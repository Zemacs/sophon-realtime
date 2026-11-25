"""Client tool bridge: lets a gateway outside the serving process see the model's tool calls and hand back results.

Engine-agnostic contract (the same two calls are implemented by every engine adapter):
  GET  /bridge/<key>/events?wait=<s>   -> {"events": [{"type": "tool_call", "call_id", "name", "arguments"}]}  (long poll)
  POST /bridge/<key>/result            <- {"call_id": str|null, "output": str, "speak": bool}
                                       -> 200 {"ok": true, "status": "accepted"|"queued"}  ("queued": the reply that made the call is still being spoken; the result is held for it)
                                          404 {"ok": false, "status": "rejected", "reason": "session_not_ready"|"unknown_session"|"turn_discarded"|"unknown_call_id"|"no_pending_call"}
  POST /bridge/<key>/say               <- {"text": str}   speak this text verbatim in the session voice (kept in history)
<key> is the session's extra_body.bridge_key. Enabled when SOPHON_BRIDGE_PORT is set; binds 127.0.0.1 only."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("sophon.bridge")
PORT = int(os.environ.get("SOPHON_BRIDGE_PORT", "0") or 0)


class Bridge:
    def __init__(self, on_result, on_say=None):
        self.on_result = on_result                 # (key, call_id, output, speak) -> bool
        self.on_say = on_say                       # (key, text) -> bool
        self.q: dict[str, list] = {}
        self.cv = threading.Condition()
        self.seen: dict[str, float] = {}
        self._srv = None

    def start(self):
        if not PORT or self._srv is not None:
            return
        br = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj, ensure_ascii=False).encode()
                try:
                    self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                except (BrokenPipeError, ConnectionResetError):
                    pass                                  # the gateway closed the long poll (session ended)

            def do_GET(self):
                u = urlparse(self.path); p = u.path.strip("/").split("/")
                if len(p) == 3 and p[0] == "bridge" and p[2] == "events":
                    wait = min(float(parse_qs(u.query).get("wait", ["20"])[0]), 60.0)
                    return self._send(200, {"events": br.pop(p[1], wait)})
                self._send(404, {"error": "not found"})

            def _result_reply(self, code, obj, **fields):
                # the full access log stays off (log_message above); only /result gets one structured line: status code, status and reason
                log.info("bridge.result %s", json.dumps({"status_code": code, "status": obj.get("status"), "reason": obj.get("reason"), "t_mono": time.monotonic(), "t_wall": time.time(), **fields}, sort_keys=True))
                self._send(code, obj)

            def do_POST(self):
                p = urlparse(self.path).path.strip("/").split("/")
                n = int(self.headers.get("Content-Length") or 0)
                is_result = len(p) == 3 and p[0] == "bridge" and p[2] == "result"
                try:
                    j = json.loads(self.rfile.read(n) or b"{}")
                except Exception:  # noqa: BLE001
                    if is_result: return self._result_reply(400, {"ok": False, "status": "rejected", "reason": "bad_json"}, key=p[1])
                    return self._send(400, {"error": "bad json"})
                if is_result:
                    f = {"key": p[1], "call_id": j.get("call_id"), "speak": bool(j.get("speak", True)), "output_len": len(str(j.get("output", "")))}
                    try:
                        r = br.on_result(p[1], j.get("call_id"), str(j.get("output", "")), bool(j.get("speak", True)))
                    except Exception:  # noqa: BLE001
                        log.exception("bridge.result handler failed")
                        return self._result_reply(500, {"ok": False, "status": "rejected", "reason": "internal_error"}, **f)
                    ok, why = r if isinstance(r, tuple) else (bool(r), None)       # on_result may return a bare bool or (ok, status-or-reason)
                    if ok: return self._result_reply(200, {"ok": True, "status": why or "accepted"}, **f)
                    return self._result_reply(404, {"ok": False, "status": "rejected", "reason": why or "rejected"}, **f)
                if len(p) == 3 and p[0] == "bridge" and p[2] == "say" and br.on_say is not None:
                    ok = br.on_say(p[1], str(j.get("text", "")))
                    return self._send(200 if ok else 404, {"ok": ok})
                self._send(404, {"error": "not found"})

        self._srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True, name="sophon-bridge").start()

    def push(self, key, ev):
        with self.cv:
            self.q.setdefault(key, []).append(ev); self.cv.notify_all()

    def pop(self, key, wait):
        end = time.monotonic() + wait
        with self.cv:
            self.seen[key] = time.monotonic()
            while not self.q.get(key):
                left = end - time.monotonic()
                if left <= 0:
                    return []
                self.cv.wait(left)
            out, self.q[key] = self.q[key], []
            return out
