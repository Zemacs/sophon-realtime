"""A tool-backed question as the very first utterance: the tool call is announced while the first reply is still being generated, so the session's
history is still empty when the gateway posts the result. The old engine answered 404; the new one holds the result for that call and speaks it
through the normal follow-up path (tool_followup -> next append with allow_emit -> plan_append)."""
import asyncio
import json
import logging
import socket
import sys
import types
import urllib.error
import urllib.request

import numpy as np

import duplex_harness as h

CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


class _Tok:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return json.dumps(messages, ensure_ascii=False)


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)


def config(key):
    return types.SimpleNamespace(instructions=None, temperature=None, max_tokens=None, tools=None, voice=None,
                                 extra_body={"bridge_key": key, "tool_mode": "client", "language": "en"})


def session(m, key="k1"):
    """What the manager does on open (prepare_runtime_config, then create_session_state), against a real bridge on a free loopback port."""
    plugin = m.SophonRealtimeDuplexPlugin(lambda *a, **k: b""); plugin.tok = _Tok()
    br = sys.modules[m.__name__.rsplit(".", 1)[0] + ".bridge"]
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); br.PORT = s.getsockname()[1]
    asyncio.run(plugin.prepare_runtime_config(config(key), model_config=None))
    buf = plugin.create_session_state().audio_buffer; buf.tool_mode = "client"
    if key not in plugin._keys:                                   # the previous code only did this (and started the bridge) in plan_append, after a turn opened
        buf.bridge_key = key; plugin._keys[key] = buf; plugin.bridge.start()
    return plugin, buf, f"http://127.0.0.1:{br.PORT}/bridge/{key}/result"


def first_reply_in_flight(m, plugin, buf):
    """The thinker has written 'let me check' + a tool call; the reply is not finished, history is empty. Returns the announced call id."""
    info = {"kind": "reply", "buf": buf, "user": np.zeros(h.SR, np.float32), "a_audio": [], "a_text": "Let me check.", "raw": "Let me check." + CALL,
            "t_plan": 0.0, "spec": False, "released": True, "dropped": False, "held": []}
    plugin._reqs["rid1"] = info; assert not buf.history
    plugin._announce(info)
    return plugin.bridge.q["k1"][0]["call_id"]


def run(m):
    plugin, buf, url = session(m); cid = first_reply_in_flight(m, plugin, buf)
    code, body = post(url, {"call_id": cid, "output": "Paris: 18C, sunny", "speak": True})
    plugin._finish_reply("rid1", completed=True)                       # the first reply ends: history gets its entry
    got = []
    for _ in range(3):
        m.time.t += 0.1
        r = buf.prepare_append(h.payload(np.zeros(1600, np.float32)), operation_id="op", chunk_period_ms=100, allow_emit=True)
        if r is not None: got.append(r)
    return code, body, got, plugin, buf


def test_old_code_rejects_a_result_when_history_is_empty(old):
    code, body, got, _, buf = run(old)
    assert code == 404 and body == {"ok": False} and not got and not buf.tool_followup


def test_new_code_holds_the_result_and_produces_one_response(new):
    code, body, got, plugin, buf = run(new)
    assert code == 200 and body == {"ok": True, "status": "queued"}
    assert len(got) == 1 and got[0].payload["tool_followup"] is True and got[0].payload["sophon_kind"] == "reply"
    plan = plugin.plan_append(request_id="r2", fence=None, session_config=None, runtime_config={"lang": "en"}, seq=1, turn_seq=1, payload=got[0].payload, final=False, sampling_params=None)
    assert "Paris: 18C, sunny" in plan.prompt["prompt"] and "get_weather" in plan.prompt["prompt"]


def test_the_front_audio_gate_is_not_bypassed(new):
    plugin, buf, url = session(new); cid = first_reply_in_flight(new, plugin, buf)
    assert post(url, {"call_id": cid, "output": "x", "speak": True})[0] == 200
    plugin._finish_reply("rid1", completed=True)
    new.time.t += 0.1
    gated = buf.prepare_append(h.payload(np.zeros(1600, np.float32)), operation_id="op", chunk_period_ms=100, allow_emit=False)
    assert gated is None and buf.tool_followup                                        # allow_emit=False: nothing is emitted, the follow-up stays queued
    assert buf.prepare_append(h.payload(np.zeros(1600, np.float32)), operation_id="op", chunk_period_ms=100, allow_emit=True) is not None


def test_unknown_ids_are_rejected_with_a_reason(new):
    plugin, buf, url = session(new); first_reply_in_flight(new, plugin, buf)
    code, body = post(url, {"call_id": "call_nope", "output": "x", "speak": True})
    assert code == 404 and body == {"ok": False, "status": "rejected", "reason": "unknown_call_id"} and not buf.early_results
    code, body = post(url.replace("/k1/", "/nokey/"), {"call_id": "call_nope", "output": "x"})
    assert code == 404 and body["reason"] == "unknown_session"


def test_null_call_id_needs_exactly_one_pending_call(new):
    plugin, buf, url = session(new)
    assert post(url, {"call_id": None, "output": "x"})[1]["reason"] == "no_pending_call"
    cid = first_reply_in_flight(new, plugin, buf)
    assert post(url, {"call_id": None, "output": "x"}) == (200, {"ok": True, "status": "queued"}) and cid in buf.early_results


def test_a_later_result_for_a_finished_reply_still_follows_the_existing_path(new):
    plugin, buf, url = session(new); cid = first_reply_in_flight(new, plugin, buf)
    post(url, {"call_id": cid, "output": "progress", "speak": False}); plugin._finish_reply("rid1", completed=True)
    assert post(url, {"call_id": cid, "output": "final answer", "speak": True}) == (200, {"ok": True, "status": "accepted"})
    assert [st["results"] for st in buf.history[-1]["chain"]] == [["progress"], ["final answer"]] and buf.tool_followup


def test_the_bridge_key_is_registered_when_the_session_opens_not_when_a_turn_opens(new, old):
    plugin, buf, url = session(new)
    assert plugin._keys["k1"] is buf and buf.bridge_key == "k1" and not buf.history       # no audio, no turn, no plan_append yet
    assert post(url, {"call_id": "call_x", "output": "x"}) == (404, {"ok": False, "status": "rejected", "reason": "unknown_call_id"})
    p_old = old.SophonRealtimeDuplexPlugin(lambda *a, **k: b""); p_old.tok = _Tok()
    asyncio.run(p_old.prepare_runtime_config(config("k1"), model_config=None)); p_old.create_session_state()
    assert "k1" not in p_old._keys and p_old._bridge_result("k1", "call_x", "x", True) is False


def test_rejection_reasons_distinguish_the_branches(new):
    plugin, buf, url = session(new)
    plugin.runtime_config_for_update(config("k2"), {})                                     # key seen in a session.update, buffer not bound yet
    assert post(url.replace("/k1/", "/k2/"), {"call_id": "c", "output": "x"})[1]["reason"] == "session_not_ready"
    assert post(url.replace("/k1/", "/zz/"), {"call_id": "c", "output": "x"})[1]["reason"] == "unknown_session"
    cid = first_reply_in_flight(new, plugin, buf); plugin._finish_reply("rid1", completed=True); buf.history.clear()
    assert post(url, {"call_id": cid, "output": "x"})[1]["reason"] == "turn_discarded"     # the call's turn was voided and history is empty


def test_every_result_is_logged_with_status_code_and_reason(new, caplog):
    caplog.set_level(logging.INFO)
    plugin, buf, url = session(new); cid = first_reply_in_flight(new, plugin, buf)
    post(url, {"call_id": "call_nope", "output": "x"}); post(url, {"call_id": cid, "output": "ok"})
    lines = [json.loads(r.getMessage().split("bridge.result ", 1)[1]) for r in caplog.records if r.name == "sophon.bridge"]
    assert [(l["status_code"], l["reason"], l["status"]) for l in lines] == [(404, "unknown_call_id", "rejected"), (200, None, "queued")]
    assert all(k in lines[0] for k in ("key", "call_id", "speak", "output_len", "t_mono", "t_wall"))
    assert any(r.levelno == logging.WARNING and "REJECTED" in r.getMessage() and "unknown_call_id" in r.getMessage() for r in caplog.records)
