"""Per-turn latency trace: delegation → route (JEV) → worker → tool → result handed to the front → spoken.

Pure bookkeeping (no I/O, no audio). The gateway marks stages; ``TraceBook`` turns the marks of one
delegation into a single machine-readable log line ``TRACE {json}``; ``scripts/trace_summary.py`` aggregates them.

Stages (ms since the delegation reached the gateway, first mark wins):
  deleg        0, the front called delegate()
  route        admit() returned (includes the JEV round trip, reported separately as ``jev_ms``)
  worker_start the Worker job began
  worker_done  the Worker returned (per-tool ``ms`` / ``source`` are in ``tools``)
  say          the Coordinator decided what to say (answer / clarify / status)
  posted       the text was handed to the voice front (tool_result accepted by the engine)
  speech_start the first audio/text of the response that carries it
  speech_end   that response was heard to the end (or ``speech_cut`` when interrupted/abandoned)
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any

log = logging.getLogger("sophon_live.trace")

STAGES = ("deleg", "route", "worker_start", "worker_done", "say", "posted", "speech_start", "speech_end", "speech_cut", "silent")


class TurnTrace:
    def __init__(self, session: str, turn: int, did: str, *, clock=time.monotonic) -> None:  # noqa: ANN001
        self._clock = clock
        self.t0 = clock()
        self.session, self.turn, self.did = session, turn, did
        self.stages: dict[str, int] = {"deleg": 0}
        self.meta: dict[str, Any] = {}
        self.tools: list[dict[str, Any]] = []
        self.emitted = False

    def mark(self, stage: str) -> None:
        self.stages.setdefault(stage, round((self._clock() - self.t0) * 1000))

    def set(self, **kv: Any) -> None:
        self.meta.update({k: v for k, v in kv.items() if v is not None})

    def add_tool(self, name: str, ms: int | None, ok: bool, **extra: Any) -> None:
        self.tools.append({"tool": name, "ms": ms, "ok": ok, **{k: v for k, v in extra.items() if v is not None}})

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"session": self.session, "turn": self.turn, "did": self.did, "stages": dict(self.stages)}
        d.update(self.meta)
        if self.tools:
            d["tools"] = list(self.tools)
        return d

    def emit(self) -> None:
        if self.emitted:
            return
        self.emitted = True
        log.info("TRACE %s", json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":")))


class TraceBook:
    """All traces of one session, plus the speech hand-off queue (which trace the next response belongs to)."""

    def __init__(self, session: str, *, clock=time.monotonic) -> None:  # noqa: ANN001
        self.session = session
        self._clock = clock
        self._turn = 0
        self.by_did: dict[str, TurnTrace] = {}
        self._said: deque[TurnTrace] = deque()          # posted to the front, response not started yet
        self._speaking: dict[str, TurnTrace] = {}       # response id -> trace

    def start(self, did: str) -> TurnTrace:
        self._turn += 1
        tr = TurnTrace(self.session, self._turn, did, clock=self._clock)
        self.by_did[did] = tr
        return tr

    def get(self, did: str) -> TurnTrace | None:
        return self.by_did.get(did)

    def posted(self, did: str) -> None:
        tr = self.by_did.get(did)
        if tr is not None:
            tr.mark("posted")
            self._said.append(tr)

    def silent(self, did: str) -> None:
        """Closed without speaking (FAST / stale / cancelled)."""
        tr = self.by_did.get(did)
        if tr is not None:
            tr.mark("silent")
            tr.emit()

    def response_started(self, rid: str) -> None:
        if self._said:
            tr = self._said.popleft()
            tr.mark("speech_start")
            self._speaking[rid] = tr

    def response_played(self, rid: str) -> None:
        tr = self._speaking.pop(rid, None)
        if tr is not None:
            tr.mark("speech_end")
            tr.emit()

    def cut(self, *, close: bool = False) -> None:
        """Barge-in (or session close): responses in flight are cut; posted-but-unstarted ones stay expected unless closing."""
        for tr in list(self._speaking.values()):
            tr.mark("speech_cut")
            tr.set(cut="close" if close else "barge_in")
            tr.emit()
        self._speaking.clear()
        if close:
            for tr in list(self._said):
                tr.set(cut="close_unspoken")
                tr.emit()
            self._said.clear()
            for tr in self.by_did.values():
                tr.emit()  # anything still open: partial trace is better than none
