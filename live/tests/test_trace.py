"""Round 3 / C: per-turn latency trace (pure bookkeeping + the gateway hooks)."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from tests import jev_standin  # noqa: E402  (scripted JEV for control-flow tests)
from sophon_live.assistant_tools import toolmeta
from sophon_live.coordinator import AdmitRequest, Ledger, WorkerResult, api
from sophon_live.coordinator.trace import TraceBook
from tests.test_gateway_coord_control import _session
from tests.test_r0_coord_boundary import _arm
from tests.worker_stub import as_worker


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000


def _lines(cm) -> list[dict]:
    return [json.loads(l.split("TRACE ", 1)[1]) for l in cm.output if "TRACE " in l]


class TraceBookTests(unittest.TestCase):
    def test_full_turn_stages_and_one_line(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            tr = book.start("d1")
            clk.advance(40); tr.mark("route"); tr.set(route="WORKER", via="jev", jev_ms=31)
            clk.advance(5); tr.mark("worker_start")
            clk.advance(900); tr.mark("worker_done"); tr.add_tool("get_weather", 880, True, source="sojson")
            clk.advance(3); tr.mark("say")
            clk.advance(20); book.posted("d1")
            clk.advance(450); book.response_started("r1")
            clk.advance(3000); book.response_played("r1")
        (t,) = _lines(cm)
        self.assertEqual(t["stages"], {"deleg": 0, "route": 40, "worker_start": 45, "worker_done": 945, "say": 948, "posted": 968, "speech_start": 1418, "speech_end": 4418})
        self.assertEqual((t["session"], t["turn"], t["did"], t["via"], t["jev_ms"]), ("s1", 1, "d1", "jev", 31))
        self.assertEqual(t["tools"], [{"tool": "get_weather", "ms": 880, "ok": True, "source": "sojson"}])

    def test_first_mark_wins_and_one_emit_per_turn(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        tr = book.start("d1")
        clk.advance(10); tr.mark("route")
        clk.advance(10); tr.mark("route")
        self.assertEqual(tr.stages["route"], 10)
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            tr.emit(); tr.emit()
        self.assertEqual(len(_lines(cm)), 1)

    def test_responses_are_matched_to_traces_in_order(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        a, b = book.start("a"), book.start("b")
        book.response_started("pad")             # a pad sentence before anything was posted: nobody's
        book.posted("a"); book.posted("b")
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            book.response_started("r1"); book.response_started("r2")
            clk.advance(10); book.response_played("r2"); book.response_played("r1")
        self.assertEqual({t["did"] for t in _lines(cm)}, {"a", "b"})
        self.assertIn("speech_start", a.stages)
        self.assertNotIn("pad", book._speaking)

    def test_barge_in_marks_cut_not_end(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.start("a"); book.posted("a"); book.response_started("r1")
        clk.advance(700)
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            book.cut()
        (t,) = _lines(cm)
        self.assertIn("speech_cut", t["stages"])
        self.assertNotIn("speech_end", t["stages"])
        self.assertEqual(t["cut"], "barge_in")

    def test_close_flushes_everything_still_open(self) -> None:
        book = TraceBook("s1")
        book.start("a"); book.start("b"); book.posted("b")
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            book.cut(close=True)
        self.assertEqual({t["did"] for t in _lines(cm)}, {"a", "b"})

    def test_silent_close(self) -> None:
        book = TraceBook("s1")
        book.start("a")
        with self.assertLogs("sophon_live.trace", "INFO") as cm:
            book.silent("a")
        self.assertIn("silent", _lines(cm)[0]["stages"])


class ToolMetaTests(unittest.IsolatedAsyncioTestCase):
    async def test_note_and_take_are_per_task(self) -> None:
        async def tool():
            toolmeta.note(source="wttr")

        toolmeta.take()
        await asyncio.ensure_future(tool())          # a different task: must not leak into this one
        self.assertEqual(toolmeta.take(), {})
        await tool()
        self.assertEqual(toolmeta.take(), {"source": "wttr"})
        self.assertEqual(toolmeta.take(), {})


class GatewayTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_turn_produces_one_complete_trace(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()

        async def worker(job, instructions="", history=None):
            return WorkerResult(job.task_id, job.version, "completed", "北京明天晴。",
                                evidence=[{"tool": "get_weather", "args": {}, "ok": True, "ms": 812, "source": "sojson"}],
                                freshness_token=job.freshness_token)

        session._trace().start("d1").set(task="北京明天天气")
        _arm(session, "d1", "北京明天天气")
        with self.assertLogs("sophon_live.trace", "INFO") as cm, \
                jev_standin.standin(), mock.patch.object(tools, "run_worker_job", worker):
            await session._coord_resolve("d1", "北京明天天气")
            session._delivery_armed("d1")              # what _post_update does once the engine took the result
            session._trace().response_started("r1")
            session._trace().response_played("r1")
        (t,) = _lines(cm)
        self.assertEqual(list(t["stages"]), ["deleg", "route", "worker_start", "worker_done", "say", "posted", "speech_start", "speech_end"])
        self.assertEqual((t["route"], t["decision"], t["via"], t["worker"]), ("WORKER", "NEW", "jev", "completed"))
        self.assertEqual(t["tools"], [{"tool": "get_weather", "ms": 812, "ok": True, "source": "sojson"}])
        self.assertEqual(t["say_chars"], len("北京明天晴。"))

    async def test_fast_turn_is_traced_as_silent(self) -> None:
        session = _session()
        session._trace().start("f")
        _arm(session, "f", "谢谢")
        with self.assertLogs("sophon_live.trace", "INFO") as cm, jev_standin.standin():
            await session._coord_resolve("f", "谢谢")
        (t,) = _lines(cm)
        self.assertIn("silent", t["stages"])
        self.assertEqual(t["decision"], "FAST")

    async def test_degraded_turn_trace_has_jev_state_and_degraded_reason(self) -> None:
        import sophon_live.assistant_tools as tools
        from sophon_live.coordinator import jev_client

        session = _session()

        async def worker(job, instructions="", history=None):
            return WorkerResult(job.task_id, job.version, "completed", "北京明天晴。",
                                evidence=[{"tool": "get_weather", "args": {}, "ok": True, "ms": 5}], freshness_token=job.freshness_token)

        async def refused(state, questions):  # noqa: ANN001
            raise jev_client.JevError("refused", "connection refused")

        session._trace().start("d1").set(task="北京明天天气")
        _arm(session, "d1", "北京明天天气")
        with self.assertLogs("sophon_live.trace", "INFO") as cm, mock.patch.object(api, "USE_JEV", True), \
                mock.patch.object(api, "REFINE_GUARD", False), mock.patch.object(jev_client, "system_one", refused), \
                mock.patch.object(tools, "run_worker_job", worker):
            await session._coord_resolve("d1", "北京明天天气")
            session._delivery_armed("d1")
            session._trace().response_started("r1")
            session._trace().response_played("r1")
        (t,) = _lines(cm)
        self.assertEqual((t["route"], t["decision"], t["via"], t["degraded_reason"], t["jev_state"]), ("WORKER", "NEW", "degraded", "refused", "closed"))
        self.assertEqual(t["worker"], "completed")

    async def test_normal_turn_trace_has_state_and_no_degraded_reason(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()

        async def worker(job, instructions="", history=None):
            return WorkerResult(job.task_id, job.version, "completed", "好。", evidence=[], freshness_token=job.freshness_token)

        session._trace().start("d1").set(task="讲个笑话")
        _arm(session, "d1", "讲个笑话")
        with self.assertLogs("sophon_live.trace", "INFO") as cm, jev_standin.standin(), mock.patch.object(tools, "run_worker_job", worker):
            await session._coord_resolve("d1", "讲个笑话")
            session._delivery_armed("d1")
            session._trace().response_started("r1")
            session._trace().response_played("r1")
        (t,) = _lines(cm)
        self.assertEqual((t["via"], t["jev_state"]), ("jev", "closed"))
        self.assertNotIn("degraded_reason", t)


if __name__ == "__main__":
    unittest.main()
