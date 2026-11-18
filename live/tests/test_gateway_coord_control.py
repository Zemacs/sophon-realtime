"""Round 1 / item 1 at the gateway: real LiveSession._coord_resolve with fake orch and fake engine.

Not a Live audio chain. Checks: FAST starts nothing, stale/cancelled results are not spoken,
cancel/refine stop the in-flight worker.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import AdmitResult, SpeakPlan, api
from sophon_live.coordinator.schemas import TaskStatus
from tests.worker_stub import as_worker
from tests.test_r0_coord_boundary import _arm, _bare_session


class FakeEngine:
    def __init__(self) -> None:
        self.results: list[tuple[str | None, str, bool]] = []

    async def tool_result(self, call_id, output, speak=True):  # noqa: ANN001
        self.results.append((call_id, output, speak))
        return True


def _session():
    session = _bare_session()
    session.engine = FakeEngine()

    async def flush(did: str) -> None:  # recorded only: the real flush posts to the engine
        session.flushed.append(did)

    session._flush_deleg = flush
    return session


def _commentary(session) -> list[dict]:
    return [e for e in session.sent if e.get("type") == "session.commentary.appended"]


class GatewayCoordControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_acks_silently_and_starts_nothing(self) -> None:
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord

        session = _session()
        orch_calls: list[str] = []

        async def fake_admit(req, ledger):
            return AdmitResult(speak=SpeakPlan(kind="noop"), job=None, route="FAST", ledger=ledger)

        async def fake_orch(task, instructions="", history=None):
            orch_calls.append(task)
            return "x"

        _arm(session, "f", "谢谢")
        with mock.patch.object(coord, "admit", fake_admit), mock.patch.object(tools, "run_worker_job", as_worker(fake_orch)):
            await session._coord_resolve("f", "谢谢")
        self.assertEqual(orch_calls, [])
        self.assertEqual(len(session.coord_ledger.order), 0)
        self.assertEqual(_commentary(session), [])
        self.assertEqual([(c, s) for c, _, s in session.engine.results], [("call_f", False)])

    async def test_stop_speaking_drops_playback_and_acks_silently(self) -> None:
        import sophon_live.coordinator as coord

        session = _session()

        async def fake_admit(req, ledger):
            return AdmitResult(speak=SpeakPlan(kind="stop_speaking", barge_policy="drop_playback"), route="STOP_SPEAKING", ledger=ledger)

        _arm(session, "s", "别说了")
        with mock.patch.object(coord, "admit", fake_admit):
            await session._coord_resolve("s", "别说了")
        self.assertIn(("drop", "coord stop_speaking"), session.flushed)
        self.assertEqual([(c, s) for c, _, s in session.engine.results], [("call_s", False)])

    async def test_cancel_stops_inflight_worker_and_its_result_is_not_spoken(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()
        started = asyncio.Event()
        finished: list[str] = []

        async def slow_orch(task, instructions="", history=None):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                finished.append("cancelled")
                raise
            finished.append("done")
            return "迟到的结果"

        _arm(session, "d1", "北京明天天气")
        _arm(session, "d2", "取消刚才那个")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(slow_orch)):
            t1 = asyncio.create_task(session._coord_resolve("d1", "北京明天天气"))
            await asyncio.wait_for(started.wait(), 5)
            await session._coord_resolve("d2", "取消刚才那个")
            await asyncio.wait_for(t1, 5)
        self.assertEqual(finished, ["cancelled"])
        tid = session.coord_ledger.order[0]
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.CANCELLED)
        # d2 spoke the cancel confirmation; d1 spoke nothing
        self.assertEqual(len(_commentary(session)), 1)
        self.assertEqual(session.deleg["d1"]["chunks"], [])
        self.assertIn(("call_d1", False), [(c, s) for c, _, s in session.engine.results])

    async def test_refine_supersedes_inflight_worker(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()
        started = asyncio.Event()
        asked: list[str] = []

        async def orch(task, instructions="", history=None):
            asked.append(task)
            if len(asked) == 1:
                started.set()
                await asyncio.sleep(30)
            return "北京晴"

        _arm(session, "d1", "上海明天天气")
        _arm(session, "d2", "改成北京的")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(orch)):
            t1 = asyncio.create_task(session._coord_resolve("d1", "上海明天天气"))
            await asyncio.wait_for(started.wait(), 5)
            await session._coord_resolve("d2", "改成北京的")
            await asyncio.wait_for(t1, 5)
        self.assertEqual(len(asked), 2)
        self.assertIn("改成北京的", asked[1])
        tid = session.coord_ledger.order[0]
        rec = session.coord_ledger.tasks[tid]
        self.assertEqual((rec.version, rec.spoken_answer), (2, "北京晴"))
        # exactly one spoken result: the refined one, on delegation d2
        self.assertEqual(len(_commentary(session)), 1)
        self.assertEqual(session.deleg["d1"]["chunks"], [])
        self.assertEqual(session.deleg["d2"]["chunks"], [("commentary", "北京晴")])


if __name__ == "__main__":
    unittest.main()
