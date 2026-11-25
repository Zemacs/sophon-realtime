"""Round 1 / item 2: Delivery receipts. 听完 = DELIVERED; 打断 ≠ 取消; 显式取消才取消.

Pure CPU: DeliveryTracker + Ledger + real LiveSession glue methods with a fake engine.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import Ledger, PlaybackReceipt, WorkerJob, api
from sophon_live.coordinator.api import complete_job, on_playback_receipt
from sophon_live.coordinator.delivery import DeliveryTracker
from sophon_live.coordinator.schemas import TaskStatus
from tests.test_gateway_coord_control import FakeEngine, _commentary, _session
from tests.worker_stub import as_worker
from tests.test_r0_coord_boundary import _arm


def _done_task(led: Ledger, text: str = "现在几点了", answer: str = "三点整") -> tuple[str, int]:
    rec = led.create(text)
    led.mark_running(rec.task_id)
    complete_job(led, WorkerJob(task_id=rec.task_id, version=1, objective=text), answer)
    return rec.task_id, 1


class LedgerReceiptTests(unittest.TestCase):
    def test_delivered_marks_completed_task_delivered(self) -> None:
        led = Ledger()
        tid, ver = _done_task(led)
        self.assertTrue(on_playback_receipt(led, PlaybackReceipt(tid, ver, "delivered")))
        self.assertEqual(led.tasks[tid].status, TaskStatus.DELIVERED)
        self.assertEqual(led.tasks[tid].spoken_answer, "三点整")

    def test_interrupted_keeps_completed_and_counts(self) -> None:
        led = Ledger()
        tid, ver = _done_task(led)
        on_playback_receipt(led, PlaybackReceipt(tid, ver, "interrupted"))
        rec = led.tasks[tid]
        self.assertEqual(rec.status, TaskStatus.COMPLETED)
        self.assertEqual(rec.interrupted, 1)
        self.assertEqual(rec.spoken_answer, "三点整")

    def test_abandoned_keeps_completed(self) -> None:
        led = Ledger()
        tid, ver = _done_task(led)
        on_playback_receipt(led, PlaybackReceipt(tid, ver, "abandoned"))
        self.assertEqual(led.tasks[tid].status, TaskStatus.COMPLETED)

    def test_receipt_never_cancels_or_revives(self) -> None:
        led = Ledger()
        rec = led.create("北京明天天气")
        led.mark_running(rec.task_id)
        for outcome in ("delivered", "interrupted", "abandoned"):
            on_playback_receipt(led, PlaybackReceipt(rec.task_id, 1, outcome))  # type: ignore[arg-type]
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.RUNNING)  # running job is untouched
        led.cancel(rec.task_id, 1)
        on_playback_receipt(led, PlaybackReceipt(rec.task_id, 1, "delivered"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.CANCELLED)

    def test_stale_version_receipt_is_ignored(self) -> None:
        led = Ledger()
        tid, ver = _done_task(led)
        self.assertFalse(on_playback_receipt(led, PlaybackReceipt(tid, 7, "delivered")))
        self.assertEqual(led.tasks[tid].status, TaskStatus.COMPLETED)
        self.assertFalse(on_playback_receipt(led, PlaybackReceipt("nope", 1, "delivered")))

    def test_delivered_task_stays_controllable(self) -> None:
        led = Ledger()
        tid, ver = _done_task(led)
        on_playback_receipt(led, PlaybackReceipt(tid, ver, "delivered"))
        self.assertEqual(led.latest().task_id, tid)  # type: ignore[union-attr]
        self.assertIsNotNone(led.refine(tid, "改成日期", version=ver))


class TrackerTests(unittest.TestCase):
    def test_arm_attach_play_delivers(self) -> None:
        tr = DeliveryTracker()
        tr.arm("t1", 1)
        self.assertTrue(tr.attach("r1"))
        self.assertEqual(tr.on_played("r1"), PlaybackReceipt("t1", 1, "delivered"))
        self.assertIsNone(tr.on_played("r1"))  # one receipt per delivery

    def test_unarmed_response_is_not_tracked(self) -> None:
        tr = DeliveryTracker()
        self.assertFalse(tr.attach("r_chat"))
        self.assertIsNone(tr.on_played("r_chat"))

    def test_attach_is_fifo(self) -> None:
        tr = DeliveryTracker()
        tr.arm("a", 1)
        tr.arm("b", 2)
        tr.attach("r1")
        tr.attach("r2")
        self.assertEqual(tr.on_played("r2"), PlaybackReceipt("b", 2, "delivered"))
        self.assertEqual(tr.on_played("r1"), PlaybackReceipt("a", 1, "delivered"))

    def test_barge_in_interrupts_attached_only(self) -> None:
        tr = DeliveryTracker()
        tr.arm("a", 1)
        tr.attach("r1")
        tr.arm("b", 1)  # armed, not yet speaking
        self.assertEqual(tr.on_interrupted(), [PlaybackReceipt("a", 1, "interrupted")])
        self.assertIsNone(tr.on_played("r1"))  # already receipted
        self.assertTrue(tr.attach("r2"))  # b is still expected
        self.assertEqual(tr.on_played("r2"), PlaybackReceipt("b", 1, "delivered"))

    def test_close_abandons_everything_open(self) -> None:
        tr = DeliveryTracker()
        tr.arm("a", 1)
        tr.attach("r1")
        tr.arm("b", 1)
        got = sorted((r.task_id, r.outcome) for r in tr.on_close())
        self.assertEqual(got, [("a", "abandoned"), ("b", "abandoned")])
        self.assertEqual(tr.on_close(), [])


class GatewayDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_barge_in_does_not_cancel_running_worker(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()
        gate = asyncio.Event()

        async def orch(task, instructions="", history=None):
            await gate.wait()
            return "北京明天晴。"

        _arm(session, "d1", "北京明天天气")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(orch)):
            t1 = asyncio.create_task(session._coord_resolve("d1", "北京明天天气"))
            await asyncio.sleep(0.05)
            # real LiveSession._drop_playback needs these
            session.play = type("D", (), {"clear": lambda s: None})()
            session.resp = {}
            import sophon_live.gateway as gw

            gw.LiveSession._drop_playback(session, "barge-in (gateway VAD)")
            tid = session.coord_ledger.order[0]
            self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.RUNNING)
            self.assertFalse(session.coord_workers[tid].done())
            gate.set()
            await asyncio.wait_for(t1, 5)
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.COMPLETED)
        self.assertEqual(session.deleg["d1"]["chunks"], [("commentary", "北京明天晴。")])

    async def test_explicit_cancel_does_cancel(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()
        gate = asyncio.Event()

        async def orch(task, instructions="", history=None):
            await gate.wait()
            return "x"

        _arm(session, "d1", "北京明天天气")
        _arm(session, "d2", "取消刚才那个")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(orch)):
            t1 = asyncio.create_task(session._coord_resolve("d1", "北京明天天气"))
            await asyncio.sleep(0.05)
            await session._coord_resolve("d2", "取消刚才那个")
            await asyncio.wait_for(t1, 5)
        tid = session.coord_ledger.order[0]
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.CANCELLED)

    async def test_result_post_arms_delivery_and_full_playback_marks_delivered(self) -> None:
        import sophon_live.gateway as gw

        session = _session()
        session.delivery = DeliveryTracker()
        session.coord_ledger = Ledger()
        tid, ver = _done_task(session.coord_ledger)
        _arm(session, "d1", "现在几点了")
        session.deleg["d1"]["track"] = (tid, ver)
        session.deleg["d1"]["t0"] = 0.0
        # real _post_update posts to the engine and arms the delivery
        await gw.LiveSession._post_update(session, "d1", [("commentary", "三点整")])
        self.assertEqual(session.engine.results[0][0], "call_d1")
        self.assertNotIn("track", session.deleg["d1"])
        # the engine starts speaking: a new response attaches; playing it out delivers
        session.delivery.attach("r1")
        gw.LiveSession._receipt_played(session, "r1")
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.DELIVERED)

    async def test_barge_in_during_answer_keeps_it_replayable(self) -> None:
        import sophon_live.gateway as gw

        session = _session()
        session.delivery = DeliveryTracker()
        session.coord_ledger = Ledger()
        tid, ver = _done_task(session.coord_ledger)
        session.delivery.arm(tid, ver)
        session.delivery.attach("r1")
        session.play = type("D", (), {"clear": lambda s: None})()
        session.resp = {}
        gw.LiveSession._drop_playback(session, "barge-in (gateway VAD)")
        rec = session.coord_ledger.tasks[tid]
        self.assertEqual((rec.status, rec.interrupted), (TaskStatus.COMPLETED, 1))
        self.assertEqual(rec.spoken_answer, "三点整")
        gw.LiveSession._receipt_played(session, "r1")  # a late "played" for the dropped response must not deliver
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.COMPLETED)

    async def test_status_replay_is_tracked_and_delivers(self) -> None:
        session = _session()
        session.delivery = DeliveryTracker()
        _arm(session, "d1", "北京明天天气")
        session.coord_ledger = Ledger()
        tid, ver = _done_task(session.coord_ledger, "北京明天天气", "晴，12到18度。")
        _arm(session, "d2", "刚才查得怎么样")
        with mock.patch.object(api, "USE_JEV", False):
            await session._coord_resolve("d2", "刚才查得怎么样")
        self.assertEqual(session.deleg["d2"]["chunks"], [("commentary", "晴，12到18度。")])
        self.assertEqual(session.deleg["d2"]["track"], (tid, ver))


if __name__ == "__main__":
    unittest.main()
