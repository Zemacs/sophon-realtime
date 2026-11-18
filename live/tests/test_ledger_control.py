"""Round 1 / item 1: Ledger CONTROL (version, SUPERSEDED, REFINE, CANCEL, STATUS).

Pure CPU. JEV is stubbed or off; the Live path is covered by scripts/path1_live_multiturn_suite.py.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import AdmitRequest, AdmitResult, Ledger, SpeakPlan, WorkerJob, admit
from sophon_live.coordinator import api, jev_client
from sophon_live.coordinator.api import complete_job
from sophon_live.coordinator.route import heuristic_control, heuristic_route
from sophon_live.coordinator.schemas import TaskStatus


def _req(text: str, *, uid: str = "u") -> AdmitRequest:
    return AdmitRequest(session_id="t", utterance_id=uid, text=text, delegate_task=text)


class LedgerStateTests(unittest.TestCase):
    def test_refine_bumps_version_and_archives_old_as_superseded(self) -> None:
        led = Ledger()
        rec = led.create("上海明天天气")
        led.mark_running(rec.task_id)
        new = led.refine(rec.task_id, "北京明天天气", version=1)
        assert new is not None
        self.assertEqual((new.task_id, new.version), (rec.task_id, 2))
        self.assertEqual(led.tasks[rec.task_id].version, 2)
        self.assertEqual([(t.version, t.status) for t in led.superseded], [(1, TaskStatus.SUPERSEDED)])

    def test_refine_with_stale_version_is_noop(self) -> None:
        led = Ledger()
        rec = led.create("上海明天天气")
        led.refine(rec.task_id, "北京明天天气", version=1)
        self.assertIsNone(led.refine(rec.task_id, "广州明天天气", version=1))
        self.assertEqual(led.tasks[rec.task_id].version, 2)
        self.assertEqual(led.tasks[rec.task_id].objective, "北京明天天气")

    def test_refine_cancelled_task_is_noop(self) -> None:
        led = Ledger()
        rec = led.create("讲个笑话")
        self.assertTrue(led.cancel(rec.task_id, 1))
        self.assertIsNone(led.refine(rec.task_id, "再讲一个"))

    def test_late_result_after_cancel_is_dropped_and_does_not_revive(self) -> None:
        led = Ledger()
        rec = led.create("北京明天天气")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=1, objective=rec.objective)
        self.assertTrue(led.cancel(rec.task_id, 1))
        self.assertFalse(complete_job(led, job, "迟到的结果"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.CANCELLED)
        self.assertEqual(led.tasks[rec.task_id].spoken_answer, "")

    def test_late_result_of_old_version_is_dropped(self) -> None:
        led = Ledger()
        rec = led.create("上海明天天气")
        job1 = WorkerJob(task_id=rec.task_id, version=1, objective=rec.objective)
        led.refine(rec.task_id, "北京明天天气", version=1)
        self.assertFalse(complete_job(led, job1, "上海多云"))
        self.assertEqual(led.tasks[rec.task_id].spoken_answer, "")
        job2 = WorkerJob(task_id=rec.task_id, version=2, objective="北京明天天气")
        self.assertTrue(complete_job(led, job2, "北京晴"))
        self.assertEqual(led.tasks[rec.task_id].spoken_answer, "北京晴")

    def test_cancel_completed_task_is_refused(self) -> None:
        led = Ledger()
        rec = led.create("现在几点了")
        job = WorkerJob(task_id=rec.task_id, version=1, objective=rec.objective)
        complete_job(led, job, "三点整")
        self.assertFalse(led.cancel(rec.task_id, 1))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.COMPLETED)

    def test_cancel_wrong_version_is_refused(self) -> None:
        led = Ledger()
        rec = led.create("x")
        self.assertFalse(led.cancel(rec.task_id, 9))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.QUEUED)

    def test_pending_input_returns_latest_needs_input(self) -> None:
        led = Ledger()
        a = led.create("明天天气怎么样")
        led.mark_needs_input(a.task_id)
        b = led.create("现在几点了")
        led.mark_running(b.task_id)
        self.assertEqual(led.pending_input().task_id, a.task_id)  # type: ignore[union-attr]


class HeuristicControlTests(unittest.TestCase):
    def _led(self, objective: str, status: TaskStatus, answer: str = "") -> tuple[Ledger, str]:
        led = Ledger()
        rec = led.create(objective)
        rec.status = status
        rec.spoken_answer = answer
        return led, rec.task_id

    def test_cancel_words(self) -> None:
        for text in ("取消刚才那个", "别查了", "算了不用了"):
            led, tid = self._led("北京明天天气", TaskStatus.RUNNING)
            d = heuristic_control(text, led)
            self.assertIsNotNone(d, text)
            assert d is not None
            self.assertEqual((d.op, d.task_id, d.version), ("CANCEL", tid, 1), text)

    def test_status_words(self) -> None:
        for text in ("刚才查得怎么样", "查到了吗", "结果呢", "再说一遍刚才的结果"):
            led, tid = self._led("北京明天天气", TaskStatus.COMPLETED, "晴")
            d = heuristic_control(text, led)
            self.assertIsNotNone(d, text)
            assert d is not None
            self.assertEqual(d.op, "STATUS", text)

    def test_refine_words(self) -> None:
        for text in ("改成上海的", "换成广州", "不对，是上海的", "再讲一个不一样的", "换一个"):
            led, tid = self._led("北京明天天气", TaskStatus.COMPLETED, "晴")
            d = heuristic_control(text, led)
            self.assertIsNotNone(d, text)
            assert d is not None
            self.assertEqual(d.op, "REFINE", text)

    def test_short_answer_fills_pending_slot(self) -> None:
        led, tid = self._led("明天天气怎么样", TaskStatus.NEEDS_INPUT)
        d = heuristic_control("北京的", led)
        assert d is not None
        self.assertEqual((d.op, d.task_id), ("REFINE", tid))

    def test_no_task_means_no_control(self) -> None:
        self.assertIsNone(heuristic_control("取消刚才那个", Ledger()))
        self.assertIsNone(heuristic_control("刚才查得怎么样", Ledger()))

    def test_fresh_requests_are_not_control(self) -> None:
        led, _ = self._led("北京明天天气", TaskStatus.COMPLETED, "晴")
        for text in ("现在几点了", "科技新闻", "明天天气怎么样", "把客厅灯打开"):
            self.assertIsNone(heuristic_control(text, led), text)

    def test_heuristic_route_reports_control(self) -> None:
        led, _ = self._led("北京明天天气", TaskStatus.RUNNING)
        self.assertEqual(heuristic_route("取消刚才那个", led), "CONTROL")


class AdmitControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_marks_task_and_starts_no_worker(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            started = await admit(_req("北京明天天气", uid="w"), led)
            res = await admit(_req("取消刚才那个", uid="c"), led)
        self.assertIsNone(res.job)
        self.assertEqual(res.route, "CONTROL")
        self.assertEqual(res.speak.kind, "answer")
        self.assertEqual(res.control.op, "CANCEL")  # type: ignore[union-attr]
        self.assertEqual(led.tasks[started.job.task_id].status, TaskStatus.CANCELLED)  # type: ignore[union-attr]
        self.assertEqual(len(led.tasks), 1)

    async def test_cancel_of_finished_task_says_so(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            started = await admit(_req("现在几点了"), led)
            complete_job(led, started.job, "三点整")  # type: ignore[arg-type]
            res = await admit(_req("取消刚才那个"), led)
        self.assertIsNone(res.job)
        self.assertIn("办完", res.speak.text)
        self.assertEqual(led.tasks[started.job.task_id].status, TaskStatus.COMPLETED)  # type: ignore[union-attr]

    async def test_status_returns_stored_answer(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            started = await admit(_req("北京明天天气"), led)
            complete_job(led, started.job, "北京明天晴，12到18度。")  # type: ignore[arg-type]
            res = await admit(_req("刚才查得怎么样"), led)
        self.assertIsNone(res.job)
        self.assertEqual(res.speak.kind, "answer")
        self.assertEqual(res.speak.text, "北京明天晴，12到18度。")
        self.assertEqual(len(led.tasks), 1)

    async def test_status_while_running_says_still_working(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            await admit(_req("北京明天天气"), led)
            res = await admit(_req("查到了吗"), led)
        self.assertIsNone(res.job)
        self.assertIn("还在", res.speak.text)

    async def test_refine_city_bumps_version_and_new_job_carries_both(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            first = await admit(_req("上海明天天气"), led)
            complete_job(led, first.job, "上海明天多云。")  # type: ignore[arg-type]
            res = await admit(_req("改成北京的"), led)
        assert res.job is not None and first.job is not None
        self.assertEqual(res.route, "CONTROL")
        self.assertEqual(res.job.task_id, first.job.task_id)
        self.assertEqual(res.job.version, 2)
        self.assertIn("上海明天天气", res.job.objective)
        self.assertIn("改成北京的", res.job.objective)
        self.assertEqual(led.tasks[first.job.task_id].status, TaskStatus.RUNNING)
        self.assertEqual([t.version for t in led.superseded], [1])
        # the old worker's late answer is dropped
        self.assertFalse(complete_job(led, first.job, "上海明天多云。"))

    async def test_slot_fill_refines_the_needs_input_task(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            clar = await admit(_req("明天天气怎么样"), led)
            self.assertEqual(clar.speak.kind, "clarify")
            res = await admit(_req("北京的"), led)
        assert res.job is not None
        self.assertEqual(res.job.task_id, clar.speak.task_id)
        self.assertEqual(res.job.version, 2)
        self.assertIn("明天天气怎么样", res.job.objective)
        self.assertIn("北京的", res.job.objective)
        self.assertEqual(len(led.tasks), 1)

    async def test_joke_again_refines_same_task(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            j1 = await admit(_req("讲个笑话"), led)
            complete_job(led, j1.job, "笑话一。")  # type: ignore[arg-type]
            j2 = await admit(_req("再讲一个不一样的"), led)
        assert j2.job is not None
        self.assertEqual(j2.job.version, 2)

    async def test_fresh_request_after_task_creates_new_task(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            a = await admit(_req("现在几点了"), led)
            complete_job(led, a.job, "三点")  # type: ignore[arg-type]
            b = await admit(_req("科技新闻"), led)
        self.assertEqual(len(led.tasks), 2)
        self.assertNotEqual(a.job.task_id, b.job.task_id)  # type: ignore[union-attr]


class AdmitJevControlTests(unittest.IsolatedAsyncioTestCase):
    def _jev(self, route: str, control: str | None):
        async def fake(state, questions, **kw):  # noqa: ANN001, ANN003
            ans = {"route": {"type": "choice", "choice": route}}
            if control is not None:
                ans["control"] = {"type": "choice", "choice": control}
            return {"answers": ans}

        return fake

    async def test_jev_control_without_key_falls_back_to_heuristic(self) -> None:
        led = Ledger()
        rec = led.create("北京明天天气")
        led.mark_running(rec.task_id)
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client, "system_one", self._jev("CONTROL", None)),
        ):
            res = await admit(_req("取消刚才那个"), led)
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.CANCELLED)
        self.assertIsNone(res.job)

    async def test_jev_control_new_starts_new_task(self) -> None:
        led = Ledger()
        rec = led.create("北京明天天气")
        led.mark_running(rec.task_id)
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client, "system_one", self._jev("CONTROL", "NEW")),
        ):
            res = await admit(_req("现在几点了"), led)
        assert res.job is not None
        self.assertNotEqual(res.job.task_id, rec.task_id)
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.RUNNING)

    async def test_jev_stale_refine_is_noop(self) -> None:
        led = Ledger()
        rec = led.create("上海明天天气")
        led.refine(rec.task_id, "北京明天天气", version=1)
        led.mark_running(rec.task_id)
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client, "system_one", self._jev("CONTROL", f"REFINE:{rec.task_id}:1")),
        ):
            res = await admit(_req("改成广州的"), led)
        self.assertIsNone(res.job)
        self.assertEqual(led.tasks[rec.task_id].version, 2)
        self.assertEqual(led.tasks[rec.task_id].objective, "北京明天天气")

    async def test_jev_valid_refine_applies(self) -> None:
        led = Ledger()
        rec = led.create("上海明天天气")
        led.mark_running(rec.task_id)
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client, "system_one", self._jev("CONTROL", f"REFINE:{rec.task_id}:1")),
        ):
            res = await admit(_req("改成北京的"), led)
        assert res.job is not None
        self.assertEqual(res.job.version, 2)

    async def test_jev_status_key(self) -> None:
        led = Ledger()
        rec = led.create("北京明天天气")
        job = WorkerJob(task_id=rec.task_id, version=1, objective=rec.objective)
        complete_job(led, job, "晴")
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client, "system_one", self._jev("CONTROL", f"STATUS:{rec.task_id}:1")),
        ):
            res = await admit(_req("怎么样"), led)
        self.assertEqual(res.speak.text, "晴")
        self.assertIsNone(res.job)


if __name__ == "__main__":
    unittest.main()
