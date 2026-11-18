"""Round 2 / B: a slot-only edit of the latest task is a REFINE even when JEV says NEW."""
from __future__ import annotations

import asyncio
import re
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import AdmitRequest, Ledger, admit, api, slots
from sophon_live.coordinator.route import current_ask, slot_edit
from sophon_live.coordinator.schemas import TaskStatus


def _ledger(objective: str = "明天上海会下雨吗", status: TaskStatus = TaskStatus.RUNNING):
    led = Ledger()
    rec = led.create(objective)
    led.mark_running(rec.task_id)
    if status == TaskStatus.COMPLETED:
        led.mark_completed(rec.task_id, "上海明天晴。")
    elif status == TaskStatus.DELIVERED:
        led.mark_completed(rec.task_id, "上海明天晴。")
        led.mark_delivered(rec.task_id, rec.version)
    elif status == TaskStatus.CANCELLED:
        led.cancel(rec.task_id, rec.version)
    return led, rec


def _jev(route: str, control: str | None, tool: str | None = None, faithful: str | None = None):
    """A scripted router: the test says what the MODEL answers (route / control / which tool / is the front's wording faithful)."""
    async def fake(state, questions, **kw):  # noqa: ANN001
        ans = {"route": {"type": "choice", "choice": route}}
        if control is not None:
            ans["control"] = {"type": "choice", "choice": control}
        if tool is not None:
            ans["tool"] = {"type": "choice", "choice": tool}
        if faithful is not None:
            ans["task_faithful"] = {"type": "choice", "choice": faithful}
        return {"answers": ans}

    return fake


class SlotEditTests(unittest.TestCase):
    def test_city_swap_on_a_running_task(self) -> None:
        led, rec = _ledger()
        d = slot_edit("明天北京会下雨吗", "改成北京的。", led)
        self.assertEqual((d.op, d.task_id, d.version, d.via), ("REFINE", rec.task_id, 1, "guard"))

    def test_city_swap_without_raw_words_needs_the_same_shape(self) -> None:
        led, rec = _ledger()
        self.assertIsNotNone(slot_edit("明天北京会下雨吗", "", led))

    def test_works_on_completed_and_delivered_tasks(self) -> None:
        for st in (TaskStatus.COMPLETED, TaskStatus.DELIVERED):
            led, rec = _ledger(status=st)
            self.assertEqual(slot_edit("明天北京会下雨吗", "", led).task_id, rec.task_id, st)

    def test_day_swap(self) -> None:
        led, rec = _ledger()
        self.assertEqual(slot_edit("后天上海会下雨吗", "", led).op, "REFINE")

    def test_explicit_edit_words_are_enough_when_the_front_rewrote_the_shape(self) -> None:
        led, _ = _ledger()
        self.assertIsNone(slot_edit("北京天气怎么样", "", led))  # a different shape, no cue: JEV decides
        self.assertEqual(slot_edit("北京天气怎么样", "改成北京的", led).op, "REFINE")

    def test_not_a_slot_edit(self) -> None:
        led, _ = _ledger()
        self.assertIsNone(slot_edit("明天上海会下雨吗", "", led), "same city, nothing changed")
        self.assertIsNone(slot_edit("讲个笑话", "", led))
        self.assertIsNone(slot_edit("上海和北京明天都会下雨吗", "", led), "two cities: ambiguous")
        self.assertIsNone(slot_edit("今天北京有什么新闻", "", led), "different request that merely names a city")

    def test_task_without_slots_or_dead_tasks_are_never_edited(self) -> None:
        led, _ = _ledger("讲个笑话")
        self.assertIsNone(slot_edit("明天北京会下雨吗", "改成北京的", led))
        led, _ = _ledger(status=TaskStatus.CANCELLED)
        self.assertIsNone(slot_edit("明天北京会下雨吗", "改成北京的", led))
        self.assertIsNone(slot_edit("明天北京会下雨吗", "改成北京的", Ledger()))

    def test_only_the_latest_task_qualifies(self) -> None:
        led, _ = _ledger()
        newer = led.create("今天有什么新闻")
        led.mark_running(newer.task_id)
        self.assertIsNone(slot_edit("明天北京会下雨吗", "", led))

    def test_refined_objective_edits_the_newest_wording(self) -> None:
        self.assertEqual(current_ask("明天上海会下雨吗。用户改口，以新说法为准：明天北京会下雨吗"), "明天北京会下雨吗")
        led, rec = _ledger()
        led.refine(rec.task_id, "明天上海会下雨吗。用户改口，以新说法为准：明天北京会下雨吗", version=1)
        self.assertEqual(slot_edit("明天广州会下雨吗", "", led).version, 2)

    def test_city_list_matches_the_weather_tool(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("w", _LIVE_ROOT / "sophon_live" / "assistant_tools" / "weather.py")
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)  # type: ignore[union-attr]
        self.assertEqual(set(slots.CITY_NAMES), set(w.CITIES))


class AdmitGuardTests(unittest.IsolatedAsyncioTestCase):
    def _req(self, text: str, raw: str = "") -> AdmitRequest:
        return AdmitRequest(session_id="sess_t", utterance_id="u", text=text, delegate_task=text, raw_text=raw)

    async def test_guard_beats_a_jev_that_says_new_and_never_calls_it(self) -> None:
        led, rec = _ledger()
        boom = mock.AsyncMock(side_effect=AssertionError("JEV must not be asked"))
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", True), \
                mock.patch.object(api.jev_client, "system_one", boom):
            res = await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        self.assertEqual((res.route, res.via, res.control.op), ("CONTROL", "guard", "REFINE"))
        self.assertEqual(res.job.task_id, rec.task_id)
        self.assertEqual(res.job.version, 2)
        self.assertIn("以新说法为准：明天北京会下雨吗", res.job.objective)
        self.assertEqual(res.stale_task, (rec.task_id, 1))
        self.assertEqual(len(led.order), 1)  # one task, now v2 — not a second task

    async def test_without_the_guard_jev_new_leaves_two_tasks(self) -> None:
        led, rec = _ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False), \
                mock.patch.object(api.jev_client, "system_one", _jev("WORKER", "NEW")):
            res = await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        self.assertEqual((res.via, res.control), ("jev", None))
        self.assertEqual(len(led.order), 2)  # the old Shanghai task is still alive: the bug being fixed

    async def test_unrelated_utterance_still_goes_to_jev(self) -> None:
        led, _ = _ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", True), \
                mock.patch.object(api.jev_client, "system_one", _jev("WORKER", "NEW")):
            res = await admit(self._req("讲个笑话"), led)
        self.assertEqual((res.via, res.route), ("jev", "WORKER"))

    async def test_decision_log_line_is_machine_readable(self) -> None:
        led, rec = _ledger()
        with mock.patch.object(api, "USE_JEV", True), self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        line = next(l for l in cm.output if "coord admit" in l)
        m = re.search(r"coord admit session=(\S+) .*?route=(\S+) .*?decision=(\S+) via=(\S+) admit_ms=(\d+) jev_ms=(\S+)", line)
        self.assertEqual(m.group(1, 2, 3, 4), ("sess_t", "CONTROL", "REFINE", "guard"))


class GatewayRawTextTests(unittest.TestCase):
    def test_recent_user_text_expires(self) -> None:
        from tests.test_r0_coord_boundary import _bare_session

        s = _bare_session()
        self.assertEqual(s._recent_user_text(), "")
        s.last_user = ("改成北京的。", time.monotonic())
        self.assertEqual(s._recent_user_text(), "改成北京的。")
        s.last_user = ("改成北京的。", time.monotonic() - 60)
        self.assertEqual(s._recent_user_text(), "")


if __name__ == "__main__":
    unittest.main()
