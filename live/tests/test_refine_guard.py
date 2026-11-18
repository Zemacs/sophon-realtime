"""v0.1.1: "same task, changed slot, said as a full new sentence" is the MODEL's call (JEV ``control`` question, with the user's own
words in the state). The code only enforces: the named task must exist and be live, the version must be current. No word lists.
(Round 2's ``slot_edit`` keyword guard is gone; live control accuracy is measured by tests/eval_control_judgment.py.)"""
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

from sophon_live.coordinator import AdmitRequest, Ledger, admit, api
from sophon_live.coordinator.route import build_route_questions
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


class ModelDrivenRefineTests(unittest.IsolatedAsyncioTestCase):
    def _req(self, text: str, raw: str = "") -> AdmitRequest:
        return AdmitRequest(session_id="sess_t", utterance_id="u", text=text, delegate_task=text, raw_text=raw)

    async def test_a_rewritten_sentence_is_a_refine_when_the_model_says_so(self) -> None:
        led, rec = _ledger()
        with mock.patch.object(api, "USE_JEV", True), \
                mock.patch.object(api.jev_client, "system_one", _jev("WORKER", f"REFINE:{rec.task_id}:1")):
            res = await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        self.assertEqual((res.via, res.control.op), ("jev", "REFINE"))
        self.assertEqual((res.job.task_id, res.job.version), (rec.task_id, 2))
        self.assertEqual(res.stale_task, (rec.task_id, 1))
        self.assertEqual(len(led.order), 1)

    async def test_the_model_sees_the_users_own_words_and_the_live_tasks(self) -> None:
        led, rec = _ledger()
        seen: dict = {}

        async def spy(state, questions, **kw):  # noqa: ANN001
            seen["state"], seen["questions"] = state, questions
            return {"answers": {"route": {"type": "choice", "choice": "WORKER"}, "control": {"type": "choice", "choice": "NEW"}}}

        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", spy):
            await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        self.assertEqual(seen["state"]["user_said"], "改成北京的。")
        self.assertEqual(seen["state"]["tasks"][0]["task_id"], rec.task_id)
        self.assertIn(f"REFINE:{rec.task_id}:1", seen["questions"]["control"]["criteria"])

    async def test_new_means_new_task_and_the_old_one_stays(self) -> None:
        led, _ = _ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", _jev("WORKER", "NEW")):
            res = await admit(self._req("讲个笑话"), led)
        self.assertEqual((res.via, res.route, res.control), ("jev", "WORKER", None))
        self.assertEqual(len(led.order), 2)

    async def test_a_stale_version_from_the_model_is_rejected_by_the_state_machine(self) -> None:
        led, rec = _ledger()
        with mock.patch.object(api, "USE_JEV", True), \
                mock.patch.object(api.jev_client, "system_one", _jev("CONTROL", f"REFINE:{rec.task_id}:7")):
            res = await admit(self._req("明天北京会下雨吗", "改成北京的。"), led)
        self.assertIsNone(res.job)
        self.assertIn("改过了", res.speak.text)
        self.assertEqual(led.tasks[rec.task_id].version, 1, "a version the ledger never had must not move the task")

    async def test_control_with_no_usable_option_is_a_fixed_clarify_not_a_guess(self) -> None:
        led, _ = _ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", _jev("CONTROL", None)):
            res = await admit(self._req("那个呢"), led)
        self.assertEqual((res.route, res.speak.kind, res.speak.text), ("CLARIFY", "clarify", api.DEGRADED_CLARIFY))
        self.assertEqual(len(led.order), 1)

    async def test_decision_log_line_is_machine_readable(self) -> None:
        led, rec = _ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", _jev("CONTROL", f"CANCEL:{rec.task_id}:1")), \
                self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            await admit(self._req("别查了"), led)
        line = next(l for l in cm.output if "coord admit" in l)
        m = re.search(r"coord admit session=(\S+) .*?route=(\S+) .*?decision=(\S+) via=(\S+) admit_ms=(\d+) jev_ms=(\S+)", line)
        self.assertEqual(m.group(1, 2, 3, 4), ("sess_t", "CONTROL", "CANCEL", "jev"))


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
