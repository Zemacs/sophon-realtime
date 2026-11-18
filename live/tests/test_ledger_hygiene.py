"""Ledger hygiene found in the Round 1 live run: abandoned clarifications must not capture later short utterances."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from tests import jev_standin  # noqa: E402  (scripted JEV for control-flow tests)
from sophon_live.coordinator import AdmitRequest, Ledger, admit, api
from sophon_live.coordinator.route import heuristic_control
from sophon_live.coordinator.schemas import TaskStatus


def _req(text: str) -> AdmitRequest:
    return AdmitRequest(session_id="t", utterance_id="u", text=text, delegate_task=text)


class HygieneTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_request_supersedes_an_open_clarification(self) -> None:
        with jev_standin.standin():
            led = Ledger()
            clar = await admit(_req("明天天气怎么样"), led)
            tid = clar.speak.task_id
            await admit(_req("现在几点了"), led)
        self.assertIsNone(led.pending_input())
        self.assertEqual(led.tasks[tid].status, TaskStatus.SUPERSEDED)

    async def test_ack_after_abandoned_clarification_is_not_a_slot_answer(self) -> None:
        with jev_standin.standin():
            led = Ledger()
            await admit(_req("明天天气怎么样"), led)
            await admit(_req("北京明天天气"), led)  # a complete new request
            res = await admit(_req("好的"), led)
        self.assertEqual(res.route, "FAST")
        self.assertIsNone(res.job)

    def test_acknowledgements_never_fill_a_pending_slot(self) -> None:
        led = Ledger()
        rec = led.create("明天天气怎么样")
        led.mark_needs_input(rec.task_id)
        for t in ("好的", "谢谢", "嗯", "行", "知道了", "好的谢谢"):
            self.assertIsNone(heuristic_control(t, led), t)
        self.assertIsNotNone(heuristic_control("北京的", led))

    async def test_superseded_clarification_is_not_resurrected_by_refine(self) -> None:
        led = Ledger()
        rec = led.create("明天天气怎么样")
        led.mark_needs_input(rec.task_id)
        led.supersede_pending_input()
        self.assertIsNone(led.refine(rec.task_id, "x"))


if __name__ == "__main__":
    unittest.main()
