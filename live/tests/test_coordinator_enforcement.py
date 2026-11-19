"""Round A (M-1): what the CODE enforces around the router model's closed answers. The scripted router here is a test double for the model
(each test says what the model answered); nothing in production derives an answer from the sentence."""
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

from sophon_live import gateway, model_client
from sophon_live.coordinator import api, clarify
from sophon_live.coordinator.ledger import Ledger
from sophon_live.coordinator.schemas import AdmitRequest


def _answer(route="WORKER", control=None, tool=None, slot=None):
    a = {"route": {"type": "choice", "choice": route}}
    if control is not None:
        a["control"] = {"type": "choice", "choice": control}
    if tool is not None:
        a["tool"] = {"type": "choice", "choice": tool}
    if slot is not None:
        a["missing_slot"] = {"type": "choice", "choice": slot}
    return {"answers": a}


def _run(text, ledger, router, *, language="zh", raw_text=None):
    calls: list = []

    async def fake(state, questions, **kw):  # noqa: ANN001
        calls.append((state, questions))
        r = router(state, questions) if callable(router) else router
        if isinstance(r, Exception):
            raise r
        return r

    with mock.patch.object(api.jev_client, "system_one", fake):
        res = asyncio.run(api.admit(AdmitRequest(session_id="s", utterance_id="u", text=text, language=language, **({"raw_text": raw_text} if raw_text else {})), ledger))
    return res, calls


class DegradedTests(unittest.TestCase):
    def tearDown(self) -> None:
        model_client.reset()

    def test_router_down_with_a_live_task_asks_one_fixed_question_and_starts_nothing(self) -> None:
        led = Ledger()
        led.create("查一下上海的天气")
        led.mark_running(led.latest().task_id)
        for exc in (model_client.ModelUnavailable("jev", "timeout"), RuntimeError("boom")):
            res, _ = _run("换成北京的", led, exc)
            self.assertEqual((res.route, res.speak.kind, res.speak.text, res.job, res.via), ("CLARIFY", "clarify", api.DEGRADED_CLARIFY, None, "degraded"), exc)

    def test_router_down_without_a_live_task_hands_the_sentence_over_with_nothing_forced(self) -> None:
        res, _ = _run("今天几点了", Ledger(), model_client.ModelUnavailable("jev", "breaker_open"))
        self.assertEqual((res.route, res.via, res.degraded_reason), ("WORKER", "degraded", "breaker_open"))
        self.assertIsNotNone(res.job)
        self.assertEqual((res.job.require_tools, res.job.tool_judgment, res.job.explicit_save, res.speak.text), ([], "unknown", None, ""))

    def test_a_route_that_is_not_an_offered_option_degrades_it_is_never_mapped(self) -> None:
        res, _ = _run("今天几点了", Ledger(), {"answers": {"route": {"type": "choice", "choice": "weather"}}})
        self.assertEqual((res.via, res.degraded_reason), ("degraded", "non_option_answer"))


class RefineToolTests(unittest.TestCase):
    def _live(self, tools):
        led = Ledger()
        rec = led.create("查一下上海的天气", require_tools=tools, tool_judgment="model")
        led.mark_running(rec.task_id)
        return led, f"REFINE:{rec.task_id}:{rec.version}"

    def test_a_refine_keeps_the_tools_of_the_task_when_the_model_names_none(self) -> None:
        led, key = self._live(["get_weather"])
        res, _ = _run("换成北京", led, _answer("CONTROL", key, "NONE"))
        self.assertEqual((res.job.require_tools, res.job.tool_judgment), (["get_weather"], "model"))

    def test_a_refine_takes_the_new_tool_when_the_model_names_one(self) -> None:
        led, key = self._live(["get_weather"])
        res, _ = _run("改成看新闻", led, _answer("CONTROL", key, "get_news"))
        self.assertEqual((res.job.require_tools, res.job.tool_judgment), (["get_news"], "model"))
        self.assertEqual(led.tasks[res.job.task_id].require_tools, ["get_news"])

    def test_the_ledger_carries_tools_across_a_refine(self) -> None:
        led, _ = self._live(["get_weather"])
        rec = led.latest()
        new = led.refine(rec.task_id, "查一下北京的天气", version=rec.version)
        self.assertEqual((new.require_tools, new.tool_judgment), (["get_weather"], "model"))


class MissingSlotTests(unittest.TestCase):
    def tearDown(self) -> None:
        model_client.reset()

    def _clarify(self, slot, language="zh"):
        def router(state, questions):
            return _answer(slot=slot) if "missing_slot" in questions else _answer("CLARIFY")
        return _run("天气怎么样", Ledger(), router, language=language)

    def test_the_question_is_the_one_for_the_slot_the_model_named(self) -> None:
        for slot, text in (("city", clarify._ASK["city"]), ("query", clarify._ASK["query"]), ("topic", clarify._ASK["topic"])):
            res, _ = self._clarify(slot)
            self.assertEqual((res.route, res.speak.text), ("CLARIFY", text), slot)

    def test_other_an_unknown_option_or_no_answer_is_the_generic_ask_never_a_guess(self) -> None:
        for slot in ("other", "weather", "", None):
            res, _ = self._clarify(slot)
            self.assertEqual(res.speak.text, clarify._GENERIC, repr(slot))
        self.assertEqual([clarify.slots_from_answer(x) for x in ("city", "other", "bogus", None)], [["city"], [], [], []])

    def test_the_model_sees_the_users_own_utterance_and_the_language_the_caller_gave(self) -> None:
        _, calls = self._clarify("city", language="en")
        state, qs = calls[-1]
        self.assertEqual((state["utterance"], state["language"], list(qs)), ("天气怎么样", "en", ["missing_slot"]))

    def test_no_language_is_invented_when_the_caller_gave_none(self) -> None:
        _, calls = self._clarify("city", language="")
        self.assertNotIn("language", calls[-1][0])


class InvalidAnswerVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        api.JUDGE_STATS.clear()

    def tearDown(self) -> None:
        api.JUDGE_STATS.clear()
        model_client.reset()

    def test_an_unusable_tool_answer_is_counted_traced_and_shown_in_health(self) -> None:
        res, _ = _run("今天几点了", Ledger(), _answer("WORKER", tool="made_up"))
        self.assertEqual((res.judge_invalid, res.job.require_tools, res.job.tool_judgment), (["tool:invalid"], [], "unknown"))
        res, _ = _run("今天几点了", Ledger(), _answer("WORKER"))
        self.assertEqual(res.judge_invalid, ["tool:missing"])
        body = json.loads(gateway._health_body("detail=1"))
        self.assertEqual(body["router_invalid_answers"], {"tool_invalid": 1, "tool_missing": 1})


if __name__ == "__main__":
    unittest.main()
