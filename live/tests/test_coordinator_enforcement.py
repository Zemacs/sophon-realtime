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
        body = json.loads(gateway._health_body("detail=1", ("127.0.0.1", 1)))
        self.assertEqual(body["router_invalid_answers"], {"tool_invalid": 1, "tool_missing": 1})


def _raw(route="WORKER", control=None, tool=None, **extra):
    """An answer body with arbitrary JSON in the ``choice`` fields (the router is a model: it can send any shape)."""
    a = {"route": {"type": "choice", "choice": route}}
    if control is not None:
        a["control"] = {"type": "choice", "choice": control}
    if tool is not None:
        a["tool"] = {"type": "choice", "choice": tool}
    a.update(extra)
    return {"answers": a}


class WrongShapedAnswersTests(unittest.TestCase):
    """R-A' / Critic I-2: one bad field never takes the others down, is never guessed, and is never silent."""

    def setUp(self) -> None:
        api.JUDGE_STATS.clear()

    def tearDown(self) -> None:
        api.JUDGE_STATS.clear()
        model_client.reset()

    def _live(self):
        led = Ledger()
        rec = led.create("查一下上海的天气", require_tools=["get_weather"], tool_judgment="model")
        led.mark_running(rec.task_id)
        return led, f"REFINE:{rec.task_id}:{rec.version}"

    def test_1_a_non_string_control_answer_does_not_raise_and_is_counted(self) -> None:
        for bad in (5, ["REFINE"], {"k": 1}, True):
            led, _ = self._live()
            res, _ = _run("换成北京", led, _raw("CONTROL", control=bad, tool="NONE"))
            self.assertEqual((res.via, res.route, res.degraded_reason), ("degraded", "CLARIFY", "control_without_target"), bad)
            self.assertEqual(res.speak.text, api.DEGRADED_CLARIFY, bad)
            self.assertIn("control:invalid", res.judge_invalid, bad)
        self.assertEqual(api.JUDGE_STATS.get("control_invalid"), 4)

    def test_2_a_non_string_tool_answer_keeps_the_route_and_the_control_and_is_counted(self) -> None:
        for bad in (["get_weather"], 7, {"x": 1}):
            led, key = self._live()
            res, _ = _run("换成北京", led, _raw("CONTROL", control=key, tool=bad))
            self.assertEqual(res.via, "jev", bad)                                  # not "heuristic", not degraded: the route and control answers were fine
            self.assertEqual(res.control.op, "REFINE", bad)                        # the control answer still applies
            self.assertEqual((res.job.require_tools, res.job.tool_judgment), (["get_weather"], "model"), bad)   # the task keeps its own tools
            self.assertEqual(res.judge_invalid, ["tool:invalid"], bad)
        res, _ = _run("今天几点了", Ledger(), _raw("WORKER", tool=["x"]))
        self.assertEqual((res.via, res.job.tool_judgment, res.judge_invalid), ("jev", "unknown", ["tool:invalid"]))
        self.assertEqual(api.JUDGE_STATS.get("tool_invalid"), 4)

    def test_3_a_wrongly_shaped_response_or_route_degrades_with_a_trace_and_never_raises(self) -> None:
        for resp in ({"answers": [1, 2]}, {"answers": {"route": "WORKER"}}, {"answers": {"route": {"type": "choice", "choice": ["WORKER"]}}},
                     {"answers": {"route": {"type": "choice", "choice": 3}}}, [], "WORKER", None):
            res, _ = _run("今天几点了", Ledger(), resp)
            self.assertEqual((res.via, res.degraded_reason, res.route), ("degraded", "non_option_answer", "WORKER"), resp)
            self.assertTrue(res.judge_invalid, resp)
        res, _ = _run("今天几点了", Ledger(), _raw("weather"))
        self.assertEqual((res.via, res.judge_invalid), ("degraded", ["route:invalid"]))
        self.assertGreaterEqual(api.JUDGE_STATS.get("route_invalid", 0), 1)

    def test_4_control_with_nothing_to_control_goes_through_the_degraded_new_task_branch_and_keeps_the_models_tool(self) -> None:
        res, _ = _run("今天几点了", Ledger(), _raw("CONTROL", tool="get_datetime"))          # no live task: the control question is not even asked
        self.assertEqual((res.route, res.via, res.degraded_reason), ("WORKER", "degraded", "control_without_target"))
        self.assertEqual((res.job.require_tools, res.job.tool_judgment, res.speak.kind), (["get_datetime"], "model", "pad"))
        self.assertEqual(res.judge_invalid, [])                                             # nothing was invalid: the model simply had no target to name
        led, _ = self._live()                                                               # a live task and no usable target: one fixed question
        res, _ = _run("嗯", led, _raw("CONTROL", control="NOPE:1:2", tool="NONE"))
        self.assertEqual((res.route, res.speak.text, res.job, res.judge_invalid), ("CLARIFY", api.DEGRADED_CLARIFY, None, ["control:invalid"]))

    def test_5_a_control_string_that_is_not_an_offered_key_is_counted_and_not_applied(self) -> None:
        for key in ("REFINE:x:abc", "BOGUS:t:1", "REFINE", "a:b:c:d"):
            led, _ = self._live()
            res, _ = _run("换成北京", led, _raw("WORKER", control=key, tool="get_weather"))
            self.assertIsNone(res.control, key)
            self.assertEqual((res.route, res.judge_invalid), ("WORKER", ["control:invalid"]), key)

    def test_6_the_client_helpers_never_raise_on_a_wrong_shape(self) -> None:
        from sophon_live.coordinator import jev_client
        for resp in (None, [], "x", {"answers": None}, {"answers": []}, {"answers": {"k": "s"}}, {"answers": {"k": {"type": "noul", "noul": "x"}}}):
            self.assertIsNone(jev_client.choice_answer(resp, "k"), resp)
            self.assertIsNone(jev_client.raw_choice(resp, "k"), resp)
            self.assertIsNone(jev_client.noul_prob(resp, "k"), resp)
        self.assertEqual(jev_client.raw_choice(_raw(tool=["a"]), "tool"), ["a"])
        self.assertIsNone(jev_client.choice_answer(_raw(tool=["a"]), "tool"))
        self.assertEqual(jev_client.choice_answer(_raw(tool="NONE"), "tool"), "NONE")
        for key in (5, ["REFINE:a:1"], {"a": 1}, True, None, ""):
            self.assertIsNone(api._parse_control_key(key), key)


if __name__ == "__main__":
    unittest.main()
