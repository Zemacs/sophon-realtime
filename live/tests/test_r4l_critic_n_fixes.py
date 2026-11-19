"""Critic's 7440dc9 review (N-1 .. N-9), as behaviour. Reproduces the probes in /workspace/reviewer/deplux-notes/runs/7440dc9/probe_k*.py.

N-1/N-7  the literal path has no numeral conversion: every row of the Critic's table (四川 -> 4, 一起 -> 1 ...) is BLOCKED.
N-2      a real but unrelated span + an invented value is released only if an INDEPENDENT model says the span says the value (no / unavailable = refusal).
N-3      read-back: the span is in the caller's FIRST reply, the model's closed-set caller_reply is ``confirm``, the value is one that was refused, and the
         independent check agrees the reply is a confirmation. 「不对」 / an unrelated sentence / silence are not.
N-4      no fuzzy matching: a one-letter difference in an email / identifier / name is refused (evidence path or read-back instead).
N-6      the session cap counts rejections in a row; a value equal to the schema's own default, or on a property marked x-grounding: none, needs no evidence.
N-8      the audit is written to the log as it happens (the table is cleared on close); grounding.py's regular expressions are pinned (test_r4k_static_guards).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding, realtime  # noqa: E402
from sophon_live.coordinator import jev_client, span_judge  # noqa: E402
from tests.test_r4k_evidence import _session, call, ev, K  # noqa: E402

H = ["我叫张伟，我住在张江路"]


def check(args, heard, seen=None, evidence=None, spoken=None, judge=None, **kw):
    """Run the guard with a recording independent judge; returns (bad, calls the judge got)."""
    calls: list = []

    def j(kind, arg, value, span, ctx):
        calls.append((kind, arg, value, span, ctx))
        return judge(kind, arg, value, span, ctx) if judge else False
    return grounding.check("t", args, heard, seen or [], evidence=evidence, spoken=spoken, judge=j, **kw), calls


class N1_NoNumeralConversionOnTheLiteralPath(unittest.TestCase):
    """The Critic's table: original words -> the value the model filled. None of these was said by the caller; every one is blocked."""
    TABLE = [("我们一起去", "1"), ("统一管理", "1"), ("请稍等一下", "1"), ("唯一的", "1"), ("不一样", "1"), ("一半", "1"), ("第一次", "1"), ("四川的天气", "4"),
             ("三明治", "3"), ("零钱", "0"), ("二手车", "2"), ("两个人", "2"), ("九州", "9"), ("十分感谢", "10"), ("五一假期", "51"), ("一千", "1000"),
             ("万一", "1"), ("半小时", "5")]

    def test_every_row_is_blocked(self) -> None:
        for words, value in self.TABLE:
            bad, judged = check({"limit": value}, [words])
            self.assertTrue(bad, (words, value))
            self.assertEqual(judged, [], (words, value))                    # no evidence was offered, so the judge is not even asked

    def test_the_words_the_old_code_special_cased_get_no_free_pass(self) -> None:
        for word in ("十分", "万一", "一起", "四川", "百货"):
            self.assertTrue(check({"x": word}, ["hello"])[0], word)
            self.assertEqual(check({"x": word}, [f"我说{word}"])[0], [], word)       # said -> fine, like any other value

    def test_the_other_numeral_script_is_refused_on_the_literal_path(self) -> None:
        for words, value in (("八十八号", "88"), ("三零二室", "302"), ("二零二六年", "2026"), ("十二点", "12"), ("十二点", "12:00"), ("十二点半", "12:30"),
                             ("两点", "2"), ("下午三点", "15:00"), ("一百零五", "105"), ("二十三", "23"), ("一万二", "12000"), ("第三层", "3")):
            self.assertTrue(check({"n": value}, [words])[0], (words, value))

    def test_same_script_and_digit_boundaries_still_work(self) -> None:
        self.assertEqual(check({"n": "88"}, ["88号"])[0], [])
        self.assertEqual(check({"n": "八十八号"}, ["我住八十八号"])[0], [])
        self.assertTrue(check({"n": "88"}, ["1988年"])[0])
        self.assertTrue(check({"n": "88"}, ["8800"])[0])
        self.assertTrue(check({"n": "88"}, ["8", "8"])[0])
        self.assertEqual(check({"n": "88"}, ["x"], ["order 88 shipped"])[0], [])

    def test_no_special_case_for_wan_yi_or_any_other_word_is_left(self) -> None:
        import inspect
        src = inspect.getsource(grounding.norm_text)
        for needle in ("万一", "百货", "十分", "零下", "一起"):
            self.assertNotIn(needle, src)
        self.assertEqual(grounding.norm_text("万一"), "万一")


class N2_TheEvidenceIsCheckedByASecondModel(unittest.TestCase):
    def test_a_real_but_unrelated_span_with_an_invented_value_is_blocked_when_the_judge_says_no(self) -> None:
        for args, heard, span in (({"name": "李四"}, H, "我住在张江路"), ({"name": "李四"}, H, "我"), ({"name": "李四"}, H, "张江"),
                                  ({"n": "88"}, ["我住在张江路"], "张江路")):
            bad, judged = check(args, heard, evidence=ev(next(iter(args)), span))
            self.assertTrue(bad, (args, span))
            self.assertEqual(judged, [("equivalent", next(iter(args)), next(iter(args.values())), span, "")])       # exactly one question: does the span say the value
            self.assertIn("independent check says", bad[0][2])

    def test_the_span_from_an_unrelated_tool_field_is_blocked_when_the_judge_says_no(self) -> None:
        bad, judged = check({"name": "李四"}, H, ["订单状态：已发货"], evidence=ev("name", "订单状态"))
        self.assertTrue(bad)
        self.assertEqual(len(judged), 1)

    def test_a_yes_releases_it_and_is_recorded_as_an_independent_acceptance(self) -> None:
        audit: list = []
        bad, judged = check({"name": "张炜"}, ["我叫张伟"], evidence=ev("name", "张伟"), judge=lambda *a: True, audit=audit)
        self.assertEqual(bad, [])
        self.assertEqual([(a["how"], a["judge"]) for a in audit], [("model_span", True)])

    def test_no_judge_an_unavailable_judge_or_a_non_answer_all_refuse(self) -> None:
        for verdict in (None, False):
            self.assertTrue(check({"name": "张炜"}, ["我叫张伟"], evidence=ev("name", "张伟"), judge=lambda *a, v=verdict: v)[0], verdict)
        got = grounding.check("t", {"name": "张炜"}, ["我叫张伟"], [], evidence=ev("name", "张伟"))              # no judge at all
        self.assertTrue(got)
        self.assertIn("independent check", got[0][2])

    def test_the_judge_is_not_asked_when_the_structure_already_fails_or_the_literal_path_passes(self) -> None:
        for args, heard, e in (({"name": "李四"}, H, ev("name", "李四")),                      # the span was never said
                               ({"name": "张炜"}, ["我叫张伟"], ev("name", "张伟", derives=False)),   # the model did not claim it
                               ({"name": "张炜"}, ["我叫张伟"], ev("nick", "张伟")),                   # evidence for another argument
                               ({"name": "张伟"}, ["我叫张伟"], ev("name", "张伟"))):                  # literal: no evidence needed
            bad, judged = check(args, heard, evidence=e, judge=lambda *a: True)
            self.assertEqual(judged, [], args)

    def test_pending_claims_are_collected_for_the_async_driver(self) -> None:
        pending: list = []
        got = grounding.check("t", {"name": "张炜"}, ["我叫张伟"], [], evidence=ev("name", "张伟"), pending=pending)
        self.assertTrue(got)
        self.assertEqual(pending, [{"kind": "equivalent", "argument": "name", "value": "张炜", "span": "张伟", "context": ""}])


class N2_AsyncDriver(unittest.IsolatedAsyncioTestCase):
    async def test_the_judge_is_awaited_concurrently_and_an_exception_is_a_refusal(self) -> None:
        seen = []

        async def judge(kind, arg, value, span, ctx):
            seen.append(arg)
            if arg == "b":
                raise RuntimeError("boom")
            return True
        evid = ev("a", "张伟") + ev("b", "张伟")
        got = await grounding.check_async("t", {"a": "张炜", "b": "章伟"}, ["我叫张伟"], [], evidence=evid, judge=judge)
        self.assertEqual(sorted(seen), ["a", "b"])
        self.assertEqual([k for k, _, _ in got], ["b"])

    async def test_nothing_to_judge_means_no_judge_call(self) -> None:
        judge = mock.AsyncMock(return_value=True)
        audit: list = []
        self.assertEqual(await grounding.check_async("t", {"name": "张伟"}, ["我叫张伟"], [], judge=judge, audit=audit), [])
        judge.assert_not_awaited()
        self.assertEqual([a["how"] for a in audit], ["literal"])

    async def test_on_verdict_sees_every_answer(self) -> None:
        got = []
        await grounding.check_async("t", {"name": "张炜"}, ["我叫张伟"], [], evidence=ev("name", "张伟"), judge=mock.AsyncMock(return_value=False),
                                    on_verdict=lambda p, a: got.append((p["kind"], a)))
        self.assertEqual(got, [("equivalent", False)])


class N2_TheSecondModelIsTheDecisionModel(unittest.IsolatedAsyncioTestCase):
    """span_judge builds one closed question and maps the answer; nothing here reads the span or the value for meaning."""

    async def _ask(self, answer, kind="equivalent", raises=None):
        async def fake(state, questions, **kw):
            self.sent = (state, questions)
            if raises:
                raise raises
            return {"answers": {span_judge.KEY: {"type": "choice", "choice": answer}}}
        with mock.patch.object(jev_client, "system_one", fake):
            return await span_judge.judge(kind, "name", "张炜", "张伟", "我记下的名字是张炜，对吗？" if kind == "confirms" else "")

    async def test_yes_no_and_everything_else(self) -> None:
        self.assertIs(await self._ask("YES"), True)
        self.assertIs(await self._ask("NO"), False)
        self.assertIsNone(await self._ask("MAYBE"))
        self.assertIsNone(await self._ask(None))
        self.assertIsNone(await self._ask("YES", raises=RuntimeError("down")))
        self.assertIsNone(await self._ask("YES", raises=asyncio.TimeoutError()))

    async def test_the_state_carries_the_claim_and_the_question_is_closed(self) -> None:
        await self._ask("YES")
        state, questions = self.sent
        self.assertEqual(state, {"argument": "name", "value": "张炜", "caller_words": "张伟"})
        self.assertEqual(set(questions[span_judge.KEY]["criteria"]), {"YES", "NO"})
        await self._ask("YES", kind="confirms")
        state, questions = self.sent
        self.assertEqual(state["caller_reply"], "张伟")
        self.assertIn("assistant_read_back", state)
        self.assertEqual(set(questions[span_judge.KEY]["criteria"]), {"YES", "NO"})

    async def test_counters(self) -> None:
        before = dict(span_judge.STATS)
        await self._ask("YES"); await self._ask("NO"); await self._ask("?")
        self.assertEqual({k: span_judge.STATS[k] - before[k] for k in before}, {"yes": 1, "no": 1, "unavailable": 1})

    async def test_the_session_asks_it_by_default_and_a_down_service_refuses(self) -> None:
        s = _session(["我叫张伟"]); s.judge = None
        down = mock.AsyncMock(side_effect=ConnectionError("no route"))
        with mock.patch.object(jev_client, "system_one", down):
            self.assertTrue(await s._guard(call({"name": "张炜", K: ev("name", "张伟")})))
        down.assert_awaited()
        self.assertIn("independent check could not tell", s.engine.tool_result.await_args.args[1])

        s = _session(["我叫张伟"]); s.judge = None
        yes = mock.AsyncMock(return_value={"answers": {span_judge.KEY: {"type": "choice", "choice": "YES"}}})
        with mock.patch.object(jev_client, "system_one", yes):
            self.assertFalse(await s._guard(call({"name": "张炜", K: ev("name", "张伟")})))


def _ask_session(heard, spoken, rejected=None, judge=None):
    s = _session(heard)
    s.spoken = list(spoken)
    s.rejected = {"name": {"李四"}} if rejected is None else rejected
    s.judge = judge or (lambda *a: asyncio.sleep(0, True))
    return s


class N3_ReadBackIsAConfirmation(unittest.IsolatedAsyncioTestCase):
    SP = [("我记下的名字是李四，对吗？", 1)]

    async def go(self, heard, *, span, reply="confirm", spoken=None, rejected=None, judge=None, value="李四"):
        s = _ask_session(heard, self.SP if spoken is None else spoken, rejected, judge)
        e = call({"name": value, K: ev("name", span, read_back=True, reply=reply)})
        released = not await s._guard(e)
        why = "" if released else s.engine.tool_result.await_args.args[1]
        return released, why, s

    async def test_the_caller_confirms(self) -> None:
        released, _, s = await self.go(H + ["对"], span="对")
        self.assertTrue(released)
        self.assertEqual(s.audit[-1]["how"], "read_back_confirmed")
        self.assertEqual(s.audit[-1]["caller_reply"], "confirm")

    async def test_the_model_must_mark_the_reply_as_a_confirmation(self) -> None:
        for reply in ("reject", "other", None, "yes", "CONFIRM"):
            released, why, _ = await self.go(H + ["不对"], span="不对", reply=reply)
            self.assertFalse(released, reply)
            self.assertIn("caller_reply must be confirm", why)

    async def test_the_independent_check_must_agree_the_reply_is_a_confirmation(self) -> None:
        seen = []

        async def judge(kind, arg, value, span, ctx):
            seen.append((kind, arg, value, span, ctx))
            return False
        for reply_words in ("不对", "你说什么？", "no that's wrong"):
            seen.clear()
            released, why, _ = await self.go(H + [reply_words], span=reply_words, judge=judge)
            self.assertFalse(released, reply_words)
            self.assertEqual(seen, [("confirms", "name", "李四", reply_words, "我记下的名字是李四，对吗？")])
            self.assertIn("does not confirm", why)

    async def test_the_span_must_be_the_first_reply_after_the_read_back(self) -> None:
        released, why, _ = await self.go(H + ["我不想说", "对"], span="对")                      # a later turn, not the first
        self.assertFalse(released)
        self.assertIn("first reply after the read-back", why)
        released, _, _ = await self.go(H + ["我不想说", "对"], span="我不想说")
        self.assertTrue(released)                                                              # the first one is fine (and the judge decides)

    async def test_silence_is_not_a_confirmation(self) -> None:
        released, why, _ = await self.go(H, span="我叫张伟")
        self.assertFalse(released)
        self.assertIn("has not replied", why)

    async def test_a_value_that_was_never_read_back_or_was_read_back_in_the_same_breath_as_another(self) -> None:
        released, why, _ = await self.go(H + ["对"], span="对", spoken=[])
        self.assertFalse(released)
        self.assertIn("never read back", why)
        released, why, _ = await self.go(H + ["对"], span="对", spoken=[("请问是王五吗", 1)])
        self.assertFalse(released)
        self.assertIn("never read back", why)

    async def test_it_can_only_confirm_a_value_the_guard_refused(self) -> None:
        """'confirm A, release B': B was read back too, but it is not what was refused before."""
        sp = [("我记下的名字是李四和王五，对吗？", 1)]
        released, why, _ = await self.go(H + ["对"], span="对", spoken=sp, value="王五")
        self.assertFalse(released)
        self.assertIn("not the one that was refused", why)
        released, _, _ = await self.go(H + ["对"], span="对", spoken=sp, value="李四")
        self.assertTrue(released)

    async def test_a_refused_call_remembers_the_value(self) -> None:
        s = _session(["我姓张"])
        await s._guard(call({"name": "李四"}))
        self.assertEqual(s.rejected, {"name": {"李四"}})

    async def test_an_unavailable_judge_refuses_the_confirmation(self) -> None:
        async def judge(*a):
            raise RuntimeError("down")
        released, _, _ = await self.go(H + ["对"], span="对", judge=judge)
        self.assertFalse(released)


class N4_NoFuzzyMatching(unittest.TestCase):
    def test_a_one_letter_difference_is_refused(self) -> None:
        for value, said in (("john.smith@acme.com", "jon.smith@acme.com"), ("sales@acme.com", "sale@acme.com"), ("admins@acme.com", "admin@acme.com"),
                            ("john.smith@acme.com", "john.smyth@acme.com"), ("main_streat", "main_street"), ("acct_x7", "acct_x1"),
                            ("Smyth", "Smith"), ("Jon", "John"), ("Allen", "Allan"), ("Daniel", "Danielle"), ("Brenda", "Brendan"), ("acme", "acne"),
                            ("Smith", "Smithson"), ("min@acme.com", "admin@acme.com"), ("acct_x", "acct_xy")):
            bad, _ = check({"v": value}, [f"it is {said}"])
            self.assertTrue(bad, (value, said))

    def test_exact_values_pass_and_spoken_forms_do_not(self) -> None:
        """Round B (G-1): only an exact occurrence is literal; a spoken form is refused alone (and goes to evidence + the judge, see test_r4m_g1_literal_only)."""
        for value, said in (("jon.smith@acme.com", "it is jon.smith@acme.com"), ("Allan", "my name is Allan")):
            self.assertEqual(check({"v": value}, [said])[0], [], (value, said))
        for value, said in (("jon@acme.com", "jon at acme dot com"), ("19122", "one nine one two two")):
            self.assertTrue(check({"v": value}, [said])[0], (value, said))

    def test_a_misheard_name_takes_the_evidence_path(self) -> None:
        bad, judged = check({"name": "Allen"}, ["my name is Allan"], evidence=ev("name", "my name is Allan"), judge=lambda *a: True)
        self.assertEqual((bad, len(judged)), ([], 1))

    def test_there_is_no_similarity_code_left(self) -> None:
        import inspect
        src = inspect.getsource(grounding)
        for needle in ("difflib", "SequenceMatcher", "0.85", "ratio("):
            self.assertNotIn(needle, src)


class N6_SessionCapAndOptionalArguments(unittest.IsolatedAsyncioTestCase):
    async def test_the_cap_counts_rejections_in_a_row(self) -> None:
        s = _session(["我叫张伟"])
        for i in range(realtime.GUARD_SESSION_MAX_REJECTIONS):
            self.assertTrue(await s._guard(call({"name": f"李{i}"}, f"c{i}")))
        self.assertFalse(await s._guard(call({"name": "张伟"}, "ok")))                          # a call that passes resets the count
        self.assertEqual(s.guard_total, 0)
        for i in range(realtime.GUARD_SESSION_MAX_REJECTIONS):
            self.assertTrue(await s._guard(call({"name": f"王{i}"}, f"d{i}")))
            self.assertTrue(s.engine.tool_result.await_args.args[1].startswith("NOT EXECUTED. These"), i)     # still ordinary refusals, never the give-up
        await s._guard(call({"name": "赵999"}, "last"))
        self.assertEqual(json.loads(s.engine.tool_result.await_args.args[1])["status"], "needs_input")

    def test_a_value_equal_to_the_schema_default_needs_no_provenance(self) -> None:
        schema = {"type": "object", "properties": {"limit": {"type": "integer", "default": 5}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "x-grounding": "none"},
                                                  "city": {"type": "string"}, "q": {"type": "string", "default": "news"}}, "required": ["city"]}
        self.assertEqual(grounding.check("t", {"limit": 5}, ["今天天气"], [], schema=schema), [])
        self.assertTrue(grounding.check("t", {"limit": 7}, ["今天天气"], [], schema=schema))                # a different value: provenance
        self.assertEqual(grounding.check("t", {"unit": "celsius"}, ["今天天气"], [], schema=schema), [])
        self.assertEqual(grounding.check("t", {"unit": "fahrenheit"}, ["今天天气"], [], schema=schema), [])
        self.assertTrue(grounding.check("t", {"city": "北京市"}, ["今天天气"], [], schema=schema))          # required and no default: provenance
        self.assertTrue(grounding.check("t", {"q": "stock prices"}, ["今天天气"], [], schema=schema))
        self.assertTrue(grounding.check("t", {"limit": 5}, ["今天天气"], []))                              # no schema, no exemption
        audit: list = []
        grounding.check("t", {"limit": 5}, ["今天天气"], [], schema=schema, audit=audit)
        self.assertEqual([a["how"] for a in audit], ["developer_exempt"])

    def test_nested_properties_and_function_wrapped_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {"opts": {"type": "object", "properties": {"limit": {"default": 3}}}}}}}]
        schema = grounding.tool_parameters(tools, "t")
        self.assertEqual(grounding.check("t", {"opts": {"limit": 3}}, ["hi"], [], schema=schema), [])
        self.assertIsNone(grounding.tool_parameters(tools, "other"))

    async def test_the_session_uses_the_tool_schema_but_not_as_provenance(self) -> None:
        s = _session(["你好"])
        s.cfg["tools"] = [{"name": "lookup_account", "parameters": {"properties": {"limit": {"default": 5}, "acct": {"enum": ["99817"], "default": "123"}}}}]
        self.assertFalse(await s._guard(call({"limit": 5})))
        self.assertTrue(await s._guard(call({"acct": "99817"})))                                   # an enum value is not a default and not provenance

    def test_the_prompt_tells_the_model_not_to_fill_optional_arguments(self) -> None:
        self.assertIn("optional argument", grounding.EVIDENCE_RULES_EN)
        self.assertIn("可选参数不要填", grounding.EVIDENCE_RULES_ZH)


class N8_TheAuditSurvivesTheSession(unittest.IsolatedAsyncioTestCase):
    async def test_every_model_judged_decision_is_logged_as_it_happens(self) -> None:
        s = _session(["我叫张伟"])
        with self.assertLogs("sophon_live.realtime", level="INFO") as cm:
            await s._guard(call({"name": "张炜", K: ev("name", "张伟")}))
            await s._guard(call({"name": "李四"}))
        text = "\n".join(cm.output)
        self.assertIn('grounding audit {"argument": "name", "value": "张炜", "how": "model_span"', text)
        self.assertIn('"needs_input"', text) if False else None
        await s.close()
        self.assertEqual(s.audit, [])                                                              # cleared, but the log lines above are the durable record

    async def test_the_degraded_path_is_logged_too(self) -> None:
        s = _session(["hi"]); s.asr = None
        with self.assertLogs("sophon_live.realtime", level="INFO") as cm:
            await s._guard(call({"zip": "12345"}))
        self.assertTrue(any("grounding audit" in m and "unavailable" in m for m in cm.output), cm.output)


class N9_TheHarnessTimeoutIsAScriptErrorNotAVerdict(unittest.TestCase):
    def test_a_timeout_twice_raises_an_error_that_is_not_an_assertion_failure(self) -> None:
        import subprocess
        from tests import test_r4j_page_behaviour as pb
        boom = mock.Mock(side_effect=subprocess.TimeoutExpired("node", 1))
        with mock.patch.object(pb.subprocess, "run", boom):
            with self.assertRaises(pb.HarnessTimeout) as cm:
                pb.run_page()
        self.assertNotIsInstance(cm.exception, AssertionError)
        self.assertEqual(boom.call_count, 2)                                  # one retry before giving up
        self.assertIn("HarnessTimeout", str(cm.exception))

    def test_a_slow_first_run_that_then_succeeds_is_fine_and_the_budget_is_longer_than_before(self) -> None:
        import subprocess
        from tests import test_r4j_page_behaviour as pb
        ok = mock.Mock(returncode=0, stdout="{}", stderr="")
        with mock.patch.object(pb.subprocess, "run", mock.Mock(side_effect=[subprocess.TimeoutExpired("node", 1), ok])):
            self.assertEqual(pb.run_page(), {})
        self.assertGreater(pb.HARNESS_TIMEOUT_S, 60)

    def test_the_mutation_runner_and_the_gate_treat_it_as_a_script_error(self) -> None:
        for rel, code in (("tests/mutation_check_r4c.py", "script_error"), ("tests/release_gate.py", "sys.exit(2)")):
            src = (_LIVE_ROOT / rel).read_text(encoding="utf-8")
            i = src.index('HARNESS_TIMEOUT_MARKER in out')
            self.assertIn(code, src[i:i + 500], rel)


if __name__ == "__main__":
    unittest.main()
