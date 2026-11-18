"""rc3: regression tests for Critic's rc2 findings RC2-3 (breaker cancel), RC2-6 (backstop for a wrong router answer), RC2-7 (numbers),
RC2-8 (invalid router answers are counted), plus the mutation-survivor tests A2 / A4 / D9 / F3."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import model_client as mc
from sophon_live.coordinator import AdmitRequest, Ledger, WorkerJob, WorkerResult, admit, api, finish_job, grounding
from sophon_live.coordinator.schemas import NUMBERS_UNVERIFIED


def run(c):
    return asyncio.run(c)


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class BreakerCancelTests(unittest.IsolatedAsyncioTestCase):
    """RC2-3: a cancelled half-open probe must not leave the breaker stuck in half_open (every later call breaker_open)."""

    def setUp(self) -> None:
        mc.reset()
        self.clock = Clock()
        b = mc.breaker("jev")
        b._clock = self.clock

    async def _open_then_half_open(self) -> None:
        async def bad():
            raise ConnectionRefusedError()

        for _ in range(3):
            with self.assertRaises(mc.ModelUnavailable):
                await mc.call("jev", bad, retries=0)
        self.assertEqual(mc.breaker("jev").state, "open")
        self.clock.t += 1000                                  # past open_s: the next call is the half-open probe

    async def test_cancelled_probe_frees_the_slot(self) -> None:
        await self._open_then_half_open()
        started = asyncio.Event()

        async def hang():
            started.set()
            await asyncio.sleep(60)

        task = asyncio.create_task(mc.call("jev", hang, retries=0, total_s=30))
        await started.wait()
        self.assertTrue(mc.breaker("jev").probing)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(mc.breaker("jev").probing, "RC2-3: probing must be released on cancel")

        async def ok():
            return "fine"

        self.assertEqual(await mc.call("jev", ok, retries=0), "fine")      # the next healthy call becomes the probe and closes the breaker
        self.assertEqual(mc.breaker("jev").state, "closed")

    async def test_wait_for_around_the_probe_does_not_stick_either(self) -> None:
        await self._open_then_half_open()

        async def hang():
            await asyncio.sleep(60)

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(mc.call("jev", hang, retries=0, total_s=30), 0.1)
        self.assertFalse(mc.breaker("jev").probing)

    async def test_cancel_while_closed_leaves_no_residue_and_counts_nothing(self) -> None:
        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(mc.call("jev", hang, retries=0, total_s=30))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        snap = mc.snapshot()["jev"]
        self.assertEqual((snap["state"], snap["failed"], snap["ok"]), ("closed", 0, 0))

    async def test_a_timeout_raised_by_the_call_itself_is_not_retried_F3(self) -> None:
        n = {"c": 0}

        async def fn():
            n["c"] += 1
            raise asyncio.TimeoutError()                      # e.g. the HTTP client's own timeout, with plenty of budget left

        with self.assertRaises(mc.ModelUnavailable) as cm:
            await mc.call("jev", fn, retries=3, total_s=30)
        self.assertEqual((cm.exception.reason, n["c"]), ("timeout", 1))


class ConnectionClassifierTests(unittest.TestCase):
    def test_a_timeout_is_never_a_connection_error_F3b(self) -> None:
        from sophon_live import model_client as mc
        self.assertFalse(mc.is_connection_error(asyncio.TimeoutError()))
        self.assertFalse(mc.is_connection_error(TimeoutError()))
        self.assertTrue(mc.is_connection_error(ConnectionResetError()))
        self.assertFalse(mc.is_connection_error(ValueError("bad body")))


class NumbersTests(unittest.TestCase):
    """RC2-7: one normalisation entry; no "the source mentions 一/两 so 1 and 2 are sourced"."""

    def t(self, spoken, source, **kw):
        return grounding.unverified_numbers(spoken, [source], **kw)

    def test_chinese_decimal_matches_the_arabic_source(self) -> None:
        self.assertEqual(self.t("三十八点五度", "38.5"), [])
        self.assertEqual(self.t("38.5度", "三十八点五"), [])

    def test_a_made_up_two_is_caught_even_when_the_source_has_the_character_one(self) -> None:
        self.assertEqual(self.t("比昨天高两度", "第一条：气温1度"), [2.0])       # D9: no 1/2 injection from 一 / 两 in the source

    def test_english_number_words(self) -> None:
        self.assertEqual(self.t("twenty degrees", "temperature 20"), [])
        self.assertEqual(self.t("twenty degrees", "temperature 25"), [20.0])
        self.assertEqual(self.t("one hundred and five", "105"), [])

    def test_percent_and_signs(self) -> None:
        self.assertEqual(self.t("百分之五", "5%"), [])
        self.assertEqual(self.t("上涨百分之三", "5%"), [3.0])
        self.assertEqual(self.t("零下三度", "-3°C"), [])

    def test_lone_quantifier_one_is_not_data_but_a_lone_two_is(self) -> None:
        self.assertEqual(self.t("我帮你查一下", "x"), [])
        self.assertEqual(self.t("两度", "x"), [2.0])

    def test_clock_readings_match_in_12_and_24_hour_form_only_for_a_clock(self) -> None:
        self.assertEqual(self.t("现在是下午三点十分", "现在是15:10", clock12=True), [])
        self.assertEqual(self.t("现在是下午四点", "现在是15:10", clock12=True), [4.0])
        self.assertEqual(self.t("现在是下午三点十分", "现在是15:10"), [3.0])

    def test_big_numbers_and_rounding(self) -> None:
        self.assertEqual(self.t("1.2万亿", "12000亿"), [])
        self.assertEqual(self.t("七点二", "7.1983"), [])
        self.assertEqual(self.t("三十五", "35000"), [35.0])


def _job(tools=(), judgment="model", objective="今天比特币多少钱"):
    return WorkerJob(task_id="t", version=1, objective=objective, require_tools=list(tools), tool_judgment=judgment)


def _res(job, text, evidence=()):
    return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=text, evidence=list(evidence),
                        freshness_token=job.freshness_token)


def _led(job):
    led = Ledger()
    rec = led.create(job.objective)
    led.mark_running(rec.task_id)
    job.task_id, job.version = rec.task_id, rec.version
    job.freshness_token = f"{rec.task_id}:{rec.version}"
    return led


class BackstopTests(unittest.IsolatedAsyncioTestCase):
    """RC2-6: a router that wrongly says "no tool" no longer leaves a fabricated fact unchecked."""

    def _scripted(self, verdict):
        async def fake(state, questions, **kw):
            if verdict == "boom":
                raise RuntimeError("router down")
            return {"answers": {api.ANSWER_FACT_KEY: {"type": "choice", "choice": verdict}}}

        return mock.patch.object(api.jev_client, "system_one", new=fake)

    async def test_no_tool_judgment_plus_invented_price_is_held_back_when_the_model_says_fact(self) -> None:
        job = _job(); led = _led(job); res = _res(job, "比特币现在大约三万五千美元。")
        with self._scripted("FACT"):
            await api.review_numbers(job, res)
        self.assertTrue(job.check_numbers)
        self.assertEqual(finish_job(led, job, res).text, NUMBERS_UNVERIFIED)

    async def test_a_story_number_is_left_alone_when_the_model_says_not_a_fact(self) -> None:
        job = _job(objective="讲个笑话"); led = _led(job); res = _res(job, "从前有三只小熊。")
        with self._scripted("NOT_FACT"):
            await api.review_numbers(job, res)
        self.assertFalse(job.check_numbers)
        self.assertEqual(finish_job(led, job, res).text, "从前有三只小熊。")

    async def test_router_down_is_not_a_pass(self) -> None:
        job = _job(); res = _res(job, "比特币现在大约三万五千美元。")
        with self._scripted("boom"):
            await api.review_numbers(job, res)
        self.assertTrue(job.check_numbers)

    async def test_non_option_verdict_is_not_a_pass(self) -> None:
        job = _job(); res = _res(job, "比特币现在大约三万五千美元。")
        with self._scripted("也许吧"):
            await api.review_numbers(job, res)
        self.assertTrue(job.check_numbers)

    async def test_an_answer_whose_numbers_are_the_users_own_asks_nobody(self) -> None:
        job = _job(objective="把音量调到五十"); res = _res(job, "好的，音量调到五十。")
        boom = mock.AsyncMock(side_effect=AssertionError("no model call needed"))
        with mock.patch.object(api.jev_client, "system_one", boom):
            await api.review_numbers(job, res)
        self.assertFalse(job.check_numbers)

    def test_news_and_clock_answers_are_checked_against_their_tool_output(self) -> None:
        for tool, out, said, bad in (
                ("get_news", "第1条：某公司发布新品。第2条：A股成交额1.2万亿", "A股涨了5%，成交额两万亿", True),
                ("get_news", "第1条：某公司发布新品。第2条：A股成交额1.2万亿", "A股成交额一点二万亿", False),
                ("get_datetime", "现在是15:10。", "现在是下午四点", True),
                ("get_datetime", "现在是15:10。", "现在是下午三点十分", False)):
            job = _job(tools=[tool]); led = _led(job)
            res = _res(job, said, [{"tool": tool, "ok": True, "out": out}])
            self.assertEqual(finish_job(led, job, res).text == NUMBERS_UNVERIFIED, bad, (tool, said))

    def test_a_job_with_no_tool_and_no_review_is_unchecked_by_design(self) -> None:
        job = _job(); led = _led(job); res = _res(job, "从前有三只小熊。")
        self.assertEqual(finish_job(led, job, res).text, "从前有三只小熊。")


class InvalidRouterAnswersTests(unittest.IsolatedAsyncioTestCase):
    """RC2-8 + A2: a missing / non-option answer is never replaced by a guess and is counted + traced."""

    async def _admit(self, answers, text="明天天气怎么样北京", raw=""):
        async def fake(state, questions, **kw):
            return {"answers": {k: {"type": "choice", "choice": v} for k, v in answers.items()}}

        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", new=fake):
            return await admit(AdmitRequest(session_id="s", utterance_id="u", text=text, delegate_task=text, raw_text=raw), Ledger())

    async def test_missing_tool_answer_is_unknown_and_counted_A2(self) -> None:
        api.JUDGE_STATS.clear()
        res = await self._admit({"route": "WORKER"}, text="明天天气怎么样北京")         # weather words in the text, no tool answer
        self.assertEqual((res.job.require_tools, res.job.tool_judgment), ([], "unknown"))
        self.assertEqual(res.judge_invalid, ["tool:missing"])
        self.assertEqual(api.JUDGE_STATS.get("tool_missing"), 1)

    async def test_non_option_tool_answer_and_faithful_answer_are_counted(self) -> None:
        api.JUDGE_STATS.clear()
        res = await self._admit({"route": "WORKER", "tool": "天气", "task_faithful": "也许"}, text="北京天气", raw="你好")
        self.assertEqual(sorted(res.judge_invalid), ["task_faithful:invalid", "tool:invalid"])
        self.assertEqual(res.job.objective, "北京天气")                              # not guessed: the front's wording stands
        self.assertEqual(api.JUDGE_STATS, {"tool_invalid": 1, "task_faithful_invalid": 1})


if __name__ == "__main__":
    unittest.main()
