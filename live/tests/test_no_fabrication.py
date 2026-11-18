"""体验版 v1 / 1: a tool-backed answer is never spoken without its tool result.

- the Coordinator (coordinator.intent) decides which requests are tool-backed and stamps WorkerJob.require_tools,
  including on REFINE (the job is re-run with the new slots and must call the tool again);
- the Worker forces the first tool call and refuses to answer a tool-backed job without it;
- finish_job rejects a completed answer that has no tool evidence (honest failure, ledger FAILED, never the made-up text).
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

from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.assistant_tools.registry import list_tool_names
from sophon_live.coordinator import AdmitRequest, Ledger, WorkerJob, WorkerResult, admit, api, finish_job, intent
from sophon_live.coordinator.schemas import FAILED_GENERIC, NO_TOOL_ANSWER, TaskStatus
from tests.test_orch_worker import _job, _say, _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession
from tests.test_refine_guard import _jev, _ledger


class IntentTests(unittest.TestCase):
    def test_tool_backed_requests(self) -> None:
        cases = {
            "明天上海会下雨吗": "get_weather", "北京明天天气": "get_weather", "明天天气。用户补充：北京": "get_weather",
            "现在几点了": "get_datetime", "今天有什么科技新闻": "get_news", "关掉客厅的灯": "device_control",
            "把客厅灯打开": "device_control", "用户上次说喜欢吃什么": "memory_note", "用搜索查一下今天美元兑人民币大概多少": "web_search",
            "what's the weather in Paris": "get_weather",
        }
        for text, tool in cases.items():
            self.assertEqual(intent.required_tools(text)[:1], [tool], text)

    def test_chitchat_jokes_and_knowledge_need_no_tool(self) -> None:
        for text in ("讲个笑话", "讲一个不一样的笑话", "你好", "谁发明了电灯", "北京是中国的首都吗", "灯为什么会亮", ""):
            self.assertEqual(intent.required_tools(text), [], text)

    # Critic I-8: realtime facts without the words 搜索 / 汇率 / 股价. Each one must be tool-backed (web_search), so the guard in
    # finish_job applies: no successful tool evidence -> honest failure, never an answer from the model's head.
    REALTIME_FACT_QUESTIONS = (
        "今天美元兑人民币是多少", "比特币现在多少钱", "A股开盘了吗",                       # the three the review found classified as []
        "现在金价多少", "今天油价多少", "上证指数今天涨了吗", "昨晚湖人比分是多少", "这周末英超有什么比赛", "现在日元汇率",
        "一百美元能换多少人民币", "特斯拉股价", "今天的票房冠军是谁", "北京到上海的航班晚点吗", "现在堵车吗", "现任美国总统是谁",
        "纳斯达克收盘了没有", "iPhone 17 多少钱", "苹果最新市值多少", "今天有什么热搜", "港股今天开市吗", "明天限行吗",
        "今天欧元兑人民币多少", "美元现在对人民币什么价", "黄金今天跌了没有", "茅台现在股价", "最新一任奥运冠军是谁",
        "what's the bitcoin price", "dollar to yuan exchange rate today", "did the Lakers win, what's the score",
    )
    NOT_REALTIME_FACTS = (
        "讲个笑话", "你好", "你叫什么名字", "给我唱首歌", "现在心情怎么样", "今天过得怎么样", "帮我翻译一下这句话", "一加一等于几",
        "李白是哪个朝代的", "水的沸点是多少度", "光速是多少", "地球到月亮多远", "帮我写首诗", "你会做什么", "谢谢你", "再说一遍",
        "谁发明了电灯", "北京是中国的首都吗", "灯为什么会亮",
    )

    def test_realtime_fact_questions_are_tool_backed(self) -> None:
        for text in self.REALTIME_FACT_QUESTIONS:
            self.assertEqual(intent.required_tools(text)[:1], ["web_search"], text)
            self.assertEqual(intent.pad_for(text), "我帮你查一下。", text)

    def test_chitchat_and_stable_knowledge_stay_tool_free(self) -> None:
        for text in self.NOT_REALTIME_FACTS:
            self.assertEqual(intent.required_tools(text), [], text)

    def test_realtime_fact_without_the_tool_is_never_answered(self) -> None:
        """End to end through admit + finish_job: 「今天美元兑人民币是多少」, the Worker answers a number with no tool call."""
        async def go():
            led = Ledger()
            with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", new=_jev("WORKER", None)):
                res = await admit(AdmitRequest(session_id="s", utterance_id="u", text="今天美元兑人民币是多少", delegate_task="今天美元兑人民币是多少"), led)
            self.assertEqual(res.job.require_tools, ["web_search"])
            led.mark_running(res.job.task_id)
            fake = WorkerResult(task_id=res.job.task_id, version=res.job.version, status="completed", spoken_answer="今天美元兑人民币大约七点二。", freshness_token=res.job.freshness_token)
            return finish_job(led, res.job, fake)
        out = asyncio.run(go())
        self.assertEqual((out.text, out.rejected), (NO_TOOL_ANSWER, "no_tool_result"))

    def test_tool_names_exist(self) -> None:
        known = set(list_tool_names())
        for name, _ in intent._PATTERNS:
            self.assertIn(name, known)


class AdmitStampsRequireToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_weather_job_requires_the_tool(self) -> None:
        led = Ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", new=_jev("WORKER", None)):
            res = await admit(AdmitRequest(session_id="s", utterance_id="u", text="明天上海会下雨吗", delegate_task="明天上海会下雨吗"), led)
        self.assertEqual(res.job.require_tools, ["get_weather"])

    async def test_joke_job_requires_nothing(self) -> None:
        led = Ledger()
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", new=_jev("WORKER", None)):
            res = await admit(AdmitRequest(session_id="s", utterance_id="u", text="讲个笑话", delegate_task="讲个笑话"), led)
        self.assertEqual(res.job.require_tools, [])

    async def test_refine_reruns_the_job_with_the_tool_required(self) -> None:
        # "改成北京的" after a Shanghai weather task: REFINE (guard), the re-run job must call get_weather again
        for raw in ("改成北京的。", ""):
            led, rec = _ledger("明天上海会下雨吗")
            res = await admit(AdmitRequest(session_id="s", utterance_id="u", text="明天北京会下雨吗", delegate_task="明天北京会下雨吗", raw_text=raw), led)
            self.assertEqual(res.control.op, "REFINE", raw)
            self.assertEqual(res.via, "guard")
            self.assertEqual(res.job.require_tools, ["get_weather"], raw)
            self.assertEqual(res.job.version, 2)

    async def test_refine_by_jev_with_a_bare_slot_still_requires_the_old_tool(self) -> None:
        led, rec = _ledger("明天上海会下雨吗")
        with mock.patch.object(api, "REFINE_GUARD", False), mock.patch.object(api, "USE_JEV", True), \
                mock.patch.object(api.jev_client, "system_one", new=_jev("CONTROL", f"REFINE:{rec.task_id}:1")):
            res = await admit(AdmitRequest(session_id="s", utterance_id="u", text="北京的", delegate_task="北京的"), led)
        self.assertEqual(res.control.op, "REFINE")
        self.assertEqual(res.job.require_tools, ["get_weather"])


class WorkerGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_call_is_forced_by_name(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, _tool_call("get_weather", '{"city":"北京","when":"tomorrow"}')),
            _FakeResp(200, _say("北京明天多云，1到8度。")),
        ])
        job = _job("明天上海会下雨吗。用户改口，以新说法为准：明天北京会下雨吗", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="北京明天多云，气温1到8度。")):
            res = await orch.run_job(job)
        self.assertEqual(res.status, "completed")
        self.assertEqual(sess.posts[0]["json"]["tool_choice"], {"type": "function", "function": {"name": "get_weather"}})
        self.assertNotIn("tool_choice", sess.posts[1]["json"])      # once the tool ran, the model speaks freely
        self.assertIn("MUST call get_weather", sess.posts[0]["json"]["messages"][1]["content"])

    async def test_jobs_without_requirements_are_not_forced(self) -> None:
        sess = _FakeSession([_FakeResp(200, _say("从前有座山。"))])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            res = await orch.run_job(_job("讲个笑话"))
        self.assertEqual((res.status, res.spoken_answer), ("completed", "从前有座山。"))
        self.assertNotIn("tool_choice", sess.posts[0]["json"])

    async def test_tool_less_answer_is_retried_once_then_withheld(self) -> None:
        fake = "明天北京是晴天，气温在零下三到十一度之间。"
        sess = _FakeSession([_FakeResp(200, _say(fake)), _FakeResp(200, _say(fake))])
        rt = mock.AsyncMock(return_value="never")
        job = _job("明天北京会下雨吗", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(job)
        self.assertEqual(len(sess.posts), 2)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.spoken_answer, NO_TOOL_ANSWER)
        self.assertNotIn("零下", res.spoken_answer)
        rt.assert_not_called()

    async def test_tool_less_answer_then_tool_call_succeeds_on_retry(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, _say("明天北京晴，零下三度。")),
            _FakeResp(200, _tool_call("get_weather", '{"city":"北京","when":"tomorrow"}')),
            _FakeResp(200, _say("北京明天多云，1到8度。")),
        ])
        job = _job("明天北京会下雨吗", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="北京明天多云，气温1到8度。")):
            res = await orch.run_job(job)
        self.assertEqual((res.status, res.spoken_answer), ("completed", "北京明天多云，1到8度。"))
        self.assertEqual(res.evidence[0]["tool"], "get_weather")

    async def test_server_rejecting_tool_choice_falls_back_but_the_guard_still_holds(self) -> None:
        sess = _FakeSession([
            _FakeResp(400, {"error": "tool_choice unsupported"}),
            _FakeResp(200, _say("编出来的天气。")),
            _FakeResp(200, _say("还是编出来的天气。")),
        ])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            res = await orch.run_job(_job("明天北京会下雨吗", require_tools=["get_weather"]))
        self.assertNotIn("tool_choice", sess.posts[1]["json"])
        self.assertEqual((res.status, res.spoken_answer), ("failed", NO_TOOL_ANSWER))


class FinishJobGuardTests(unittest.TestCase):
    def _run(self, evidence: list[dict], status: str = "completed", require=("get_weather",), say: str = "北京晴，零下三到十一度。"):
        led = Ledger()
        rec = led.create("明天北京会下雨吗")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=list(require))
        res = WorkerResult(task_id=rec.task_id, version=rec.version, status=status, spoken_answer=say, evidence=evidence)
        return led, rec, finish_job(led, job, res)

    def test_answer_without_tool_evidence_is_rejected(self) -> None:
        led, rec, out = self._run([])
        self.assertEqual((out.kind, out.text, out.rejected), ("answer", NO_TOOL_ANSWER, "no_tool_result"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.FAILED)
        self.assertEqual(led.tasks[rec.task_id].spoken_answer, NO_TOOL_ANSWER)

    def test_unrelated_tool_evidence_does_not_count(self) -> None:
        _, _, out = self._run([{"tool": "get_datetime", "ok": True}])
        self.assertEqual(out.rejected, "no_tool_result")

    def test_answer_with_its_tool_is_spoken(self) -> None:
        led, rec, out = self._run([{"tool": "get_weather", "ok": True}])
        self.assertEqual((out.kind, out.rejected, out.text), ("answer", "", "北京晴，零下三到十一度。"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.COMPLETED)

    def test_failed_tool_call_is_not_evidence_either(self) -> None:
        # Critic B-1 (r4): "successful tool evidence". After a failed tool the model only guesses, so even an
        # apologetic-looking completed answer is replaced by the plain failure sentence. (Replaces the r1 test that
        # let a completed answer through when the tool had failed: a deliberate tightening, not a relaxed expectation.)
        led, rec, out = self._run([{"tool": "get_weather", "ok": False}])
        self.assertEqual((out.text, out.rejected), (NO_TOOL_ANSWER, "tool_failed"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.FAILED)

    def test_refused_or_slot_less_call_is_not_evidence(self) -> None:
        for ev in ({"tool": "get_weather", "ok": False, "refused": "out_of_scope"}, {"tool": "get_weather", "missing": ["city"]}):
            _, _, out = self._run([ev])
            self.assertEqual(out.text, NO_TOOL_ANSWER, ev)

    def test_failed_then_successful_call_is_enough(self) -> None:
        _, _, out = self._run([{"tool": "get_weather", "ok": False}, {"tool": "get_weather", "ok": True}])
        self.assertEqual(out.rejected, "")

    def test_jobs_with_no_requirement_are_untouched(self) -> None:
        _, _, out = self._run([], require=())
        self.assertEqual((out.kind, out.rejected), ("answer", ""))

    def test_failed_result_with_a_fixed_honest_sentence_is_spoken(self) -> None:
        # (r1 expected a failed result to pass its free text through, Q3 fake included; Critic I-5 tightened it:
        # only fixed honest sentences or a failed required tool's own sentence are ever spoken for a FAILED job.)
        led = Ledger()
        rec = led.create("明天北京会下雨吗")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["get_weather"])
        res = WorkerResult(task_id=rec.task_id, version=rec.version, status="failed", spoken_answer="后台模型暂时不可用，请稍后再试。")
        out = finish_job(led, job, res)
        self.assertEqual((out.text, out.rejected), ("后台模型暂时不可用，请稍后再试。", ""))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.FAILED)

    def test_failed_required_tool_own_sentence_is_spoken_only_when_the_evidence_carries_it(self) -> None:
        sentence = "暂时查不到北京的天气，请稍后再试。"
        _, _, out = self._run([{"tool": "get_weather", "ok": False, "failure_text": sentence}], status="failed", say=sentence)
        self.assertEqual((out.text, out.rejected), (sentence, ""))
        _, _, out = self._run([{"tool": "get_weather", "ok": False}], status="failed", say=sentence)          # same words, no evidence of them
        self.assertEqual((out.text, out.rejected), (NO_TOOL_ANSWER, "failed_text"))


class MutationFakeNeverSpokenTests(unittest.TestCase):
    """Critic I-5, finish_job as the LAST line of defence: the Q3 fake sentence must never reach the user,
    whatever the (mutated / dishonest) Worker reports."""

    FAKE = "北京晴，零下三到十一度。"

    def _out(self, status: str, evidence: list[dict]):
        led = Ledger()
        rec = led.create("明天北京会下雨吗")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["get_weather"])
        res = WorkerResult(task_id=rec.task_id, version=rec.version, status=status, spoken_answer=self.FAKE, evidence=evidence)
        return led, rec, finish_job(led, job, res)

    def test_the_fake_is_blocked_in_every_shape(self) -> None:
        shapes = {
            "completed, no evidence": ("completed", []),
            "completed, tool without ok": ("completed", [{"tool": "get_weather"}]),            # Critic's hole 1: missing ok != success
            "completed, ok=None": ("completed", [{"tool": "get_weather", "ok": None}]),
            "completed, ok='yes' (not True)": ("completed", [{"tool": "get_weather", "ok": "yes"}]),
            "completed, ok=False": ("completed", [{"tool": "get_weather", "ok": False}]),
            "failed, no evidence": ("failed", []),                                              # Critic's hole 2: failed passed free text
            "failed, ok=False": ("failed", [{"tool": "get_weather", "ok": False}]),
            "failed, ok=True": ("failed", [{"tool": "get_weather", "ok": True}]),
            "failed, failure_text is something else": ("failed", [{"tool": "get_weather", "ok": False, "failure_text": "别的话"}]),
        }
        for name, (status, ev) in shapes.items():
            led, rec, out = self._out(status, ev)
            self.assertNotIn("零下", out.text, name)
            self.assertNotIn(self.FAKE, led.tasks[rec.task_id].spoken_answer, name)
            self.assertEqual(out.text, NO_TOOL_ANSWER, name)
            self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.FAILED, name)

    def test_the_only_way_through_is_a_successful_call(self) -> None:
        _, _, out = self._out("completed", [{"tool": "get_weather", "ok": True}])
        self.assertEqual(out.text, self.FAKE)           # (the numbers themselves are checked by the acceptance judge, not here)

    def test_non_tool_job_failure_gets_the_generic_sentence_not_model_text(self) -> None:
        led = Ledger()
        rec = led.create("讲个笑话")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective)
        out = finish_job(led, job, WorkerResult(rec.task_id, rec.version, "failed", "编出来的话"))
        self.assertEqual((out.text, out.rejected), (FAILED_GENERIC, "failed_text"))

    def test_needs_input_text_from_the_worker_is_never_spoken(self) -> None:
        led = Ledger()
        rec = led.create("明天天气")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["get_weather"])
        out = finish_job(led, job, WorkerResult(rec.task_id, rec.version, "needs_input", "晴，零下三度", needs_slots=["city"]))
        self.assertEqual(out.kind, "clarify")
        self.assertNotIn("零下", out.text)


if __name__ == "__main__":
    unittest.main()
