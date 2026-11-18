"""体验版 v1 / 1: a tool-backed answer is never spoken without its tool result.

- the Coordinator (coordinator.intent) decides which requests are tool-backed and stamps WorkerJob.require_tools,
  including on REFINE (the job is re-run with the new slots and must call the tool again);
- the Worker forces the first tool call and refuses to answer a tool-backed job without it;
- finish_job rejects a completed answer that has no tool evidence (honest failure, ledger FAILED, never the made-up text).
"""
from __future__ import annotations

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
from sophon_live.coordinator.api import NO_TOOL_ANSWER
from sophon_live.coordinator.schemas import TaskStatus
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
    def _run(self, evidence: list[dict], status: str = "completed", require=("get_weather",)):
        led = Ledger()
        rec = led.create("明天北京会下雨吗")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=list(require))
        res = WorkerResult(task_id=rec.task_id, version=rec.version, status=status, spoken_answer="北京晴，零下三到十一度。", evidence=evidence)
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

    def test_failed_and_needs_input_results_pass_through(self) -> None:
        _, _, out = self._run([], status="failed")
        self.assertEqual(out.rejected, "")


if __name__ == "__main__":
    unittest.main()
