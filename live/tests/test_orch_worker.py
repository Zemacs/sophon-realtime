"""Round 1 / item 4: the orchestrator is a Worker. It executes a WorkerJob and never routes.

- no routing / keyword / refine / memory decisions in orch's prompt or imports;
- a missing required slot comes back as needs_input and the C layer words the CLARIFY;
- results are stamped with (task_id, version); stale ones are dropped by the Coordinator.
"""
from __future__ import annotations

import asyncio
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import route as route_mod  # noqa: E402
from tests import jev_standin  # noqa: E402  (scripted JEV for control-flow tests)
from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.coordinator import Ledger, WorkerJob, WorkerResult, clarify_for_slots, finish_job
from sophon_live.coordinator.schemas import TaskStatus
from tests.test_gateway_coord_control import _commentary, _session
from tests.test_orchestrator import _FakeResp, _FakeSession
from tests.test_r0_coord_boundary import _arm


def _tool_call(name: str, args: str) -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}}]}


def _say(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _job(objective: str = "北京明天天气", **kw) -> WorkerJob:
    return WorkerJob(task_id="t1", version=2, objective=objective, **kw)


class WorkerBoundaryTests(unittest.TestCase):
    def test_prompt_makes_no_routing_or_memory_decisions(self) -> None:
        low = orch.SYS.lower()
        for banned in ("ask which city", "preferred_city", "memory_note", "disliked", "ask the caller in"):
            self.assertNotIn(banned, low, banned)
        self.assertIn("never ask", low)

    def test_orchestrator_source_does_not_reach_into_routing(self) -> None:
        src = (_LIVE_ROOT / "sophon_live" / "assistant_tools" / "orchestrator.py").read_text(encoding="utf-8")
        for banned in ("heuristic_route", "heuristic_control", "from ..coordinator.route", "from ..coordinator.api", "from ..coordinator.ledger", "jev_client", "_route_task"):
            self.assertNotIn(banned, src, banned)
        # no keyword switch on the caller's wording either
        self.assertIsNone(re.search(r'if\s+"[^"]*[\u4e00-\u9fff][^"]*"\s+in\s+task', src))


class RunJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_result_is_stamped_and_carries_evidence(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, _tool_call("get_weather", '{"city":"北京","when":"tomorrow"}')),
            _FakeResp(200, _say("北京明天多云，12度。")),
        ])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="北京明天 多云 12度")):
            res = await orch.run_job(_job())
        self.assertIsInstance(res, WorkerResult)
        self.assertEqual((res.task_id, res.version, res.status), ("t1", 2, "completed"))
        self.assertEqual(res.spoken_answer, "北京明天多云，12度。")
        ev = res.evidence[0]
        self.assertEqual((ev["tool"], ev["args"], ev["ok"]), ("get_weather", {"city": "北京", "when": "tomorrow"}, True))
        self.assertIsInstance(ev["ms"], int)  # latency of the tool call is part of the evidence
        # the job objective is the only task text the model sees
        self.assertIn("北京明天天气", sess.posts[0]["json"]["messages"][1]["content"])

    async def test_missing_required_slot_is_reported_not_asked(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"when":"tomorrow"}'))])
        rt = mock.AsyncMock(return_value="should not run")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(_job("明天天气怎么样"))
        self.assertEqual(res.status, "needs_input")
        self.assertEqual(res.needs_slots, ["city"])
        self.assertEqual(res.spoken_answer, "")
        rt.assert_not_awaited()
        self.assertEqual(len(sess.posts), 1)  # stopped at once, no second LLM round

    async def test_invented_city_is_a_missing_slot_not_a_forecast(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"上海","when":"tomorrow"}'))])
        rt = mock.AsyncMock(return_value="should not run")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(_job("明天天气怎么样"))
        self.assertEqual((res.status, res.needs_slots), ("needs_input", ["city"]))
        rt.assert_not_awaited()

    async def test_city_from_the_request_history_or_translation_is_fine(self) -> None:
        cases = [
            ("上海明天天气", None, '{"city":"上海"}'),
            ("那明天呢", [{"task": "北京今天天气", "answer": "北京今天晴"}], '{"city":"北京"}'),
            ("北京明天天气", None, '{"city":"Beijing"}'),   # the model may translate a city the user did say
        ]
        for obj, hist, args in cases:
            sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", args)), _FakeResp(200, _say("好"))])
            rt = mock.AsyncMock(return_value="ok")
            with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
                res = await orch.run_job(_job(obj), history=hist)
            self.assertEqual(res.status, "completed", obj)
            rt.assert_awaited_once()

    async def test_tool_outside_job_scope_is_refused(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, _tool_call("device_control", '{"op":"x"}')),
            _FakeResp(200, _say("这个我办不了。")),
        ])
        rt = mock.AsyncMock(return_value="ran")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(_job(tool_scope=["get_weather"]))
        rt.assert_not_awaited()
        self.assertEqual(res.evidence[0]["refused"], "out_of_scope")
        names = [t["function"]["name"] for t in sess.posts[0]["json"]["tools"]]
        self.assertEqual(names, ["get_weather"])

    async def test_llm_outage_is_a_failed_result_not_a_crash(self) -> None:
        sess = _FakeSession([_FakeResp(503, {"error": "busy"})])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            res = await orch.run_job(_job())
        self.assertEqual(res.status, "failed")
        self.assertIn("不可用", res.spoken_answer)
        self.assertEqual((res.task_id, res.version), ("t1", 2))

    async def test_legacy_text_entry_still_works_and_words_clarify_via_c_layer(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", "{}"))])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            text = await orch.resolve_task("明天天气怎么样")
        self.assertEqual(text, clarify_for_slots(["city"]))


class FinishJobTests(unittest.TestCase):
    def _running(self):
        led = Ledger()
        rec = led.create("北京明天天气")
        led.mark_running(rec.task_id)
        return led, WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective)

    def test_completed(self) -> None:
        led, job = self._running()
        out = finish_job(led, job, WorkerResult(job.task_id, job.version, "completed", "晴。"))
        self.assertEqual((out.kind, out.text), ("answer", "晴。"))
        self.assertEqual(led.tasks[job.task_id].status, TaskStatus.COMPLETED)

    def test_needs_input_becomes_a_c_layer_clarify_and_keeps_the_task_open(self) -> None:
        led, job = self._running()
        out = finish_job(led, job, WorkerResult(job.task_id, job.version, "needs_input", "随便什么 worker 的话", needs_slots=["city"]))
        self.assertEqual(out.kind, "clarify")
        self.assertEqual(out.text, "请问您想查哪个城市的天气？")  # the worker's own words are ignored
        self.assertEqual(led.tasks[job.task_id].status, TaskStatus.NEEDS_INPUT)
        self.assertIs(led.pending_input(), led.tasks[job.task_id])

    def test_failed_is_recorded_and_status_can_replay_it(self) -> None:
        led, job = self._running()
        out = finish_job(led, job, WorkerResult(job.task_id, job.version, "failed", "后台模型暂时不可用，请稍后再试。"))
        self.assertEqual(out.kind, "answer")
        self.assertEqual(led.tasks[job.task_id].status, TaskStatus.FAILED)

    def test_stale_results_are_dropped(self) -> None:
        led, job = self._running()
        led.refine(job.task_id, "改成上海", version=job.version)  # v2 now
        self.assertEqual(finish_job(led, job, WorkerResult(job.task_id, job.version, "completed", "旧")).kind, "stale")
        led2, job2 = self._running()
        led2.cancel(job2.task_id, job2.version)
        self.assertEqual(finish_job(led2, job2, WorkerResult(job2.task_id, job2.version, "completed", "旧")).kind, "stale")
        led3, job3 = self._running()  # a result stamped for another version is also not accepted
        self.assertEqual(finish_job(led3, job3, WorkerResult(job3.task_id, job3.version + 1, "completed", "?")).kind, "stale")


class GatewayWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_slot_from_worker_is_spoken_as_c_layer_clarify_then_slot_fill_runs(self) -> None:
        import sophon_live.assistant_tools as tools
        from sophon_live.coordinator import api

        session = _session()
        seen: list[str] = []
        real_route = route_mod.heuristic_route

        async def worker(job, instructions="", history=None):
            seen.append(job.objective)
            if "北京" not in job.objective:
                return WorkerResult(job.task_id, job.version, "needs_input", "", needs_slots=["city"], freshness_token=job.freshness_token)
            return WorkerResult(job.task_id, job.version, "completed", "北京明天晴。", evidence=[{"tool": "get_weather", "ok": True}], freshness_token=job.freshness_token)

        _arm(session, "d1", "明天天气怎么样")
        _arm(session, "d2", "北京的")
        # 「明天天气怎么样」 already gets a C-layer CLARIFY from admit; force the Worker path instead
        with jev_standin.standin(), mock.patch.object(tools, "run_worker_job", worker), \
                mock.patch.object(route_mod, "heuristic_route", lambda text, led: "WORKER" if text == "明天天气怎么样" else real_route(text, led)):
            await session._coord_resolve("d1", "明天天气怎么样")
            tid = session.coord_ledger.order[0]
            self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.NEEDS_INPUT)
            self.assertEqual(len(_commentary(session)), 1)  # exactly one thing was said: the C-worded question
            said = "".join(c for _, c in session.deleg["d1"]["chunks"])
            self.assertIn("请问您想查哪个城市的天气？", said)
            self.assertIn("问出口", said)  # the front is told to voice it as a question, not to paraphrase it away
            await session._coord_resolve("d2", "北京的")
        self.assertEqual(len(seen), 2)
        self.assertIn("用户补充：北京的", seen[1])
        self.assertEqual(session.coord_ledger.tasks[tid].status, TaskStatus.COMPLETED)
        self.assertEqual(session.coord_ledger.tasks[tid].version, 2)


if __name__ == "__main__":
    unittest.main()
