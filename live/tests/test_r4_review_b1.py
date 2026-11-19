"""Round 4 / Critic B-1: a realtime query (weather ...) is never answered without a SUCCESSFUL tool call, REFINE included.

Mutation-style: the exact fake answer from docs/r3-quality-issues.md Q3 ("晴天，零下三到十一度" while the tool said
"多云，1 到 8 度") is pushed through every layer that could let it out; each layer must refuse it on its own.
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
from sophon_live.coordinator import Ledger, WorkerJob, WorkerResult, api, finish_job
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER
from sophon_live.coordinator.schemas import TaskStatus, freshness_token_for
from tests._tool_fakes import failing_tool
from tests.test_gateway_coord_control import _commentary, _session
from tests.test_orch_worker import _say, _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession
from tests.test_r0_coord_boundary import _arm
from tests.test_refine_guard import _ledger

Q3_FAKE = "北京明天晴天，气温在零下三到十一度之间。"      # Critic / Q3: the tool really said 多云，1 到 8 度
REAL = "北京明天多云，气温1到8度。"


def _refine_job(led: Ledger, rec) -> WorkerJob:
    new = led.refine(rec.task_id, rec.objective + "。用户改口，以新说法为准：改成北京的", version=rec.version)
    led.mark_running(new.task_id)
    return WorkerJob(task_id=new.task_id, version=new.version, objective=new.objective, require_tools=["get_weather"],
                     freshness_token=f"{new.task_id}:{new.version}")


class Q3ThroughEveryLayerTests(unittest.IsolatedAsyncioTestCase):
    async def test_layer1_worker_refuses_toolless_refine_answer(self) -> None:
        led, rec = _ledger("明天上海会下雨吗", status=TaskStatus.COMPLETED)
        job = _refine_job(led, rec)
        sess = _FakeSession([_FakeResp(200, _say(Q3_FAKE)), _FakeResp(200, _say(Q3_FAKE))])
        rt = mock.AsyncMock(return_value="never called")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(job)
        self.assertEqual((res.status, res.spoken_answer), ("failed", NO_TOOL_ANSWER))
        self.assertEqual(res.evidence, [])
        out = finish_job(led, job, res)
        self.assertEqual(out.text, NO_TOOL_ANSWER)
        self.assertNotIn("零下", out.text)
        self.assertEqual(led.tasks[job.task_id].status, TaskStatus.FAILED)

    def test_layer2_finish_job_refuses_even_if_the_worker_lets_it_through(self) -> None:
        """The Worker is mutated to be dishonest (completed, no evidence): the Coordinator is the last line."""
        led, rec = _ledger("明天上海会下雨吗", status=TaskStatus.COMPLETED)
        job = _refine_job(led, rec)
        res = WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=Q3_FAKE, evidence=[], freshness_token=job.freshness_token)
        out = finish_job(led, job, res)
        self.assertEqual((out.text, out.rejected), (NO_TOOL_ANSWER, "no_tool_result"))
        self.assertEqual(led.tasks[job.task_id].spoken_answer, NO_TOOL_ANSWER)

    def test_layer2_mutation_guard_removed_would_speak_it(self) -> None:
        """Sanity of the mutation itself: with the guard disabled the fake WOULD be spoken, so the tests above do bite."""
        led, rec = _ledger("明天上海会下雨吗", status=TaskStatus.COMPLETED)
        job = _refine_job(led, rec)
        res = WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=Q3_FAKE, evidence=[], freshness_token=job.freshness_token)
        with mock.patch.object(api, "_ungrounded", return_value=False):
            out = finish_job(led, job, res)
        self.assertEqual(out.text, Q3_FAKE)

    async def test_layer3_gateway_speaks_the_honest_sentence_for_a_refine(self) -> None:
        """Shanghai weather done -> user says 改成北京的 -> real admit (the scripted model says REFINE) -> real Worker with a lying LLM."""
        import sophon_live.coordinator as coord  # noqa: F401

        session = _session()
        led, rec = _ledger("明天上海会下雨吗", status=TaskStatus.COMPLETED)
        session.coord_ledger = led
        sess = _FakeSession([_FakeResp(200, _say(Q3_FAKE)), _FakeResp(200, _say(Q3_FAKE))])
        _arm(session, "d2", "明天北京会下雨吗")
        session._trace().start("d2")
        session.last_user = ("改成北京的。", 1e12)
        from tests.test_refine_guard import _jev

        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(api.jev_client, "system_one", new=_jev("WORKER", f"REFINE:{rec.task_id}:1", tool="get_weather")), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="never")), \
                mock.patch.object(session, "_recent_user_text", return_value="改成北京的。"):
            await session._coord_resolve("d2", "明天北京会下雨吗")
        spoken = " ".join(c for _, c in session.deleg["d2"]["chunks"])
        self.assertIn(NO_TOOL_ANSWER, spoken)
        self.assertNotIn("零下", spoken)
        self.assertNotIn("十一度", spoken)
        task = next(t for t in led.tasks.values() if t.version == 2)
        self.assertEqual((task.status, task.spoken_answer), (TaskStatus.FAILED, NO_TOOL_ANSWER))
        tr = session._trace().get("d2").as_dict()
        self.assertEqual(tr["worker"], "failed")
        self.assertEqual(tr["require"], "get_weather")


class FreshnessTokenTests(unittest.TestCase):
    """Critic S-3: freshness_token was written into WorkerJob and never checked."""

    def _job(self):
        led, rec = _ledger("北京明天天气")
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["get_weather"],
                        freshness_token=freshness_token_for(rec.task_id, rec.version))
        return led, rec, job

    def _res(self, job, token):
        return WorkerResult(job.task_id, job.version, "completed", REAL, evidence=[{"tool": "get_weather", "ok": True}], freshness_token=token)

    def test_matching_token_is_accepted(self) -> None:
        led, rec, job = self._job()
        self.assertEqual(finish_job(led, job, self._res(job, job.freshness_token)).kind, "answer")

    def test_result_that_does_not_echo_the_token_is_dropped(self) -> None:
        for bad in ("", "other:1", f"{'x'}:{1}"):
            led, rec, job = self._job()
            self.assertEqual(finish_job(led, job, self._res(job, bad)).kind, "stale", bad)
            self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.RUNNING)      # nothing was written to the Ledger

    def test_forged_job_token_is_dropped(self) -> None:
        led, rec, job = self._job()
        job.freshness_token = "t:9"
        self.assertEqual(finish_job(led, job, self._res(job, "t:9")).kind, "stale")


class ToolFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_required_tool_ends_the_job_with_the_tools_own_sentence_and_no_model_retelling(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"北京","when":"tomorrow"}'))])   # ONE LLM call only
        tool_says = "暂时查不到北京的天气，请稍后再试。"
        job = WorkerJob(task_id="t1", version=2, objective="明天北京会下雨吗", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=failing_tool(tool_says)):
            res = await orch.run_job(job)
        self.assertEqual(len(sess.posts), 1)                      # the model never got to paraphrase / invent
        self.assertEqual((res.status, res.spoken_answer), ("failed", tool_says))
        self.assertFalse(res.evidence[0]["ok"])

    def test_orchestrator_counts_only_ok_true_as_success(self) -> None:
        job = WorkerJob(task_id="t", version=1, objective="北京天气", require_tools=["get_weather"])
        for ev in ({"tool": "get_weather"}, {"tool": "get_weather", "ok": None}, {"tool": "get_weather", "ok": False}):
            self.assertFalse(orch._evidence_has_required(job, [ev]), ev)
        self.assertTrue(orch._evidence_has_required(job, [{"tool": "get_weather", "ok": True}]))

    async def test_failed_required_tool_records_its_own_sentence_as_evidence(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"北京"}'))])
        job = WorkerJob(task_id="t1", version=1, objective="北京天气", require_tools=["get_weather"])
        said = "暂时查不到北京的天气，请稍后再试。"
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=failing_tool(said)):
            res = await orch.run_job(job)
        self.assertEqual(res.evidence[0]["failure_text"], said)
        led = Ledger(); rec = led.create("北京天气"); led.mark_running(rec.task_id)
        job2 = WorkerJob(task_id=rec.task_id, version=rec.version, objective="北京天气", require_tools=["get_weather"])
        res.task_id, res.version = rec.task_id, rec.version
        self.assertEqual(finish_job(led, job2, res).text, said)       # spoken, because the evidence vouches for it

    async def test_machine_looking_tool_failure_falls_back_to_the_generic_sentence(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"北京"}'))])
        job = WorkerJob(task_id="t1", version=1, objective="北京天气", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value='{"ok": false, "error": "boom"}')):
            res = await orch.run_job(job)
        self.assertEqual((res.status, res.spoken_answer), ("failed", NO_TOOL_ANSWER))

    async def test_failed_tool_that_is_not_required_keeps_the_old_flow(self) -> None:
        """A non-required tool failing (here: no requirement at all) is still told to the model, as before."""
        sess = _FakeSession([
            _FakeResp(200, _tool_call("get_weather", '{"city":"北京"}')),
            _FakeResp(200, _say("现在查不到天气，稍后再试。")),
        ])
        job = WorkerJob(task_id="t1", version=1, objective="随便聊聊，说说北京的天气", require_tools=[])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=failing_tool("暂时查不到北京的天气，请稍后再试。")):
            res = await orch.run_job(job)
        self.assertEqual((res.status, res.spoken_answer), ("completed", "现在查不到天气，稍后再试。"))

    async def test_the_real_worker_echoes_the_jobs_freshness_token(self) -> None:
        sess = _FakeSession([_FakeResp(200, _say("好的。"))])
        job = WorkerJob(task_id="t1", version=3, objective="讲个笑话", freshness_token="t1:3")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            res = await orch.run_job(job)
        self.assertEqual(res.freshness_token, "t1:3")

    async def test_success_still_speaks_the_models_wording(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, _tool_call("get_weather", '{"city":"北京","when":"tomorrow"}')),
            _FakeResp(200, _say(REAL)),
        ])
        job = WorkerJob(task_id="t1", version=1, objective="明天北京会下雨吗", require_tools=["get_weather"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), \
                mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="北京明天多云，气温1到8度。")):
            res = await orch.run_job(job)
        self.assertEqual((res.status, res.spoken_answer), ("completed", REAL))
        self.assertTrue(res.evidence[0]["ok"])


class ToolFactsInTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_weather_facts_reach_the_trace_so_a_spoken_answer_can_be_checked(self) -> None:
        import sophon_live.assistant_tools as tools

        session = _session()
        _arm(session, "d1", "北京明天天气")
        session._trace().start("d1")

        async def worker(job, instructions="", history=None):  # noqa: ANN001
            ev = {"tool": "get_weather", "ok": True, "ms": 120, "source": "sojson", "city": "北京", "tmin": 1, "tmax": 8, "now": None,
                  "args": {"city": "北京"}}
            return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=REAL, evidence=[ev],
                                freshness_token=job.freshness_token)

        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", worker):
            await session._coord_resolve("d1", "北京明天天气")
        tool = session._trace().get("d1").as_dict()["tools"][0]
        self.assertEqual((tool["city"], tool["tmin"], tool["tmax"]), ("北京", 1, 8))
        self.assertNotIn("now", tool)                         # None is dropped


class WeatherMetaFactsTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_detailed_reports_city_and_temperatures(self) -> None:
        import tests.test_weather_sources as tw

        tw.weather.STATS.clear(); tw.weather.CACHE.clear()
        env = dict(tw.ENV, SOPHON_WEATHER_SOURCES="open-meteo")
        with mock.patch.dict("os.environ", env), mock.patch.object(tw.weather, "_fetch_json", tw._router(om=tw.OM)):
            text, meta = await tw.weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual((meta["city"], meta["tmin"], meta["tmax"]), ("北京", "4", "12"))
        self.assertIn("4到12度", text)


if __name__ == "__main__":
    unittest.main()
