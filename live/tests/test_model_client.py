"""v0.1.0 model client: budgets, retry rules, breaker, health snapshot, and the model-free JEV degrade."""
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
from sophon_live.coordinator import AdmitRequest, Ledger, admit, api
from sophon_live.coordinator.schemas import TaskStatus


def run(coro):
    return asyncio.run(coro)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class BreakerTests(unittest.TestCase):
    def setUp(self) -> None:
        mc.reset()

    def test_opens_after_n_consecutive_failures_then_half_open_probe(self) -> None:
        clk = Clock()
        b = mc.Breaker("x", clock=clk)
        for _ in range(b.threshold):
            self.assertTrue(b.allow())
            b.failure("boom")
        self.assertEqual(b.state, "open")
        self.assertFalse(b.allow())                      # fast fail
        clk.t += b.open_s + 0.1
        self.assertTrue(b.allow())                       # the one probe
        self.assertEqual(b.state, "half_open")
        self.assertFalse(b.allow())                      # others still rejected while probing
        b.failure("still down")
        self.assertEqual(b.state, "open")
        clk.t += b.open_s + 0.1
        self.assertTrue(b.allow())
        b.success()
        self.assertEqual(b.state, "closed")
        self.assertEqual(b.fails, 0)

    def test_success_resets_consecutive_count(self) -> None:
        b = mc.Breaker("y")
        b.failure("a"); b.failure("b"); b.success(); b.failure("c")
        self.assertEqual(b.state, "closed")

    def test_env_overrides(self) -> None:
        with mock.patch.dict("os.environ", {"SOPHON_LIVE_BREAKER_Z_N": "1", "SOPHON_LIVE_BREAKER_Z_OPEN_S": "2", "SOPHON_LIVE_TIMEOUT_Z": "0.7"}):
            b = mc.Breaker("z")
            self.assertEqual((b.threshold, b.open_s, mc.budget("z")), (1, 2.0, 0.7))
        self.assertEqual((mc.budget("jev"), mc.budget("orch"), mc.budget("tag")), (1.5, 30.0, 15.0))


class CallTests(unittest.TestCase):
    def setUp(self) -> None:
        mc.reset()

    def test_connection_error_is_retried_once_with_backoff(self) -> None:
        n = {"c": 0}

        async def fn():
            n["c"] += 1
            if n["c"] == 1:
                raise ConnectionRefusedError()
            return "ok"

        self.assertEqual(run(mc.call("jev", fn, retries=1)), "ok")
        self.assertEqual(n["c"], 2)

    def test_timeout_is_not_retried(self) -> None:
        n = {"c": 0}

        async def fn():
            n["c"] += 1
            await asyncio.sleep(5)

        with self.assertRaises(mc.ModelUnavailable) as cm:
            run(mc.call("jev", fn, retries=3, total_s=0.2))
        self.assertEqual((cm.exception.reason, n["c"]), ("timeout", 1))

    def test_application_error_is_not_retried(self) -> None:
        n = {"c": 0}

        async def fn():
            n["c"] += 1
            raise RuntimeError("HTTP 500")

        with self.assertRaises(mc.ModelUnavailable) as cm:
            run(mc.call("orch", fn, retries=3))
        self.assertEqual((cm.exception.reason, n["c"]), ("error", 1))

    def test_breaker_open_fails_fast_without_calling(self) -> None:
        async def bad():
            raise ConnectionRefusedError()

        for _ in range(3):
            with self.assertRaises(mc.ModelUnavailable):
                run(mc.call("jev", bad, retries=0))
        called = {"c": 0}

        async def fn():
            called["c"] += 1

        with self.assertRaises(mc.ModelUnavailable) as cm:
            run(mc.call("jev", fn))
        self.assertEqual((cm.exception.reason, called["c"]), ("breaker_open", 0))
        snap = mc.snapshot()
        self.assertEqual(snap["jev"]["state"], "open")
        self.assertEqual(snap["jev"]["rejected"], 1)
        self.assertEqual(set(snap), {"jev", "orch", "tag"})


def _req(text: str) -> AdmitRequest:
    return AdmitRequest(session_id="s", utterance_id="u", text=text, delegate_task=text)


class JevDegradeTests(unittest.TestCase):
    """JEV failing never reaches the keyword router; it degrades model-free."""

    def setUp(self) -> None:
        mc.reset()

    def _admit(self, ledger: Ledger, text: str, jev):
        async def go():
            return await admit(_req(text), ledger)
        with mock.patch.object(api.jev_client, "system_one", jev), \
                mock.patch("sophon_live.coordinator.route.heuristic_route", side_effect=AssertionError("heuristic_route must not run")) as hr:
            res = run(go())
            hr.assert_not_called()
        return res

    def _live_ledger(self) -> Ledger:
        led = Ledger()
        rec = led.create("明天上海会下雨吗")
        led.mark_running(rec.task_id)
        return led

    def _assert_new_task(self, res, led: Ledger, reason: str) -> None:
        self.assertEqual((res.via, res.degraded_reason, res.route), ("degraded", reason, "WORKER"))
        self.assertIsNotNone(res.job)
        self.assertEqual(len(led.tasks), 1)

    def _assert_clarify(self, res, reason: str) -> None:
        self.assertEqual((res.via, res.degraded_reason, res.route), ("degraded", reason, "CLARIFY"))
        self.assertEqual((res.speak.kind, res.speak.text, res.job), ("clarify", api.DEGRADED_CLARIFY, None))

    def test_timeout_no_task_goes_to_orch_as_new_task(self) -> None:
        async def slow(state, questions, **kw):
            raise mc.ModelUnavailable("jev", "timeout")
        led = Ledger()
        self._assert_new_task(self._admit(led, "今天北京天气怎么样", slow), led, "timeout")

    def test_timeout_with_live_task_is_fixed_clarify(self) -> None:
        async def slow(state, questions, **kw):
            raise mc.ModelUnavailable("jev", "timeout")
        led = self._live_ledger()
        self._assert_clarify(self._admit(led, "改成北京的吧", slow), "timeout")
        self.assertEqual(len(led.tasks), 1)

    def test_connection_refused_degrades(self) -> None:
        async def refused(state, questions, **kw):
            raise ConnectionRefusedError()
        led = Ledger()
        self._assert_new_task(self._admit(led, "几点了", refused), led, "error")
        led2 = self._live_ledger()
        self._assert_clarify(self._admit(led2, "取消刚才那个", refused), "error")

    def test_non_option_answer_degrades(self) -> None:
        for resp in ({"answers": {"route": {"type": "choice", "choice": "BANANA"}}}, {"answers": {"route": {"type": "noul", "noul": 0.9}}}, {}):
            async def odd(state, questions, _r=resp, **kw):
                return _r
            led = self._live_ledger()
            self._assert_clarify(self._admit(led, "随便说点什么", odd), "non_option_answer")

    def test_breaker_open_degrades_fast_through_real_client(self) -> None:
        # production jev_client against a closed port: connection failures first, then the breaker opens and calls stop reaching the network
        import time
        from tests.jev_stub import real_system_one
        times = []
        with mock.patch.object(api.jev_client, "system_one", real_system_one), \
                mock.patch.object(api.jev_client, "DEFAULT_URL", "http://127.0.0.1:9/v1/systemone"):
            for _ in range(5):
                t0 = time.monotonic()
                res = run(admit(_req("几点了"), Ledger()))
                times.append(time.monotonic() - t0)
                self.assertEqual(res.via, "degraded")
        self.assertEqual(mc.snapshot()["jev"]["state"], "open")
        self.assertEqual(res.degraded_reason, "breaker_open")
        self.assertLess(times[-1], 0.05)

    def test_heuristic_route_is_not_called_by_production_code(self) -> None:
        src = (_LIVE_ROOT / "sophon_live" / "coordinator" / "api.py").read_text()
        self.assertNotIn("heuristic_route(", src)


class OrchAndHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        mc.reset()

    def test_orch_down_is_honest_fixed_sentence_then_breaker_fast_fail(self) -> None:
        import time
        from sophon_live.assistant_tools import orchestrator as orch
        from sophon_live.coordinator import WorkerJob
        job = WorkerJob(task_id="t1", version=1, objective="北京明天天气")
        with mock.patch.dict("os.environ", {"SOPHON_LIVE_ORCH_URL": "http://127.0.0.1:9/v1/chat/completions"}):
            for _ in range(3):
                res = run(orch.run_job(job))
                self.assertEqual(res.status, "failed")
                self.assertIn("后台模型", repr(res))
            t0 = time.monotonic()
            res = run(orch.run_job(job))
            self.assertLess(time.monotonic() - t0, 0.1)
            self.assertEqual(res.status, "failed")
        self.assertEqual(mc.snapshot()["orch"]["state"], "open")

    def test_tag_connect_failure_raises_after_retry_and_trips_breaker(self) -> None:
        from sophon_live.adapters import VllmOmniAdapter
        ad = VllmOmniAdapter("ws://127.0.0.1:9/v1/realtime", "http://127.0.0.1:9")
        for _ in range(3):
            with self.assertRaises(mc.ModelUnavailable):
                run(ad.start(instructions="", voice="v", lang="zh", tools=[], history=[]))
        self.assertEqual(mc.snapshot()["tag"]["state"], "open")

    def test_health_detail_is_json_with_downstream(self) -> None:
        import json
        from sophon_live import gateway
        body = json.loads(gateway._health_detail())
        self.assertEqual(body["gateway"], "ok")
        self.assertEqual(set(body["downstream"]), {"jev", "orch", "tag"})
        self.assertIn("state", body["downstream"]["jev"])


if __name__ == "__main__":
    unittest.main()
