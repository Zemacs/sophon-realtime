"""v0.1.0 model client (round 1, infrastructure only): budgets, retry rules, breaker (incl. the half-open cancel fix), health snapshot, and the
wiring of the three downstream clients (JEV / orch / Tag). Nothing here reads the user's words."""
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
from sophon_live.coordinator import jev_client


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

    def tearDown(self) -> None:
        mc.reset()                                       # breaker state is process-global: never leave an open breaker for the next test file

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

    def tearDown(self) -> None:
        mc.reset()                                       # breaker state is process-global: never leave an open breaker for the next test file

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


class BreakerCancelTests(unittest.IsolatedAsyncioTestCase):
    """RC2-3: a cancelled half-open probe must not leave the breaker stuck in half_open (every later call breaker_open)."""

    def setUp(self) -> None:
        mc.reset()
        self.clock = Clock()
        b = mc.breaker("jev")
        b._clock = self.clock

    def tearDown(self) -> None:
        mc.reset()

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


class JevClientWiringTests(unittest.TestCase):
    """The production JEV client goes through model_client (budget, one retry on connection errors, breaker). Callers (the Coordinator) still
    catch the exception; what they do with it is not decided here."""

    def setUp(self) -> None:
        mc.reset()

    def tearDown(self) -> None:
        mc.reset()                                       # breaker state is process-global: never leave an open breaker for the next test file

    def test_closed_port_raises_model_unavailable_then_breaker_opens_and_stops_hitting_the_network(self) -> None:
        import time
        times = []
        from tests.jev_stub import real_system_one          # ``tests`` installs a scripted router; this test exercises the production client
        with mock.patch.object(jev_client, "system_one", real_system_one), mock.patch.object(jev_client, "DEFAULT_URL", "http://127.0.0.1:9/v1/systemone"):
            for _ in range(5):
                t0 = time.monotonic()
                with self.assertRaises(mc.ModelUnavailable) as cm:
                    run(jev_client.system_one({}, {}))
                times.append(time.monotonic() - t0)
        self.assertEqual(mc.snapshot()["jev"]["state"], "open")
        self.assertEqual(cm.exception.reason, "breaker_open")
        self.assertLess(times[-1], 0.05)

    def test_jev_budget_default_is_1_5_s_and_the_old_env_still_wins(self) -> None:
        self.assertIsNone(jev_client.DEFAULT_TIMEOUT)             # unset: model_client.budget("jev") applies
        self.assertEqual(mc.budget("jev"), 1.5)

    def test_http_error_status_is_an_application_error_not_retried(self) -> None:
        calls = {"n": 0}

        async def fake_call(name, fn, *, retries=1, total_s=None):
            calls["n"] += 1
            self.assertEqual((name, retries), ("jev", 1))
            raise mc.ModelUnavailable("jev", "error", "HTTP 500")

        from tests.jev_stub import real_system_one
        with mock.patch.object(mc, "call", fake_call):
            with self.assertRaises(mc.ModelUnavailable):
                run(real_system_one({}, {}))
        self.assertEqual(calls["n"], 1)


class HealthRouteTests(unittest.TestCase):
    def test_only_detail_1_returns_the_json_body(self) -> None:
        import json
        from sophon_live import gateway
        self.assertEqual(json.loads(gateway._health_body("detail=1"))["gateway"], "ok")
        self.assertEqual(json.loads(gateway._health_body("x=1&detail=1"))["gateway"], "ok")
        for q in ("", "detail=10", "detail=0", "xdetail=1"):
            self.assertIsNone(gateway._health_body(q), q)

    def test_orch_budget_default_is_30_s(self) -> None:
        from sophon_live.assistant_tools import orchestrator as orch
        self.assertEqual(orch._DEFAULT_TIMEOUT, 30.0)


class GatewayDefaultHostTests(unittest.TestCase):
    def test_host_defaults_to_loopback(self) -> None:
        from sophon_live import gateway
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SOPHON_LIVE_HOST", None)
            self.assertEqual(gateway._default_host(), "127.0.0.1")


class OrchAndHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        mc.reset()

    def tearDown(self) -> None:
        mc.reset()                                       # breaker state is process-global: never leave an open breaker for the next test file

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


class OrchStatusWiringTests(unittest.TestCase):
    """The orchestrator's HTTP statuses map onto the breaker: a 5xx is "the service is sick" (counts), a 4xx is the caller's / model's
    problem (does not), and a 400 caused by the optional request fields is retried once without them."""

    def setUp(self) -> None:
        mc.reset()

    def tearDown(self) -> None:
        mc.reset()                                       # breaker state is process-global: never leave an open breaker for the next test file

    def _serve(self, handler):
        from aiohttp import web
        import socket

        async def go(coro_fn):
            app = web.Application()
            app.router.add_post("/v1/chat/completions", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            try:
                return await coro_fn(f"http://127.0.0.1:{port}/v1/chat/completions")
            finally:
                await runner.cleanup()
        return go

    def _run_job(self, handler, n=1):
        from sophon_live.assistant_tools import orchestrator as orch
        from sophon_live.coordinator import WorkerJob
        job = WorkerJob(task_id="t1", version=1, objective="你好")

        async def body(url):
            res = None
            with mock.patch.dict("os.environ", {"SOPHON_LIVE_ORCH_URL": url}):
                for _ in range(n):
                    res = await orch.run_job(job)
            return res
        return run(self._serve(handler)(body))

    def test_5xx_counts_against_the_breaker(self) -> None:
        from aiohttp import web

        async def h(request):
            return web.json_response({"error": "boom"}, status=503)
        res = self._run_job(h, n=3)
        self.assertEqual(res.status, "failed")
        self.assertEqual(mc.snapshot()["orch"]["failed"], 3)
        self.assertEqual(mc.snapshot()["orch"]["state"], "open")

    def test_4xx_does_not_count(self) -> None:
        from aiohttp import web

        async def h(request):
            return web.json_response({"error": "bad"}, status=404)
        res = self._run_job(h, n=4)
        self.assertEqual(res.status, "failed")
        self.assertEqual((mc.snapshot()["orch"]["failed"], mc.snapshot()["orch"]["state"]), (0, "closed"))

    def test_400_is_retried_once_without_the_optional_fields(self) -> None:
        from aiohttp import web
        seen = []

        async def h(request):
            body = await request.json()
            seen.append(sorted(k for k in ("chat_template_kwargs", "tool_choice") if k in body))
            if "chat_template_kwargs" in body:
                return web.json_response({"error": "unknown field"}, status=400)
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "你好呀"}}]})
        res = self._run_job(h)
        self.assertEqual(seen, [["chat_template_kwargs"], []])
        self.assertEqual(res.status, "completed")


if __name__ == "__main__":
    unittest.main()
