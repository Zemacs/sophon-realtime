"""Stage 0 (JEV reliability): the client (typed failures, connection-only retry, circuit breaker with /healthz probing), the
degraded decision in admit() (no keyword rule runs), the counters / per-answer logging, and the gateway's start-up wait.
Everything talks to a real local HTTP server (aiohttp.web) that misbehaves on demand; nothing touches the network or a GPU."""
from __future__ import annotations

import asyncio
import inspect
import json
import socket
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from aiohttp import web  # noqa: E402

from sophon_live import gateway  # noqa: E402
from sophon_live.coordinator import AdmitRequest, Ledger, admit, api, jev_client, route as route_mod  # noqa: E402
from sophon_live.coordinator.jev_client import JevError  # noqa: E402

GOOD = {"answers": {"route": {"type": "choice", "choice": "WORKER", "confidence": 0.69}}}


class FakeJev:
    """A JEV HTTP server whose behaviour for /v1/systemone and /healthz is set per test."""

    def __init__(self) -> None:
        self.post_mode = "ok"            # ok | 500 | 503 | 422 | badjson | array | hang | noroute | noul | nocontrol
        self.health_mode = "ok"          # ok | 503 | badjson | notok | hang
        self.posts = 0
        self.gets = 0
        self.control_choice = None
        self._runner: web.AppRunner | None = None
        self.port = 0
        self.release = asyncio.Event()     # set on stop(): lets hanging handlers finish

    async def start(self) -> "FakeJev":
        app = web.Application()
        app.router.add_post("/v1/systemone", self._post)
        app.router.add_get("/healthz", self._health)
        self._runner = web.AppRunner(app, shutdown_timeout=0.5)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = self._runner.addresses[0][1]
        return self

    async def stop(self) -> None:
        self.release.set()
        if self._runner:
            await self._runner.cleanup()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/systemone"

    async def _post(self, req: web.Request) -> web.Response:
        self.posts += 1
        body = await req.json()
        m = self.post_mode
        if m == "hang":
            await self.release.wait()
        if m in ("500", "503", "422"):
            return web.Response(status=int(m), text=json.dumps({"error": "boom"}), content_type="application/json")
        if m == "badjson":
            return web.Response(status=200, text="<html>not json</html>")
        if m == "array":
            return web.json_response([1, 2])
        if m == "noroute":
            return web.json_response({"answers": {}})
        if m == "noul":
            return web.json_response({"answers": {"route": {"type": "noul", "noul": 0.2}}})
        resp: dict = {"answers": {"route": {"type": "choice", "choice": "WORKER", "confidence": 0.69}}}
        if "control" in (body.get("questions") or {}):
            if m == "nocontrol":
                resp["answers"]["route"]["choice"] = "CONTROL"
            else:
                key = self.control_choice or "NEW"
                resp["answers"]["control"] = {"type": "choice", "choice": key}
                if key != "NEW":
                    resp["answers"]["route"]["choice"] = "CONTROL"
        return web.json_response(resp)

    async def _health(self, req: web.Request) -> web.Response:
        self.gets += 1
        m = self.health_mode
        if m == "hang":
            await self.release.wait()
        if m == "503":
            return web.Response(status=503, text="down")
        if m == "badjson":
            return web.Response(status=200, text="nope")
        if m == "notok":
            return web.json_response({"ok": False})
        return web.json_response({"ok": True, "model": "karpo-decision"})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        jev_client.stats.reset()
        jev_client.breaker.reset()
        jev_client.register_secondary(None)
        self.fake = await FakeJev().start()
        self.addAsyncCleanup(self.fake.stop)
        self.patches = [
            mock.patch.object(jev_client, "DEFAULT_URL", self.fake.url),
            mock.patch.object(jev_client.CFG, "timeout", 1.5),
            mock.patch.object(jev_client.CFG, "retry_delay", 0.01),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    async def call(self) -> dict:
        return await jev_client.system_one({"utterance": "x"}, {"route": {"type": "choice", "instructions": "x", "criteria": {"A": "a"}}})

    async def reason_of_call(self) -> str:
        with self.assertRaises(JevError) as cm:
            await self.call()
        return cm.exception.reason


class ClientFailureTests(Base):
    async def test_ok_returns_the_parsed_answers(self) -> None:
        r = await self.call()
        self.assertEqual(r["answers"]["route"]["choice"], "WORKER")
        self.assertEqual(self.fake.posts, 1)

    async def test_500_and_503_are_http_5xx_and_are_not_retried(self) -> None:
        for mode in ("500", "503"):
            self.fake.post_mode, before = mode, self.fake.posts
            self.assertEqual(await self.reason_of_call(), "http_5xx")
            self.assertEqual(self.fake.posts - before, 1, "an HTTP error is not a connection error: no retry")

    async def test_422_is_http_4xx_and_does_not_count_towards_the_breaker(self) -> None:
        self.fake.post_mode = "422"
        for _ in range(jev_client.CFG.breaker_fails + 2):
            self.assertEqual(await self.reason_of_call(), "http_4xx")
        self.assertEqual(jev_client.breaker.state, "closed")

    async def test_bad_json_and_non_object_json(self) -> None:
        self.fake.post_mode = "badjson"
        self.assertEqual(await self.reason_of_call(), "bad_json")
        self.fake.post_mode = "array"
        self.assertEqual(await self.reason_of_call(), "bad_json")

    async def test_refused_is_retried_exactly_once(self) -> None:
        dead = f"http://127.0.0.1:{free_port()}/v1/systemone"
        calls = {"n": 0}
        real = jev_client._post_once

        async def counting(*a, **k):  # noqa: ANN002, ANN003
            calls["n"] += 1
            return await real(*a, **k)

        with mock.patch.object(jev_client, "_post_once", counting):
            t0 = time.monotonic()
            with self.assertRaises(JevError) as cm:
                await jev_client.system_one({}, {"route": {}}, url=dead)
        self.assertEqual(cm.exception.reason, "refused")
        self.assertEqual(calls["n"], 2, "one try + one retry")
        self.assertLess(time.monotonic() - t0, 1.5)

    async def test_hang_times_out_once_and_is_never_retried(self) -> None:
        self.fake.post_mode = "hang"
        with mock.patch.object(jev_client.CFG, "timeout", 0.3):
            t0 = time.monotonic()
            self.assertEqual(await self.reason_of_call(), "timeout")
            took = time.monotonic() - t0
        self.assertEqual(self.fake.posts, 1, "a timed-out call is never re-sent")
        self.assertLess(took, 0.9)

    async def test_connect_timeout_is_its_own_reason_and_is_not_retried(self) -> None:
        with mock.patch.object(jev_client.aiohttp.ClientSession, "post", side_effect=jev_client.aiohttp.ConnectionTimeoutError("connect")) as post:
            self.assertEqual(await self.reason_of_call(), "connect_timeout")
        self.assertEqual(post.call_count, 1)

    async def test_default_budget_is_0_3_connect_1_5_total_and_is_what_the_request_uses(self) -> None:
        self.assertEqual((jev_client.Config.connect_timeout, jev_client.Config.timeout), (0.3, 1.5))
        seen = {}
        real = jev_client.aiohttp.ClientTimeout

        def spy(*a, **k):  # noqa: ANN002, ANN003
            seen.update(k)
            return real(*a, **k)

        with mock.patch.object(jev_client.aiohttp, "ClientTimeout", spy):
            await self.call()
        self.assertAlmostEqual(seen["sock_connect"], 0.3, places=3)
        self.assertAlmostEqual(seen["total"], 1.5, places=2)  # deadline-based: total = what is left of the 1.5 s budget

    async def test_import_time_defaults_in_a_fresh_interpreter(self) -> None:
        """setUp resets SECONDARY and tests patch Config, so the shipped defaults are read in a clean process with no SOPHON_JEV_* env."""
        import subprocess, sys, os, json as _json
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON_JEV")}
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        code = ("import json; from sophon_live.coordinator import jev_client as j; c = j.Config; "
                "print(json.dumps([c.timeout, c.connect_timeout, c.connect_retries, c.breaker_fails, j.SECONDARY is None]))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60, cwd=os.getcwd())
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(_json.loads(out.stdout.strip().splitlines()[-1]), [1.5, 0.3, 1, 3, True])

    async def test_every_failure_is_counted_by_reason(self) -> None:
        self.fake.post_mode = "500"
        await self.reason_of_call()
        self.fake.post_mode = "badjson"
        await self.reason_of_call()
        self.assertEqual(jev_client.stats.fail, {"http_5xx": 1, "bad_json": 1})
        self.assertEqual(jev_client.stats.n_fail, 2)


class BreakerTests(Base):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.clk = Clock()
        p = mock.patch.object(jev_client, "_clock", self.clk)
        p.start()
        self.addCleanup(p.stop)

    async def trip(self) -> None:
        self.fake.post_mode = "500"
        for _ in range(jev_client.CFG.breaker_fails):
            await self.reason_of_call()
        self.assertEqual(jev_client.breaker.state, "open")

    async def test_opens_after_n_consecutive_failures_and_then_short_circuits(self) -> None:
        await self.trip()
        posts = self.fake.posts
        for _ in range(5):
            self.assertEqual(await self.reason_of_call(), "breaker_open")
        self.assertEqual(self.fake.posts, posts, "an open breaker sends nothing")
        self.assertEqual(self.fake.gets, 0, "and does not probe before the window is over")

    async def test_a_success_in_between_resets_the_count(self) -> None:
        self.fake.post_mode = "500"
        for _ in range(jev_client.CFG.breaker_fails - 1):
            await self.reason_of_call()
        self.fake.post_mode = "ok"
        await self.call()
        self.fake.post_mode = "500"
        for _ in range(jev_client.CFG.breaker_fails - 1):
            await self.reason_of_call()
        self.assertEqual(jev_client.breaker.state, "closed")

    async def test_after_the_window_healthz_then_one_trial_closes_it(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.post_mode = "ok"
        r = await self.call()
        self.assertEqual(r["answers"]["route"]["choice"], "WORKER")
        self.assertEqual(self.fake.gets, 1, "one /healthz probe first")
        self.assertEqual(jev_client.breaker.state, "closed")

    async def test_failed_probe_keeps_it_open_without_a_real_request(self) -> None:
        await self.trip()
        posts = self.fake.posts
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.health_mode = "503"
        self.assertEqual(await self.reason_of_call(), "breaker_open")
        self.assertEqual(self.fake.posts, posts)
        self.assertEqual(jev_client.breaker.state, "open")
        self.assertEqual(await self.reason_of_call(), "breaker_open", "the window restarted")
        self.assertEqual(self.fake.gets, 1)

    async def test_healthz_ok_but_trial_fails_reopens(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.health_mode, self.fake.post_mode = "ok", "500"
        self.assertEqual(await self.reason_of_call(), "http_5xx")
        self.assertEqual(jev_client.breaker.state, "open")
        self.assertEqual(await self.reason_of_call(), "breaker_open")

    async def test_healthz_that_is_not_ok_does_not_count_as_ready(self) -> None:
        for mode in ("notok", "badjson", "503"):
            self.fake.health_mode = mode
            ok, _ms, detail = await jev_client.probe(self.fake.url)
            self.assertFalse(ok, mode)

    async def test_half_open_lets_exactly_one_trial_through(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.post_mode = "ok"
        gate = asyncio.Event()
        real = jev_client._post_once

        async def slow(*a, **k):  # noqa: ANN002, ANN003
            await gate.wait()
            return await real(*a, **k)

        with mock.patch.object(jev_client, "_post_once", slow):
            first = asyncio.create_task(self.call())
            await asyncio.sleep(0.2)                       # the trial is in flight
            self.assertEqual(await self.reason_of_call(), "breaker_open")
            gate.set()
            await first
        self.assertEqual(jev_client.breaker.state, "closed")

    # --- a cancelled / crashed probe or trial must never leave `busy` stuck (Critic B-2 on the other implementation) ---
    async def _assert_not_stuck(self) -> None:
        self.assertFalse(jev_client.breaker.busy, "probing/busy flag stuck True")
        self.fake.health_mode = "ok"
        self.fake.post_mode = "ok"
        await self.call()                                   # the next caller must be able to probe + trial, and close it
        self.assertEqual(jev_client.breaker.state, "closed")

    async def test_cancelled_healthz_probe_does_not_leave_the_breaker_stuck(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.health_mode = "hang"
        t = asyncio.create_task(self.call())
        await asyncio.sleep(0.2)
        self.assertTrue(jev_client.breaker.busy)           # the probe is in flight
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        await self._assert_not_stuck()

    async def test_probe_that_raises_unexpectedly_does_not_leave_the_breaker_stuck(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)

        async def boom(*a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("probe blew up")

        with mock.patch.object(jev_client, "probe", boom):
            with self.assertRaises(RuntimeError):
                await self.call()
        await self._assert_not_stuck()

    async def test_cancelled_half_open_trial_does_not_leave_the_breaker_stuck(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.post_mode = "hang"
        t = asyncio.create_task(self.call())
        await asyncio.sleep(0.2)
        self.assertEqual(jev_client.breaker.state, "half_open")
        self.assertTrue(jev_client.breaker.busy)           # the trial is in flight
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertEqual(jev_client.breaker.state, "half_open")   # a cancel says nothing about availability
        await self._assert_not_stuck()

    async def test_trial_with_a_4xx_releases_the_slot(self) -> None:
        await self.trip()
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.post_mode = "422"
        self.assertEqual(await self.reason_of_call(), "http_4xx")
        await self._assert_not_stuck()


class ProbeLoopAndStartupTests(Base):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.clk = Clock()
        p = mock.patch.object(jev_client, "_clock", self.clk)
        p.start()
        self.addCleanup(p.stop)

    async def nosleep(self, s: float) -> None:
        self.clk.advance(s)

    async def test_wait_ready_returns_as_soon_as_healthz_is_ok(self) -> None:
        ready, waited, detail = await jev_client.wait_ready(self.fake.url, max_wait=5, sleep=self.nosleep)
        self.assertTrue(ready)
        self.assertEqual(self.fake.gets, 1)

    async def test_wait_ready_waits_for_a_late_server_but_only_up_to_the_bound(self) -> None:
        self.fake.health_mode = "503"
        n = {"i": 0}

        async def sleep_then_heal(s: float) -> None:
            self.clk.advance(s)
            n["i"] += 1
            if n["i"] == 3:
                self.fake.health_mode = "ok"

        ready, waited, _ = await jev_client.wait_ready(self.fake.url, max_wait=10, interval=1.0, sleep=sleep_then_heal)
        self.assertTrue(ready)
        self.assertAlmostEqual(waited, 3.0, places=1)
        self.fake.health_mode = "503"
        ready, waited, detail = await jev_client.wait_ready(self.fake.url, max_wait=4, interval=1.0, sleep=self.nosleep)
        self.assertFalse(ready)
        self.assertLessEqual(waited, 4.0)
        self.assertIn("503", detail)

    async def test_wait_ready_on_a_dead_port_gives_up_and_never_raises(self) -> None:
        ready, waited, detail = await jev_client.wait_ready(f"http://127.0.0.1:{free_port()}", max_wait=2, interval=1.0, sleep=self.nosleep)
        self.assertFalse(ready)
        self.assertIn("refused", detail)

    async def test_gateway_start_up_line_says_ready_or_degraded(self) -> None:
        with mock.patch.object(jev_client, "_clock", time.monotonic):     # real clock: nothing here sleeps
            msg, ready = await gateway.jev_startup_check(max_wait=0.5, url=self.fake.url)
            self.assertTrue(ready)
            self.assertTrue(msg.startswith("JEV ready"), msg)
            self.fake.health_mode = "503"
            msg, ready = await gateway.jev_startup_check(max_wait=0.5, url=self.fake.url)
        self.assertFalse(ready)
        self.assertIn("JEV NOT READY", msg)
        self.assertIn("DEGRADED", msg)

    def test_start_up_probe_only_when_the_coordinator_will_ask_jev(self) -> None:
        self.assertTrue(gateway.jev_startup_wanted({"SOPHON_LIVE_COORD": "1", "SOPHON_LIVE_JEV": "1"}))
        self.assertFalse(gateway.jev_startup_wanted({"SOPHON_LIVE_COORD": "1", "SOPHON_LIVE_JEV": "0"}))
        self.assertFalse(gateway.jev_startup_wanted({"SOPHON_LIVE_JEV": "1"}))

    async def test_prober_opens_the_breaker_when_healthz_keeps_failing_and_closes_it_when_the_model_answers(self) -> None:
        self.fake.health_mode = "503"
        await jev_client.probe_loop(self.fake.url, interval=1, sleep=self.nosleep, rounds=jev_client.CFG.breaker_fails)
        self.assertEqual(jev_client.breaker.state, "open")
        # still down after the window: stays open, and no real request is sent while /healthz fails
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        posts = self.fake.posts
        await jev_client.probe_loop(self.fake.url, interval=0, sleep=self.nosleep, rounds=1)
        self.assertEqual((jev_client.breaker.state, self.fake.posts), ("open", posts))
        # healthz ok but the model does not answer: stays open (deep probe = healthz + a tiny real request)
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.health_mode, self.fake.post_mode = "ok", "500"
        await jev_client.probe_loop(self.fake.url, interval=0, sleep=self.nosleep, rounds=1)
        self.assertEqual(jev_client.breaker.state, "open")
        # both fine: closed again without any user turn
        self.clk.advance(jev_client.CFG.breaker_open_s + 1)
        self.fake.post_mode = "ok"
        await jev_client.probe_loop(self.fake.url, interval=0, sleep=self.nosleep, rounds=1)
        self.assertEqual(jev_client.breaker.state, "closed")


def _req(text: str, uid: str = "u1") -> AdmitRequest:
    return AdmitRequest(session_id="s", utterance_id=uid, text=text, delegate_task=text, playback_active=False)


class DegradedDecisionTests(Base):
    """Whatever goes wrong with JEV, admit() never consults a keyword rule and says so (via=degraded + reason)."""

    def spies(self):
        def forbidden(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("a keyword heuristic was consulted on the degraded path")
        return (mock.patch.object(route_mod, "heuristic_route", forbidden), mock.patch.object(api, "heuristic_control", forbidden),
                mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False))

    async def admit_degraded(self, text: str, ledger: Ledger | None = None):
        a, b, c, d = self.spies()
        with a, b, c, d, self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            res = await admit(_req(text), ledger if ledger is not None else Ledger())
        return res, "\n".join(cm.output)

    async def _each_failure(self):
        """(label, setup, expected reason) for timeout / non-choice (missing + noul) / refused / 5xx / bad json / breaker open."""
        yield "timeout", lambda: setattr(self.fake, "post_mode", "hang"), "timeout"
        yield "refused", lambda: setattr(jev_client, "DEFAULT_URL", f"http://127.0.0.1:{free_port()}/v1/systemone"), "refused"
        yield "5xx", lambda: setattr(self.fake, "post_mode", "500"), "http_5xx"
        yield "bad json", lambda: setattr(self.fake, "post_mode", "badjson"), "bad_json"
        yield "missing route", lambda: setattr(self.fake, "post_mode", "noroute"), "not_choice:missing"
        yield "non-choice (noul)", lambda: setattr(self.fake, "post_mode", "noul"), "not_choice:noul"

    async def test_no_active_task_the_utterance_becomes_a_new_task_for_the_orchestrator(self) -> None:
        async for label, setup, reason in self._each_failure():
            self.fake.post_mode = "ok"
            jev_client.breaker.reset()
            with mock.patch.object(jev_client.CFG, "timeout", 0.3), mock.patch.object(jev_client, "DEFAULT_URL", self.fake.url):
                setup()
                led = Ledger()
                res, logs = await self.admit_degraded("比特币现在多少钱", led)
            with self.subTest(label):
                self.assertEqual((res.via, res.degraded_reason, res.route), ("degraded", reason, "WORKER"))
                self.assertIsNotNone(res.job)
                self.assertEqual(res.job.objective, "比特币现在多少钱")
                self.assertEqual(res.speak.kind, "pad")
                self.assertEqual(res.speak.text, api.DEGRADED_PAD)
                self.assertEqual(len(led.tasks), 1)
                self.assertIn("via=degraded", logs)
                self.assertIn(f"reason={reason}", logs)

    async def test_with_an_active_task_it_asks_the_fixed_sentence_and_changes_nothing(self) -> None:
        async for label, setup, reason in self._each_failure():
            self.fake.post_mode = "ok"
            jev_client.breaker.reset()
            led = Ledger()
            rec = led.create("查北京明天天气")
            led.mark_running(rec.task_id)
            with mock.patch.object(jev_client.CFG, "timeout", 0.3), mock.patch.object(jev_client, "DEFAULT_URL", self.fake.url):
                setup()
                res, logs = await self.admit_degraded("取消吧", led)
            with self.subTest(label):
                self.assertEqual((res.via, res.degraded_reason, res.route), ("degraded", reason, "CLARIFY"))
                self.assertIsNone(res.job)
                self.assertEqual(res.speak.kind, "clarify")
                self.assertEqual(res.speak.text, "我这边判断模块暂时不可用，你是想改刚才那个，还是问个新的？")
                self.assertEqual(list(led.tasks), [rec.task_id])
                self.assertEqual((led.tasks[rec.task_id].version, led.tasks[rec.task_id].status.value), (1, "RUNNING"))

    async def test_breaker_open_goes_straight_to_degraded_without_touching_the_server(self) -> None:
        self.fake.post_mode = "500"
        for _ in range(jev_client.CFG.breaker_fails):
            await self.reason_of_call()
        posts = self.fake.posts
        res, _ = await self.admit_degraded("现在几点了")
        self.assertEqual((res.via, res.degraded_reason, res.jev_state), ("degraded", "breaker_open", "open"))
        self.assertEqual(self.fake.posts, posts)

    async def test_jev_disabled_is_degraded_too(self) -> None:
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(route_mod, "heuristic_route", side_effect=AssertionError("rule")):
            res = await admit(_req("现在几点了"), Ledger())
        self.assertEqual((res.via, res.degraded_reason, res.jev_state), ("degraded", "jev_disabled", "disabled"))

    async def test_admit_source_never_calls_heuristic_route(self) -> None:
        src = inspect.getsource(api)
        self.assertNotIn("heuristic_route(", src)
        self.assertFalse(hasattr(api, "heuristic_route"))
        self.assertTrue(callable(route_mod.heuristic_route), "stays defined (stage 3/4 removes it), just uncalled")

    async def test_no_reference_to_heuristic_route_under_any_name(self) -> None:
        """AST check: catches `from .route import heuristic_route as hr`, `route.heuristic_route(...)`, and imports inside functions."""
        import ast
        for mod in (api, jev_client):
            for node in ast.walk(ast.parse(inspect.getsource(mod))):
                names = []
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name.split(".")[-1] for a in node.names]
                elif isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in ("heuristic_route", "route.heuristic_route"):
                    names = [node.value]  # getattr(route, "heuristic_route")
                self.assertNotIn("heuristic_route", names, f"{mod.__name__}:{getattr(node, 'lineno', '?')} references heuristic_route")

    async def test_a_registered_secondary_decides_when_jev_does_not(self) -> None:
        calls = []

        async def secondary(state, questions):  # noqa: ANN001
            calls.append(state["utterance"])
            return {"answers": {"route": {"type": "choice", "choice": "WORKER"}}}

        self.fake.post_mode = "500"
        jev_client.register_secondary(secondary)
        a, b, c, d = self.spies()
        with a, b, c, d:
            res = await admit(_req("今天原油价格"), Ledger())
        self.assertEqual((res.via, res.route, calls), ("secondary", "WORKER", ["今天原油价格"]))

    async def test_no_secondary_is_configured_by_default(self) -> None:
        self.assertIsNone(jev_client.SECONDARY)
        src = inspect.getsource(jev_client)
        self.assertNotIn("environ.get(\"SOPHON_JEV_SECONDARY", src, "no env switch can point it at a paid resource")

    async def test_a_failing_secondary_falls_back_to_degraded(self) -> None:
        async def broken(state, questions):  # noqa: ANN001
            raise RuntimeError("down")

        self.fake.post_mode = "500"
        jev_client.register_secondary(broken)
        res, _ = await self.admit_degraded("现在几点了")
        self.assertEqual((res.via, res.degraded_reason), ("degraded", "http_5xx"))


class CountersAndVisibilityTests(Base):
    async def test_ok_fail_degraded_counters_and_per_answer_type(self) -> None:
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False):
            with self.assertLogs("sophon_live.coordinator.jev", "INFO") as cm:
                await admit(_req("现在几点了", "a"), Ledger())
            self.assertTrue(any("jev answers route=choice(conf=0.69)" in o for o in cm.output), cm.output)
            self.fake.post_mode = "noul"
            await admit(_req("现在几点了", "b"), Ledger())
            self.fake.post_mode = "500"
            await admit(_req("现在几点了", "c"), Ledger())
        s = jev_client.stats
        self.assertEqual(s.ok, 1)
        self.assertEqual(s.fail, {"not_choice": 1, "http_5xx": 1})
        self.assertEqual(s.degraded, {"not_choice:noul": 1, "http_5xx": 1})
        self.assertEqual((s.n_fail, s.n_degraded), (2, 2))
        self.assertEqual(s.answers, {"route:choice": 1, "route:noul": 1})
        self.assertEqual(s.line(), "jev_ok=1 jev_fail=2 jev_degraded=2")

    async def test_the_non_choice_path_is_no_longer_silent_and_no_longer_mislabelled(self) -> None:
        self.fake.post_mode = "noul"
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False), \
                self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            res = await admit(_req("现在几点了"), Ledger())
        text = "\n".join(cm.output)
        self.assertNotEqual(res.via, "jev")
        self.assertEqual(res.via, "degraded")
        self.assertIn("route is not a choice (type=noul)", text)
        self.assertIn("via=degraded", text)
        self.assertIn("reason=not_choice:noul", text)
        self.assertIn("jev_ok=0 jev_fail=1 jev_degraded=1", text)

    async def test_admit_log_line_carries_the_counters_and_state(self) -> None:
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False), \
                self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            res = await admit(_req("现在几点了"), Ledger())
        line = [o for o in cm.output if "coord admit" in o][0]
        self.assertIn("via=jev", line)
        self.assertIn("jev_ok=1 jev_fail=0 jev_degraded=0", line)
        self.assertEqual((res.via, res.jev_state, res.degraded_reason), ("jev", "closed", None))

    async def test_control_without_a_usable_target_is_visible_not_silent(self) -> None:
        led = Ledger()
        rec = led.create("查北京明天天气")
        led.mark_running(rec.task_id)
        self.fake.post_mode = "nocontrol"
        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False), \
                self.assertLogs("sophon_live.coordinator", "INFO") as cm:
            res = await admit(_req("取消吧"), led)
        text = "\n".join(cm.output)
        self.assertIn("control answer is missing; falling back to heuristic_control", text)
        self.assertEqual(jev_client.stats.degraded, {"control_not_choice:missing": 1})
        self.assertEqual(res.degraded_reason, "control_not_choice:missing")

    async def test_confidence_is_logged_but_never_changes_the_route(self) -> None:
        for conf in (0.01, 0.99, None):
            ans = {"type": "choice", "choice": "WORKER"}
            if conf is not None:
                ans["confidence"] = conf
            resp = {"answers": {"route": ans}}
            self.assertEqual(jev_client.confidence(resp, "route"), conf)
            with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "REFINE_GUARD", False), \
                    mock.patch.object(jev_client, "system_one", mock.AsyncMock(return_value=resp)):
                res = await admit(_req("现在几点了", f"c{conf}"), Ledger())
            self.assertEqual((res.via, res.route), ("jev", "WORKER"), conf)
        self.assertNotIn("confidence", inspect.getsource(api).replace("confidence is", ""), "no routing code reads it")


if __name__ == "__main__":
    unittest.main()
