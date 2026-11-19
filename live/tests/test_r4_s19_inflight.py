"""Round 4c / S-19 (cap < pool is not enough: the cap is taken before queueing and the threads are a dedicated pool) and S-21
(the release-on-exception / release-on-cancel mutants of the in-flight cap must be killed)."""
from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest import mock

from sophon_live.assistant_tools import inflight, news, weather, web_search


def _all_slots_free(sem: threading.BoundedSemaphore, n: int) -> bool:
    got = 0
    while sem.acquire(blocking=False):
        got += 1
    for _ in range(got):
        sem.release()
    return got == n


class DedicatedPoolAndCap(unittest.IsolatedAsyncioTestCase):
    def test_the_pool_is_dedicated_and_as_big_as_the_cap(self) -> None:
        self.assertEqual(inflight.EXECUTOR._max_workers, inflight.MAX_INFLIGHT)
        self.assertTrue(inflight.EXECUTOR._thread_name_prefix.startswith("sophon-outbound"))

    async def test_the_13th_request_fails_at_once_and_asr_is_not_starved(self) -> None:
        n = inflight.MAX_INFLIGHT
        gate = threading.Event()
        started = threading.Semaphore(0)

        def stuck() -> str:
            started.release()
            gate.wait(10)
            return "late"

        tasks = [asyncio.ensure_future(inflight.run(stuck)) for _ in range(n)]
        try:
            for _ in range(n):
                self.assertTrue(await asyncio.to_thread(started.acquire, True, 3), "outbound threads did not all start")
            t0 = time.monotonic()
            with self.assertRaises(OSError):
                await inflight.run(lambda: "13th")                         # the (n+1)th: refused, not queued behind the stuck sockets
            self.assertLess(time.monotonic() - t0, 0.5)
            t0 = time.monotonic()
            self.assertEqual(await asyncio.to_thread(lambda: "asr"), "asr")      # the default pool (ASR, device) is untouched
            self.assertLess(time.monotonic() - t0, 0.5)
        finally:
            gate.set()
            await asyncio.gather(*tasks)
        await asyncio.sleep(0.05)
        self.assertTrue(_all_slots_free(inflight.SHARED, n))

    async def test_web_search_news_and_weather_all_fail_fast_when_the_cap_is_full(self) -> None:
        full = threading.BoundedSemaphore(1)
        full.acquire()
        boom = AssertionError("must not even start a socket")
        with mock.patch.object(inflight, "SHARED", full), mock.patch.object(web_search, "_inflight", full), \
                mock.patch.object(web_search, "_urlopen_unbounded", side_effect=boom), mock.patch.object(news, "_urlopen", side_effect=boom), \
                mock.patch.object(weather, "_http_get_json", side_effect=boom):
            for coro in (web_search._fetch_bytes("https://x", 1.0), news._fetch_rss("https://x"), weather._fetch_json("https://x", 1.0)):
                t0 = time.monotonic()
                with self.assertRaises(OSError):
                    await coro
                self.assertLess(time.monotonic() - t0, 0.5)

    async def test_weather_holds_a_slot_of_the_shared_cap_while_it_is_in_flight(self) -> None:
        sem = threading.BoundedSemaphore(1)
        gate = threading.Event()
        started = threading.Event()

        def slow_get(url, timeout=1.0):  # noqa: ANN001
            started.set()
            gate.wait(5)
            return {}

        with mock.patch.object(inflight, "SHARED", sem), mock.patch.object(weather, "_http_get_json", slow_get):
            w = asyncio.ensure_future(weather._fetch_json("https://x", 3.0))
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            with self.assertRaises(OSError):                 # the only slot is the weather request's: a search / news request is refused
                await inflight.run(lambda: "search")
            gate.set()
            self.assertEqual(await w, {})
            await asyncio.sleep(0.05)
            self.assertTrue(_all_slots_free(sem, 1))


class SlotsAreAlwaysReleased(unittest.IsolatedAsyncioTestCase):
    """Mutants of inflight.run that must not survive: no release after an exception; no release after the awaiter is cancelled."""

    async def test_slot_is_released_after_the_work_raises(self) -> None:
        sem = threading.BoundedSemaphore(2)

        def bad() -> None:
            raise ValueError("socket blew up")

        for _ in range(5):                                     # more failures than slots: a leak would exhaust the cap
            with self.assertRaises(ValueError):
                await inflight.run(bad, sem=sem)
        self.assertTrue(_all_slots_free(sem, 2))
        self.assertEqual(await inflight.run(lambda: 7, sem=sem), 7)

    async def test_cancelled_awaiter_keeps_the_slot_until_the_thread_ends_then_releases_it(self) -> None:
        sem = threading.BoundedSemaphore(1)
        gate = threading.Event()
        started = threading.Event()

        def stuck() -> None:
            started.set()
            gate.wait(5)

        task = asyncio.ensure_future(inflight.run(stuck, sem=sem))
        self.assertTrue(await asyncio.to_thread(started.wait, 3))
        task.cancel()                                          # wait_for / hedge gave up; the socket thread is still running
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(sem.acquire(blocking=False), "slot given back while the thread still holds a socket")
        gate.set()
        for _ in range(100):
            await asyncio.sleep(0.02)
            if sem.acquire(blocking=False):
                sem.release()
                break
        else:
            self.fail("slot never released after the cancelled thread finished (leak)")

    async def test_work_cancelled_before_it_started_releases_the_slot(self) -> None:
        sem = threading.BoundedSemaphore(1)

        class Queued:
            def submit(self, fn):                             # the pool accepted the item but never ran it, then it was cancelled
                f: Future = Future()
                f.cancel()
                return f

        with mock.patch.object(inflight, "EXECUTOR", Queued()):
            with self.assertRaises(asyncio.CancelledError):
                await inflight.run(lambda: 1, sem=sem)
        self.assertTrue(_all_slots_free(sem, 1))

    async def test_submit_failure_releases_the_slot(self) -> None:
        sem = threading.BoundedSemaphore(1)
        pool = ThreadPoolExecutor(1)
        pool.shutdown()                                        # submit raises RuntimeError
        with mock.patch.object(inflight, "EXECUTOR", pool):
            with self.assertRaises(RuntimeError):
                await inflight.run(lambda: 1, sem=sem)
        self.assertTrue(_all_slots_free(sem, 1))


if __name__ == "__main__":
    unittest.main()
