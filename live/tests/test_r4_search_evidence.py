"""Round 4b (Critic v1 review S-11, S-9): a search that found nothing is not evidence; a captcha page does not end the hedge;
cancelled hedges cannot pile up threads."""
from __future__ import annotations

import asyncio
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.assistant_tools import web_search
from sophon_live.coordinator import Ledger, WorkerJob, finish_job
from tests.test_orch_worker import _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession
from tests.test_web_search_fallback import BING_HTML

CAPTCHA = b"<html><body>captcha: please verify you are human</body></html>"


def _sequence(pages: list):
    """Fake _fetch_bytes: the n-th call gets pages[n] (bytes, or an Exception to raise); the last entry repeats."""
    async def fetch(url: str, timeout: float = 5.5) -> bytes:
        i = min(len(fetch.calls), len(pages) - 1)
        fetch.calls.append(url)
        v = pages[i]
        if isinstance(v, Exception):
            raise v
        return v
    fetch.calls = []
    return fetch


class CaptchaDoesNotEndTheHedge(unittest.IsolatedAsyncioTestCase):
    async def test_empty_page_then_good_page_is_answered_from_the_good_page(self) -> None:
        fetch = _sequence([CAPTCHA, BING_HTML])
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search._bing("美元兑人民币", 3, True, 5.0)
        self.assertEqual(len(fetch.calls), 2)                   # the captcha page did not stop the search at the first attempt
        self.assertIn("外汇牌价", out[0])

    async def test_error_empty_error_then_good(self) -> None:
        fetch = _sequence([OSError("reset"), CAPTCHA, BING_HTML])
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search._bing("美元兑人民币", 3, True, 5.0)
        self.assertTrue(out)

    async def test_only_captcha_pages_is_no_results_not_an_outage(self) -> None:
        fetch = _sequence([CAPTCHA])
        with mock.patch.object(web_search, "_fetch_bytes", fetch), mock.patch.object(web_search, "_BING_HEDGE_S", 0.05):
            self.assertEqual(await web_search._bing("美元兑人民币", 3, True, 2.0), [])
        self.assertGreater(len(fetch.calls), 1)                 # it used its retries before giving up

    async def test_only_errors_is_still_an_outage(self) -> None:
        fetch = _sequence([OSError("down")])
        with mock.patch.object(web_search, "_fetch_bytes", fetch), mock.patch.object(web_search, "_BING_HEDGE_S", 0.05):
            with self.assertRaises(OSError):
                await web_search._bing("美元兑人民币", 3, True, 2.0)


class NoResultsIsNotEvidence(unittest.IsolatedAsyncioTestCase):
    NOTHING = "没有搜到关于「美元兑人民币」的可靠摘要。"

    async def _run(self, tool_out: str, model_after: str = "今天美元兑人民币大约七点二。"):
        job = WorkerJob(task_id="t1", version=1, objective="今天美元兑人民币是多少", require_tools=["web_search"])
        sess = _FakeSession([_FakeResp(200, _tool_call("web_search", '{"query":"美元兑人民币"}')),
                             _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": model_after}}]})])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value=tool_out)):
            return await orch.run_job(job)

    async def test_nothing_found_is_a_failed_job_with_the_tools_own_sentence(self) -> None:
        res = await self._run(self.NOTHING)
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.evidence[0]["ok"])
        self.assertEqual(res.spoken_answer, self.NOTHING)
        self.assertNotIn("七点二", res.spoken_answer)             # the model's improvisation never gets a turn

    async def test_english_no_results_too(self) -> None:
        res = await self._run("I couldn't find useful snippets for 'usd cny'.")
        self.assertEqual(res.status, "failed")

    async def test_what_the_user_hears_is_the_honest_sentence(self) -> None:
        res = await self._run(self.NOTHING)
        led = Ledger(); rec = led.create("今天美元兑人民币是多少"); led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["web_search"])
        res.task_id, res.version = rec.task_id, rec.version
        out = finish_job(led, job, res)
        self.assertEqual(out.text, self.NOTHING)

    async def test_real_snippets_are_still_a_success(self) -> None:
        res = await self._run("关于「美元兑人民币」，中国银行外汇牌价：美元 713.20。")
        self.assertTrue(res.evidence[0]["ok"])


class InflightCap(unittest.TestCase):
    def test_requests_beyond_the_cap_fail_at_once_and_release_on_finish(self) -> None:
        gate = threading.Event()
        started = threading.Semaphore(0)

        def slow(url, timeout):  # noqa: ANN001
            started.release()
            gate.wait(5)
            return b"ok"

        with mock.patch.object(web_search, "_inflight", threading.BoundedSemaphore(2)), mock.patch.object(web_search, "_urlopen_unbounded", slow):
            ts = [threading.Thread(target=web_search._urlopen, args=("https://x", 1.0)) for _ in range(2)]
            for t in ts:
                t.start()
            for _ in range(2):
                self.assertTrue(started.acquire(timeout=2))
            with self.assertRaises(OSError):
                web_search._urlopen("https://x", 1.0)           # the third does not even start a socket
            gate.set()
            for t in ts:
                t.join(2)
            self.assertEqual(web_search._urlopen("https://x", 1.0), b"ok")      # slots were released


if __name__ == "__main__":
    unittest.main()
