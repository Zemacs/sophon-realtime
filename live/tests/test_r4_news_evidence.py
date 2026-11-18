"""Round 4b+ (Critic v1-news-budget S-14, S-16): news failures stay honest at every layer.

  * an empty / unparsable feed is a failure sentence, never a headline list
  * the news tool shares the in-flight cap with web_search
  * a feed that answers after the budget moved on (production path: a blocked ``to_thread(_urlopen)`` thread) is never written in
  * C layer: a get_news failure cannot be turned into made-up headlines (orch + finish_job)
Budget tests need ``news._budget_s`` (v1/7, builder/v1-news-budget); on a tree without it they skip and run on the merged tree.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import inflight, news, web_search
from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.coordinator import Ledger, WorkerJob, WorkerResult, finish_job
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER
from tests.test_orch_worker import _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession

DOWN_ZH = "新闻源暂时连不上，稍后再帮你看看头条吧。"
FAKE = "今天的头条：第1条，某某公司宣布破产；第2条，某地发生重大事故。"
GOOD_RSS = "<?xml version='1.0'?><rss><channel><item><title>科技：真实头条一</title></item><item><title>科技：真实头条二</title></item></channel></rss>".encode()
LATE_RSS = "<?xml version='1.0'?><rss><channel><item><title>迟到的头条</title></item></channel></rss>".encode()
EMPTY_RSS = b"<?xml version='1.0'?><rss><channel></channel></rss>"


def _run(coro):
    return asyncio.run(coro)


class EmptyFeedsAreFailures(unittest.TestCase):
    def test_empty_rss_everywhere_is_the_unreachable_sentence(self) -> None:
        for lang, want in (("zh", "连不上"), ("en", "unreachable")):
            for body in (EMPTY_RSS, b"", b"<html>blocked</html>"):
                with mock.patch.object(news, "_fetch_rss", new=mock.AsyncMock(return_value=body)):
                    out = _run(news.run({"topic": "科技" if lang == "zh" else "tech", "lang": lang}))
                self.assertIn(want, out, (lang, body))
                self.assertNotIn("第1条", out)
                self.assertNotIn("Headlines on", out)

    def test_headlines_still_come_through_when_the_feed_has_items(self) -> None:
        with mock.patch.object(news, "_fetch_rss", new=mock.AsyncMock(return_value=GOOD_RSS)):
            out = _run(news.run({"topic": "科技", "lang": "zh"}))
        self.assertIn("第1条", out)
        self.assertIn("真实头条一", out)


class SharedInflightCap(unittest.TestCase):
    def test_news_and_search_share_one_cap(self) -> None:
        self.assertIs(inflight.SHARED, web_search._inflight)

    def test_news_over_the_cap_fails_honestly_and_at_once(self) -> None:
        full = threading.BoundedSemaphore(1)
        full.acquire()                                                 # every slot taken (by searches, say)
        with mock.patch.object(inflight, "SHARED", full), mock.patch.object(news, "_urlopen", side_effect=AssertionError("must not even start")):
            t0 = time.monotonic()
            out = _run(news.run({"topic": "科技", "lang": "zh"}))
        self.assertIn("连不上", out)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_slots_are_released_after_a_news_request(self) -> None:
        sem = threading.BoundedSemaphore(2)
        with mock.patch.object(inflight, "SHARED", sem), mock.patch.object(news, "_urlopen", return_value=GOOD_RSS):
            for _ in range(5):
                _run(news.run({"topic": "科技", "lang": "zh"}))
        self.assertTrue(sem.acquire(blocking=False) and sem.acquire(blocking=False))     # both slots free again


@unittest.skipUnless(hasattr(news, "_budget_s"), "news budget comes with v1/7 (builder/v1-news-budget)")
class LateFeedIsNeverWrittenIn(unittest.TestCase):
    def test_blocked_thread_answering_after_the_budget_moved_on_is_ignored(self) -> None:
        """The production path: ``to_thread(_urlopen)``. wait_for cancels the awaiting side, the blocked thread cannot be cancelled and
        returns its (late) page later; that page must not appear in the answer."""
        done_late = threading.Event()
        calls: list[str] = []

        def urlopen(url: str) -> bytes:
            calls.append(url)
            if len(calls) == 1:
                time.sleep(1.2)                                         # the first feed stalls past its slice
                done_late.set()
                return LATE_RSS
            return GOOD_RSS

        with mock.patch.object(news, "_urlopen", urlopen), mock.patch.object(news, "_PER_FEED_S", 0.3):
            async def timed():                                           # (asyncio.run itself waits for the executor thread on exit; time the call, not the loop)
                t0 = time.monotonic()
                got = await news.run({"topic": "科技", "lang": "zh"})
                return got, time.monotonic() - t0

            out, took = _run(timed())
        self.assertIn("真实头条一", out)
        self.assertNotIn("迟到的头条", out)
        self.assertLess(took, 1.0)                                      # it did not wait for the stalled thread
        self.assertTrue(done_late.wait(3))                              # the late thread did finish meanwhile...
        self.assertNotIn("迟到的头条", out)                               # ...and the answer already returned is unchanged
        self.assertEqual(len(calls), 2)

    def test_budget_spent_is_the_unreachable_sentence_not_a_partial_list(self) -> None:
        def urlopen(url: str) -> bytes:
            time.sleep(0.8)
            return LATE_RSS

        with mock.patch.dict(os.environ, {"SOPHON_NEWS_BUDGET": "2"}), mock.patch.object(news, "_budget_s", return_value=0.5), \
                mock.patch.object(news, "_PER_FEED_S", 0.2), mock.patch.object(news, "_urlopen", urlopen):
            out = _run(news.run({"topic": "科技", "lang": "zh"}))
        self.assertIn("连不上", out)
        self.assertNotIn("迟到的头条", out)
        time.sleep(1.0)                                                 # let the stragglers finish (no thread error, nothing leaks into state)


class NewsFailureCannotBecomeHeadlines(unittest.IsolatedAsyncioTestCase):
    async def _orch(self, tool_out: str, model_after: str = FAKE):
        job = WorkerJob(task_id="t1", version=1, objective="今天有什么科技新闻", require_tools=["get_news"])
        sess = _FakeSession([_FakeResp(200, _tool_call("get_news", '{"topic":"科技"}')),
                             _FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": model_after}}]})])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value=tool_out)):
            return await orch.run_job(job)

    async def test_orch_failure_is_the_tools_sentence_and_the_model_gets_no_turn(self) -> None:
        for out in (DOWN_ZH, "News feeds are unreachable right now. Try again in a moment."):
            res = await self._orch(out)
            self.assertEqual(res.status, "failed")
            self.assertFalse(res.evidence[0]["ok"])
            self.assertEqual(res.spoken_answer, out)
            self.assertNotIn("第1条", res.spoken_answer)

    def _finish(self, status: str, said: str, ev: dict):
        led = Ledger()
        rec = led.create("今天有什么科技新闻")
        led.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=["get_news"])
        return finish_job(led, job, WorkerResult(rec.task_id, rec.version, status, said, evidence=[ev]))

    def test_finish_job_never_speaks_made_up_headlines_after_a_news_failure(self) -> None:
        shapes = {
            "completed + ok=False": ("completed", {"tool": "get_news", "ok": False}),
            "completed + no ok": ("completed", {"tool": "get_news"}),
            "failed + ok=False": ("failed", {"tool": "get_news", "ok": False}),
            "failed + failure_text is the fake": ("failed", {"tool": "get_news", "ok": False, "failure_text": FAKE}),
        }
        for name, (status, ev) in shapes.items():
            out = self._finish(status, FAKE, ev)
            self.assertEqual(out.text, NO_TOOL_ANSWER, name)
            self.assertNotIn("第1条", out.text, name)

    def test_finish_job_speaks_the_tools_own_unreachable_sentence(self) -> None:
        out = self._finish("failed", DOWN_ZH, {"tool": "get_news", "ok": False, "args": {"topic": "科技"}, "failure_text": DOWN_ZH})
        self.assertEqual((out.text, out.rejected), (DOWN_ZH, ""))


if __name__ == "__main__":
    unittest.main()
