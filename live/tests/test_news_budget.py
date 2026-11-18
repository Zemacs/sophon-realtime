"""体验版 v1：get_news 整体有时间预算，慢源不拖到 77 s。"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import news

RSS = """<?xml version="1.0"?><rss><channel><item><title>示例科技公司发布新芯片</title></item><item><title>另一条科技新闻</title></item></channel></rss>""".encode()


class NewsBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_feeds_fail_honestly_within_the_budget(self) -> None:
        async def stall(url: str) -> bytes:
            await asyncio.sleep(60)
            return RSS

        with mock.patch.dict(os.environ, {"SOPHON_NEWS_BUDGET": "2"}), mock.patch.object(news, "_fetch_rss", stall), \
                mock.patch.object(news, "_PER_FEED_S", 0.6):
            t0 = time.monotonic()
            out = await news.run({"topic": "科技", "lang": "zh"})
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertIn("连不上", out)

    async def test_a_slow_first_feed_is_skipped_and_the_next_one_answers(self) -> None:
        calls: list[str] = []

        async def fetch(url: str) -> bytes:
            calls.append(url)
            if len(calls) == 1:
                await asyncio.sleep(60)
            return RSS

        with mock.patch.dict(os.environ, {"SOPHON_NEWS_BUDGET": "5"}), mock.patch.object(news, "_fetch_rss", fetch), \
                mock.patch.object(news, "_PER_FEED_S", 0.5):
            out = await news.run({"topic": "科技", "lang": "zh"})
        self.assertIn("新芯片", out)
        self.assertGreaterEqual(len(calls), 2)

    def test_default_budget(self) -> None:
        os.environ.pop("SOPHON_NEWS_BUDGET", None)
        self.assertEqual(news._budget_s(), 9.0)


if __name__ == "__main__":
    unittest.main()


class NewsFollowUpTests(unittest.IsolatedAsyncioTestCase):
    """Critic S-14 / S-16."""

    async def test_an_empty_feed_is_nothing_found_not_an_outage(self) -> None:
        empty = b'<?xml version="1.0"?><rss><channel></channel></rss>'

        async def fetch(url: str) -> bytes:
            return empty

        with mock.patch.object(news, "_fetch_rss", fetch):
            out = await news.run({"topic": "科技", "lang": "zh"})
        self.assertEqual(out, "没有找到相关头条。")
        self.assertNotIn("第1条", out)

    async def test_unreachable_feeds_still_say_so(self) -> None:
        async def fetch(url: str) -> bytes:
            raise OSError("down")

        with mock.patch.object(news, "_fetch_rss", fetch):
            self.assertIn("连不上", await news.run({"topic": "科技", "lang": "zh"}))

    async def test_a_late_result_after_the_budget_is_never_in_the_answer(self) -> None:
        async def slow(url: str) -> bytes:
            await asyncio.sleep(0.8)
            return RSS

        with mock.patch.dict(os.environ, {"SOPHON_NEWS_BUDGET": "2"}), mock.patch.object(news, "_fetch_rss", slow), \
                mock.patch.object(news, "_PER_FEED_S", 0.3):
            out = await news.run({"topic": "科技", "lang": "zh"})
        self.assertIn("连不上", out)
        self.assertNotIn("新芯片", out)

    async def test_socket_timeout_is_the_remaining_budget_and_shares_the_search_cap(self) -> None:
        from sophon_live.assistant_tools import web_search

        seen: list[float] = []

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return RSS

        def fake_urlopen(req, timeout=0):
            seen.append(timeout)
            return Resp()

        token = news._DEADLINE.set(time.monotonic() + 1.2)
        try:
            with mock.patch.object(news.urllib.request, "urlopen", fake_urlopen):
                news._urlopen("https://example.com/rss")
                self.assertLessEqual(seen[-1], 1.25)                       # not the fixed 5 s
                for _ in range(web_search._INFLIGHT_MAX):                  # exhaust the shared cap
                    web_search._inflight.acquire(blocking=False)
                try:
                    with self.assertRaises(OSError):
                        news._urlopen("https://example.com/rss")
                finally:
                    for _ in range(web_search._INFLIGHT_MAX):
                        try:
                            web_search._inflight.release()
                        except ValueError:
                            break
        finally:
            news._DEADLINE.reset(token)
