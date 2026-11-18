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
