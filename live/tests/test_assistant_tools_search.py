"""Offline tests for web_search, news, and memory assistant tools."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sophon_live.assistant_tools import memory, news, web_search


def _run(coro):
    return asyncio.run(coro)


class _FakeResp:
    def __init__(self, body: bytes, content_type: str = "application/json"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class WebSearchTests(unittest.TestCase):
    def test_instant_answer_summary(self) -> None:
        payload = json.dumps(
            {
                "AbstractText": "Python is a programming language.",
                "Heading": "Python",
                "RelatedTopics": [{"Text": "Python (programming language) - high level language."}],
            }
        ).encode()
        lite = b"<html></html>"

        def fake_urlopen(req, timeout=20):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "api.duckduckgo.com" in url:
                return _FakeResp(payload)
            if "lite.duckduckgo.com" in url:
                return _FakeResp(lite, "text/html")
            raise AssertionError(url)

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _run(web_search.run({"query": "Python language", "max_results": 2}))
        self.assertIn("Python", out)
        self.assertIn("programming language", out)

    def test_network_error_spoken(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            out = _run(web_search.run({"query": "测试", "max_results": 3}))
        self.assertIn("网络", out)
        self.assertNotIn("offline", out)

    def test_lite_html_fallback(self) -> None:
        instant = json.dumps({"AbstractText": "", "RelatedTopics": []}).encode()
        lite = """
        <html><body>
        <a class='result-link' href="/l/1">First title</a>
        <span class='result-snippet'>First snippet text</span>
        <a class='result-link' href="/l/2">Second title</a>
        <span class='result-snippet'>Second snippet</span>
        </body></html>
        """.encode()

        def fake_urlopen(req, timeout=20):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "api.duckduckgo.com" in url:
                return _FakeResp(instant)
            if "lite.duckduckgo.com" in url:
                return _FakeResp(lite, "text/html")
            raise AssertionError(url)

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _run(web_search.run({"query": "fallback query", "max_results": 2, "lang": "en"}))
        self.assertIn("First title", out)
        self.assertIn("Second title", out)


class NewsTests(unittest.TestCase):
    def test_rss_headlines_zh(self) -> None:
        rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
        <item><title>芯片出口新规 - Source A</title></item>
        <item><title>AI 监管框架公布 - Source B</title></item>
        <item><title>电动车销量创新高 - Source C</title></item>
        </channel></rss>""".encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=_FakeResp(rss, "application/rss+xml")):
            out = _run(news.run({"topic": "科技", "lang": "zh", "max_items": 3}))
        self.assertIn("芯片出口新规", out)
        self.assertNotIn("Source A", out)  # the trailing 「 - 来源」 is dropped, not the headline
        self.assertIn("第1条", out)
        self.assertIn("第2条", out)

    def test_publisher_is_taken_from_the_source_element_in_either_order(self) -> None:
        """Critic S-1: with <source>, remove exactly the publisher, whichever side of " - " it is on; never the headline."""
        rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
        <item><title>芯片出口新规 - 新华社</title><source url="https://x">新华社</source></item>
        <item><title>路透社 - 特斯拉发布新电池</title><source url="https://y">路透社</source></item>
        <item><title>特斯拉 - 马斯克新计划 - 路透社</title><source url="https://y">路透社</source></item>
        <item><title>某某报 - 与来源不符的标题</title><source url="https://z">别家</source></item>
        </channel></rss>""".encode("utf-8")
        titles = news._titles_from_rss(rss)
        self.assertEqual(titles, ["芯片出口新规", "特斯拉发布新电池", "特斯拉 - 马斯克新计划", "某某报 - 与来源不符的标题"])

    def test_title_keeps_inner_dashes_and_untagged_titles(self) -> None:
        rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
        <item><title>特斯拉 - 马斯克新计划 - 路透社</title></item>
        <item><title>没有来源后缀的标题</title></item>
        </channel></rss>""".encode("utf-8")
        self.assertEqual(news._titles_from_rss(rss), ["特斯拉 - 马斯克新计划", "没有来源后缀的标题"])

    def test_network_error_en(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            out = _run(news.run({"topic": "NVIDIA", "lang": "en"}))
        self.assertIn("unreachable", out.lower())


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "assistant_memory.json"
        self._orig_env = os.environ.get("SOPHON_LIVE_MEMORY_PATH")
        os.environ["SOPHON_LIVE_MEMORY_PATH"] = str(self._path)

    def tearDown(self) -> None:
        if self._orig_env is None:
            os.environ.pop("SOPHON_LIVE_MEMORY_PATH", None)
        else:
            os.environ["SOPHON_LIVE_MEMORY_PATH"] = self._orig_env
        self._tmpdir.cleanup()

    def test_save_recall_list(self) -> None:
        out_save = _run(memory.run({"op": "save", "key": "city", "value": "上海"}))
        self.assertIn("记住", out_save)
        self.assertTrue(self._path.is_file())

        out_recall = _run(memory.run({"op": "recall", "key": "city"}))
        self.assertIn("上海", out_recall)

        out_list = _run(memory.run({"op": "list"}))
        self.assertIn("city", out_list)

    def test_recall_missing(self) -> None:
        out = _run(memory.run({"op": "recall", "key": "missing"}))
        self.assertIn("还没有", out)


if __name__ == "__main__":
    unittest.main()