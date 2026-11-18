"""Offline tests for live.sophon_live.assistant_tools (no GPU)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIVE))

from sophon_live.assistant_tools import list_tool_names, run_task, run_tool  # noqa: E402
from sophon_live.assistant_tools import memory as memory_mod  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_names(self):
        names = list_tool_names()
        for n in ("get_datetime", "get_weather", "web_search", "get_news", "device_control", "memory_note"):
            self.assertIn(n, names)


class TestDatetimeDeviceMemory(unittest.TestCase):
    def test_datetime(self):
        text = asyncio.run(run_tool("get_datetime", {"what": "both"}))
        self.assertTrue(text)
        self.assertRegex(text, r"点|AM|PM|Today|现在")

    def test_device_mock(self):
        text = asyncio.run(run_tool("device_control", {"device": "灯", "action": "打开"}))
        self.assertIn("模拟", text)

    def test_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mem.json"
            os.environ["SOPHON_LIVE_MEMORY_PATH"] = str(path)
            try:
                self.assertIn("记住", asyncio.run(run_tool("memory_note", {"op": "save", "key": "city", "value": "上海"})))
                self.assertIn("上海", asyncio.run(run_tool("memory_note", {"op": "recall", "key": "city"})))
                self.assertIn("city", asyncio.run(run_tool("memory_note", {"op": "list"})))
            finally:
                os.environ.pop("SOPHON_LIVE_MEMORY_PATH", None)


class TestRouting(unittest.TestCase):
    def test_route_time(self):
        with mock.patch("sophon_live.assistant_tools.runner.run_tool", new=mock.AsyncMock(return_value="现在是1点")) as m:
            out = asyncio.run(run_task("现在几点了"))
            self.assertEqual(out, "现在是1点")
            self.assertEqual(m.await_args.args[0], "get_datetime")

    def test_route_news(self):
        with mock.patch("sophon_live.assistant_tools.runner.run_tool", new=mock.AsyncMock(return_value="新闻")) as m:
            out = asyncio.run(run_task("查一下关于黄仁勋的新闻"))
            self.assertEqual(out, "新闻")
            self.assertEqual(m.await_args.args[0], "get_news")


if __name__ == "__main__":
    unittest.main()
