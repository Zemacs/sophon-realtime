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

from sophon_live.assistant_tools import list_tool_names, run_tool  # noqa: E402
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


class TestLegacyHeuristicRoutingIsGone(unittest.IsolatedAsyncioTestCase):
    """Round 4g: the SOPHON_LIVE_BUILTIN_TOOLS demo (keyword regexes mapping a task to a tool) was deleted. Which tool a request needs is the
    model's decision (orchestrator); the runner only executes by name. These replace the two TestRouting tests that pinned the old behavior."""

    def test_the_entry_points_no_longer_exist(self):
        import sophon_live.assistant_tools as tools
        from sophon_live.assistant_tools import runner

        for mod in (tools, runner):
            self.assertFalse(hasattr(mod, "run_task"), mod.__name__)
        self.assertFalse(hasattr(runner, "_route_task"))
        self.assertNotIn("run_task", tools.__all__)
        with self.assertRaises(ImportError):
            from sophon_live.assistant_tools import run_task  # noqa: F401

    def test_no_source_file_mentions_the_flag_or_its_helpers(self):
        root = LIVE / "sophon_live"
        needles = ("SOPHON_LIVE_BUILTIN_TOOLS", "use_builtin", "_builtin_resolve", "_route_task", "run_task", '"builtin"')
        for path in list(root.rglob("*.py")) + list(root.rglob("*.html")):
            text = path.read_text(encoding="utf-8")
            for n in needles:
                self.assertNotIn(n, text, f"{path.relative_to(LIVE)} still mentions {n}")

    async def test_the_flag_in_the_environment_has_no_effect(self):
        """COORD and ORCH off + the old flag on: the gateway starts no resolver and announces no backend (before: backend=builtin + a task)."""
        from tests.test_r0_coord_boundary import _bare_session
        import sophon_live.gateway as gw

        session = _bare_session()
        session._cut_user = lambda: None
        session._trace = lambda: mock.MagicMock()
        started: list = []
        env = {"SOPHON_LIVE_BUILTIN_TOOLS": "1", "SOPHON_LIVE_COORD": "", "SOPHON_LIVE_ORCH": ""}
        with mock.patch.dict(os.environ, env), mock.patch.object(gw.asyncio, "create_task", side_effect=lambda c: (started.append(c), c.close())[0]), \
                mock.patch.object(gw, "DELEG_MODE", "client"):
            await session._delegate({"call_id": "c1", "arguments": {"task": "现在几点了"}})
        created = [e for e in session.sent if e.get("type") == "session.delegation.created"]
        self.assertEqual(len(created), 1)
        self.assertNotIn("backend", created[0]["delegation"])
        self.assertEqual(started, [])
        # and the real resolvers are still wired when their own flags are set
        for flag, backend in (("SOPHON_LIVE_ORCH", "orch"), ("SOPHON_LIVE_COORD", "coord")):
            session.sent.clear()
            env2 = {"SOPHON_LIVE_BUILTIN_TOOLS": "1", "SOPHON_LIVE_COORD": "", "SOPHON_LIVE_ORCH": "", flag: "1"}
            with mock.patch.dict(os.environ, env2), mock.patch.object(gw.asyncio, "create_task", side_effect=lambda c: (started.append(c), c.close())[0]), \
                    mock.patch.object(gw, "DELEG_MODE", "client"):
                await session._delegate({"call_id": "c2", "arguments": {"task": "x"}})
            created = [e for e in session.sent if e.get("type") == "session.delegation.created"]
            self.assertEqual(created[0]["delegation"]["backend"], backend)
        self.assertEqual(len(started), 2)


if __name__ == "__main__":
    unittest.main()
