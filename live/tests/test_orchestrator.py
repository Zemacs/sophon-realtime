"""Unit tests for Path-1 co-located orchestrator (mocked LLM HTTP)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import orchestrator as orch  # noqa: E402


class _FakeResp:
    def __init__(self, status: int, payload: dict | str):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):  # noqa: ANN001, ARG002
        if isinstance(self._payload, Exception):
            raise self._payload
        if not isinstance(self._payload, dict):
            raise json.JSONDecodeError("expect object", "", 0)
        return self._payload

    async def text(self) -> str:
        if isinstance(self._payload, dict):
            return json.dumps(self._payload)
        return str(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):  # noqa: ANN002
        return False


class _FakeSession:
    def __init__(self, responses: list[_FakeResp]):
        self._responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None):  # noqa: A002, ANN001
        self.posts.append({"url": url, "json": json, "headers": headers})
        if not self._responses:
            raise AssertionError("unexpected LLM call")
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):  # noqa: ANN002
        return False


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_task(self) -> None:
        out = await orch.resolve_task("  ")
        self.assertIn("没有收到", out)

    async def test_direct_answer_strips_say(self) -> None:
        sess = _FakeSession([
            _FakeResp(200, {"choices": [{"message": {"content": "SAY: 现在是下午三点。"}}]}),
        ])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            out = await orch.resolve_task("现在几点？")
        self.assertEqual(out, "现在是下午三点。")
        self.assertEqual(len(sess.posts), 1)
        self.assertIn("tools", sess.posts[0]["json"])

    async def test_tool_then_final(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_datetime", "arguments": '{"what":"time"}'},
        }
        sess = _FakeSession([
            _FakeResp(200, {"choices": [{"message": {"content": "", "tool_calls": [tool_call]}}]}),
            _FakeResp(200, {"choices": [{"message": {"content": "现在是下午两点。"}}]}),
        ])
        with (
            mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess),
            mock.patch.object(orch, "run_tool", new=mock.AsyncMock(return_value="现在是 14:00。")) as rt,
        ):
            out = await orch.resolve_task("几点了")
        self.assertEqual(out, "现在是下午两点。")
        rt.assert_awaited_once_with("get_datetime", {"what": "time"})
        self.assertEqual(len(sess.posts), 2)
        tool_msg = sess.posts[1]["json"]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("14:00", tool_msg["content"])

    async def test_http_error(self) -> None:
        sess = _FakeSession([_FakeResp(503, {"error": "busy"})])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess):
            out = await orch.resolve_task("查天气")
        self.assertIn("不可用", out)

    async def test_cfg_reads_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SOPHON_LIVE_ORCH_URL": "http://127.0.0.1:9999/v1/chat/completions",
                "SOPHON_LIVE_ORCH_MODEL": "test-model",
                "SOPHON_LIVE_ORCH_STEPS": "2",
            },
            clear=False,
        ):
            url, model, _key, steps, _timeout = orch._cfg()
        self.assertEqual(url, "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(model, "test-model")
        self.assertEqual(steps, 2)


if __name__ == "__main__":
    unittest.main()
