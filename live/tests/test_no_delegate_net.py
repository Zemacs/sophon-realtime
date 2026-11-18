"""v0.1.0-rc1: the gateway delegates a tool-backed request itself when the voice front ended its turn with only a lead-in."""
from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

import sophon_live.gateway as gw
from tests.test_r0_coord_boundary import _bare_session


def _session(user_text: str | None, age_s: float = 0.2):
    s = _bare_session()
    s.closed = False
    s.id = "sess_net"
    s.resp = {"r1": {"asr": None, "deleg_n0": s.deleg_calls, "t0": time.monotonic() - 0.5}}
    s.last_user = (user_text, time.monotonic() - age_s) if user_text is not None else None
    calls: list = []

    async def fake_delegate(e):
        calls.append(e)

    s._delegate = fake_delegate
    return s, calls


def _run(s) -> None:
    gw.NET_GRACE_S = 0.0
    asyncio.run(s._no_delegate_net("r1"))


class NoDelegateNet(unittest.TestCase):
    def test_delegates_when_the_front_only_said_the_lead_in(self) -> None:
        for text in ("你还记得我上次跟你说我喜欢吃什么吗", "明天上海会下雨吗", "关掉客厅的灯"):
            s, calls = _session(text)
            _run(s)
            self.assertEqual(len(calls), 1, text)
            self.assertEqual(calls[0]["arguments"]["task"], text)
            self.assertIsNone(calls[0]["call_id"])
            self.assertEqual(calls[0]["net"], "no_delegate")

    def test_not_when_the_front_called_delegate_in_that_turn(self) -> None:
        s, calls = _session("明天上海会下雨吗")
        s.deleg_calls = 1                                    # the tool call arrived after the response started
        _run(s)
        self.assertEqual(calls, [])

    def test_not_for_chit_chat_jokes_or_knowledge(self) -> None:
        for text in ("你好呀", "讲个笑话", "光速是多少"):
            s, calls = _session(text)
            _run(s)
            self.assertEqual(calls, [], text)

    def test_not_without_a_fresh_transcript_or_twice_for_the_same_utterance(self) -> None:
        s, calls = _session(None)
        _run(s)
        self.assertEqual(calls, [])
        s, calls = _session("明天上海会下雨吗", age_s=60)    # a stale transcript from an earlier turn
        _run(s)
        self.assertEqual(calls, [])
        s, calls = _session("明天上海会下雨吗")
        s.last_deleg_asr_t = s.last_user[1]                   # this utterance was already delegated
        _run(s)
        self.assertEqual(calls, [])

    def test_a_failing_net_never_raises(self) -> None:
        s, _ = _session("明天上海会下雨吗")

        async def boom(_e):
            raise RuntimeError("x")

        s._delegate = boom
        _run(s)                                              # logged, not raised


if __name__ == "__main__":
    unittest.main()
