"""v0.1.0-rc2: the gateway's no-delegate safety net.

TRIGGER = structure only (the response is done, the front called no delegate in that turn, its output is lead-in sized, there is a fresh
un-delegated transcript). DECISION = the router model (``coordinator.api.judge_tools``, scripted here). No keyword test of the user's words.
"""
from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

import sophon_live.gateway as gw
from tests.test_r0_coord_boundary import _bare_session


def _session(user_text: str | None, age_s: float = 0.2, front_said: str = "我帮你查一下。"):
    s = _bare_session()
    s.closed = False
    s.id = "sess_net"
    s.resp = {"r1": {"asr": None, "deleg_n0": s.deleg_calls, "t0": time.monotonic() - 0.5, "text": front_said}}
    s.last_user = (user_text, time.monotonic() - age_s) if user_text is not None else None
    calls: list = []

    async def fake_delegate(e):
        calls.append(e)

    s._delegate = fake_delegate
    return s, calls


def _run(s) -> None:
    gw.NET_GRACE_S = 0.0
    asyncio.run(s._no_delegate_net("r1"))


def _model_says(tools):
    calls: list = []

    async def fake(text):
        calls.append(text)
        return tools

    return mock.patch("sophon_live.coordinator.api.judge_tools", new=fake), calls


class NoDelegateNet(unittest.TestCase):
    def test_delegates_the_users_own_words_when_the_model_names_a_tool(self) -> None:
        for text, tools in (("你还记得我上次跟你说我喜欢吃什么吗", ["memory_note"]), ("明天上海会下雨吗", ["get_weather"]), ("关掉客厅的灯", ["device_control"])):
            s, calls = _session(text)
            p, asked = _model_says(tools)
            with p:
                _run(s)
            self.assertEqual(asked, [text])                         # the model is asked about the user's own words
            self.assertEqual(len(calls), 1, text)
            self.assertEqual(calls[0]["arguments"]["task"], text)
            self.assertIsNone(calls[0]["call_id"])
            self.assertEqual(calls[0]["net"], "no_delegate")

    def test_not_when_the_model_says_no_tool_or_is_unavailable(self) -> None:
        for tools in ([], None):
            s, calls = _session("你好呀")
            p, asked = _model_says(tools)
            with p:
                _run(s)
            self.assertEqual((asked, calls), (["你好呀"], []), tools)

    def test_structural_triggers_gate_the_model_call(self) -> None:
        """Each of these ends the net BEFORE any model call (no keyword look at the words, no JEV spend)."""
        def case(s):
            p, asked = _model_says(["get_weather"])
            with p:
                _run(s)
            return asked

        s, calls = _session("明天上海会下雨吗")
        s.deleg_calls = 1                                    # the front called delegate in that turn
        self.assertEqual((case(s), calls), ([], []))
        s, calls = _session("明天上海会下雨吗", front_said="明天上海有雨，带把伞吧，气温大概十二度左右呢。")   # the front answered by itself
        self.assertEqual((case(s), calls), ([], []))
        s, calls = _session(None)
        self.assertEqual((case(s), calls), ([], []))
        s, calls = _session("明天上海会下雨吗", age_s=60)    # a stale transcript from an earlier turn
        self.assertEqual((case(s), calls), ([], []))
        s, calls = _session("明天上海会下雨吗")
        s.last_deleg_asr_t = s.last_user[1]                   # this utterance was already delegated
        self.assertEqual((case(s), calls), ([], []))

    def test_a_failing_net_never_raises(self) -> None:
        s, _ = _session("明天上海会下雨吗")

        async def boom(_e):
            raise RuntimeError("x")

        s._delegate = boom
        p, _ = _model_says(["get_weather"])
        with p:
            _run(s)                                          # logged, not raised

    def test_the_gateway_has_no_keyword_test_of_the_user_utterance(self) -> None:
        import inspect
        src = inspect.getsource(gw.LiveSession._no_delegate_net)
        self.assertNotIn("required_tools", src)
        self.assertNotIn("re.", src)


class FrontWordedTaskTests(unittest.IsolatedAsyncioTestCase):
    """Live: delegate(task='我记得你说过你喜欢吃火锅') for the user's 「你还记得我上次跟你说我喜欢吃什么吗」 was routed FAST and went silent.
    rc2: the MODEL judges whether the front's wording is faithful (task_faithful); the code only enforces its answer."""

    async def _admit(self, task: str, raw: str, faithful: str | None, tool: str = "memory_note"):
        from sophon_live.coordinator import AdmitRequest, Ledger, admit
        from sophon_live.coordinator import api
        from tests.test_refine_guard import _jev

        asked: list = []
        inner = _jev("WORKER", None, tool=tool, faithful=faithful)

        async def spy(state, questions, **kw):
            asked.append((dict(state), sorted(questions)))
            return await inner(state, questions, **kw)

        with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api.jev_client, "system_one", new=spy):
            res = await admit(AdmitRequest(session_id="s", utterance_id="u", text=task, delegate_task=task, raw_text=raw), Ledger())
        return res, asked

    async def test_an_unfaithful_front_task_is_replaced_by_the_users_words(self) -> None:
        res, asked = await self._admit("我记得你说过你喜欢吃火锅", "你还记得我上次跟你说我喜欢吃什么吗", "DIFFERENT")
        self.assertEqual(res.job.objective, "你还记得我上次跟你说我喜欢吃什么吗")
        self.assertEqual(res.job.require_tools, ["memory_note"])
        self.assertEqual(asked[0][0]["user_said"], "你还记得我上次跟你说我喜欢吃什么吗")      # the model saw both wordings
        self.assertIn("task_faithful", asked[0][1])

    async def test_a_faithful_front_task_is_left_alone(self) -> None:
        res, _ = await self._admit("讲个笑话", "来个笑话吧", "SAME", tool="NONE")
        self.assertEqual(res.job.objective, "讲个笑话")

    async def test_no_faithfulness_question_when_the_wording_is_the_users_own(self) -> None:
        res, asked = await self._admit("明天上海会下雨吗", "明天上海会下雨吗", None, tool="get_weather")
        self.assertNotIn("task_faithful", asked[0][1])
        self.assertEqual(res.job.objective, "明天上海会下雨吗")


if __name__ == "__main__":
    unittest.main()
