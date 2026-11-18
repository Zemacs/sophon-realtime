"""体验版 v1 / 3: no stray 。 at the start of a reply (root cause: a closing quote after the last sentence mark)."""
from __future__ import annotations

import asyncio
import collections
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.gateway import lead_for
from tests.test_r0_coord_boundary import _bare_session


class LeadForTests(unittest.TestCase):
    def test_finished_sentences_get_no_lead(self) -> None:
        for prev in ("等于 Dec 25 呀！", "现在是21点21分。", "真的吗？", "ok.", "好，", "走吧…"):
            self.assertEqual(lead_for(prev, "zh"), "", prev)
            self.assertEqual(lead_for(prev, "en"), "", prev)

    def test_closing_quote_or_bracket_after_the_sentence_mark_is_still_finished(self) -> None:
        # these are the real endings that leaked a 。 into the next reply (mt_r3_3 / r4_1 / r4_3 / r5_2 / r5_3)
        for prev in ("胖呗，还系什么腰带啊！\"", "胖呗，还系什么腰带啊！”", "熬到它自己消失。\"", "他说“好。”", "（好的。）", "「行！」 "):
            self.assertEqual(lead_for(prev, "zh"), "", prev)

    def test_nothing_or_only_closers_is_not_dangling(self) -> None:
        for prev in ("", "   ", "”", "\""):
            self.assertEqual(lead_for(prev, "zh"), "", repr(prev))

    def test_a_cut_off_sentence_still_gets_a_separator(self) -> None:
        self.assertEqual(lead_for("我想想", "zh"), "。")
        self.assertEqual(lead_for("他说：“好", "zh"), "。")
        self.assertEqual(lead_for("I think", "en"), ". ")


class _Engine:
    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()


class NextReplyTextTests(unittest.IsolatedAsyncioTestCase):
    async def _replies(self, first: str, second: str) -> dict:
        s = _bare_session()
        s.engine = _Engine()
        s.closed = False
        s.rate = 24000
        s.cur_rid = None
        s.resp = {}
        s.play = collections.deque()
        s.last_text = ""
        s._cut_user = lambda: None
        s._delivery = lambda: type("D", (), {"attach": lambda self, rid: None})()
        s._trace = lambda: type("T", (), {"response_started": lambda self, rid: None})()
        for rid, text in (("r1", first), ("r2", second)):
            s.engine.q.put_nowait({"t": "text", "rid": rid, "delta": text})
            s.engine.q.put_nowait({"t": "done", "rid": rid})
        task = asyncio.ensure_future(s._engine_events())
        for _ in range(50):
            await asyncio.sleep(0)
            if s.engine.q.empty():
                break
        await asyncio.sleep(0)
        s.closed = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return s.resp

    async def test_reply_after_a_quoted_joke_has_no_leading_period(self) -> None:
        resp = await self._replies("我帮你查一下。有一天，0 遇到 8，说：\"胖就胖呗，还系什么腰带啊！\"", "我帮你查一下。现在是21点21分。")
        self.assertEqual(resp["r2"]["text"], "我帮你查一下。现在是21点21分。")

    async def test_reply_after_a_cut_off_one_keeps_its_separator(self) -> None:
        resp = await self._replies("我帮你查一", "现在是21点21分。")
        self.assertEqual(resp["r2"]["text"], "。现在是21点21分。")

    async def test_every_ending_seen_in_the_logs(self) -> None:
        for first in ("等于 Dec 25 呀！", "腰带啊！”", "消失。\""):
            resp = await self._replies(first, "好的。")
            self.assertEqual(resp["r2"]["text"], "好的。", first)


if __name__ == "__main__":
    unittest.main()
