"""体验版 v1 / 6: TRACE vad_end (user stopped talking) and pad_start (first sound heard after that)."""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator.trace import TraceBook
from tests.test_r0_coord_boundary import _bare_session
from tests.test_trace import Clock


class TraceBookMarkerTests(unittest.TestCase):
    def test_vad_end_and_pad_start_are_negative_offsets_from_the_delegation(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_vad_end(clk.t)                       # user stops
        clk.advance(900); book.note_voice_start(clk.t)  # lead-in becomes audible 0.9 s later
        clk.advance(600)                                # the front calls delegate 0.6 s after that
        tr = book.start("d1")
        self.assertEqual(tr.stages["vad_end"], -1500)
        self.assertEqual(tr.stages["pad_start"], -600)
        self.assertEqual(tr.stages["deleg"], 0)
        self.assertEqual(tr.as_dict()["stages"]["pad_start"] - tr.as_dict()["stages"]["vad_end"], 900)   # time to first sound

    def test_no_lead_in_means_no_pad_start(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_vad_end(clk.t); clk.advance(700)
        tr = book.start("d1")
        self.assertEqual(tr.stages, {"deleg": 0, "vad_end": -700})

    def test_previous_turns_tail_is_not_a_pad(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_voice_start(clk.t)                   # an old response started
        clk.advance(4000); book.note_vad_end(clk.t)    # the user spoke after it
        clk.advance(500)
        tr = book.start("d1")
        self.assertIn("vad_end", tr.stages)
        self.assertNotIn("pad_start", tr.stages)

    def test_markers_are_claimed_by_one_turn_only(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_vad_end(clk.t); clk.advance(300); book.note_voice_start(clk.t); clk.advance(100)
        book.start("d1")
        clk.advance(100)
        t2 = book.start("d2")
        self.assertEqual(t2.stages, {"deleg": 0})

    def test_stale_vad_end_is_ignored(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_vad_end(clk.t); clk.advance(120_000)
        self.assertEqual(book.start("d1").stages, {"deleg": 0})

    def test_summary_json_carries_them(self) -> None:
        clk = Clock()
        book = TraceBook("s1", clock=clk)
        book.note_vad_end(clk.t); clk.advance(250)
        d = json.loads(json.dumps(book.start("d1").as_dict()))
        self.assertEqual(d["stages"]["vad_end"], -250)


class GatewayHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_cut_user_notes_the_last_voiced_frame(self) -> None:
        s = _bare_session()
        s.timeline_ms = 10_000
        s.user_t0 = 8_000
        voiced = (np.sin(np.arange(16000) / 5) * 0.3).astype(np.float32)        # 1 s of "speech": 8000..9000
        s.user_audio = [voiced, np.zeros(16000, np.float32)]                    # then 1 s of silence: 9000..10000
        s.asr_task = None
        s.asr = object()

        async def no_asr(a, start, end) -> None:                                # the transcription itself is not under test
            return None

        s._asr = no_asr
        await s._cut_user()
        book = s._trace()
        before = time.monotonic()
        tr = book.start("d1")
        self.assertIn("vad_end", tr.stages)
        # speech ended ~1.0 s (timeline) before "now", plus the 3.2 ms of test runtime
        self.assertTrue(-1100 <= tr.stages["vad_end"] <= -950, tr.stages)
        self.assertLess(time.monotonic() - before, 1)


if __name__ == "__main__":
    unittest.main()
