"""The turn readout only reports (FrontPort F2); it no longer starts a reply.

Run from the repository root:
    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tag/tests -p no:cacheprovider
(also runs with python3 -m unittest discover -s tag/tests).
No GPU, no vllm_omni: see fork_harness.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fork_harness import Call, Readouts, load_duplex

dx = load_duplex()
F2_FIELDS = {"endpoint_id", "turn_action", "turn_check_ms"}


def f2(ev):
    return {k: v for k, v in ev.items() if k != "type"}


class Defaults(unittest.TestCase):
    """The take line is the existing checkpoint table (duplex.py:36-37, :43)."""

    def test_checkpoint_table_is_the_existing_default(self):
        self.assertEqual(dx.CHECKS, [0.2, 0.35, 0.5, 0.8, 1.3, 2.0])
        self.assertEqual(dx.THRS, [0.95, 0.85, 0.7, 0.5, 0.35, 0.25])
        self.assertEqual(dx.FALLBACK_S, 2.0)
        self.assertEqual(dx.PERIODIC_S, 0.0)
        self.assertEqual(dx.PERIODIC_THR, 0.9)


class F2Event(unittest.TestCase):
    def test_fields_are_only_the_three_the_fork_owns(self):
        ev = dx.f2_event("fork-ep-1", "take", 81.6)
        self.assertEqual(f2(ev).keys(), F2_FIELDS)
        self.assertEqual(ev["turn_check_ms"], 82)
        self.assertIsInstance(ev["turn_check_ms"], int)

    def test_action_is_a_closed_two_way_choice(self):
        with self.assertRaises(ValueError):
            dx.f2_event("fork-ep-1", "reject", 10)

    def test_negative_duration_is_clamped(self):
        self.assertEqual(dx.f2_event("e", "no_take", -3)["turn_check_ms"], 0)


class Take(unittest.TestCase):
    def test_readout_over_threshold_reports_take_and_does_not_start_a_reply(self):
        c = Call(dx, Readouts((0.99, 0.0)))
        c.speech(500)
        c.silence(200)
        self.assertFalse(c.buf.pending_reply, "the readout itself must not arm a reply")
        c.silence(100)
        self.assertEqual(len(c.events), 1)
        ev = c.events[0]
        self.assertEqual(ev["turn_action"], "take")
        self.assertEqual(f2(ev).keys(), F2_FIELDS)
        self.assertGreaterEqual(ev["turn_check_ms"], 10)
        self.assertFalse(c.buf.pending_reply)
        c.silence(2500)
        self.assertEqual(c.reply_appends(), [])
        self.assertEqual(len(c.events), 1, "one F2 per endpoint")
        self.assertEqual(
            c.dx.turn_prob_http.calls, 1, "no more readouts once the endpoint is taken"
        )

    def test_threshold_is_per_checkpoint(self):
        c = Call(dx, Readouts((0.9, 0.0), (0.9, 0.0)))
        c.speech(500)
        c.silence(200)
        self.assertEqual(c.events, [], "0.9 is below the first checkpoint's 0.95")
        c.silence(200)
        self.assertEqual(
            [e["turn_action"] for e in c.events],
            ["take"],
            "0.9 is above the second checkpoint's 0.85",
        )


class NoTake(unittest.TestCase):
    def test_no_take_is_reported_at_the_fallback_silence_and_never_starts_a_reply(
        self,
    ):
        c = Call(dx, Readouts((0.1, 0.0)))
        c.speech(500)
        seen = []
        for _ in range(25):
            c.feed("silence")
            seen.append(len(c.events))
        self.assertEqual(seen[:19], [0] * 19, "nothing before the fallback silence")
        self.assertEqual(seen[19:], [1] * 6, "one no_take from FALLBACK_S (2.0 s) on")
        self.assertEqual(c.events[0]["turn_action"], "no_take")
        self.assertEqual(f2(c.events[0]).keys(), F2_FIELDS)
        self.assertFalse(c.buf.pending_reply)
        self.assertEqual(c.reply_appends(), [])

    def test_user_resuming_closes_the_endpoint_with_no_take(self):
        c = Call(dx, Readouts((0.1, 0.0)))
        c.speech(500)
        c.silence(300)
        self.assertEqual(c.events, [])
        c.speech(200)
        self.assertEqual([e["turn_action"] for e in c.events], ["no_take"])

    def test_no_completed_readout_means_no_event(self):
        c = Call(dx, Readouts(None))
        c.speech(500)
        c.silence(300)
        c.speech(200)
        self.assertEqual(c.events, [])

    def test_each_endpoint_gets_its_own_id(self):
        c = Call(dx, Readouts((0.99, 0.0)))
        c.speech(500)
        c.silence(300)
        c.speech(500)
        c.silence(300)
        self.assertEqual([e["turn_action"] for e in c.events], ["take", "take"])
        self.assertEqual(len({e["endpoint_id"] for e in c.events}), 2)


class AcousticReject(unittest.TestCase):
    def test_short_non_turn_sound_is_no_take(self):
        old = dx.IGNORE_SHORT
        dx.IGNORE_SHORT = True
        try:
            c = Call(dx, Readouts((0.1, 0.9)))
            c.speech(300)
            c.silence(2200)
        finally:
            dx.IGNORE_SHORT = old
        self.assertEqual([e["turn_action"] for e in c.events], ["no_take"])
        self.assertEqual(f2(c.events[0]).keys(), F2_FIELDS)
        self.assertEqual(c.reply_appends(), [])


if __name__ == "__main__":
    unittest.main()
