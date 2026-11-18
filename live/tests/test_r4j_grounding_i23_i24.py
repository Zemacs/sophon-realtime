"""Round 4j / I-23 + I-24.

I-23: every argument value goes through provenance, whatever its script or length. Chinese values (which ``_norm`` used to erase) and values of
more than 3 words (which used to be skipped) are checked against what the caller said / a tool returned. Positive and negative cases for both.
I-24: the realtime guard never releases an unprovenanced call: from the 3rd identical rejection on, the answer is a structured needs_input.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding, realtime  # noqa: E402


def bad(args: dict, heard: list[str], seen: list[str] | None = None) -> list:
    return grounding.check("t", args, heard, seen or [])


class ChineseValuesAreChecked(unittest.TestCase):
    HEARD = ["我叫李明，住在北京市海淀区中关村大街1号，邮箱是 bob at gmail dot com"]

    def test_a_chinese_value_the_caller_said_passes(self) -> None:
        self.assertEqual(bad({"name": "李明"}, self.HEARD), [])
        self.assertEqual(bad({"address": "北京市海淀区中关村大街1号"}, self.HEARD), [])
        self.assertEqual(bad({"address": "海淀区"}, self.HEARD), [])

    def test_an_invented_chinese_value_is_blocked(self) -> None:
        for args in ({"name": "张伟"}, {"address": "上海市浦东新区世纪大道1号"}, {"address": "上海市 浦东新区 世纪大道 1号"}, {"city": "广州"}):
            got = bad(args, self.HEARD)
            self.assertTrue(got, args)
            self.assertTrue(all("not said by the caller" in why for _, _, why in got))

    def test_a_chinese_value_from_a_tool_result_passes(self) -> None:
        self.assertEqual(bad({"name": "张伟"}, ["look it up"], ["{'customer': '张伟'}"]), [])
        self.assertTrue(bad({"name": "张伟"}, ["look it up"], ["{'customer': '李伟'}"]))

    def test_a_value_in_a_different_numeral_script_is_not_literal_provenance(self) -> None:
        """Round 4k+ / N-1 replaces "一号 == 1号 on the literal path": converting Chinese numerals is a judgment about what a character means here (四川 -> 4), so the
        literal path compares text as written. The same number in the other script goes through the model's evidence + the independent check."""
        heard = ["我住在上海市浦东新区世纪大道一号"]
        self.assertEqual(bad({"address": "上海市浦东新区世纪大道一号"}, heard), [])
        self.assertTrue(bad({"address": "上海市浦东新区世纪大道1号"}, heard))
        ev = [{"argument": "address", "span": "上海市浦东新区世纪大道一号", "value_comes_from_span": True}]
        self.assertEqual(grounding.check("t", {"address": "上海市浦东新区世纪大道1号"}, heard, [], evidence=ev, judge=lambda *a: True), [])
        self.assertTrue(bad({"address": "北京市朝阳区建国路八十八号"}, heard))

    def test_mixed_chinese_and_latin_checks_both_halves(self) -> None:
        self.assertEqual(bad({"name": "李明 Li"}, ["我叫李明 Li"]), [])
        self.assertTrue(bad({"name": "李明 Zhang"}, ["我叫李明 Li"]))        # the Latin half is invented
        self.assertTrue(bad({"name": "张伟 Li"}, ["我叫李明 Li"]))            # the Chinese half is invented

    def test_a_single_character_has_no_provenance_signal_and_is_not_checked(self) -> None:
        self.assertEqual(bad({"x": "京"}, ["hello"]), [])


class LongValuesAreChecked(unittest.TestCase):
    HEARD = ["my email is bob at gmail dot com and I live at 742 Evergreen Terrace Springfield"]

    def test_a_long_value_made_of_what_the_caller_said_passes(self) -> None:
        self.assertEqual(bad({"address": "742 Evergreen Terrace Springfield"}, self.HEARD), [])
        # Round 4k: the 50% token-overlap rule is gone. A composed value (words the caller never said) needs the model's evidence span instead.
        self.assertTrue(bad({"note": "bob lives at Evergreen Terrace in Springfield"}, self.HEARD))
        ev = [{"argument": "note", "span": "I live at 742 Evergreen Terrace Springfield", "value_comes_from_span": True}]
        args = {"note": "bob lives at Evergreen Terrace in Springfield"}
        self.assertTrue(grounding.check("t", args, self.HEARD, [], evidence=ev))                    # N-2: no independent check, no release
        self.assertEqual(grounding.check("t", args, self.HEARD, [], evidence=ev, judge=lambda *a: True), [])

    def test_an_invented_long_value_is_blocked(self) -> None:
        for args in ({"address": "55 Maple Avenue Shelbyville Ohio"}, {"name": "Dr Alice Marie Carter Jones"}, {"address": "9 Quiet Lane Portland Oregon"}):
            self.assertTrue(bad(args, self.HEARD), args)

    def test_a_number_in_a_long_value_must_itself_be_grounded(self) -> None:
        self.assertTrue(bad({"address": "999 Evergreen Terrace Springfield Illinois"}, self.HEARD))

    def test_digits_read_out_one_by_one_are_one_value(self) -> None:
        self.assertEqual(bad({"phone": "4 8 2 9 1 3 7"}, ["it is four eight two nine one three seven"]), [])
        self.assertTrue(bad({"phone": "4 8 2 9 1 3 7"}, ["my number is 5 5 5 1 2 3 4"]))

    def test_a_summary_reusing_the_callers_words_passes_and_an_invented_one_does_not(self) -> None:
        said = ["I want a refund for the blender I bought last week it is broken"]
        self.assertEqual(bad({"summary": "refund for the broken blender bought last week"}, said), [])
        self.assertTrue(bad({"summary": "wants to upgrade the premium subscription tier immediately"}, said))

    def test_short_latin_values_are_unchanged(self) -> None:
        self.assertTrue(bad({"order_id": "A98231"}, ["hello"]))
        self.assertEqual(bad({"order_id": "A98231"}, ["order A98231 please"]), [])
        self.assertEqual(bad({"state": "TX"}, ["hello"]), [])             # under 3 characters: nothing to look up


# ------------------------------------------------------------------ I-24
def _session(heard: list[str]) -> realtime.RealtimeSession:
    s = realtime.RealtimeSession.__new__(realtime.RealtimeSession)
    s.id = "t"; s.heard = list(heard); s.seen = []; s.asr = object(); s.asr_tasks = []; s.guard_counts = {}
    s.guard_total = 0; s.audit = []; s.spoken = []; s.degraded_warned = False
    s.cfg = {"instructions": "", "tools": [{"name": "lookup_account"}]}
    s.engine = mock.MagicMock(); s.engine.tool_result = mock.AsyncMock()
    return s


class GuardNeverReleasesAnUnprovenancedCall(unittest.IsolatedAsyncioTestCase):
    CALL = {"name": "lookup_account", "call_id": "c", "arguments": {"email": "mary.jones@gmail.com", "zip": "73301"}}

    async def test_the_same_call_is_never_let_through_however_often_it_is_repeated(self) -> None:
        s = _session(["hi, I need help with my bill"])
        for i in range(1, 30):
            self.assertTrue(await s._guard(dict(self.CALL, call_id=f"c{i}")), i)          # always rejected: nothing is ever released to the tool

    async def test_first_rejections_use_the_normal_message_then_needs_input(self) -> None:
        s = _session(["hi, I need help with my bill"])
        outs = []
        for i in range(1, 6):
            await s._guard(dict(self.CALL, call_id=f"c{i}"))
            outs.append(s.engine.tool_result.await_args.args[1])
        self.assertEqual(realtime.GUARD_MAX_REJECTIONS, 2)
        for o in outs[:2]:
            self.assertTrue(o.startswith("NOT EXECUTED. These argument values"), o)
            self.assertFalse(o.lstrip().startswith("{"))
        for o in outs[2:]:
            body = json.loads(o)
            self.assertEqual((body["ok"], body["status"]), (False, "needs_input"))
            self.assertIn("NOT EXECUTED and will not be", body["message"])
            self.assertIn("mary.jones@gmail.com", body["message"])

    async def test_needs_input_is_a_structured_failure_for_the_failure_reader(self) -> None:
        s = _session(["hi"])
        for i in range(3):
            await s._guard(dict(self.CALL, call_id=f"c{i}"))
        self.assertTrue(grounding.output_failed({"output": s.engine.tool_result.await_args.args[1]}))

    async def test_the_count_key_is_the_tool_and_its_exact_arguments(self) -> None:
        s = _session(["hi"])
        for i in range(2):
            await s._guard(dict(self.CALL, call_id=f"a{i}"))
        other = dict(self.CALL, arguments={"email": "mary.jones@gmail.com", "zip": "73302"})           # a different argument: its own count
        await s._guard(other)
        self.assertTrue(s.engine.tool_result.await_args.args[1].startswith("NOT EXECUTED. These"))      # still the first-level message
        self.assertEqual(len(s.guard_counts), 2)
        await s._guard(dict(self.CALL, call_id="a3"))
        self.assertEqual(json.loads(s.engine.tool_result.await_args.args[1])["status"], "needs_input")

    async def test_a_grounded_call_is_still_released(self) -> None:
        s = _session(["my email is mary.jones@gmail.com and zip 73301"])
        for i in range(6):
            self.assertFalse(await s._guard(dict(self.CALL, call_id=f"g{i}")))
        s.engine.tool_result.assert_not_awaited()

    async def test_the_threshold_is_exactly_two(self) -> None:
        s = _session(["hi"])
        kinds = []
        for i in range(4):
            await s._guard(dict(self.CALL, call_id=f"k{i}"))
            kinds.append("needs_input" if s.engine.tool_result.await_args.args[1].lstrip().startswith("{") else "normal")
        self.assertEqual(kinds, ["normal", "normal", "needs_input", "needs_input"])


if __name__ == "__main__":
    unittest.main()
