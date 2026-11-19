"""Round B / Critic G-1: the grounding LITERAL path is an exact comparison and nothing else.

Before: ``word_numbers`` (four thousand -> 4000), ``spoken_forms`` (currency words -> codes, months / ordinals -> MM-DD, "twenty twenty six" -> 2026, a spoken
e-mail rebuilt from "at" / "dot", seats, grouped number words), ``joins_n`` (their output as extra literal candidates), the lenient literal rules (a date whose
year was never said, an e-mail whose domain was only said up to its first label, a multi-word value whose parts occur anywhere) and the short-value
exemption. Each of those is a judgment about meaning made by a table. Now an equivalence beyond an exact occurrence is the model's claim (``grounding_evidence``)
plus an independent model's yes (``judge``); a no, a timeout, an exception or no judge = not grounded (fail closed).
The rows below are Critic's probes from the b11afd4 review (probe_literal.out) plus the removed-rule boundaries.
"""
from __future__ import annotations

import ast
import asyncio
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding  # noqa: E402

K = grounding.EVIDENCE_KEY


def ev(argument: str, span: str) -> list:
    return [{"argument": argument, "span": span, "value_comes_from_span": True}]


def literal_only(args: dict, heard: list[str], seen: list[str] | None = None) -> list:
    """No evidence, no judge: exactly the literal path (a refused value is returned)."""
    return grounding.check("t", args, heard, seen or [])


# (value, what the caller said) the OLD literal path let through and the new one must not
REMOVED_FREE_PASSES = [
    ({"amount": "4000"}, "four thousand"),                                   # word_numbers
    ({"amount": "250"}, "two hundred fifty dollars"),
    ({"amount": "4000"}, "four thousand dollars"),
    ({"currency": "krw"}, "we won the game"),                                # _CURRENCY: 'won' -> krw
    ({"currency": "brl"}, "for real this time"),                             # _CURRENCY: 'real' -> brl
    ({"currency": "usd"}, "twenty bucks"),
    ({"date": "12-14"}, "december 14th"),                                    # month / ordinal tables
    ({"date": "1999-12-14"}, "december 14th"),                               # the year was never said ("year usually not said")
    ({"date": "1999-12-14"}, "see you on 12/14"),                            # lenient date rule: MMDD said, year never
    ({"date": "2026-12-14"}, "december fourteenth twenty twenty six"),
    ({"date": "2026-12-14"}, "the fourteenth of december"),
    ({"seat": "14c"}, "fourteen C please"),                                  # seats
    ({"email": "jon@acme.com"}, "jon at acme dot com"),                      # spoken e-mail rebuilt
    ({"email": "john.smith31@gmail.com"}, "john dot smith thirty one at gmail dot com"),
    ({"email": "john.smith@acme.org"}, "my address is john.smith at acme"),  # domain only matched up to its first label
    ({"zip": "19122"}, "one nine one two two"),                              # NUM_WORDS / grouped number words
    ({"zip": "816"}, "eight sixteen"),
    ({"name": "yusuf"}, "Y, U, S, U, F"),                                    # spelled_joins used as a literal candidate
    ({"name": "Li 88"}, "my name is Li and the room is 5 then later 88"),     # multi-word: parts anywhere
    ({"name": "Wang Li Si"}, "Wang called, Li answered, Si hung up"),
]


class NoSpokenFormIsLiteralProvenance(unittest.TestCase):
    def test_every_form_the_old_tables_converted_is_refused_on_the_literal_path(self) -> None:
        for args, said in REMOVED_FREE_PASSES:
            self.assertTrue(literal_only(args, [said]), (args, said))

    def test_the_same_values_are_released_by_evidence_plus_an_independent_yes_and_by_nothing_less(self) -> None:
        for args, said in REMOVED_FREE_PASSES:
            (k, _), = args.items()
            e = ev(k, said)
            self.assertEqual(grounding.check("t", args, [said], [], evidence=e, judge=lambda *a: True), [], (args, said))
            self.assertTrue(grounding.check("t", args, [said], [], evidence=e, judge=lambda *a: False), ("no", args, said))
            self.assertTrue(grounding.check("t", args, [said], [], evidence=e, judge=lambda *a: None), ("unavailable", args, said))
            self.assertTrue(grounding.check("t", args, [said], [], evidence=e), ("no judge", args, said))

    def test_the_judge_is_asked_the_equivalence_question_with_the_span_and_the_value(self) -> None:
        calls: list = []

        def judge(kind, arg, value, span, ctx):
            calls.append((kind, arg, value, span, ctx))
            return True
        grounding.check("t", {"amount": "4000"}, ["four thousand"], [], evidence=ev("amount", "four thousand"), judge=judge)
        self.assertEqual(calls, [(grounding.JUDGE_EQUIVALENT, "amount", "4000", "four thousand", "")])

    def test_a_claim_that_needs_the_judge_is_collected_not_decided(self) -> None:
        pending: list = []
        bad = grounding.check("t", {"amount": "4000"}, ["four thousand"], [], evidence=ev("amount", "four thousand"), pending=pending)
        self.assertTrue(bad)
        self.assertEqual([(p["kind"], p["value"], p["span"]) for p in pending], [(grounding.JUDGE_EQUIVALENT, "4000", "four thousand")])


class FailClosedThroughTheAsyncDriver(unittest.IsolatedAsyncioTestCase):
    ARGS, HEARD = {"amount": "4000"}, ["four thousand"]

    async def _run(self, judge):
        return await grounding.check_async("t", self.ARGS, self.HEARD, [], evidence=ev("amount", "four thousand"), judge=judge)

    async def test_yes_releases_it(self) -> None:
        async def yes(*a):
            return True
        self.assertEqual(await self._run(yes), [])

    async def test_no_a_non_answer_an_exception_and_a_timeout_all_refuse(self) -> None:
        async def no(*a):
            return False

        async def none(*a):
            return None

        async def boom(*a):
            raise RuntimeError("router down")

        async def timeout(*a):
            raise asyncio.TimeoutError()

        async def junk(*a):
            return "yes"                                   # not a bool: never read as a yes
        for j in (no, none, boom, timeout, junk):
            bad = await self._run(j)
            self.assertEqual([b[0] for b in bad], ["amount"], j.__name__)
            self.assertIn("not said by the caller", bad[0][2], j.__name__)

    async def test_no_evidence_means_the_judge_is_not_even_asked(self) -> None:
        asked: list = []

        async def judge(*a):
            asked.append(a)
            return True
        bad = await grounding.check_async("t", self.ARGS, self.HEARD, [], judge=judge)
        self.assertTrue(bad)
        self.assertEqual(asked, [])

    def test_the_refusal_the_model_reads_is_the_fixed_sentence_naming_the_evidence_path(self) -> None:
        bad = grounding.check("t", self.ARGS, self.HEARD, [])
        msg = grounding.guard_message(bad, self.HEARD)
        self.assertTrue(msg.startswith("NOT EXECUTED. These argument values were not provided by the caller"))
        self.assertIn(K, msg)
        self.assertIn("a second model checks that the span says the value", msg)


class NoShortValueExemption(unittest.TestCase):
    def test_short_values_need_provenance_in_every_script(self) -> None:
        for v in ("TX", "京", "5", "b", "7号", "OK"):
            self.assertTrue(literal_only({"x": v}, ["hello"]), v)

    def test_and_pass_when_they_occur_exactly(self) -> None:
        for v, said in (("TX", "I live in TX"), ("京", "我在京"), ("5", "room 5"), ("OK", "say OK")):
            self.assertEqual(literal_only({"x": v}, [said]), [], v)

    def test_a_short_latin_value_is_a_whole_word_only(self) -> None:
        self.assertTrue(literal_only({"state": "TX"}, ["text me"]))        # a piece of a longer word
        self.assertTrue(literal_only({"n": "5"}, ["room 15"]))

    def test_only_punctuation_is_skipped_because_nothing_is_left_to_compare(self) -> None:
        self.assertEqual(literal_only({"x": "--"}, ["hello"]), [])


class WhatStaysLiteral(unittest.TestCase):
    def test_format_only_normalisation_is_still_exact_provenance(self) -> None:
        self.assertEqual(literal_only({"n": "88"}, ["号码是８８"]), [])                  # full-width
        self.assertEqual(literal_only({"n": "Smith"}, ["it is smith"]), [])               # case
        self.assertEqual(literal_only({"n": "1000"}, ["it costs 1,000"]), [])             # punctuation
        self.assertEqual(literal_only({"e": "jon.smith@acme.com"}, ["it is jon.smith@acme.com"]), [])
        self.assertEqual(literal_only({"z": "90210"}, ["look"], ["{'zip': '90210'}"]), [])  # a tool result

    def test_spaces_are_format_so_digits_said_one_by_one_as_digits_are_the_same_text(self) -> None:
        """``norm_text`` drops whitespace and punctuation (Critic: format only, acceptable): '1 9 1 2 2' is the text 19122. Number WORDS are not."""
        self.assertEqual(literal_only({"code": "19122"}, ["1 9 1 2 2"]), [])
        self.assertTrue(literal_only({"code": "19122"}, ["one nine one two two"]))

    def test_whole_word_and_digit_boundaries_stay(self) -> None:
        for v, said in (("88", "1988"), ("88", "8800"), ("Daniel", "Danielle"), ("min@acme.com", "admin@acme.com"), ("john.smith@acme.com", "john.smyth@acme.com")):
            self.assertTrue(literal_only({"v": v}, [said]), (v, said))

    def test_no_cross_turn_assembly(self) -> None:
        self.assertTrue(literal_only({"n": "88"}, ["8", "8"]))

    def test_the_other_numeral_script_is_still_not_literal(self) -> None:
        self.assertTrue(literal_only({"n": "88"}, ["八十八"]))
        self.assertTrue(literal_only({"n": "八十八"}, ["88"]))


class TheRemovedMachineryStaysRemoved(unittest.TestCase):
    TREE = ast.parse((Path(__file__).resolve().parents[1] / "sophon_live" / "grounding.py").read_text(encoding="utf-8"))

    def test_the_converters_and_tables_are_gone(self) -> None:
        for n in ("word_numbers", "spoken_forms", "NUM_WORDS", "_UNITS", "_TENS", "_SCALE", "_MONTHS", "_ORD", "_CURRENCY", "_CJK", "_norm"):
            self.assertFalse(hasattr(grounding, n), n)

    def test_check_has_no_joins_and_no_length_or_script_exemption(self) -> None:
        fn = next(n for n in ast.walk(self.TREE) if isinstance(n, ast.FunctionDef) and n.name == "check")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(fn) if isinstance(n, ast.arg)}
        self.assertFalse(names & {"joins", "joins_n", "words", "spelled_joins", "word_numbers", "spoken_forms"}, names)
        for node in ast.walk(fn):                                                   # no ``len(vn) < N`` style exemption
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call) and getattr(node.left.func, "id", "") == "len":
                self.assertFalse(any(isinstance(c, ast.Constant) and isinstance(c.value, int) and c.value <= 3 for c in node.comparators), ast.unparse(node))

    def test_spelled_joins_is_display_only(self) -> None:
        """It is used by the hint texts and is never a provenance candidate: ``check`` does not call it."""
        fn = next(n for n in ast.walk(self.TREE) if isinstance(n, ast.FunctionDef) and n.name == "check")
        calls = {getattr(c.func, "id", "") for c in ast.walk(fn) if isinstance(c, ast.Call)}
        self.assertNotIn("spelled_joins", calls)
        self.assertEqual(grounding.spelled_joins("Y, U, S, U, F"), ["yusuf"])
        self.assertEqual(grounding.spelled_joins("four eight two"), [])          # number WORDS are not read by code


if __name__ == "__main__":
    unittest.main()
