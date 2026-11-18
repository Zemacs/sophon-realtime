"""Critic I-9: a measured misclassification gate for intent.required_tools' realtime-fact routing (web_search).

Two error kinds, with different tolerance:
  * a realtime question that is NOT routed to web_search can be answered from the model's head (a fabrication risk): 0 allowed.
  * a knowledge / chit-chat question routed to web_search costs a lookup (latency, outbound requests, a spoken lead-in): a small
    rate is tolerated, not a large one. Round 4b's first I-8 version over-triggered on 15/15 of Critic's probes (100%) and 21/44
    (48%) of the harder set written for I-9.
Numbers now: realtime 100/100 routed, free 102/102 clean (101 scored: 1 weather-keyword case excluded) (0.0% over-trigger). The gate allows <= 3%.
The set sizes are asserted too, so the gate cannot be passed by shrinking the sets.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import intent
from tests import intent_cases as c

# 「今天天气好适合打球」 is routed to get_weather by the long-standing weather keyword (not a realtime-fact decision, not web_search).
OUT_OF_SCOPE_FREE = {"今天天气好适合打球"}
MAX_OVER_TRIGGER_RATE = 0.03
POS = c.REALTIME + c.HELDOUT_REALTIME + c.CONVERSION_REALTIME
NEG = c.FREE + c.HELDOUT_FREE + c.CONVERSION_FREE


def _routes_to_search(text: str) -> bool:
    return "web_search" in intent.required_tools(text)


class IntentGate(unittest.TestCase):
    def test_label_sets_are_not_shrunk(self) -> None:
        self.assertGreaterEqual(len(POS), 100)
        self.assertGreaterEqual(len(NEG), 102)
        self.assertEqual((len(c.OFFICIAL_REALTIME), len(c.OFFICIAL_FREE), len(c.CRITIC_REALTIME), len(c.CRITIC_FREE)), (29, 19, 12, 15))
        everything = POS + NEG
        self.assertEqual(len(everything), len(set(everything)), "a question is labeled twice")

    def test_no_realtime_question_is_missed(self) -> None:
        pos = POS
        missed = [t for t in pos if intent.required_tools(t)[:1] != ["web_search"]]
        self.assertEqual(missed, [], f"{len(missed)}/{len(pos)} realtime questions not routed to web_search")

    def test_over_trigger_rate_stays_under_the_gate(self) -> None:
        neg = [t for t in NEG if t not in OUT_OF_SCOPE_FREE]
        over = [t for t in neg if _routes_to_search(t)]
        rate = len(over) / len(neg)
        self.assertLessEqual(rate, MAX_OVER_TRIGGER_RATE, f"{len(over)}/{len(neg)} = {rate:.1%} over-triggered: {over}")

    def test_critics_fifteen_probes_and_the_official_negatives_are_all_clean(self) -> None:
        for t in c.CRITIC_FREE + c.OFFICIAL_FREE:
            self.assertEqual(intent.required_tools(t), [], t)

    def test_a_bare_now_word_or_topic_word_is_not_a_signal(self) -> None:
        for t in ("现在", "今天", "现在怎么样", "今天是几号的生日", "地震", "世界杯", "NBA", "基金", "指数", "疫情", "航班", "限行"):
            self.assertFalse(_routes_to_search(t), t)        # (price nouns like 汇率 / 股价 / 票房 are signals by themselves, on purpose)

    def test_explanations_stay_free_but_a_now_word_restores_the_lookup(self) -> None:
        self.assertFalse(_routes_to_search("油价为什么会涨"))
        self.assertTrue(_routes_to_search("为什么今天油价涨了"))
        self.assertFalse(_routes_to_search("汇率是什么意思"))
        self.assertTrue(_routes_to_search("今天汇率是多少"))

    def test_conversion_words_exclude_unit_conversion_but_not_currency(self) -> None:
        """Critic I-9 acceptance: 「换算」 must not turn unit conversion into a lookup; currency conversion still is one."""
        for t in c.CONVERSION_FREE:
            self.assertEqual(intent.required_tools(t), [], t)
        for t in c.CONVERSION_REALTIME:
            self.assertEqual(intent.required_tools(t)[:1], ["web_search"], t)

    def test_acceptance_critics_15_free_and_12_realtime(self) -> None:
        """The two numbers the acceptance is stated in: the 15 over-triggered probes no longer go online, the 12 extra realtime phrasings 0 misses."""
        self.assertEqual((len(c.CRITIC_FREE), len(c.CRITIC_REALTIME)), (15, 12))
        self.assertEqual([t for t in c.CRITIC_FREE if intent.required_tools(t)], [])
        self.assertEqual([t for t in c.CRITIC_REALTIME if intent.required_tools(t)[:1] != ["web_search"]], [])
        for t in ("地震", "地震是怎么形成的"):
            self.assertEqual(intent.required_tools(t), [], t)                              # topic alone: no
        self.assertEqual(intent.required_tools("刚才地震了吗")[:1], ["web_search"])          # happening now: yes
        self.assertEqual(intent.required_tools("今天是谁的生日"), [])

    def test_explicit_search_requests_still_search(self) -> None:
        for t in ("帮我搜索一下地震是怎么形成的", "搜一下NBA是什么意思", "google 一下光速"):
            self.assertIn("web_search", intent.required_tools(t), t)


if __name__ == "__main__":
    unittest.main()
