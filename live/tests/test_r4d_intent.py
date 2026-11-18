"""Round 4d (Critic incr-5 I-11 / I-10 residual): the fallback no longer swallows casual / task sentences; colloquial and English phrasing."""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from sophon_live.coordinator import intent
from tests import intent_dev2 as dev2
from tests import intent_probe5 as probe
from tests import intent_stress as stress

_T = Path(__file__).resolve().parent


def _search(t: str) -> bool:
    return "web_search" in intent.required_tools(t)


class FallbackStress(unittest.TestCase):
    def test_critic_stress_over_trigger_rate_is_at_most_10_percent(self) -> None:
        over = [t for t in stress.STRESS_FREE if _search(t)]
        self.assertEqual(len(stress.STRESS_FREE), 40)
        self.assertLessEqual(len(over), 4, over)            # target <= 10%; measured 0/40 (f7f2c58: 29/40)

    def test_categories_not_keywords(self) -> None:
        """Fresh sentences of the same CATEGORIES (personal arrangement / task / status of me / schedule), not in any set."""
        for t in ("今天可以收工了吗", "现在要不要出门", "你们现在有几个人在线", "明天需要带证件吗", "最近还顺利吗", "这次的方案通过了吗", "今天几楼开会", "现在还来得及吗",
                  "我今天走了多少路", "这周要写几份总结", "你现在在干嘛呢", "现在我们几点集合"):
            self.assertFalse(_search(t), t)


class ArithmeticNeverSearches(unittest.TestCase):
    def test_currency_arithmetic_and_loan_calculation(self) -> None:
        for t in ("二百美元加三百美元是多少", "五十欧元乘以三是多少", "1000日元减去200日元等于多少", "800元除以4是多少", "10 dollars plus 20 dollars is how much"):
            self.assertFalse(_search(t), t)

    def test_a_real_conversion_still_searches(self) -> None:
        self.assertTrue(_search("二百美元现在能换多少人民币"))


class ColloquialAndEnglish(unittest.TestCase):
    def test_dev2(self) -> None:
        self.assertEqual([t for t in dev2.DEV2_REALTIME if not _search(t)], [])
        self.assertEqual([t for t in dev2.DEV2_FREE if _search(t)], [])
        self.assertEqual([t for t in dev2.DEV2_EN_REALTIME if not _search(t)], [])
        self.assertEqual([t for t in dev2.DEV2_EN_FREE if _search(t)], [])

    def test_probe_set_is_pinned_and_clean(self) -> None:
        # run once (4/30 misses, 1/30 over-triggers) before its misses were folded back into the rules; now a regression set, not blind
        self.assertEqual(hashlib.sha256((_T / "intent_probe5.py").read_bytes()).hexdigest(), (_T / "intent_probe5.sha256").read_text().strip())
        self.assertEqual([t for t in probe.PROBE_REALTIME if not _search(t)], [])
        self.assertEqual([t for t in probe.PROBE_FREE if _search(t)], [])

    def test_colloquial_forms_are_normalised(self) -> None:
        self.assertEqual(intent._colloquial("美元今儿啥价"), "美元今天什么价")
        self.assertEqual(intent._colloquial("这会儿咋样"), "现在怎么样")

    def test_a_specific_tool_still_owns_its_request(self) -> None:
        self.assertEqual(intent.required_tools("明天北京会下雨吗"), ["get_weather"])
        self.assertEqual(intent.required_tools("现在几点了"), ["get_datetime"])
        self.assertEqual(intent.required_tools("it is raining in london right now"), ["get_weather"])


class FallbackFRequiresAValueNoun(unittest.TestCase):
    def test_now_word_plus_quantity_alone_is_not_enough(self) -> None:
        for t in ("现在几个人", "今天来了多少客人", "最近有几个新同事"):
            self.assertFalse(intent._fallback_realtime(t), t)
        self.assertTrue(intent._fallback_realtime("今天白银多少钱一克") or intent.is_realtime_fact("今天白银多少钱一克"))


if __name__ == "__main__":
    unittest.main()
