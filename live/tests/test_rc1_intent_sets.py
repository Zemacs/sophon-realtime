"""v0.1.0-rc1 / I-10: realtime miss rate and false-trigger rate on two labeled sets (target <= 5% each).
- Critic incr-4's independent 92 (38 realtime + 54 free): no longer blind (the rules were tuned with it visible), so it is a regression ceiling.
- rc1 held-out (40 + 50): written before the rc1 edits, run once before (1/40 miss, 1/50 over) and once after (0 / 0).
Conservative rule: when unsure whether a question is time-sensitive, search."""
from __future__ import annotations

import unittest

from sophon_live.coordinator import intent
from tests import intent_critic92 as critic
from tests import intent_rc1_heldout as held


def _search(t: str) -> bool:
    return "web_search" in intent.required_tools(t)


class RateTests(unittest.TestCase):
    def _rates(self, pos, neg):
        miss = [t for t in pos if not _search(t)]
        over = [t for t in neg if _search(t)]
        return miss, over

    def test_critic_independent_set(self) -> None:
        self.assertEqual((len(critic.CRITIC_REALTIME), len(critic.CRITIC_FREE)), (38, 54))
        miss, over = self._rates(critic.CRITIC_REALTIME, critic.CRITIC_FREE)
        self.assertLessEqual(len(miss) / len(critic.CRITIC_REALTIME), 0.05, miss)
        self.assertLessEqual(len(over) / len(critic.CRITIC_FREE), 0.05, over)

    def test_rc1_heldout_set(self) -> None:
        miss, over = self._rates(held.HELDOUT_REALTIME, held.HELDOUT_FREE)
        self.assertLessEqual(len(miss) / len(held.HELDOUT_REALTIME), 0.05, miss)
        self.assertLessEqual(len(over) / len(held.HELDOUT_FREE), 0.05, over)


class NamedCases(unittest.TestCase):
    def test_incr4_regressions_and_i10_family_words(self) -> None:
        for t in ("上海今天限号是多少", "今天沪深两市成交额多少", "今天人民币中间价", "美联储最新利率决议是什么", "今晚有什么足球比赛", "今晚湖人对谁"):
            self.assertTrue(_search(t), t)

    def test_narrative_property_and_effect_questions_are_not_lookups(self) -> None:
        for t in ("我昨天看了NBA比赛很精彩", "汇率会影响出口吗", "黄金的密度是多少", "高铁的最高时速是多少", "世界杯多少年举办一次"):
            self.assertFalse(_search(t), t)

    def test_a_narrative_with_a_question_is_still_a_lookup(self) -> None:
        self.assertTrue(_search("我昨天看的那场NBA比赛谁赢了"))


if __name__ == "__main__":
    unittest.main()
