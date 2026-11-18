"""Round 4c / I-10 + S-20: the realtime fallback, the two named regressions, currency vs unit conversion (with a non-vacuous veto test)."""
from __future__ import annotations

import re
import unittest
from unittest import mock

from sophon_live.coordinator import intent
from tests import intent_dev as dev
from tests import intent_frozen as frozen


def _search(t: str) -> bool:
    return "web_search" in intent.required_tools(t)


class NamedRegressions(unittest.TestCase):
    def test_critic_incr4_regressions_route_to_search(self) -> None:
        for t in ("上海今天限号是多少", "今天沪深两市成交额多少"):
            self.assertTrue(_search(t), t)


class DevSetIsClean(unittest.TestCase):
    """The only set the rules were tuned on. Not an acceptance number (that is the frozen set / Critic's independent set)."""

    def test_dev_set(self) -> None:
        self.assertEqual([t for t in dev.DEV_REALTIME if not _search(t)], [])
        self.assertEqual([t for t in dev.DEV_FREE if _search(t)], [])


class FrozenSetCeiling(unittest.TestCase):
    """Held-out (frozen BEFORE the change, run once after it): measured 2/57 misses and 2/61 over-triggers (baseline at 4b5ec1b: 22/57, 1/61).
    These are ceilings, so a later change cannot silently make them worse; do not 'fix' them by editing the frozen set."""

    def test_misses_and_over_triggers_do_not_exceed_the_measured_numbers(self) -> None:
        miss = [t for t in frozen.FROZEN_REALTIME if not _search(t)]
        over = [t for t in frozen.FROZEN_FREE if _search(t)]
        self.assertLessEqual(len(miss), 2, miss)
        self.assertLessEqual(len(over), 2, over)


class FallbackDesign(unittest.TestCase):
    def test_a_specific_tool_keeps_the_request(self) -> None:
        # 「今天/明天 + 吗」 is the fallback's shape, but the weather / time / news tools own these
        self.assertEqual(intent.required_tools("明天北京会下雨吗"), ["get_weather"])
        self.assertEqual(intent.required_tools("现在几点了"), ["get_datetime"])
        self.assertEqual(intent.required_tools("今天有什么新闻"), ["get_news"])

    def test_personal_creative_and_stable_sentences_are_not_searched(self) -> None:
        for t in ("今天你过得怎么样", "我现在有多少钱啊我也不知道", "帮我写首关于今天股市的诗", "今天是谁的生日", "光速现在是多少",
                  "NBA有多少支球队", "世界杯是哪年开始的", "现在怎么样", "今天好无聊啊"):
            self.assertFalse(_search(t), t)

    def test_realtime_ish_but_unlisted_topics_are_searched(self) -> None:
        for t in ("今天人民币汇率中间价是多少", "现在去杭州的火车还有票吗", "今年的奥斯卡最佳影片是什么", "what's the price of ethereum now"):
            self.assertTrue(_search(t), t)


class CurrencyVersusUnitConversion(unittest.TestCase):
    CURRENCY = ("一百欧元折合成人民币是多少", "USD兑CNY今天多少", "把五百港币兑换成人民币是多少", "100美元折算成日元是多少",
                "一克黄金折合多少人民币", "一克黄金多少钱", "convert 100 USD to CNY", "how much is 50 euros in dollars", "把100 USD换成CNY")
    UNIT = ("把一盎司黄金换算成克", "一盎司白银等于多少克", "一吨原油等于多少桶", "一桶汽油折合多少升", "五公斤白银是多少克", "convert 5 miles to kilometers")

    def test_currency_conversion_is_a_lookup(self) -> None:
        for t in self.CURRENCY:
            self.assertTrue(_search(t), t)

    def test_unit_conversion_is_not(self) -> None:
        for t in self.UNIT:
            self.assertFalse(_search(t), t)

    def test_metals_alone_are_not_money(self) -> None:
        self.assertIsNone(intent._MONEYISH.search("一盎司黄金换算成克"))
        self.assertIsNone(intent._MONEYISH.search("五公斤白银是多少克"))
        self.assertIsNotNone(intent._MONEYISH.search("一克黄金多少钱"))

    def test_the_unit_conversion_veto_is_load_bearing(self) -> None:
        # S-21 mutant: without the 换算 veto the metal-to-grams sentences go online; the tests above must see exactly that
        never = re.compile(r"(?!x)x")
        with mock.patch.object(intent, "_UNIT_CONVERT", never):
            self.assertTrue(_search("把一盎司黄金换算成克"))
            self.assertTrue(_search("一盎司白银折合多少克"))
            self.assertTrue(any(_search(t) for t in self.UNIT))
        self.assertFalse(any(_search(t) for t in self.UNIT))


if __name__ == "__main__":
    unittest.main()
