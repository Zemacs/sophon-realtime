"""体验版 v1 / 4: the lead-in before a delegated task depends on the kind of request (minimal)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import intent
from sophon_live.gateway import DELEGATE


class PadForTests(unittest.TestCase):
    def test_lookups_say_let_me_check(self) -> None:
        for t in ("明天上海会下雨吗", "今天有什么科技新闻", "用搜索查一下今天美元兑人民币大概多少"):
            self.assertEqual(intent.pad_for(t), "我帮你查一下。", t)

    def test_device_control_says_ok_right_away(self) -> None:
        for t in ("关掉客厅的灯", "把客厅灯打开"):
            self.assertEqual(intent.pad_for(t), "好的，马上。", t)

    def test_time_jokes_and_chitchat_get_no_filler(self) -> None:
        for t in ("现在几点了", "讲个笑话", "讲一个不一样的笑话", "你好"):
            self.assertEqual(intent.pad_for(t), "", t)

    def test_memory_recall_is_a_lookup(self) -> None:
        self.assertEqual(intent.pad_for("用户上次说喜欢吃什么"), "我帮你查一下。")


class FrontInstructionTests(unittest.TestCase):
    """The lead-in is spoken by the voice front before it calls delegate; its tool description is where we steer it."""

    def test_zh_description_assigns_lead_ins_by_type(self) -> None:
        d = DELEGATE["zh"]["description"]
        self.assertIn("我帮你查一下", d)
        self.assertIn("好的，马上", d)
        self.assertIn("不要说过渡语", d)

    def test_no_blanket_lead_in_rule_remains(self) -> None:
        self.assertNotIn("先用一句很短的口语告诉用户你去处理一下", DELEGATE["zh"]["description"])
        self.assertNotIn("say one very short spoken phrase that you're on it", DELEGATE["en"]["description"])


class SystemPromptTests(unittest.TestCase):
    def test_session_prompt_carries_the_per_type_lead_in_rule(self) -> None:
        from tests.test_r0_coord_boundary import _bare_session
        s = _bare_session()
        s.instructions, s.lang = "你是小助手。", "zh"
        p = s._sys()
        self.assertTrue(p.startswith("你是小助手。"))
        self.assertIn("我帮你查一下", p)
        self.assertIn("好的，马上", p)
        self.assertIn("讲笑话，不要说任何过渡语", p)
        s.lang = "en"
        self.assertIn("a joke, say nothing", s._sys())


if __name__ == "__main__":
    unittest.main()
