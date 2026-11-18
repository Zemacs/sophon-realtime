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
    """The filler follows the tool the ROUTER MODEL chose (an option key we offered), never the user's words."""

    def test_lookups_say_let_me_check(self) -> None:
        for tool in ("get_weather", "get_news", "web_search"):
            self.assertEqual(intent.pad_for_tools([tool]), "我帮你查一下。", tool)

    def test_device_control_says_ok_right_away(self) -> None:
        self.assertEqual(intent.pad_for_tools(["device_control"]), "好的，马上。")

    def test_time_memory_jokes_and_unknown_get_no_filler(self) -> None:
        # memory: with a lead-in the front sometimes ended the turn after it and never called delegate (zh_memory 4/4 failed on c88195c)
        for tools in (["get_datetime"], ["memory_note"], [], None):
            self.assertEqual(intent.pad_for_tools(tools), "", tools)

    def test_the_model_answer_maps_to_tools_by_option_key_only(self) -> None:
        self.assertEqual(intent.tools_from_answer("web_search"), ["web_search"])
        self.assertEqual(intent.tools_from_answer("NONE"), [])
        self.assertIsNone(intent.tools_from_answer(None))
        self.assertIsNone(intent.tools_from_answer("无关紧要"))        # a non-option answer is "unknown", never guessed


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
        self.assertIn("问我记得什么（记忆），不要说任何过渡语", p)
        self.assertIn("绝不能只说过渡语就结束", p)         # v0.1.0: lead-in alone must never end the turn (zh_memory / zh_light regression)
        s.lang = "en"
        self.assertIn("what you remember about the user (memory), say nothing", s._sys())


if __name__ == "__main__":
    unittest.main()
