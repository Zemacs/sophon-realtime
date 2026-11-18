"""Critic v1 review S-12: v1/4b's test only asserted that some Chinese text is inside ``_sys()``.

What can be checked on a CPU, behaviorally, about 「过渡语按类型说」:
  1. the prompt the ENGINE receives (through the real ``on_event`` -> ``engine.set_instructions`` path, not a direct ``_sys()`` call)
     carries the per-type rule AFTER whatever instructions the client / Tag preamble put in, with its override clause, so a
     conflicting example earlier in the prompt ("比如我帮你查一下" for every tool) is explicitly overruled;
  2. the lead-in the rule prescribes for each kind of request is the SAME one the Coordinator would pad with (intent.pad_for):
     the prompt and the code cannot drift apart (weather/news/search/memory -> 我帮你查一下; device -> 好的，马上; time/joke -> none);
  3. it is still the last thing in the prompt after later ``session.instructions.append`` updates.
What it cannot show: that the Tag model obeys the rule. That needs the live 54-turn run (not on this box, see PR body).

Skipped on trees without gateway.FILLER_RULE (v1/4b, branch builder/v1-filler-sys); it runs on the merged tree r4 + 4b.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

import sophon_live.gateway as gw
from sophon_live.coordinator import intent
from tests.test_r0_coord_boundary import _bare_session

TAG_PREAMBLE = "调用工具前，先用一句很短的口语告诉用户你要做什么（比如“我帮你查一下”）。"   # the Tag plugin's fixed example (tools.py:21)


class _Engine:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def set_instructions(self, text: str) -> None:
        self.prompts.append(text)


def _session(lang: str = "zh", instructions: str = ""):
    s = _bare_session()
    s.lang = lang
    s.instructions = instructions
    s.engine = _Engine()
    s.args = None
    s.started = True

    async def send(_ev: dict) -> None:
        return None

    s.send = send
    return s


@unittest.skipUnless(hasattr(gw, "FILLER_RULE"), "gateway.FILLER_RULE comes with v1/4b (builder/v1-filler-sys)")
class LeadInRuleReachesTheEngine(unittest.TestCase):
    def _append(self, s, text: str) -> str:
        asyncio.run(gw.LiveSession.on_event(s, {"type": "session.instructions.append", "content": text, "event_id": "e1"}))
        return s.engine.prompts[-1]

    def test_rule_comes_after_a_conflicting_example_and_overrules_it(self) -> None:
        s = _session("zh", "你是小助手。" + TAG_PREAMBLE)
        prompt = self._append(s, "用户喜欢简短回答。")
        self.assertLess(prompt.index("比如“我帮你查一下”"), prompt.index("调用 delegate 之前只说一句很短的过渡语"))   # the rule is the later, last word
        self.assertTrue(prompt.rstrip().endswith("（这条优先于其他关于过渡语的举例）。"))
        self.assertIn("用户喜欢简短回答。", prompt)                                                     # client instructions are kept
        self.assertIn("你的回复会被直接合成为语音说出来", prompt)

    def test_rule_survives_later_appends_and_is_per_language(self) -> None:
        for lang, marker in (("zh", "这条优先于其他关于过渡语的举例"), ("en", "this overrides any other example of a lead-in")):
            s = _session(lang, "x")
            for i in range(3):
                prompt = self._append(s, f"note {i}")
                self.assertTrue(prompt.rstrip().endswith(marker + (")." if lang == "en" else "）。")), (lang, i))

    def test_prompt_and_coordinator_prescribe_the_same_lead_in_per_type(self) -> None:
        rule = gw.FILLER_RULE["zh"]
        by_lead_in = {
            "我帮你查一下。": ["明天北京天气怎么样", "今天有什么科技新闻", "今天美元兑人民币是多少", "比特币现在多少钱"],
            "好的，马上。": ["把客厅的灯关掉"],
            "": ["现在几点了", "讲个笑话", "你还记得我喜欢吃什么吗"],
        }
        for lead_in, texts in by_lead_in.items():
            for text in texts:
                self.assertEqual(intent.pad_for(text), lead_in, text)                    # what the Coordinator pads with
                if lead_in:
                    self.assertIn(lead_in.rstrip("。"), rule, text)                      # ...is what the prompt tells the front to say
        self.assertIn("问时间或日期、讲笑话、问我记得什么（记忆），不要说任何过渡语", rule)           # and the no-lead-in types are the same three (memory: v0.1.0, live-tested)


if __name__ == "__main__":
    unittest.main()
