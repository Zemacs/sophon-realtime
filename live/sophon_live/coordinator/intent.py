"""Which requests need a tool (their answer must come from a tool result, never from the model's head)?

v0.1.0-rc2: **the model judges, the code enforces** (deplux AGENTS.md hard constraint). This module contains NO keyword,
regex or word-list judgement of what the user said. The one semantic decision it frames -- "does this request need a
lookup / a device / the memory, and which tool?" -- is one multiple-choice question put to the router model (JEV / Karpo)
together with the routing question; ``tools_from_answer`` maps the model's enumerated answer (an option key we offered) to
tool names. That mapping is a lookup over our own option keys, not a reading of the user's words.

What the code does with the model's answer (enforcement, in ``coordinator.api`` / ``coordinator.grounding``):
  - stamp ``WorkerJob.require_tools`` (the Worker must call one of them; its first call is forced);
  - reject a Worker answer that carries no successful tool evidence for such a job;
  - reject a spoken number that no tool result of this job contains (number-containment, ``grounding.py``).
When the router is unavailable the answer is ``None`` (= unknown): no tool is forced, and the number-containment check
applies to every answer of the job.
Keep the tool names in sync with ``assistant_tools.registry`` (a test enforces it).
"""
from __future__ import annotations

from typing import Any

TOOL_KEY = "tool"

# Option key -> what the router sees. The text is PROMPT (instructions + few-shot examples for the model); no code reads it.
# The examples are deliberately not taken from the evaluation sets (tests/intent_*.py), which only measure.
NONE = "NONE"
MEMORY_SAVE = "memory_save"      # option key: the memory_note tool, and the user explicitly asked to keep it long-term
TOOL_CRITERIA: dict[str, str] = {
    NONE: "不用查：闲聊、问候、感谢、观点、常识历史、概念解释、算术换算、讲自己的事、创作（笑话、故事）。例：「光在真空里多快」「十二乘以八」",
    "get_weather": "天气、气温、下雨下雪、带伞、穿什么（任意城市日期）",
    "get_news": "新闻、头条、快讯",
    "get_datetime": "当前时间、日期、星期几",
    "device_control": "开关或调节家里设备：灯、空调、窗帘、电视、风扇、插座",
    "web_search": "答案会随时间变、必须现查才准：价格行情汇率、股市、比分赛果、票房榜单、路况航班、突发事件、现任人物，或用户要求上网搜。"
                  "拿不准会不会变就选这项。例：「瑞士法郎今天兑人民币多少」「现在三环堵不堵」",
    "memory_note": "回忆用户以前说过的话和偏好（只读）",
    MEMORY_SAVE: "用户明确要求长期记住一件事或偏好（以后/默认/总是…也算）。例：「帮我记一下我对花生过敏」",
}

# What the user hears while the Worker runs: by the tool the router chose (our own table, keyed by option key).
_PAD_BY_TOOL = {
    "get_weather": "我帮你查一下。",
    "get_news": "我帮你查一下。",
    "web_search": "我帮你查一下。",
    "memory_note": "",
    "device_control": "好的，马上。",
    "get_datetime": "",
}


def build_tool_question() -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "判断这位用户的请求需要用哪个工具才能答对；以 user_said（用户原话，没有时看 utterance）为准。选一项。",
        "criteria": dict(TOOL_CRITERIA),
    }


def tools_from_answer(choice: str | None) -> list[str] | None:
    """The router's enumerated answer -> tool names. ``[]`` = the model said no tool; ``None`` = unknown (no / non-option answer)."""
    if choice is None or choice not in TOOL_CRITERIA:
        return None
    if choice == NONE:
        return []
    return ["memory_note"] if choice == MEMORY_SAVE else [choice]


def explicit_save_from_answer(choice: str | None) -> bool | None:
    """The model's judgment "the user explicitly asked to remember this long-term" (memory_gate enforces it). None = unknown -> never writes."""
    if choice is None or choice not in TOOL_CRITERIA:
        return None
    return choice == MEMORY_SAVE


def pad_for_tools(tools: list[str] | None) -> str:
    """Filler by the tool the model chose. Lookups 查一下; device 好的，马上; time / memory / jokes / chit-chat none."""
    if not tools:
        return ""
    return _PAD_BY_TOOL.get(tools[0], "")
