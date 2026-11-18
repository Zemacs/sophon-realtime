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
TOOL_CRITERIA: dict[str, str] = {
    NONE: "不需要查任何东西：闲聊、问候、感谢、观点、常识和历史知识、概念解释、不随时间变化的事实、创作（讲笑话、编故事）、"
          "算术和单位换算、用户在讲述自己的经历。例：「给我讲个冷笑话」「光在真空里多快」「二战哪年结束」「我今天跑了五公里」「十二乘以八」「谢谢你」",
    "get_weather": "问天气、气温、下雨下雪、要不要带伞、穿什么（任意城市或日期）。例：「后天杭州冷吗」「出门要带伞吗」",
    "get_news": "想听新闻、头条、快讯。例：「来点体育新闻」「今天有什么大事」",
    "get_datetime": "问当前的时间、日期、星期几。例：「现在几点了」「今天星期几」",
    "device_control": "控制家里的设备：开关或调节灯、空调、窗帘、电视、风扇、插座。例：「把卧室的灯调暗一点」「关掉空调」",
    "web_search": "答案会随时间变化、必须现查才准的事实：价格、行情、汇率、股市是否开盘、比分和赛果、票房和榜单、路况/限号/航班车次、"
                  "地震台风等突发事件、现任职务的人，或者用户明确要求上网搜索。拿不准答案会不会变的时候，选这一项（宁可多查一次，也不要凭记忆报数字）。"
                  "例：「瑞士法郎今天兑人民币多少」「今天沪市成交了多少」「周杰伦演唱会还有票吗」「现在三环堵不堵」「帮我搜一下这家店的评价」",
    "memory_note": "让助手记住一件事，或回忆用户以前说过的话和偏好。例：「帮我记一下明天要交报告」「我之前跟你说过我爱吃什么来着」",
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
    return [] if choice == NONE else [choice]


def pad_for_tools(tools: list[str] | None) -> str:
    """Filler by the tool the model chose. Lookups 查一下; device 好的，马上; time / memory / jokes / chit-chat none."""
    if not tools:
        return ""
    return _PAD_BY_TOOL.get(tools[0], "")
