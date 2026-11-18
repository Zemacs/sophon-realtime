"""Which requests are *tool-backed* (their answer must come from a tool result, never from the model's head)?

The Coordinator decides this, not the Worker: a weather / time / news / device / search / memory request that is
answered without the matching tool call is fabricated. ``required_tools`` is a deterministic, keyword-level
judgement (zh + en, no model), used to
  - stamp ``WorkerJob.require_tools`` (the Worker must call one of them; its first call is forced), and
  - reject a Worker answer that carries no tool evidence for a tool-scoped job (``finish_job``).
Jokes, chit-chat and general knowledge need no tool and return an empty list.
Keep the tool names in sync with ``assistant_tools.registry`` (a test enforces it).
"""
from __future__ import annotations

import re

# (tool name, pattern) in priority order; the first matching pattern wins for the *primary* tool.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("get_weather", re.compile(r"天气|下雨|下雪|下不下|气温|温度|冷不冷|热不热|带伞|穿什么|weather|rain|snow|temperature|forecast", re.I)),
    ("get_news", re.compile(r"新闻|头条|快讯|headline|\bnews\b", re.I)),
    ("get_datetime", re.compile(r"几点|现在时间|当前时间|什么时间|几号|星期几|周几|礼拜几|今天日期|今天的日期|what time|the time|what day|today'?s date|\bdate\b", re.I)),
    ("device_control", re.compile(r"(打开|关掉|关闭|关上|调高|调低|调亮|调暗|开|关)[^，。！？]{0,6}(灯|空调|窗帘|电视|风扇|插座)|(灯|空调|窗帘|电视|风扇|插座)[^，。！？]{0,3}(打开|关掉|关闭|关上|开|关)|音量|闹钟|设个提醒|设置提醒|turn (on|off)|\blights?\b|volume|alarm|curtain|\btv\b", re.I)),
    ("web_search", re.compile(r"搜索|搜一下|搜一搜|百度一下|汇率|股价|股票|google|\bsearch\b|exchange rate", re.I)),
    ("memory_note", re.compile(r"记住|记下|别忘了|你还?记得|上次说|之前说|我的偏好|喜欢吃什么|喜欢喝什么|remember|recall", re.I)),
)


def required_tools(text: str) -> list[str]:
    """Tool names a request needs, primary first. Empty for chit-chat / jokes / general knowledge."""
    t = (text or "").strip()
    if not t:
        return []
    out: list[str] = []
    for name, pat in _PATTERNS:
        if pat.search(t) and name not in out:
            out.append(name)
    return out


def is_tool_scoped(text: str) -> bool:
    return bool(required_tools(text))


# What the user hears while the Worker runs (the front says it before calling delegate): by intent, minimal.
_PAD_BY_TOOL = {
    "get_weather": "我帮你查一下。",
    "get_news": "我帮你查一下。",
    "web_search": "我帮你查一下。",
    "memory_note": "我帮你查一下。",
    "device_control": "好的，马上。",
    "get_datetime": "",
}


def pad_for(text: str) -> str:
    """Filler for a request by intent. 查询类 查一下; 设备 好的，马上; 时间 / 笑话 / 闲聊 none."""
    tools = required_tools(text)
    if not tools:
        return ""
    return _PAD_BY_TOOL.get(tools[0], "")
