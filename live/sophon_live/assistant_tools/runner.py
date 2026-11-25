"""Dispatch built-in tools by name, and map free-text delegate tasks onto tools."""
from __future__ import annotations

import json
import re
from typing import Any

from . import datetime_tool, device, memory, news, weather, web_search

_HANDLERS = {
    "get_datetime": datetime_tool.run,
    "get_weather": weather.run,
    "web_search": web_search.run,
    "get_news": news.run,
    "device_control": device.run,
    "memory_note": memory.run,
}


# Tool arguments that describe the current request: recorded as task-scoped slots (never as memory).
_SLOT_ARGS = {
    "get_weather": ("city", "when"),
    "get_news": ("topic", "lang"),
    "get_datetime": ("what",),
    "web_search": ("query",),
    "device_control": ("device", "action"),
}


def _capture_slots(name: str, args: dict[str, Any]) -> None:
    from ..coordinator.memory_gate import current_task

    ctx = current_task()
    if ctx is None:
        return
    for k in _SLOT_ARGS.get(name, ()):
        v = args.get(k)
        if v not in (None, ""):
            ctx.ledger.set_slot(ctx.task_id, ctx.version, k, str(v))


async def run_tool(name: str, args: dict[str, Any] | None = None) -> str:
    """Execute one registered tool; return a short spoken-friendly Chinese/English string or JSON error."""
    fn = _HANDLERS.get(name)
    if fn is None:
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)
    _capture_slots(name, args or {})
    try:
        return await fn(args or {})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "tool": name, "error": str(e)[:300]}, ensure_ascii=False)


def _route_task(task: str) -> tuple[str, dict[str, Any]] | None:
    """Heuristic mapping from GPT-Live delegate(task) text → (tool_name, args)."""
    t = (task or "").strip()
    if not t:
        return None
    s = t.lower()
    zh = bool(re.search(r"[一-鿿]", t))

    if re.search(r"天气|下雨|气温|温度|冷不冷|热不热|weather|rain|temperature", s):
        when = "tomorrow" if re.search(r"明天|tomorrow", s) else "today"
        city = None
        for zh_name, en in weather.CITIES.items():
            if zh_name in t or en.lower() in s:
                city = en
                break
        return "get_weather", {"city": city or ("Beijing" if zh else "Beijing"), "when": when}

    if re.search(r"几点|时间|what time|the time", s):
        return "get_datetime", {"what": "time"}
    if re.search(r"日期|几号|星期|周几|礼拜|date|what day", s):
        return "get_datetime", {"what": "date"}

    if re.search(r"灯|空调|音量|窗帘|电视|播放|闹钟|提醒|light|volume|alarm|remind|ac\b|curtain|tv\b", s):
        return "device_control", {"device": t, "action": "do", "value": ""}

    if re.search(r"记住|记下|保存偏好|save memory|remember that", s):
        return "memory_note", {"op": "save", "key": "note", "value": t}
    if re.search(r"我喜欢|我的偏好|recall|你还记得", s):
        return "memory_note", {"op": "list"}

    if re.search(r"新闻|头条|headline|news", s):
        topic = re.sub(r"^(帮我|请|查一下|查询|搜索|搜一下)", "", t).strip() or t
        return "get_news", {"topic": topic, "lang": "zh" if zh else "en", "max_items": 3}

    if re.search(r"搜索|查一下|搜一下|search|google|百度|查找|关于", s):
        q = re.sub(r"^(帮我|请|搜索|搜一下|查一下|查找|search)\s*", "", t, flags=re.I).strip() or t
        return "web_search", {"query": q, "max_results": 3}

    return None


async def run_task(task: str) -> str | None:
    """Resolve a free-text delegation task. Returns None if no built-in tool matches."""
    routed = _route_task(task)
    if not routed:
        return None
    name, args = routed
    return await run_tool(name, args)
