"""Dispatch built-in tools by name. Which tool a request needs is the model's decision (orchestrator); this module only executes by name."""
from __future__ import annotations

import json
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
