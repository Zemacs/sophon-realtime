"""Side-channel for tool facts that are not part of the spoken text (which source answered, how long).

A tool calls ``note(source=...)`` while it runs; the Worker calls ``take()`` right after the tool returns.
Both run in the same asyncio task, so a ContextVar is enough and nothing leaks across concurrent jobs.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_meta: ContextVar[dict[str, Any] | None] = ContextVar("sophon_tool_meta", default=None)


def note(**kv: Any) -> None:
    cur = dict(_meta.get() or {})
    cur.update(kv)
    _meta.set(cur)


def take() -> dict[str, Any]:
    cur = _meta.get() or {}
    _meta.set(None)
    return cur
