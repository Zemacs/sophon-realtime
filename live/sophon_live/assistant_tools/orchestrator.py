"""Path-1 co-located delegation orchestrator: text LLM selects tools; tools run on the GPU host.

Used when ``SOPHON_LIVE_ORCH=1``. Replaces keyword ``run_task`` routing. Speaks OpenAI Chat
Completions + tools against a local vLLM (see docs/sophon-path1-colocated-backend.md).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiohttp

from .registry import TOOL_SCHEMAS
from .runner import run_tool

log = logging.getLogger("sophon_live.orch")

_DEFAULT_URL = "http://127.0.0.1:8101/v1/chat/completions"
_DEFAULT_MODEL = "qwen3.8-27b-bo"
_DEFAULT_KEY = "local"
_DEFAULT_STEPS = 6
_DEFAULT_TIMEOUT = 120.0

SYS = (
    "You are the back-office agent behind a voice assistant on a live call. "
    "The voice front already heard the caller; you only see the task text it relayed. "
    "Use tools when you need facts or actions. When done, reply with ONLY the short spoken "
    "answer for the caller (one or two natural sentences). No markdown, no tool names, "
    "no 'SAY:' prefix."
)


def _cfg() -> tuple[str, str, str, int, float]:
    """Read env on each call so tests and restarts without re-import stay honest."""
    url = os.environ.get("SOPHON_LIVE_ORCH_URL", _DEFAULT_URL).strip() or _DEFAULT_URL
    model = os.environ.get("SOPHON_LIVE_ORCH_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    key = os.environ.get("SOPHON_LIVE_ORCH_KEY", _DEFAULT_KEY).strip() or _DEFAULT_KEY
    try:
        steps = int(os.environ.get("SOPHON_LIVE_ORCH_STEPS", str(_DEFAULT_STEPS)))
    except ValueError:
        steps = _DEFAULT_STEPS
    try:
        timeout = float(os.environ.get("SOPHON_LIVE_ORCH_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    except ValueError:
        timeout = _DEFAULT_TIMEOUT
    return url, model, key, max(1, steps), max(5.0, timeout)


def _openai_tools() -> list[dict]:
    out = []
    for t in TOOL_SCHEMAS:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return out


def _say(text: str) -> str:
    t = (text or "").strip()
    i = t.upper().rfind("SAY:")
    if i >= 0:
        t = t[i + 4:].strip()
    return t.strip().strip('"') or "好的，我查过了。"


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        args = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}
    return args if isinstance(args, dict) else {}


async def resolve_task(task: str, *, instructions: str = "") -> str:
    """Run LLM tool loop for one Path-1 delegation task; return commentary text for the front."""
    task = (task or "").strip()
    if not task:
        return "我这边没有收到具体任务，请再说一次。"
    url, model, key, max_steps, timeout = _cfg()
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SYS + (("\n\nExtra policy:\n" + instructions) if instructions else "")},
        {"role": "user", "content": f"Front relayed this task from the caller:\n{task}"},
    ]
    tools = _openai_tools()
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as cs:
        for step in range(max_steps):
            body = {"model": model, "messages": msgs, "temperature": 0.2, "tools": tools}
            try:
                async with cs.post(url, json=body, headers=headers) as r:
                    status = r.status
                    try:
                        j = await r.json(content_type=None)
                    except Exception as e:  # noqa: BLE001
                        raw = await r.text()
                        log.warning("orch LLM bad JSON HTTP %s: %r body=%s", status, e, raw[:300])
                        return "后台模型暂时不可用，请稍后再试。"
                if status >= 400:
                    log.warning("orch LLM HTTP %s: %s", status, str(j)[:300])
                    return "后台模型暂时不可用，请稍后再试。"
                choices = j.get("choices") if isinstance(j, dict) else None
                if not choices:
                    log.warning("orch LLM empty choices: %s", str(j)[:300])
                    return "后台模型暂时不可用，请稍后再试。"
                m = choices[0].get("message") or {}
            except Exception as e:  # noqa: BLE001
                log.warning("orch LLM call failed: %r", e)
                return "后台模型连接失败，请稍后再试。"
            tcs = m.get("tool_calls") or []
            if not tcs:
                return _say(m.get("content") or "")
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = _parse_args(fn.get("arguments"))
                out = await run_tool(name, args)
                log.info(
                    "orch step=%d tool=%s args=%s out=%s",
                    step,
                    name,
                    json.dumps(args, ensure_ascii=False)[:120],
                    out[:160],
                )
                msgs.append({"role": "tool", "tool_call_id": tc.get("id") or name, "content": out})
    return "我还在查，请稍等一下再问一次。"
