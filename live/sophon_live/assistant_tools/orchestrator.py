"""Path-1 co-located delegation orchestrator: text LLM selects tools; tools run on the GPU host.

The orchestrator is the Worker: it executes one ``WorkerJob`` and returns a ``WorkerResult``.
It makes no routing, cancel/refine or keyword decisions. Those belong to the Coordinator (C layer / JEV);
a missing required slot is reported as ``needs_input`` and the C layer words the CLARIFY.
Speaks OpenAI Chat Completions + tools against a local vLLM (see docs/sophon-path1-colocated-backend.md).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import aiohttp

from ..coordinator import slots
from ..coordinator.clarify import clarify_for_slots
from ..coordinator.schemas import NO_TOOL_ANSWER, WorkerJob, WorkerResult, tool_succeeded
from . import toolmeta
from .registry import TOOL_SCHEMAS
from .runner import run_tool

log = logging.getLogger("sophon_live.orch")

_DEFAULT_URL = "http://127.0.0.1:8101/v1/chat/completions"
_DEFAULT_MODEL = "qwen3.8-27b-bo"
_DEFAULT_KEY = "local"
_DEFAULT_STEPS = 6
_DEFAULT_TIMEOUT = 120.0

SYS = (
    "You are the tool-executing worker behind a voice assistant. "
    "You receive ONE job objective that the coordinator already routed and refined; "
    "you never decide whether to ask, cancel, refine or remember anything. "
    "Do exactly the objective with the tools you are given.\n"
    "- Pick the tool that fits the objective (time/date: get_datetime; weather: get_weather; news: get_news; "
    "search: web_search; device: device_control).\n"
    "- If a required tool parameter is not in the objective, call the tool WITHOUT that parameter. "
    "Never guess it and never ask the caller; the coordinator asks.\n"
    "- If the objective says the caller changed their mind or added detail, the newest wording wins.\n"
    "- Plain chat or jokes need no tool: answer directly, and do not repeat an answer already in the recent turns.\n"
    "- If a tool returns headlines or weather facts, speak them immediately; never call the same tool again in that job.\n"
    "- If a tool returns an error JSON or '暂时/不可用/unreachable', apologize briefly and stop retrying the same tool.\n"
    "- Match the caller's language; for Chinese objectives keep get_news lang=zh.\n"
    "When done, reply with ONLY the short spoken answer for the caller (one or two natural sentences). "
    "No markdown, no tool names, no 'SAY:' prefix."
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
    # strip common reasoning wrappers if a model leaks them
    t = re.sub(r"^<think>[\s\S]*?</think>", "", t, flags=re.I).strip()
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


def _history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-6:]:
        task = (h.get("task") or "").strip()
        ans = (h.get("answer") or "").strip()
        if task or ans:
            lines.append(f"- task: {task}\n  answer: {ans}")
    if not lines:
        return ""
    return "Recent turns in this call:\n" + "\n".join(lines)


def _missing_required(name: str, args: dict) -> list[str]:
    """Required tool parameters the model left out (schema-driven, no keywords)."""
    for t in TOOL_SCHEMAS:
        if t["name"] == name:
            req = (t.get("parameters") or {}).get("required") or []
            return [k for k in req if not str(args.get(k) or "").strip()]
    return []


def _ungrounded_city(name: str, args: dict, msgs: list[dict[str, Any]]) -> bool:
    """True when the model made up a weather city: not in what the user said, the recent turns, or a tool result.

    "明天天气怎么样" must not become 上海 because the model likes Shanghai. Lenient on purpose: if any known city is
    mentioned anywhere (the model may translate 北京 to Beijing), the argument is treated as grounded."""
    if name != "get_weather":
        return False
    city = str(args.get("city") or "").strip()
    if not city:
        return False
    seen = "\n".join(str(m.get("content") or "") for m in msgs if m.get("role") in ("user", "tool"))
    bare = city.rstrip("市")
    if city in seen or (bare and bare in seen):
        return False
    return not any(c in seen for c in slots.CITY_NAMES)


def _evidence_has_required(job: WorkerJob, evidence: list[dict]) -> bool:
    """True when the job needs no tool, or at least one of its required tools was called AND succeeded (``ok``)."""
    req = [t for t in (job.require_tools or []) if _in_scope(job, t)]
    return not req or any(ev.get("tool") in req and tool_succeeded(ev) for ev in evidence)


def _tool_failure_text(out: str) -> str:
    """What to say when a required tool ran and failed: the tool's own plain sentence (it is honest by construction),
    never model text. Machine-looking output (JSON) falls back to the generic sentence."""
    t = (out or "").strip()
    if not t or t.startswith(("{", "[")) or len(t) > 120 or "\n" in t:
        return NO_TOOL_ANSWER
    return t


def _force_choice(job: WorkerJob) -> Any:
    """OpenAI ``tool_choice`` that forces the Coordinator-required tool (one tool: by name; several: any tool)."""
    req = [t for t in (job.require_tools or []) if _in_scope(job, t)]
    if not req:
        return None
    if len(req) == 1:
        return {"type": "function", "function": {"name": req[0]}}
    return "required"


def _in_scope(job: WorkerJob, name: str) -> bool:
    scope = job.tool_scope or ["*"]
    return "*" in scope or name in scope


def _result(job: WorkerJob, status: str, text: str, evidence: list[dict], slots: list[str] | None = None) -> WorkerResult:
    return WorkerResult(
        task_id=job.task_id,
        version=job.version,
        status=status,  # type: ignore[arg-type]
        spoken_answer=text,
        evidence=evidence,
        needs_slots=slots or [],
        freshness_token=job.freshness_token,
    )


async def run_job(
    job: WorkerJob,
    *,
    instructions: str = "",
    history: list[dict[str, str]] | None = None,
) -> WorkerResult:
    """Run the LLM tool loop for one WorkerJob. Returns a WorkerResult stamped with the job's (task_id, version)."""
    task = (job.objective or "").strip()
    evidence: list[dict] = []
    if not task:
        return _result(job, "failed", "我这边没有收到具体任务，请再说一次。", evidence)
    url, model, key, max_steps, timeout = _cfg()
    hist = _history_block(history)
    user = f"Job objective:\n{task}"
    if hist:
        user = hist + "\n\n" + user
    required = [t for t in (job.require_tools or []) if _in_scope(job, t)]
    if required:
        user += (
            f"\n\nThis job is tool-backed: you MUST call {' or '.join(required)} and answer ONLY from its result. "
            "Never answer facts (weather, time, news, device state, memory) from your own knowledge."
        )
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SYS + (("\n\nExtra policy:\n" + instructions) if instructions else "")},
        {"role": "user", "content": user},
    ]
    tools = [t for t in _openai_tools() if _in_scope(job, t["function"]["name"])]
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    unavailable = "后台模型暂时不可用，请稍后再试。"
    forced = False   # the forced first call has been attempted (once); a second tool-less answer is rejected
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as cs:
        for step in range(max_steps):
            body: dict[str, Any] = {
                "model": model,
                "messages": msgs,
                "temperature": 0.2,
                "tools": tools,
            }
            choice = _force_choice(job)
            if choice is not None and not _evidence_has_required(job, evidence):
                body["tool_choice"] = choice          # the Coordinator says this answer needs its tool: do not let the model skip it
            # Prefer non-thinking for low-latency spoken answers when the server supports it.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                async with cs.post(url, json=body, headers=headers) as r:
                    status = r.status
                    try:
                        j = await r.json(content_type=None)
                    except Exception as e:  # noqa: BLE001
                        raw = await r.text()
                        log.warning("orch LLM bad JSON HTTP %s: %r body=%s", status, e, raw[:300])
                        return _result(job, "failed", unavailable, evidence)
                if status >= 400:
                    # retry once without chat_template_kwargs / tool_choice if rejected (the guard below still applies)
                    if status == 400 and ("chat_template_kwargs" in body or "tool_choice" in body):
                        body.pop("chat_template_kwargs", None)
                        body.pop("tool_choice", None)
                        async with cs.post(url, json=body, headers=headers) as r2:
                            status = r2.status
                            j = await r2.json(content_type=None)
                        if status >= 400:
                            log.warning("orch LLM HTTP %s: %s", status, str(j)[:300])
                            return _result(job, "failed", unavailable, evidence)
                    else:
                        log.warning("orch LLM HTTP %s: %s", status, str(j)[:300])
                        return _result(job, "failed", unavailable, evidence)
                choices = j.get("choices") if isinstance(j, dict) else None
                if not choices:
                    log.warning("orch LLM empty choices: %s", str(j)[:300])
                    return _result(job, "failed", unavailable, evidence)
                m = choices[0].get("message") or {}
            except Exception as e:  # noqa: BLE001
                log.warning("orch LLM call failed: %r", e)
                return _result(job, "failed", "后台模型连接失败，请稍后再试。", evidence)
            tcs = m.get("tool_calls") or []
            if not tcs:
                if not _evidence_has_required(job, evidence):
                    # A tool-backed job answered without calling its tool: never accept that (fabricated numbers).
                    if not forced:
                        forced = True
                        log.warning("orch step=%d job %s answered without %s -> forcing the tool call", step, job.task_id, required)
                        msgs.append({"role": "assistant", "content": m.get("content") or ""})
                        msgs.append({"role": "user", "content": f"You did not call the tool. Call {' or '.join(required)} now; do not answer without it."})
                        continue
                    log.warning("orch step=%d job %s still no %s call -> failed (answer withheld)", step, job.task_id, required)
                    return _result(job, "failed", NO_TOOL_ANSWER, evidence)
                return _result(job, "completed", _say(m.get("content") or ""), evidence)
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = _parse_args(fn.get("arguments"))
                if not _in_scope(job, name):
                    out = json.dumps({"ok": False, "error": f"tool {name} is not permitted for this job"})
                    evidence.append({"tool": name, "args": args, "ok": False, "refused": "out_of_scope"})
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id") or name, "content": out})
                    continue
                missing = _missing_required(name, args)
                if not missing and _ungrounded_city(name, args, msgs):
                    log.info("orch step=%d get_weather city=%r is not grounded in the request -> needs_input", step, args.get("city"))
                    missing = ["city"]
                if missing:
                    # the Worker never asks the caller: report the gap, the C layer words the CLARIFY
                    log.info("orch step=%d tool=%s missing=%s -> needs_input", step, name, missing)
                    evidence.append({"tool": name, "args": args, "ok": False, "missing": missing})
                    return _result(job, "needs_input", "", evidence, missing)
                t_tool = time.monotonic()
                toolmeta.take()
                out = await run_tool(name, args)
                tool_ms = round((time.monotonic() - t_tool) * 1000)
                tool_meta = toolmeta.take()
                log.info(
                    "orch step=%d tool=%s args=%s out=%s",
                    step,
                    name,
                    json.dumps(args, ensure_ascii=False)[:120],
                    out[:160],
                )
                msgs.append({"role": "tool", "tool_call_id": tc.get("id") or name, "content": out})
                failed = (
                    '"ok": false' in out
                    or "暂时连不上" in out
                    or "暂时查不到" in out
                    or "暂时没有" in out
                    or "unreachable" in out.lower()
                    or "unavailable" in out.lower()
                    or "不可用" in out
                )
                evidence.append({"tool": name, "args": args, "ok": not failed, "ms": tool_ms, **tool_meta})
                if failed and name in required:
                    # A tool-backed job whose tool failed: say so plainly. The model never gets to word it, it could
                    # "recover" with numbers of its own (Critic B-1).
                    log.warning("orch step=%d required tool %s failed -> failed with the tool's own sentence", step, name)
                    text = _tool_failure_text(out)
                    evidence[-1]["failure_text"] = text      # the Coordinator may speak exactly this sentence for the failed job
                    return _result(job, "failed", text, evidence)
                news_ok = name == "get_news" and ("第1条" in out or "Headlines on" in out) and not failed
                weather_ok = name == "get_weather" and ("度" in out or "°" in out or "°C" in out) and not failed
                if news_ok or weather_ok:
                    msgs.append({
                        "role": "user",
                        "content": "Tool succeeded. Speak one or two short oral sentences from this result now. Do NOT call the same tool again.",
                    })
                elif name in ("get_weather", "web_search", "get_news") and failed:
                    msgs.append({
                        "role": "user",
                        "content": "The tool failed. Tell the caller briefly without calling the same tool again.",
                    })
    return _result(job, "failed", "我还在查，请稍等一下再问一次。", evidence)


async def resolve_task(
    task: str,
    *,
    instructions: str = "",
    history: list[dict[str, str]] | None = None,
) -> str:
    """Legacy text-in/text-out entry for the non-Coordinator ``SOPHON_LIVE_ORCH=1`` path.

    Wraps the text as an anonymous WorkerJob; a missing slot comes back as the C layer's clarify wording.
    """
    task = (task or "").strip()
    if not task:
        return "我这边没有收到具体任务，请再说一次。"
    res = await run_job(WorkerJob(task_id="", version=0, objective=task), instructions=instructions, history=history)
    if res.status == "needs_input":
        return clarify_for_slots(res.needs_slots)
    return res.spoken_answer
