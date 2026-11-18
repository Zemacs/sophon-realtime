"""Stable contracts between Frontend ↔ Coordinator ↔ Worker ↔ Delivery.

Keep this module free of gateway/audio/tool I/O so layers stay decoupled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    NEEDS_INPUT = "NEEDS_INPUT"
    COMPLETED = "COMPLETED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


@dataclass
class AdmitRequest:
    session_id: str
    utterance_id: str
    text: str
    language: str = "zh"
    delegate_task: str | None = None
    playback_active: bool = False
    now_ms: int = 0
    raw_text: str = ""  # what the user actually said (ASR), when known; `text` may be the front's rewrite


@dataclass
class SpeakPlan:
    kind: Literal["pad", "clarify", "answer", "stop_speaking", "noop"]
    text: str = ""
    barge_policy: Literal["keep", "drop_playback"] = "keep"
    task_id: str | None = None
    version: int | None = None


# The only sentences a *failed* WorkerResult may carry to the user's ears (besides a failed required tool's own sentence,
# which the Worker records as evidence ``failure_text``). Model free text is never spoken for a failed job (Critic I-5).
NO_TOOL_ANSWER = "这个我这边没查到准确结果，你再问我一次吧。"
FAILED_GENERIC = "这个我这边没办成，你再问我一次吧。"
FIXED_FAILURES = frozenset({
    NO_TOOL_ANSWER,
    FAILED_GENERIC,
    "后台模型暂时不可用，请稍后再试。",
    "后台模型连接失败，请稍后再试。",
    "我还在查，请稍等一下再问一次。",
    "后台编排出错了，请稍后再试。",
})


# The failure sentences OUR tools produce themselves (assistant_tools: weather / news / web_search). The Coordinator speaks a Worker-reported
# ``failure_text`` only if it is exactly one of these (Critic S-13): a Worker cannot launder a fabricated sentence by writing it into
# evidence as the "tool's own text". Group 1, where present, is what the tool echoes back (city / query) and must come from the request.
TOOL_FAILURE_TEMPLATES = tuple(re.compile(p) for p in (
    r"暂时查不到([\u4e00-\u9fffA-Za-z·\- ]{1,24})的天气，请稍后再试。",
    r"Weather for ([\u4e00-\u9fffA-Za-z·\- ]{1,40}) is unavailable right now \([A-Za-z0-9:_,\- ]{0,120}\)\.",
    r"找不到([\u4e00-\u9fffA-Za-z·\- ]{1,24})的天气信息。",
    r"No weather data for ([\u4e00-\u9fffA-Za-z·\- ]{1,40})\.",
    r"新闻源暂时连不上，稍后再帮你看看头条吧。",
    r"News feeds are unreachable right now\. Try again in a moment\.",
    r"网络暂时不可用，没法完成网页搜索，请稍后再试。",
    r"Web search is unavailable right now due to a network problem\. Please try again later\.",
    r"没有搜到关于「([^」\n]{1,60})」的可靠摘要。",
    r"I couldn't find useful snippets for '([^'\n]{1,60})'\.",
))


def is_tool_failure_sentence(text: str, ev: dict, objective: str = "") -> bool:
    """``text`` is one of our tools' own failure sentences, and what it echoes (city / query) is something the request actually said."""
    for pat in TOOL_FAILURE_TEMPLATES:
        m = pat.fullmatch(text or "")
        if not m:
            continue
        echoed = m.group(1) if m.groups() else ""
        if not echoed:
            return True
        asked = " ".join(str(v) for v in ((ev.get("args") or {}).values() if isinstance(ev.get("args"), dict) else ())) + " " + (objective or "")
        return echoed.strip().lower() in asked.lower()
    return False


def tool_succeeded(ev: dict) -> bool:
    """A tool call counts as evidence only when the Worker says ``ok`` is exactly True (a missing ``ok`` is NOT success)."""
    return ev.get("ok") is True and not ev.get("missing") and not ev.get("refused")


def freshness_token_for(task_id: str, version: int) -> str:
    """The one place that spells the token: it names exactly one (task_id, version) of the Ledger."""
    return f"{task_id}:{version}"


@dataclass
class WorkerJob:
    task_id: str
    version: int
    objective: str
    language: str = "zh"
    tool_scope: list[str] = field(default_factory=lambda: ["*"])
    constraints: list[dict[str, Any]] = field(default_factory=list)
    freshness_token: str = ""
    # Tool-backed request (weather / time / news / device / search / memory): the answer MUST come from a tool result.
    # Set by the Coordinator (coordinator.intent); the Worker forces the first call, finish_job rejects a tool-less answer.
    require_tools: list[str] = field(default_factory=list)


@dataclass
class WorkerResult:
    task_id: str
    version: int
    status: Literal["completed", "needs_input", "failed", "cancelled"]
    spoken_answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    needs_slots: list[str] = field(default_factory=list)
    freshness_token: str = ""  # the Worker echoes the job's token; the Coordinator refuses a result whose token is not the job's


@dataclass
class PlaybackReceipt:
    task_id: str
    version: int
    outcome: Literal["delivered", "interrupted", "abandoned"]


@dataclass
class ControlDecision:
    """A CONTROL operation aimed at one (task_id, version) of the Ledger."""

    op: Literal["CANCEL", "STATUS", "REFINE"]
    task_id: str
    version: int
    via: Literal["jev", "heuristic", "guard"] = "heuristic"

    @property
    def key(self) -> str:
        return f"{self.op}:{self.task_id}:{self.version}"


@dataclass
class TaskRecord:
    task_id: str
    version: int
    objective: str
    status: TaskStatus = TaskStatus.QUEUED
    spoken_answer: str = ""
    constraints: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)  # task-scoped slots (this task version only; never long-term)
    interrupted: int = 0  # how many times playback of the answer was cut off (answer stays re-playable)
