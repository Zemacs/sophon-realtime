"""Stable contracts between Frontend ↔ Coordinator ↔ Worker ↔ Delivery.

Keep this module free of gateway/audio/tool I/O so layers stay decoupled.
"""
from __future__ import annotations

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
