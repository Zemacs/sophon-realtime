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


@dataclass
class SpeakPlan:
    kind: Literal["pad", "clarify", "answer", "stop_speaking", "noop"]
    text: str = ""
    barge_policy: Literal["keep", "drop_playback"] = "keep"
    task_id: str | None = None
    version: int | None = None


@dataclass
class WorkerJob:
    task_id: str
    version: int
    objective: str
    language: str = "zh"
    tool_scope: list[str] = field(default_factory=lambda: ["*"])
    constraints: list[dict[str, Any]] = field(default_factory=list)
    freshness_token: str = ""


@dataclass
class WorkerResult:
    task_id: str
    version: int
    status: Literal["completed", "needs_input", "failed", "cancelled"]
    spoken_answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    needs_slots: list[str] = field(default_factory=list)


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
    via: Literal["jev", "heuristic"] = "heuristic"

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
