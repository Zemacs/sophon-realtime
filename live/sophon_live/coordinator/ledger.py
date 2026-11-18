"""In-session Task Ledger (versioned). Code enforces; models only pick options."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .schemas import TaskRecord, TaskStatus


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Ledger:
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def active(self) -> list[TaskRecord]:
        live = {
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.NEEDS_INPUT,
            TaskStatus.COMPLETED,
        }
        return [self.tasks[i] for i in self.order if self.tasks[i].status in live]

    def latest(self) -> TaskRecord | None:
        act = self.active()
        return act[-1] if act else None

    def create(self, objective: str, *, constraints: list | None = None) -> TaskRecord:
        tid = _new_id()
        rec = TaskRecord(
            task_id=tid,
            version=1,
            objective=objective,
            status=TaskStatus.QUEUED,
            constraints=list(constraints or []),
        )
        self.tasks[tid] = rec
        self.order.append(tid)
        return rec

    def refine(self, task_id: str, objective: str) -> TaskRecord | None:
        rec = self.tasks.get(task_id)
        if rec is None:
            return None
        if rec.status in (TaskStatus.CANCELLED, TaskStatus.SUPERSEDED):
            return None
        old_ver = rec.version
        rec.status = TaskStatus.SUPERSEDED
        new = TaskRecord(
            task_id=task_id,
            version=old_ver + 1,
            objective=objective,
            status=TaskStatus.QUEUED,
            constraints=list(rec.constraints),
        )
        # keep same id, bump version in place for lookup simplicity
        self.tasks[task_id] = new
        return new

    def mark_running(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec:
            rec.status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, answer: str) -> None:
        rec = self.tasks.get(task_id)
        if rec and rec.status != TaskStatus.SUPERSEDED:
            rec.status = TaskStatus.COMPLETED
            rec.spoken_answer = answer

    def mark_needs_input(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec:
            rec.status = TaskStatus.NEEDS_INPUT

    def cancel(self, task_id: str, version: int) -> bool:
        rec = self.tasks.get(task_id)
        if rec is None or rec.version != version:
            return False
        rec.status = TaskStatus.CANCELLED
        return True

    def snapshot_for_jev(self) -> list[dict]:
        out = []
        for rec in self.active()[-5:]:
            out.append(
                {
                    "task_id": rec.task_id,
                    "version": rec.version,
                    "objective": rec.objective,
                    "status": rec.status.value,
                }
            )
        return out
