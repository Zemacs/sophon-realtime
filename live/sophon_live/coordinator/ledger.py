"""In-session Task Ledger (versioned). Code enforces; models only pick options."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .schemas import TaskRecord, TaskStatus

# A task in one of these states can still be asked about, refined or re-heard.
_LIVE = {
    TaskStatus.QUEUED,
    TaskStatus.RUNNING,
    TaskStatus.NEEDS_INPUT,
    TaskStatus.COMPLETED,
    TaskStatus.DELIVERED,
}
# Work is still pending: a cancel is meaningful and a worker result is still wanted.
_PENDING = {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.NEEDS_INPUT}


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Ledger:
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    superseded: list[TaskRecord] = field(default_factory=list)  # audit trail of replaced versions

    def active(self) -> list[TaskRecord]:
        return [self.tasks[i] for i in self.order if self.tasks[i].status in _LIVE]

    def latest(self) -> TaskRecord | None:
        act = self.active()
        return act[-1] if act else None

    def pending_input(self) -> TaskRecord | None:
        """The most recent task that is waiting for the user to fill a slot."""
        for tid in reversed(self.order):
            rec = self.tasks[tid]
            if rec.status == TaskStatus.NEEDS_INPUT:
                return rec
        return None

    def supersede_pending_input(self) -> int:
        """The user moved on: open clarifications are replaced by the new request and can no longer capture answers."""
        n = 0
        for tid in self.order:
            rec = self.tasks[tid]
            if rec.status == TaskStatus.NEEDS_INPUT:
                rec.status = TaskStatus.SUPERSEDED
                self.superseded.append(rec)
                n += 1
        return n

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

    def refine(self, task_id: str, objective: str, *, version: int | None = None) -> TaskRecord | None:
        """Replace the task by a new version (same id). The old version becomes SUPERSEDED.

        ``version`` (when given) must be the current one; a stale or cancelled target is a no-op.
        """
        rec = self.tasks.get(task_id)
        if rec is None:
            return None
        if rec.status in (TaskStatus.CANCELLED, TaskStatus.SUPERSEDED):
            return None
        if version is not None and rec.version != version:
            return None
        rec.status = TaskStatus.SUPERSEDED
        self.superseded.append(rec)
        new = TaskRecord(
            task_id=task_id,
            version=rec.version + 1,
            objective=objective,
            status=TaskStatus.QUEUED,
            constraints=list(rec.constraints),
        )
        self.tasks[task_id] = new  # same id, current version in place
        return new

    def mark_running(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec and rec.status in (TaskStatus.QUEUED, TaskStatus.NEEDS_INPUT, TaskStatus.RUNNING):
            rec.status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, answer: str) -> bool:
        """Only a pending task can complete; a cancelled or superseded one is never revived."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.status not in _PENDING:
            return False
        rec.status = TaskStatus.COMPLETED
        rec.spoken_answer = answer
        return True

    def mark_failed(self, task_id: str, answer: str) -> bool:
        """A pending task whose Worker gave up: keep the apology so STATUS can replay it."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.status not in _PENDING:
            return False
        rec.status = TaskStatus.FAILED
        rec.spoken_answer = answer
        return True

    def mark_needs_input(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec and rec.status in _PENDING:
            rec.status = TaskStatus.NEEDS_INPUT

    def cancel(self, task_id: str, version: int) -> bool:
        """Cancel a pending task at exactly this version. Finished work has nothing to cancel."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.version != version or rec.status not in _PENDING:
            return False
        rec.status = TaskStatus.CANCELLED
        return True

    def set_slot(self, task_id: str, version: int, key: str, value: str) -> bool:
        """Record a task-scoped slot (e.g. city) on a live pending task at exactly this version."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.version != version or rec.status not in _PENDING:
            return False
        rec.slots[str(key)] = str(value)
        return True

    def get_slots(self, task_id: str) -> dict[str, str]:
        rec = self.tasks.get(task_id)
        return dict(rec.slots) if rec else {}

    def mark_delivered(self, task_id: str, version: int) -> bool:
        """The answer was heard to the end. Only a completed task at this version can become DELIVERED."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.version != version:
            return False
        if rec.status == TaskStatus.COMPLETED:
            rec.status = TaskStatus.DELIVERED
            return True
        return rec.status == TaskStatus.DELIVERED

    def note_interrupted(self, task_id: str, version: int) -> bool:
        """Playback was cut off (barge-in / dropped). The task keeps its status: not cancelled, still re-playable."""
        rec = self.tasks.get(task_id)
        if rec is None or rec.version != version:
            return False
        rec.interrupted += 1
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
