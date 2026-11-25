"""Test helper: adapt a text-returning fake Worker into the WorkerJob → WorkerResult entry the gateway now calls."""
from __future__ import annotations

from sophon_live.coordinator.schemas import WorkerJob, WorkerResult


def as_worker(fn):
    """fn(objective, instructions=..., history=...) -> str  ⇒  async (job, instructions=..., history=...) -> WorkerResult."""

    async def run(job: WorkerJob, instructions: str = "", history=None) -> WorkerResult:  # noqa: ANN001
        text = await fn(job.objective, instructions=instructions, history=history)
        return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=text,
                            evidence=[{"tool": t, "ok": True} for t in job.require_tools],
                            freshness_token=job.freshness_token)  # a fake Worker that "ran" the required tool

    return run
