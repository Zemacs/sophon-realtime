"""Built-in assistant tools for Sophon Live (general voice assistant pack).

The GPT-Live front still only calls ``delegate(task)``.

- ``SOPHON_LIVE_ORCH=1``: GPU-side LLM orchestrator selects tools (preferred Path-1 design).

Schemas are also exported for OpenAI Realtime / BackOffice clients that want
first-class function calling.
"""
from .registry import TOOL_SCHEMAS, list_tool_names
from .runner import run_tool

__all__ = ["TOOL_SCHEMAS", "list_tool_names", "run_tool", "resolve_task_orch", "run_worker_job"]


async def resolve_task_orch(
    task: str,
    *,
    instructions: str = "",
    history: list | None = None,
) -> str:
    from .orchestrator import resolve_task
    return await resolve_task(task, instructions=instructions, history=history)


async def run_worker_job(
    job,  # coordinator.schemas.WorkerJob
    *,
    instructions: str = "",
    history: list | None = None,
):
    """Worker entry for the Coordinator: execute one WorkerJob, return a WorkerResult."""
    from .orchestrator import run_job
    return await run_job(job, instructions=instructions, history=history)
