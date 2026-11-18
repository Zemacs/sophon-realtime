"""Built-in assistant tools for Sophon Live (general voice assistant pack).

The GPT-Live front still only calls ``delegate(task)``.

- ``SOPHON_LIVE_ORCH=1``: GPU-side LLM orchestrator selects tools (preferred Path-1 design).
- ``SOPHON_LIVE_BUILTIN_TOOLS=1``: legacy heuristic ``run_task`` (demo only; not architecture).

Schemas are also exported for OpenAI Realtime / BackOffice clients that want
first-class function calling.
"""
from .registry import TOOL_SCHEMAS, list_tool_names
from .runner import run_tool, run_task

__all__ = ["TOOL_SCHEMAS", "list_tool_names", "run_tool", "run_task", "resolve_task_orch"]


async def resolve_task_orch(
    task: str,
    *,
    instructions: str = "",
    history: list | None = None,
) -> str:
    from .orchestrator import resolve_task
    return await resolve_task(task, instructions=instructions, history=history)
