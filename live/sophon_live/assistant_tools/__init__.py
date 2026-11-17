"""Built-in assistant tools for Sophon Live (general voice assistant pack).

The GPT-Live front still only calls ``delegate(task)``. When
``SOPHON_LIVE_BUILTIN_TOOLS=1``, the gateway can resolve the task with this
registry instead of (or before) the browser demo backend.

Schemas are also exported for OpenAI Realtime / BackOffice clients that want
first-class function calling.
"""
from .registry import TOOL_SCHEMAS, list_tool_names
from .runner import run_tool, run_task

__all__ = ["TOOL_SCHEMAS", "list_tool_names", "run_tool", "run_task"]
