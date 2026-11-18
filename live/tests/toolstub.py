"""Test helper: a stand-in for ``runner.run_tool`` that reports the structured outcome the real tools report.

The orchestrator decides "did the tool fail?" from ``toolmeta.note(ok=False)`` (a return code), never by reading the tool's sentence,
so a mocked tool has to set it the way a real one does.
"""
from __future__ import annotations

from sophon_live.assistant_tools import toolmeta


def tool_returning(out: str, *, ok: bool = True):
    async def run_tool(name, args):  # noqa: ANN001, ARG001
        if not ok:
            toolmeta.note(ok=False)
        return out

    return run_tool
