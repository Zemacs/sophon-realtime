"""Test doubles for the Worker's tool runner that report failure the way the real tools do: in structure (Round 4e / B-3).

The Worker no longer reads a tool's spoken text for words to decide that it failed. A real tool says so through
``toolmeta.note(failure=<code>)`` (see ``sophon_live.failcodes``) or by returning an ``{"ok": false}`` JSON envelope; a fake that only
returns a failure *sentence* is, to the Worker, a successful call with unusual text.
"""
from __future__ import annotations

from unittest import mock

from sophon_live import failcodes
from sophon_live.assistant_tools import toolmeta


def failing_tool(text: str, code: str = failcodes.NETWORK) -> mock.AsyncMock:
    async def _run(name, args=None):  # noqa: ANN001, ARG001
        toolmeta.note(failure=code)
        return text
    return mock.AsyncMock(side_effect=_run)
