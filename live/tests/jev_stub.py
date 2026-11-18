"""Test-only stand-in for the JEV router.

Older ledger/control/trace tests exercise the Coordinator's ledger behaviour, not Karpo. They used to reach the keyword
``heuristic_route`` through the JEV-unavailable fallback; production no longer does that (v0.1.0 model client), so these tests
get an explicit fake JEV that answers with the legacy keyword router. ``heuristic_route`` stays defined (uncalled in
production) only for this stub and its own unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sophon_live.coordinator import api  # noqa: E402
from sophon_live.coordinator.route import build_state as _real_build_state, heuristic_control, heuristic_route

_last: dict = {}
real_system_one = api.jev_client.system_one   # the production client, for tests that exercise it


def _build_state(text, ledger, **kw):  # noqa: ANN001
    _last["args"] = (text, ledger)
    return _real_build_state(text, ledger, **kw)


async def fake_system_one(state, questions, **kw):  # noqa: ANN001
    text, ledger = _last["args"]
    route = heuristic_route(text, ledger)
    ans: dict = {"route": {"type": "choice", "choice": route}}
    if route == "CONTROL":
        d = heuristic_control(text, ledger)
        if d is not None:
            ans["control"] = {"type": "choice", "choice": f"{d.op}:{d.task_id}:{d.version}"}
    return {"answers": ans}


def install() -> None:
    api.build_state = _build_state
    api.jev_client.system_one = fake_system_one
