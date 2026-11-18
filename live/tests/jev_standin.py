"""A scripted stand-in for JEV, for tests of the Ledger / gateway control flows.

Those tests need *some* decision for "cancel", "改成北京", "进度呢" … Before stage 0 they got it from the (frozen) heuristics because JEV was off.
Product code no longer routes by heuristics (stage 0 / I-12), so the tests now play JEV explicitly: this stand-in answers like a model
would, using the frozen ``route.heuristic_route`` / ``heuristic_control`` as its *script*. It exists only under tests/."""
from __future__ import annotations

import contextlib
from unittest import mock

from sophon_live.coordinator import api, route as route_mod


async def _ask(text, req, ledger, questions, meta):  # noqa: ANN001
    r = route_mod.heuristic_route(text, ledger)
    answers: dict = {"route": {"type": "choice", "choice": r}}
    if r == "CONTROL":
        d = route_mod.heuristic_control(text, ledger)
        answers["control"] = {"type": "choice", "choice": "NEW" if d is None else f"{d.op}:{d.task_id}:{d.version}"}
    meta["via"] = "jev"
    return {"answers": answers}, None


@contextlib.contextmanager
def standin():
    """Replaces ``api.USE_JEV = False`` in tests that need the old decisions."""
    with mock.patch.object(api, "USE_JEV", True), mock.patch.object(api, "_ask_jev", _ask):
        yield
