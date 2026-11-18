"""Round 4j / S-39: behaviour test of the browser test page, run in node against the REAL <script> of static/index.html (stubbed DOM,
WebSocket, fetch / XHR / beacon, virtual timers; see page_harness.js).

For a delegation whose backend is ``coord`` or ``orch`` the page must send NOTHING to the gateway and make NO network call, even after a minute of
virtual time (so a send inside Promise.then, an aliased setInterval, a renamed helper or any keyword demo is caught by what it DOES, not by how it is
written); no manual box is shown. Any other backend only shows the manual box, still sends nothing by itself; the user's own manual answer is sent
(positive control that the harness can see a send).

Needs ``node``. Without it the test is skipped (reported as skipped, never as passed); the string checks in test_r4i_demo_removed_retired_env.py still run."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
PAGE = LIVE / "sophon_live" / "static" / "index.html"
HARNESS = LIVE / "tests" / "page_harness.js"
CASES = [
    {"name": "coord", "delegation": {"backend": "coord"}},
    {"name": "orch", "delegation": {"backend": "orch"}},
    {"name": "none", "delegation": {}, "type_manual": True},
    {"name": "builtin", "delegation": {"backend": "builtin"}},
    {"name": "other", "delegation": {"backend": "something-else", "task": "what's the weather in Beijing tomorrow?"}},
]


def run_page(page: Path = PAGE) -> dict:
    r = subprocess.run(["node", str(HARNESS), str(page), json.dumps(CASES)], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(r.stderr[-800:])
    return json.loads(r.stdout)


@unittest.skipUnless(shutil.which("node"), "node is not installed: the page behaviour test cannot run (the static string checks still do)")
class PageDoesNothingOnItsOwn(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rep = run_page()

    def test_the_harness_ran_the_page(self) -> None:
        self.assertNotIn("error", self.rep)
        self.assertEqual(set(self.rep), {c["name"] for c in CASES})

    def test_coord_and_orch_send_nothing_call_nothing_and_show_no_box(self) -> None:
        for b in ("coord", "orch"):
            self.assertEqual(self.rep[b], {"sent": [], "network": [], "manual_box_shown": False, "pending": False}, b)

    def test_any_other_backend_only_shows_the_manual_box(self) -> None:
        for b in ("none", "builtin", "other"):
            r = self.rep[b]
            self.assertEqual((r["sent"], r["network"], r["manual_box_shown"], r["pending"]), ([], [], True, True), b)

    def test_the_users_own_manual_answer_is_the_only_thing_the_page_sends(self) -> None:
        sent = self.rep["none"]["after_manual_click"]
        self.assertEqual([(m["type"], m["content"], m["delegation_id"]) for m in sent], [("session.commentary.append", "手动回答", "item_x")])


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class HarnessSeesRewrittenRegressions(unittest.TestCase):
    """The harness is tested against the rewrites a string check misses (Critic's R-1 list): each must show up as a send / network call / box."""
    BASE = PAGE.read_text(encoding="utf-8")
    HOOK = 'setR("等你手动回答…");'
    SEND = 'send({type:"session.commentary.append",content:"demo-answer",delegation_id:d.id})'

    def _run(self, html: str) -> dict:
        tmp = LIVE / "tests" / "_page_variant.html"
        tmp.write_text(html, encoding="utf-8")
        try:
            return run_page(tmp)
        finally:
            tmp.unlink()

    def test_regressions_are_caught_by_behaviour(self) -> None:
        variants = {
            "send inside Promise.then, for every backend": ("async function onDelegation(d) {", "async function onDelegation(d) { Promise.resolve().then(() => " + self.SEND + ");"),
            "aliased inject in a setInterval": (self.HOOK, self.HOOK + " const f = inject; const iv = setInterval(() => { f(d.id, 'x', 'commentary', setR); clearInterval(iv); }, 300);"),
            "network call under another name": ("async function onDelegation(d) {", "async function onDelegation(d) { globalThis['fe' + 'tch']('https://example.invalid/w');"),
            "coord short-circuit removed": ('d.backend === "orch" || d.backend === "coord"', 'd.backend === "orch"'),
            "keyword demo defined outside onDelegation": ("async function onDelegation(d) {", "const ans = t => /天气|weather/i.test(t) ? 'sunny' : null;\nasync function onDelegation(d) { setTimeout(() => { const a = ans(d.task || ''); if (a) " + self.SEND + "; }, 50);"),
        }
        for name, (old, new) in variants.items():
            self.assertIn(old, self.BASE, name)
            rep = self._run(self.BASE.replace(old, new, 1))
            clean = {"sent": [], "network": [], "manual_box_shown": False, "pending": False}
            acted = any(r["sent"] or r["network"] for r in rep.values())          # the page did something on its own, for some backend
            self.assertTrue(acted or rep["coord"] != clean or rep["orch"] != clean, name)


if __name__ == "__main__":
    unittest.main()
