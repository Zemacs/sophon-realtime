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
TASKS = {"weather": "北京今天天气怎么样", "stock": "帮我看一下今天的股票行情", "news": "最近有什么科技新闻", "english": "what's the weather in Beijing tomorrow?",
         "device": "把客厅的灯关掉", "time": "现在几点了", "empty": "", "long": "请帮我查一下这个订单的物流并且告诉我大概什么时候能到，谢谢" * 3}
CASES = ([{"name": f"{b}:{k}", "delegation": {"backend": b, "task": v}} for b in ("coord", "orch") for k, v in TASKS.items()]
         + [{"name": f"{b}:{k}", "delegation": ({"backend": b} if b else {}) | {"task": v}, "type_manual": k == "weather"}
            for b in (None, "builtin", "something-else") for k, v in TASKS.items()])


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

    def test_the_harness_ran_the_page_for_every_kind_of_task(self) -> None:
        self.assertNotIn("error", self.rep)
        self.assertEqual(set(self.rep), {c["name"] for c in CASES})
        self.assertGreaterEqual(len(TASKS), 8)

    def test_coord_and_orch_send_nothing_call_nothing_and_show_no_box_whatever_the_task(self) -> None:
        for b in ("coord", "orch"):
            for k in TASKS:
                self.assertEqual(self.rep[f"{b}:{k}"], {"sent": [], "network": [], "manual_box_shown": False, "pending": False}, (b, k))

    def test_any_other_backend_only_shows_the_manual_box_whatever_the_task(self) -> None:
        for b in ("None", "builtin", "something-else"):
            for k in TASKS:
                r = self.rep[f"{b}:{k}"]
                self.assertEqual((r["sent"], r["network"], r["manual_box_shown"], r["pending"]), ([], [], True, True), (b, k))

    def test_the_users_own_manual_answer_is_the_only_thing_the_page_sends(self) -> None:
        for b in ("None", "builtin", "something-else"):
            sent = self.rep[f"{b}:weather"]["after_manual_click"]
            self.assertEqual([(m["type"], m["content"], m["delegation_id"]) for m in sent], [("session.commentary.append", "手动回答", "item_x")], b)


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
            self.assertEqual(set(rep), {c["name"] for c in CASES}, name)
            clean = {"sent": [], "network": [], "manual_box_shown": False, "pending": False}
            acted = any(r["sent"] or r["network"] for r in rep.values())          # the page did something on its own, for some backend / task
            self.assertTrue(acted or any(rep[f"{b}:{k}"] != clean for b in ("coord", "orch") for k in TASKS), name)


if __name__ == "__main__":
    unittest.main()
