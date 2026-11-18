"""Mutation check for stage 0 (JEV reliability): applies each mutant to the source, runs the tests that must catch it, restores the file.
Run from the live/ directory: python tests/mutation_check_s0.py  (expects every line KILLED, exit 0). Not part of unittest discovery."""
import shutil, subprocess, sys, os
PY = sys.executable
T = ["tests.test_s0_jev_reliability", "tests.test_trace"]
JC = "sophon_live/coordinator/jev_client.py"; API = "sophon_live/coordinator/api.py"; GW = "sophon_live/gateway.py"
M = [
 ("retry also on timeout / connect timeout", JC, "if e.reason == REFUSED and attempt < CFG.connect_retries", "if e.reason in (REFUSED, TIMEOUT, CONNECT_TIMEOUT) and attempt < CFG.connect_retries"),
 ("no retry on refused", JC, "attempt < CFG.connect_retries and", "attempt < 0 and"),
 ("two retries", JC, "attempt < CFG.connect_retries and", "attempt < CFG.connect_retries + 1 and"),
 ("retry also on HTTP 5xx", JC, "if e.reason == REFUSED and attempt", "if e.reason in (REFUSED, HTTP_5XX) and attempt"),
 ("breaker never opens", JC, "self.consecutive >= CFG.breaker_fails", "self.consecutive >= 10**9"),
 ("open breaker does not short-circuit", JC, "                raise JevError(BREAKER_OPEN, \"open\")", "                pass"),
 ("no /healthz probe before the trial", JC, "                ok, ms, detail = await probe(endpoint)", "                ok, ms, detail = True, 0.0, 'skipped'"),
 ("half-open lets everyone through", JC, "        if self.busy:\n            raise JevError(BREAKER_OPEN, f\"{self.state}, probe/trial in flight\")", "        if False:\n            raise JevError(BREAKER_OPEN, f\"{self.state}, probe/trial in flight\")"),
 ("4xx counts towards the breaker", JC, "_BREAKER_REASONS = {TIMEOUT,", "_BREAKER_REASONS = {HTTP_4XX, TIMEOUT,"),
 ("failed trial does not re-open", JC, "            self.state, self.opened_at = \"open\", _clock()\n            log.warning(\"jev breaker: trial failed", "            self.opened_at = _clock()\n            log.warning(\"jev breaker: trial failed"),
 ("healthz 'ok: false' accepted", JC, "if not (isinstance(data, dict) and data.get(\"ok\") is True):", "if False:"),
 ("connect timeout not applied", JC, "sock_connect=min(CFG.connect_timeout, timeout)", "sock_connect=None"),
 ("default total timeout 8 s again", JC, "_f(\"SOPHON_JEV_TIMEOUT\", 1.5)", "_f(\"SOPHON_JEV_TIMEOUT\", 8.0)"),
 ("answer types not counted", JC, "        stats.answers[f\"{k}:{t}\"] = stats.answers.get(f\"{k}:{t}\", 0) + 1\n", "        pass\n"),
 ("wait_ready never gives up in time", JC, "if waited + interval > limit:", "if waited > limit * 1000:"),
 ("secondary on by default", JC, "SECONDARY: SecondaryFn | None = None", "SECONDARY: SecondaryFn | None = (lambda s, q: None)"),
 ("non-choice route silently accepted", API, "    if jev_client.choice_answer(resp, \"route\") is None:\n        log.warning(\"jev answered but route", "    if False:\n        log.warning(\"jev answered but route"),
 ("degraded with active task starts a task", API, "    if ledger.snapshot_for_jev():\n        log.warning(\"jev degraded", "    if False:\n        log.warning(\"jev degraded"),
 ("degraded with no active task asks anyway", API, "    if ledger.snapshot_for_jev():\n        log.warning(\"jev degraded", "    if True:\n        log.warning(\"jev degraded"),
 ("clarification sentence changed", API, "你是想改刚才那个，还是问个新的？", "你想怎么样？"),
 ("degraded not counted", API, "    jev_client.note_degraded(reason)\n    if ledger.snapshot_for_jev():", "    if ledger.snapshot_for_jev():"),
 ("degraded labelled jev", API, "meta[\"via\"], meta[\"reason\"] = \"degraded\", reason", "meta[\"via\"], meta[\"reason\"] = \"jev\", reason"),
 ("heuristic_route is back in api", API, "from .route import build_route_questions, build_state, heuristic_control, slot_edit", "from .route import build_route_questions, build_state, heuristic_control, heuristic_route, slot_edit"),
 ("jev ok not counted", API, "    jev_client.note_ok()\n    meta[\"via\"] = \"jev\"", "    meta[\"via\"] = \"jev\""),
 ("trace loses degraded_reason", GW, "jev_state=result.jev_state, degraded_reason=result.degraded_reason)", "jev_state=result.jev_state)"),
 ("trace loses jev_state", GW, "jev_state=result.jev_state, degraded_reason", "degraded_reason"),
 ("probe cancelled leaves busy stuck", JC, "            except BaseException:\n                self.busy = False\n                raise", "            except BaseException:\n                raise"),
 ("cancelled/4xx trial never releases the slot", JC, "        if self.state == \"half_open\":\n            self.busy = False\n\n\nbreaker", "        pass\n\n\nbreaker"),
 ("alias import of heuristic_route on the jev_disabled path", API, "    if not USE_JEV:\n        jev_client.note_fail(\"jev_disabled\")", "    if not USE_JEV:\n        from .route import heuristic_route as _hr; _hr\n        jev_client.note_fail(\"jev_disabled\")"),
 ("alias import of heuristic_route inside _degraded", API, "def _degraded(text: str, req: AdmitRequest, ledger: Ledger, meta: dict, reason: str) -> AdmitResult:\n", "def _degraded(text: str, req: AdmitRequest, ledger: Ledger, meta: dict, reason: str) -> AdmitResult:\n    from .route import heuristic_route as _hr\n    _hr\n"),
 ("module-level alias import in api", API, "from .route import build_route_questions, build_state, heuristic_control, slot_edit", "from .route import build_route_questions, build_state, heuristic_control, slot_edit, heuristic_route as _hr"),
]
bad = 0
for name, f, old, new in M:
    src = open(f, encoding="utf-8").read()
    if old not in src:
        print("MUTANT NOT APPLICABLE:", name); bad += 1; continue
    shutil.copy(f, f + ".orig")
    try:
        open(f, "w", encoding="utf-8").write(src.replace(old, new, 1))
        env = dict(os.environ, PYTHONPATH=".")
        r = subprocess.run([PY, "-m", "unittest", *T], capture_output=True, text=True, env=env, timeout=240)
        tail = [l for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))][:3]
        killed = r.returncode != 0
        print(("KILLED  " if killed else "SURVIVED"), name, "|", "; ".join(x.split(" (")[0] for x in tail))
        bad += 0 if killed else 1
    except subprocess.TimeoutExpired:
        print("KILLED  ", name, "| (timeout = the tests hung on the mutant)")
    finally:
        shutil.move(f + ".orig", f)
print("surviving/not-applicable:", bad)
sys.exit(1 if bad else 0)
