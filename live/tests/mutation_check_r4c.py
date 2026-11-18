"""Mutation check for round 4c (S-21): applies each mutant to the source, runs the tests that must catch it, restores the file.
Run from the live/ directory: python tests/mutation_check_r4c.py  (expects every line KILLED, exit 0). Not part of unittest discovery."""
import shutil, subprocess, sys, os
ROOT = "."
PY = sys.executable
T_S19 = ["tests.test_r4_s19_inflight", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
T_S18 = ["tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1"]
T_INT = ["tests.test_r4_intent_i10", "tests.test_r4_intent_gate"]
INF = "sophon_live/assistant_tools/inflight.py"; SCH = "sophon_live/coordinator/schemas.py"; INT = "sophon_live/coordinator/intent.py"
CB = "    fut.add_done_callback(lambda _f: s.release())\n    return await asyncio.wrap_future(fut)"
M = [
 ("run: no release after the work RAISES", INF, "fut.add_done_callback(lambda _f: s.release())", "fut.add_done_callback(lambda f: s.release() if (not f.cancelled() and f.exception() is None) else None)", T_S19),
 ("run: no release after a pre-start CANCEL", INF, "fut.add_done_callback(lambda _f: s.release())", "fut.add_done_callback(lambda f: None if f.cancelled() else s.release())", T_S19),
 ("run: released when the awaiter is cancelled (thread still holds the socket)", INF, CB, "    try:\n        return await asyncio.wrap_future(fut)\n    finally:\n        s.release()", T_S19),
 ("run: no release when submit fails", INF, "    except BaseException:\n        s.release()\n        raise", "    except BaseException:\n        raise", T_S19),
 ("pool smaller than the cap", INF, "ThreadPoolExecutor(max_workers=MAX_INFLIGHT,", "ThreadPoolExecutor(max_workers=max(1, MAX_INFLIGHT - 1),", T_S19),
 ("run: blocking acquire (queues instead of failing)", INF, "s.acquire(blocking=False)", "s.acquire(timeout=2)", T_S19),
 ("echo: fullmatch -> search", SCH, "pat.fullmatch(text or \"\")", "pat.search(text or \"\")", T_S18),
 ("echo: fullmatch -> match", SCH, "pat.fullmatch(text or \"\")", "pat.match(text or \"\")", T_S18),
 ("weather slot widened to 200 chars", SCH, "(?:no source|{_WEATHER_SRC}(?:, {_WEATHER_SRC}){{0,7}})", "[A-Za-z0-9:_,\\- ]{{0,200}}", T_S18),
 ("echo: no word boundary", SCH, "    if re.match(r\"[a-z0-9]\", e[0]):\n        pat = r\"(?<![a-z0-9])\" + pat\n    if re.match(r\"[a-z0-9]\", e[-1]):\n        pat += r\"(?![a-z0-9])\"\n", "", T_S18),
 ("echo: checked against Worker args", SCH, "_said_in(echoed, objective)", "_said_in(echoed, str((ev.get('args') or {})) + ' ' + objective)", T_S18),
 ("templates not bound to the tool", SCH, "TOOL_FAILURE_TEMPLATES.get(str((ev or {}).get(\"tool\") or \"\"), ())", "[p for v in TOOL_FAILURE_TEMPLATES.values() for p in v]", T_S18),
 ("intent: unit-conversion veto removed", INT, "    if _UNIT_CONVERT.search(t) and not _MONEYISH.search(t):\n        return False                                    # unit conversion is arithmetic, never a lookup", "    if False:\n        return False", T_INT),
 ("intent: 黄金/白银 back in _MONEYISH", INT, "|USDT|多少钱|价格", "|USDT|黄金|白银|多少钱|价格", T_INT),
]
bad = 0
for name, f, old, new, tests in M:
    src = open(f, encoding="utf-8").read()
    if old not in src:
        print("MUTANT NOT APPLICABLE:", name); bad += 1; continue
    shutil.copy(f, f + ".orig")
    try:
        open(f, "w", encoding="utf-8").write(src.replace(old, new, 1))
        env = dict(os.environ, PYTHONPATH=".", SOPHON_LIVE_JEV="0")
        r = subprocess.run([PY, "-m", "unittest", *tests], capture_output=True, text=True, env=env, timeout=300)
        tail = [l for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))][:3]
        killed = r.returncode != 0
        print(("KILLED  " if killed else "SURVIVED"), name, "|", "; ".join(x.split(" (")[0] for x in tail))
        bad += 0 if killed else 1
    finally:
        shutil.move(f + ".orig", f)
print("surviving/not-applicable:", bad)
sys.exit(1 if bad else 0)
