"""Mutation check for round 4c (S-21): applies each mutant to the source, runs the tests that must catch it, restores the file.
Run from the live/ directory: python tests/mutation_check_r4c.py  (expects every line KILLED, exit 0). Not part of unittest discovery."""
import shutil, subprocess, sys, os
ROOT = "."
PY = sys.executable
T_S19 = ["tests.test_r4_s19_inflight", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
T_S18 = ["tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1"]
T_INT = ["tests.test_r4_intent_i10", "tests.test_r4_intent_gate", "tests.test_r4d_intent"]
T_D = ["tests.test_r4d_s23_s24", "tests.test_r4_news_evidence"]
T_E = ["tests.test_r4d_s23_s24", "tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
ORCH = "sophon_live/assistant_tools/orchestrator.py"
NEWS = "sophon_live/assistant_tools/news.py"; WS = "sophon_live/assistant_tools/web_search.py"; WX = "sophon_live/assistant_tools/weather.py"
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
 ("echo: checked against Worker args", SCH, "_echo_ok(m.group(1), objective, city=tool == \"get_weather\")", "_echo_ok(m.group(1), str((ev or {}).get('args')) + ' ' + objective, city=tool == \"get_weather\")", T_S18),
 ("templates not bound to the tool", SCH, "TOOL_FAILURE_TEMPLATES.get(tool, ())", "[p for v in TOOL_FAILURE_TEMPLATES.values() for p in v]", T_S18),
 ("intent: unit-conversion veto removed", INT, "    if _UNIT_CONVERT.search(t) and not _MONEYISH.search(t):\n        return False                                    # unit conversion is arithmetic, never a lookup", "    if False:\n        return False", T_INT),
 ("intent: 黄金/白银 back in _MONEYISH", INT, "|USDT|多少钱|价格", "|USDT|黄金|白银|多少钱|价格", T_INT),
 # Round 4e (B-3 / I-17): the removed rules must not come back, and the structured signals must be what drives the outcome
 ("I-17: weather slot accepts any token again", SCH, '"|".join(SOURCE_ERROR_CODES)', 'r"[A-Za-z][A-Za-z0-9_]{2,40}"', T_E),
 ("I-17: weather notes carry the exception class name", WX, "err = _error_code(e)", "err = type(e).__name__", T_E),
 ("B-3: echo rejects single characters again (a length rule)", SCH, '    return any(_said_in(n, objective or "") for n in names)', '    return len(echoed.strip()) >= 2 and any(_said_in(n, objective or "") for n in names)', T_E),
 ("B-3: echo consults the city list again (fragment of a known city rejected)", SCH, '    return any(_said_in(n, objective or "") for n in names)',
  '    from sophon_live.assistant_tools.weather import CITIES\n    if any(echoed != c and echoed in c for c in CITIES):\n        return False\n    return any(_said_in(n, objective or "") for n in names)', T_E),
 ("B-3: the worker reads failure words from the tool text again", ORCH, "failed = fail_code is not None", 'failed = fail_code is not None or "暂时连不上" in out or "unavailable" in out.lower()', T_E),
 ("B-3: the worker ignores the tool's failure code", ORCH, 'code = (tool_meta or {}).get("failure")', "code = None", T_E),
 ("B-3: the worker ignores an ok:false envelope", ORCH, 'if isinstance(obj, dict) and obj.get("ok") is False:', "if False:", T_E),
 ("B-3: news does not report its failure code", NEWS, "        toolmeta.note(failure=failcodes.NETWORK)\n", "", T_E),
 ("B-3: web_search does not report 'no results'", WS, "            toolmeta.note(failure=failcodes.NO_RESULTS)\n", "", T_E),
 ("B-3: weather does not report 'busy'", WX, '            meta["failure"] = failcodes.BUSY\n', "", T_E),
 ("S-24: news says 'feeds unreachable' when the cap refused", NEWS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: search says 'network' when the cap refused", WS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: weather says 'cannot find' when the cap refused", WX, "        if busy:\n            meta[\"failure\"] = failcodes.BUSY", "        if False:\n            meta[\"failure\"] = failcodes.BUSY", T_D),
 ("S-24: busy sentence not whitelisted for web_search", SCH, '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",\n        BUSY_ZH_RE, BUSY_EN_RE,', '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",', T_D),
 ("I-11: fallback F without the value noun", INT, "and _QTY_NOW_RE.search(t) and _VALUE_NOUN_RE.search(t)", "and _QTY_NOW_RE.search(t)", T_INT),
 ("I-11: arithmetic veto removed from the primary route", INT, "    if _ARITH_STRONG.search(t):\n        return False\n", "", T_INT),
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
