"""Mutation check (rounds 4c-4e): applies each mutant to the source, runs the tests that must catch it, restores the file.
Run from the live/ directory with pytest installed: python tests/mutation_check_r4c.py   Not part of test discovery.

Exit codes:  0 = every mutant KILLED by a real test failure;  1 = at least one mutant SURVIVED;  2 = SCRIPT ERROR (never "killed"):
  a test group that does not pass cleanly (and collect >= 1 test) on the UNMUTATED source, a mutant whose old text is not in the file,
  or a mutated run that fails to import / collect (an import error or syntax error is not a test catching the mutant) or collects 0 tests.
A mutant counts as KILLED only when >= 1 collected test FAILED and no collection error occurred."""
import re, shutil, subprocess, sys, os
ROOT = "."
PY = sys.executable
T_S19 = ["tests.test_r4_s19_inflight", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
T_S18 = ["tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1"]
T_INT = ["tests.test_r4_intent_i10", "tests.test_r4_intent_gate"]
T_D = ["tests.test_r4d_s23_s24", "tests.test_r4_news_evidence"]
T_E = ["tests.test_r4d_s23_s24", "tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
ORCH = "sophon_live/assistant_tools/orchestrator.py"
T_F = ["tests.test_r4e_b4_i18", "tests.test_orch_worker", "tests.test_orchestrator"]
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
 ("B-3: echo rejects single characters again (a length rule)", SCH, '    return any(_said_in(n, words or "") for n in names)', '    return len(value.strip()) >= 2 and any(_said_in(n, words or "") for n in names)', T_E),
 ("B-3: echo consults the city list again (fragment of a known city rejected)", SCH, '    return any(_said_in(n, words or "") for n in names)',
  '    from sophon_live.assistant_tools.weather import CITIES\n    if any(value != c and value in c for c in CITIES):\n        return False\n    return any(_said_in(n, words or "") for n in names)', T_E),
 ("B-3: the worker reads failure words from the tool text again", ORCH, "failed = fail_code is not None", 'failed = fail_code is not None or "暂时连不上" in out or "unavailable" in out.lower()', T_E),
 ("B-3: the worker ignores the tool's failure code", ORCH, 'code = (tool_meta or {}).get("failure")', "code = None", T_E),
 ("B-3: the worker ignores an ok:false envelope", ORCH, 'if isinstance(obj, dict) and obj.get("ok") is False:', "if False:", T_E),
 ("B-3: news does not report its failure code", NEWS, "        toolmeta.note(failure=failcodes.NETWORK)\n", "", T_E),
 ("B-3: web_search does not report 'no results'", WS, "            toolmeta.note(failure=failcodes.NO_RESULTS)\n", "", T_E),
 ("B-3: weather does not report 'busy'", WX, '            meta["failure"] = failcodes.BUSY\n', "", T_E),
 # Round 4f (B-4 / I-18)
 ("B-4: _ungrounded_city consults the city list again", ORCH, '    return not any(appears_in_words(v, seen, translate=True) for v in (city, bare) if v)',
  '    from sophon_live.coordinator.slots import CITY_NAMES\n    return not any(appears_in_words(v, seen, translate=True) for v in (city, bare) if v) and not any(c in seen for c in CITY_NAMES)', T_F),
 ("B-4: _ungrounded_city never fires", ORCH, '    return not any(appears_in_words(v, seen, translate=True) for v in (city, bare) if v)', "    return False", T_F),
 ("B-4: _ungrounded_city drops the alias translation", ORCH, "appears_in_words(v, seen, translate=True)", "appears_in_words(v, seen)", T_F),
 ("I-18: the speak-now nudge reads the tool text again", ORCH, "succeeded = not failed and tool_meta.get(failcodes.SUCCESS_KEY) is True",
  'succeeded = not failed and ("第1条" in out or "度" in out or "Headlines on" in out)', T_F),
 ("I-18: the nudge ignores the structured success report", ORCH, "succeeded = not failed and tool_meta.get(failcodes.SUCCESS_KEY) is True", "succeeded = False", T_F),
 ("I-18: news does not report success", NEWS, "    toolmeta.note(**{failcodes.SUCCESS_KEY: True}, count=len(picked))", "    pass", T_F),
 ("I-18: weather reports success even without temperatures", WX, "True if (not meta.get(\"failure\") and meta.get(\"tmin\") is not None and meta.get(\"tmax\") is not None) else None", "True", T_F),
 ("S-24: news says 'feeds unreachable' when the cap refused", NEWS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: search says 'network' when the cap refused", WS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: weather says 'cannot find' when the cap refused", WX, "        if busy:\n            meta[\"failure\"] = failcodes.BUSY", "        if False:\n            meta[\"failure\"] = failcodes.BUSY", T_D),
 ("S-24: busy sentence not whitelisted for web_search", SCH, '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",\n        BUSY_ZH_RE, BUSY_EN_RE,', '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",', T_D),
]
def run_group(tests, env):
    """-> (returncode, passed, failed, errors, failed_names, output). Uses pytest; module names are mapped to files."""
    files = [t.replace(".", "/") + ".py" for t in tests]
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--maxfail=50", "-rfE", *files], capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout + r.stderr
    def n(word):
        m = re.search(r"(\d+) " + word, out)
        return int(m.group(1)) if m else 0
    names = [l.split(" - ")[0].split("::", 1)[-1].strip() for l in out.splitlines() if l.startswith("FAILED ")]
    return r.returncode, n("passed"), n("failed"), n("error"), names, out


def script_error(msg):
    print("SCRIPT ERROR:", msg)
    sys.exit(2)


env = dict(os.environ, PYTHONPATH=".", SOPHON_LIVE_JEV="0")
groups = {tuple(m[4]) for m in M}
for g in sorted(groups):                                          # the unmutated source must pass every group, with tests actually collected
    rc, passed, failed, errors, _, out = run_group(list(g), env)
    if rc != 0 or failed or errors or passed == 0:
        script_error(f"baseline group {list(g)} is not green on the unmutated source (rc={rc} passed={passed} failed={failed} errors={errors})\n" + out[-1500:])
    print(f"baseline ok: {passed} tests  {list(g)}")

killed_n = survived_n = 0
for name, f, old, new, tests in M:
    src = open(f, encoding="utf-8").read()
    if old not in src:
        script_error("mutant not applicable (old text not found): " + name)
    shutil.copy(f, f + ".orig")
    try:
        open(f, "w", encoding="utf-8").write(src.replace(old, new, 1))
        rc, passed, failed, errors, names, out = run_group(tests, env)
    finally:
        shutil.move(f + ".orig", f)
    if errors or (passed + failed) == 0:
        script_error(f"mutant '{name}' broke import/collection or collected no tests (errors={errors}, ran={passed + failed}); that is not a kill\n" + out[-1200:])
    if failed:
        killed_n += 1
        print("KILLED  ", name, "|", f"{failed} failing:", "; ".join(names[:3]))
    else:
        survived_n += 1
        print("SURVIVED", name)
print(f"killed={killed_n} survived={survived_n} of {len(M)}")
sys.exit(1 if survived_n else 0)
