"""Mutation check for the rc3 changes (run by hand: ``python -m tests.mutation_rc3 [--jobs 4]``; not part of unittest discovery).

Each mutant edits one production line in a temp copy and runs the whole unit-test suite; KILLED = some test fails (for a reason that is
not an import / syntax error), SURVIVED = the suite stayed green (a test gap) and the exit code is 1. The mutants cover Critic's rc2
survivors (A2 A3 A4 D9 F3) and the rc3 behaviours (RC2-3 breaker cancel, RC2-6 backstop, RC2-7 numerals, RC2-8 invalid answers, RC2-9 net).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
A, G, GR, MC = "sophon_live/coordinator/api.py", "sophon_live/gateway.py", "sophon_live/coordinator/grounding.py", "sophon_live/model_client.py"
K = '("天气" in text or "气温" in text)'

MUTANTS: list[tuple[str, str, str, str]] = [
    ("A2 keyword fallback when the model gave no tool answer (admit)", A,
     "                tools = intent.tools_from_answer(tool_choice)\n",
     "                tools = intent.tools_from_answer(tool_choice)\n                if tools is None and " + K + ": tools = ['get_weather']\n"),
    ("A3 keyword inside judge_tools (no-delegate net)", A,
     "    return tools\n\n\n_ROUTES",
     "    if " + K.replace("text", "text") + ": return ['get_weather']\n    return tools\n\n\n_ROUTES"),
    ("A4 judge_tools falls back to a keyword when the router is down", A,
     '        log.warning("tool judgment unavailable: %r", e)\n        return None',
     '        log.warning("tool judgment unavailable: %r", e)\n        return ["get_weather"] if ' + K + " else None"),
    ("A5 tools_from_answer guesses on an illegal option", "sophon_live/coordinator/intent.py",
     "    if choice is None or choice not in TOOL_CRITERIA:\n        return None\n    if choice == NONE:",
     "    if choice is None or choice not in TOOL_CRITERIA:\n        return ['web_search']\n    if choice == NONE:"),
    ("D9 a source that mentions 一/两 makes 1 and 2 sourced", GR,
     "        vals.extend(v for v, _ in numbers(t or \"\"))\n",
     "        vals.extend(v for v, _ in numbers(t or \"\"))\n        if '一' in (t or '') or '两' in (t or ''): vals.extend((1.0, 2.0))\n"),
    ("D10 12-hour clock equivalence always on", GR, "if clock12 and dec == 0 and 1 <= value <= 12:", "if dec == 0 and 1 <= value <= 12:"),
    ("D11 Chinese decimal part counted as a second number", GR, '(?P<dec>点[零〇一二三四五六七八九]+)?(?P<dunit>', '(?P<dec>(?!)点[零〇一二三四五六七八九]+)?(?P<dunit>'),
    ("D12 lone 两 exempt again", GR, '            if tok == "一" and not m.group("dec"):', '            if tok in ("一", "两") and not m.group("dec"):'),
    ("F3 a timeout inside the call is classified as an application error", MC,
     "            except (asyncio.TimeoutError, TimeoutError):\n                br.failure(\"timeout\")",
     "            except (KeyError,):\n                br.failure(\"timeout\")"),
    ("F3b is_connection_error treats a timeout as a retryable connection error", MC,
     "    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):\n        return False\n    try:",
     "    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):\n        return True\n    try:"),
    ("RC2-3 half-open probe is not released on cancel", MC, "            br.release_probe()\n", "            pass\n"),
    ("RC2-6a number check only for numeric tools / unknown (old rule)", A,
     "    return bool(_successful_outputs(result))\n", "    return False\n"),
    ("RC2-6b review_numbers never turns the check on", A, "    if verdict != \"NOT_FACT\":\n        job.check_numbers = True", "    pass"),
    ("RC2-6c review_numbers treats an unusable verdict as NOT_FACT", A, 'if verdict != "NOT_FACT":', 'if verdict == "FACT":'),
    ("RC2-6d review_numbers skipped by the gateway", G, "        await review_numbers(job, result)", "        pass"),
    ("RC2-8 invalid tool answers are not counted", A, "                if tools is None:\n                    _invalid(", "                if False:\n                    _invalid("),
    ("RC2-9a net: answered front still gets a second answer", A, 'if front_said and jev_client.choice_answer(resp, FRONT_REPLY_KEY) != "LEAD_IN_ONLY":', "if False:"),
    ("RC2-9b net: language hard-coded again", A, '    if language:\n        state["language"] = language', '    if True:\n        state["language"] = "zh"'),
    ("RC2-9c net: a text-length proxy is back", G, "            lu = getattr(self, \"last_user\", None)\n",
     "            if len((r.get(\"text\") or \"\").strip()) > 16: return\n            lu = getattr(self, \"last_user\", None)\n"),
    ("B1 task_faithful DIFFERENT ignored", A, 'if faithful == "DIFFERENT":', "if False:"),
    ("B3 task_faithful always takes the raw words", A, 'if faithful == "DIFFERENT":', "if raw:"),
]


def run_one(m: tuple[str, str, str, str]) -> tuple[str, str]:
    name, rel, old, new = m
    d = Path(tempfile.mkdtemp(prefix="mut_rc3_"))
    try:
        shutil.copytree(LIVE, d / "live", ignore=shutil.ignore_patterns("__pycache__", ".git"))
        p = d / "live" / rel
        s = p.read_text(encoding="utf-8")
        if old not in s:
            return name, "NOT APPLICABLE (pattern moved)"
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON")}
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=d / "live", capture_output=True, text=True, env=env, timeout=900)
        invalid = any(k in r.stderr for k in ("SyntaxError", "_FailedTest", "ModuleNotFoundError", "ImportError"))
        failing = [l.split(" (")[0] for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
        status = "SURVIVED" if r.returncode == 0 else ("INVALID" if invalid else "KILLED")
        return name, f"{status} | {len(failing)} failing {failing[:2]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    survived = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(run_one, MUTANTS):
            print(f"{res.split(' ')[0]:9} {name} | {res.split('| ', 1)[-1] if '|' in res else res}", flush=True)
            survived += res.startswith(("SURVIVED", "NOT APPLICABLE", "INVALID"))
    print(f"mutants={len(MUTANTS)} not_killed={survived}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
