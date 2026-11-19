"""Mutation check for the model-driven coordinator (Round A / Critic M-1): run from live/ as ``python tests/mutation_coordinator.py [--jobs N]``.

Each mutant edits one production line in a temp copy and runs the unit suite (pytest, stop at the first failure); KILLED = some test fails for a
reason that is not an import / syntax error, SURVIVED = the suite stayed green (a test gap). Exit 0 only when every mutant is killed; 1 = a survivor;
2 = a script problem (mutant text not found, broken import, the unmutated copy not green) which is never counted as a kill.
The mutants stand for the ways a keyword rule or a guess could creep back into the intent layer.
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
A, G = "sophon_live/coordinator/api.py", "sophon_live/gateway.py"
JC = "sophon_live/coordinator/jev_client.py"
I, MG, CL, LG = "sophon_live/coordinator/intent.py", "sophon_live/coordinator/memory_gate.py", "sophon_live/coordinator/clarify.py", "sophon_live/coordinator/ledger.py"
K = '("天气" in text or "气温" in text)'

MUTANTS: list[tuple[str, str, str, str]] = [
    ("A2 keyword fallback when the model gave no tool answer (admit)", A,
     "                tools = intent.tools_from_answer(tool_choice)\n",
     "                tools = intent.tools_from_answer(tool_choice)\n                if tools is None and " + K + ": tools = ['get_weather']\n"),
    ("A5 tools_from_answer guesses on an illegal option", I,
     "    if choice is None or choice not in TOOL_CRITERIA:\n        return None\n    if choice == NONE:",
     "    if choice is None or choice not in TOOL_CRITERIA:\n        return ['web_search']\n    if choice == NONE:"),
    ("C25 tools_from_answer: memory_save no longer maps to memory_note", I, 'return ["memory_note"] if choice == MEMORY_SAVE else [choice]', "return [choice]"),
    ("P1 the filler is the lookup lead-in for every tool", I, '    return _PAD_BY_TOOL.get(tools[0], "")', '    return "我帮你查一下。"'),
    ("P2 the filler is spoken although the router could not judge", I, "    if not tools:\n        return \"\"", "    if tools == []:\n        return \"\""),
    ("RC2-8 invalid tool answers are not counted", A, "                if tool_choice is not None and tools is None:\n                    _invalid(", "                if False:\n                    _invalid("),
    ("U1 an unusable tool answer is recorded as a model judgment", A, '    judgment = "unknown" if tools is None else "model"', '    judgment = "model"'),
    ("M1 explicit save: an unknown answer writes long-term memory", MG, "ctx.explicit_save is not True", "ctx.explicit_save is False"),
    ("M2 explicit save: the gateway does not carry the model's judgment", G, ", ledger=ledger, explicit_save=job.explicit_save))", ", ledger=ledger))"),
    ("M3 explicit save: a model answer other than memory_save also counts", I, "    if choice is None or choice not in TOOL_CRITERIA:\n        return None\n    return choice == MEMORY_SAVE",
     "    if choice is None or choice not in TOOL_CRITERIA:\n        return None\n    return choice != NONE"),
    ("D1 router down + live task: a keyword decides instead of the fixed clarify", A, "    if ledger.active():\n", "    if False:\n"),
    ("D2 router down + no live task: nothing is handed to the orchestrator", A, "    job = _new_job(ledger, text, language=req.language, tools=tools, explicit_save=explicit_save)      # router down", "    job = None; return AdmitResult(speak=SpeakPlan(kind='noop', text=''), route='FAST', ledger=ledger)      # router down"),
    ("R1 a REFINE option is honoured only on the CONTROL route", A, '    if route in ("CONTROL", "WORKER"):', '    if route == "CONTROL":'),
    ("R2 CONTROL without a usable target does not ask", A, '        if control_key != "NEW":', "        if False:"),
    ("R3 a REFINE does not inherit the task's required tools", LG, "            require_tools=list(rec.require_tools),", "            require_tools=[],"),
    ("R4 a REFINE with a new model answer keeps the old tools", A, "    if tools:\n        new.require_tools, new.tool_judgment = list(tools), \"model\"", "    if False:\n        pass"),
    ("L1 the model is not shown the user's own words", A, '                state["user_said"] = req.raw_text.strip()', "                pass"),
    ("L2 the missing-slot question hard-codes the language", A, '{"utterance": text, **({"language": language} if language else {})}', '{"utterance": text, "language": "zh"}'),
    ("S1 the missing slot is guessed instead of read from the answer", CL, "    return [choice] if choice in _ASK else []", '    return ["city"]'),
    ("S2 an unusable missing-slot answer is not generic", CL, "    return [choice] if choice in _ASK else []", "    return [choice] if choice else []"),
    ("T1 the tool question is not asked", A, "            questions[intent.TOOL_KEY] = intent.build_tool_question()\n", "            pass\n"),
    # R-A' (Critic I-2): a wrongly shaped answer is unusable, counted, traced, and never takes the other answers down
    ("W1 choice_answer hands a non-string choice to the caller", JC, "    return c if isinstance(c, str) else None", "    return c"),
    ("W2 raw_choice raises on a wrongly shaped response", JC, '    answers = resp.get("answers") if isinstance(resp, dict) else None\n    ans = answers.get(key) if isinstance(answers, dict) else None\n    if isinstance(ans, dict) and ans.get("type") == "choice":\n        return ans.get("choice")',
     '    ans = (resp.get("answers") or {}).get(key) or {}\n    if ans.get("type") == "choice":\n        return ans.get("choice")'),
    ("W3 _parse_control_key accepts a non-string", A, "    if not isinstance(key, str) or not key or key == \"NEW\":", "    if not key or key == \"NEW\":"),
    ("W4 a missing / non-string answer to an asked question is not counted", A, "    if asked:\n        _invalid(question, raw, meta)", "    if False:\n        _invalid(question, raw, meta)"),
    ("W5 a control string that is not an offered key is not counted", A, '            _invalid("control", control_key, meta)', "            pass"),
    ("W6 a route string that is not an offered route is not counted", A, '                    _invalid("route", route, meta)', "                    pass"),
    ("W7 CONTROL without a target is a fixed clarify again (not the degraded branch)", A, '            return _degrade(text, req, ledger, meta, "control_without_target", tools=tools, explicit_save=explicit_save)',
     "            return AdmitResult(speak=SpeakPlan(kind=\"clarify\", text=DEGRADED_CLARIFY), route=\"CLARIFY\", ledger=ledger)"),
    ("W8 the degraded new task drops the model's tool judgment", A, "    job = _new_job(ledger, text, language=req.language, tools=tools, explicit_save=explicit_save)      # router down", "    job = _new_job(ledger, text, language=req.language, tools=None, explicit_save=explicit_save)      # router down"),
    ("W9 the control question is counted as missing even when it was not asked", A, 'asked="control" in questions', "asked=True"),
    ("W10 the turn is never marked as a router decision", A, '            if route in _ROUTES:\n                meta["via"] = "jev"\n', "            if route in _ROUTES:\n"),
    ("H1 /health?detail=1 forgets the router's invalid answers", G, '"router_invalid_answers": dict(_coord_api.JUDGE_STATS)', '"router_invalid_answers": {}'),
]


def run_one(m: tuple[str, str, str, str]) -> tuple[str, str]:
    name, rel, old, new = m
    d = Path(tempfile.mkdtemp(prefix="mut_coord_"))
    try:
        shutil.copytree(LIVE, d / "live", ignore=shutil.ignore_patterns("__pycache__", ".git"))
        p = d / "live" / rel
        s = p.read_text(encoding="utf-8")
        if old not in s:
            return name, "SCRIPT-ERROR | mutant not applicable (old text not found)"
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON")}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], cwd=d / "live", capture_output=True, text=True, env=env, timeout=900)
        out = r.stdout + r.stderr
        if r.returncode in (2, 3, 4, 5) or any(k in out for k in ("ERROR collecting", "SyntaxError", "ModuleNotFoundError", "ImportError")):
            return name, "SCRIPT-ERROR | the mutant broke import / collection (that is not a kill): " + out[-300:].replace("\n", " ")
        failing = [l for l in out.splitlines() if l.startswith("FAILED")]
        return name, ("SURVIVED | suite stayed green" if r.returncode == 0 else f"KILLED | {failing[0][7:120] if failing else 'failing'}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def baseline() -> bool:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON")}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=LIVE, capture_output=True, text=True, env=env, timeout=900)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if not baseline():
        print("SCRIPT ERROR: the unmutated suite is not green")
        return 2
    survived = errors = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(run_one, MUTANTS):
            print(f"{res.split(' ')[0]:12} {name} | {res.split('| ', 1)[-1]}", flush=True)
            survived += res.startswith("SURVIVED")
            errors += res.startswith("SCRIPT-ERROR")
    print(f"killed={len(MUTANTS) - survived - errors} survived={survived} script_errors={errors} of {len(MUTANTS)}")
    return 2 if errors else (1 if survived else 0)


if __name__ == "__main__":
    sys.exit(main())
