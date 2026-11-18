"""Mutation check for the I-10 blind runner (tools/blind_runner.py) against tests/test_blind_runner.py.
Run from the live/ directory with pytest (and the suite's other dependencies) installed: python tests/mutation_check_blind_runner.py
Not part of test discovery.

Exit codes:  0 = every mutant KILLED by a real test failure;  1 = at least one SURVIVED;  2 = SCRIPT ERROR (never "killed"): the unmutated
runner does not pass its tests, a mutant's old text is not in the file, or a mutated run fails to import / collect.

PRIVACY mutants M1..M17 are Critic's 17 privacy mutants of e74e470 (review I-3), ported to the current source. M14 as Critic wrote it
(an ERROR-level log call in the error branch) is EQUIVALENT under ``logging.disable(CRITICAL)`` -- no test can see a log call the logging module
discards. It is kept below in EQUIVALENT exactly as written (it must SURVIVE, and the script says so); M14w is the same call together with the
weaker seal it needs to matter (``disable(WARNING)``), and that one is killed. Only M-mutants in ``M`` count towards killed/survived.
The rest (R-*) pin the review's B-R1..B-R7 and suggestions: they are mutants of the new columns, the output guard, and the input handling.
Every pytest run happens in a throw-away working directory that is deleted at the end: a mutant that writes a file into its cwd leaves nothing behind.
No mutant is run in parallel; the runner source is restored in a ``finally`` and compared byte for byte at the end."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

LIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SRC = os.path.join(LIVE, "tools", "blind_runner.py")
TESTS = [os.path.join(LIVE, "tests", "test_blind_runner.py")]
EXC = 'row["error"], row["error_stage"] = type(e).__name__, stage'
WRITE = "                f.write(json.dumps(row, ensure_ascii=False) + \"\\n\")"
RUNCALL = "                                    history_sink=run_history if shared_ledger else None)\n"

M = [
 # --- Critic's 17 privacy mutants (I-3) -------------------------------------------------------------------------------------------
 ("M1 remove logging.disable", "    logging.disable(logging.CRITICAL)\n", "    pass\n"),
 ("M2 logging.disable only up to WARNING", "    logging.disable(logging.CRITICAL)\n", "    logging.disable(logging.WARNING)\n"),
 ("M3 error=str(e)", EXC, 'row["error"], row["error_stage"] = str(e), stage'),
 ("M4 print(sentence) in the error branch", EXC, EXC + "\n        print(text)"),
 ("M5 memory not redirected", 'os.environ["SOPHON_LIVE_MEMORY_PATH"] = os.devnull', "pass"),
 ("M6 print sentence to stderr", WRITE, "                print(s.text, file=sys.stderr)\n" + WRITE),
 ("M7 sentence echoed in the row", RUNCALL, RUNCALL + '                row["text"] = s.text\n'),
 ("M8 sidecar debug file next to out", WRITE, "                open(out_path + '.dbg', 'a').write(s.text)\n" + WRITE),
 ("M9b warnings not ignored and warn(sentence)", 'warnings.simplefilter("ignore")', 'warnings.simplefilter("default")'),
 ("M10 logging switch not restored", "        logging.disable(prev_disable)\n", "        pass\n"),
 ("M11 write sentence to a cwd file", WRITE, "                open('trace.txt', 'a').write(s.text)\n" + WRITE),
 ("M12 device hook not removed", "            for k in _HOOK_ENV_NAMES:\n                os.environ.pop(k, None)\n", "            pass\n"),
 ("M13 extra env var left behind", "            yield\n", '            os.environ["LASTSENT"] = "x"\n            yield\n'),
 ("M14w ERROR-level log of the sentence in the error branch (with the seal at WARNING)", EXC, EXC + '\n        logging.getLogger().error("x %s", text)'),
 ("M15 repr(e) to stderr in the error branch", EXC, EXC + "\n        print(repr(e), file=sys.stderr)"),
 ("M16 device hook set instead of removed", "            for k in _HOOK_ENV_NAMES:\n                os.environ.pop(k, None)\n", "            for k in _HOOK_ENV_NAMES:\n                os.environ[k] = '/bin/true'\n"),
 ("M17 output opened for append instead of exclusive create", "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n\n\nasync def run", "os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)\n\n\nasync def run"),
 # --- B-R1 who decided / degraded -------------------------------------------------------------------------------------------------
 ("R1a route_via says jev for any JEV answer", 'return "jev" if jev_route in enums else "jev_invalid"', 'return "jev"'),
 ("R1b route_via ignores a missing JEV route (api.py:93)", "    if not jev_route:\n        return \"heuristic\"", "    if False:\n        return \"heuristic\""),
 ("R1c control_via says jev when the heuristic picked the target", '    return "heuristic"                          # no usable target', '    return "jev"                          # no usable target'),
 ("R1d control_via ignores the applied decision's own label", "        return res.control.via\n", '        return "jev"\n'),
 ("R1e degraded ignores control_via", 'or row["control_via"] == "heuristic"\n                                            or res.via', "\n                                            or res.via"),
 ("R1f degraded ignores an invalid JEV route", 'row["route_via"] in ("heuristic", "jev_invalid")', 'row["route_via"] == "heuristic"'),
 ("R1g degraded reason for a missing route dropped", 'reason = "no_usable_route"', "reason = None"),
 ("R1h control_no_target reason dropped", 'reason = "control_no_target"', "reason = None"),
 ("R1i guard counted as route_via heuristic", '    if res.via == "guard":\n        return "guard"', '    if res.via == "guard":\n        return "heuristic"'),
 # --- B-R7 degraded kinds + errors keep what was obtained -------------------------------------------------------------------------
 ("R7a timeout reported as conn", 'return "timeout"', 'return "conn"'),
 ("R7b http_status reported as conn", 'return "http_status"', 'return "conn"'),
 ("R7c conn reported as http_status", 'return "conn"', 'return "http_status"'),
 ("S2f bad_json not told from runtime_error", 'return "bad_json" if e.__cause__ is not None else "runtime_error"', 'return "runtime_error"'),
 ("S2g the message decides the kind", 'return "bad_json" if e.__cause__ is not None else "runtime_error"', 'return "bad_json" if "json" in str(e) else "runtime_error"'),
 ("S2h degraded_exc not recorded", 'row["degraded_reason"], row["degraded_exc"] = reason, rec.error_exc', 'row["degraded_reason"], row["degraded_exc"] = reason, None'),
 ("R7d the JEV error is swallowed by the spy (production behaviour changed)", "            rec.error_kind, rec.error_exc = jev_error_kind(e), type(e).__name__\n            raise\n", "            rec.error_kind, rec.error_exc = jev_error_kind(e), type(e).__name__\n            return {}\n"),
 ("R7e an error wipes the row", EXC, "row = _blank_row(sid, mode, []); " + EXC),
 ("R7f error_stage always admit", EXC, 'row["error"], row["error_stage"] = type(e).__name__, "admit"'),
 # --- B-R2 fidelity ---------------------------------------------------------------------------------------------------------------
 ("R2a history not passed to the worker", "history=list((history or [])[-HISTORY_TURNS:])", "history=[]"),
 ("R2b instructions not passed to the worker", "instructions=instructions, history=list(", 'instructions="", history=list('),
 ("R2c more than the last 6 turns", "HISTORY_TURNS = 6", "HISTORY_TURNS = 12"),
 ("R2d the rewritten task ignored", "delegate = task if task is not None else text", "delegate = text"),
 ("R2e raw_text is the rewrite", "raw_text=text), ledger)", "raw_text=delegate), ledger)"),
 ("R2f playback_active not passed", "playback_active=playback_active, raw_text", "playback_active=False, raw_text"),
 ("R2g history never accumulates with --shared-ledger", "            history_sink.append(", "            (lambda *_: None)("),
 ("R2i fidelity codes dropped", '"no_asr_transcript"]\n    codes.append("rewrite_from_input"', '"x"]\n    codes.append("rewrite_from_input"'),
 ("S1a playback_active value not recorded", 'str(bool(playback_active)).lower()', '"false"'),
 ("S1b language value not recorded", 'f"language={language},jev_state_language=zh"', '"language=zh,jev_state_language=zh"'),
 ("S1c run() does not pass the values to fidelity", "shared_ledger=shared_ledger, playback_active=playback_active, language=language)", "shared_ledger=shared_ledger)"),
 ("R2j rewrite not flagged", '"rewrite_from_input" if has_task else "no_frontend_rewrite"', '"no_frontend_rewrite"'),
 ("R2k fake_* codes dropped", 'codes += ["fake_jev", "fake_worker"]', "pass"),
 # --- B-R4 output guard -----------------------------------------------------------------------------------------------------------
 ("R4b an existing output not refused", "return any(os.path.lexists(p) for p in (out_path, out_path + \".done\"))", "return False"),
 ("R4c a stale .done not refused", '(out_path, out_path + ".done"))', "(out_path,))"),
 ("R4d output mode 0644", "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n\n\nasync def run", "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)\n\n\nasync def run"),
 ("R4e .done mode 0644", 'out_path + ".done", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)', 'out_path + ".done", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)'),
 ("R4f done marker says not complete", '"complete": True', '"complete": False'),
 ("R4g done marker written before the run finishes", "        with os.fdopen(fd, \"w\", encoding=\"utf-8\") as f:\n", "        open(out_path + '.done', 'w').write('{}')\n        with os.fdopen(fd, \"w\", encoding=\"utf-8\") as f:\n"),
 # --- B-R5 / B-R6 columns ---------------------------------------------------------------------------------------------------------
 ("R5 required_tools not recorded", 'row["required_tools"] = list(job.require_tools or [])', "pass"),
 ("R6 fake mode fills tools/tool_ok", 'if mode != "fake":', "if True:"),
 ("R6b mode not recorded", '"id": sid, "mode": mode,', '"id": sid, "mode": None,'),
 # --- S-2 input handling ----------------------------------------------------------------------------------------------------------
 ("S2a BOM not skipped", 'with open(path, encoding="utf-8-sig") as f:\n        for n, line in enumerate(f, 1):\n            line = line.strip()', 'with open(path, encoding="utf-8") as f:\n        for n, line in enumerate(f, 1):\n            line = line.strip()'),
 ("S2b text may be null/non-string", "if not isinstance(text, str) or not text.strip() or", "if False or"),
 ("S2c ids number file lines", "out.append(Sentence(f\"s{len(out) + 1:04d}\", line, None))", "out.append(Sentence(f\"s{n:04d}\", line, None))"),
 ("S2d --fake-route not validated", "if a.fake_route not in ROUTE_CRITERIA:", "if False:"),
 ("S2e history lines not validated", "if not isinstance(o, dict) or not isinstance(o.get(\"task\", \"\"), str) or not isinstance(o.get(\"answer\", \"\"), str):", "if False:"),
 # --- S-3 the no-judgment guard sees what it should --------------------------------------------------------------------------------
 ("S3a from re import in the runner", "import argparse\n", "import argparse\nfrom re import search\n"),
 ("S3b a comparison with the sentence", "    stage = \"admit\"\n", "    stage = \"admit\"\n    _ = \"x\" in text\n"),
 ("S3c a method call on the sentence", "    stage = \"admit\"\n", "    stage = \"admit\"\n    _ = text.find(\"x\")\n"),
 ("S3d a CJK literal", "    stage = \"admit\"\n", "    stage = \"admit\"\n    _ = \"天气\"\n"),
 # --- I-1 outcome columns / history like gateway.py -----------------------------------------------------------------------------
 ("I1a history appended on every outcome", 'if history_sink is not None and out.kind == "answer":', "if history_sink is not None:"),
 ("I1b history not capped", "del history_sink[:-HISTORY_KEEP]", "pass"),
 ("I1c history cap 20", "HISTORY_KEEP = 12", "HISTORY_KEEP = 20"),
 ("I1d outcome_kind not recorded", 'row["outcome_kind"], row["outcome_rejected"] = out.kind, out.rejected or None', 'row["outcome_kind"], row["outcome_rejected"] = None, None'),
 ("I1e outcome_rejected not recorded", 'out.kind, out.rejected or None', "out.kind, None"),
 # --- I-2 keyword-function conclusions --------------------------------------------------------------------------------------------
 ("I2a clarify_slots not recorded", "rec.slots = (rec.slots or []) + [list(r or [])]", "pass"),
 ("I2b save intent not recorded", "rec.save_intent = (rec.save_intent or []) + [bool(r)]", "pass"),
 ("I2c the wrapper changes the keyword function's answer", "            return r\n        return wrapper", "            return r if sink != 'slots' else []\n        return wrapper"),
 ("I2d the wrappers are not restored", "        for o, n, v in saved:\n            setattr(o, n, v)", "        pass"),
 ("I2e the memory_gate function is not wrapped", 'recording(memory_gate.has_explicit_save_intent, "save"))', "memory_gate.has_explicit_save_intent)"),
 ("I2f the wrapper records the ARGUMENT", "[bool(r)]", "[str(a)]"),
 ("I2g clarify_slots lost on the non-job path", '        row["clarify_slots"] = rec.slots\n        if res.job is None:', "        if res.job is None:"),
 # --- S-4 metamorphic: logic that forks on the sentence is found whatever the variables are called --------------------------------
 ("S4a forks on the sentence length (no comparison node)", 'row["control_via"] = _control_via(res, rec.route, rec.control)\n', 'row["control_via"] = _control_via(res, rec.route, rec.control)\n        if len(text) % 2:\n            row["control_via"] = "jev"\n'),
 ("S4b forks on the sentence through an alias", 'row["control_via"] = _control_via(res, rec.route, rec.control)\n', 'row["control_via"] = _control_via(res, rec.route, rec.control)\n        t = text\n        row["decision"] = "X" if ord(t[0]) % 2 else row["decision"]\n'),
 ("S4c route_via forks on the sentence", 'row["route_via"] = _route_via(res, rec.route, use_jev, set(ROUTE_CRITERIA))\n', 'row["route_via"] = _route_via(res, rec.route, use_jev, set(ROUTE_CRITERIA))\n        row["route_via"] = "heuristic" if sum(map(ord, text)) % 3 == 0 else row["route_via"]\n'),
 # --- S-5 -------------------------------------------------------------------------------------------------------------------------
 ("S5a a foreign .done is a traceback not exit 6", "            raise SystemExit(EXIT_DONE_FAILED)\n        with os.fdopen(done_fd", "            raise\n        with os.fdopen(done_fd"),
 ("S5b an unreadable input is a traceback", "    except OSError as e:\n        print(f\"blind_runner: cannot read", "    except ZeroDivisionError as e:\n        print(f\"blind_runner: cannot read"),
 ("S5c an unwritable output is a traceback", "    except OSError as e:                         # e.g.", "    except ZeroDivisionError as e:                         # e.g."),
 ("S5d the exit code for a lost race is not 5", "    except FileExistsError:                      # lost a race for the output file itself\n        print(\"blind_runner: refusing to write the output: it already exists (nothing was changed)\", file=sys.stderr)\n        return EXIT_OUT_REFUSED", "    except FileExistsError:                      # lost a race for the output file itself\n        return 0"),
 # --- I-1' Critic's bypasses of the AST guard (865bb78 re-check): found by the runtime probe and/or the metamorphic test ------------
 ("G1 alias then compare, no output effect (probe)", WRITE, "                t = s.text; _ = 'x' in t\n" + WRITE),
 ("G2 async helper(q).startswith, no output effect (probe)", WRITE, "                await _chk(s.text)\n" + WRITE),
 ("G8 len(s.text) alias compare (probe)", WRITE, "                n_ = len(s.text); _ = n_ > 5\n" + WRITE),
 ("G9 slice s.text[:3] (probe)", WRITE, "                _ = s.text[:3]\n" + WRITE),
 ("G10 getattr(s, 'text').find (probe)", WRITE, "                _ = getattr(s, 'text').find('x')\n" + WRITE),
 ("H1 alias: handled_by changes when the text contains x", WRITE, "                t = s.text\n                if 'x' in t: row['handled_by'] = 'fast'\n" + WRITE),
 ("H5 async helper(q) changes via", WRITE, "                if await _chk(s.text): row['via'] = 'guard'\n" + WRITE),
 ("H6 sys.modules['re'].search changes required_tools (C-level: only the metamorphic test sees it)", WRITE, "                if sys.modules['re'].search('a', s.text): row['required_tools'] = []\n" + WRITE),
 # --- S-2' invalid input exits 2 ------------------------------------------------------------------------------------------------------
 ("S2'a invalid input JSON is not mapped to exit 2", "        if isinstance(e.code, str):\n            print(\"blind_runner: \" + e.code, file=sys.stderr)\n            return EXIT_BAD_ARGS\n", "        if False:\n            return EXIT_BAD_ARGS\n"),
]
# Mutants that are equivalent to the unmutated runner BY CONSTRUCTION (the tests cannot, and must not, see them). They must survive.
EQUIVALENT = [
 ("M14 ERROR-level log of the sentence in the error branch (Critic's original: invisible under logging.disable(CRITICAL))", EXC, EXC + '\n        logging.getLogger().error("x %s", text)'),
]
# G2 / H5 need the helper that Critic's bypass used (a renamed parameter)
ANCHOR = "\nALLOW_REAL_ENV"
HELPER_G2 = "\nasync def _chk(q):\n    return q.startswith('x')\n"
HELPER_H5 = "\nasync def _chk(q):\n    return q.startswith('a')\n"
# M14 needs the weaker seal to be observable (see the module docstring)
COMBO = {"M14w ERROR-level log of the sentence in the error branch (with the seal at WARNING)": ("    logging.disable(logging.CRITICAL)\n", "    logging.disable(logging.WARNING)\n")}
# M9b needs the warning itself
EXTRA = {"M9b warnings not ignored and warn(sentence)": (WRITE, "                warnings.warn(s.text)\n" + WRITE),
         "G2 async helper(q).startswith, no output effect (probe)": (ANCHOR, HELPER_G2 + ANCHOR),
         "H5 async helper(q) changes via": (ANCHOR, HELPER_H5 + ANCHOR)}


WORK = tempfile.mkdtemp(prefix="blind_runner_mut_")      # the cwd of every pytest run; deleted at the end


def run_tests(env):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--maxfail=50", "-rfE", *TESTS], capture_output=True, text=True, env=env,
                       cwd=WORK, timeout=600)
    out = r.stdout + r.stderr
    def n(word):
        m = re.search(r"(\d+) " + word, out)
        return int(m.group(1)) if m else 0
    names = [l.split(" - ")[0].split("::", 1)[-1].strip() for l in out.splitlines() if l.startswith("FAILED ")]
    return r.returncode, n("passed"), n("failed"), n("error"), names, out


def script_error(msg):
    print("SCRIPT ERROR:", msg)
    sys.exit(2)


env = {k: v for k, v in os.environ.items() if not k.startswith("SOPHON_LIVE_")}
env.update(PYTHONPATH=LIVE, PYTHONDONTWRITEBYTECODE="1")
def clean_pycache():
    for _d, _dn, _fn in os.walk(LIVE):
        if os.path.basename(_d) == "__pycache__":
            shutil.rmtree(_d, ignore_errors=True)


clean_pycache()
rc, passed, failed, errors, _, out = run_tests(env)
if rc != 0 or failed or errors or passed == 0:
    script_error(f"baseline is not green on the unmutated runner (rc={rc} passed={passed} failed={failed} errors={errors})\n" + out[-1500:])
print(f"baseline ok: {passed} tests {TESTS}")

orig_src = open(SRC, encoding="utf-8").read()
def apply(old, new, name):
    src = orig_src
    pairs = [(old, new)] + ([COMBO[name]] if name in COMBO else []) + ([EXTRA[name]] if name in EXTRA else [])
    for o, nw in pairs:
        if o not in src:
            script_error("mutant not applicable (old text not found): " + name + " :: " + o[:60])
        src = src.replace(o, nw, 1)
    shutil.copy(SRC, SRC + ".orig")
    try:
        open(SRC, "w", encoding="utf-8").write(src)
        res = run_tests(env)
    finally:
        shutil.move(SRC + ".orig", SRC)
        clean_pycache()
    rc, passed, failed, errors, names, out = res
    if errors or (passed + failed) == 0:
        script_error(f"mutant '{name}' broke import/collection or collected no tests (errors={errors}, ran={passed + failed}); that is not a kill\n" + out[-1200:])
    return failed, names


for _name, _old, _new in M + EQUIVALENT:                  # every mutant must apply before any test run is spent
    for _o in [_old] + [x[0] for x in (COMBO.get(_name), EXTRA.get(_name)) if x]:
        if _o not in orig_src:
            script_error("mutant not applicable (old text not found): " + _name + " :: " + _o[:60])
killed_n = survived_n = 0
try:
    for name, old, new in M:
        failed, names = apply(old, new, name)
        if failed:
            killed_n += 1
            print("KILLED  ", name, "|", f"{failed} failing:", "; ".join(names[:2]))
        else:
            survived_n += 1
            print("SURVIVED", name)
    for name, old, new in EQUIVALENT:
        failed, names = apply(old, new, name)
        print("EQUIVALENT (expected to survive):", name, "->", "SURVIVED, as proven" if not failed else f"killed by {names[:2]} (the equivalence claim is wrong; investigate)")
finally:
    if open(SRC, encoding="utf-8").read() != orig_src:
        open(SRC, "w", encoding="utf-8").write(orig_src)
        print("SCRIPT ERROR: the runner source was not restored by a mutant run; it has been rewritten from the original text")
        sys.exit(2)
    shutil.rmtree(WORK, ignore_errors=True)
print(f"killed={killed_n} survived={survived_n} of {len(M)}")
sys.exit(1 if survived_n else 0)
