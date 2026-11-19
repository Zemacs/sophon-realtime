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
WRITE = "                f.write(_dump_row(row) + \"\\n\")"
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
 ("R1a degraded is never recorded as true", 'row["degraded"] = bool(use_jev and res.via == "degraded")', 'row["degraded"] = False'),
 ("R1b degraded ignores that the JEV was switched off (not an outage)", 'bool(use_jev and res.via == "degraded")', 'bool(res.via == "degraded")'),
 ("R1c control_via says jev for a decision that was not applied", "return res.control.via if res.control is not None else None", 'return res.control.via if res.control is not None else "jev"'),
 ("R1d control_via ignores the applied decision's own label", "return res.control.via if res.control is not None else None", 'return "jev" if res.control is not None else None'),
 ("R1e the degraded reason is derived, not production's", 'row["degraded_reason"] = (res.degraded_reason or None) if row["degraded"] else None', 'row["degraded_reason"] = "timeout" if row["degraded"] else None'),
 ("R1f a reason is recorded for a run that was not degraded", 'row["degraded_reason"] = (res.degraded_reason or None) if row["degraded"] else None', 'row["degraded_reason"] = res.degraded_reason or None'),
 ("R1g judge_invalid not recorded", 'row["judge_invalid"] = list(res.judge_invalid)', "pass"),
 ("R1h the tool judgment and explicit_save not recorded", 'row["tool_judgment"], row["explicit_save"] = job.tool_judgment, job.explicit_save', "pass"),
 ("R1k the JEV tool answer not recorded", 'row["jev_route"], row["jev_control"], row["jev_tool"] = rec.route, rec.control, rec.tool', 'row["jev_route"], row["jev_control"], row["jev_tool"] = rec.route, rec.control, None'),
 ("R1l a later JEV answer without a route blanks the recorded route", 'rec.route = jev_client.choice_answer(resp, "route") or rec.route', 'rec.route = jev_client.choice_answer(resp, "route")'),
 # --- B-R7 degraded kinds + errors keep what was obtained -------------------------------------------------------------------------
 ("R7a degraded_exc records the message instead of the class name", "rec.error_exc = type(e).__name__", "rec.error_exc = str(e)"),
 ("S2h degraded_exc not recorded", 'row["degraded_exc"] = rec.error_exc if row["degraded"] else None', 'row["degraded_exc"] = None'),
 ("R7d the JEV error is swallowed by the spy (production behaviour changed)", "            rec.error_exc = type(e).__name__\n            raise\n", "            rec.error_exc = type(e).__name__\n            return {}\n"),
 ("R7e an error wipes the row", EXC, "row = _blank_row(sid, mode, []); " + EXC),
 ("R7f error_stage always admit", EXC, 'row["error"], row["error_stage"] = type(e).__name__, "admit"'),
 # --- B-R2 fidelity ---------------------------------------------------------------------------------------------------------------
 ("R2a history not passed to the worker", "history=_worker_history(history or [])", "history=[]"),
 ("R2b instructions not passed to the worker", "instructions=instructions, history=_worker_history(", 'instructions="", history=_worker_history('),
 ("R2c more than the last 6 turns", "HISTORY_TURNS = 6", "HISTORY_TURNS = 12"),
 ("R2d the rewritten task ignored", "delegate = raw if task is None else reveal(task)", "delegate = raw"),
 ("R2e raw_text is the rewrite", "playback_active=playback_active, raw_text=raw)", "playback_active=playback_active, raw_text=delegate)"),
 ("R2f playback_active not passed", "playback_active=playback_active, raw_text=raw)", "playback_active=False, raw_text=raw)"),
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
 ("S2c ids number file lines", 'sid = f"s{len(out) + 1:04d}"\n                item = Sentence(sid, Opaque(line), None)', 'sid = f"s{n:04d}"\n                item = Sentence(sid, Opaque(line), None)'),
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
 ("I2a the tool answer not recorded", 'rec.tool = jev_client.choice_answer(resp, intent.TOOL_KEY) or rec.tool', "pass"),
 ("I2d the JEV is not restored after the sentence", "        for o, n, v in saved:\n            setattr(o, n, v)", "        pass"),
 ("I2e the spy changes the JEV's answer", "        return resp\n\n    patches", "        return {}\n\n    patches"),
 ("I2f the spy records the REQUEST instead of the answer", 'rec.control = jev_client.choice_answer(resp, "control") or rec.control', "rec.control = str(state)"),
 # --- S-4 metamorphic: logic that forks on the sentence is found whatever the variables are called --------------------------------
 ("S4a forks on the sentence length (no comparison node)", 'row["control_via"] = _control_via(res)\n', 'row["control_via"] = _control_via(res)\n        if len(text) % 2:\n            row["control_via"] = "jev"\n'),
 ("S4b forks on the sentence through an alias", 'row["control_via"] = _control_via(res)\n', 'row["control_via"] = _control_via(res)\n        t = text\n        row["decision"] = "X" if ord(t[0]) % 2 else row["decision"]\n'),
 ("S4c via forks on the sentence", 'row["route"], row["via"] = res.route, res.via\n', 'row["route"], row["via"] = res.route, res.via\n        row["via"] = "degraded" if sum(map(ord, text)) % 3 == 0 else row["via"]\n'),
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
 ("H6 sys.modules['re'].search changes required_tools (C-level: only the metamorphic test sees it)", WRITE, "                if sys.modules['re'].search('a', s.text): row['required_tools'] = ['x']\n" + WRITE),
 # --- S-2' invalid input exits 2 ------------------------------------------------------------------------------------------------------
 ("S2'a invalid input JSON is not mapped to exit 2", "        if isinstance(e.code, str):\n            print(\"blind_runner: \" + e.code, file=sys.stderr)\n            return EXIT_BAD_ARGS\n", "        if False:\n            return EXIT_BAD_ARGS\n"),
 # --- I-1'' the sentence is Opaque: Critic's surviving bypasses (and G3/G7) now crash or are caught, none can run ---------------------------------
 ("J3 sys.modules['re'].search on a multi-char word changes required_tools", WRITE, "                if sys.modules['re'].search('qzxw', s.text): row['required_tools'] = ['x']\n" + WRITE),
 ("J5 json.dumps(sentence) alias compare changes via", WRITE, "                d_ = json.dumps(s.text)\n                if 'qzxw' in d_: row['via'] = 'guard'\n" + WRITE),
 ("J6 bytes(sentence) alias compare changes via", WRITE, "                b_ = bytes(s.text, 'utf-8')\n                if b'qzxw' in b_: row['via'] = 'guard'\n" + WRITE),
 ("J9 module-level helper with unbound str.startswith changes via", WRITE, "                if _h(s.text): row['via'] = 'guard'\n" + WRITE),
 ("G3 unbound str.find(s.text, 'x') (no output effect)", WRITE, "                _ = str.find(s.text, 'x')\n" + WRITE),
 ("G7 sys.modules['re'].search (no output effect)", WRITE, "                _ = sys.modules['re'].search('x', s.text)\n" + WRITE),
 ("K1 reaching into the wrapper's slot", WRITE, "                if s.text._v.startswith('qzxw'): row['via'] = 'guard'\n" + WRITE),
 ("K2 reveal() called outside _admit_request (no effect)", WRITE, "                _ = reveal(s.text)\n" + WRITE),
 ("K3 the wrapper is iterable again", "    __iter__ = __len__ = __bool__", "    __len__ = __bool__"),
 ("K4 the wrapper can be compared again", "__getitem__ = __eq__ = __ne__", "__getitem__ = __ne__"),
 ("K5 str(wrapper) works again", "    __hash__ = __str__ = __format__", "    __hash__ = __format__"),
 ("K6 the wrapper's repr leaks the content", 'return "<Opaque sentence>"', 'return f"<Opaque {self._v}>"'),
 ("K7 reveal accepts anything", "    if type(o) is not Opaque:\n        raise TypeError(\"reveal() takes an Opaque value\")\n", ""),
 ("K8 the reader does not wrap a plain line", "item = Sentence(sid, Opaque(line), None)", "item = Sentence(sid, line, None)"),
 ("K9 the reader does not wrap a JSON text", "item = Sentence(sid, Opaque(text),", "item = Sentence(sid, text,"),
 ("U1 an invalid UTF-8 input file is not checked", '    _utf8_or_exit(path, "input file")\n', "    pass\n"),
 ("U2 an invalid UTF-8 history file is not checked", '    _utf8_or_exit(path, "history file")\n', "    pass\n"),
 ("U3 an invalid UTF-8 instructions file is not checked", '            _utf8_or_exit(a.instructions_file, "instructions file")\n', "            pass\n"),
 ("U4 the UTF-8 message line number is off by one", "count(b\"\\n\") + 1", "count(b\"\\n\")"),
 ("U5 the UTF-8 message names no byte offset", ", byte offset {e.start})", ")"),
 ("D1 duplicate ids are not detected", "            if sid in seen:\n", "            if False:\n"),
 ("D2 the duplicate-id message has no line number", 'raise SystemExit(f"input line {n}: id is not unique")', 'raise SystemExit("input ids are not unique")'),
 ("D3 the duplicate-id message echoes the id", 'raise SystemExit(f"input line {n}: id is not unique")', 'raise SystemExit(f"input line {n}: id {sid} is not unique")'),
 # --- I-1 (triple prime): production texts are Opaque at the line they are read (Critic's N1-N4 and the unwrapped variants) -----------------------
 ("N1 fork on job.objective", "        objective = Opaque(job.objective)", "        objective = Opaque(job.objective)\n        if 'qzxw' in job.objective:\n            row['handled_by'] = 'fast'"),
 ("N2 fork on job.objective at the history sink", 'history_sink.append({"task": objective, "answer": reply})', 'history_sink.append({"task": objective, "answer": reply})\n            if \'qzxw\' in job.objective:\n                row["via"] = "guard"'),
 ("N3 fork on out.text", "        reply = Opaque(out.text)", "        reply = Opaque(out.text)\n        if 'qzxw' in out.text:\n            row['handled_by'] = 'fast'"),
 ("N4 fork on res.speak.text", 'row["reply"] = Opaque(res.speak.text)', 'row["reply"] = Opaque(res.speak.text)\n            if \'qzxw\' in res.speak.text:\n                row["handled_by"] = "fast"'),
 ("N5 fork on the Opaque reply", "        reply = Opaque(out.text)", "        reply = Opaque(out.text)\n        if 'qzxw' in reply:\n            row['handled_by'] = 'fast'"),
 ("N6 job.objective is not wrapped", "        objective = Opaque(job.objective)", "        objective = job.objective"),
 ("N7 the clarify/fast reply is not wrapped", 'row["reply"] = Opaque(res.speak.text)', 'row["reply"] = res.speak.text'),
 ("N8 out.text is not wrapped", "        reply = Opaque(out.text)", "        reply = out.text"),
 ("N9 the history sink gets plain text", 'history_sink.append({"task": objective, "answer": reply})', 'history_sink.append({"task": job.objective, "answer": out.text})'),
 ("N10 the TaskContext reads job.objective itself", "utterance=reveal(objective)", "utterance=job.objective"),
 ("N11 the JSON row is dumped without the reply being revealed", 'return json.dumps({**row, "reply": reveal(row["reply"])}, ensure_ascii=False)', "return json.dumps(row, ensure_ascii=False)"),
 ("N12 the Worker's history is plain again (a stored sentence is read raw)", '        out.append({"task": reveal(h["task"]), "answer": reveal(h["answer"])})', '        out.append({"task": h["task"], "answer": h["answer"]})'),
 ("N14 fork on str(res.speak) (a production field that is not audited)", "        if res.job is None:\n", "        if 'qzxw' in str(res.speak):\n            row['handled_by'] = 'fast'\n        if res.job is None:\n"),
 ("N13 the history file's texts are not wrapped", 'out.append({"task": Opaque(o.get("task", "")), "answer": Opaque(o.get("answer", ""))})', 'out.append({"task": o.get("task", ""), "answer": o.get("answer", "")})'),
 # --- I-2 (triple prime): reveal checks its caller; the exit bodies are whitelisted (Critic's L1-L4, L6) ---------------------------------------
 ("L1 object.__getattribute__ takes the text out of the slot, a C-level search forks on it", WRITE, "                _v = object.__getattribute__(s.text, '_' + 'v')\n                if sys.modules['re'].search('qzxw', _v): row['via'] = 'guard'\n" + WRITE),
 ("L2 sys.modules[__name__].reveal(sentence) then a bytes() alias fork", WRITE, "                _v = sys.modules[__name__].reveal(s.text)\n                _b = bytes(_v, 'utf-8')\n                if b'qzxw' in _b: row['via'] = 'guard'\n" + WRITE),
 ("L3 a C-level search on the revealed text inside _admit_request rewrites it", "    raw = reveal(text)\n", "    raw = reveal(text)\n    if sys.modules['re'].search('qzxw', raw):\n        raw = 'x'\n"),
 ("L4 json.dumps alias compare on the revealed text inside _admit_request rewrites it", "    raw = reveal(text)\n", "    raw = reveal(text)\n    d_ = json.dumps(raw)\n    if 'qzxw' in d_:\n        raw = 'x'\n"),
 ("L6 the spy forks on the JEV request's utterance with a C-level search", "        rec.route = jev_client.choice_answer(resp, \"route\") or rec.route\n", "        if sys.modules['re'].search('qzxw', state['utterance']):\n            resp = {}\n        rec.route = jev_client.choice_answer(resp, \"route\") or rec.route\n"),
 ("L7 reveal does not check its caller", "        if fn.__code__ is caller:\n", "        if True:\n"),
 ("L8 reveal compares code objects by value (an identical clone passes)", "        if fn.__code__ is caller:\n", "        if fn.__code__ == caller:\n"),
 ("L9 reveal becomes an exit of its own", "_EXIT_FUNCTIONS = (_admit_request, _task_context, _worker_history, _dump_row)", "_EXIT_FUNCTIONS = (_admit_request, _task_context, _worker_history, _dump_row, reveal)"),
 ("L10 _task_context does a keyword test on the text", "    return C.TaskContext(", "    _y = reveal(objective).startswith('q')\n    return C.TaskContext("),
 ("L11 the spy compares what it forwards", "            resp = await jev(state, questions, **kw)\n", "            resp = await jev(state, questions, **kw)\n            if 'qzxw' in str(state):\n                resp = {}\n"),
 # --- I-3 (triple prime): the readers' stage -------------------------------------------------------------------------------------------------
 ("K12 read_sentences drops a line by its content", "            line = line.strip()\n", "            line = line.strip()\n            if 'qzxw' in line:\n                continue\n"),
 ("K13 read_sentences changes an id by the content", '                sid = f"s{len(out) + 1:04d}"\n                item', '                sid = f"s{len(out) + 1:04d}"\n                if \'qzxw\' in line:\n                    sid += "x"\n                item'),
 ("K14 the raw line is stashed in a module-level list and a fork reads it later", "            line = line.strip()\n", "            line = line.strip()\n            _SEEN.append(line)\n"),
 ("K15 read_history drops a line by its content", "            if not line.strip():\n                continue\n", "            if 'qzxw' in line:\n                continue\n            if not line.strip():\n                continue\n"),
 ("K16 read_sentences keeps an alias of the raw line", '                sid = f"s{len(out) + 1:04d}"\n                item', '                sid = f"s{len(out) + 1:04d}"\n                _alias = line\n                item'),
 ("K17 _utf8_or_exit looks at the raw bytes", "    try:\n        data.decode(\"utf-8-sig\")", "    if b'qzxw' in data:\n        raise SystemExit('x')\n    try:\n        data.decode(\"utf-8-sig\")"),
 # --- S-2 (triple prime): one way of counting lines ------------------------------------------------------------------------------------------
 ("S2c-1 the UTF-8 line number counts only LF again", 'data[:e.start].replace(b"\\r\\n", b"\\n").replace(b"\\r", b"\\n").count(b"\\n") + 1', 'data[:e.start].count(b"\\n") + 1'),
 ("S2c-2 the UTF-8 line number counts CRLF twice", 'data[:e.start].replace(b"\\r\\n", b"\\n").replace(b"\\r", b"\\n")', 'data[:e.start].replace(b"\\r", b"\\n")'),
]
# Mutants that are equivalent to the unmutated runner BY CONSTRUCTION (the tests cannot, and must not, see them). They must survive.
EQUIVALENT = [
 ("M14 ERROR-level log of the sentence in the error branch (Critic's original: invisible under logging.disable(CRITICAL))", EXC, EXC + '\n        logging.getLogger().error("x %s", text)'),
]
# G2 / H5 need the helper that Critic's bypass used (a renamed parameter)
ANCHOR = "\nALLOW_REAL_ENV"
HELPER_G2 = "\nasync def _chk(q):\n    return q.startswith('x')\n"
HELPER_J9 = "\n_h = lambda q: str.startswith(q, 'qzxw')\n"
HELPER_H5 = "\nasync def _chk(q):\n    return q.startswith('a')\n"
# M14 needs the weaker seal to be observable (see the module docstring)
COMBO = {"K14 the raw line is stashed in a module-level list and a fork reads it later": (WRITE, "                if _SEEN and 'qzxw' in _SEEN[0]: row['via'] = 'guard'\n" + WRITE),
         "M14w ERROR-level log of the sentence in the error branch (with the seal at WARNING)": ("    logging.disable(logging.CRITICAL)\n", "    logging.disable(logging.WARNING)\n")}
# M9b needs the warning itself
EXTRA = {"K14 the raw line is stashed in a module-level list and a fork reads it later": (ANCHOR, "\n_SEEN = []\n" + ANCHOR),
         "M9b warnings not ignored and warn(sentence)": (WRITE, "                warnings.warn(s.text)\n" + WRITE),
         "G2 async helper(q).startswith, no output effect (probe)": (ANCHOR, HELPER_G2 + ANCHOR),
         "H5 async helper(q) changes via": (ANCHOR, HELPER_H5 + ANCHOR),
         "J9 module-level helper with unbound str.startswith changes via": (ANCHOR, HELPER_J9 + ANCHOR)}


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
