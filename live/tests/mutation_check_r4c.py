"""Mutation check (rounds 4c-4e): applies each mutant to the source, runs the tests that must catch it, restores the file.
Run from the live/ directory with pytest installed: python tests/mutation_check_r4c.py   Not part of test discovery.

Exit codes:  0 = every mutant KILLED by a real test failure;  1 = at least one mutant SURVIVED;  2 = SCRIPT ERROR (never "killed"):
  a test group that does not pass cleanly (and collect >= 1 test) on the UNMUTATED source, a mutant whose old text is not in the file,
  or a mutated run that fails to import / collect (an import error or syntax error is not a test catching the mutant) or collects 0 tests.
A mutant counts as KILLED only when >= 1 collected test FAILED and no collection error occurred."""
import re, shutil, subprocess, sys, os
HARNESS_TIMEOUT_MARKER = "the page harness did not finish in"      # same text as tests/test_r4j_page_behaviour.py
ROOT = "."
PY = sys.executable
T_S19 = ["tests.test_r4_s19_inflight", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
T_S18 = ["tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1"]
T_D = ["tests.test_r4d_s23_s24", "tests.test_r4_news_evidence"]
T_E = ["tests.test_r4d_s23_s24", "tests.test_r4_s18_s21", "tests.test_no_fabrication", "tests.test_r4_review_b1", "tests.test_r4_search_evidence", "tests.test_r4_news_evidence"]
ORCH = "sophon_live/assistant_tools/orchestrator.py"
GR = "sophon_live/grounding.py"; RT = "sophon_live/realtime.py"; GW = "sophon_live/gateway.py"
T_G = ["tests.test_r4g_failure_hint", "tests.test_assistant_tools"]
T_H = ["tests.test_r4h_grounding_news"]
T_I = ["tests.test_r4i_demo_removed_retired_env"]
IDX = "sophon_live/static/index.html"
T_JF = ["tests.test_r4j_failure_signals"]; T_JG = ["tests.test_r4j_grounding_i23_i24"]; T_JP = ["tests.test_r4j_page_behaviour"]
T_KE = ["tests.test_r4k_evidence"]; T_KL = ["tests.test_r4l_critic_n_fixes"]; T_KS = ["tests.test_r4k_static_guards"]; T_KF = ["tests.test_r4k_failure_contract"]
T_JN = ["tests.test_assistant_tools_search", "tests.test_web_search_fallback"]
BK = "sophon_live/backend.py"; AD = "sophon_live/adapters.py"; WS_ = "sophon_live/assistant_tools/web_search.py"; WEATHER = "sophon_live/assistant_tools/weather.py"
FA = "failed = fail_code is not None"
T_F = ["tests.test_r4e_b4_i18", "tests.test_orch_worker", "tests.test_orchestrator"]
NEWS = "sophon_live/assistant_tools/news.py"; WS = "sophon_live/assistant_tools/web_search.py"; WX = "sophon_live/assistant_tools/weather.py"
T_R1 = ["tests.test_model_client"]
T_GB = ["tests.test_r4m_g1_literal_only"]
INF = "sophon_live/assistant_tools/inflight.py"; SCH = "sophon_live/coordinator/schemas.py"
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
  '    return not any(appears_in_words(v, seen, translate=True) for v in (city, bare) if v) and not any(c in seen for c in ("北京", "上海", "广州", "深圳"))', T_F),
 ("B-4: _ungrounded_city never fires", ORCH, '    return not any(appears_in_words(v, seen, translate=True) for v in (city, bare) if v)', "    return False", T_F),
 ("B-4: _ungrounded_city drops the alias translation", ORCH, "appears_in_words(v, seen, translate=True)", "appears_in_words(v, seen)", T_F),
 ("I-18: the speak-now nudge reads the tool text again", ORCH, "succeeded = not failed and tool_meta.get(failcodes.SUCCESS_KEY) is True",
  'succeeded = not failed and ("第1条" in out or "度" in out or "Headlines on" in out)', T_F),
 ("I-18: the nudge ignores the structured success report", ORCH, "succeeded = not failed and tool_meta.get(failcodes.SUCCESS_KEY) is True", "succeeded = False", T_F),
 ("I-18: news does not report success", NEWS, "    toolmeta.note(**{failcodes.SUCCESS_KEY: True}, count=len(picked))", "    pass", T_F),
 ("I-18: weather reports success even without temperatures", WX, "True if (not meta.get(\"failure\") and meta.get(\"tmin\") is not None and meta.get(\"tmax\") is not None) else None", "True", T_F),
 # Round 4g (failure_hint structure / legacy builtin path deleted)
 ("I-20: failure_hint fires for every output", GR, '    if not failed:\n        return ""', '    if False:\n        return ""', T_G),
 ("I-20: output_failed reads the output's words again", GR, '    if item.get("error") or item.get("status") in ("failed", "incomplete"):\n        return True',
  '    if item.get("error") or item.get("status") in ("failed", "incomplete") or re.search(r"not found|no user|invalid|error", str(item.get("output", "")), re.I):\n        return True', T_G),
 ("I-20: output_failed ignores a JSON ok:false", GR, 'return out.get("ok") is False or out.get("success") is False or bool(out.get("error"))', 'return False', T_G),
 ("I-20: realtime marks every output failed", RT, 'grounding.output_failed(item)))', 'True))', T_G),
 ("I-20: realtime ignores the per-item flag", RT, "grounding.failure_hint(failed, self.heard)", "grounding.failure_hint(False, self.heard)", T_G),
 ("legacy: the builtin backend is announced again", GW, '        deleg_body = {"id": did, "type": "delegation", "target": "client", "task": what}\n',
  '        deleg_body = {"id": did, "type": "delegation", "target": "client", "task": what}\n        if os.environ.get("SOPHON_LIVE_BUILTIN_TOOLS"):\n            deleg_body["backend"] = "builtin"\n', T_G),
 ("legacy: run_task is back in the runner", "sophon_live/assistant_tools/runner.py", "async def run_tool(", "async def run_task(task):\n    return None\n\n\nasync def run_tool(", T_G),
 ("S-24: news says 'feeds unreachable' when the cap refused", NEWS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: search says 'network' when the cap refused", WS, "        if busy:\n            toolmeta.note(failure=failcodes.BUSY)", "        if False:\n            toolmeta.note(failure=failcodes.BUSY)", T_D),
 ("S-24: weather says 'cannot find' when the cap refused", WX, "        if busy:\n            meta[\"failure\"] = failcodes.BUSY", "        if False:\n            meta[\"failure\"] = failcodes.BUSY", T_D),
 ("S-24: busy sentence not whitelisted for web_search", SCH, '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",\n        BUSY_ZH_RE, BUSY_EN_RE,', '    "web_search": tuple(re.compile(p) for p in (\n        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",', T_D),
 # Round 4h (I-20 grounding provenance only / news topic is a schema parameter)
 ("I-20: a free-text key is exempt from grounding again", GR, "        vn = norm_text(v)\n        if not vn:", '        if k.lower() in {"summary", "reason", "query", "note", "notes", "message", "description", "comment", "content"}:\n            continue\n        vn = norm_text(v)\n        if not vn:', T_H),
 ("I-20: placeholder-looking values are blocked even when the caller said them", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])",
  '        if vn in {"12345", "johndoe", "john", "doe", "example.com", "5551234567"}:\n            return False\n        if re.fullmatch(r"(?=.*[A-Za-z])', T_H),
 ("I-20: placeholder-looking values are allowed even when the caller never said them", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])",
  '        if vn in {"12345", "johndoe", "john", "doe", "example.com", "testerexample.com"}:\n            return True\n        if re.fullmatch(r"(?=.*[A-Za-z])', T_H),
 ("I-20: grounding accepts everything (provenance not enforced)", GR, "    def literal(v: str, vn: str) -> bool:\n", "    def literal(v: str, vn: str) -> bool:\n        return bool(vn)\n", T_H),
 ("news: a keyword filter decides which headlines are kept again", NEWS, "                titles = _titles_from_rss(await fetch(fb))\n",
  "                titles = [t for t in _titles_from_rss(await fetch(fb)) if topic in t or '科技' in t]\n", T_H),
 ("news: a category falls back to another category's feed", NEWS, '    "tech": ("http://www.people.com.cn/rss/it.xml",),', '    "tech": ("http://www.people.com.cn/rss/it.xml", "http://www.people.com.cn/rss/finance.xml"),', T_H),
 ("news: a value outside the enum is accepted as a topic", NEWS, "    if topic not in CATEGORIES:", "    if not topic:", T_H),
 ("news: the search fallback uses the raw topic, not the category label", NEWS, "_rss_url(label, lang)", "_rss_url(topic, lang)", T_H),
 ("orchestrator: a required enum value outside the enum is accepted", ORCH, "                if not v or (enum and v not in enum):", "                if not v:", T_H),
 ("orchestrator: the job language no longer fills get_news lang", ORCH, 'args.get("lang") not in ("zh", "en") and job.language in ("zh", "en")', "False", T_H),
 ("schema: get_news topic is a free string again", "sophon_live/assistant_tools/registry.py", '"topic": {"type": "string", "enum": ["tech", "finance", "world", "general"], ', '"topic": {"type": "string", ', T_H),
  # Round 4i (I-22 demo backend deleted / retired env var warning)
 ("I-22: the page short-circuits orch only (coord gets the demo path again)", IDX, 'd.backend === "orch" || d.backend === "coord"', 'd.backend === "orch"', T_I),
 ("I-22: the demo backend code is back in the page", IDX, "async function onDelegation(d) {", "async function demoBackend(task) { return null; }\nasync function onDelegation(d) {", T_I),
 ("I-22: the page answers a delegation by itself after a delay", IDX, 'setR("等你手动回答…");', 'setR("等你手动回答…"); setTimeout(() => inject(d.id, "demo", "commentary", setR), 800);', T_I),
 ("I-22: the page fetches from a weather API again", IDX, "async function onDelegation(d) {", "async function wx() { return await fetch('https://api.open-meteo.com'); }\nasync function onDelegation(d) {", T_I),
 ("retired env: main no longer runs the startup check", GW, "    warn_retired_env()\n", "", T_I),
 ("retired env: the warning is never logged", GW, "        log.warning(\"%s is set but", "        log.debug(\"%s is set but", T_I),
 ("retired env: the warning echoes the value", GW, 'remove it from the environment", n)', 'remove it from the environment" + os.environ[n], n)', T_I),
 ("retired env: an empty value is treated as unset", GW, "hit = [n for n in RETIRED_ENV_VARS if n in os.environ]", "hit = [n for n in RETIRED_ENV_VARS if os.environ.get(n)]", T_I),
  # Round 4j (B-5 text judgments removed / I-23 provenance for CJK + long values / I-24 guard never releases / I-25 failure survivors / S-39 page behaviour)
 ("B-5: backend counts a failure when the output starts with 'error' again", BK, 'grounding.output_failed({"output": out}):', 'grounding.output_failed({"output": out}) or out.lower().startswith("error") or "not found" in out.lower()[:200]:', T_JF),
 ("B-5: backend stops only get_/find_/search_ tools again", BK, "if self.fails.get(name, 0) >= MAX_FAILS:", 'if self.fails.get(name, 0) >= MAX_FAILS and name.startswith(("get_", "find_", "search_")):', T_JF),
 ("B-5: backend never counts a failure", BK, "self.fails[name] = self.fails.get(name, 0) + 1", "pass", T_JF),
 ("B-5: realtime closes the session when the message says 'closed'", RT, 'if e.get("fatal") is True:', 'if "closed" in (e.get("message") or ""):', T_JF),
 ("B-5: gateway closes the session when the message says 'closed'", GW, 'if e.get("fatal") is True:', 'if "closed" in (e.get("message") or ""):', T_JF),
 ("B-5: realtime closes on any truthy fatal", RT, 'if e.get("fatal") is True:', 'if e.get("fatal"):', T_JF),
 ("B-5: the adapter no longer flags a lost connection as fatal", AD, '"message": "engine connection closed", "fatal": True}', '"message": "engine connection closed"}', T_JF),
 ("B-5: the timeout envelope is prose again", RT, 'out = json.dumps({"ok": False, "error": "no_result_from_system"})', 'out = "Error: no result from the system"', T_JF),
 ("B-5: news guesses the publisher from segment shape again", NEWS, "    return title          # no <source> to go by", '    return title.rsplit(" - ", 1)[0] if " - " in title else title          # no <source> to go by', T_JN),
 ("B-5: web_search strips a Chinese relative-time prefix again", WS_, 'snippet = _text(p.group(1)) if p else ""', 'snippet = re.sub(r"^\\s*\\d+\\s*天前\\s*[·・]\\s*", "", _text(p.group(1))) if p else ""', T_JN),
 ("I-24: the guard releases after 0 rejections", RT, "GUARD_MAX_REJECTIONS = 2", "GUARD_MAX_REJECTIONS = 0", T_JG),
 ("I-24: the guard escalates only after 20 rejections", RT, "GUARD_MAX_REJECTIONS = 2", "GUARD_MAX_REJECTIONS = 20", T_JG),
 ("I-24: after the threshold the call is released to the tool", RT, "            return True\n        msg = grounding.guard_message(bad, self.heard)", "            return False\n        msg = grounding.guard_message(bad, self.heard)", T_JG),
 ("I-24: after the threshold there is no needs_input (same message again)", RT, "if capped or n > GUARD_MAX_REJECTIONS:", "if False:", T_JG),
 ("I-24: needs_input is prose, not a structured failure", RT, 'json.dumps({"ok": False, "status": "needs_input", "message": grounding.needs_input_message(bad, give_up=capped)})', "grounding.needs_input_message(bad, give_up=capped)", T_JG),
 ("I-25: orchestrator adds a NEW failure word (失败)", ORCH, FA, FA + ' or "失败" in out', T_JF),
 ("I-25: orchestrator lowercase English probe", ORCH, FA, FA + ' or "unavail" in out.lower()', T_JF),
 ("I-25: orchestrator regex probe", ORCH, FA, FA + ' or __import__("re").search("暂时|无法", out) is not None', T_JF),
 ("I-25: orchestrator any() over a word tuple", ORCH, FA, FA + ' or any(k in out for k in ("连不上", "查不到"))', T_JF),
 ("I-25: orchestrator raw substring '\"ok\": false'", ORCH, FA, FA + " or '\"ok\": false' in out", T_JF),
 ("I-25: any truthy 'failure' key in JSON counts", ORCH, 'if isinstance(obj, dict) and obj.get("ok") is False:', 'if isinstance(obj, dict) and (obj.get("ok") is False or obj.get("failure")):', T_JF),
 ("I-25: web_search judged a failure by its sentence", ORCH, FA, FA + ' or (name == "web_search" and "没有搜到" in out)', T_JF),
 ("I-25: weather error code read from str(e)", "sophon_live/assistant_tools/weather.py", '    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):      # before OSError: TimeoutError is an OSError', '    if "timed out" in str(e):\n        return "timeout"\n    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):', T_JF),
 ("S-39: page sends from a Promise.then (direct send)", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { Promise.resolve().then(() => send({type:"session.commentary.append",content:"x",delegation_id:d.id}));', T_JP),
 ("S-39: page injects through an aliased setInterval", IDX, 'setR("等你手动回答…");', 'setR("等你手动回答…"); const f = inject; const iv = setInterval(() => { f(d.id, "x", "commentary", setR); clearInterval(iv); }, 300);', T_JP),
 ("S-39: page makes a network call under another name", IDX, "async function onDelegation(d) {", "async function onDelegation(d) { globalThis['fe' + 'tch']('https://example.invalid/w');", T_JP),
 ("S-39: coord gets the manual box / demo path (node behaviour)", IDX, 'd.backend === "orch" || d.backend === "coord"', 'd.backend === "orch"', T_JP),
  # Round 4k (R-1 normaliser / R-2 evidence span + read-back / R-3 Critic's mutants / R-4 item structure / R-5 degraded path, cap, cleanup / R-6, R-7)
 ("N-1: a Chinese numeral table is back in the normaliser (一 -> 1, 四 -> 4)", GR, 'return re.sub(r"[^\\w@.]", "", unicodedata.normalize("NFKC", str(s)).lower())', 'return re.sub(r"[^\\w@.]", "", unicodedata.normalize("NFKC", str(s)).lower().replace("一", "1").replace("四", "4"))', T_KE + T_KL),
 ("N-1: the literal path converts 十分 / 万一 specially", GR, "    def literal(v: str, vn: str) -> bool:\n", "    def literal(v: str, vn: str) -> bool:\n        if vn in (\"十分\", \"万一\"): return True\n", T_KE + T_KL),
 ("R-1: a number may be a piece of a longer number", GR, "        if not ((value_n[0].isdigit() and i > 0 and corpus_n[i - 1].isdigit()) or (value_n[-1].isdigit() and j < len(corpus_n) and corpus_n[j].isdigit())):", "        if True:", T_KE),
 ("R-1: turns are joined without a separator (a match can straddle two turns)", GR, 'return "|".join(norm_text(x) for x in turns)', 'return "".join(norm_text(x) for x in turns)', T_KE),
 ("R-2: a span needs no occurrence in what was said", GR, "            if occurs(span_n, corpus):", "            if True:", T_KE),
 ("R-2: the model need not state the value comes from the span", GR, 'if en.get("value_comes_from_span") is not True:', "if False:", T_KE),
 ("R-2: any truthy value_comes_from_span counts", GR, 'if en.get("value_comes_from_span") is not True:', 'if not en.get("value_comes_from_span"):', T_KE),
 ("R-2: evidence for one argument is accepted for any argument", GR, "entries = evidence.get(k) or []", "entries = [x for v_ in evidence.values() for x in v_]", T_KE),
 ("R-2: the span is compared without normalisation", GR, 'span_n = norm_text(span)', 'span_n = span', T_KE),
 ("R-2: a read-back is accepted without the value ever being said aloud", GR, "                if j < 0:", "                if False:", T_KE),
 ("R-2: a read-back confirmation may be older than the read-back", GR, "if not occurs(span_n, norm_text(heard[first])):", "if not occurs(span_n, _turns_n(heard)):", T_KE + T_KL),
 ("R-2: the read-back flag is ignored (plain span path only)", GR, 'if en.get("read_back_to_caller") is True:', "if False:", T_KE),
 ("R-2: the evidence argument is not stripped before the client sees the call", RT, "            e[\"arguments\"] = args                                 # the client's tool never sees the reserved argument", "            pass", T_KE),
 ("R-2: the engine is not shown the evidence field", RT, "tools = grounding.with_evidence_field(tools)        #", "tools = tools        #", T_KE),
 ("R-2: the prompt does not explain the evidence rule", RT, "                rule += grounding.EVIDENCE_RULES_ZH if self.lang == \"zh\" else grounding.EVIDENCE_RULES_EN", "                pass", T_KE),
 ("R-2: model-judged acceptances are not audited", RT, '                self._audit(dict(d, call=e.get("name")))', "                pass", T_KE),
 ("R-2: the refusal does not mention the evidence path", GR, "short paraphrase), call again and list it in {EVIDENCE_KEY}", "short paraphrase), call again", T_KE),
 ("R-2: needs_input does not ask for a read-back", GR, 'Read each value back to the caller', 'Say something to the caller', T_KE),
 ("R-2: the assistant's spoken text is not tracked for read-backs", RT, 'if r.get("text"): self.spoken.append((r["text"], len(self.heard)))', "pass", T_KE),
 ("R-4: the item-level failure flag is dropped on the delegation path", RT, "fut.set_result(grounding.tool_text(item))", 'fut.set_result(str(item.get("output", "")))', T_KE),
 ("R-4: the back office ignores the item-level failure flag", BK, 'if getattr(out, "failed", False) or grounding.output_failed({"output": out}):', 'if grounding.output_failed({"output": out}):', T_KE),
 ("R-4: tool_text never marks failure", GR, "    s.failed = output_failed(item)", "    s.failed = False", T_KE),
 ("R-5: no warning on the degraded path", RT, "            if not self.degraded_warned:", "            if False:", T_KE),
 ("R-5: the degraded path is not recorded", RT, '            self._audit({"call": e.get("name"), "grounding": "unavailable", "reason": reason})', "            pass", T_KE),
 ("R-5: no per-session cap on rejections", RT, "capped = self.guard_total > GUARD_SESSION_MAX_REJECTIONS", "capped = False", T_KE),
 ("R-5: the per-session cap is 1000", RT, "GUARD_SESSION_MAX_REJECTIONS = 12", "GUARD_SESSION_MAX_REJECTIONS = 1000", T_KE),
 ("R-5: the counter table is not cleared when the session ends", RT, 'self.guard_counts.clear(); self.audit.clear(); self.spoken.clear(); getattr(self, "rejected", {}).clear()', "pass", T_KE),
 ("N-4: fuzzy matching of Latin words is back (0.85)", GR, "        return occurs(vn, corpus) or appears_in_words(v, said)", "        import difflib\n        if vn.isascii() and not any(c.isdigit() for c in vn) and max((difflib.SequenceMatcher(None, vn, w).ratio() for w in said.lower().split()), default=0.0) >= 0.85:\n            return True\n        return occurs(vn, corpus) or appears_in_words(v, said)", T_KL),
 ("R-7: the instructions count as provenance again", RT, 'bad = await grounding.check_async(e.get("name", ""), args, self.heard, self.seen,', 'bad = await grounding.check_async(e.get("name", ""), args, self.heard, self.seen + [self.cfg.get("instructions") or ""],', T_KE),
 ("R-7: the tool schema counts as provenance again", RT, 'bad = await grounding.check_async(e.get("name", ""), args, self.heard, self.seen,', 'bad = await grounding.check_async(e.get("name", ""), args, self.heard, self.seen + [json.dumps(self.cfg.get("tools") or [])],', T_KE),
 ("R-3: needs_input is not spoken", RT, "grounding.needs_input_message(bad, give_up=capped)}), speak=True)", "grounding.needs_input_message(bad, give_up=capped)}), speak=False)", T_KE),
 ("R-3: the guard key ignores the arguments", RT, 'key = e.get("name", "") + json.dumps(args, sort_keys=True)', 'key = e.get("name", "")', T_KE + T_JG),
 ("R-3: web_search drops snippets by a regex (广告|推广)", WS_, 'snippet = _text(p.group(1)) if p else ""', 'snippet = _text(p.group(1)) if p else ""\n        if re.search("广告|推广", snippet): continue', T_KS),
 ("R-3: web_search drops snippets by an in-test (no re)", WS_, 'snippet = _text(p.group(1)) if p else ""', 'snippet = _text(p.group(1)) if p else ""\n        if "广告" in snippet: continue', T_KS),
 ("R-3: web_search age prefix stripped by split on the dot", WS_, 'snippet = _text(p.group(1)) if p else ""', 'snippet = _text(p.group(1)) if p else ""\n        snippet = snippet.split("·", 1)[1].strip() if "天前 ·" in snippet else snippet', T_JN + T_KS),
 ("R-3: a word probe inside _tool_failure_text", ORCH, '    t = (out or "").strip()\n    if not t or t.startswith(("{", "["))', '    t = (out or "").strip()\n    if "error" in t.lower(): return NO_TOOL_ANSWER\n    if not t or t.startswith(("{", "["))', T_KS),
 ("R-3: toolmeta detail read as text in _tool_failure_code", ORCH, '    code = (tool_meta or {}).get("failure")', '    code = (tool_meta or {}).get("failure") or ("error" if "不存在" in str((tool_meta or {}).get("detail", "")) else None)', T_KS + T_JF),
 ("R-3: failure read through msgs[-1]", ORCH, "                fail_code = _tool_failure_code(out, tool_meta)", '                fail_code = _tool_failure_code(out, tool_meta)\n                if fail_code is None and "失败" in msgs[-1]["content"]: fail_code = "error"', T_JF + T_KS),
 ("R-3: failure by str(obj) probe on the parsed JSON", ORCH, 'if isinstance(obj, dict) and obj.get("ok") is False:', 'if isinstance(obj, dict) and (obj.get("ok") is False or "fail" in str(obj)):', T_JF + T_KS),
 ("R-3: failure by json.dumps(obj) probe", ORCH, 'if isinstance(obj, dict) and obj.get("ok") is False:', 'if isinstance(obj, dict) and (obj.get("ok") is False or "fail" in json.dumps(obj)):', T_JF + T_KS),
 ("R-3: failure by the length of the output", ORCH, "                fail_code = _tool_failure_code(out, tool_meta)", "                fail_code = _tool_failure_code(out, tool_meta) or (\"error\" if len(out) < 3 else None)", T_JF + T_KS),
 ("R-3: failure by the first word of the output", ORCH, '    if t.startswith("{"):', '    if t.split(" ")[0] == "Error:": return "error"\n    if t.startswith("{"):', T_JF + T_KS),
 ("R-3: output_failed accepts status 'error'", GR, '("failed", "incomplete")', '("failed", "incomplete", "error")', T_KF),
 ("R-3: output_failed treats a falsy ok as failure", GR, 'out.get("ok") is False', 'not out.get("ok", True)', T_KF),
 ("R-3: backend counts a failure from the first five characters", BK, 'if getattr(out, "failed", False) or grounding.output_failed({"output": out}):', 'if getattr(out, "failed", False) or grounding.output_failed({"output": out}) or out[:5] == "Error":', T_JF),
 ("R-3: page: a keyword demo only for stock tasks", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { if (/股票/.test(d.task||"")) send({type:"session.commentary.append",content:"demo",delegation_id:d.id});', T_JP),
 ("R-3: page: a demo only for news tasks on the coord backend", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { if ((d.task||"").includes("新闻")) send({type:"session.commentary.append",content:"demo",delegation_id:d.id});', T_JP),
 ("R-3: page: a demo timer at 90 seconds", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { setTimeout(() => send({type:"session.commentary.append",content:"demo",delegation_id:d.id}), 90000);', T_JP),
 ("R-3: page: a second WebSocket sends", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { new WebSocket("ws://x/y").send(JSON.stringify({type:"x"}));', T_JP),
 ("R-3: page: a demo on page load", IDX, "async function onDelegation(d) {", 'window.addEventListener("load", () => { send({type:"session.commentary.append",content:"demo"}); });\nasync function onDelegation(d) {', T_JP),
 ("R-3: page: a demo only when the task is empty", IDX, "async function onDelegation(d) {", 'async function onDelegation(d) { if (!d.task) send({type:"session.commentary.append",content:"demo",delegation_id:d.id});', T_JP),

 # Critic's 7440dc9 review N-1..N-9 (round 4l)
 ("N-2: a span is accepted without asking the independent judge", GR, 'ans = verdict(JUDGE_EQUIVALENT, k, v, span, "")\n                if ans is not True:', 'ans = True\n                if ans is not True:', T_KE + T_KL),
 ("N-2: an unavailable judge (None) counts as yes", GR, 'ans = verdict(JUDGE_EQUIVALENT, k, v, span, "")\n                if ans is not True:', 'ans = verdict(JUDGE_EQUIVALENT, k, v, span, "")\n                if ans is False:', T_KE + T_KL),
 ("N-2: the judge is asked about the wrong claim (span and value swapped)", GR, 'ans = verdict(JUDGE_EQUIVALENT, k, v, span, "")', 'ans = verdict(JUDGE_EQUIVALENT, k, span, v, "")', T_KL),
 ("N-2: check_async pass 1 collects nothing", GR, "bad = check(name, args, heard, seen, evidence, spoken, None, pending=pending,", "bad = check(name, args, heard, seen, evidence, spoken, None, pending=None,", T_KL),
 ("N-2: check_async counts a judge exception as yes", GR, "table[(p[\"kind\"], p[\"argument\"], p[\"value\"], p[\"span\"])] = r if isinstance(r, bool) else None", "table[(p[\"kind\"], p[\"argument\"], p[\"value\"], p[\"span\"])] = r if isinstance(r, bool) else True", T_KL),
 ("N-2: the session never asks the second model by default", RT, 'judge=getattr(self, "judge", None) or span_judge.judge,', 'judge=getattr(self, "judge", None) or (lambda *a: asyncio.sleep(0, True)),', T_KL),
 ("N-2: span_judge maps NO to yes", "sophon_live/coordinator/span_judge.py", '        STATS["no"] += 1\n        return False', '        STATS["no"] += 1\n        return True', T_KL),
 ("N-2: span_judge treats anything but NO as yes", "sophon_live/coordinator/span_judge.py", 'if ans == "YES":', 'if ans != "NO":', T_KL),
 ("N-2: span_judge counts an exception as yes", "sophon_live/coordinator/span_judge.py", '        log.warning("span check unavailable (%s): %r", kind, e)\n        return None', '        log.warning("span check unavailable (%s): %r", kind, e)\n        return True', T_KL),
 ("N-3: the caller_reply enum is not checked", GR, 'if en.get("caller_reply") != "confirm":', "if False:", T_KL),
 ("N-3: the read-back span may be any later turn", GR, "if not occurs(span_n, norm_text(heard[first])):", "if not occurs(span_n, _turns_n(heard[first:])):", T_KL),
 ("N-3: a read-back may confirm a value that was never refused", GR, "if rejected is not None and v not in rejected.get(k, ()):", "if False:", T_KL),
 ("N-3: the independent check of the confirmation is skipped", GR, 'ans = verdict(JUDGE_CONFIRMS, k, v, span, spoken[j][0])\n                if ans is not True:', 'ans = True\n                if ans is not True:', T_KL),
 ("N-3: silence after the read-back is not noticed", GR, "                if first >= len(heard):", "                if first > len(heard) + 99:", T_KL),
 ("N-3: a refused value is not remembered", RT, "rej.setdefault(k_, set()).add(v_)", "pass", T_KL),
 ("N-4: a Latin token may be a piece of a longer word again", GR, '        if re.fullmatch(r"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+", v.strip()):', "        if False:", T_KL),
 ("N-6: any value on a property with a default is exempt", GR, 'return "default" in prop and str(prop["default"]) == str(value)', 'return "default" in prop', T_KL),
 ("N-6: x-grounding: none is ignored", GR, 'if prop.get("x-grounding") == "none":', "if False:", T_KL),
 ("N-6: the session does not give the guard the tool schema", RT, 'schema=grounding.tool_parameters(self.cfg.get("tools"), e.get("name", "")),', "schema=None,", T_KL),
 ("N-6: the cap counts rejections over the whole session again", RT, "            self.guard_total = 0                                  #", "            pass                                                  #", T_KL),
 ("N-6: the give-up message no longer teaches the read-back", GR, '"Ask the caller plainly for the missing information; they may also prefer to end the call. " + way)', '"Ask the caller plainly for the missing information; they may also prefer to end the call.")', T_KE + T_KL),
 ("N-6: the prompt no longer says to leave optional arguments out", GR, "and do not fill an optional argument the caller did not mention (leave it out)", "", T_KL),
 ("N-8: the audit is not written to the log", RT, 'log.info("realtime %s: grounding audit %s", self.id, json.dumps(rec, ensure_ascii=False, default=str)[:500])', "pass", T_KL),
 ("N-8: a keyword regex is added to grounding.py (the old allowlist waved it through)", GR, "def norm_text(s: str) -> str:", 'WEATHER = re.compile("天气|气温|下雨")\n\n\ndef norm_text(s: str) -> str:', T_KS),
 ("N-9: the harness is not retried after a timeout (and a second timeout is not an error)", "tests/test_r4j_page_behaviour.py", "    for attempt in (1, 2):", "    for attempt in (1,):", T_KL),
 ('R1 model_client: a timeout inside the call is classified as an application error', "sophon_live/model_client.py", '            except (asyncio.TimeoutError, TimeoutError):\n                br.failure("timeout")', '            except (KeyError,):\n                br.failure("timeout")', T_R1),
 ('R1 model_client: a timeout counts as a retryable connection error', "sophon_live/model_client.py", '    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):\n        return False\n    try:', '    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):\n        return True\n    try:', T_R1),
 ('R1 model_client (RC2-3): a cancelled half-open probe is not released', "sophon_live/model_client.py", '            br.release_probe()\n', '            pass\n', T_R1),
 ('R1 model_client: the breaker never opens', "sophon_live/model_client.py", '        if self.state == "half_open" or self.fails >= self.threshold:', '        if False:', T_R1),
 ('R1 model_client: half-open lets every caller probe', "sophon_live/model_client.py", '            if self.probing:                       # one probe at a time; everyone else fails fast', '            if False:', T_R1),
 ('R1 model_client: application errors are retried like connection errors', "sophon_live/model_client.py", '                if is_connection_error(e) and attempt < retries:', '                if attempt < retries:', T_R1),
 ('R1 model_client: the JEV budget default is 8 s again', "sophon_live/model_client.py", 'DEFAULT_BUDGET_S = {"jev": 1.5,', 'DEFAULT_BUDGET_S = {"jev": 8.0,', T_R1),
 ('R1 jev_client: bypasses the model client (no budget / breaker)', "sophon_live/coordinator/jev_client.py", '    return await model_client.call("jev", _once, retries=1, total_s=DEFAULT_TIMEOUT)', '    return await _once()', T_R1),
 ('R1 orchestrator: the LLM step bypasses the breaker', ORCH, 'await model_client.call("orch", _step, retries=1, total_s=timeout)', 'await _step()', T_R1),
 ('R1 orchestrator: a 5xx is not counted against the breaker', ORCH, '                if status >= 500:', '                if False:', T_R1),
 ('R1 orchestrator: a 400 is not retried without the optional fields', ORCH, '                if status == 400 and ("chat_template_kwargs" in body or "tool_choice" in body):', '                if False:', T_R1),
 ('R1 orchestrator: the worker budget default is 120 s again', ORCH, '_DEFAULT_TIMEOUT = 30.0', '_DEFAULT_TIMEOUT = 120.0', T_R1),
 ('R1 adapters: the Tag connect is reported under another breaker name', AD, 'model_client.call("tag",', 'model_client.call("jev",', T_R1),
 ('R1 gateway: /health?detail=1 never returns the detail', GW, '    return _health_detail() if "detail=1" in query.split("&") else None', '    return None', T_R1),
 ('R1 gateway: any query returns the detail', GW, '    return _health_detail() if "detail=1" in query.split("&") else None', '    return _health_detail()', T_R1),
 ('R1 gateway: the downstream snapshot is left out of the detail', GW, '"downstream": model_client.snapshot(),', '"downstream": {},', T_R1),
 ('R1 gateway: the bind address defaults to all interfaces', GW, 'or "127.0.0.1"\n', 'or "0.0.0.0"\n', T_R1),
 # Round B (Critic G-1): the literal path is an exact comparison; each removed rule must stay removed (tests/test_r4m_g1_literal_only.py)
 ("G-1: a date whose year was never said passes when MMDD was said", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        dm = re.fullmatch(r\"\\d{4}-(\\d{2})-(\\d{2})\", v.strip())\n        if dm and (dm.group(1) + dm.group(2)) in corpus:\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: an e-mail passes when its local part and the FIRST label of the domain were said", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        loc, _, dom = v.strip().partition(\"@\")\n        if loc and dom and appears_in_words(loc, said) and appears_in_words(dom.split(\".\")[0], said):\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: a multi-word value passes when every part occurs anywhere", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        parts = [norm_text(p) for p in re.split(r\"[\\s,]+\", v) if norm_text(p)]\n        if len(parts) > 1 and all(occurs(p, corpus) for p in parts):\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: a spelled-out run is a literal candidate again", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        if vn in {norm_text(j) for h in heard for j in spelled_joins(h)}:\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: a number word is special-cased on the literal path", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        if vn == \"4000\" and \"fourthousand\" in corpus:\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: a currency word is special-cased on the literal path", GR, "        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", "        if vn == \"krw\" and \"won\" in corpus:\n            return True\n        if re.fullmatch(r\"(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+\", v.strip()):", T_GB),
 ("G-1: a value under 3 characters is skipped again", GR, "        vn = norm_text(v)\n        if not vn:", "        vn = norm_text(v)\n        if not vn or len(vn) < 3:", T_GB + T_KE),
 ("G-1: a single character is skipped again", GR, "        vn = norm_text(v)\n        if not vn:", "        vn = norm_text(v)\n        if not vn or len(vn) < 2:", T_GB + T_KE),
 ("G-1: a Latin token only has to occur as a substring", GR, "            return appears_in_words(unicodedata.normalize(\"NFKC\", v), unicodedata.normalize(\"NFKC\", said))", "            return occurs(vn, corpus)", T_GB + T_KL),
 ("G-1: the generic whole-word rule becomes a substring test", GR, "        return occurs(vn, corpus) or appears_in_words(v, said)", "        return occurs(vn, corpus) or v in said", T_GB),
 ("G-1: a non-bool judge answer counts as a verdict", GR, 'table[(p["kind"], p["argument"], p["value"], p["span"])] = r if isinstance(r, bool) else None', 'table[(p["kind"], p["argument"], p["value"], p["span"])] = bool(r) if r is not None and not isinstance(r, BaseException) else None', T_GB),
 ("G-1: a judge exception counts as a yes", GR, 'table[(p["kind"], p["argument"], p["value"], p["span"])] = r if isinstance(r, bool) else None', 'table[(p["kind"], p["argument"], p["value"], p["span"])] = True if isinstance(r, Exception) else (r if isinstance(r, bool) else None)', T_GB),
 ("G-1: a number-word table is back in grounding.py", GR, "from .coordinator.schemas import appears_in_words\n", "from .coordinator.schemas import appears_in_words\n_UNITS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten'.split())}\n", T_GB + T_KS),
 ("G-1: a currency-word table is back in grounding.py", GR, "from .coordinator.schemas import appears_in_words\n", "from .coordinator.schemas import appears_in_words\n_CURRENCY = {'dollar': 'usd', 'euro': 'eur'}\n", T_GB + T_KS),
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


env = dict(os.environ, PYTHONPATH=".", SOPHON_LIVE_JEV="0", PYTHONDONTWRITEBYTECODE="1")   # a stale .pyc (same size, same second) must never stand in for the mutated or the restored source
for _d, _dn, _fn in os.walk("."):
    if os.path.basename(_d) == "__pycache__":
        shutil.rmtree(_d, ignore_errors=True)
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
    if HARNESS_TIMEOUT_MARKER in out:
        script_error(f"mutant '{name}': the page harness timed out (machine busy?); that is not a kill. Re-run on an idle machine or raise SOPHON_PAGE_HARNESS_TIMEOUT\n" + out[-600:])
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
