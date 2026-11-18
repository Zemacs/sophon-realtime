# Word-list / keyword-rule inventory (sophon-realtime `live/`)

Written for Critic's B-4 (commit `7cce424`) and updated in Round 4g (failure_hint + legacy builtin path) and Round 4h (grounding blacklists, news topic, `index.html` demo backend). Snapshot of branch `builder/r4-review-fix`; line numbers are for the commit that updates this file.

**Principle (user, 2026-11-19 00:27):** no hard-coded keyword / regex / word-list semantic judgment anywhere. The model judges; code only
enforces (provenance, structured return codes, state-machine version checks; literal comparison only to verify that a model-produced value
comes from the user's words or a tool's output, never to decide what to do).

**Status values:** `REMOVED` (gone in this round or earlier on this branch) · `KEPT-provenance` (allowed: verifies a model value, decides nothing) ·
`KEPT-data` (a tool's own data table, not a judgment about user words) · `KEPT-benign` (script/format test, no semantic decision) ·
`FROZEN → stage N` (a semantic rule on user words; frozen, no new entries, removal planned in that stage, per Critic's adjudication:
"anything belonging to intent/routing stays frozen for stage 3/4") · `OPEN` (not intent/routing; needs a decision).

Method: `grep` for `CITY_NAMES`/`CITIES` references, then for module-level tuples / frozensets / `re.compile` / `re.search` / `any(x in text …)` over
`sophon_live/**/*.py` (tests excluded, listed separately). Inference: the list is only as complete as those patterns; a rule built another way
(e.g. a dict of phrases, a chain of `if "…" in t`) could be missed. The `if "…" in t` pattern was grepped separately and found only the lines below.

## 1. `CITY_NAMES` / `CITIES` references (the B-4 ask)

| Where | What | Status |
|---|---|---|
| `assistant_tools/orchestrator.py:128` `_ungrounded_city` (was `:142` `slots.CITY_NAMES`) | "is the model's weather city grounded?" | **REMOVED this round.** Provenance only: `appears_in_words(city, user/tool messages, translate=True)`; the import of `coordinator.slots` is gone. |
| `coordinator/schemas.py:142,147` `_city_aliases` | translates 北京 ⇄ Beijing so provenance can be checked | KEPT-provenance (reads the weather tool's own `CITIES`; a test pins that it is the only reader in `schemas.py`) |
| `assistant_tools/weather.py:28,146,351` `CITIES` | the weather tool's geocoding / canonical-name table | KEPT-data |
| ~~`assistant_tools/runner.py:65`~~ | `_route_task` picked a city by scanning the task for table names | **REMOVED in Round 4g** (whole legacy `run_task` path deleted, see §2b) |
| `coordinator/slots.py:12` `CITY_NAMES` | city words for "same request, one slot changed" | FROZEN → stage 3/4 |
| `coordinator/slots.py:43` `skeleton()` | strips `CITY_NAMES + DAY_WORDS` before comparing requests | FROZEN → stage 3/4 |
| `coordinator/route.py:173` `slot_edit()` | REFINE guard: finds old/new city (and day) words | FROZEN → stage 3/4 |
| `coordinator/route.py:56,138` `_CITIES` | second, shorter city list (7 entries) inside `heuristic_route` | FROZEN → stage 3/4 (dies with `heuristic_route`) |
| `tests/test_refine_guard.py:99` | asserts `slots.CITY_NAMES == weather.CITIES` keys (keeps the frozen copy in sync) | goes with `slots.py` in stage 3/4 |
| `tests/test_r4d_s23_s24.py`, `tests/test_r4e_b4_i18.py` | AST / patch tests that prove the removed rules stay removed | permanent guards |

After this round `CITY_NAMES` has **no reader outside the intent/routing layer** (`slots.py`, `route.py`), apart from the sync test.

## 2. Other word lists and keyword rules

### 2a. Intent / routing (FROZEN → stage 3/4; no new entries)

| File:line | Construct | Decides |
|---|---|---|
| `coordinator/intent.py:21-88` | `_TIME`, `_ASSET`, `_QTY`, `_PRICE_NOUN`, `_CHARTS`, `_SPORT*`, `_OUTCOME_NOW`, `_EVENT`, `_TRANSPORT`, `_OFFICE`, `_FRESH`, `_EN*`, `_STRONG_VETO`, `_UNIT_CONVERT`, `_MONEYISH`, `_WEAK_VETO`, `_NOW*`, `_ENTITY*`, `_QUESTION`, `_QTY_*`, `_PERSONAL`, `_STABLE`, `_ARITH`, `_COUNT_WEAK` | `is_realtime_fact()` (`:91-122`): must a factual question be looked up |
| `coordinator/intent.py:123-130` `_PATTERNS` | one regex per tool (天气 / 新闻 / 几点 / 打开 灯 / 搜索 / 记住) | `required_tools()` (`:133`), `is_tool_scoped()` (`:147`): which tool a request needs |
| `coordinator/route.py:55-94` | `_CLARIFY_HINTS`, `_WORKER_HINTS`, `_NEW_REQUEST_HINTS`, `_CANCEL_HINTS`, `_STATUS_HINTS`, `_REFINE_HINTS`, `_SWAP_RE`, `_ACK_WORDS` | `heuristic_control()` (`:105`), `heuristic_route()` (`:134`): cancel / status / refine / worker vs fast. Callers: `api.py:93,99,101,136` (stage 0 of the decision-ownership work, deplux PR #28, replaces the `heuristic_route` calls with the degraded path; the functions themselves are removed in stage 3/4) |
| `coordinator/route.py:147` | `_EDIT_CUES` | `slot_edit()` (`:160`): is this an edit of the last task |
| `coordinator/slots.py:16,18-20` | `DAY_WORDS`, `_PUNCT`, `_FILLER`, `_SLOT_PREFIX` | `skeleton()` / `similarity()` for `slot_edit` |
| `coordinator/clarify.py:22` | `"天气" in text` → missing slot `city` | `missing_slots_for_text()`, used by `api.py:35` |
| `coordinator/memory_gate.py:24,33` | `EXPLICIT_SAVE_HINTS` | `has_explicit_save_intent()`: may a preference be saved long-term |
| `coordinator/memory_gate.py:27-28` | `TASK_SLOT_KEYS`, `PREFERENCE_PREFIXES` | which *tool-argument key names* are task-scoped: schema keys, not user words; KEPT-benign |

### 2b. Tool side

| File:line | Construct | Decides | Status |
|---|---|---|---|
| ~~`assistant_tools/runner.py` `_route_task` / `run_task`~~ | legacy `SOPHON_LIVE_BUILTIN_TOOLS=1` demo: regexes mapping a task to a tool | tool choice | **REMOVED in Round 4g.** Verified read-only on the server (by the user's side, before this round): the flag is unset on every running gateway (:8780 pid 3548220 and pid 3564091), and the branch was reachable only with COORD and ORCH both off. Deleted: `run_task`, `_route_task`, the `use_builtin` branch and `_builtin_resolve` in `gateway.py`, the `backend="builtin"` line in `static/index.html`, the `__init__.py` doc line and `run_task` export, and the `ASSISTANT_TOOLS.md` mentions. The env var now has no effect (test: `test_assistant_tools.py::TestLegacyHeuristicRoutingIsGone`). |
| ~~`assistant_tools/news.py` `_TECH_KEYS`, `_FIN_KEYS`, `_WORLD_KEYS` → `_bucket(topic)`~~ | which RSS feed a topic uses | feed selection | **REMOVED in Round 4h** (this was the "Round 4f" item in the 4g inventory; the label was wrong, it is Round 4h). `get_news.topic` is now a closed-enum tool parameter (`tech` / `finance` / `world` / `general`, `news.CATEGORIES`, in the tool schema) that the orchestrator model fills; the code fetches by that parameter only (`news._ZH_FEEDS[topic]`, then a Google RSS query for the category's fixed label). A value outside the enum is rejected by the orchestrator (`_missing_required`, schema-driven) and by `news.run` (returns the "which topic?" sentence, nothing fetched). Tests: `test_r4h_grounding_news.py`; mutants in `mutation_check_r4c.py`. |
| ~~`assistant_tools/news.py` `_filter_topic`~~ | keeps headlines containing the topic's tokens | relevance filter | **REMOVED in Round 4h** (same change). Headlines are taken as the category feed gives them; no keyword filter, no cross-category fallback. Behavior change: asking for news about a specific person/company/event is no longer served by `get_news` (the model uses `web_search` for it, said in the tool description and orchestrator prompt). |
| `coordinator/schemas.py:66-92` `TOOL_FAILURE_TEMPLATES` | the exact sentences our tools speak on failure | provenance of a spoken failure sentence | KEPT-provenance (templates, not a reading of user words) |
| `grounding.py:196` `failure_hint` (regex `not found|no user|…|error` over a tool's output text) | whether a client tool output counts as failed, to append a spell-it-out hint (Realtime face; the Coordinator/Worker path was already structured) | **REMOVED in Round 4g.** `failure_hint(failed, heard)` now takes a flag; the flag is `grounding.output_failed(item)`: a truthy item `error`, item `status` `failed`/`incomplete`, or an output that is a JSON object with `ok:false` / `success:false` / non-empty `error`. Prose that merely says "not found" is not a failure. `realtime.py` carries the flag per queued output. |
| ~~`grounding.py` `PLACEHOLDER_DOMAINS`, `PLACEHOLDER_NAMES`, `PLACEHOLDER_VALUES`, `FREE_TEXT_KEYS`~~ (+ the `SOPHON_LIVE_GROUND_MODE=placeholder` mode) | values the model may not invent as tool arguments; argument keys exempt from checking | argument-grounding guard on the Realtime face | **REMOVED in Round 4h.** Provenance only: a short value passes if it occurs, as a whole word (`schemas.appears_in_words`), in what the caller said or a tool returned, or in a spoken/normalised form of it (the pre-existing spoken-digits / `at`-`dot` / fuzzy-misheard logic, which is provenance); otherwise it is blocked, whatever it looks like and whatever the argument is called. 12345 / John Doe / example.com are allowed when the caller said them and blocked when the model invented them. Still structural, not lexical: values of more than 3 words and normalised values under 3 characters are out of scope. Tests: `test_r4h_grounding_news.py::GroundingIsProvenanceOnly`. |
| `static/index.html:170-207` `demoBackend` (+ `weather`, `CITIES`) | JS regexes on the delegated task (weather / time / date / device words) and a 21-city table, used by the browser test page to fake a backend answer | tool choice in a demo backend | **FROZEN (not changed this round, by instruction); not in the product path.** Reachability checked: the gateway serves this file at `GET /` and `/index.html` on the same port as the WebSocket (`gateway.py` `process_request`), so it ships and is reachable by anyone who opens `http://HOST:n/` (the README calls it "a browser test page"; the docs reach it through an SSH tunnel). It runs only in that page, only when the delegation event has `backend` other than `orch` **and** the page's "后端" select is on the default "内置演示后端" (`onDelegation`, line 201 short-circuits `orch` only). **Residual to know:** the gateway sends `backend: "coord"` when `SOPHON_LIVE_COORD` is on, and the page does NOT short-circuit `coord`, so on a coord gateway the test page still runs `demoBackend` and injects its answer as commentary next to the real coordinator result (unless the select is set to manual). Phone / Realtime-API / LiveKit / Pipecat clients never load it. Plan: stage 3/4 with the other routing lists; the minimal fix then is to short-circuit `coord` like `orch` (or delete the demo backend). Docs already say it is demo-only (`docs/sophon-tool-extension-paths.md:70`). |
| `assistant_tools/orchestrator.py` `SYS` prompt | instructions to the model | n/a | not a rule; prompts are the sanctioned mechanism |

### 2c. Script / format tests (KEPT-benign)

`gateway.py:38,135,138,233` (`CJK` regex: language by script, sentence splitting), `news.py:44-45`, `web_search.py:41-42`, `device.py:61`,
`weather.py:149`, `runner.py:60` (`[一-鿿]` script test), `schemas.py:103,105` (ASCII word boundary in `_said_in`), `web_search.py:115-144`
(HTML parsing), `news.py:107-123` (publisher-name length bounds). They pick a language or parse markup; none decides what the user wants.

## 3. What this round removed (cumulative on this branch)

* B-3 (`2ed385b`): city-list decisions in `schemas.py` (`_known_cities`, `_is_whole_entity`, `_split_objective`, `_REFINE_MARKERS`); text probes for tool failure in `orchestrator.py`.
* B-4 (this round): `slots.CITY_NAMES` in `_ungrounded_city`.
* I-18 (this round): text probes for tool success in `orchestrator.py` (`"第1条"`, `"Headlines on"`, `"度"`, `"°"`, `"°C"`).

* Round 4g: `grounding.failure_hint` text regex; the legacy `SOPHON_LIVE_BUILTIN_TOOLS` / `run_task` keyword router (and the `runner.py:65` city scan inside it).

## 4. Guard rails so they stay gone

* `tests/test_r4e_b4_i18.py`: `_ungrounded_city` answers are identical with `slots.CITY_NAMES` patched to `()` or to a hostile list; AST check that `orchestrator.py` references neither `CITY_NAMES` nor `CITIES` nor imports `slots`; no success-text probe strings in the source.
* `tests/test_r4d_s23_s24.py`: AST check that `schemas.py` reads a place table only inside `_city_aliases`; no `in out` failure probe in `run_job`.
* `tests/test_r4g_failure_hint.py` (structured-only hint, including the real `RealtimeSession` path) and `tests/test_assistant_tools.py::TestLegacyHeuristicRoutingIsGone` (entry points gone, no source mentions the flag, env var has no effect).
* `tests/mutation_check_r4c.py`: mutants that reintroduce each of the above (42 mutants; see the script header for exit codes).
