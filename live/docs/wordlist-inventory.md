# Word-list / keyword-rule inventory (sophon-realtime `live/`)

Written in Round 4f for Critic's B-4. Snapshot of branch `builder/r4-review-fix` (parent of this commit: `2ed385b`; line numbers are for this commit).

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
| `assistant_tools/runner.py:65` | `_route_task` picks a city by scanning the task for table names | FROZEN → stage 3/4 (see §3, legacy `run_task`) |
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
| `assistant_tools/runner.py:54-92`, `:95` | `_route_task` / `run_task`: regex heuristic mapping a task to a tool (legacy `SOPHON_LIVE_BUILTIN_TOOLS=1` demo, documented as "not architecture", `docs/ASSISTANT_TOOLS.md:32`) | tool choice | **OPEN, deletable.** Not deleted this round: it is a live gateway code path (`gateway.py:604,613,624,853-857`) behind an env flag whose value on the GPU server I did not check (no server actions), and two tests (`test_assistant_tools.py::TestRouting`) pin it. Proposal: delete with the flag in stage 3/4 once the server env is confirmed unset. |
| `assistant_tools/news.py:39-41,56-62` | `_TECH_KEYS`, `_FIN_KEYS`, `_WORLD_KEYS` → `_bucket(topic)` | which RSS feed a topic uses | **OPEN.** Tool-internal topic routing over a *model-produced* `topic` argument. Proposal: stage 4, let the model pass a structured `category` enum argument. |
| `assistant_tools/news.py:150-158` `_filter_topic` | keeps headlines containing the topic's tokens (drops "新闻/头条/资讯/news/headlines") | relevance filter | **OPEN**, same proposal as above. |
| `coordinator/schemas.py:66-92` `TOOL_FAILURE_TEMPLATES` | the exact sentences our tools speak on failure | provenance of a spoken failure sentence | KEPT-provenance (templates, not a reading of user words) |
| `grounding.py:13` `PLACEHOLDER_DOMAINS`, `:196` `failure_hint` regex (`not found|no user|…|error`) | reads a tool's *output text* for failure words (legacy realtime/grounding path, not the Coordinator/Worker path) | whether a tool output counts as failed, for a hint to the model | **OPEN.** Same defect class as the B-3 text probes. Proposal: stage 3/4, switch that path to structured return codes like the Worker (`failcodes`). |
| `assistant_tools/orchestrator.py` `SYS` prompt | instructions to the model | n/a | not a rule; prompts are the sanctioned mechanism |

### 2c. Script / format tests (KEPT-benign)

`gateway.py:38,135,138,233` (`CJK` regex: language by script, sentence splitting), `news.py:44-45`, `web_search.py:41-42`, `device.py:61`,
`weather.py:149`, `runner.py:60` (`[一-鿿]` script test), `schemas.py:103,105` (ASCII word boundary in `_said_in`), `web_search.py:115-144`
(HTML parsing), `news.py:107-123` (publisher-name length bounds). They pick a language or parse markup; none decides what the user wants.

## 3. What this round removed (cumulative on this branch)

* B-3 (`2ed385b`): city-list decisions in `schemas.py` (`_known_cities`, `_is_whole_entity`, `_split_objective`, `_REFINE_MARKERS`); text probes for tool failure in `orchestrator.py`.
* B-4 (this round): `slots.CITY_NAMES` in `_ungrounded_city`.
* I-18 (this round): text probes for tool success in `orchestrator.py` (`"第1条"`, `"Headlines on"`, `"度"`, `"°"`, `"°C"`).

## 4. Guard rails so they stay gone

* `tests/test_r4e_b4_i18.py`: `_ungrounded_city` answers are identical with `slots.CITY_NAMES` patched to `()` or to a hostile list; AST check that `orchestrator.py` references neither `CITY_NAMES` nor `CITIES` nor imports `slots`; no success-text probe strings in the source.
* `tests/test_r4d_s23_s24.py`: AST check that `schemas.py` reads a place table only inside `_city_aliases`; no `in out` failure probe in `run_job`.
* `tests/mutation_check_r4c.py`: mutants that reintroduce each of the above (35 mutants; see the script header for exit codes).
