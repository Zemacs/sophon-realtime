# Round A (M-1): the routing / intent layer is the model's, the code only enforces

Status: **implemented in Round A** of the v0.1.0 r4 mainline (PR #32). Replaces the earlier "comparison only" draft.
Numbers marked *(relayed)* come from Critic's reviews and were not re-measured (no server / GPU access, no live run).
Sources: r4 = `7440dc9`; rc3 = `7ec617c` (sophon #31), rc2 `0650877` (#30) its ancestor.

The rule (user): no keyword, regex, word-list or special-cased number/phrase decides anything. The model judges; the code enforces.

## 1. What changed (behaviour 「行为变化」)

| Decision | Before (r4) | Now |
|---|---|---|
| Which tool a request needs | `intent.py` regex tables (`_PATTERNS`, `_REALTIME_RE`, vetoes ...) | the router (JEV) answers a closed `tool` question (`intent.TOOL_CRITERIA`: NONE / get_weather / get_news / get_datetime / device_control / web_search / memory_note / memory_save). Code maps option key -> tool names (`tools_from_answer`), stamps `WorkerJob.require_tools`, and `finish_job` rejects a tool-less answer. |
| Filler / lead-in | `pad_for(text)` read the sentence | `pad_for_tools(tools)`: a table keyed by the model's option key. Router unusable -> no filler |
| Route, control target | JEV, **falling back to keyword code** (`heuristic_route` / `heuristic_control`), and `slot_edit` (city/day word lists, similarity 0.8) ran **before** JEV | JEV only. JEV down / breaker open / non-option answer -> `api._degrade`: no live task = the sentence becomes a new task for the orchestrator (nothing forced, tool judgment recorded `unknown`); a live task = one fixed clarify sentence. A REFINE option is honoured on either route; code only checks task exists / version current / state machine |
| Explicit "remember this long-term" | `memory_gate.EXPLICIT_SAVE_HINTS` | the model's `memory_save` option; `memory_gate` writes long-term only on `explicit_save is True`; unknown never writes |
| Missing slot for CLARIFY | `"天气" in text` | the model's `missing_slot` question (city / query / topic / other); unknown -> generic ask. Only asked on a CLARIFY route (+1 JEV request, rare) |
| Unusable router answers | n/a | never guessed, never silent: counted in `api.JUDGE_STATS` (`/health?detail=1` -> `router_invalid_answers`) and traced per turn (`judge_invalid`) |
| The user's own words | n/a | the model also sees `user_said` (= `raw_text`) when the front rewrote the task |

Consequence users can notice: when the router is down, requests that used to be recognised by keywords (cancel / status / refine words) now
get the fixed clarify sentence while a task is live, and a new task otherwise. This is deliberate (no guess).

Removed from production: `coordinator/slots.py` (whole file), `route.heuristic_route` / `heuristic_control` / `slot_edit` / `HINTS` / `_SWAP_RE`,
`intent._PATTERNS` / `_REALTIME_RE` / `is_realtime_fact` / `required_tools` / `pad_for`, `clarify.missing_slots_for_text`,
`memory_gate.EXPLICIT_SAVE_HINTS` / `has_explicit_save_intent`. `tests/test_no_semantic_rules.py` (whole-package AST guard, digest-keyed allowlist)
pins that none of them comes back and that nothing under `sophon_live` imports `jev_stub`.

## 2. Ported from rc3, excluded, deferred

* **Ported (structure):** `intent.py` (`TOOL_CRITERIA`, `tools_from_answer`, `explicit_save_from_answer`, `pad_for_tools`), `route.py`, `clarify.py` (`MISSING_SLOT_KEY`), `ledger.py` (tools / judgment / explicit_save carried across REFINE), `memory_gate.py`, `api.py` (`_degrade`, `JUDGE_STATS`), `schemas.py`, the scripted router `tests/jev_stub.py` + `toolstub.py` (test doubles: **nothing under `sophon_live` may import them**), and the measurement scripts `tests/eval_tool_judgment.py` / `eval_control_judgment.py` (need a live JEV; not part of the unit suite).
* **r4's N-series protections are kept** (`grounding.py`, `realtime.py`, `coordinator/span_judge.py`): they are independent of the intent layer and untouched.
* **memory_note filler:** rc3 gave "" for memory recall but r4's front prompt still tells the front to lead in, so r4's lookup lead-in "我帮你查一下。" is kept for memory_note (prompt fix deferred).
* **Excluded:** rc3's numeral normaliser (Critic C-1/C-2); `task_faithful`; the no-delegate net; FILLER_RULE prompt changes; `review_numbers` / number containment (RC2-6/7) and the answer-side parts that need it; `eval_blind.py`.
* **Deferred:** RC2-6 ("treat as FACT when the router is unavailable") has no consumer here (no answer-side number check); it lands with Round C (independent answer-provenance question).
* Latency of the merged call: `docs/admit-merged-call-latency.md`.

## 3. Every deleted test has a named replacement (or an honest gap)

Accuracy of "which tool / which control" is now the **model's**: it cannot be unit-tested with a scripted router. What unit tests can pin is that the
code enforces whatever closed answer arrives. Accuracy is measured by `tests/eval_tool_judgment.py` / `eval_control_judgment.py` on the live server
(blocked until the user lifts the freeze); the labelled sets stay (`intent_cases.py`, `intent_dev*.py`, `intent_frozen.py` + pin, `intent_tools_set.py`, `intent_critic92.py`, `intent_rc1_heldout.py`).

| Deleted test(s) | Measured | Replacement |
|---|---|---|
| `test_r4_intent_gate.py` (9) | regex gate accuracy / over-trigger | live eval `eval_tool_judgment.py` (**gap: no unit-time accuracy**); enforcement: `test_filler.py::test_the_model_answer_maps_to_tools_by_option_key_only`, `test_no_fabrication.py::test_every_option_names_a_registered_tool`, `::test_no_model_answer_is_unknown_not_a_guess`; label sets still pinned by `test_r4_intent_frozen_pin.py` |
| `test_r4_intent_i10.py` (10) | named regressions, currency vs unit veto | same sets to the model (live eval); **gap**: no unit-time equivalent |
| `test_refine_guard.py` (13: city/day swap, skeleton, `slot_edit`, guard-beats-JEV ...) | word-based REFINE guard | `test_a_rewritten_sentence_is_a_refine_when_the_model_says_so`, `test_the_model_sees_the_users_own_words_and_the_live_tasks`, `test_new_means_new_task_and_the_old_one_stays`, `test_a_stale_version_from_the_model_is_rejected_by_the_state_machine`, `test_control_with_no_usable_option_is_a_fixed_clarify_not_a_guess`; plus `test_coordinator_enforcement.py::RefineToolTests` |
| `test_ledger_control.py` (8: cancel/status/refine words, short answer fills slot, ...) | `heuristic_*` | the model answers `control`; code enforcement in `test_ledger_control.py::test_jev_control_without_a_usable_option_is_a_fixed_clarify_and_the_task_is_untouched` + the remaining state-machine tests; words themselves -> `eval_control_judgment.py` (live) |
| `test_ledger_hygiene.py::test_acknowledgements_never_fill_a_pending_slot` | ack word list | model's `control` (live eval); code: a slot-fill is a REFINE of a NEEDS_INPUT task only by the model's option |
| `test_memory_gate.py::test_explicit_save_intent` | `EXPLICIT_SAVE_HINTS` | `test_explicit_save_is_the_models_answer_and_unknown_is_not_a_yes`, `test_the_gate_keeps_a_write_as_a_task_slot_unless_the_model_said_save` |
| `test_no_fabrication.py` (5: `required_tools` on sentences) | regex tool decision | `test_every_option_names_a_registered_tool` (replaces `test_tool_names_exist`); sentence-level -> live eval |
| `test_filler.py::test_time_jokes_and_chitchat_get_no_filler` | `pad_for(text)` | `test_time_jokes_chitchat_and_unknown_get_no_filler` (by option key) |
| `test_blind_runner.py` (10: heuristic degraded kinds, keyword-function wrappers, refine guard as `via=guard`) | wrappers around keyword functions | the functions are gone: `test_a_jev_failure_is_recorded_as_degraded_with_production_s_own_reason`, `test_the_degraded_reason_does_not_depend_on_the_message`, `test_jev_switched_off_is_not_degraded_and_says_so`, `test_control_without_a_usable_target_is_a_fixed_clarify_decided_by_the_model_s_route`, `test_there_is_no_guard_any_more_...`, `test_the_model_s_tool_answer_is_recorded_not_derived`, `test_the_spy_passes_the_request_and_the_answer_through_unchanged`, `test_the_jev_is_restored_after_every_run` / `_even_when_the_run_crashes` |

New in Round A: `tests/test_coordinator_enforcement.py` (11 tests: degraded paths, REFINE tool inheritance, missing-slot, invalid-answer visibility),
`tests/mutation_coordinator.py` (22 mutants, wired into `release_gate.py`), `tests/test_no_semantic_rules.py` (13).
No assertion was weakened to get green: the deleted ones tested code that no longer exists.
