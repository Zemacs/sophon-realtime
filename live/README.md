# Live mode: a drop-in for the OpenAI GPT-Live API

`sophon_live` is a gateway that speaks the **OpenAI GPT-Live protocol** (`/v1/live/sessions`, client delegation) and runs a Sophon-Realtime engine (Tag or Flash) behind it. Clients written for GPT-Live connect by changing only the URL and key.

Sophon keeps the voice: it listens, decides when it's its turn, and talks. When a request needs a lookup or an action (time, weather, memory, news, device control), it says a short line such as "Let me check that", hands a task to **your** backend through `session.delegation.created`, and speaks the result in its own words when your backend answers with `session.commentary.append`.

```
GPT-Live client ──GPT-Live protocol──▶ sophon_live.gateway ──▶ Tag / Flash engine (/v1/realtime + tool bridge)
                 ◀── delegation.created ──┘        ▲
your backend ─── commentary.append / thinking.append ┘
```

## Run

Live mode is not a separate install: it is a small gateway that runs next to a Tag or Flash engine started with the one-command scripts, in the same Python environment. The engine's `start.sh` already opens the tool bridge the gateway needs (on `PORT + 700`, 127.0.0.1 only).

```bash
# 1. start the engine as usual (Tag here; Flash works the same way)
cd sophon-realtime/tag && GPU=0 PORT=8000 bash start.sh

# 2. start Live mode next to it
cd ../live && BUILD=tag PORT=8000 LIVE_PORT=8780 bash start_live.sh
```

GPT-Live clients connect to `ws://HOST:8780/v1/live/sessions`; `http://HOST:8780/` is a browser test page (microphone, delegation cards, a manual-answer box and the protocol event log). Stop with `LIVE_PORT=8780 bash stop_live.sh`.

The gateway loads a small ASR model for the user transcript (`Qwen/Qwen3-ASR-0.6B-hf`, about 2 GB of GPU memory). In mainland China, download it from ModelScope first: `modelscope download --model Qwen/Qwen3-ASR-0.6B-hf --local_dir ./asr`, then `SOPHON_LIVE_ASR=./asr bash start_live.sh`.

## Protocol coverage

- Client events: `session.start` (instructions, voice, `delegation: {type: client}`, `input` history), `session.input_audio.append`, `session.commentary.append` (spoken result), `session.thinking.append` (background context, not read out), `session.instructions.append`, `session.close`.
- Server events: `session.started`, `session.output_audio.delta` (continuous ~100 ms frames, silence when idle), `session.output_transcript.delta` (paced with playback), `session.input_transcript.delta`, `session.delegation.created` (adds an extra `task` field with what Sophon handed over), `session.commentary.appended`, `session.usage.updated`, `session.closed`, `error`.
- Barge-in is handled server-side: when the user talks over Sophon, playback stops.
- Optional settings: `SOPHON_LIVE_API_KEYS` (comma-separated bearer keys), `SOPHON_LIVE_ALLOW_ORIGINS` (browser origins allowed; other `Origin` headers are rejected, as in GPT-Live), `SOPHON_LIVE_VOICE`.

## OpenAI Realtime face: LiveKit and Pipecat

The same gateway also serves an **OpenAI Realtime** compatible endpoint at `ws://HOST:8780/v1/realtime` (GA event names, function calling, barge-in truncation, PCM16 and G.711). Voice-agent frameworks that support OpenAI Realtime work by changing the base URL, no new plugin needed:

| Framework | What to change | Example | Tested |
|---|---|---|---|
| LiveKit Agents | `openai.realtime.RealtimeModel(base_url="http://HOST:8780/v1", ...)` | [examples/livekit_agent.py](examples/livekit_agent.py) | livekit-agents 1.8.11 |
| Pipecat | `OpenAIRealtimeLLMService(base_url="ws://HOST:8780/v1/realtime", ...)` | [examples/pipecat_bot.py](examples/pipecat_bot.py) | pipecat-ai 1.15.0 |

Both frameworks add no noticeable latency: Pipecat measured the same as a direct WebSocket client, LiveKit within about 0.1 s.

Set `SOPHON_LIVE_REALTIME_KEYS_FILE` (one key per line) to require a bearer key on `/v1/realtime`.

## Front + back office (optional)

Set `SOPHON_LIVE_BACKEND=<model>` with an OpenAI-compatible endpoint (`SOPHON_LIVE_BACKEND_URL`, `SOPHON_LIVE_BACKEND_KEY`) and the gateway answers delegations itself with a larger model: Sophon stays on the line, the back office reasons and calls tools. With DeepSeek V4 Pro as the back office, Big Bench Audio rises from 81.2 to about 83.

## Test

`python live_test.py --url ws://127.0.0.1:8780/v1/live/sessions` plays the 14 recorded questions in `cases/` (7 that should be delegated, 7 that should not) with a mock backend and prints whether each one was delegated, the transcript and the time to first audio.

Measured on one H100 (mock backend, 14 recorded questions):

| Engine | Lookups delegated and answered | Answered directly when it should | First audio, direct / delegated |
|---|---|---|---|
| Tag | 7 / 7 | 6 / 7 ("tell me a joke" was delegated) | about 0.4–0.6 s / 0.6–0.8 s |
| Flash | 7 / 7 | 5 / 7 ("tell me a joke" and "why is the sky blue?" were delegated) | about 0.33 s / 0.7 s |

## Limits

- Sometimes delegates a question it could answer itself, or says a filler line without delegating.
- The first session right after the engine starts may not respond; run one warm-up session (`client_example.py` does this).
- Sound tags appear in the output transcript (for example `[laughter]`), the way GPT-Live shows acoustic events.

### Release gate

`python tests/release_gate.py` (from `live/`, with pytest, httpx, websockets, fastapi, uvicorn, numpy, aiohttp installed) fails unless **node** is installed (the page behaviour tests need it), no page test is skipped, the full suite passes with `SOPHON_LIVE_*` cleared, and `tests/mutation_check_r4c.py` kills every mutant. There is no CI in this repository; run the gate before releasing.

### Blind runner (I-10 routing test)

`python -m tools.blind_runner SENTENCES OUT.jsonl` (from `live/`) records, per sentence, what the Coordinator, the JEV and the Worker did, and **never scores**. Full column list and definitions: the docstring of `tools/blind_runner.py`; behaviour and privacy are pinned by `tests/test_blind_runner.py`, and `python tests/mutation_check_blind_runner.py` must kill every mutant (run inside `tests/release_gate.py`; Critic's original M14 is provably equivalent and is reported as such).

* **Modes.** The default **fake mode** needs no server (a constant fake JEV, a fake Worker; `tools`/`tool_ok` are `null` because nothing is called). **Real mode** (`--mode real --jev-url ... [--orch-url ... --orch-model ...]`) is refused unless `BLIND_RUNNER_ALLOW_REAL=1` and is meant for after the server freeze is lifted. `--mode gateway` is unavailable: the gateway and the Realtime face take speech only.
* **Third-party traffic.** In real mode every sentence is sent to the JEV and orchestrator URLs **and to whatever tools the Worker decides to call (web search, weather, news services)**. That is the system under test's own traffic. There is no tools-offline switch; point the run only at servers you control and only use sentences you are willing to send out.
* **Who decided.** `route_via` / `control_via` / `via` / `degraded` / `degraded_reason` separate a decision of the JEV from one of the keyword tables: a JEV answer with no usable route, a CONTROL without a target, an invalid route, and a failed JEV call (`degraded_reason` timeout / conn / http_status / bad_json / runtime_error, chosen by exception class and cause only, never the message; `runtime_error` is jev_client's ">=400" answer, which the class cannot tell from another plain RuntimeError; `http_status` cannot occur with the current production `jev_client`) are all `degraded`; the refine guard is `via=guard` (not degraded, the JEV is never asked). `required_tools` records what `intent.required_tools` (a regex table in production) told the Worker to call. Two more keyword tables are made visible by wrapping the production functions for the duration of one sentence and keeping only their **return values**: `clarify_slots` (`missing_slots_for_text`, one slot list per call) and `memory_gate_explicit` (`has_explicit_save_intent`, one bool per call; only a real Worker's memory tool calls it, so it is null in fake mode). `outcome_kind` (answer / clarify / stale) and `outcome_rejected` come from `finish_job` (`handled_by` is the admit stage's handler and stays `worker` when the Worker's result later becomes a clarification or is dropped as stale: read `outcome_kind`); with `--shared-ledger` only an `answer` outcome is appended to the history, like the gateway does (last 12 kept).
* **Fidelity.** Every row carries `fidelity`, the list of ways the run differs from the production path (no front-model rewrite unless the input gives a `task`, the actual constant `playback_active=<bool>` / `language=<x>,jev_state_language=zh` (the JEV state's language is hard-wired by `build_state`), no ASR transcript, fake JEV/Worker, per-sentence ledger...). The conclusions of a blind run hold only for the Coordinator layer + JEV + orchestrator, not for the voice front or the Realtime face. Pass `--instructions-file` (the front system prompt) and `--history-jsonl` (`{"task","answer"}` lines; the last 6 are used) to get closer to production.
* **`--shared-ledger`.** By default each sentence gets a **new** Ledger, so the JEV always sees `tasks: []` and CONTROL / REFINE / CANCEL sentences **cannot be tested**. With `--shared-ledger` one Ledger (and, like the gateway, one growing history) is kept across sentences, so a sentence sees the earlier ones as **already COMPLETED** tasks: the Worker is awaited to the end, so "change/cancel a task that is still running" is not testable, and memory goes to `/dev/null`, so "remember X, then ask about it" is not testable either. Put follow-up sentences next to the one they follow in the same file.
* **Input.** UTF-8 (a BOM is ignored). Plain lines, or JSONL `{"id", "text", "task"}` where `text` must be a non-empty string; ids `s0001`... number the sentences (blank lines do not count).
* **Output.** `OUT.jsonl` must not exist (the runner never overwrites, and refuses the input file, since it exists) and is created `0600`; exit code 5 when refused. Other exit codes: 2 (bad arguments, an unreadable or invalid input file such as bad JSON or empty text, with a one-line message that names the line number only, or an output that cannot be created, e.g. no such directory), 3 (gateway mode), 4 (real mode refused), 6 (the run finished but `OUT.jsonl.done` could not be created because it appeared during the run; the output is complete, the foreign `.done` is left alone). A run that finished also writes `OUT.jsonl.done` (counts only): **no `.done` means the run was interrupted** and the output is partial. Sentence text is never logged, printed or persisted anywhere but the reply field of the output file.

The runner holds no judgment about the sentence; this is checked, not proven. `tests/test_blind_runner.py` has (1) an AST guard (name-based: it misses aliases, renamed parameters and `sys.modules['re']`), (2) a **runtime probe** that wraps each sentence in a `str` subclass whose methods record any call made from the runner's own frames (zero hits required; it does not see unbound calls like `str.find(s.text, ...)` or C-level calls like `re.search(pattern, s.text)`), and (3) a **metamorphic test**: every single character of the character set, and every sentence of the repo's labelled corpus, must produce the same row as a neutral sentence (except `reply`, `required_tools`, `clarify_slots`, which carry production's own keyword answers). A fork on a multi-character word that the corpus does not contain is not found. **Code review remains the backstop.**
