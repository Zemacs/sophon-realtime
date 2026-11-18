"""Blind-test runner for Critic's I-10 routing test (one self-contained file; touches no production code).

It feeds sentences, one per run, through the Coordinator (``coordinator.admit``: the JEV route decision, the guard, the heuristic
fallback) and, for a job, through the Worker, and RECORDS what the system did. It never scores, never compares to an expected
route, and contains no word list or regex about meaning.

Usage (from ``live/``)::

    python -m tools.blind_runner SENTENCES.txt OUT.jsonl                       # fake mode (default): nothing leaves the process
    python -m tools.blind_runner SENTENCES.jsonl OUT.jsonl --fake-route FAST   # a different constant fake JEV answer
    BLIND_RUNNER_ALLOW_REAL=1 python -m tools.blind_runner S.txt OUT.jsonl --mode real --jev-url http://HOST:8110 \\
        --orch-url http://HOST:8101/v1/chat/completions [--instructions-file F] [--history-jsonl H] [--shared-ledger]

Input (read as UTF-8, a BOM is ignored): a text file (one sentence per line, blank lines skipped; ids ``s0001``... number the SENTENCES, not the
file lines) or JSONL, one ``{"id": "<opaque>", "text": "...", "task": "..."}`` per line (a line starting with ``{`` is JSON). ``text`` must be a
non-empty string; ``id`` and ``task`` are optional (``task`` = the front model's rewritten task description, see Fidelity).
Output, one JSON object per sentence, written ONLY to OUT.jsonl (which must not exist yet, and must not be the input; created 0600)::

    {"id", "mode" (fake|real), "fidelity" (list of codes, see below),
     "route"  (the Coordinator's route: WORKER|FAST|CLARIFY|CONTROL|STOP_SPEAKING), "route_via" (who decided the route, see below),
     "decision" (NEW / REFINE / CANCEL / STATUS / the route), "jev_route", "jev_control" (what the JEV answered, even if a guard overrode it),
     "via" (jev | guard | heuristic: the production label), "control_via" (who picked the CONTROL target, or null),
     "degraded" (JEV was expected but something else decided), "degraded_reason", "degraded_exc" (class name of the JEV exception, else null),
     "handled_by" (fast | clarify | control | worker | stop),
     "required_tools" (what ``intent.required_tools`` -- a regex table in production -- told the Worker it must call; null when no job),
     "clarify_slots" (RETURN VALUES of ``clarify.missing_slots_for_text`` -- a keyword test in production -- one slot list per call, null if it was
                      not called), "memory_gate_explicit" (return values of ``memory_gate.has_explicit_save_intent`` -- a keyword table -- one bool per
                      call; only a REAL Worker's memory tool calls it, so it is null in fake mode),
     "tools" / "tool_ok" (the tools the Worker really called and their ok flags; null in fake mode: nothing was called),
     "worker_status", "outcome_kind" (finish_job's answer | clarify | stale; null when no job), "outcome_rejected" (finish_job's ``rejected`` code,
     e.g. no_tool_result / tool_failed, else null), "reply", "error" (exception class name only, else null), "error_stage" (admit | worker | finish)}

The two keyword functions are wrapped for the duration of one sentence exactly like ``jev_client.system_one`` is: called unchanged, only their
return values are kept (never their arguments), and restored afterwards.

On an error every field obtained before the failure is kept; only ``error`` / ``error_stage`` are added.

Who decided (all derived from structure, never from the sentence)
  route_via     guard      the refine guard (``slot_edit``) answered before the JEV was asked (``jev_route`` is then null)
                jev        the JEV answered one of the five route enums (the keys of ``route.ROUTE_CRITERIA``)
                heuristic  the JEV failed or is switched off, or answered no route: ``heuristic_route`` (keyword tables) decided
                jev_invalid  the JEV answered something that is not a route enum; production falls through to a WORKER job
  control_via   jev | heuristic | guard (the production label of the applied ControlDecision); when the JEV said CONTROL but nothing was applied:
                jev (it said NEW or named a task the Ledger does not have) or heuristic (``heuristic_control`` ran and found nothing)
  degraded      the JEV was expected (SOPHON_LIVE_JEV on) and route_via is heuristic/jev_invalid, or control_via is heuristic, or via is heuristic.
                A guard decision is NOT degraded (the JEV was never asked) -- read ``route_via``.
  degraded_reason  the JEV call raised (chosen by exception CLASS and cause only, never the message; ``degraded_exc`` has the class name):
                timeout | conn | http_status (aiohttp.ClientResponseError) | bad_json (RuntimeError with a cause: the unparsable body) |
                runtime_error (a RuntimeError without a cause: jev_client's >=400 answer -- and any other plain RuntimeError, the class cannot
                tell them apart) | jev_error (anything else);  or, without an exception: no_usable_route | invalid_route | control_no_target | null

Fidelity: what differs from the production path (every row lists the codes that apply to this run)
  fake_jev, fake_worker      nothing real was asked
  no_frontend_rewrite        the sentence is also the task description (production: the voice front model rewrites it); with ``task`` in the
                             input the rewrite is supplied by the test author instead: rewrite_from_input (not a real front model)
  no_instructions / instructions_from_file     the front system prompt given to the Worker
  no_history / history_static / history_accumulates     the Worker's recent turns (last 6 are passed). static = from --history-jsonl each
                             sentence; accumulates = --shared-ledger, each job whose outcome is an answer is appended (a clarify question or a stale result is not,
                             last 12 kept) like the gateway does
  playback_active=<bool>, language=<x>,jev_state_language=zh     the ACTUAL values, constant for the whole run (production's follow the call);
                             ``--language`` only changes the job's language, the JEV state's language is hard-wired "zh" by ``build_state``
  no_asr_transcript          raw_text == text; the Realtime face's speech-transcript grounding (``heard``) is not exercised
  ledger_per_sentence / ledger_shared     a new Ledger for every sentence (the JEV sees no tasks, CONTROL cannot be tested) / one Ledger

Modes
  fake (default)  a constant fake JEV (``--fake-route``, default WORKER) and a fake Worker that "calls" the tools the Coordinator required
                  and answers with a fixed string. No network, no model. This is what the tests use. ``tools``/``tool_ok`` are null.
  real            the real JEV (``--jev-url``) and the real Worker/orchestrator (``--orch-url``/``--orch-model``). Refused unless the
                  environment has ``BLIND_RUNNER_ALLOW_REAL=1``. Real runs are for AFTER the server freeze is lifted; do not run them before.
  gateway URL     NOT available: the GPT-Live gateway and the Realtime face take speech only (text items are ignored), so there is no text
                  turn interface to drive. ``--mode gateway`` exits with code 3 and says so.

Exit codes: 0 done; 2 bad arguments, an unreadable input file, or an output that cannot be created (e.g. no such directory); 3 gateway mode;
4 real mode refused; 5 output refused (exists / is the input); 6 the run finished but OUT.jsonl.done could not be created (it appeared during the run).
A run that finished also writes ``OUT.jsonl.done`` (counts only); a run that was interrupted has none.

Privacy (what the runner guarantees, and the tests that pin it: ``tests/test_blind_runner.py::Privacy``)
  * sentence text is written to no place except the reply field of OUT.jsonl (the sentence itself is not even echoed there);
  * all logging is switched off for the run (``logging.disable(CRITICAL)``), warnings are ignored, and the runner prints only counts;
  * the memory store is pointed at the null device, the device hook is unset, and errors are recorded as a class name only;
  * REAL mode necessarily sends sentences to the JEV / orchestrator URLs and to the tools those use (web search, weather, news): that is the
    system under test's own traffic, not the runner's; point it only at servers you control. There is no tools-offline switch.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import warnings
from typing import Any, Iterable, NamedTuple

ALLOW_REAL_ENV = "BLIND_RUNNER_ALLOW_REAL"
FAKE_REPLY = "[fake worker reply]"
_HOOK_ENV_NAMES = ("SOPHON_LIVE_DEVICE_HOOK",)
HISTORY_TURNS = 6          # the Worker prompt uses the last 6 turns (orchestrator._history_block)
HISTORY_KEEP = 12          # the gateway keeps the last 12 (gateway.py)
EXIT_BAD_ARGS, EXIT_GATEWAY, EXIT_REAL_REFUSED, EXIT_OUT_REFUSED, EXIT_DONE_FAILED = 2, 3, 4, 5, 6


class Sentence(NamedTuple):
    id: str
    text: str
    task: str | None = None


def read_sentences(path: str) -> list[Sentence]:
    """[(id, text, task)]. A line that starts with ``{`` is JSON ({"id", "text", "task"}); anything else is a plain sentence."""
    out: list[Sentence] = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    o = json.loads(line)
                    text, task = o["text"], o.get("task")
                    if not isinstance(text, str) or not text.strip() or (task is not None and (not isinstance(task, str) or not task.strip())):
                        raise TypeError("text/task must be non-empty strings")
                    sid = str(o["id"]) if o.get("id") is not None else f"s{len(out) + 1:04d}"
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise SystemExit(f"input line {n}: expected JSON with a non-empty string 'text' (optional string 'task')")   # never echo the line
                out.append(Sentence(sid, text, task))
            else:
                out.append(Sentence(f"s{len(out) + 1:04d}", line, None))
    ids = [s.id for s in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("input ids are not unique")
    return out


def read_history(path: str) -> list[dict[str, str]]:
    """``{"task": "...", "answer": "..."}`` per line, the shape the gateway keeps in ``orch_history``. Never echoed."""
    out: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                if not isinstance(o, dict) or not isinstance(o.get("task", ""), str) or not isinstance(o.get("answer", ""), str):
                    raise TypeError
                out.append({"task": o.get("task", ""), "answer": o.get("answer", "")})
            except (ValueError, TypeError):
                raise SystemExit(f"history line {n}: expected JSON {{\"task\": str, \"answer\": str}}")
    return out


@contextlib.contextmanager
def quiet_and_sealed(real: bool, orch_url: str | None = None, orch_model: str | None = None):
    """No logging, no warnings, no memory writes, no device hook, for the duration of the run (everything restored after)."""
    saved_env = {k: os.environ.get(k) for k in ("SOPHON_LIVE_MEMORY_PATH", *_HOOK_ENV_NAMES, "SOPHON_LIVE_ORCH_URL", "SOPHON_LIVE_ORCH_MODEL")}
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["SOPHON_LIVE_MEMORY_PATH"] = os.devnull
            for k in _HOOK_ENV_NAMES:
                os.environ.pop(k, None)
            if orch_url:
                os.environ["SOPHON_LIVE_ORCH_URL"] = orch_url
            if orch_model:
                os.environ["SOPHON_LIVE_ORCH_MODEL"] = orch_model
            yield
    finally:
        logging.disable(prev_disable)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def jev_error_kind(e: BaseException) -> str:
    """Why a JEV call failed, by exception CLASS and cause only (the message may quote the sentence and is never read).
    timeout | conn | http_status (aiohttp.ClientResponseError) | bad_json (a RuntimeError that has a cause: jev_client raises the unparsable-body
    error ``from`` the parse error) | runtime_error (a RuntimeError without a cause: what jev_client raises for a >=400 answer, and also any other
    plain RuntimeError -- the class cannot tell them apart) | jev_error (anything else)."""
    import aiohttp
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(e, (aiohttp.ClientConnectionError, OSError)):
        return "conn"
    if isinstance(e, aiohttp.ClientResponseError):
        return "http_status"
    if isinstance(e, RuntimeError):
        return "bad_json" if e.__cause__ is not None else "runtime_error"
    return "jev_error"


class _Recorder:
    """Remembers what the JEV answered for the current sentence (the Coordinator may then override it) and why a JEV call failed."""
    def __init__(self) -> None:
        self.route: str | None = None
        self.control: str | None = None
        self.error_kind: str | None = None
        self.error_exc: str | None = None
        self.slots: list[list[str]] | None = None        # return values of api.missing_slots_for_text, one per call
        self.save_intent: list[bool] | None = None       # return values of memory_gate.has_explicit_save_intent, one per call


async def _fake_jev(route: str):
    async def system_one(state, questions, *, url=None, model=None):   # noqa: ANN001, ARG001
        return {"answers": {"route": {"type": "choice", "choice": route}}}
    return system_one


async def _fake_worker(job, *, instructions="", history=None):   # noqa: ANN001, ARG001
    from sophon_live.coordinator.schemas import WorkerResult
    ev = [{"tool": t, "ok": True} for t in (job.require_tools or [])]
    return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=FAKE_REPLY,
                        evidence=ev, freshness_token=job.freshness_token)


def fidelity_codes(*, mode: str, has_task: bool, instructions: str, history_mode: str, shared_ledger: bool,
                   playback_active: bool = False, language: str = "zh") -> list[str]:
    # the ACTUAL values, constant for the whole run (production's change with the call); the JEV state's language is hard-wired "zh" in build_state
    codes = [f"playback_active={str(bool(playback_active)).lower()}", f"language={language},jev_state_language=zh", "no_asr_transcript"]
    codes.append("rewrite_from_input" if has_task else "no_frontend_rewrite")
    codes.append("instructions_from_file" if instructions else "no_instructions")
    codes.append(history_mode)
    codes.append("ledger_shared" if shared_ledger else "ledger_per_sentence")
    if mode == "fake":
        codes += ["fake_jev", "fake_worker"]
    return sorted(codes)


def _blank_row(sid: str, mode: str, fidelity: list[str]) -> dict[str, Any]:
    return {"id": sid, "mode": mode, "fidelity": fidelity, "route": None, "route_via": None, "decision": None, "jev_route": None,
            "jev_control": None, "via": None, "control_via": None, "degraded": None, "degraded_reason": None, "degraded_exc": None, "handled_by": None,
            "required_tools": None, "clarify_slots": None, "memory_gate_explicit": None, "tools": None, "tool_ok": None, "worker_status": None, "outcome_kind": None, "outcome_rejected": None, "reply": "", "error": None, "error_stage": None}


def _route_via(res, jev_route: str | None, use_jev: bool, enums: set[str]) -> str:   # noqa: ANN001
    if res.via == "guard":
        return "guard"
    if res.via == "heuristic" or not use_jev:
        return "heuristic"
    if not jev_route:
        return "heuristic"                      # api: ``choice_answer(...) or heuristic_route(...)``
    return "jev" if jev_route in enums else "jev_invalid"


def _control_via(res, jev_route: str | None, jev_control: str | None) -> str | None:   # noqa: ANN001
    from sophon_live.coordinator import api
    if res.control is not None:
        return res.control.via
    if jev_route != "CONTROL" or res.via != "jev":
        return None
    if jev_control == "NEW" or api._parse_control_key(jev_control) is not None:
        return "jev"                            # said NEW, or named a task the Ledger does not have: a new job, JEV's call
    return "heuristic"                          # no usable target: heuristic_control ran and found nothing


async def run_one(sid: str, text: str, *, ledger, jev, worker, language: str = "zh", mode: str = "fake", task: str | None = None,
                  instructions: str = "", history: list[dict[str, str]] | None = None, playback_active: bool = False,
                  fidelity: list[str] | None = None, history_sink: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Run one sentence; ALWAYS returns a row (on an error the fields obtained so far are kept, plus ``error``/``error_stage``)."""
    from sophon_live import coordinator as C
    from sophon_live.coordinator import api, jev_client, memory_gate
    from sophon_live.coordinator.route import ROUTE_CRITERIA
    rec = _Recorder()
    row = _blank_row(sid, mode, fidelity if fidelity is not None else [])
    stage = "admit"

    async def spy(state, questions, **kw):   # noqa: ANN001
        try:
            resp = await jev(state, questions, **kw)
        except Exception as e:   # noqa: BLE001   re-raised unchanged: production behaviour is not altered
            rec.error_kind, rec.error_exc = jev_error_kind(e), type(e).__name__
            raise
        rec.route = jev_client.choice_answer(resp, "route")
        rec.control = jev_client.choice_answer(resp, "control")
        return resp

    def recording(fn, sink: str):   # noqa: ANN001   wraps a production keyword function: calls it unchanged, keeps ONLY its return value
        def wrapper(*a, **k):   # noqa: ANN002, ANN003
            r = fn(*a, **k)
            if sink == "slots":
                rec.slots = (rec.slots or []) + [list(r or [])]
            else:
                rec.save_intent = (rec.save_intent or []) + [bool(r)]
            return r
        return wrapper

    patches = [(jev_client, "system_one", spy), (api, "missing_slots_for_text", recording(api.missing_slots_for_text, "slots")),
               (memory_gate, "has_explicit_save_intent", recording(memory_gate.has_explicit_save_intent, "save"))]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, f in patches:
        setattr(o, n, f)
    try:
        delegate = task if task is not None else text
        res = await C.admit(C.AdmitRequest(session_id="blind", utterance_id=sid, text=delegate, delegate_task=delegate, language=language,
                                           playback_active=playback_active, raw_text=text), ledger)
        use_jev = bool(api.USE_JEV)
        row["jev_route"], row["jev_control"] = rec.route, rec.control
        row["route"], row["via"] = res.route, res.via
        row["decision"] = res.control.op if res.control is not None else ("NEW" if res.job is not None else res.route)
        row["route_via"] = _route_via(res, rec.route, use_jev, set(ROUTE_CRITERIA))
        row["control_via"] = _control_via(res, rec.route, rec.control)
        reason = rec.error_kind
        if reason is None and use_jev and res.via != "guard":
            if row["route_via"] == "heuristic":
                reason = "no_usable_route"
            elif row["route_via"] == "jev_invalid":
                reason = "invalid_route"
            elif row["control_via"] == "heuristic":
                reason = "control_no_target"
        row["degraded_reason"], row["degraded_exc"] = reason, rec.error_exc
        row["degraded"] = bool(use_jev and (row["route_via"] in ("heuristic", "jev_invalid") or row["control_via"] == "heuristic"
                                            or res.via == "heuristic" or rec.error_kind is not None))
        row["clarify_slots"] = rec.slots
        if res.job is None:
            row["handled_by"] = {"STOP_SPEAKING": "stop", "FAST": "fast", "CLARIFY": "clarify"}.get(res.route, "control" if res.control else res.route.lower())
            row["reply"] = res.speak.text or ""
            return row
        job = res.job
        row["handled_by"] = "worker"
        row["required_tools"] = list(job.require_tools or [])
        stage = "worker"
        with C.bind_task(C.TaskContext(task_id=job.task_id, version=job.version, utterance=job.objective, ledger=ledger)):
            wr = await worker(job, instructions=instructions, history=list((history or [])[-HISTORY_TURNS:]))
        row["worker_status"] = wr.status
        if mode != "fake":                       # a fake Worker calls nothing: its "evidence" is just the required tools echoed back
            row["tools"] = [str(e.get("tool")) for e in wr.evidence]
            row["tool_ok"] = [e.get("ok") for e in wr.evidence]
        row["memory_gate_explicit"] = rec.save_intent
        stage = "finish"
        out = C.finish_job(ledger, job, wr)
        row["outcome_kind"], row["outcome_rejected"] = out.kind, out.rejected or None
        row["reply"] = out.text
        row["clarify_slots"] = rec.slots
        if history_sink is not None and out.kind == "answer":      # like gateway.py: a clarify question or a stale result never enters the history
            history_sink.append({"task": job.objective, "answer": out.text})
            del history_sink[:-HISTORY_KEEP]
    except Exception as e:   # noqa: BLE001   class name only: a message may quote the sentence
        row["error"], row["error_stage"] = type(e).__name__, stage
        row["clarify_slots"], row["memory_gate_explicit"] = rec.slots, rec.save_intent
    finally:
        for o, n, v in saved:
            setattr(o, n, v)
    return row


def output_refused(out_path: str) -> bool:
    """True when OUT / OUT.done already exists (a symlink counts as existing). The input file always exists, so OUT == the input -- by any
    spelling, link or hard link -- is refused by the same test: the runner never overwrites."""
    return any(os.path.lexists(p) for p in (out_path, out_path + ".done"))


def open_output(out_path: str) -> int:
    """Create OUT exclusively with mode 0600. Refuses (SystemExit 5) when it is the input or already exists; never truncates or appends."""
    if output_refused(out_path):
        raise SystemExit(EXIT_OUT_REFUSED)
    return os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


async def run(sentences: Iterable[Sentence], out_path: str, *, mode: str = "fake", fake_route: str = "WORKER",
              jev_url: str | None = None, shared_ledger: bool = False, instructions: str = "",
              history: list[dict[str, str]] | None = None, language: str = "zh", playback_active: bool = False) -> int:
    from sophon_live import coordinator as C
    from sophon_live.coordinator import api, jev_client
    if mode == "fake":
        jev = await _fake_jev(fake_route)
        worker = _fake_worker
        use_jev = True
    else:
        real_system_one = jev_client.system_one

        async def jev(state, questions, **kw):   # noqa: ANN001
            return await real_system_one(state, questions, url=jev_url, **kw)
        from sophon_live.assistant_tools import run_worker_job as worker
        use_jev = api.USE_JEV
    base_history = list(history or [])
    history_mode = "history_accumulates" if shared_ledger else ("history_static" if base_history else "no_history")
    n = n_err = 0
    ledger = C.Ledger()
    run_history = list(base_history)
    prev_use = api.USE_JEV
    api.USE_JEV = use_jev
    fd = open_output(out_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for s in sentences:
                if not shared_ledger:
                    ledger = C.Ledger()
                fid = fidelity_codes(mode=mode, has_task=s.task is not None, instructions=instructions, history_mode=history_mode,
                                     shared_ledger=shared_ledger, playback_active=playback_active, language=language)
                row = await run_one(s.id, s.text, ledger=ledger, jev=jev, worker=worker, language=language, mode=mode, task=s.task,
                                    instructions=instructions, history=run_history, playback_active=playback_active, fidelity=fid,
                                    history_sink=run_history if shared_ledger else None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                n_err += 1 if row["error"] else 0
        try:
            done_fd = os.open(out_path + ".done", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:                  # someone else created it during the run: it is not ours, so it proves nothing about THIS run
            raise SystemExit(EXIT_DONE_FAILED)
        with os.fdopen(done_fd, "w", encoding="utf-8") as g:
            json.dump({"n": n, "errors": n_err, "mode": mode, "complete": True}, g)
    finally:
        api.USE_JEV = prev_use
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="blind_runner", description="Record what the system does with each sentence (no scoring).")
    ap.add_argument("sentences"); ap.add_argument("out")
    ap.add_argument("--mode", choices=("fake", "real", "gateway"), default="fake")
    ap.add_argument("--fake-route", default="WORKER", help="fake mode: the constant route the fake JEV answers (one of the five route enums)")
    ap.add_argument("--jev-url"); ap.add_argument("--orch-url"); ap.add_argument("--orch-model")
    ap.add_argument("--shared-ledger", action="store_true", help="keep one ledger across the sentences (they then see each other as earlier, COMPLETED turns)")
    ap.add_argument("--instructions-file", help="the front model's system prompt, passed to the Worker as production does (read only, never echoed)")
    ap.add_argument("--history-jsonl", help='recent turns {"task","answer"} per line, passed to the Worker (last 6; read only, never echoed)')
    ap.add_argument("--language", default="zh"); ap.add_argument("--playback-active", action="store_true")
    a = ap.parse_args(argv)
    from sophon_live.coordinator.route import ROUTE_CRITERIA
    if a.fake_route not in ROUTE_CRITERIA:
        print(f"blind_runner: --fake-route must be one of {sorted(ROUTE_CRITERIA)}", file=sys.stderr)
        return EXIT_BAD_ARGS
    if a.mode == "gateway":
        print("blind_runner: no text-turn interface exists on the gateway or the Realtime face (speech only); use fake or real mode", file=sys.stderr)
        return EXIT_GATEWAY
    if a.mode == "real":
        if os.environ.get(ALLOW_REAL_ENV) != "1":
            print(f"blind_runner: real mode needs {ALLOW_REAL_ENV}=1 (only after the server freeze is lifted)", file=sys.stderr)
            return EXIT_REAL_REFUSED
        if not a.jev_url:
            print("blind_runner: real mode needs --jev-url", file=sys.stderr)
            return EXIT_REAL_REFUSED
    try:
        sents = read_sentences(a.sentences)
        instructions = ""
        if a.instructions_file:
            with open(a.instructions_file, encoding="utf-8-sig") as fh:
                instructions = fh.read()
        history = read_history(a.history_jsonl) if a.history_jsonl else []
    except OSError as e:
        print(f"blind_runner: cannot read an input file ({type(e).__name__})", file=sys.stderr)
        return EXIT_BAD_ARGS
    try:
        with quiet_and_sealed(a.mode == "real", a.orch_url, a.orch_model):
            n = asyncio.run(run(sents, a.out, mode=a.mode, fake_route=a.fake_route, jev_url=a.jev_url, shared_ledger=a.shared_ledger,
                                instructions=instructions, history=history, language=a.language, playback_active=a.playback_active))
    except SystemExit as e:
        if e.code == EXIT_OUT_REFUSED:
            print("blind_runner: refusing to write the output: it already exists (nothing was changed)", file=sys.stderr)
            return EXIT_OUT_REFUSED
        if e.code == EXIT_DONE_FAILED:
            print("blind_runner: the run finished and OUT.jsonl is complete, but OUT.jsonl.done could not be created because it appeared during the run;"
                  " it is not this run's completion marker", file=sys.stderr)
            return EXIT_DONE_FAILED
        raise
    except FileExistsError:                      # lost a race for the output file itself
        print("blind_runner: refusing to write the output: it already exists (nothing was changed)", file=sys.stderr)
        return EXIT_OUT_REFUSED
    except OSError as e:                         # e.g. the output directory does not exist, or is not writable
        print(f"blind_runner: cannot write the output file ({type(e).__name__}); is the output directory there and writable?", file=sys.stderr)
        return EXIT_BAD_ARGS
    print(f"blind_runner: {n} sentences -> output file ({a.mode} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
