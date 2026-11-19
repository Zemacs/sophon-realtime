"""Blind-test runner for Critic's I-10 routing test (one self-contained file; touches no production code).

It feeds sentences, one per run, through the Coordinator (``coordinator.admit``: the JEV route / tool / control decision, or the model-free degraded
path when the JEV cannot answer) and, for a job, through the Worker and ``finish_job``, and RECORDS what the
system did. It never scores, never compares to an expected route, and contains no word list or regex about meaning.

Adapted to the v0.1.0 production API (rc3 removed ``heuristic_route`` / ``slot_edit`` / ``required_tools`` / ``missing_slots_for_text`` /
``has_explicit_save_intent``: the model decides, so there is no keyword function left to record, and ``via`` is ``jev`` or ``degraded``).

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
     "route"  (the Coordinator's route: WORKER|FAST|CLARIFY|CONTROL|STOP_SPEAKING),
     "decision" (NEW / REFINE / CANCEL / STATUS / the route), "jev_route", "jev_control", "jev_tool" (what the JEV answered, even when the Coordinator
     then degraded or overrode it; null = no usable answer),
     "via" (jev | degraded: production's label; degraded = the JEV could not give a route and the model-free path ran), "control_via" (production's label
     of the applied ControlDecision, or null),
     "degraded" (JEV was expected but ``via`` is degraded), "degraded_reason" (production's own reason: breaker_open | timeout | connection | error |
     non_option_answer; never derived here), "degraded_exc" (class name of the JEV exception, else null),
     "judge_invalid" (the router answers that were not an offered option, production's own list, e.g. ``tool:missing``),
     "handled_by" (fast | clarify | control | worker | stop; the ADMIT stage's handler. It stays "worker" when the Worker's result is later turned into
                   a clarification or dropped as stale -- read "outcome_kind" (answer | clarify | stale) for what happened to the job),
     "required_tools" (the tools the MODEL said the Worker must call; null when no job), "tool_judgment" (model | unknown: the router answered the tool
                      question or could not), "explicit_save" (the model's judgment, true | false | null),
     "tools" / "tool_ok" (the tools the Worker really called and their ok flags; null in fake mode: nothing was called),
     "worker_status", "outcome_kind" (finish_job's answer | clarify | stale; null when no job), "outcome_rejected" (finish_job's ``rejected`` code,
     e.g. no_tool_result / tool_failed, else null), "reply", "error" (exception class name only, else null),
     "error_stage" (admit | worker | finish)}

The JEV is wrapped for the duration of one sentence: called unchanged, only the choice answers it returns are kept (never its arguments), and it is
restored afterwards.

On an error every field obtained before the failure is kept; only ``error`` / ``error_stage`` are added.

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
  fake (default)  a constant fake JEV (``--fake-route``, default WORKER; it answers the route question and nothing else, so the tool, control and
                  review questions come back as "no usable answer" and are recorded as such) and a fake Worker that "calls" the tools the Coordinator
                  required and answers with a fixed string. No network, no model. This is what the tests use. ``tools``/``tool_ok`` are null.
  real            the real JEV (``--jev-url``) and the real Worker/orchestrator (``--orch-url``/``--orch-model``). Refused unless the
                  environment has ``BLIND_RUNNER_ALLOW_REAL=1``. Real runs are for AFTER the server freeze is lifted; do not run them before.
  gateway URL     NOT available: the GPT-Live gateway and the Realtime face take speech only (text items are ignored), so there is no text
                  turn interface to drive. ``--mode gateway`` exits with code 3 and says so.

Exit codes: 0 done; 2 bad arguments, an unreadable or invalid input file (bad JSON / empty text / duplicate id / bad history line / not valid UTF-8,
in the sentences, --instructions-file or --history-jsonl: one line on stderr naming the line NUMBER (and, for UTF-8, the byte offset), never its
content and never an id's value; line breaks (CRLF, a lone CR, LF) are counted the way the readers' text mode counts them), or an output that cannot be created (e.g. no such directory); 3 gateway mode;
4 real mode refused; 5 output refused (exists / is the input); 6 the run finished but OUT.jsonl.done could not be created (it appeared during the run).
A run that finished also writes ``OUT.jsonl.done`` (counts only); a run that was interrupted has none.

What the runner can and cannot do with a sentence (Opaque = an accident guard, NOT "unreadable")
  Every sentence and ``task`` is wrapped by ``read_sentences`` in ``Opaque``; so is every text production hands back that may quote a sentence
  (``job.objective``, the clarify/fast reply ``res.speak.text``, the Worker's ``out.text``), at the very line it is read, and so are the history
  entries (``task``/``answer``). An Opaque is not iterable, not comparable, not hashable, has no ``len``/truth value/string methods, is not
  formattable/copyable/picklable and is not a ``str``: ``re.search(p, s)``, ``json.dumps(s)``, ``bytes(s)``, ``str.find(s, x)`` (unbound), ``x in s``,
  ``s == x``, ``f"{s}"`` ... all raise ``TypeError``, C-level paths included, so a runner that tries them CRASHES instead of forking.
  The ONLY way out is ``reveal()``, and it checks its caller at run time: only the code objects of four fixed exit functions may call it
  (``_admit_request`` and ``_task_context`` and ``_worker_history`` hand text to production; ``_dump_row`` writes the JSONL line). Their bodies are
  pinned by an AST whitelist (``reveal`` plus building the one object, nothing else), as are the bodies of the reading functions (on the raw line only
  ``strip()``, ``startswith("{")`` and ``json.loads``; no ``in``/comparison/indexing on content; no module-level mutable containers) and of the spy
  and recorder that watch production. The file may not import ``gc``/``inspect``/``ctypes``/``re``..., nor name ``object``, ``globals``, ``sys.modules``,
  ``__getattribute__`` ... (Python cannot hide the wrapper's slot; these spellings are how it would be taken out).
  Still relying on REVIEW (and a human reads this file before the blind run): the production objects the runner is handed (it is the system under
  test, and it holds them as ``res``/``job``/``wr``), the structure tests' own blind spots, and whatever a test-suite change might weaken.
  ``--instructions-file`` is PLAINTEXT and not protected: it is the test author's front prompt, not a blind input. The history file's texts are
  wrapped, but they are the test author's too.

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


class Opaque:
    """A guard against ACCIDENTS (an accident guard, not a secret): a blind-test sentence, or any text production derived from one, that the runner carries but
    must not read by mistake. It is not iterable, not comparable, not hashable, has no ``len``/truth value/string methods, cannot be formatted,
    copied or pickled, and is no ``str``: ``re.search``, ``json.dumps``, ``bytes()``, an unbound ``str.find(x, ...)``, ``in``, ``==`` and every
    other way of looking inside raise ``TypeError`` (C-level paths included). Python cannot hide the slot (``object.__getattribute__``, ``gc``,
    ``ctypes`` ... get it out); tests/test_blind_runner.py forbids those spellings in this file and a person reads the file before the blind run.
    The ONE way out is ``reveal``, which only the fixed exit functions may call (checked at run time and by an AST test)."""
    __slots__ = ("_v",)

    def __init__(self, v: str | None) -> None:
        self._v = "" if v is None else v

    def _no(self, *a: Any, **k: Any):   # noqa: ANN002, ANN003
        raise TypeError("a blind-test sentence is opaque to the runner")
    __iter__ = __len__ = __bool__ = __contains__ = __getitem__ = __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _no
    __hash__ = __str__ = __format__ = __bytes__ = __reduce__ = __reduce_ex__ = __copy__ = __deepcopy__ = __add__ = __radd__ = __mod__ = __mul__ = _no
    __int__ = __float__ = __index__ = __fspath__ = __getstate__ = _no

    def __repr__(self) -> str:
        return "<Opaque sentence>"                  # neutral: no content, not even the length


def reveal(o: Opaque) -> str:
    """The only unwrap. It checks its CALLER at run time: only the code objects of the fixed exit functions (``_EXIT_FUNCTIONS``: the AdmitRequest, the
    TaskContext, the Worker's history and the JSONL row) may call it, however the call is spelled (an alias, ``globals()[...]``, a module attribute).
    Anyone else gets a TypeError. The check is by code object, not by name."""
    if type(o) is not Opaque:
        raise TypeError("reveal() takes an Opaque value")
    caller = sys._getframe(1).f_code
    for fn in _EXIT_FUNCTIONS:
        if fn.__code__ is caller:
            return o._v
    raise TypeError("reveal() may only be called from a fixed exit function")


class Sentence(NamedTuple):
    id: str
    text: Opaque
    task: Opaque | None = None


def _utf8_or_exit(path: str, what: str) -> None:
    """Refuse a file that is not valid UTF-8 with ONE line naming the position, never the content. The line number counts line breaks the way the
    readers' text mode does (``\\r\\n``, a lone ``\\r`` and ``\\n`` are one break each), so it agrees with the numbers they report."""
    with open(path, "rb") as fb:
        data = fb.read()
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        line_no = data[:e.start].replace(b"\r\n", b"\n").replace(b"\r", b"\n").count(b"\n") + 1
        raise SystemExit(f"{what} is not valid UTF-8 (line {line_no}, byte offset {e.start})")


def read_sentences(path: str) -> list[Sentence]:
    """[(id, text, task)], text and task wrapped in ``Opaque``. A line that starts with ``{`` is JSON ({"id", "text", "task"}); anything else is a
    plain sentence."""
    _utf8_or_exit(path, "input file")
    out: list[Sentence] = []
    seen: set[str] = set()
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
                item = Sentence(sid, Opaque(text), Opaque(task) if task is not None else None)
            else:
                sid = f"s{len(out) + 1:04d}"
                item = Sentence(sid, Opaque(line), None)
            if sid in seen:
                raise SystemExit(f"input line {n}: id is not unique")            # the line number, never the id's value
            seen.add(sid)
            out.append(item)
    return out


def read_history(path: str) -> list[dict[str, Opaque]]:
    """``{"task": "...", "answer": "..."}`` per line, the shape the gateway keeps in ``orch_history``, each text wrapped in ``Opaque``. Never echoed."""
    _utf8_or_exit(path, "history file")
    out: list[dict[str, Opaque]] = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                if not isinstance(o, dict) or not isinstance(o.get("task", ""), str) or not isinstance(o.get("answer", ""), str):
                    raise TypeError
                out.append({"task": Opaque(o.get("task", "")), "answer": Opaque(o.get("answer", ""))})
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


class _Recorder:
    """Remembers what the JEV answered for the current sentence (the Coordinator may then override it) and why a JEV call failed."""
    def __init__(self) -> None:
        self.route: str | None = None
        self.control: str | None = None
        self.tool: str | None = None             # the JEV's answer to the tool question (an option key or None)
        self.error_kind: str | None = None
        self.error_exc: str | None = None


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
    return {"id": sid, "mode": mode, "fidelity": fidelity, "route": None, "decision": None, "jev_route": None,
            "jev_control": None, "via": None, "control_via": None, "degraded": None, "degraded_reason": None, "degraded_exc": None, "handled_by": None,
            "jev_tool": None, "judge_invalid": None, "required_tools": None, "tool_judgment": None, "explicit_save": None, "tools": None, "tool_ok": None, "worker_status": None, "outcome_kind": None, "outcome_rejected": None, "reply": Opaque(""), "error": None, "error_stage": None}


def _control_via(res) -> str | None:   # noqa: ANN001
    """Who picked the CONTROL target that was applied (production's label), else None: a NEW job or a fixed clarify has no target."""
    return res.control.via if res.control is not None else None


def _admit_request(sid: str, text: Opaque, task: Opaque | None, language: str, playback_active: bool):
    """EXIT 1 (to production): builds the production AdmitRequest from the opaque sentence (``task`` = the front's rewrite when the input gives one;
    ``raw_text`` stays the original sentence). The body is pinned by an AST whitelist: ``reveal`` and ``C.AdmitRequest`` only."""
    from sophon_live import coordinator as C
    raw = reveal(text)
    delegate = raw if task is None else reveal(task)
    return C.AdmitRequest(session_id="blind", utterance_id=sid, text=delegate, delegate_task=delegate, language=language,
                          playback_active=playback_active, raw_text=raw)


def _task_context(job, objective: Opaque, ledger):   # noqa: ANN001
    """EXIT 2 (to production): the TaskContext the Worker runs under. AST whitelist: ``reveal`` and ``C.TaskContext`` only."""
    from sophon_live import coordinator as C
    return C.TaskContext(task_id=job.task_id, version=job.version, utterance=reveal(objective), ledger=ledger)


def _worker_history(history: list[dict[str, Opaque]]) -> list[dict[str, str]]:
    """EXIT 3 (to production): the last HISTORY_TURNS turns as the plain ``{"task","answer"}`` dicts the Worker prompt takes.
    AST whitelist: ``reveal`` and ``list.append`` only."""
    out: list[dict[str, str]] = []
    for h in history[-HISTORY_TURNS:]:
        out.append({"task": reveal(h["task"]), "answer": reveal(h["answer"])})
    return out


def _dump_row(row: dict[str, Any]) -> str:
    """EXIT 4 (to the output file): the one JSONL line of a row; the reply is the only Opaque field. AST whitelist: ``reveal`` and ``json.dumps`` only."""
    return json.dumps({**row, "reply": reveal(row["reply"])}, ensure_ascii=False)


_EXIT_FUNCTIONS = (_admit_request, _task_context, _worker_history, _dump_row)      # a tuple of the only functions reveal() serves


async def run_one(sid: str, text: Opaque, *, ledger, jev, worker, language: str = "zh", mode: str = "fake", task: Opaque | None = None,
                  instructions: str = "", history: list[dict[str, Opaque]] | None = None, playback_active: bool = False,
                  fidelity: list[str] | None = None, history_sink: list[dict[str, Opaque]] | None = None) -> dict[str, Any]:
    """Run one sentence; ALWAYS returns a row (on an error the fields obtained so far are kept, plus ``error``/``error_stage``)."""
    from sophon_live import coordinator as C
    from sophon_live.coordinator import api, intent, jev_client
    rec = _Recorder()
    row = _blank_row(sid, mode, fidelity if fidelity is not None else [])
    stage = "admit"

    async def spy(state, questions, **kw):   # noqa: ANN001
        try:
            resp = await jev(state, questions, **kw)
        except Exception as e:   # noqa: BLE001   re-raised unchanged: production behaviour is not altered
            rec.error_exc = type(e).__name__
            raise
        rec.route = jev_client.choice_answer(resp, "route") or rec.route
        rec.control = jev_client.choice_answer(resp, "control") or rec.control
        rec.tool = jev_client.choice_answer(resp, intent.TOOL_KEY) or rec.tool
        return resp

    patches = [(jev_client, "system_one", spy)]
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, f in patches:
        setattr(o, n, f)
    try:
        res = await C.admit(_admit_request(sid, text, task, language, playback_active), ledger)
        use_jev = bool(api.USE_JEV)
        row["jev_route"], row["jev_control"], row["jev_tool"] = rec.route, rec.control, rec.tool
        row["route"], row["via"] = res.route, res.via
        row["decision"] = res.control.op if res.control is not None else ("NEW" if res.job is not None else res.route)
        row["control_via"] = _control_via(res)
        row["degraded"] = bool(use_jev and res.via == "degraded")
        row["degraded_reason"] = (res.degraded_reason or None) if row["degraded"] else None     # production's own reason, never derived here
        row["degraded_exc"] = rec.error_exc if row["degraded"] else None
        row["judge_invalid"] = list(res.judge_invalid)
        if res.job is None:
            row["handled_by"] = {"STOP_SPEAKING": "stop", "FAST": "fast", "CLARIFY": "clarify"}.get(res.route, "control" if res.control else res.route.lower())
            row["reply"] = Opaque(res.speak.text)        # read here, wrapped at once: production's text may quote the sentence
            return row
        job = res.job
        objective = Opaque(job.objective)                # read here, wrapped at once
        row["handled_by"] = "worker"
        row["required_tools"] = list(job.require_tools or [])
        row["tool_judgment"], row["explicit_save"] = job.tool_judgment, job.explicit_save
        stage = "worker"
        with C.bind_task(_task_context(job, objective, ledger)):
            wr = await worker(job, instructions=instructions, history=_worker_history(history or []))
        row["worker_status"] = wr.status
        if mode != "fake":                       # a fake Worker calls nothing: its "evidence" is just the required tools echoed back
            row["tools"] = [str(e.get("tool")) for e in wr.evidence]
            row["tool_ok"] = [e.get("ok") for e in wr.evidence]
        stage = "finish"
        out = C.finish_job(ledger, job, wr)
        row["outcome_kind"], row["outcome_rejected"] = out.kind, out.rejected or None
        reply = Opaque(out.text)                         # read here, wrapped at once
        row["reply"] = reply
        if history_sink is not None and out.kind == "answer":      # like gateway.py: a clarify question or a stale result never enters the history
            history_sink.append({"task": objective, "answer": reply})
            del history_sink[:-HISTORY_KEEP]
    except Exception as e:   # noqa: BLE001   class name only: a message may quote the sentence
        row["error"], row["error_stage"] = type(e).__name__, stage
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
              history: list[dict[str, Opaque]] | None = None, language: str = "zh", playback_active: bool = False) -> int:
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
                f.write(_dump_row(row) + "\n")
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
            _utf8_or_exit(a.instructions_file, "instructions file")
            with open(a.instructions_file, encoding="utf-8-sig") as fh:
                instructions = fh.read()
        history = read_history(a.history_jsonl) if a.history_jsonl else []
    except OSError as e:
        print(f"blind_runner: cannot read an input file ({type(e).__name__})", file=sys.stderr)
        return EXIT_BAD_ARGS
    except SystemExit as e:                      # the readers' own one-line messages ("input line N: ...", duplicate ids, bad history): never the line itself
        if isinstance(e.code, str):
            print("blind_runner: " + e.code, file=sys.stderr)
            return EXIT_BAD_ARGS
        raise
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
