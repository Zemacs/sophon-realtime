"""Coordinator admit(): Frontend → route/ledger → SpeakPlan and/or WorkerJob."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from .. import model_client
from . import intent, jev_client
from .clarify import MISSING_SLOT_KEY, build_missing_slot_question, clarify_for_slots, slots_from_answer
from .ledger import Ledger
from .route import build_route_questions, build_state
from .schemas import AdmitRequest, ControlDecision, PlaybackReceipt, SpeakPlan, TaskStatus, WorkerJob, WorkerResult, freshness_token_for
from .schemas import FAILED_GENERIC, FIXED_FAILURES, NO_TOOL_ANSWER, is_tool_failure_sentence, tool_succeeded

log = logging.getLogger("sophon_live.coordinator")

USE_JEV = os.environ.get("SOPHON_LIVE_JEV", "1").strip().lower() in ("1", "true", "yes", "on")


_ROUTES = frozenset({"STOP_SPEAKING", "CLARIFY", "FAST", "CONTROL", "WORKER"})
# Fixed, model-free sentence used when the router is unavailable and a task is already live: asking is safer than guessing.
DEGRADED_CLARIFY = "我没完全听明白，你是想改刚才那件事，还是问一件新的事？"


@dataclass
class AdmitResult:
    speak: SpeakPlan
    job: WorkerJob | None = None
    route: str = "FAST"
    ledger: Ledger | None = None
    control: ControlDecision | None = None  # set when a CONTROL op was applied (or refused)
    stale_task: tuple[str, int] | None = None  # (task_id, version) whose in-flight worker must stop
    via: str = "heuristic"  # who decided: guard | jev | heuristic
    jev_ms: float | None = None  # Karpo round trip, when JEV was asked
    judge_invalid: list = field(default_factory=list)   # RC2-8: router answers that were not an offered option ("tool:invalid", "tool:missing")
    degraded_reason: str = ""  # via=degraded: breaker_open | timeout | connection | error | non_option_answer | jev_disabled


async def _clarify_text(text: str, language: str | None = None) -> str:
    """The question to ask for a CLARIFY route: the MODEL names the missing slot (clarify.MISSING_SLOT_KEY); unknown -> generic ask."""
    slots: list[str] = []
    try:
        resp = await jev_client.system_one({"utterance": text, **({"language": language} if language else {})}, {MISSING_SLOT_KEY: build_missing_slot_question()})
        slots = slots_from_answer(jev_client.choice_answer(resp, MISSING_SLOT_KEY))
    except Exception as e:  # noqa: BLE001
        log.warning("missing-slot judgment unavailable: %r", e)
    return clarify_for_slots(slots)


def _clarify_text_for_task(objective: str) -> str:
    """Re-asking for a task that already waits for input: the Worker's own report named the slots; the generic ask is the safe repeat."""
    return clarify_for_slots(None)


def _parse_control_key(key: str | None) -> ControlDecision | None:
    if not key or key == "NEW":
        return None
    parts = key.split(":")
    if len(parts) != 3 or parts[0] not in ("CANCEL", "STATUS", "REFINE"):
        return None
    try:
        ver = int(parts[2])
    except ValueError:
        return None
    return ControlDecision(op=parts[0], task_id=parts[1], version=ver, via="jev")  # type: ignore[arg-type]


def _decision_label(res: "AdmitResult") -> str:
    if res.control is not None:
        return res.control.op
    if res.job is not None:
        return "NEW"
    return res.route


async def admit(req: AdmitRequest, ledger: Ledger) -> AdmitResult:
    """Frontend → Coordinator. Wraps the routing with one structured, per-utterance decision log line."""
    meta: dict = {"via": "heuristic", "jev_ms": None}
    t0 = time.monotonic()
    res = await _admit(req, ledger, meta)
    res.via = meta["via"]
    res.jev_ms = meta["jev_ms"]
    res.degraded_reason = meta.get("degraded_reason", "")
    res.judge_invalid = list(meta.get("judge_invalid", []))
    log.info(
        "coord admit session=%s utt=%r route=%s decision=%s via=%s%s admit_ms=%d jev_ms=%s",
        req.session_id, (req.delegate_task or req.text or "")[:60], res.route, _decision_label(res), res.via, (":" + res.degraded_reason) if res.degraded_reason else "",
        (time.monotonic() - t0) * 1000, "-" if meta["jev_ms"] is None else int(meta["jev_ms"]),
    )
    return res


# RC2-8: how often the router model answers a question with something that is not an offered option (visible in /health via JUDGE_STATS
# and per turn in the trace as ``judge_invalid``); the code never guesses a replacement.
JUDGE_STATS: dict[str, int] = {}


def _invalid(question: str, answer: object, meta: dict) -> None:
    kind = "missing" if answer is None else "invalid"
    JUDGE_STATS[f"{question}_{kind}"] = JUDGE_STATS.get(f"{question}_{kind}", 0) + 1
    meta.setdefault("judge_invalid", []).append(f"{question}:{kind}")
    log.warning("router gave a %s answer to the %r question (%r): treated as unknown, not guessed", kind, question, str(answer)[:40])


async def _admit(req: AdmitRequest, ledger: Ledger, meta: dict) -> AdmitResult:
    text = (req.delegate_task or req.text or "").strip()
    # JEV decides the route. When it cannot (down, slow, breaker open, or an answer that is not one of the offered options)
    # the Coordinator never guesses from keywords: it degrades to a model-free safe behaviour (see _degrade).
    route: str | None = None
    control_key: str | None = None
    tools: list[str] | None = None            # None = the model gave no tool answer (unknown)
    explicit_save: bool | None = None         # the model's judgment: did the user explicitly ask to remember it long-term? None = unknown
    reason = "jev_disabled"
    if USE_JEV:
        try:
            state = build_state(text, ledger, playback_active=req.playback_active)
            questions = build_route_questions(ledger)
            # the model, not a word list, says which tool the request needs (and whether the user explicitly asked to keep something long-term).
            # It reads the user's own words (``user_said``) when the front rewrote the task; the tool question tells it to trust those.
            if (req.raw_text or "").strip():
                state["user_said"] = req.raw_text.strip()
            questions[intent.TOOL_KEY] = intent.build_tool_question()
            t_jev = time.monotonic()
            resp = await jev_client.system_one(state, questions)
            meta["jev_ms"] = (time.monotonic() - t_jev) * 1000
            route = jev_client.choice_answer(resp, "route")
            if route in _ROUTES:
                control_key = jev_client.choice_answer(resp, "control")
                tool_choice = jev_client.choice_answer(resp, intent.TOOL_KEY)
                tools = intent.tools_from_answer(tool_choice)
                explicit_save = intent.explicit_save_from_answer(tool_choice)
                # RC2-8: an answer that is not one of the offered options is never guessed AND never silent: it is counted and traced
                if tools is None:
                    _invalid("tool", tool_choice, meta)
                meta["via"] = "jev"
                log.info("jev route=%s control=%s utterance=%r", route, control_key, text[:80])
            else:
                route, reason = None, "non_option_answer"
        except model_client.ModelUnavailable as e:
            reason = e.reason
            log.warning("jev unavailable (%s): %s", e.reason, e)
        except Exception as e:  # noqa: BLE001
            reason = "error"
            log.warning("jev call failed: %r", e)
    if route is None:
        return _degrade(text, req, ledger, meta, reason)

    if route == "STOP_SPEAKING":
        return AdmitResult(
            speak=SpeakPlan(kind="stop_speaking", text="", barge_policy="drop_playback"),
            route=route,
            ledger=ledger,
        )

    if route == "CLARIFY":
        # optional: still register a NEEDS_INPUT task shell
        rec = ledger.create(text)
        ledger.mark_needs_input(rec.task_id)
        return AdmitResult(
            speak=SpeakPlan(
                kind="clarify",
                text=await _clarify_text(text, req.language),
                task_id=rec.task_id,
                version=rec.version,
            ),
            route=route,
            ledger=ledger,
        )

    if route == "FAST":
        return AdmitResult(
            speak=SpeakPlan(kind="noop", text=""),
            route=route,
            ledger=ledger,
        )

    if route in ("CONTROL", "WORKER"):
        # The model's ``control`` answer names which live task (if any) this utterance edits / cancels / asks about. A rewritten full
        # sentence ("明天北京会下雨吗" after "…上海…") is the model's call too, so a REFINE option is honoured whichever of the two
        # routes it came with; the code only enforces (task exists, version current, state machine) in _apply_control.
        decision = _parse_control_key(control_key)
        if decision is not None:
            return _apply_control(decision, text, req, ledger, tools, explicit_save)
    if route == "CONTROL":
        if control_key != "NEW":
            # JEV said CONTROL but named no usable target: only the router could tell what to do, so ask (model-free, fixed sentence).
            log.warning("route=CONTROL without a usable control option (%r): fixed clarify", control_key)
            return AdmitResult(speak=SpeakPlan(kind="clarify", text=DEGRADED_CLARIFY), route="CLARIFY", ledger=ledger)

    # WORKER or CONTROL→NEW or CONTROL with nothing to control
    job = _new_job(ledger, text, language=req.language, tools=tools, explicit_save=explicit_save)
    return AdmitResult(
        speak=SpeakPlan(
            kind="pad",
            text=intent.pad_for_tools(tools),
            task_id=job.task_id,
            version=job.version,
        ),
        job=job,
        route="WORKER",
        ledger=ledger,
    )


def _degrade(text: str, req: AdmitRequest, ledger: Ledger, meta: dict, reason: str) -> AdmitResult:
    """Router unavailable: model-free safe behaviour, no keyword semantics.

    No live task -> the utterance is a new task for the orchestrator (a model reads it). A live task -> one fixed
    clarify sentence, because only the router could tell a refinement/cancel from a new request.
    """
    meta["via"] = "degraded"
    meta["degraded_reason"] = reason
    if ledger.active():
        log.warning("coord degraded (%s): live task -> fixed clarify", reason)
        return AdmitResult(speak=SpeakPlan(kind="clarify", text=DEGRADED_CLARIFY), route="CLARIFY", ledger=ledger)
    log.warning("coord degraded (%s): no live task -> new task for orch", reason)
    job = _new_job(ledger, text, language=req.language, tools=None)      # router down: nothing is forced and the judgment is recorded as unknown
    return AdmitResult(
        speak=SpeakPlan(kind="pad", text="", task_id=job.task_id, version=job.version),
        job=job, route="WORKER", ledger=ledger,
    )


def _answer(text: str, ledger: Ledger, decision: ControlDecision, *, version: int | None = None) -> AdmitResult:
    return AdmitResult(
        speak=SpeakPlan(kind="answer", text=text, task_id=decision.task_id, version=version if version is not None else decision.version),
        route="CONTROL",
        ledger=ledger,
        control=decision,
    )


def _apply_control(decision: ControlDecision, text: str, req: AdmitRequest, ledger: Ledger, tools: list[str] | None = None, explicit_save: bool | None = None) -> AdmitResult:
    rec = ledger.tasks.get(decision.task_id)
    if rec is None:
        # unknown id (e.g. a hallucinated option): nothing to control, treat as a new request
        job = _new_job(ledger, text, language=req.language, tools=tools, explicit_save=explicit_save)
        return AdmitResult(
            speak=SpeakPlan(kind="pad", text=intent.pad_for_tools(tools), task_id=job.task_id, version=job.version),
            job=job,
            route="WORKER",
            ledger=ledger,
        )

    if decision.op == "CANCEL":
        if rec.status in (TaskStatus.COMPLETED, TaskStatus.DELIVERED):
            return _answer("那个已经办完了，没有可取消的。", ledger, decision)
        if rec.version != decision.version:
            return _answer("这个任务刚刚已经变了，我没法按旧版本取消。", ledger, decision)
        if ledger.cancel(decision.task_id, decision.version):
            res = _answer("好的，这个任务取消了。", ledger, decision)
            res.stale_task = (decision.task_id, decision.version)
            return res
        return _answer("这个任务已经结束了，没有可取消的。", ledger, decision)

    if decision.op == "STATUS":
        if rec.status in (TaskStatus.COMPLETED, TaskStatus.DELIVERED) and rec.spoken_answer:
            return _answer(rec.spoken_answer, ledger, decision, version=rec.version)
        if rec.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            return _answer("还在查，请稍等一下。", ledger, decision, version=rec.version)
        if rec.status == TaskStatus.NEEDS_INPUT:
            return _answer(_clarify_text_for_task(rec.objective), ledger, decision, version=rec.version)
        if rec.status == TaskStatus.FAILED and rec.spoken_answer:
            return _answer(rec.spoken_answer, ledger, decision, version=rec.version)
        return _answer("这个任务已经取消了，没有结果。", ledger, decision, version=rec.version)

    # REFINE
    if rec.version != decision.version:
        return _answer("这个任务刚刚已经改过了，我按最新的来。", ledger, decision, version=rec.version)
    objective = _refined_objective(rec.objective, text, slot_fill=rec.status == TaskStatus.NEEDS_INPUT)
    new = ledger.refine(decision.task_id, objective, version=decision.version)
    if new is None:
        job = _new_job(ledger, text, language=req.language, tools=tools, explicit_save=explicit_save)
        return AdmitResult(
            speak=SpeakPlan(kind="answer", text="原任务已经失效，我按新请求来办。", task_id=job.task_id, version=job.version),
            job=job,
            route="WORKER",
            ledger=ledger,
            control=decision,
        )
    if tools:
        new.require_tools, new.tool_judgment = list(tools), "model"
        if explicit_save is not None:
            new.explicit_save = explicit_save
    ledger.mark_running(new.task_id)
    return AdmitResult(
        speak=SpeakPlan(kind="pad", text="好的，按你改的来。", task_id=new.task_id, version=new.version),
        job=WorkerJob(
            task_id=new.task_id,
            version=new.version,
            objective=new.objective,
            language=req.language,
            freshness_token=freshness_token_for(new.task_id, new.version),
            # a REFINE re-runs the job with the new slots: what the router said for the new words, else what it said for the old task
            require_tools=tools or list(rec.require_tools),
            tool_judgment="model" if tools is not None else rec.tool_judgment,
            explicit_save=new.explicit_save,
        ),
        route="CONTROL",
        ledger=ledger,
        control=decision,
        stale_task=(decision.task_id, decision.version),
    )


def _refined_objective(old: str, utterance: str, *, slot_fill: bool) -> str:
    """Compose the Worker objective for a refined task: keep the original ask, make the new words win."""
    if slot_fill:
        return f"{old}。用户补充：{utterance}"
    return f"{old}。用户改口，以新说法为准：{utterance}"


def _new_job(ledger: Ledger, objective: str, *, language: str = "zh", tools: list[str] | None = None, explicit_save: bool | None = None) -> WorkerJob:
    ledger.supersede_pending_input()  # a fresh request replaces any open clarification
    judgment = "unknown" if tools is None else "model"
    rec = ledger.create(objective, require_tools=tools or [], tool_judgment=judgment, explicit_save=explicit_save)
    ledger.mark_running(rec.task_id)
    return WorkerJob(
        task_id=rec.task_id,
        version=rec.version,
        objective=objective,
        language=language,
        freshness_token=freshness_token_for(rec.task_id, rec.version),
        require_tools=list(tools or []),
        tool_judgment=judgment,
        explicit_save=explicit_save,
    )


def complete_job(ledger: Ledger, job: WorkerJob, answer: str) -> bool:
    """Record a Worker result. Returns False (and records nothing) for a stale/cancelled job."""
    rec = ledger.tasks.get(job.task_id)
    if rec is None:
        return False
    if rec.version != job.version or rec.status in (TaskStatus.SUPERSEDED, TaskStatus.CANCELLED):
        log.info("drop stale worker result %s v%s (task now v%s %s)", job.task_id, job.version, rec.version, rec.status.value)
        return False
    return ledger.mark_completed(job.task_id, answer)


@dataclass
class JobOutcome:
    """What the Coordinator wants said for a finished WorkerResult."""

    kind: str  # "answer" | "clarify" | "stale"
    text: str = ""
    task_id: str = ""
    version: int = 0
    rejected: str = ""  # why a Worker answer was not spoken ("no_tool_result"), for the trace


def _ungrounded(job: WorkerJob, result: WorkerResult) -> bool:
    """A completed answer for a tool-scoped job with no evidence of a *successful* call of one of its tools.

    A call that failed (``ok`` false) is not evidence either: after a failed tool the model would only be guessing."""
    if not job.require_tools or result.status != "completed":
        return False
    return not any(ev.get("tool") in job.require_tools and tool_succeeded(ev) for ev in result.evidence)


def _failure_text(job: WorkerJob, result: WorkerResult) -> str:
    """What may be said for a FAILED result: a fixed honest sentence, or the failed required tool's own sentence
    (recorded by the Worker as evidence ``failure_text``). Anything else is model free text: replaced (Critic I-5)."""
    said = (result.spoken_answer or "").strip()
    if said in FIXED_FAILURES:
        return said
    if said and any(ev.get("tool") in (job.require_tools or []) and ev.get("ok") is False and ev.get("failure_text") == said
                    and is_tool_failure_sentence(said, ev, job.objective)       # S-13: a Worker's say-so is not enough, it must be OUR tool's sentence
                    for ev in result.evidence):
        return said
    return NO_TOOL_ANSWER if job.require_tools else FAILED_GENERIC


def finish_job(ledger: Ledger, job: WorkerJob, result: WorkerResult) -> JobOutcome:
    """Worker → Coordinator. Stale or cancelled results are dropped; needs_input becomes a C-worded CLARIFY.

    The Worker only reports status and missing slots; wording and ledger state are decided here.
    """
    rec = ledger.tasks.get(job.task_id)
    stale = JobOutcome(kind="stale", task_id=job.task_id, version=job.version)
    if rec is None or (result.task_id, result.version) != (job.task_id, job.version):
        return stale
    if job.freshness_token and (job.freshness_token != freshness_token_for(job.task_id, job.version) or result.freshness_token != job.freshness_token):
        # the token was written by the Coordinator for exactly this (task_id, version); a result that does not echo it
        # is not an answer to this job (S-3: the token used to be written and never checked)
        log.warning("drop worker result %s v%s: freshness token mismatch (job=%r result=%r)", job.task_id, job.version, job.freshness_token, result.freshness_token)
        return stale
    if rec.version != job.version or rec.status in (TaskStatus.SUPERSEDED, TaskStatus.CANCELLED):
        log.info("drop stale worker result %s v%s (task now v%s %s)", job.task_id, job.version, rec.version, rec.status.value)
        return stale
    if result.status == "cancelled":
        return stale
    if result.status == "needs_input":
        ledger.mark_needs_input(job.task_id)
        return JobOutcome(kind="clarify", text=clarify_for_slots(result.needs_slots), task_id=job.task_id, version=job.version)
    if _ungrounded(job, result):
        # Never speak a tool-backed answer the model produced without calling its tool (fabricated numbers).
        log.warning("reject ungrounded worker answer task=%s v%s require=%s answer=%r", job.task_id, job.version, job.require_tools, (result.spoken_answer or "")[:60])
        if not ledger.mark_failed(job.task_id, NO_TOOL_ANSWER):
            return stale
        tried = any(ev.get("tool") in job.require_tools for ev in result.evidence)
        return JobOutcome(kind="answer", text=NO_TOOL_ANSWER, task_id=job.task_id, version=job.version,
                          rejected="tool_failed" if tried else "no_tool_result")
    if result.status == "failed":
        text = _failure_text(job, result)
        replaced = text != (result.spoken_answer or "").strip()
        if replaced:
            log.warning("replace free-text failure task=%s v%s answer=%r", job.task_id, job.version, (result.spoken_answer or "")[:60])
        if not ledger.mark_failed(job.task_id, text):
            return stale
        return JobOutcome(kind="answer", text=text, task_id=job.task_id, version=job.version, rejected="failed_text" if replaced else "")
    text = (result.spoken_answer or "").strip() or "好的，我这边查过了。"
    if not ledger.mark_completed(job.task_id, text):
        return stale
    return JobOutcome(kind="answer", text=text, task_id=job.task_id, version=job.version)


def on_playback_receipt(ledger: Ledger, receipt: PlaybackReceipt) -> bool:
    """Delivery → Coordinator. 听完 = DELIVERED; 打断 / 放弃 never cancel anything and keep the answer re-playable.

    Returns True when the receipt matched a live (task_id, version).
    """
    if receipt.outcome == "delivered":
        return ledger.mark_delivered(receipt.task_id, receipt.version)
    return ledger.note_interrupted(receipt.task_id, receipt.version)
