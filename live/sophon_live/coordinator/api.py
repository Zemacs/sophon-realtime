"""Coordinator admit(): Frontend → route/ledger → SpeakPlan and/or WorkerJob."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import jev_client
from .ledger import Ledger
from .route import build_route_questions, build_state, heuristic_control, heuristic_route
from .schemas import AdmitRequest, ControlDecision, PlaybackReceipt, SpeakPlan, TaskStatus, WorkerJob

log = logging.getLogger("sophon_live.coordinator")

USE_JEV = os.environ.get("SOPHON_LIVE_JEV", "1").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AdmitResult:
    speak: SpeakPlan
    job: WorkerJob | None = None
    route: str = "FAST"
    ledger: Ledger | None = None
    control: ControlDecision | None = None  # set when a CONTROL op was applied (or refused)
    stale_task: tuple[str, int] | None = None  # (task_id, version) whose in-flight worker must stop


def _clarify_text(text: str) -> str:
    if "天气" in text:
        return "请问您想查哪个城市的天气？"
    return "还缺一点信息，方便再说具体一点吗？"


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


async def admit(req: AdmitRequest, ledger: Ledger) -> AdmitResult:
    text = (req.delegate_task or req.text or "").strip()
    route = "WORKER"
    control_key: str | None = None

    if USE_JEV:
        try:
            state = build_state(text, ledger, playback_active=req.playback_active)
            questions = build_route_questions(ledger)
            resp = await jev_client.system_one(state, questions)
            route = jev_client.choice_answer(resp, "route") or heuristic_route(text, ledger)
            control_key = jev_client.choice_answer(resp, "control")
            log.info("jev route=%s control=%s utterance=%r", route, control_key, text[:80])
        except Exception as e:  # noqa: BLE001
            log.warning("jev unavailable, heuristic: %r", e)
            route = heuristic_route(text, ledger)
    else:
        route = heuristic_route(text, ledger)

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
                text=_clarify_text(text),
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

    if route == "CONTROL":
        decision = _parse_control_key(control_key)
        # JEV said CONTROL but named no usable target (or NEW): the Ledger decides what exists.
        if decision is None and control_key != "NEW":
            decision = heuristic_control(text, ledger)
        if decision is not None:
            return _apply_control(decision, text, req, ledger)

    # WORKER or CONTROL→NEW or CONTROL with nothing to control
    job = _new_job(ledger, text, language=req.language)
    return AdmitResult(
        speak=SpeakPlan(
            kind="pad",
            text="我帮你查一下。",
            task_id=job.task_id,
            version=job.version,
        ),
        job=job,
        route="WORKER",
        ledger=ledger,
    )


def _answer(text: str, ledger: Ledger, decision: ControlDecision, *, version: int | None = None) -> AdmitResult:
    return AdmitResult(
        speak=SpeakPlan(kind="answer", text=text, task_id=decision.task_id, version=version if version is not None else decision.version),
        route="CONTROL",
        ledger=ledger,
        control=decision,
    )


def _apply_control(decision: ControlDecision, text: str, req: AdmitRequest, ledger: Ledger) -> AdmitResult:
    rec = ledger.tasks.get(decision.task_id)
    if rec is None:
        # unknown id (e.g. a hallucinated option): nothing to control, treat as a new request
        job = _new_job(ledger, text, language=req.language)
        return AdmitResult(
            speak=SpeakPlan(kind="pad", text="我帮你查一下。", task_id=job.task_id, version=job.version),
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
            return _answer(_clarify_text(rec.objective), ledger, decision, version=rec.version)
        return _answer("这个任务已经取消了，没有结果。", ledger, decision, version=rec.version)

    # REFINE
    if rec.version != decision.version:
        return _answer("这个任务刚刚已经改过了，我按最新的来。", ledger, decision, version=rec.version)
    objective = _refined_objective(rec.objective, text, slot_fill=rec.status == TaskStatus.NEEDS_INPUT)
    new = ledger.refine(decision.task_id, objective, version=decision.version)
    if new is None:
        job = _new_job(ledger, text, language=req.language)
        return AdmitResult(
            speak=SpeakPlan(kind="answer", text="原任务已经失效，我按新请求来办。", task_id=job.task_id, version=job.version),
            job=job,
            route="WORKER",
            ledger=ledger,
            control=decision,
        )
    ledger.mark_running(new.task_id)
    return AdmitResult(
        speak=SpeakPlan(kind="pad", text="好的，按你改的来。", task_id=new.task_id, version=new.version),
        job=WorkerJob(
            task_id=new.task_id,
            version=new.version,
            objective=new.objective,
            language=req.language,
            freshness_token=f"{new.task_id}:{new.version}",
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


def _new_job(ledger: Ledger, objective: str, *, language: str = "zh") -> WorkerJob:
    ledger.supersede_pending_input()  # a fresh request replaces any open clarification
    rec = ledger.create(objective)
    ledger.mark_running(rec.task_id)
    return WorkerJob(
        task_id=rec.task_id,
        version=rec.version,
        objective=objective,
        language=language,
        freshness_token=f"{rec.task_id}:{rec.version}",
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


def on_playback_receipt(ledger: Ledger, receipt: PlaybackReceipt) -> bool:
    """Delivery → Coordinator. 听完 = DELIVERED; 打断 / 放弃 never cancel anything and keep the answer re-playable.

    Returns True when the receipt matched a live (task_id, version).
    """
    if receipt.outcome == "delivered":
        return ledger.mark_delivered(receipt.task_id, receipt.version)
    return ledger.note_interrupted(receipt.task_id, receipt.version)
