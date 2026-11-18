"""Coordinator admit(): Frontend → route/ledger → SpeakPlan and/or WorkerJob."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import jev_client
from .ledger import Ledger
from .route import build_route_questions, build_state, heuristic_route
from .schemas import AdmitRequest, SpeakPlan, TaskStatus, WorkerJob

log = logging.getLogger("sophon_live.coordinator")

USE_JEV = os.environ.get("SOPHON_LIVE_JEV", "1").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AdmitResult:
    speak: SpeakPlan
    job: WorkerJob | None = None
    route: str = "FAST"
    ledger: Ledger | None = None


def _clarify_text(text: str) -> str:
    if "天气" in text:
        return "请问您想查哪个城市的天气？"
    return "还缺一点信息，方便再说具体一点吗？"


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

    if route == "CONTROL" and control_key and control_key != "NEW":
        parts = control_key.split(":")
        if len(parts) == 3:
            op, tid, ver_s = parts
            try:
                ver = int(ver_s)
            except ValueError:
                ver = -1
            if op == "CANCEL":
                ok = ledger.cancel(tid, ver)
                msg = "好的，这个任务取消了。" if ok else "这个任务已经变了，我没法按旧版本取消。"
                return AdmitResult(
                    speak=SpeakPlan(kind="answer", text=msg, task_id=tid, version=ver),
                    route=route,
                    ledger=ledger,
                )
            if op == "STATUS":
                rec = ledger.tasks.get(tid)
                if rec and rec.spoken_answer:
                    return AdmitResult(
                        speak=SpeakPlan(
                            kind="answer",
                            text=rec.spoken_answer,
                            task_id=tid,
                            version=rec.version,
                        ),
                        route=route,
                        ledger=ledger,
                    )
                return AdmitResult(
                    speak=SpeakPlan(kind="answer", text="这个任务还没有结果。"),
                    route=route,
                    ledger=ledger,
                )
            if op == "REFINE":
                new = ledger.refine(tid, text)
                if new is None:
                    return AdmitResult(
                        speak=SpeakPlan(kind="answer", text="原任务已经失效，我按新请求来办。"),
                        job=_new_job(ledger, text),
                        route="WORKER",
                        ledger=ledger,
                    )
                ledger.mark_running(new.task_id)
                return AdmitResult(
                    speak=SpeakPlan(
                        kind="pad",
                        text="好的，按你改的来。",
                        task_id=new.task_id,
                        version=new.version,
                    ),
                    job=WorkerJob(
                        task_id=new.task_id,
                        version=new.version,
                        objective=new.objective,
                        language=req.language,
                        freshness_token=f"{new.task_id}:{new.version}",
                    ),
                    route=route,
                    ledger=ledger,
                )

    # WORKER or CONTROL→NEW or fallback
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


def _new_job(ledger: Ledger, objective: str, *, language: str = "zh") -> WorkerJob:
    rec = ledger.create(objective)
    ledger.mark_running(rec.task_id)
    return WorkerJob(
        task_id=rec.task_id,
        version=rec.version,
        objective=objective,
        language=language,
        freshness_token=f"{rec.task_id}:{rec.version}",
    )


def complete_job(ledger: Ledger, job: WorkerJob, answer: str) -> None:
    rec = ledger.tasks.get(job.task_id)
    if rec is None:
        return
    if rec.version != job.version or rec.status == TaskStatus.SUPERSEDED:
        log.info("drop stale worker result %s v%s", job.task_id, job.version)
        return
    ledger.mark_completed(job.task_id, answer)
