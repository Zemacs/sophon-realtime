"""R0 CPU checks. Invariants must hold. Target cases record actuals and assert the contract.

A failing target whose actual matches the known gap is a known red, not an invariant.
Improvement makes the target assertion pass; that is not Live acceptance.
Gateway tests call the real LiveSession._coord_resolve. Stub admit is not the Live path.
The heuristic admit test is a narrow integration into that method with a fake orch.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.coordinator import AdmitRequest, AdmitResult, Ledger, SpeakPlan, WorkerJob, admit
from sophon_live.coordinator import api
from sophon_live.coordinator import jev_client
from sophon_live.coordinator.api import complete_job
from sophon_live.coordinator.schemas import TaskStatus
from tests.worker_stub import as_worker


def record(case_id: str, actual: dict) -> None:
    path = os.environ.get("R0_OBS_PATH")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": case_id, "actual": actual}, ensure_ascii=False) + "\n")


def _req(text: str, *, uid: str) -> AdmitRequest:
    return AdmitRequest(
        session_id="r0",
        utterance_id=uid,
        text=text,
        delegate_task=text,
        playback_active=False,
    )


class InvariantTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_result_dropped_and_cancel_is_versioned(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            first = await admit(_req("现在几点了", uid="t1"), led)
        self.assertIsNotNone(first.job)
        assert first.job is not None
        new = led.refine(first.job.task_id, "改成日期")
        self.assertIsNotNone(new)
        assert new is not None
        self.assertEqual(new.version, 2)
        complete_job(led, first.job, "迟到的旧答案")
        current = led.tasks[first.job.task_id]
        self.assertEqual(current.version, 2)
        self.assertEqual(current.spoken_answer, "")
        self.assertFalse(led.cancel(first.job.task_id, 1))
        self.assertTrue(led.cancel(first.job.task_id, 2))
        self.assertEqual(current.status, TaskStatus.CANCELLED)

    async def test_jev_stub_called_and_socket_attempts_stay_zero(self) -> None:
        calls = {"n": 0}

        class BoomSession:
            def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                calls["n"] += 1
                raise RuntimeError("blocked before request")

        log_path = os.environ.get("R0_NET_LOG")
        before = Path(log_path).read_text(encoding="utf-8") if log_path and Path(log_path).exists() else ""
        with (
            mock.patch.object(api, "USE_JEV", True),
            mock.patch.object(jev_client.aiohttp, "ClientSession", BoomSession),
        ):
            led = Ledger()
            result = await admit(_req("现在几点了", uid="jev"), led)
        after = Path(log_path).read_text(encoding="utf-8") if log_path and Path(log_path).exists() else ""
        self.assertEqual(calls["n"], 1)
        self.assertEqual(after, before)
        self.assertEqual(result.route, "WORKER")
        self.assertIsNotNone(result.job)

    def test_two_ledger_objects_do_not_share_tasks(self) -> None:
        """Object isolation only. This is not a session-id association guarantee."""
        left = Ledger()
        right = Ledger()
        left.create("左任务")
        self.assertEqual(len(left.tasks), 1)
        self.assertEqual(len(right.tasks), 0)
        self.assertIsNot(left, right)


class ControlStatusTargetTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_keeps_one_task_and_does_not_start_worker(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            started = await admit(_req("北京明天天气", uid="w"), led)
            self.assertEqual(started.route, "WORKER")
            self.assertEqual(len(led.tasks), 1)
            cancelled = await admit(_req("取消刚才那个", uid="c"), led)
        actual = {
            "task_count": len(led.tasks),
            "cancelled": sum(task.status == TaskStatus.CANCELLED for task in led.tasks.values()),
            "statuses": [task.status.value for task in led.tasks.values()],
            "route": cancelled.route,
            "has_job": cancelled.job is not None,
        }
        record("target.cancel", actual)
        self.assertEqual(actual["task_count"], 1)
        self.assertEqual(actual["cancelled"], 1)
        self.assertEqual(actual["has_job"], False)
        self.assertEqual(actual["statuses"], ["CANCELLED"])

    async def test_status_answers_from_existing_result_without_new_task(self) -> None:
        with mock.patch.object(api, "USE_JEV", False):
            led = Ledger()
            started = await admit(_req("北京明天天气", uid="w"), led)
            assert started.job is not None
            complete_job(led, started.job, "北京明天晴，12到18度。")
            stored = led.tasks[started.job.task_id].spoken_answer
            status = await admit(_req("刚才查得怎么样", uid="s"), led)
        actual = {
            "task_count": len(led.tasks),
            "has_job": status.job is not None,
            "route": status.route,
            "kind": status.speak.kind,
            "text": status.speak.text,
        }
        record("target.status", actual)
        self.assertEqual(actual["task_count"], 1)
        self.assertEqual(actual["has_job"], False)
        self.assertEqual(actual["kind"], "answer")
        self.assertEqual(actual["text"], stored)


def _bare_session():
    import sophon_live.gateway as gw

    session = gw.LiveSession.__new__(gw.LiveSession)
    session.id = "sess_r0"
    session.instructions = ""
    session.timeline_ms = 0
    session.playing = False
    session.lang = "zh"
    session.deleg = {}
    session.engine = None
    session.last_voice_t = None
    session.last_said = ""
    sent: list = []
    flushed: list = []

    async def send(ev: dict) -> None:
        sent.append(ev)

    async def flush(did: str) -> None:
        flushed.append(did)
        if session.engine is not None:
            raise AssertionError("engine touched")

    session.send = send
    session._flush_deleg = flush
    session._drop_playback = lambda why: flushed.append(("drop", why))
    session.sent = sent
    session.flushed = flushed
    return session


def _arm(session, did: str, text: str) -> None:
    session.deleg[did] = {
        "call_id": "call_" + did,
        "chunks": [],
        "timer": None,
        "task": text,
        "t0": 0,
    }


class GatewayBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Real LiveSession._coord_resolve only. Not a Live audio or routing chain."""

    async def test_stub_admit_one_job_is_not_duplicated(self) -> None:
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord
        import sophon_live.coordinator.api as capi

        session = _bare_session()
        new_jobs = {"n": 0}
        orch_calls: list[str] = []
        real_new = capi._new_job

        def counting_new(*args, **kwargs):
            new_jobs["n"] += 1
            return real_new(*args, **kwargs)

        async def fake_admit(req, ledger):
            rec = ledger.create(req.delegate_task or req.text)
            ledger.mark_running(rec.task_id)
            job = WorkerJob(
                task_id=rec.task_id,
                version=rec.version,
                objective=rec.objective,
                freshness_token=f"{rec.task_id}:{rec.version}",
            )
            return AdmitResult(
                speak=SpeakPlan(kind="pad", text="我帮你查一下。", task_id=rec.task_id, version=rec.version),
                job=job,
                route="WORKER",
                ledger=ledger,
            )

        async def fake_orch(task, instructions="", history=None):
            orch_calls.append(task)
            return "现在是三点。"

        _arm(session, "d1", "现在几点了")
        with (
            mock.patch.object(coord, "admit", fake_admit),
            mock.patch.object(capi, "_new_job", counting_new),
            mock.patch.object(tools, "run_worker_job", as_worker(fake_orch)),
        ):
            await session._coord_resolve("d1", "现在几点了")
        self.assertEqual(new_jobs["n"], 0)
        self.assertEqual(orch_calls, ["现在几点了"])
        self.assertEqual(len(session.coord_ledger.tasks), 1)
        self.assertIsNone(session.engine)

    async def test_real_heuristic_admit_into_coord_resolve_with_fake_orch(self) -> None:
        """Narrow integration: real admit heuristic, real _coord_resolve, fake orch. Not Live."""
        import sophon_live.assistant_tools as tools

        async def fake_orch(task, instructions="", history=None):
            return "桩回答：" + task

        left = _bare_session()
        right = _bare_session()
        _arm(left, "a", "现在几点了")
        _arm(right, "b", "科技新闻")
        with (
            mock.patch.object(api, "USE_JEV", False),
            mock.patch.object(tools, "run_worker_job", as_worker(fake_orch)),
        ):
            await left._coord_resolve("a", "现在几点了")
            await right._coord_resolve("b", "科技新闻")
        self.assertIsNot(left.coord_ledger, right.coord_ledger)
        self.assertEqual(len(left.coord_ledger.tasks), 1)
        self.assertEqual(len(right.coord_ledger.tasks), 1)
        self.assertFalse(set(left.coord_ledger.tasks) & set(right.coord_ledger.tasks))

    async def test_fast_without_job_does_not_call_new_job(self) -> None:
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord
        import sophon_live.coordinator.api as capi

        session = _bare_session()
        new_jobs = {"n": 0}
        orch_calls: list[str] = []
        real_new = capi._new_job

        def counting_new(*args, **kwargs):
            new_jobs["n"] += 1
            return real_new(*args, **kwargs)

        async def fake_admit(req, ledger):
            return AdmitResult(speak=SpeakPlan(kind="noop", text=""), job=None, route="FAST", ledger=ledger)

        async def fake_orch(task, instructions="", history=None):
            orch_calls.append(task)
            return "桩"

        _arm(session, "f", "谢谢")
        with (
            mock.patch.object(coord, "admit", fake_admit),
            mock.patch.object(capi, "_new_job", counting_new),
            mock.patch.object(tools, "run_worker_job", as_worker(fake_orch)),
        ):
            await session._coord_resolve("f", "谢谢")
        actual = {
            "new_job_calls": new_jobs["n"],
            "orch_calls": len(orch_calls),
            "ledger_len": len(session.coord_ledger.order),
        }
        record("target.fast_no_job", actual)
        self.assertEqual(actual["new_job_calls"], 0)
        self.assertEqual(actual["orch_calls"], 0)
        self.assertEqual(actual["ledger_len"], 0)


if __name__ == "__main__":
    unittest.main()
