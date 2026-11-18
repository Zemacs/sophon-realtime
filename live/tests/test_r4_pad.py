"""Round 4 / Critic I-2: the Coordinator's SpeakPlan(kind="pad") reaches the user when a WorkerJob runs (gateway side).

Real LiveSession._coord_resolve with a fake engine and a fake Worker. Not an audio chain: whether the voice front
says it (or <wait>s) is its decision and needs the live :8780 run.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

import sophon_live.gateway as gw
from sophon_live.coordinator import AdmitResult, SpeakPlan, WorkerJob, WorkerResult
from tests.test_r0_coord_boundary import _arm, _bare_session


class _Engine:
    def __init__(self) -> None:
        self.results: list[tuple[str | None, str, bool]] = []

    async def tool_result(self, call_id, output, speak=True):  # noqa: ANN001
        self.results.append((call_id, output, speak))
        return True


def _session():
    s = _bare_session()
    s.engine = _Engine()
    s.lang = "zh"
    s.last_said = ""

    async def flush(did: str) -> None:       # the real flush -> _flush_model -> _post_update; keep the real one
        await gw.LiveSession._flush_deleg(s, did)

    s._flush_deleg = flush
    s._flush_model = lambda did: gw.LiveSession._flush_model(s, did)
    s._post_update = lambda did, chunks, pad=False: gw.LiveSession._post_update(s, did, chunks, pad=pad)
    s._delivery_armed_calls = []
    s._delivery_armed = lambda did: (s._delivery_armed_calls.append(did), gw.LiveSession._delivery_armed(s, did))
    return s


def _admit(pad_text: str, kind: str = "pad"):
    async def fake_admit(req, ledger):
        rec = ledger.create(req.delegate_task or req.text)
        ledger.mark_running(rec.task_id)
        job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, freshness_token=f"{rec.task_id}:{rec.version}")
        return AdmitResult(speak=SpeakPlan(kind=kind, text=pad_text, task_id=rec.task_id, version=rec.version), job=job, route="WORKER", ledger=ledger)

    return fake_admit


def _worker(delay: float, answer: str = "北京明天多云，1到8度。"):
    async def run(job, instructions="", history=None):  # noqa: ANN001
        await asyncio.sleep(delay)
        return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=answer,
                            evidence=[{"tool": "get_weather", "ok": True}], freshness_token=job.freshness_token)

    return run


class PadTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, session, admit, worker, did="p1", text="北京明天天气"):
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord

        _arm(session, did, text)
        session._trace().start(did)
        with mock.patch.object(coord, "admit", admit), mock.patch.object(tools, "run_worker_job", worker), \
                mock.patch.object(gw, "PAD_GRACE_S", 0.05), mock.patch.object(gw, "DELEG_MODE", "model"):
            await session._coord_resolve(did, text)

    async def test_slow_worker_gets_the_pad_before_the_result(self) -> None:
        s = _session()
        await self._run(s, _admit("我帮你查一下。"), _worker(0.4))
        texts = [o for _, o, _ in s.engine.results]
        self.assertEqual(len(texts), 2)
        self.assertIn("我帮你查一下。", texts[0])                 # the filler first ...
        self.assertNotIn("1到8度", texts[0])
        self.assertIn("北京明天多云，1到8度。", texts[1])         # ... then the answer
        self.assertTrue(all(sp for _, _, sp in s.engine.results))
        self.assertEqual(s._delivery_armed_calls, ["p1"])        # only the ANSWER arms a delivery receipt, the pad does not
        tr = s._trace().get("p1").as_dict()
        self.assertEqual((tr["pad"], tr["pad_text"]), ("posted", "我帮你查一下。"))
        self.assertEqual(len(s._trace()._said), 1)               # one hand-off queued for the next response: the ANSWER's, not the pad's

    async def test_fast_worker_gets_no_pad(self) -> None:
        s = _session()
        await self._run(s, _admit("我帮你查一下。"), _worker(0.0))
        texts = [o for _, o, _ in s.engine.results]
        self.assertEqual(len(texts), 1)
        self.assertNotIn("我帮你查一下。", texts[0])
        self.assertNotIn("pad", s._trace().get("p1").as_dict())

    async def test_empty_pad_is_not_posted(self) -> None:
        s = _session()
        await self._run(s, _admit(""), _worker(0.3), text="现在几点了")
        self.assertEqual(len(s.engine.results), 1)

    async def test_non_pad_speak_kinds_are_not_posted(self) -> None:
        s = _session()
        await self._run(s, _admit("原任务已经失效，我按新请求来办。", kind="answer"), _worker(0.3))
        self.assertEqual(len(s.engine.results), 1)

    async def test_front_that_just_spoke_is_not_interrupted(self) -> None:
        import time

        s = _session()
        s.last_voice_t = time.monotonic()                        # the front said its own lead-in a moment ago
        await self._run(s, _admit("我帮你查一下。"), _worker(0.3))
        self.assertEqual(len(s.engine.results), 1)
        self.assertEqual(s._trace().get("p1").as_dict()["pad"], "front_spoke")

    async def test_pad_never_follows_a_cancelled_worker(self) -> None:
        s = _session()
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord

        _arm(s, "p1", "北京明天天气")
        s._trace().start("p1")

        async def slow(job, instructions="", history=None):  # noqa: ANN001
            await asyncio.sleep(30)

        async def run():
            with mock.patch.object(coord, "admit", _admit("我帮你查一下。")), mock.patch.object(tools, "run_worker_job", slow), \
                    mock.patch.object(gw, "PAD_GRACE_S", 5.0), mock.patch.object(gw, "DELEG_MODE", "model"):
                await s._coord_resolve("p1", "北京明天天气")

        t = asyncio.create_task(run())
        await asyncio.sleep(0.1)
        for w in list(s.coord_workers.values()):
            w.cancel()
        await asyncio.wait_for(t, 5)
        await asyncio.sleep(0.1)
        self.assertEqual([(c, sp) for c, _, sp in s.engine.results], [("call_p1", False)])    # silent ack only, never a pad
        self.assertIsNone(s.deleg["p1"].get("pad_task"))


class RuntimeBannerTests(unittest.TestCase):
    """Critic S-6: SOPHON_LIVE_COORD is off by default in code; a restart without it must say so loudly."""

    def test_default_env_is_flagged_as_the_legacy_path(self) -> None:
        text, intended = gw.runtime_banner({})
        self.assertFalse(intended)
        self.assertIn("SOPHON_LIVE_COORD OFF", text)
        self.assertIn("SOPHON_LIVE_COORD=1", text)

    def test_coord_on_is_reported_with_the_other_switches(self) -> None:
        text, intended = gw.runtime_banner({"SOPHON_LIVE_COORD": "1", "SOPHON_LIVE_JEV": "1", "SOPHON_LIVE_ORCH": "1"})
        self.assertTrue(intended)
        self.assertIn("COORD on", text)
        self.assertIn("JEV=1", text)
        self.assertIn("ORCH=1", text)

    def test_falsy_values_do_not_count_as_on(self) -> None:
        for v in ("0", "", "false", "no"):
            self.assertFalse(gw.runtime_banner({"SOPHON_LIVE_COORD": v})[1], v)


if __name__ == "__main__":
    unittest.main()
