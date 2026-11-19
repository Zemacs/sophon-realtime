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
                            evidence=[{"tool": "get_weather", "ok": True, "out": answer}], freshness_token=job.freshness_token)

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


class _SlowPadEngine(_Engine):
    """tool_result takes `delay` s when the update carries the pad text: models the HTTP round trip (the await in which a Worker can finish)."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    async def tool_result(self, call_id, output, speak=True):  # noqa: ANN001
        if "要转达给用户的内容：我帮你查一下。" in output:
            await asyncio.sleep(self.delay)
        return await super().tool_result(call_id, output, speak)


def _kinds(session) -> list[str]:
    return ["pad" if "我帮你查一下。" in o and "1到8度" not in o else "answer" if "1到8度" in o else "other" for _, o, _ in session.engine.results]


class PadRaceTests(unittest.IsolatedAsyncioTestCase):
    """Critic I-6: the Worker finishing while the pad is on its way (past the point where cancel still works)."""

    async def _go(self, session, worker_delay: float, did="r1"):
        import sophon_live.assistant_tools as tools
        import sophon_live.coordinator as coord

        _arm(session, did, "北京明天天气")
        session._trace().start(did)
        with mock.patch.object(coord, "admit", _admit("我帮你查一下。")), mock.patch.object(tools, "run_worker_job", _worker(worker_delay)), \
                mock.patch.object(gw, "PAD_GRACE_S", 0.05), mock.patch.object(gw, "DELEG_MODE", "model"):
            await session._coord_resolve(did, "北京明天天气")
        for _ in range(40):                                        # let a pad still in flight settle, then judge the order
            await asyncio.sleep(0.02)
            if not session.deleg[did].get("pad_posting"):
                break
        await asyncio.sleep(0.05)

    async def test_worker_finishing_during_the_pad_post_never_puts_the_pad_behind_the_result(self) -> None:
        s = _session()
        s.engine = _SlowPadEngine(0.30)                            # pad starts at 0.05 s and is still in flight when the Worker ends at 0.15 s
        await self._go(s, 0.15)
        self.assertEqual(_kinds(s), ["pad", "answer"])             # in order: the result waited for the pad's turn, it did not overtake it
        self.assertEqual(s._trace().get("r1").as_dict()["pad"], "posted")

    async def test_critics_probe_shape_delay_before_the_post_the_pad_is_dropped(self) -> None:
        """runs/incr-1/pad_race_probe2.py: the pad's _post_update is held back 0.25 s, the Worker ends at 0.12 s."""
        s = _session()
        real = s._post_update

        async def held_back(did, chunks, pad=False):  # noqa: ANN001
            if pad:
                await asyncio.sleep(0.25)
            return await real(did, chunks, pad=pad)

        s._post_update = held_back
        await self._go(s, 0.12)
        kinds = _kinds(s)
        self.assertEqual(kinds, ["answer"])                        # the late pad is dropped, it never lands after the result
        self.assertEqual(s._trace().get("r1").as_dict()["pad"], "skipped_result_first")

    async def test_pad_never_appears_after_the_result_whatever_the_timing(self) -> None:
        for worker_delay in (0.0, 0.04, 0.06, 0.10, 0.20, 0.34, 0.50):
            for pad_http in (0.0, 0.1, 0.3):
                s = _session()
                s.engine = _SlowPadEngine(pad_http)
                await self._go(s, worker_delay)
                kinds = _kinds(s)
                self.assertEqual(kinds.count("answer"), 1, (worker_delay, pad_http, kinds))
                if "pad" in kinds:
                    self.assertLess(kinds.index("pad"), kinds.index("answer"), (worker_delay, pad_http, kinds))

    async def test_pad_post_after_cancel_or_result_is_refused_under_the_lock(self) -> None:
        for how in ("said_result", "cancelled"):
            s = _session()
            _arm(s, "c1", "北京明天天气")
            d = s.deleg["c1"]
            if how == "said_result":
                d["said_result"] = True
            else:
                gw.LiveSession._cancel_pad(d)
            sent = await gw.LiveSession._post_update(s, "c1", [("commentary", "我帮你查一下。")], pad=True)
            self.assertFalse(sent, how)
            self.assertEqual(s.engine.results, [], how)

    async def test_silent_ack_waits_for_a_pad_already_on_its_way(self) -> None:
        s = _session()
        s.engine = _SlowPadEngine(0.2)
        _arm(s, "a1", "北京明天天气")
        pad = asyncio.ensure_future(gw.LiveSession._post_update(s, "a1", [("commentary", "我帮你查一下。")], pad=True))
        await asyncio.sleep(0.05)
        await gw.LiveSession._ack_delegation(s, "a1", "已取消")
        await pad
        self.assertEqual([sp for _, _, sp in s.engine.results], [True, False])          # pad first, silent ack after


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
