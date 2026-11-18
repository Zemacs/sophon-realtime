"""I-10 blind runner: behaviour, who-decided columns, fidelity, privacy (sentence text goes to the output file and nowhere else), no judgment in the runner."""
from __future__ import annotations

import ast
import asyncio
import io
import json
import logging
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import blind_runner as br

LIVE = pathlib.Path(__file__).resolve().parents[1]
# distinctive, never in any source file or fixture; the Chinese ones go through the real Coordinator logging paths
SENTENCES = ["ZQXV7731今天北京天气怎么样", "KJWB5528好的谢谢", "记住我喜欢PLMN4406", "NHTD9902帮我查一下上海的新闻"]
MARKERS = ["ZQXV7731", "KJWB5528", "PLMN4406", "NHTD9902"]


def _write(d: str, lines: list[str], name: str = "in.txt") -> str:
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def _rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _jev_answering(route: str | None, control: str | None = None, *, raw: dict | None = None):
    async def system_one(state, questions, **kw):   # noqa: ANN001
        if raw is not None:
            return raw
        ans: dict = {}
        if route is not None:
            ans["route"] = {"type": "choice", "choice": route}
        if control is not None:
            ans["control"] = {"type": "choice", "choice": control}
        return {"answers": ans}
    return system_one


def _jev_raising(exc: BaseException):
    async def system_one(state, questions, **kw):   # noqa: ANN001
        raise exc
    return system_one


class _Base(unittest.TestCase):
    """Every test must leave os.environ and the logging switch exactly as it found them (a leak in the FIRST run is a failure too)."""

    def _callTestMethod(self, method) -> None:   # noqa: ANN001   (inside the test call, so a leak is a FAILURE of that test, not a teardown error)
        env, disable = dict(os.environ), logging.root.manager.disable
        try:
            super()._callTestMethod(method)
        finally:
            leaked = (dict(os.environ), logging.root.manager.disable)
            os.environ.clear(); os.environ.update(env); logging.disable(disable)          # do not let one leak fail every later test
        self.assertEqual(leaked, (env, disable), "the runner leaked environment or logging state")

    def run_one(self, text: str, *, jev, ledger=None, worker=None, use_jev: bool = True, **kw) -> dict:
        from sophon_live import coordinator as C
        from sophon_live.coordinator import api
        prev = api.USE_JEV
        api.USE_JEV = use_jev
        try:
            with br.quiet_and_sealed(False):
                task = kw.pop("task", None)
                return asyncio.run(br.run_one(kw.pop("sid", "x"), br.Opaque(text), ledger=ledger or C.Ledger(), jev=jev, worker=worker or br._fake_worker,
                                              task=br.Opaque(task) if task is not None else None, **kw))
        finally:
            api.USE_JEV = prev


class Behaviour(_Base):
    def test_plain_lines_get_sentence_numbers_and_fake_mode_records_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            lines = [SENTENCES[0], "", SENTENCES[1], "", "", SENTENCES[2], SENTENCES[3], ""]
            self.assertEqual(br.main([_write(d, lines), out]), 0)
            rows = _rows(out)
        self.assertEqual([r["id"] for r in rows], ["s0001", "s0002", "s0003", "s0004"])          # numbers the sentences, not the file lines
        keys = ("id", "mode", "fidelity", "route", "route_via", "decision", "jev_route", "jev_control", "via", "control_via", "degraded",
                "degraded_reason", "degraded_exc", "handled_by", "required_tools", "clarify_slots", "memory_gate_explicit", "tools", "tool_ok",
                "worker_status", "outcome_kind", "outcome_rejected", "reply", "error", "error_stage")
        for r in rows:
            self.assertEqual(sorted(r), sorted(keys))
            self.assertEqual((r["mode"], r["route"], r["route_via"], r["via"], r["degraded"], r["jev_route"], r["error"]),
                             ("fake", "WORKER", "jev", "jev", False, "WORKER", None))
            self.assertEqual((r["reply"], r["outcome_kind"], r["outcome_rejected"]), (br.FAKE_REPLY, "answer", None))
            self.assertIsNone(r["memory_gate_explicit"])                 # only a real Worker's memory tool reaches has_explicit_save_intent
            # B-R6: nothing was really called in fake mode, so the columns that say "what was called" are null, not a copy of required_tools
            self.assertIsNone(r["tools"]); self.assertIsNone(r["tool_ok"])
            self.assertIsInstance(r["required_tools"], list)
            self.assertIn("fake_worker", r["fidelity"]); self.assertIn("fake_jev", r["fidelity"])
        self.assertEqual(rows[0]["required_tools"], ["get_weather"])         # B-R5: the regex-derived requirement has its own column

    def test_jsonl_keeps_opaque_ids_and_a_constant_fake_route_is_recorded_not_judged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, [json.dumps({"id": "case-A", "text": SENTENCES[0]}), json.dumps({"id": "7", "text": SENTENCES[1]})], "in.jsonl")
            out = os.path.join(d, "out.jsonl")
            self.assertEqual(br.main([inp, out, "--fake-route", "FAST"]), 0)
            rows = _rows(out)
        self.assertEqual([r["id"] for r in rows], ["case-A", "7"])
        self.assertTrue(all(r["route"] == "FAST" and r["handled_by"] == "fast" and r["reply"] == "" and r["required_tools"] is None for r in rows))

    def test_the_output_does_not_echo_the_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            br.main([_write(d, SENTENCES), out])
            blob = open(out, encoding="utf-8").read()
        for m in MARKERS:
            self.assertNotIn(m, blob)

    # ---- B-R1 / B-R7: who decided, and why it is degraded -------------------------------------------------------------------------
    def test_a_jev_failure_is_recorded_as_degraded_heuristic_with_its_kind_chosen_by_class_and_cause_only(self) -> None:
        import aiohttp
        try:
            try:
                raise ValueError("not json")
            except ValueError as v:
                raise RuntimeError("jev bad json: ...") from v
        except RuntimeError as bad_json:
            chained = bad_json
        status = aiohttp.ClientResponseError(None, (), status=502, message="x")
        cases = ((RuntimeError("jev HTTP 500: body"), "runtime_error"), (chained, "bad_json"), (status, "http_status"), (asyncio.TimeoutError(), "timeout"),
                 (TimeoutError(), "timeout"), (ConnectionRefusedError(), "conn"), (aiohttp.ClientConnectionError(), "conn"), (KeyError("k"), "jev_error"))
        for exc, kind in cases:
            row = self.run_one("现在几点了", jev=_jev_raising(exc))
            self.assertEqual((row["via"], row["route_via"], row["degraded"], row["degraded_reason"], row["degraded_exc"], row["jev_route"], row["error"]),
                             ("heuristic", "heuristic", True, kind, type(exc).__name__, None, None), exc)

    def test_the_degraded_kind_does_not_depend_on_the_message(self) -> None:
        a = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("timeout")))
        b = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("jev bad json: x")))
        self.assertEqual((a["degraded_reason"], b["degraded_reason"]), ("runtime_error", "runtime_error"))

    def test_the_jev_error_message_is_never_read_or_recorded(self) -> None:
        row = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("echo ZQXV7731")))
        self.assertNotIn("ZQXV7731", json.dumps(row, ensure_ascii=False))

    def test_a_jev_answer_without_a_usable_route_is_not_recorded_as_a_jev_decision(self) -> None:      # api.py:93 ``or heuristic_route``
        row = self.run_one("现在几点了", jev=_jev_answering(None))
        self.assertEqual((row["via"], row["route_via"], row["jev_route"], row["degraded"], row["degraded_reason"]), ("jev", "heuristic", None, True, "no_usable_route"))
        row = self.run_one("现在几点了", jev=_jev_answering(None, raw={"answers": {}}))
        self.assertEqual((row["route_via"], row["degraded"]), ("heuristic", True))

    def test_a_jev_answer_that_is_not_a_route_enum_is_invalid_not_a_jev_decision(self) -> None:
        row = self.run_one("好的谢谢", jev=_jev_answering("BOGUS"))
        self.assertEqual((row["via"], row["route_via"], row["jev_route"], row["route"], row["degraded"], row["degraded_reason"]),
                         ("jev", "jev_invalid", "BOGUS", "WORKER", True, "invalid_route"))

    def test_a_valid_jev_route_is_a_jev_decision_and_not_degraded(self) -> None:
        for r in ("WORKER", "FAST", "CLARIFY", "STOP_SPEAKING"):
            row = self.run_one("今天天气", jev=_jev_answering(r))
            self.assertEqual((row["route_via"], row["degraded"], row["degraded_reason"], row["control_via"]), ("jev", False, None, None), r)

    def _ledger_with_a_finished_task(self):
        from sophon_live import coordinator as C
        ledger = C.Ledger()
        first = self.run_one("明天上海天气怎么样", jev=_jev_answering("WORKER"), ledger=ledger)
        self.assertEqual(first["worker_status"], "completed")
        return ledger

    def test_control_without_a_target_is_decided_by_the_heuristic_and_says_so(self) -> None:           # api.py:136
        ledger = self._ledger_with_a_finished_task()
        row = self.run_one("取消吧", jev=_jev_answering("CONTROL"), ledger=ledger)
        self.assertEqual((row["route_via"], row["control_via"], row["degraded"], row["degraded_reason"], row["via"]), ("jev", "heuristic", True, "control_no_target", "jev"))
        self.assertEqual(row["decision"], "CANCEL")

    def test_control_where_the_heuristic_finds_nothing_is_still_marked_heuristic(self) -> None:
        row = self.run_one("嗯", jev=_jev_answering("CONTROL"))                    # empty ledger: heuristic_control ran, found nothing, a new job follows
        self.assertEqual((row["decision"], row["control_via"], row["degraded"], row["degraded_reason"]), ("NEW", "heuristic", True, "control_no_target"))

    def test_control_with_a_jev_named_target_or_new_is_a_jev_decision(self) -> None:
        ledger = self._ledger_with_a_finished_task()
        latest = ledger.latest()
        row = self.run_one("进度", jev=_jev_answering("CONTROL", f"STATUS:{latest.task_id}:{latest.version}"), ledger=ledger)
        self.assertEqual((row["decision"], row["control_via"], row["degraded"], row["jev_control"]), ("STATUS", "jev", False, f"STATUS:{latest.task_id}:{latest.version}"))
        row = self.run_one("另一件事", jev=_jev_answering("CONTROL", "NEW"), ledger=ledger)
        self.assertEqual((row["decision"], row["control_via"], row["degraded"]), ("NEW", "jev", False))
        row = self.run_one("另一件事", jev=_jev_answering("CONTROL", "CANCEL:nope:1"), ledger=ledger)          # a task the Ledger does not have
        self.assertEqual((row["decision"], row["control_via"], row["degraded"]), ("NEW", "jev", False))

    def test_the_refine_guard_is_recorded_as_a_guard_decision_and_the_jev_is_not_asked(self) -> None:
        ledger = self._ledger_with_a_finished_task()
        row = self.run_one("明天北京天气怎么样", jev=_jev_raising(AssertionError("must not be asked")), ledger=ledger)
        self.assertEqual((row["via"], row["route_via"], row["control_via"], row["jev_route"], row["degraded"], row["error"]), ("guard", "guard", "guard", None, False, None))

    def test_jev_switched_off_is_not_degraded_but_route_via_says_heuristic(self) -> None:
        row = self.run_one("现在几点了", jev=_jev_answering("FAST"), use_jev=False)
        self.assertEqual((row["via"], row["route_via"], row["degraded"], row["degraded_reason"]), ("heuristic", "heuristic", False, None))

    def test_an_error_keeps_what_was_already_recorded(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("secret ZQXV7731 text")
        row = self.run_one("现在几点了", jev=_jev_answering("WORKER"), worker=bad_worker)
        self.assertEqual((row["error"], row["error_stage"], row["route"], row["route_via"], row["jev_route"], row["via"], row["handled_by"], row["degraded"]),
                         ("ValueError", "worker", "WORKER", "jev", "WORKER", "jev", "worker", False))
        self.assertIsInstance(row["required_tools"], list)
        self.assertNotIn("ZQXV7731", json.dumps(row, ensure_ascii=False))
        row = self.run_one("现在几点了", jev=_jev_raising(TimeoutError()), worker=bad_worker)             # the degraded reason survives the later crash
        self.assertEqual((row["error"], row["via"], row["degraded"], row["degraded_reason"]), ("ValueError", "heuristic", True, "timeout"))

    def test_a_crash_in_the_run_is_recorded_as_a_class_name_only(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("secret ZQXV7731 text")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            with mock.patch.object(br, "_fake_worker", bad_worker):
                br.main([_write(d, [SENTENCES[0]]), out])
            (r,) = _rows(out)
            self.assertEqual((r["id"], r["error"], r["error_stage"], r["route"], r["route_via"], r["jev_route"], r["via"]), ("s0001", "ValueError", "worker", "WORKER", "jev", "WORKER", "jev"))
            self.assertNotIn("ZQXV7731", json.dumps(r, ensure_ascii=False))
            self.assertEqual(json.load(open(out + ".done")), {"n": 1, "errors": 1, "mode": "fake", "complete": True})

    # ---- B-R2: fidelity with the production path ----------------------------------------------------------------------------------
    def test_instructions_history_and_the_rewritten_task_reach_the_worker_like_in_production(self) -> None:
        seen: list[dict] = []
        async def spy_worker(job, *, instructions="", history=None):   # noqa: ANN001
            seen.append({"objective": job.objective, "instructions": instructions, "history": list(history or [])})
            return await br._fake_worker(job)
        hist = [{"task": f"t{i}", "answer": f"a{i}"} for i in range(9)]
        row = self.run_one("今天北京天气", jev=_jev_answering("WORKER"), worker=spy_worker, task="查询北京今天的天气", instructions="SYS PROMPT", history=hist,
                           fidelity=br.fidelity_codes(mode="fake", has_task=True, instructions="x", history_mode="history_static", shared_ledger=False))
        (s,) = seen
        self.assertEqual((s["objective"], s["instructions"]), ("查询北京今天的天气", "SYS PROMPT"))
        self.assertEqual(s["history"], hist[-6:])
        self.assertIn("rewrite_from_input", row["fidelity"]); self.assertNotIn("no_frontend_rewrite", row["fidelity"])
        self.assertIn("instructions_from_file", row["fidelity"]); self.assertIn("history_static", row["fidelity"])

    def test_the_raw_text_stays_the_sentence_when_a_rewrite_is_supplied(self) -> None:
        got: list = []
        from sophon_live.coordinator import api
        orig = api.slot_edit
        def spy(text, raw, ledger):   # noqa: ANN001
            got.append((text, raw))
            return orig(text, raw, ledger)
        with mock.patch.object(api, "slot_edit", spy):
            self.run_one("原话ZQXV", jev=_jev_answering("WORKER"), task="改写后的任务")
        self.assertEqual(got, [("改写后的任务", "原话ZQXV")])

    def test_playback_active_and_language_are_parameters_and_always_listed_as_fixed(self) -> None:
        got: list = []
        from sophon_live.coordinator import route
        orig = route.build_state
        from sophon_live.coordinator import api
        def spy(text, ledger, *, playback_active):   # noqa: ANN001
            got.append(playback_active)
            return orig(text, ledger, playback_active=playback_active)
        with mock.patch.object(api, "build_state", spy):
            row = self.run_one("今天", jev=_jev_answering("FAST"), playback_active=True,
                               fidelity=br.fidelity_codes(mode="fake", has_task=False, instructions="", history_mode="no_history", shared_ledger=False,
                                                     playback_active=True, language="en"))
        self.assertEqual(got, [True])
        for code in ("playback_active=true", "language=en,jev_state_language=zh", "no_asr_transcript", "no_frontend_rewrite", "no_instructions", "no_history", "ledger_per_sentence"):
            self.assertIn(code, row["fidelity"])
        self.assertNotIn("playback_active=false", row["fidelity"])
        self.assertIn("playback_active=false", br.fidelity_codes(mode="fake", has_task=False, instructions="", history_mode="no_history", shared_ledger=False))
        self.assertIn("language=zh,jev_state_language=zh", br.fidelity_codes(mode="fake", has_task=False, instructions="", history_mode="no_history", shared_ledger=False))

    def test_main_records_the_actual_playback_and_language_values_in_fidelity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.jsonl")
            self.assertEqual(br.main([_write(d, ["一句"]), out, "--playback-active", "--language", "en"]), 0)
            (r,) = _rows(out)
        self.assertIn("playback_active=true", r["fidelity"]); self.assertIn("language=en,jev_state_language=zh", r["fidelity"])

    def test_main_passes_files_through_static_history_resets_per_sentence_and_shared_history_accumulates(self) -> None:
        seen: list[list] = []
        real_fake = br._fake_worker
        async def spy_worker(job, *, instructions="", history=None):   # noqa: ANN001
            seen.append([instructions, list(history or [])])
            return await real_fake(job)
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, ["一句", "两句"])
            ins = _write(d, ["SYS"], "ins.txt")
            hist = _write(d, [json.dumps({"task": "t0", "answer": "a0"})], "h.jsonl")
            with mock.patch.object(br, "_fake_worker", spy_worker):
                self.assertEqual(br.main([inp, os.path.join(d, "o1.jsonl"), "--instructions-file", ins, "--history-jsonl", hist]), 0)
                rows1 = _rows(os.path.join(d, "o1.jsonl"))
                self.assertEqual(br.main([inp, os.path.join(d, "o2.jsonl"), "--instructions-file", ins, "--history-jsonl", hist, "--shared-ledger"]), 0)
                rows2 = _rows(os.path.join(d, "o2.jsonl"))
        self.assertEqual(seen[0], ["SYS\n", [{"task": "t0", "answer": "a0"}]])
        self.assertEqual(seen[1], ["SYS\n", [{"task": "t0", "answer": "a0"}]])                  # not accumulated without --shared-ledger
        self.assertEqual(seen[2][1], [{"task": "t0", "answer": "a0"}])
        self.assertEqual(len(seen[3][1]), 2)                                                      # the first finished job was appended
        self.assertIn("history_static", rows1[0]["fidelity"]); self.assertIn("history_accumulates", rows2[0]["fidelity"])
        self.assertIn("ledger_shared", rows2[0]["fidelity"])

    # ---- I-1: outcome columns and the history only grows on answers, like gateway.py ----------------------------------------------
    def _needs_input_worker(self, slots=("city",)):
        from sophon_live.coordinator.schemas import WorkerResult
        async def w(job, **kw):   # noqa: ANN001
            return WorkerResult(task_id=job.task_id, version=job.version, status="needs_input", spoken_answer="", needs_slots=list(slots), freshness_token=job.freshness_token)
        return w

    def test_a_needs_input_outcome_is_recorded_as_clarify(self) -> None:
        from sophon_live.coordinator.clarify import clarify_for_slots
        row = self.run_one("今天天气", jev=_jev_answering("WORKER"), worker=self._needs_input_worker())
        self.assertEqual((row["outcome_kind"], row["worker_status"], row["reply"], row["handled_by"]), ("clarify", "needs_input", clarify_for_slots(["city"]), "worker"))

    def test_a_stale_result_is_recorded_as_stale(self) -> None:
        from sophon_live.coordinator.schemas import WorkerResult
        async def w(job, **kw):   # noqa: ANN001
            return WorkerResult(task_id=job.task_id, version=job.version + 1, status="completed", spoken_answer="x", freshness_token=job.freshness_token)
        row = self.run_one("好的", jev=_jev_answering("WORKER"), worker=w)
        self.assertEqual((row["outcome_kind"], row["reply"]), ("stale", ""))

    def test_a_rejected_answer_records_why_finish_job_refused_it(self) -> None:
        from sophon_live.coordinator.schemas import WorkerResult
        async def w(job, **kw):   # noqa: ANN001
            return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer="北京今天二十度", evidence=[], freshness_token=job.freshness_token)
        row = self.run_one("今天北京天气怎么样", jev=_jev_answering("WORKER"), worker=w)
        self.assertEqual((row["outcome_kind"], row["outcome_rejected"]), ("answer", "no_tool_result"))

    def test_a_needs_input_text_never_reaches_a_later_workers_history_but_an_answer_does(self) -> None:
        from sophon_live.coordinator.clarify import clarify_for_slots
        from sophon_live.coordinator.schemas import WorkerResult
        seen: list[list] = []
        real_fake = br._fake_worker
        stale_wanted = {"n": 0}
        async def w(job, *, instructions="", history=None):   # noqa: ANN001
            seen.append([dict(h) for h in (history or [])])
            k = len(seen)
            if k == 1:
                return await self._needs_input_worker()(job)
            if k == 3:                                               # a stale result in the middle
                return WorkerResult(task_id=job.task_id, version=job.version + 1, status="completed", spoken_answer="STALEANSWER", freshness_token=job.freshness_token)
            return await real_fake(job)
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, ["第一句", "第二句", "第三句", "第四句"])
            out = os.path.join(d, "o.jsonl")
            with mock.patch.object(br, "_fake_worker", w):
                self.assertEqual(br.main([inp, out, "--shared-ledger"]), 0)
            rows = _rows(out)
        self.assertEqual([r["outcome_kind"] for r in rows], ["clarify", "answer", "stale", "answer"])
        clarify_text = clarify_for_slots(["city"])
        self.assertEqual(seen[0], [])
        self.assertEqual(seen[1], [])                                                   # the clarify question was not appended
        self.assertEqual(len(seen[2]), 1)                                               # the answer of sentence 2 was
        self.assertEqual(seen[2][0]["answer"], br.FAKE_REPLY)
        self.assertEqual(len(seen[3]), 1)                                               # the stale result was not
        self.assertNotIn(clarify_text, json.dumps(seen[1:], ensure_ascii=False))
        self.assertNotIn("STALEANSWER", json.dumps(seen, ensure_ascii=False))

    def test_the_history_is_capped_at_twelve_like_the_gateway(self) -> None:
        seen: list[int] = []
        real_fake = br._fake_worker
        async def w(job, *, instructions="", history=None):   # noqa: ANN001
            seen.append(len(history or []))
            return await real_fake(job)
        hist = [{"task": f"t{i}", "answer": f"a{i}"} for i in range(11)]
        sink = list(hist)
        from sophon_live import coordinator as C
        async def go():
            for k in range(3):
                await br.run_one(f"s{k}", br.Opaque(f"句{k}"), ledger=C.Ledger(), jev=_jev_answering("WORKER"), worker=br._fake_worker, history=sink, history_sink=sink)
        with br.quiet_and_sealed(False):
            asyncio.run(go())
        self.assertEqual(len(sink), 12)
        self.assertEqual(sink[-1]["answer"], br.FAKE_REPLY)

    # ---- I-2: the keyword functions' conclusions are recorded, nothing else, and everything is restored -----------------------------
    def test_the_keyword_functions_return_values_are_recorded_and_the_functions_restored(self) -> None:
        from sophon_live.coordinator import api, clarify, memory_gate
        orig_slots, orig_mem = api.missing_slots_for_text, memory_gate.has_explicit_save_intent
        sent = "今天天气怎么样"
        row = self.run_one(sent, jev=_jev_answering("CLARIFY"))
        self.assertEqual(row["clarify_slots"], [clarify.missing_slots_for_text(sent)])
        self.assertEqual(row["handled_by"], "clarify")
        self.assertIsNone(self.run_one("好的", jev=_jev_answering("FAST"))["clarify_slots"])                   # not called: null, not []
        async def mem_worker(job, **kw):   # noqa: ANN001
            memory_gate.has_explicit_save_intent(job.objective); memory_gate.has_explicit_save_intent("")
            return await br._fake_worker(job)
        for sent in ("记住我喜欢咖啡", "随便聊聊"):
            row = self.run_one(sent, jev=_jev_answering("WORKER"), worker=mem_worker)
            self.assertEqual(row["memory_gate_explicit"], [orig_mem(sent), False])             # the production function's own answers, as returned
        self.assertIs(api.missing_slots_for_text, orig_slots); self.assertIs(memory_gate.has_explicit_save_intent, orig_mem)

    def test_the_wrappers_are_restored_even_when_the_run_crashes(self) -> None:
        from sophon_live.coordinator import api, jev_client, memory_gate
        before = (api.missing_slots_for_text, memory_gate.has_explicit_save_intent, jev_client.system_one)
        async def bad(job, **kw):   # noqa: ANN001
            raise ValueError("x")
        row = self.run_one("今天", jev=_jev_answering("WORKER"), worker=bad)
        self.assertEqual(row["error"], "ValueError")
        self.assertEqual((api.missing_slots_for_text, memory_gate.has_explicit_save_intent, jev_client.system_one), before)

    def test_the_wrapped_functions_pass_everything_through_unchanged(self) -> None:
        from sophon_live.coordinator import api, memory_gate
        calls: list = []
        async def w(job, **kw):   # noqa: ANN001
            calls.append((api.missing_slots_for_text("天气"), memory_gate.has_explicit_save_intent("记住"), memory_gate.has_explicit_save_intent(None)))
            return await br._fake_worker(job)
        self.run_one("x", jev=_jev_answering("WORKER"), worker=w)
        self.assertEqual(calls, [(["city"], True, False)])

    # ---- gates, exit codes ----------------------------------------------------------------------------------------------------
    def test_real_mode_is_refused_without_the_env_gate_and_gateway_mode_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p, out = _write(d, SENTENCES), os.path.join(d, "out.jsonl")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(br.ALLOW_REAL_ENV, None)
                self.assertEqual(br.main([p, out, "--mode", "real", "--jev-url", "http://127.0.0.1:1"]), 4)
                os.environ[br.ALLOW_REAL_ENV] = "1"
                self.assertEqual(br.main([p, out, "--mode", "real"]), 4)             # no --jev-url
                os.environ[br.ALLOW_REAL_ENV] = "true"
                self.assertEqual(br.main([p, out, "--mode", "real", "--jev-url", "http://127.0.0.1:1"]), 4)     # only the literal 1 opens the gate
            self.assertEqual(br.main([p, out, "--mode", "gateway"]), 3)
            self.assertFalse(os.path.exists(out)); self.assertFalse(os.path.exists(out + ".done"))

    # ---- B-R4: the output never overwrites anything ---------------------------------------------------------------------------
    def test_the_output_may_not_be_the_input_or_an_existing_file_and_nothing_is_changed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, SENTENCES)
            before = open(inp, "rb").read()
            self.assertEqual(br.main([inp, inp]), 5)
            self.assertEqual(br.main([inp, os.path.join(d, ".", os.path.basename(inp))]), 5)         # same file by another spelling
            link = os.path.join(d, "link.jsonl"); os.symlink(inp, link)
            self.assertEqual(br.main([inp, link]), 5)
            self.assertEqual(open(inp, "rb").read(), before)
            existing = os.path.join(d, "mine.jsonl"); open(existing, "w").write("KEEP\n")
            self.assertEqual(br.main([inp, existing]), 5)
            self.assertEqual(open(existing).read(), "KEEP\n")
            out = os.path.join(d, "new.jsonl"); open(out + ".done", "w").write("old")
            self.assertEqual(br.main([inp, out]), 5)                                              # a stale .done would make an interrupted run look complete
            self.assertFalse(os.path.exists(out))

    def test_the_exclusive_create_is_the_last_line_of_defence_even_if_the_early_check_is_bypassed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            existing = os.path.join(d, "mine.jsonl"); open(existing, "w").write("KEEP\n")
            with mock.patch.object(br, "output_refused", lambda p: False):
                with self.assertRaises(FileExistsError):
                    br.open_output(existing)
            self.assertEqual(open(existing).read(), "KEEP\n")
            with self.assertRaises(SystemExit) as cm:
                br.open_output(existing)
            self.assertEqual(cm.exception.code, 5)

    def test_the_output_and_done_files_are_private_and_done_marks_a_finished_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp, out = _write(d, SENTENCES), os.path.join(d, "out.jsonl")
            old = os.umask(0o022)
            try:
                self.assertEqual(br.main([inp, out]), 0)
            finally:
                os.umask(old)
            self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(out + ".done").st_mode), 0o600)
            self.assertEqual(json.load(open(out + ".done")), {"n": 4, "errors": 0, "mode": "fake", "complete": True})
            out2 = os.path.join(d, "out2.jsonl")
            async def interrupted(*a, **k):   # noqa: ANN002, ANN003
                raise KeyboardInterrupt
            with mock.patch.object(br, "run_one", interrupted), self.assertRaises(KeyboardInterrupt):
                br.main([inp, out2])
            self.assertTrue(os.path.exists(out2)); self.assertFalse(os.path.exists(out2 + ".done"))   # partial output, no completion marker

    def test_a_missing_output_directory_or_input_is_a_clear_exit_code_not_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, SENTENCES)
            se = io.StringIO()
            with mock.patch("sys.stderr", se):
                self.assertEqual(br.main([inp, os.path.join(d, "no_such_dir", "out.jsonl")]), 2)
                self.assertEqual(br.main([os.path.join(d, "nope.txt"), os.path.join(d, "o.jsonl")]), 2)
                self.assertEqual(br.main([inp, os.path.join(d, "o2.jsonl"), "--instructions-file", os.path.join(d, "nope")]), 2)
                self.assertEqual(br.main([inp, os.path.join(d, "o3.jsonl"), "--history-jsonl", os.path.join(d, "nope")]), 2)
            self.assertIn("FileNotFoundError", se.getvalue()); self.assertNotIn("Traceback", se.getvalue())
            self.assertFalse(os.path.exists(os.path.join(d, "o.jsonl")))

    def test_a_done_marker_that_appears_during_the_run_fails_clearly_and_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp, out = _write(d, SENTENCES), os.path.join(d, "out.jsonl")
            real_fake = br._fake_worker
            async def w(job, **kw):   # noqa: ANN001
                if not os.path.exists(out + ".done"):
                    open(out + ".done", "w").write("FOREIGN")
                return await real_fake(job)
            se = io.StringIO()
            with mock.patch.object(br, "_fake_worker", w), mock.patch("sys.stderr", se):
                self.assertEqual(br.main([inp, out]), 6)
            self.assertEqual(open(out + ".done").read(), "FOREIGN")                    # not ours: never overwritten, never claimed
            self.assertEqual(len(_rows(out)), 4)                                       # the output itself is complete
            self.assertIn("could not be created", se.getvalue()); self.assertNotIn("Traceback", se.getvalue())

    def test_a_lost_race_for_the_output_file_itself_is_a_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp, out = _write(d, SENTENCES), os.path.join(d, "out.jsonl")
            def racing(path):                                    # the file appears between the early check and the exclusive create
                open(path, "w").write("THEIRS")
                return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with mock.patch.object(br, "open_output", racing):
                self.assertEqual(br.main([inp, out]), 5)
            self.assertEqual(open(out).read(), "THEIRS")

    # ---- S-4 / I-1': metamorphic tests (independent of how the runner's code names things) ---------------------------------------
    BASELINE = "占位占位"

    @staticmethod
    def _char_sentences(full_cjk: bool) -> list[str]:
        import string
        ascii_chars = [c for c in string.printable if not c.isspace() and c != "{"]           # a line starting with { is read as JSON
        cjk = [chr(c) for c in range(0x4E00, 0xA000)]
        return ascii_chars + (cjk if full_cjk else cjk[::16])

    @staticmethod
    def _corpus_sentences() -> list[str]:
        from tests import intent_cases as ic                       # the repo's own labelled utterances: real words, Chinese and English
        out = []
        for name in ("OFFICIAL_REALTIME", "OFFICIAL_FREE", "CRITIC_REALTIME", "CRITIC_FREE", "HARDER_REALTIME", "HARDER_FREE", "HELDOUT_REALTIME", "HELDOUT_FREE",
                     "CONVERSION_REALTIME", "CONVERSION_FREE"):
            out += [x for x in getattr(ic, name, ()) if isinstance(x, str) and x.strip() and not x.lstrip().startswith("{")]
        return list(dict.fromkeys(out))

    def _rows_for(self, sents: list[str], route: str) -> list[dict]:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.jsonl")
            so = io.StringIO()
            with mock.patch("sys.stdout", so):
                self.assertEqual(br.main([_write(d, sents), out, "--fake-route", route]), 0)
            return _rows(out)

    def test_every_single_character_and_every_corpus_sentence_is_treated_exactly_like_a_neutral_sentence(self) -> None:
        """The runner must not fork on WHAT the sentence says. Every single character of the character set (so any 'contains X' / 'starts with X' /
        'ends with X' fork on a one-character X fires), and every sentence of the repo's labelled corpus (so forks on a word the corpus has fire), gets a
        row that equals the row of a neutral sentence in every column except ``reply``, ``required_tools`` and ``clarify_slots`` -- the three that
        legitimately carry the production keyword functions' answers. ``required_tools`` must equal ``intent.required_tools`` of the same sentence,
        and a fake Worker's reply is constant. A fork on a multi-character word the corpus does not contain is NOT found; code review remains the backstop."""
        from sophon_live.coordinator import intent
        from sophon_live.coordinator.route import ROUTE_CRITERIA
        corpus = self._corpus_sentences()
        self.assertGreater(len(corpus), 150)
        for route in sorted(ROUTE_CRITERIA):
            sents = self._char_sentences(full_cjk=route == "WORKER") + corpus
            self.assertEqual(len(set(sents)), len(sents))
            base, *rows = self._rows_for([self.BASELINE] + sents, route)
            self.assertEqual(len(rows), len(sents))
            for sent, r in zip(sents, rows):
                if route == "WORKER":
                    self.assertEqual(r["reply"], br.FAKE_REPLY, sent)
                    self.assertEqual(r["required_tools"], intent.required_tools(sent), sent)
                for key in set(r) - {"id", "reply", "required_tools", "clarify_slots"}:
                    self.assertEqual(r[key], base[key], f"{route} {key} differs for {sent!r}")

    def test_the_metamorphic_test_can_see_a_character_fork(self) -> None:
        """Not vacuous: a runner that forks on one character gives a different row for exactly that character."""
        orig_run_one = br.run_one
        async def run_one(sid, text, **kw):   # noqa: ANN001
            row = await orig_run_one(sid, text, **kw)
            if "x" in br.reveal(text):                                 # a planted judgment (in the TEST, never in the runner)
                row["handled_by"] = "fast"
            return row
        with mock.patch.object(br, "run_one", run_one):
            rows = self._rows_for([self.BASELINE, "x", "y"], "WORKER")
        self.assertEqual([r["handled_by"] for r in rows], ["worker", "fast", "worker"])

    # ---- I-1': a runtime probe: does the runner's OWN code touch the sentence? ---------------------------------------------------
    @staticmethod
    def _probe_type(hits: list):
        """A str subclass whose every method records a call made FROM a frame of tools/blind_runner.py (production modules are ignored)."""
        runner = os.path.realpath(br.__file__)
        names = ["__contains__", "__getitem__", "__len__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__iter__", "__add__", "__radd__",
                 "__mod__", "__mul__", "__format__", "__str__", "__repr__", "find", "rfind", "index", "rindex", "count", "startswith", "endswith", "lower",
                 "upper", "casefold", "split", "rsplit", "splitlines", "partition", "rpartition", "strip", "lstrip", "rstrip", "replace", "translate",
                 "encode", "isascii", "isdigit", "isalpha", "isalnum", "isspace", "title", "swapcase", "removeprefix", "removesuffix", "format", "zfill",
                 "ljust", "rjust", "center", "expandtabs"]
        def mk(name):
            base = getattr(str, name)
            def m(self, *a, **k):
                f = sys._getframe(1)
                if os.path.realpath(f.f_code.co_filename) == runner:
                    hits.append(name)
                return base(self, *a, **k)
            m.__name__ = name
            return m
        ns = {n: mk(n) for n in names if hasattr(str, n)}
        ns["__hash__"] = str.__hash__
        return type("TouchProbe", (str,), ns)

    def test_the_probe_is_not_vacuous_it_sees_a_touch_from_runner_code_and_ignores_other_code(self) -> None:
        hits: list = []
        T = self._probe_type(hits)
        t = T("abc")
        _ = "x" in t; _ = t.startswith("a"); _ = t[:1]; _ = len(t)                      # this test file is not the runner: no hits
        self.assertEqual(hits, [])
        ns: dict = {}
        exec(compile("def touch(t):\n    return ('x' in t, t.find('a'), t[:2], len(t), t == 'q', t.startswith('a'))\n", os.path.realpath(br.__file__), "exec"), ns)
        ns["touch"](t)
        self.assertEqual(sorted(set(hits)), sorted({"__contains__", "find", "__getitem__", "__len__", "__eq__", "startswith"}))

    def test_the_runners_own_code_never_touches_the_sentence_in_any_route_ledger_mode_or_error_path(self) -> None:
        from sophon_live.coordinator.route import ROUTE_CRITERIA
        hits: list = []
        T = self._probe_type(hits)
        real_fake = br._fake_worker
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("x")
        async def fake_jev_failing(route):   # noqa: ANN001, ARG001
            return _jev_raising(RuntimeError("down"))
        async def go(route, shared, worker, jev_factory=None):
            sents = [br.Sentence("a", br.Opaque(T("今天天气怎么样 zq")), None), br.Sentence("b", br.Opaque(T("好的谢谢")), br.Opaque(T("改写后的任务"))),
                     br.Sentence("c", br.Opaque(T("取消吧")), None)]
            with tempfile.TemporaryDirectory() as d:
                with mock.patch.object(br, "_fake_worker", worker):
                    if jev_factory is not None:
                        with mock.patch.object(br, "_fake_jev", jev_factory):
                            return await br.run(sents, os.path.join(d, "o.jsonl"), fake_route=route, shared_ledger=shared)
                    return await br.run(sents, os.path.join(d, "o.jsonl"), fake_route=route, shared_ledger=shared)
        with br.quiet_and_sealed(False):
            for route in sorted(ROUTE_CRITERIA):
                for shared in (False, True):
                    self.assertEqual(asyncio.run(go(route, shared, real_fake)), 3)
                    self.assertEqual(asyncio.run(go(route, shared, bad_worker)), 3)
            self.assertEqual(asyncio.run(go("WORKER", True, real_fake, fake_jev_failing)), 3)
        self.assertEqual(hits, [], "the runner's own code touched the sentence text")

    # ---- S-2: input handling --------------------------------------------------------------------------------------------------
    def test_input_handling_bom_null_text_bad_ids_bad_route_and_bad_history(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bom = os.path.join(d, "bom.jsonl")
            open(bom, "w", encoding="utf-8-sig").write(json.dumps({"id": "b", "text": "ZQXV7731"}) + "\n")
            self.assertEqual([(s.id, br.reveal(s.text)) for s in br.read_sentences(bom)], [("b", "ZQXV7731")])           # parsed as JSON, not taken as a sentence
            for bad in ({"id": "a", "text": None}, {"id": "a", "text": ""}, {"id": "a", "text": "  "}, {"id": "a", "text": 5}, {"id": "a"},
                        {"id": "a", "text": "ZQXV7731", "task": 7}, {"id": "a", "text": "ZQXV7731", "task": ""}):
                p = _write(d, [json.dumps(bad)], "bad.jsonl")
                with self.assertRaises(SystemExit) as cm:
                    br.read_sentences(p)
                self.assertNotIn("ZQXV7731", str(cm.exception))
            self.assertEqual([s.id for s in br.read_sentences(_write(d, [json.dumps({"text": "x"}), "", "plain", json.dumps({"id": "k", "text": "y"})], "m.jsonl"))],
                             ["s0001", "s0002", "k"])
            h = _write(d, ['{"task": 1, "answer": "ZQXV7731"}'], "h.jsonl")
            with self.assertRaises(SystemExit) as cm:
                br.read_history(h)
            self.assertNotIn("ZQXV7731", str(cm.exception))
            so, se = io.StringIO(), io.StringIO()
            with mock.patch("sys.stderr", se):
                self.assertEqual(br.main([_write(d, ["x"]), os.path.join(d, "o.jsonl"), "--fake-route", "BOGUS"]), 2)
            self.assertFalse(os.path.exists(os.path.join(d, "o.jsonl")))

    def test_invalid_input_exits_2_with_one_line_naming_the_line_number_and_never_the_content(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cases = {"bad json": ['{"id": "a", "text": "ZQXV7731"', ], "null text": [json.dumps({"id": "a", "text": None})],
                     "dup ids": [json.dumps({"id": "a", "text": "ZQXV7731"}), json.dumps({"id": "a", "text": "y"})]}
            for name, lines in cases.items():
                out = os.path.join(d, name.replace(" ", "_") + ".jsonl")
                se = io.StringIO()
                with mock.patch("sys.stderr", se):
                    self.assertEqual(br.main([_write(d, lines, "in_" + name.replace(" ", "_") + ".jsonl"), out]), 2, name)
                err = se.getvalue()
                self.assertEqual(len(err.strip().splitlines()), 1, err)
                self.assertNotIn("ZQXV7731", err); self.assertNotIn("Traceback", err)
                self.assertFalse(os.path.exists(out) or os.path.exists(out + ".done"))
            se = io.StringIO()
            with mock.patch("sys.stderr", se):
                br.main([_write(d, ["ok", json.dumps({"text": ""})], "second.jsonl"), os.path.join(d, "o9.jsonl")])
            self.assertTrue(se.getvalue().startswith("blind_runner: input line"))
            h = _write(d, ['{"task": 1, "answer": "ZQXV7731"}'], "h.jsonl")
            se = io.StringIO()
            with mock.patch("sys.stderr", se):
                self.assertEqual(br.main([_write(d, ["ok"], "okin.txt"), os.path.join(d, "o10.jsonl"), "--history-jsonl", h]), 2)
            self.assertNotIn("ZQXV7731", se.getvalue()); self.assertEqual(len(se.getvalue().strip().splitlines()), 1)

    def test_a_file_that_is_not_utf8_exits_2_with_one_line_naming_the_position_for_all_three_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            def bad(name: str, before: bytes) -> str:
                path = os.path.join(d, name)
                open(path, "wb").write(before + b"\xff\xfe tail\n")
                return path
            ok = _write(d, ["fine"], "ok.txt")
            def case(name, what, prefix, line_no, argv_fn):
                return name, (argv_fn(bad(name.replace(" ", "_") + ".bin", prefix)), what, line_no, len(prefix))
            cases = dict([
                case("input", "input file", b"", 1, lambda f: [f, os.path.join(d, "o1.jsonl")]),
                case("input line 3", "input file", b"ZQXV7731 fine\nsecond\n", 3, lambda f: [f, os.path.join(d, "o2.jsonl")]),
                case("instructions", "instructions file", b"SYSZQXV7731 ", 1, lambda f: [ok, os.path.join(d, "o3.jsonl"), "--instructions-file", f]),
                case("history", "history file", b'{"task": "a", "answer": "ZQXV7731"}\n', 2, lambda f: [ok, os.path.join(d, "o4.jsonl"), "--history-jsonl", f]),
            ])
            for name, (argv, what, line_no, offset) in cases.items():
                se = io.StringIO()
                with mock.patch("sys.stderr", se):
                    self.assertEqual(br.main(argv), 2, name)
                err = se.getvalue()
                self.assertEqual(err.strip(), f"blind_runner: {what} is not valid UTF-8 (line {line_no}, byte offset {offset})", name)
                self.assertNotIn("Traceback", err); self.assertNotIn("ZQXV7731", err)
                self.assertFalse(os.path.exists(argv[1]) or os.path.exists(argv[1] + ".done"), name)

    def test_a_duplicate_id_is_reported_with_its_line_number_never_its_value(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, [json.dumps({"id": "SECRETID77", "text": "a"}), "", json.dumps({"id": "b", "text": "c"}), json.dumps({"id": "SECRETID77", "text": "z"})], "dup.jsonl")
            with self.assertRaises(SystemExit) as cm:
                br.read_sentences(inp)
            self.assertEqual(str(cm.exception), "input line 4: id is not unique")
            se = io.StringIO()
            with mock.patch("sys.stderr", se):
                self.assertEqual(br.main([inp, os.path.join(d, "o.jsonl")]), 2)
            self.assertEqual(se.getvalue().strip(), "blind_runner: input line 4: id is not unique")
            clash = _write(d, ["plain one", json.dumps({"id": "s0001", "text": "x"})], "clash.jsonl")        # a manual id that collides with an automatic one
            with self.assertRaises(SystemExit) as cm:
                br.read_sentences(clash)
            self.assertEqual(str(cm.exception), "input line 2: id is not unique")

    def test_duplicate_ids_and_bad_json_are_refused_without_echoing_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            dup = _write(d, [json.dumps({"id": "a", "text": "ZQXV7731"}), json.dumps({"id": "a", "text": "y"})], "d.jsonl")
            with self.assertRaises(SystemExit):
                br.read_sentences(dup)
            bad = _write(d, ['{"id": "a", "oops": "ZQXV7731"'], "b.jsonl")
            with self.assertRaises(SystemExit) as cm:
                br.read_sentences(bad)
            self.assertNotIn("ZQXV7731", str(cm.exception))

    def test_the_environment_and_logging_are_restored_after_a_run(self) -> None:
        before = (dict(os.environ), logging.root.manager.disable)
        with tempfile.TemporaryDirectory() as d:
            br.main([_write(d, SENTENCES), os.path.join(d, "o.jsonl")])
        self.assertEqual((dict(os.environ), logging.root.manager.disable), before)

    def test_the_seal_itself_logging_off_at_every_level_hook_unset_memory_on_the_null_device_and_all_restored(self) -> None:
        with mock.patch.dict(os.environ, {"SOPHON_LIVE_DEVICE_HOOK": "/bin/true", "SOPHON_LIVE_MEMORY_PATH": "/tmp/x-mem.json"}):
            with br.quiet_and_sealed(False):
                self.assertGreaterEqual(logging.root.manager.disable, logging.CRITICAL)
                self.assertNotIn("SOPHON_LIVE_DEVICE_HOOK", os.environ)
                self.assertEqual(os.environ["SOPHON_LIVE_MEMORY_PATH"], os.devnull)
            self.assertEqual((os.environ["SOPHON_LIVE_DEVICE_HOOK"], os.environ["SOPHON_LIVE_MEMORY_PATH"]), ("/bin/true", "/tmp/x-mem.json"))


class Privacy(_Base):
    """Sentence text may appear in the output file and nowhere else: stdout, stderr, log records, any other file, a device hook."""

    def _scan(self, root: str, skip: set[str]) -> list[str]:
        hits = []
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                p = os.path.abspath(os.path.join(dp, fn))
                if p in skip:
                    continue
                data = open(p, "rb").read().decode("utf-8", "ignore")
                if any(m in data for m in MARKERS):
                    hits.append(p)
        return hits

    def sealed_run(self, *, worker=None, jev=None, lines=None, expect_rc: int = 0, with_hook: bool = True) -> list[dict]:
        """One in-process run with every observer attached; asserts that no sentence text reached any of them. Returns the output rows."""
        records: list[logging.LogRecord] = []

        class H(logging.Handler):
            def emit(self, record):   # noqa: ANN001
                records.append(record)
        h = H(level=0)
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(h); root.setLevel(logging.DEBUG)
        mem_default = LIVE / "logs" / "assistant_memory.json"
        mem_before = mem_default.read_bytes() if mem_default.exists() else None
        patches = []
        if worker is not None:
            patches.append(mock.patch.object(br, "_fake_worker", worker))
        if jev is not None:
            async def fake_jev(route):   # noqa: ANN001, ARG001
                return jev
            patches.append(mock.patch.object(br, "_fake_jev", fake_jev))
        try:
            with tempfile.TemporaryDirectory() as d:
                out, inp = os.path.join(d, "out.jsonl"), _write(d, lines or SENTENCES)
                hook_marker = os.path.join(d, "hook_was_called")
                hook = os.path.join(d, "hook.sh")
                with open(hook, "w") as f:
                    f.write(f"#!/bin/sh\necho called > '{hook_marker}'\n")
                os.chmod(hook, 0o755)
                cwd = os.getcwd()
                os.chdir(d)
                so, se = io.StringIO(), io.StringIO()
                env = {"SOPHON_LIVE_DEVICE_HOOK": hook} if with_hook else {}
                try:
                    with mock.patch.dict(os.environ, env), mock.patch("sys.stdout", so), mock.patch("sys.stderr", se):
                        for p in patches:
                            p.start()
                        try:
                            self.assertEqual(br.main([inp, out]), expect_rc)
                        finally:
                            for p in patches:
                                p.stop()
                finally:
                    os.chdir(cwd)
                blob = so.getvalue() + se.getvalue()
                self.assertFalse([m for m in MARKERS if m in blob], blob)
                self.assertEqual(records, [], [r.getMessage() for r in records][:3])
                self.assertFalse(os.path.exists(hook_marker), "the device hook ran during the blind run")
                self.assertEqual(self._scan(d, skip={os.path.abspath(inp), os.path.abspath(out)}), [])
                rows = _rows(out)
        finally:
            root.removeHandler(h); root.setLevel(old_level)
        self.assertEqual(mem_default.read_bytes() if mem_default.exists() else None, mem_before)
        return rows

    def test_in_process_nothing_leaks_to_stdout_stderr_log_records_files_or_the_device_hook(self) -> None:
        rows = self.sealed_run()
        self.assertEqual(len(rows), len(SENTENCES))

    def test_a_worker_that_crashes_with_the_sentence_in_the_message_leaks_nothing_either(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError(f"boom {job.objective}")
        rows = self.sealed_run(worker=bad_worker)
        self.assertTrue(all(r["error"] == "ValueError" for r in rows))

    def test_a_jev_that_crashes_with_the_sentence_in_the_message_leaks_nothing_either(self) -> None:   # admit swallows it into a log line
        async def jev(state, questions, **kw):   # noqa: ANN001
            raise RuntimeError(f"jev HTTP 500: {state['utterance']}")
        rows = self.sealed_run(jev=jev)
        self.assertTrue(all(r["via"] == "heuristic" and r["degraded_reason"] == "runtime_error" and r["degraded_exc"] == "RuntimeError" for r in rows))

    def test_error_and_critical_level_log_calls_with_the_sentence_are_swallowed(self) -> None:
        real_fake = br._fake_worker
        async def noisy_worker(job, **kw):   # noqa: ANN001
            lg = logging.getLogger("sophon_live.anything")
            lg.error("ERROR %s", job.objective); lg.critical("CRITICAL %s", job.objective); logging.getLogger().error("root %s", job.objective)
            return await real_fake(job)
        rows = self.sealed_run(worker=noisy_worker)
        self.assertEqual([r["error"] for r in rows], [None] * len(SENTENCES))

    def test_the_keyword_function_wrappers_keep_arguments_out_of_every_observer(self) -> None:
        from sophon_live.coordinator import memory_gate
        real_fake = br._fake_worker
        async def w(job, **kw):   # noqa: ANN001
            memory_gate.has_explicit_save_intent(job.objective)
            return await real_fake(job)
        rows = self.sealed_run(worker=w)
        self.assertEqual([r["error"] for r in rows], [None] * len(SENTENCES))
        self.assertTrue(all(r["memory_gate_explicit"] in ([True], [False]) for r in rows))
        self.assertFalse([m for m in MARKERS if m in json.dumps(rows, ensure_ascii=False)])

    def test_a_real_device_call_inside_the_run_does_not_reach_a_configured_hook(self) -> None:
        from sophon_live.assistant_tools import device
        real_fake = br._fake_worker
        async def device_worker(job, **kw):   # noqa: ANN001
            await device.run({"device": "light", "action": "on"})
            return await real_fake(job)
        rows = self.sealed_run(worker=device_worker)
        self.assertEqual([r["error"] for r in rows], [None] * len(SENTENCES))

    def test_the_hook_check_is_not_vacuous_the_hook_does_run_when_the_environment_is_not_sealed(self) -> None:
        from sophon_live.assistant_tools import device
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "m")
            hook = os.path.join(d, "hook.sh")
            open(hook, "w").write(f"#!/bin/sh\necho called > '{marker}'\n"); os.chmod(hook, 0o755)
            with mock.patch.dict(os.environ, {"SOPHON_LIVE_DEVICE_HOOK": hook}):
                asyncio.run(device.run({"device": "light", "action": "on"}))
            self.assertTrue(os.path.exists(marker))

    def test_the_check_is_not_vacuous_the_system_does_log_sentence_text_when_logging_is_on(self) -> None:
        """Without the runner's seal the same path DOES put the sentence in log records, so the tests above really protect something."""
        records: list[str] = []

        class H(logging.Handler):
            def emit(self, record):   # noqa: ANN001
                records.append(record.getMessage())
        h = H(level=0)
        root = logging.getLogger()
        old = root.level
        root.addHandler(h); root.setLevel(logging.DEBUG)
        try:
            from sophon_live import coordinator as C
            from sophon_live.coordinator import api
            use = api.USE_JEV
            api.USE_JEV = True
            try:
                from sophon_live.coordinator import jev_client
                orig = jev_client.system_one
                jev_client.system_one = asyncio.run(br._fake_jev("WORKER"))
                try:
                    asyncio.run(C.admit(C.AdmitRequest(session_id="s", utterance_id="u", text=SENTENCES[0], delegate_task=SENTENCES[0]), C.Ledger()))
                finally:
                    jev_client.system_one = orig
            finally:
                api.USE_JEV = use
        finally:
            root.removeHandler(h); root.setLevel(old)
        self.assertTrue(any("ZQXV7731" in m for m in records), records)

    def test_subprocess_run_leaks_nothing_to_stdout_stderr_or_any_other_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            inp = _write(d, SENTENCES)
            # cwd, TMPDIR and HOME are all inside the scanned directory, so a stray file written to any of them is found
            env = dict(os.environ, PYTHONPATH=str(LIVE), PYTHONDONTWRITEBYTECODE="1", SOPHON_LIVE_MEMORY_PATH=os.path.join(d, "mem.json"),
                       PYTHONWARNINGS="default", SOPHON_LIVE_DEVICE_HOOK=os.path.join(d, "no-such-hook"), TMPDIR=d, HOME=d)
            p = subprocess.run([sys.executable, "-m", "tools.blind_runner", inp, out], cwd=d, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            self.assertFalse([m for m in MARKERS if m in p.stdout + p.stderr], p.stdout + p.stderr)
            # the input file holds the sentences by definition; everything else in the tmp dir must be clean of them
            self.assertEqual(self._scan(d, skip={os.path.abspath(inp), os.path.abspath(out)}), [])
            self.assertFalse(os.path.exists(os.path.join(d, "mem.json")))        # the memory store was redirected to the null device

    def test_a_refused_output_prints_no_sentence_and_no_path_contents(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, SENTENCES)
            so, se = io.StringIO(), io.StringIO()
            with mock.patch("sys.stdout", so), mock.patch("sys.stderr", se):
                self.assertEqual(br.main([inp, inp]), 5)
            self.assertFalse([m for m in MARKERS if m in so.getvalue() + se.getvalue()])

    def test_a_memory_write_goes_to_the_null_device_not_to_the_configured_store(self) -> None:
        from sophon_live.assistant_tools import memory
        with tempfile.TemporaryDirectory() as d:
            mine = os.path.join(d, "m.json")
            with mock.patch.dict(os.environ, {"SOPHON_LIVE_MEMORY_PATH": mine}):
                with br.quiet_and_sealed(False):
                    self.assertEqual(str(memory._memory_path()), os.devnull)
                self.assertEqual(str(memory._memory_path()), mine)               # restored


class OpaqueSentence(unittest.TestCase):
    """I-1'': the runner carries each sentence as an Opaque object; every way of looking inside it raises TypeError (C-level paths included)."""

    def test_every_way_of_looking_inside_raises_type_error(self) -> None:
        import copy
        import operator
        import pickle
        o = br.Opaque("qzxw 天气 SECRETOPAQUE")
        other = br.Opaque("x")
        probes = {
            "re.search": lambda: re.search("qzxw", o), "re.match": lambda: re.match("q", o), "re.compile.search": lambda: re.compile("q").search(o),
            "json.dumps": lambda: json.dumps(o), "json.dumps(list)": lambda: json.dumps([o]), "bytes()": lambda: bytes(o), "bytes(o, enc)": lambda: bytes(o, "utf-8"),
            "str()": lambda: str(o), "format": lambda: format(o), "f-string": lambda: f"{o}", "%": lambda: "%s" % o, "str.format": lambda: "{}".format(o),
            "unbound str.find": lambda: str.find(o, "x"), "unbound str.startswith": lambda: str.startswith(o, "q"), "unbound str.__contains__": lambda: str.__contains__(o, "q"),
            "in": lambda: "q" in o, "operator.contains": lambda: operator.contains(o, "q"), "iter": lambda: iter(o), "for": lambda: [c for c in o], "list()": lambda: list(o),
            "len": lambda: len(o), "bool": lambda: bool(o), "not": lambda: not o, "if": lambda: 1 if o else 2, "index": lambda: o[0], "slice": lambda: o[:3],
            "==": lambda: o == "x", "== opaque": lambda: o == other, "!=": lambda: o != "x", "<": lambda: o < "x", ">=": lambda: o >= other, "hash": lambda: hash(o),
            "dict key": lambda: {o: 1}, "set": lambda: {o}, "list.index": lambda: [o].index(other), "x in list": lambda: "x" in [o], "+": lambda: o + "x", "radd": lambda: "x" + o,
            "*": lambda: o * 2, "int": lambda: int(o), "float": lambda: float(o), "operator.index": lambda: operator.index(o), "os.fspath": lambda: os.fspath(o),
            "copy": lambda: copy.copy(o), "deepcopy": lambda: copy.deepcopy(o), "pickle": lambda: pickle.dumps(o), "sorted": lambda: sorted([o, other]),
            "max": lambda: max(o, other), "str.join": lambda: "".join([o]), "sep.join": lambda: ",".join(o), "print": lambda: print(o, file=io.StringIO()),
            "open": lambda: open(o), "encode": lambda: o.encode(), "find": lambda: o.find("q"), "startswith": lambda: o.startswith("q"), "split": lambda: o.split(),
            "__dict__": lambda: o.__dict__, "vars": lambda: vars(o), "setattr": lambda: setattr(o, "extra", 1),
        }
        for name, fn in probes.items():
            with self.assertRaises((TypeError, AttributeError), msg=name):
                fn()
        for name, fn in probes.items():                                       # not merely "raises something": the sentence is never in the error text either
            try:
                fn()
            except (TypeError, AttributeError, ValueError) as e:
                self.assertNotIn("SECRETOPAQUE", str(e), name)

    def test_repr_is_neutral_and_reveal_is_the_one_way_out_and_only_for_an_opaque(self) -> None:
        o = br.Opaque("SECRETOPAQUE abc")
        self.assertEqual(repr(o), "<Opaque sentence>")
        self.assertEqual(repr(br.Opaque("x")), repr(br.Opaque("a much longer sentence")))          # not even the length
        self.assertEqual(br.reveal(o), "SECRETOPAQUE abc")
        for bad in ("a plain str", None, 3, ["x"]):
            with self.assertRaises(TypeError):
                br.reveal(bad)  # type: ignore[arg-type]

    def test_the_reader_wraps_every_sentence_and_the_runner_hands_production_a_real_str(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, ["plain", json.dumps({"id": "j", "text": "json one", "task": "a rewrite"})])
            sents = br.read_sentences(inp)
        self.assertTrue(all(type(s.text) is br.Opaque for s in sents))
        self.assertTrue(type(sents[1].task) is br.Opaque and sents[0].task is None)
        got: list = []
        from sophon_live.coordinator import api
        real_admit = api.admit
        async def spy(req, ledger):   # noqa: ANN001
            got.append((type(req.text), type(req.delegate_task), type(req.raw_text)))
            return await real_admit(req, ledger)
        with mock.patch.object(api, "admit", spy), mock.patch("sophon_live.coordinator.admit", spy):
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(br.main([_write(d, ["plain"]), os.path.join(d, "o.jsonl")]), 0)
        self.assertEqual(got, [(str, str, str)])


class RunnerHoldsNoJudgment(unittest.TestCase):
    """A self-check guard, NOT a security boundary: it stops the obvious ways of putting a rule about the sentence into the runner."""
    SRC = (LIVE / "tools" / "blind_runner.py").read_text(encoding="utf-8")
    TREE = ast.parse(SRC)
    BANNED_MODULES = {"re", "regex", "fnmatch", "difflib", "importlib", "unicodedata"}
    SENTENCE_NAMES = {"text", "line", "sentence", "utterance", "task", "delegate", "raw_text", "s"}

    def test_no_regex_like_imports_in_any_spelling(self) -> None:
        for n in ast.walk(self.TREE):
            if isinstance(n, ast.Import):
                self.assertFalse({a.name.split(".")[0] for a in n.names} & self.BANNED_MODULES, ast.dump(n))
            if isinstance(n, ast.ImportFrom):
                self.assertNotIn((n.module or "").split(".")[0], self.BANNED_MODULES, ast.dump(n))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.assertNotIn(n.func.id, {"__import__", "eval", "exec", "compile"})

    def test_no_cjk_string_literal_outside_the_docstring(self) -> None:
        doc = ast.get_docstring(self.TREE) or ""
        cjk = [n.value for n in ast.walk(self.TREE) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc
               and re.search(r"[\u4e00-\u9fff]", n.value)]
        self.assertEqual(cjk, [])

    def test_the_sentence_is_only_passed_on_never_compared_searched_or_sliced(self) -> None:
        """Outside the two readers (which only tell a JSON line from a plain line and validate types) no sentence variable may be
        compared, used with ``in``, indexed, or have a method called on it."""
        for fn in [n for n in ast.walk(self.TREE) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in ("read_sentences", "read_history")]:
            for n in ast.walk(fn):
                if isinstance(n, ast.Compare) and not all(isinstance(o, (ast.Is, ast.IsNot)) for o in n.ops):       # ``x is None`` is not a judgment
                    names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
                    self.assertFalse(names & self.SENTENCE_NAMES, f"{fn.name}: comparison on a sentence variable")
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in self.SENTENCE_NAMES:
                    self.assertIn((fn.name, n.attr), {("run", "id"), ("run", "text"), ("run", "task")}, f"{fn.name}: method/attribute use on a sentence variable")
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                    self.assertNotIn(n.value.id, self.SENTENCE_NAMES, f"{fn.name}: indexing a sentence variable")

    def test_reveal_is_called_from_exactly_one_function_and_nothing_reaches_into_the_wrapper(self) -> None:
        tree = self.TREE
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        def inside(node: ast.AST, fn: ast.AST) -> bool:
            return any(x is node for x in ast.walk(fn))
        opaque_init = [n for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "Opaque" for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"][0]
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "reveal"]
        self.assertTrue(calls)
        for c in calls:
            self.assertTrue(inside(c, funcs["_admit_request"]), f"reveal() called outside _admit_request at line {c.lineno}")
        names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "reveal"]
        self.assertEqual(len(names), len(calls), "reveal is referenced other than by a direct call (an alias)")
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == "_v":
                self.assertTrue(inside(n, funcs["reveal"]) or inside(n, opaque_init), f"the wrapper's slot is touched at line {n.lineno}")
        for n in ast.walk(tree):                      # reflection on a sentence (the patching code's getattr(module, name) is fine)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "setattr", "delattr", "vars", "object", "type"):
                for arg in n.args:
                    for x in ast.walk(arg):
                        self.assertFalse(isinstance(x, ast.Name) and x.id in self.SENTENCE_NAMES, f"reflection on a sentence variable at line {n.lineno}")
                        self.assertFalse(isinstance(x, ast.Attribute) and x.attr in ("text", "task"), f"reflection on a sentence field at line {n.lineno}")
        self.assertEqual(len([n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "_v"]), 1)         # only Opaque's ``__slots__ = ("_v",)``

    def test_the_guard_is_not_vacuous_it_sees_a_planted_comparison_and_a_from_import(self) -> None:
        planted = ast.parse("from re import search\ndef f(text):\n    return 'x' in text\n")
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "re" for n in ast.walk(planted)))
        self.assertTrue(any(isinstance(n, ast.Compare) and {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & self.SENTENCE_NAMES for n in ast.walk(planted)))


if __name__ == "__main__":
    unittest.main()
