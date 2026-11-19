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


def peek(o) -> str:   # noqa: ANN001   test code looks inside an Opaque directly (reveal() serves only the runner's fixed exits)
    return o._v


def _opq_hist(hist: list[dict]) -> list[dict]:
    return [{"task": br.Opaque(h["task"]), "answer": br.Opaque(h["answer"])} for h in hist]


def _plain_hist(hist: list[dict]) -> list[dict]:
    return [{"task": peek(h["task"]), "answer": peek(h["answer"])} for h in hist]


def _write(d: str, lines: list[str], name: str = "in.txt") -> str:
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def _rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _jev_answering(route: str | None, control: str | None = None, *, raw: dict | None = None, tool: str | None = None, **others):
    """A scripted JEV: it answers by question KEY (route / control / tool / any other key given as a keyword), never by reading the state."""
    async def system_one(state, questions, **kw):   # noqa: ANN001
        if raw is not None:
            return raw
        ans: dict = {}
        for key, val in (("route", route), ("control", control), ("tool", tool), *others.items()):
            if val is not None and key in questions:
                ans[key] = {"type": "choice", "choice": val}
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
                if kw.get("history") is not None:
                    kw["history"] = _opq_hist(kw["history"])
                row = asyncio.run(br.run_one(kw.pop("sid", "x"), br.Opaque(text), ledger=ledger or C.Ledger(), jev=jev, worker=worker or br._fake_worker,
                                             task=br.Opaque(task) if task is not None else None, **kw))
                row["reply"] = peek(row["reply"])                  # the row as written to the file (br._dump_row), for the assertions
                return row
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
        keys = ("id", "mode", "fidelity", "route", "decision", "jev_route", "jev_control", "jev_tool", "via", "control_via", "degraded",
                "degraded_reason", "degraded_exc", "judge_invalid", "handled_by", "required_tools", "tool_judgment", "explicit_save",
                "tools", "tool_ok", "worker_status", "outcome_kind", "outcome_rejected", "reply", "error", "error_stage")
        for r in rows:
            self.assertEqual(sorted(r), sorted(keys))
            self.assertEqual((r["mode"], r["route"], r["via"], r["degraded"], r["jev_route"], r["error"]), ("fake", "WORKER", "jev", False, "WORKER", None))
            self.assertEqual((r["reply"], r["outcome_kind"], r["outcome_rejected"]), (br.FAKE_REPLY, "answer", None))
            # B-R6: nothing was really called in fake mode, so the columns that say "what was called" are null, not a copy of required_tools
            self.assertIsNone(r["tools"]); self.assertIsNone(r["tool_ok"])
            # the fake JEV answers the route question only: the tool question has no usable answer, production counts it and records the judgment as unknown
            self.assertEqual((r["required_tools"], r["tool_judgment"], r["jev_tool"], r["judge_invalid"]), ([], "unknown", None, ["tool:missing"]))
            self.assertIn("fake_worker", r["fidelity"]); self.assertIn("fake_jev", r["fidelity"])

    def test_the_model_s_tool_answer_is_recorded_not_derived(self) -> None:
        row = self.run_one("今天天气", jev=_jev_answering("WORKER", tool="get_weather"))
        self.assertEqual((row["jev_tool"], row["required_tools"], row["tool_judgment"], row["judge_invalid"], row["explicit_save"]), ("get_weather", ["get_weather"], "model", [], False))
        row = self.run_one("今天天气", jev=_jev_answering("WORKER", tool="NONE"))
        self.assertEqual((row["required_tools"], row["tool_judgment"]), ([], "model"))
        row = self.run_one("今天天气", jev=_jev_answering("WORKER", tool="not an option"))               # never guessed: unknown + counted
        self.assertEqual((row["jev_tool"], row["required_tools"], row["tool_judgment"], row["judge_invalid"]), ("not an option", [], "unknown", ["tool:invalid"]))
        row = self.run_one("记一下", jev=_jev_answering("WORKER", tool="memory_save"))
        self.assertEqual((row["required_tools"], row["explicit_save"]), (["memory_note"], True))

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
    def test_a_jev_failure_is_recorded_as_degraded_with_production_s_own_reason(self) -> None:
        """The reason is production's (``AdmitResult.degraded_reason``: ModelUnavailable carries it, any other exception is ``error``); the runner
        derives nothing from the exception, it records the CLASS NAME only."""
        from sophon_live.model_client import ModelUnavailable
        cases = ((ModelUnavailable("jev", "timeout"), "timeout"), (ModelUnavailable("jev", "connection"), "connection"),
                 (ModelUnavailable("jev", "breaker_open"), "breaker_open"), (ModelUnavailable("jev", "error"), "error"),
                 (RuntimeError("jev HTTP 500: body"), "error"), (TimeoutError(), "error"), (KeyError("k"), "error"))
        for exc, reason in cases:
            row = self.run_one("现在几点了", jev=_jev_raising(exc))
            self.assertEqual((row["via"], row["degraded"], row["degraded_reason"], row["degraded_exc"], row["jev_route"], row["error"]),
                             ("degraded", True, reason, type(exc).__name__, None, None), exc)

    def test_the_degraded_reason_does_not_depend_on_the_message(self) -> None:
        a = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("timeout")))
        b = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("jev bad json: x")))
        self.assertEqual((a["degraded_reason"], b["degraded_reason"]), ("error", "error"))

    def test_the_jev_error_message_is_never_read_or_recorded(self) -> None:
        row = self.run_one("现在几点了", jev=_jev_raising(RuntimeError("echo ZQXV7731")))
        self.assertNotIn("ZQXV7731", json.dumps(row, ensure_ascii=False))

    def test_a_jev_answer_without_a_usable_route_is_not_recorded_as_a_jev_decision(self) -> None:
        row = self.run_one("现在几点了", jev=_jev_answering(None))
        self.assertEqual((row["via"], row["jev_route"], row["degraded"], row["degraded_reason"]), ("degraded", None, True, "non_option_answer"))
        row = self.run_one("现在几点了", jev=_jev_answering(None, raw={"answers": {}}))
        self.assertEqual((row["via"], row["degraded"]), ("degraded", True))

    def test_a_jev_answer_that_is_not_a_route_enum_is_invalid_not_a_jev_decision(self) -> None:
        row = self.run_one("好的谢谢", jev=_jev_answering("BOGUS"))
        self.assertEqual((row["via"], row["jev_route"], row["route"], row["degraded"], row["degraded_reason"]), ("degraded", "BOGUS", "WORKER", True, "non_option_answer"))

    def test_a_valid_jev_route_is_a_jev_decision_and_not_degraded(self) -> None:
        for r in ("WORKER", "FAST", "CLARIFY", "STOP_SPEAKING"):
            row = self.run_one("今天天气", jev=_jev_answering(r))
            self.assertEqual((row["via"], row["degraded"], row["degraded_reason"], row["control_via"], row["jev_route"]), ("jev", False, None, None, r), r)

    def _ledger_with_a_finished_task(self):
        from sophon_live import coordinator as C
        ledger = C.Ledger()
        first = self.run_one("明天上海天气怎么样", jev=_jev_answering("WORKER"), ledger=ledger)
        self.assertEqual(first["worker_status"], "completed")
        return ledger

    def test_control_without_a_target_with_a_live_task_is_still_not_guessed(self) -> None:
        ledger = self._ledger_with_a_finished_task()
        row = self.run_one("取消吧", jev=_jev_answering("CONTROL"), ledger=ledger)
        self.assertEqual((row["route"], row["control_via"], row["via"], row["degraded"], row["handled_by"]), ("CLARIFY", None, "jev", False, "clarify"))
        self.assertNotEqual(row["decision"], "CANCEL")

    def test_control_without_a_usable_target_is_a_fixed_clarify_decided_by_the_model_s_route(self) -> None:
        row = self.run_one("嗯", jev=_jev_answering("CONTROL"))                    # no target, no ledger task: production asks (fixed sentence), it does not guess
        self.assertEqual((row["route"], row["decision"], row["control_via"], row["via"], row["degraded"], row["handled_by"]), ("CLARIFY", "CLARIFY", None, "jev", False, "clarify"))

    def test_control_via_is_the_applied_decisions_own_label_not_a_constant(self) -> None:
        """The runner records production's label for the applied ControlDecision verbatim. Production now only builds ``via="jev"`` controls, so a stand-in
        with another label proves the runner passes it through rather than writing "jev" itself (mutant R1d)."""
        from types import SimpleNamespace as NS
        self.assertEqual(br._control_via(NS(control=NS(via="a-label-production-may-add"))), "a-label-production-may-add")
        self.assertEqual(br._control_via(NS(control=NS(via="jev"))), "jev")
        self.assertIsNone(br._control_via(NS(control=None)))

    def test_control_with_a_jev_named_target_or_new_is_a_jev_decision(self) -> None:
        ledger = self._ledger_with_a_finished_task()
        latest = ledger.latest()
        row = self.run_one("进度", jev=_jev_answering("CONTROL", f"STATUS:{latest.task_id}:{latest.version}"), ledger=ledger)
        self.assertEqual((row["decision"], row["control_via"], row["degraded"], row["jev_control"]), ("STATUS", "jev", False, f"STATUS:{latest.task_id}:{latest.version}"))
        # NEW, or a task the Ledger does not have: a new job. No ControlDecision is applied, so control_via is null (production labels only an applied
        # decision; before rc3 the runner derived "jev" here from the heuristic fall-through, which no longer exists)
        row = self.run_one("另一件事", jev=_jev_answering("CONTROL", "NEW"), ledger=ledger)
        self.assertEqual((row["decision"], row["control_via"], row["degraded"], row["via"], row["jev_control"]), ("NEW", None, False, "jev", "NEW"))
        row = self.run_one("另一件事", jev=_jev_answering("CONTROL", "CANCEL:nope:1"), ledger=ledger)
        self.assertEqual((row["decision"], row["control_via"], row["degraded"], row["via"]), ("NEW", None, False, "jev"))

    def test_there_is_no_guard_any_more_the_model_is_always_asked_and_a_refine_is_its_call(self) -> None:
        """rc3 removed ``slot_edit`` (a keyword guard that answered before the JEV): the JEV is asked even when a task is live, and a REFINE option it
        names is applied by the Coordinator (code only checks the task exists and the version is current)."""
        ledger = self._ledger_with_a_finished_task()
        latest = ledger.latest()
        asked: list = []
        async def jev(state, questions, **kw):   # noqa: ANN001
            asked.append(sorted(questions))
            return {"answers": {"route": {"type": "choice", "choice": "WORKER"}, "control": {"type": "choice", "choice": f"REFINE:{latest.task_id}:{latest.version}"}}}
        row = self.run_one("明天北京天气怎么样", jev=jev, ledger=ledger)
        self.assertTrue(asked)
        self.assertEqual((row["via"], row["jev_route"], row["decision"], row["control_via"], row["degraded"], row["error"]), ("jev", "WORKER", "REFINE", "jev", False, None))

    def test_jev_switched_off_is_not_degraded_and_says_so(self) -> None:
        row = self.run_one("现在几点了", jev=_jev_answering("FAST"), use_jev=False)
        self.assertEqual((row["via"], row["degraded"], row["degraded_reason"]), ("degraded", False, None))      # production: via degraded / reason jev_disabled; not an outage

    def test_an_error_keeps_what_was_already_recorded(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("secret ZQXV7731 text")
        row = self.run_one("现在几点了", jev=_jev_answering("WORKER", tool="get_datetime"), worker=bad_worker)
        self.assertEqual((row["error"], row["error_stage"], row["route"], row["jev_route"], row["via"], row["handled_by"], row["degraded"], row["required_tools"]),
                         ("ValueError", "worker", "WORKER", "WORKER", "jev", "worker", False, ["get_datetime"]))
        self.assertNotIn("ZQXV7731", json.dumps(row, ensure_ascii=False))
        from sophon_live.model_client import ModelUnavailable
        row = self.run_one("现在几点了", jev=_jev_raising(ModelUnavailable("jev", "timeout")), worker=bad_worker)             # the degraded reason survives the later crash
        self.assertEqual((row["error"], row["via"], row["degraded"], row["degraded_reason"]), ("ValueError", "degraded", True, "timeout"))

    def test_a_crash_in_the_run_is_recorded_as_a_class_name_only(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("secret ZQXV7731 text")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            with mock.patch.object(br, "_fake_worker", bad_worker):
                br.main([_write(d, [SENTENCES[0]]), out])
            (r,) = _rows(out)
            self.assertEqual((r["id"], r["error"], r["error_stage"], r["route"], r["jev_route"], r["via"]), ("s0001", "ValueError", "worker", "WORKER", "WORKER", "jev"))
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
        """Was: ``slot_edit`` received (rewritten task, raw words). ``slot_edit`` is gone; the same fact is now what the JEV is shown."""
        seen: list = []
        async def jev(state, questions, **kw):   # noqa: ANN001
            seen.append((state.get("user_said"), state.get("utterance")))
            return {"answers": {"route": {"type": "choice", "choice": "WORKER"}, "tool": {"type": "choice", "choice": "NONE"}}}
        row = self.run_one("原话ZQXV", jev=jev, task="改写后的任务")
        self.assertEqual(seen[0], ("原话ZQXV", "改写后的任务"))      # the JEV is shown the user's own words next to the front's rewrite
        self.assertEqual(row["judge_invalid"], [])

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
            return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer="北京今天20度", evidence=[], freshness_token=job.freshness_token)
        row = self.run_one("今天北京天气怎么样", jev=_jev_answering("WORKER", tool="get_weather"), worker=w)           # the model said weather, no tool was called
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
        sink = _opq_hist(hist)
        from sophon_live import coordinator as C
        async def go():
            for k in range(3):
                await br.run_one(f"s{k}", br.Opaque(f"句{k}"), ledger=C.Ledger(), jev=_jev_answering("WORKER"), worker=br._fake_worker, history=sink, history_sink=sink)
        with br.quiet_and_sealed(False):
            asyncio.run(go())
        self.assertEqual(len(sink), 12)
        self.assertEqual(peek(sink[-1]["answer"]), br.FAKE_REPLY)

    # ---- I-2: the keyword functions' conclusions are recorded, nothing else, and everything is restored -----------------------------
    def test_the_jev_is_restored_after_every_run(self) -> None:
        from sophon_live.coordinator import api, jev_client
        before = jev_client.system_one
        row = self.run_one("今天比特币多少钱", jev=_jev_answering("WORKER", tool="NONE"))
        self.assertEqual((row["jev_tool"], row["required_tools"], row["tool_judgment"]), ("NONE", [], "model"))
        self.assertIs(jev_client.system_one, before)

    def test_the_jev_is_restored_even_when_the_run_crashes(self) -> None:
        from sophon_live.coordinator import jev_client
        before = jev_client.system_one
        async def bad(job, **kw):   # noqa: ANN001
            raise ValueError("x")
        row = self.run_one("今天", jev=_jev_answering("WORKER"), worker=bad)
        self.assertEqual(row["error"], "ValueError")
        self.assertIs(jev_client.system_one, before)

    def test_the_spy_passes_the_request_and_the_answer_through_unchanged(self) -> None:
        seen: list = []
        async def jev(state, questions, **kw):   # noqa: ANN001
            seen.append((dict(state), sorted(questions), dict(kw)))
            return {"answers": {"route": {"type": "choice", "choice": "FAST"}}}
        row = self.run_one("x", jev=jev, task="任务")
        self.assertEqual(row["jev_route"], "FAST")
        (state, keys, kw), = seen
        self.assertEqual((state["user_said"], "route" in keys, "tool" in keys, kw), ("x", True, True, {}))

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
        row that equals the row of a neutral sentence in EVERY column except ``id``. (Before rc3 ``required_tools`` and ``clarify_slots`` were exempt:
        they carried the answers of production's keyword functions. Those functions are gone: with a constant fake JEV no column may depend on the
        sentence, not even the tool judgment.) A fork on a multi-character word the corpus does not contain is NOT found; code review remains the backstop."""
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
                for key in set(r) - {"id"}:
                    self.assertEqual(r[key], base[key], f"{route} {key} differs for {sent!r}")

    def test_the_metamorphic_test_can_see_a_character_fork(self) -> None:
        """Not vacuous: a runner that forks on one character gives a different row for exactly that character."""
        orig_run_one = br.run_one
        async def run_one(sid, text, **kw):   # noqa: ANN001
            row = await orig_run_one(sid, text, **kw)
            if "x" in peek(text):                                 # a planted judgment (in the TEST, never in the runner)
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
            self.assertEqual([(s.id, peek(s.text)) for s in br.read_sentences(bom)], [("b", "ZQXV7731")])           # parsed as JSON, not taken as a sentence
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
            raise RuntimeError(f"jev HTTP 500: {state}")
        rows = self.sealed_run(jev=jev)
        self.assertTrue(all(r["via"] == "degraded" and r["degraded_reason"] == "error" and r["degraded_exc"] == "RuntimeError" for r in rows))

    def test_error_and_critical_level_log_calls_with_the_sentence_are_swallowed(self) -> None:
        real_fake = br._fake_worker
        async def noisy_worker(job, **kw):   # noqa: ANN001
            lg = logging.getLogger("sophon_live.anything")
            lg.error("ERROR %s", job.objective); lg.critical("CRITICAL %s", job.objective); logging.getLogger().error("root %s", job.objective)
            return await real_fake(job)
        rows = self.sealed_run(worker=noisy_worker)
        self.assertEqual([r["error"] for r in rows], [None] * len(SENTENCES))

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
        for bad in ("a plain str", None, 3, ["x"]):
            with self.assertRaises(TypeError):
                br.reveal(bad)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):                              # an Opaque, but a caller that is not one of the fixed exits
            br.reveal(o)

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


# ---------------------------------------------------------------------------------------------------------------------------------------
# Structure checks (AST whitelists). They are SELF-CHECKS against accidents, not a security boundary: each one is pinned by mutants in
# tests/mutation_check_blind_runner.py, and each is shown not to be vacuous on a planted violation below.
# ---------------------------------------------------------------------------------------------------------------------------------------
def _parents(tree: ast.AST) -> dict:
    pm: dict = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            pm[c] = n
    return pm


def _funcs(tree: ast.AST) -> dict:
    out: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, n)
    return out


def _body_nodes(fn: ast.AST) -> list:
    """All nodes of a function body, annotations excluded."""
    skip: set = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.AnnAssign):
            skip.update(ast.walk(n.annotation))
    return [n for stmt in fn.body for n in ast.walk(stmt) if n not in skip]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


_NONE_COMPARE_OPS = (ast.Is, ast.IsNot)


def _is_none_compare(n: ast.Compare) -> bool:
    return all(isinstance(o, _NONE_COMPARE_OPS) for o in n.ops) and all(isinstance(c, ast.Constant) and c.value is None for c in n.comparators)


EXIT_SPECS = {
    "_admit_request": {"calls": {"reveal"}, "attrs": {("C", "AdmitRequest")}},
    "_task_context": {"calls": {"reveal"}, "attrs": {("C", "TaskContext"), ("job", "task_id"), ("job", "version")}},
    "_worker_history": {"calls": {"reveal"}, "attrs": {("out", "append")}},
    "_dump_row": {"calls": {"reveal"}, "attrs": {("json", "dumps")}},
}
_EXIT_NODES = (ast.Expr, ast.ImportFrom, ast.alias, ast.Assign, ast.AnnAssign, ast.Return, ast.For, ast.Call, ast.keyword, ast.Attribute, ast.Name, ast.Constant,
               ast.Dict, ast.List, ast.Subscript, ast.Slice, ast.UnaryOp, ast.USub, ast.IfExp, ast.Compare, ast.Is, ast.IsNot, ast.Load, ast.Store)


def exit_violations(fn: ast.AST, spec: dict) -> list[str]:
    """An exit function may do ONE thing: ``reveal`` its Opaque arguments and build the one object it hands on (an AdmitRequest, a TaskContext, the
    Worker's history dicts, the JSON line). No other call, no comparison but ``is None``, no import but the coordinator, no name outside its own."""
    v: list[str] = []
    nodes = _body_nodes(fn)
    params = {a.arg for a in fn.args.args}
    local = {t.id for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign, ast.For)) for t in ast.walk(n.target if not isinstance(n, ast.Assign) else n.targets[0])
             if isinstance(t, ast.Name)}
    bases = {b for b, _ in spec["attrs"]}
    ok_names = params | local | spec["calls"] | bases | {"C", "json", "HISTORY_TURNS"}
    for n in nodes:
        if not isinstance(n, _EXIT_NODES):
            v.append(f"{fn.name}: node {type(n).__name__} is not allowed")
        if isinstance(n, ast.ImportFrom) and not (n.module == "sophon_live" and [(a.name, a.asname) for a in n.names] == [("coordinator", "C")]):
            v.append(f"{fn.name}: import")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id not in spec["calls"]:
                    v.append(f"{fn.name}: call {n.func.id}")
            elif not (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and (n.func.value.id, n.func.attr) in spec["attrs"]):
                v.append(f"{fn.name}: call of an attribute outside the whitelist")
        if isinstance(n, ast.Attribute):
            if not (isinstance(n.value, ast.Name) and (n.value.id, n.attr) in spec["attrs"]):
                v.append(f"{fn.name}: attribute {n.attr}")
        if isinstance(n, ast.Name) and n.id not in ok_names:
            v.append(f"{fn.name}: name {n.id}")
        if isinstance(n, ast.Compare) and not _is_none_compare(n):
            v.append(f"{fn.name}: comparison")
        if isinstance(n, ast.Subscript):
            sl = n.slice
            okslice = (isinstance(sl, ast.Constant) and isinstance(sl.value, str)) or (
                isinstance(sl, ast.Slice) and sl.upper is None and isinstance(sl.lower, ast.UnaryOp) and isinstance(sl.lower.operand, ast.Name)
                and sl.lower.operand.id == "HISTORY_TURNS")
            if not okslice:
                v.append(f"{fn.name}: subscript")
    return v


_SPY_NODES = (ast.Try, ast.ExceptHandler, ast.Await, ast.Call, ast.keyword, ast.Name, ast.Attribute, ast.Tuple, ast.Assign, ast.Raise, ast.Return, ast.Constant,
              ast.If, ast.Compare, ast.Eq, ast.Starred, ast.Load, ast.Store, ast.BoolOp, ast.Or, ast.BinOp, ast.Add, ast.List, ast.Expr)
_SPY_ATTRS = ({("rec", a) for a in ("route", "control", "tool", "error_exc")} | {("jev_client", "choice_answer")}
              | {("intent", "TOOL_KEY")})
_SPY_CALLS = {"jev", "type", "choice_answer"}


def watcher_violations(fn: ast.AST, tree: ast.AST) -> list[str]:
    """The spy sees production's state (the JEV request and its answers). It may only forward its arguments to the JEV it wraps and keep the
    option keys of the answers: no comparison, no indexing, no other call, no use of ``state`` / ``questions`` except by being forwarded."""
    v: list[str] = []
    pm = _parents(tree)
    for n in _body_nodes(fn):
        if not isinstance(n, _SPY_NODES):
            v.append(f"{fn.name}: node {type(n).__name__} is not allowed")
        if isinstance(n, ast.Compare):
            v.append(f"{fn.name}: comparison")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id not in _SPY_CALLS:
                v.append(f"{fn.name}: call {n.func.id}")
            if isinstance(n.func, ast.Attribute) and not (isinstance(n.func.value, ast.Name) and (n.func.value.id, n.func.attr) in _SPY_ATTRS):
                v.append(f"{fn.name}: attribute call {n.func.attr}")
        if isinstance(n, ast.Attribute) and not ((isinstance(n.value, ast.Name) and (n.value.id, n.attr) in _SPY_ATTRS)
                                                  or (n.attr == "__name__" and isinstance(n.value, ast.Call) and _call_name(n.value) == "type")):
            v.append(f"{fn.name}: attribute {n.attr}")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in {"state", "questions", "kw", "resp", "e"}:
            p = pm[n]
            while isinstance(p, (ast.Starred, ast.keyword, ast.BoolOp)):
                p = pm[p]
            allowed = {"state": {"jev"}, "questions": {"jev"}, "kw": {"jev"}, "resp": {"choice_answer"}, "e": {"type"}}[n.id]
            if not ((isinstance(p, ast.Call) and _call_name(p) in allowed) or (n.id == "resp" and isinstance(p, ast.Return))):
                v.append(f"{fn.name}: {n.id} is used other than by being forwarded")
    return v


_PRODUCTION_TEXT_ATTRS = {"objective", "text", "spoken_answer"}
_PRODUCTION_OBJECTS = {"res", "job", "out", "wr", "speak"}
_PRODUCTION_PATHS = {"job.objective", "job.require_tools", "job.tool_judgment", "job.explicit_save",
                     "res.degraded_reason", "res.judge_invalid", "out.kind", "out.rejected", "out.text", "res.control", "res.control.op", "res.job", "res.route",
                     "res.route.lower", "res.speak.text", "res.via", "wr.evidence", "wr.status"}
_PRODUCTION_CALLEES = {"worker", "_task_context", "_control_via", "finish_job", "admit", "_admit_request"}


def production_text_violations(run_one: ast.AST, tree: ast.AST) -> list[str]:
    """What run_one reads from production: a text that may quote the sentence is wrapped in Opaque at the line it is read; production's objects
    are only handed on (or an attribute that is not a text is read)."""
    v: list[str] = []
    pm = _parents(tree)
    for n in _body_nodes(run_one):
        if isinstance(n, ast.Attribute) and n.attr in _PRODUCTION_TEXT_ATTRS:
            p = pm[n]
            if not (isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "Opaque" and p.args == [n] and not p.keywords):
                v.append(f"line {n.lineno}: production text .{n.attr} is read other than as Opaque(...)")
        if isinstance(n, ast.Attribute) and not isinstance(pm.get(n), ast.Attribute):          # the outermost link of an attribute chain
            parts, x = [], n
            while isinstance(x, ast.Attribute):
                parts.append(x.attr)
                x = x.value
            if isinstance(x, ast.Name) and x.id in _PRODUCTION_OBJECTS and ".".join([x.id] + parts[::-1]) not in _PRODUCTION_PATHS:
                v.append(f"line {n.lineno}: reads {'.'.join([x.id] + parts[::-1])}, which is not one of the audited production fields")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in _PRODUCTION_OBJECTS:
            p = pm[n]
            if not (isinstance(p, ast.Attribute) or (isinstance(p, ast.Call) and _call_name(p) in _PRODUCTION_CALLEES)):
                v.append(f"line {n.lineno}: {n.id} is used other than by attribute or being handed on")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in {"objective", "reply"}:
            p = pm[n]
            if not ((isinstance(p, ast.Call) and _call_name(p) == "_task_context") or isinstance(p, (ast.Assign, ast.Dict))):
                v.append(f"line {n.lineno}: {n.id} (an Opaque) is used other than by being stored or handed to an exit")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "setattr", "delattr"):
            if not all(isinstance(a, ast.Name) and a.id in {"o", "n", "f", "v"} for a in n.args):
                v.append(f"line {n.lineno}: reflection")
    return v


_ALLOWED_TOP_IMPORTS = {"__future__", "argparse", "asyncio", "contextlib", "json", "logging", "os", "sys", "warnings", "typing"}
_ALLOWED_DUNDER_ATTRS = {"__name__", "__cause__"}
_BANNED_NAMES = {"object", "globals", "locals", "vars", "__import__", "eval", "exec", "compile", "breakpoint", "memoryview", "bytearray", "bytes", "repr", "ascii"}
_BANNED_ATTRS = {"modules", "_getframe", "f_locals", "f_globals", "gi_frame", "cr_frame", "get_referents", "get_objects", "getmembers", "getsource", "__code__", "__dict__"}


def whole_file_violations(tree: ast.AST) -> list[str]:
    """Imports are a fixed list (no gc/inspect/ctypes/re/operator/functools/copy/pickle/io...); no name or attribute that takes a value out of a slot
    or reaches the module's own namespace; no ``global``; no module-level (or default-argument) mutable container to stash a sentence in."""
    v: list[str] = []
    fns = _funcs(tree)
    reveal_nodes = set(ast.walk(fns["reveal"]))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in _ALLOWED_TOP_IMPORTS | {"aiohttp"}:
                    v.append(f"line {n.lineno}: import {a.name}")
        if isinstance(n, ast.ImportFrom):
            mod = (n.module or "").split(".")[0]
            if mod not in _ALLOWED_TOP_IMPORTS | {"sophon_live"} or n.level:
                v.append(f"line {n.lineno}: from {n.module} import")
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            v.append(f"line {n.lineno}: global/nonlocal")
        if isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
            v.append(f"line {n.lineno}: name {n.id}")
        if isinstance(n, ast.Attribute):
            if n.attr in _BANNED_ATTRS and n not in reveal_nodes:
                v.append(f"line {n.lineno}: attribute {n.attr}")
            if n.attr.startswith("__") and n.attr.endswith("__") and n.attr not in _ALLOWED_DUNDER_ATTRS and n not in reveal_nodes:
                v.append(f"line {n.lineno}: dunder attribute {n.attr}")
            if isinstance(n.ctx, ast.Store) and not (isinstance(n.value, ast.Name) and (n.value.id in {"self", "rec"} or (n.value.id, n.attr) == ("api", "USE_JEV"))):
                v.append(f"line {n.lineno}: attribute store on something other than self/rec/api.USE_JEV")
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(n.args.defaults) + [d for d in n.args.kw_defaults if d is not None]:
                if not isinstance(d, (ast.Constant, ast.Tuple)):
                    v.append(f"line {n.lineno}: a default argument that is not a constant")
    mutable = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Call)
    scopes = [tree] + [c for c in tree.body if isinstance(c, ast.ClassDef)]
    for scope in scopes:
        for st in scope.body:
            if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None and isinstance(st.value, mutable):
                v.append(f"line {st.lineno}: a module-level or class-level mutable container / call")
    return v


_READ_NODES = (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.For, ast.If, ast.With, ast.withitem, ast.Try, ast.ExceptHandler, ast.Raise, ast.Continue,
               ast.Call, ast.keyword, ast.Attribute, ast.Name, ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.Subscript, ast.Slice, ast.UnaryOp, ast.Not,
               ast.BoolOp, ast.Or, ast.And, ast.IfExp, ast.Compare, ast.Is, ast.IsNot, ast.In, ast.BinOp, ast.Add, ast.JoinedStr, ast.FormattedValue,
               ast.Load, ast.Store)
_READ_CALLS_NAMES = {"isinstance", "str", "len", "enumerate", "open", "Opaque", "Sentence", "SystemExit", "TypeError", "set", "_utf8_or_exit"}
_READ_CALLS_ATTRS = {"strip", "startswith", "get", "loads", "add", "append", "read", "decode", "replace", "count"}
_CONTENT_NAMES = {"line", "text", "task", "o", "data"}
_CONTENT_METHODS = {"strip", "startswith", "get", "decode", "replace", "count"}


def _flows_into_content_names_only(node: ast.AST, pm: dict) -> bool:
    """True when ``node`` (a content value) is assigned only to content names (``line = line.strip()``, ``o = json.loads(line)``); an alias such as
    ``x = line`` is how a value would escape this check."""
    p = pm[node]
    while isinstance(p, ast.Tuple):
        p = pm[p]
    if not isinstance(p, ast.Assign):
        return False
    return all(isinstance(t, ast.Name) and t.id in _CONTENT_NAMES or isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) and x.id in _CONTENT_NAMES for x in t.elts)
               for t in p.targets)


def read_violations(fn: ast.AST, tree: ast.AST) -> list[str]:
    """The reading stage sees the raw line before it is wrapped. On it: ``strip()``, ``startswith("{")``, ``json.loads``, wrapping in Opaque, and for the
    parsed JSON only a constant-key lookup or ``get("...")``. No ``in``/comparison/indexing/f-string on content. ``sid in seen`` is the one membership test."""
    v: list[str] = []
    pm = _parents(tree)
    for n in _body_nodes(fn):
        if not isinstance(n, _READ_NODES):
            v.append(f"{fn.name}: node {type(n).__name__} is not allowed")
        if isinstance(n, ast.Compare):
            sid_in_seen = (isinstance(n.left, ast.Name) and n.left.id == "sid" and [type(o) for o in n.ops] == [ast.In]
                           and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == "seen")
            if not (_is_none_compare(n) or sid_in_seen):
                v.append(f"{fn.name}: comparison / membership test (line {n.lineno})")
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id not in _READ_CALLS_NAMES:
                v.append(f"{fn.name}: call {n.func.id}")
            if isinstance(n.func, ast.Attribute):
                if n.func.attr not in _READ_CALLS_ATTRS:
                    v.append(f"{fn.name}: method {n.func.attr}")
                if n.func.attr == "startswith" and not (len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and n.args[0].value == "{"):
                    v.append(f"{fn.name}: startswith with something other than the JSON marker")
                if n.func.attr in _CONTENT_METHODS and not all(isinstance(a, ast.Constant) for a in n.args):
                    v.append(f"{fn.name}: {n.func.attr} with a non-constant argument")
            if _call_name(n) in _CONTENT_METHODS | {"loads"}:                      # the RESULT of a content method is content too: where may it go?
                p = pm[n]
                ok = (isinstance(p, (ast.If, ast.IfExp, ast.BoolOp, ast.Expr, ast.Attribute, ast.BinOp))
                      or (isinstance(p, (ast.Assign, ast.Tuple)) and _flows_into_content_names_only(n, pm))
                      or (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not))
                      or (isinstance(p, ast.Compare) and _is_none_compare(p))
                      or (isinstance(p, ast.Call) and _call_name(p) in ("isinstance", "Opaque")))
                if not ok:
                    v.append(f"{fn.name}: the result of {_call_name(n)}() goes somewhere not allowed (line {n.lineno})")
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            v.append(f"{fn.name}: attribute store")
        if isinstance(n, ast.FormattedValue) and any(isinstance(x, ast.Name) and x.id in _CONTENT_NAMES for x in ast.walk(n.value)):
            v.append(f"{fn.name}: content in an f-string")
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "data" and not isinstance(pm[n], ast.Attribute):
            v.append(f"{fn.name}: a slice of the raw bytes goes somewhere not allowed (line {n.lineno})")
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "o":          # o["text"]: the value is content
            p = pm[n]
            if not ((isinstance(p, (ast.Tuple, ast.Assign)) and _flows_into_content_names_only(n, pm))
                    or (isinstance(p, ast.Call) and _call_name(p) in ("str", "Opaque", "isinstance"))):
                v.append(f"{fn.name}: a looked-up JSON value goes somewhere not allowed (line {n.lineno})")
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in _CONTENT_NAMES:
            p = pm[n]
            ok = ((isinstance(p, ast.Attribute) and p.attr in _CONTENT_METHODS and isinstance(pm[p], ast.Call) and pm[p].func is p)
                  or (isinstance(p, ast.Call) and _call_name(p) in ("loads", "Opaque", "isinstance"))
                  or (isinstance(p, ast.Compare) and _is_none_compare(p))
                  or (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not))
                  or (isinstance(p, ast.Subscript) and p.value is n and ((isinstance(p.slice, ast.Constant) and n.id == "o")
                                                                       or (isinstance(p.slice, ast.Slice) and n.id == "data")))
                  or (isinstance(p, (ast.Tuple, ast.Assign)) and _flows_into_content_names_only(n, pm)))
            if not ok:
                v.append(f"{fn.name}: content name {n.id} used at line {n.lineno} other than strip()/startswith/json.loads/Opaque/constant-key lookup")
    return v


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

    def test_the_guard_is_not_vacuous_it_sees_a_planted_comparison_and_a_from_import(self) -> None:
        planted = ast.parse("from re import search\ndef f(text):\n    return 'x' in text\n")
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "re" for n in ast.walk(planted)))
        self.assertTrue(any(isinstance(n, ast.Compare) and {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & self.SENTENCE_NAMES for n in ast.walk(planted)))


class StructureWhitelists(unittest.TestCase):
    """I-1''' / I-2''' / I-3''': the runner's own source obeys the whitelists above, and each whitelist sees a planted violation."""
    SRC = (LIVE / "tools" / "blind_runner.py").read_text(encoding="utf-8")
    TREE = ast.parse(SRC)

    def _planted(self, old: str, new: str) -> ast.AST:
        self.assertEqual(self.SRC.count(old), 1, old)
        return ast.parse(self.SRC.replace(old, new))

    def test_the_exit_functions_are_the_only_callers_of_reveal_and_their_bodies_are_whitelisted(self) -> None:
        fns = _funcs(self.TREE)
        for name, spec in EXIT_SPECS.items():
            self.assertEqual(exit_violations(fns[name], spec), [], name)
        tuple_node = [n for n in self.TREE.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_EXIT_FUNCTIONS"][0].value
        self.assertEqual(sorted(e.id for e in tuple_node.elts), sorted(EXIT_SPECS))                      # the runtime set == the audited set
        self.assertEqual(sorted(f.__name__ for f in br._EXIT_FUNCTIONS), sorted(EXIT_SPECS))
        exits = {name: set(ast.walk(fns[name])) for name in EXIT_SPECS}
        calls = [n for n in ast.walk(self.TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "reveal"]
        self.assertEqual(len(calls), 6)
        for c in calls:
            self.assertTrue(any(c in nodes for nodes in exits.values()), f"reveal() called outside an exit function at line {c.lineno}")
        names = [n for n in ast.walk(self.TREE) if isinstance(n, ast.Name) and n.id == "reveal"]
        self.assertEqual(len(names), len(calls), "reveal is referenced other than by a direct call (an alias)")
        inits = [set(ast.walk(f)) for f in ast.walk(self.TREE) if isinstance(f, ast.FunctionDef) and f.name == "__init__"]
        for n in ast.walk(self.TREE):
            if isinstance(n, ast.Attribute) and n.attr == "_v":
                self.assertTrue(n in set(ast.walk(fns["reveal"])) or any(n in nodes for nodes in inits), f"the wrapper's slot is touched at line {n.lineno}")
        self.assertEqual(len([n for n in ast.walk(self.TREE) if isinstance(n, ast.Constant) and n.value == "_v"]), 1)         # only ``__slots__ = ("_v",)``

    def test_reveal_checks_its_caller_by_code_object_at_run_time(self) -> None:
        import types
        o = br.Opaque("SECRETOPAQUE")
        alias = br.reveal
        for how in (lambda: br.reveal(o), lambda: alias(o), lambda: getattr(sys.modules[br.__name__], "reveal")(o), lambda: vars(br)["reveal"](o),
                    lambda: br.__dict__["reveal"](o)):
            with self.assertRaises(TypeError):
                how()
        # a function with an IDENTICAL body and an equal (but not the same) code object is not an exit either
        clone = types.FunctionType(br._dump_row.__code__.replace(), vars(br), "clone")
        with self.assertRaises(TypeError):
            clone({"reply": o})
        self.assertEqual(br._dump_row({"id": "a", "reply": o}), json.dumps({"id": "a", "reply": "SECRETOPAQUE"}))          # a real exit still works
        self.assertEqual(br._worker_history([{"task": br.Opaque("t"), "answer": br.Opaque("a")}]), [{"task": "t", "answer": "a"}])
        with self.assertRaises(TypeError):                                           # even an exit function takes Opaque values only
            br._dump_row({"id": "a", "reply": "a plain string"})
        with self.assertRaises(TypeError):
            br._worker_history([{"task": "plain", "answer": "plain"}])

    def test_the_spy_and_the_recorder_only_forward_what_they_watch(self) -> None:
        fns = _funcs(self.TREE)
        self.assertEqual(watcher_violations(fns["spy"], self.TREE), [], "spy")
        self.assertNotIn("wrapper", fns)             # the recorder of the removed keyword functions is gone: nothing wraps production code but the JEV

    def test_run_one_wraps_every_production_text_where_it_reads_it(self) -> None:
        run_one = _funcs(self.TREE)["run_one"]
        self.assertEqual(production_text_violations(run_one, self.TREE), [])
        reads = [n for n in _body_nodes(run_one) if isinstance(n, ast.Attribute) and n.attr in _PRODUCTION_TEXT_ATTRS]
        self.assertEqual(sorted(n.attr for n in reads), ["objective", "text", "text"])               # job.objective, res.speak.text, out.text: all wrapped

    def test_the_whole_file_has_no_way_into_a_slot_or_the_modules_namespace_and_no_stash(self) -> None:
        self.assertEqual(whole_file_violations(self.TREE), [])

    def test_the_readers_only_strip_test_for_the_json_marker_parse_and_wrap(self) -> None:
        fns = _funcs(self.TREE)
        for name in ("read_sentences", "read_history", "_utf8_or_exit"):
            self.assertEqual(read_violations(fns[name], self.TREE), [], name)

    def test_each_whitelist_sees_a_planted_violation(self) -> None:
        def seen(kind, old, new, fname=None):
            tree = self._planted(old, new)
            fns = _funcs(tree)
            if kind == "exit":
                return exit_violations(fns[fname], EXIT_SPECS[fname])
            if kind == "watch":
                return watcher_violations(fns[fname], tree)
            if kind == "prod":
                return production_text_violations(fns["run_one"], tree)
            if kind == "whole":
                return whole_file_violations(tree)
            return read_violations(fns[fname], tree)
        planted = [
            ("exit", "    raw = reveal(text)\n", "    raw = reveal(text)\n    if 'qzxw' in raw:\n        raw = ''\n", "_admit_request"),
            ("exit", "    raw = reveal(text)\n", "    raw = reveal(text)\n    _x = json.dumps(raw)\n", "_admit_request"),
            ("exit", "    return C.TaskContext(", "    _y = reveal(objective).startswith('q')\n    return C.TaskContext(", "_task_context"),
            ("exit", 'return json.dumps({**row, "reply": reveal(row["reply"])}, ensure_ascii=False)',
             'return json.dumps({**row, "reply": reveal(row["reply"]), "n": len(row)}, ensure_ascii=False)', "_dump_row"),
            ("watch", '        rec.route = jev_client.choice_answer(resp, "route") or rec.route\n',
             "        if 'qzxw' in state['utterance']:\n            rec.route = 'x'\n" + '        rec.route = jev_client.choice_answer(resp, "route") or rec.route\n', "spy"),
            ("watch", "            rec.error_exc = type(e).__name__\n", "            rec.error_exc = type(e).__name__ + str(e)\n", "spy"),
            ("watch", "        return resp\n", "        _n = len(state)\n        return resp\n", "spy"),
            ("prod", "        job = res.job\n", "        job = res.job\n        if 'qzxw' in job.objective:\n            row['handled_by'] = 'fast'\n"),
            ("prod", "        reply = Opaque(out.text)", "        reply = Opaque(out.text)\n        if 'qzxw' in out.text:\n            pass\n"),
            ("prod", "        if res.job is None:\n", "        if 'qzxw' in str(res.speak):\n            row['handled_by'] = 'fast'\n        if res.job is None:\n"),
            ("whole", "import json\n", "import json\nimport gc\n"),
            ("whole", "ALLOW_REAL_ENV = ", "_STASH = []\nALLOW_REAL_ENV = "),
            ("whole", "def read_history(path: str)", "def _k(x, cache=[]):\n    return cache\n\n\ndef read_history(path: str)"),
            ("read", "            if not line:\n                continue\n", "            if 'qzxw' in line:\n                continue\n            if not line:\n                continue\n", "read_sentences"),
            ("read", '                sid = f"s{len(out) + 1:04d}"\n', '                sid = f"s{len(out) + 1:04d}"\n                _alias = line\n', "read_sentences"),
            ("read", "            if not line.strip():\n                continue\n", "            if len(line) > 3 and not line.strip():\n                continue\n", "read_history"),
        ]
        for case in planted:
            kind, old, new, *rest = case
            self.assertTrue(seen(kind, old, new, rest[0] if rest else None), f"not seen: {kind} {new[:50]!r}")

    def test_inputs_with_other_characters_but_the_same_line_structure_read_the_same_way(self) -> None:
        """I-3''': what the readers return (the number of sentences, their ids, whether each has a task, the history length) depends on the line
        structure only (blank lines, a leading brace, the JSON keys), never on what the lines say."""
        def structure(words: list[str]) -> tuple:
            w = words
            lines = [w[0], "", w[1], json.dumps({"id": "case-1", "text": w[2]}), "   ", json.dumps({"text": w[3], "task": w[4]}), "  " + w[5] + "  ",
                     json.dumps({"id": "z9", "text": w[6]}), w[7]]
            with tempfile.TemporaryDirectory() as d:
                inp = _write(d, lines, "in.jsonl")
                sents = br.read_sentences(inp)
                hist = _write(d, [json.dumps({"task": w[0], "answer": w[1]}), "", json.dumps({"task": w[2]}), json.dumps({"answer": w[3]})], "h.jsonl")
                h = br.read_history(hist)
            return ([s.id for s in sents], [s.task is not None for s in sents], [type(s.text) is br.Opaque for s in sents], len(h),
                    [sorted(x) for x in h])
        base = structure(["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"])
        # characters of every kind, words a runner might fork on, JSON-looking and brace-leading texts (inside a string value), very long and very short
        sets = [["qzxw", "weather", "天气怎么样", "取消", "记住我喜欢", "}", "{", "x"], ["a" * 500, "好", "\u3000x", "。", "NULL", "true", "{\"a\": 1}", "[1]"],
                ["今天北京天气", "stop", "停", "cancel", "你好", "zq", "qzxw qzxw", "id"], ["#", "//", "--", "\\", "'", "\"", "{}", "%s"],
                ["\U0001F600", "é", "ß", "İ", "ǅ", "٣", "ⅷ", "x\u200b"], ["8", "88号", "eighty", "两万三", "http://x", "a@b.c", "C:\\x", "../.."]]
        for words in sets:
            self.assertEqual(structure(words), base, words)

    def test_the_utf8_error_line_number_counts_breaks_the_way_the_readers_do(self) -> None:
        """S-2''': a lone CR, CRLF and LF are one break each in the readers' text mode; the UTF-8 message uses the same count, so the numbers agree."""
        seps = [b"\n", b"\r\n", b"\r", b"\r\r\n", b"\n\r"]
        with tempfile.TemporaryDirectory() as d:
            for i, sep in enumerate(seps):
                parts = [b"one", b"two", b"th\xffree", b"four"]
                path = os.path.join(d, f"bad{i}.bin")
                open(path, "wb").write(sep.join(parts))
                with self.assertRaises(SystemExit) as cm:
                    br._utf8_or_exit(path, "input file")
                with open(path, "rb") as fb:                                    # what a text-mode reader makes of the same file, up to the bad byte
                    good = fb.read().split(b"\xff")[0]
                lines_before = len(io.StringIO(good.decode("utf-8"), newline=None).readlines())
                self.assertIn(f"line {lines_before}", str(cm.exception), sep)
                # and a JSON error on the same physical line is reported with the same number by read_sentences
                good_path = os.path.join(d, f"badjson{i}.txt")
                open(good_path, "wb").write(sep.join([b"one", b"two", b"{not json", b"four"]))
                with self.assertRaises(SystemExit) as cm2:
                    br.read_sentences(good_path)
                self.assertEqual(str(cm2.exception).split(":")[0], f"input line {lines_before}", sep)


if __name__ == "__main__":
    unittest.main()
