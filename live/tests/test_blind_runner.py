"""I-10 blind runner: behaviour, privacy (sentence text goes to the output file and nowhere else), and no judgment in the runner."""
from __future__ import annotations

import ast
import asyncio
import io
import json
import logging
import os
import pathlib
import re
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


class Behaviour(unittest.TestCase):
    def test_plain_lines_get_ids_and_fake_mode_records_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            self.assertEqual(br.main([_write(d, SENTENCES + [""]), out]), 0)
            rows = _rows(out)
        self.assertEqual([r["id"] for r in rows], ["s0001", "s0002", "s0003", "s0004"])
        for r in rows:
            for k in ("route", "decision", "jev_route", "via", "degraded", "handled_by", "tools", "tool_ok", "worker_status", "reply"):
                self.assertIn(k, r)
            self.assertEqual((r["route"], r["via"], r["degraded"], r["jev_route"]), ("WORKER", "jev", False, "WORKER"))
            self.assertEqual(r["reply"], br.FAKE_REPLY)

    def test_jsonl_keeps_opaque_ids_and_a_constant_fake_route_is_recorded_not_judged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inp = _write(d, [json.dumps({"id": "case-A", "text": SENTENCES[0]}), json.dumps({"id": "7", "text": SENTENCES[1]})], "in.jsonl")
            out = os.path.join(d, "out.jsonl")
            self.assertEqual(br.main([inp, out, "--fake-route", "FAST"]), 0)
            rows = _rows(out)
        self.assertEqual([r["id"] for r in rows], ["case-A", "7"])
        self.assertTrue(all(r["route"] == "FAST" and r["handled_by"] == "fast" and r["reply"] == "" for r in rows))

    def test_the_output_does_not_echo_the_sentence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            br.main([_write(d, SENTENCES), out])
            blob = open(out, encoding="utf-8").read()
        for m in MARKERS:
            self.assertNotIn(m, blob)

    def test_a_jev_failure_is_recorded_as_degraded_heuristic(self) -> None:
        async def boom(state, questions, **kw):   # noqa: ANN001
            raise RuntimeError("jev down")
        from sophon_live import coordinator as C
        from sophon_live.coordinator import api
        api_use = api.USE_JEV
        api.USE_JEV = True
        try:
            with br.quiet_and_sealed(False):
                row = asyncio.run(br.run_one("x", "现在几点了", ledger=C.Ledger(), jev=boom, worker=br._fake_worker))
        finally:
            api.USE_JEV = api_use
        self.assertEqual((row["via"], row["degraded"], row["jev_route"]), ("heuristic", True, None))

    def test_a_crash_is_recorded_as_a_class_name_only(self) -> None:
        async def bad_worker(job, **kw):   # noqa: ANN001
            raise ValueError("secret ZQXV7731 text")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.jsonl")
            with mock.patch.object(br, "_fake_worker", bad_worker):
                br.main([_write(d, [SENTENCES[0]]), out])
            self.assertEqual(_rows(out), [{"id": "s0001", "error": "ValueError"}])

    def test_real_mode_is_refused_without_the_env_gate_and_gateway_mode_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p, out = _write(d, SENTENCES), os.path.join(d, "out.jsonl")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(br.ALLOW_REAL_ENV, None)
                self.assertEqual(br.main([p, out, "--mode", "real", "--jev-url", "http://127.0.0.1:1"]), 4)
                os.environ[br.ALLOW_REAL_ENV] = "1"
                self.assertEqual(br.main([p, out, "--mode", "real"]), 4)             # no --jev-url
            self.assertEqual(br.main([p, out, "--mode", "gateway"]), 3)
            self.assertFalse(os.path.exists(out))

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


class Privacy(unittest.TestCase):
    """Sentence text may appear in the output file and nowhere else: stdout, stderr, log records, any other file."""

    def _scan(self, root: str, skip: str) -> list[str]:
        hits = []
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                if os.path.abspath(p) == os.path.abspath(skip):
                    continue
                data = open(p, "rb").read().decode("utf-8", "ignore")
                if any(m in data for m in MARKERS):
                    hits.append(p)
        return hits

    def test_in_process_nothing_leaks_to_stdout_stderr_log_records_or_files(self) -> None:
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
        try:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "out.jsonl")
                inp = _write(d, SENTENCES)
                cwd = os.getcwd()
                os.chdir(d)
                so, se = io.StringIO(), io.StringIO()
                try:
                    with mock.patch("sys.stdout", so), mock.patch("sys.stderr", se):
                        self.assertEqual(br.main([inp, out]), 0)
                finally:
                    os.chdir(cwd)
                blob = so.getvalue() + se.getvalue()
                self.assertFalse([m for m in MARKERS if m in blob], blob)
                self.assertEqual(records, [], [r.getMessage() for r in records][:3])
                self.assertEqual([x for x in self._scan(d, skip=out) if os.path.abspath(x) != os.path.abspath(inp)], [])
                self.assertEqual(len(_rows(out)), len(SENTENCES))
        finally:
            root.removeHandler(h); root.setLevel(old_level)
        self.assertEqual(mem_default.read_bytes() if mem_default.exists() else None, mem_before)

    def test_the_check_is_not_vacuous_the_system_does_log_sentence_text_when_logging_is_on(self) -> None:
        """Without the runner's seal the same path DOES put the sentence in log records, so the test above really protects something."""
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
            env = dict(os.environ, PYTHONPATH=str(LIVE), PYTHONDONTWRITEBYTECODE="1", SOPHON_LIVE_MEMORY_PATH=os.path.join(d, "mem.json"),
                       PYTHONWARNINGS="default")
            p = subprocess.run([sys.executable, "-m", "tools.blind_runner", inp, out], cwd=str(LIVE), env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            self.assertFalse([m for m in MARKERS if m in p.stdout + p.stderr], p.stdout + p.stderr)
            # the input file holds the sentences by definition; everything else in the tmp dir must be clean of them
            self.assertEqual([x for x in self._scan(d, skip=out) if os.path.abspath(x) != os.path.abspath(inp)], [])
            self.assertFalse(os.path.exists(os.path.join(d, "mem.json")))        # the memory store was redirected to the null device

    def test_a_memory_write_goes_to_the_null_device_not_to_the_configured_store(self) -> None:
        from sophon_live.assistant_tools import memory
        with tempfile.TemporaryDirectory() as d:
            mine = os.path.join(d, "m.json")
            with mock.patch.dict(os.environ, {"SOPHON_LIVE_MEMORY_PATH": mine}):
                with br.quiet_and_sealed(False):
                    self.assertEqual(str(memory._memory_path()), os.devnull)
                self.assertEqual(str(memory._memory_path()), mine)               # restored


class RunnerHoldsNoJudgment(unittest.TestCase):
    def test_no_regex_no_cjk_literal_and_no_comparison_with_text(self) -> None:
        src = (LIVE / "tools" / "blind_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("re", imported)
        code_strings = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                code_strings.append(n.value)
        # docstring excluded (module docstring is the first Expr)
        doc = ast.get_docstring(tree) or ""
        cjk = [s for s in code_strings if s != doc and re.search(r"[\u4e00-\u9fff]", s)]
        self.assertEqual(cjk, [])
        self.assertNotIn("in text", src.replace("for sid, text in", ""))        # the runner never searches the sentence
