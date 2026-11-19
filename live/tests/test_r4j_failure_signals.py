"""Round 4j / B-5 + I-25: failure is read from structure only; no word, regex or name-prefix decides it.

* backend.py: a tool output counts as failed only by ``grounding.output_failed`` (structure); tool-name prefixes no longer pick which tools are stopped.
* realtime.py / gateway.py: an engine error closes the session only when the adapter flagged ``fatal`` (typed), never because the message says "closed".
* orchestrator: the I-25 mutants that survived on 2ed385b / 7cce424 (a new failure word, a raw '"ok": false' probe, any truthy ``failure`` key in JSON,
  web_search judged by its sentence, weather error code read from ``str(e)``) are killed by behaviour tests, and the AST guard below is real.
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import backend, failcodes, gateway, realtime  # noqa: E402
from sophon_live.assistant_tools import orchestrator as orch  # noqa: E402
from sophon_live.assistant_tools import weather  # noqa: E402
from sophon_live.coordinator.schemas import WorkerJob  # noqa: E402
from tests._tool_fakes import failing_tool  # noqa: E402
from tests.test_orch_worker import _say, _tool_call  # noqa: E402
from tests.test_orchestrator import _FakeResp, _FakeSession  # noqa: E402

ARGS = {"get_weather": {"city": "北京"}, "web_search": {"query": "北京 天气"}, "get_news": {"topic": "tech", "lang": "zh"}}
PROSE = ("Error: user not found", "error", "not found", "failed", "unavailable", "unavail", "失败", "不可用", "暂时无法", "暂时连不上", "查不到", "没有搜到",
         "没有搜到关于「北京」的可靠摘要。", "Web search is unavailable right now due to a network problem.", "I couldn't find useful snippets for 'x'.",
         'the field "ok": false is documented', "ok: false")


# --------------------------------------------------------------------------- backend.py (B-5)
class BackendFailureCountIsStructural(unittest.IsolatedAsyncioTestCase):
    async def _run(self, tool: str, outputs: list[str], n_calls: int) -> tuple[list[str], backend.BackOffice]:
        steps = [_FakeResp(200, _tool_call(tool, "{}")) for _ in range(n_calls)] + [_FakeResp(200, _say("ok"))]
        executed: list[str] = []

        async def call_tool(name: str, args: dict) -> str:
            executed.append(name)
            return outputs[min(len(executed) - 1, len(outputs) - 1)]
        bo = backend.BackOffice("t")
        with mock.patch.object(backend.aiohttp, "ClientSession", return_value=_FakeSession(steps)), mock.patch.object(backend, "_key", return_value="k"):
            await bo.run("", [{"name": tool, "parameters": {"type": "object", "properties": {}}}], "task", call_tool)
        return executed, bo

    async def test_prose_that_says_error_or_not_found_is_never_a_failure(self) -> None:
        for prose in PROSE:
            executed, bo = await self._run("get_account", [prose], 4)
            self.assertEqual(len(executed), 4, prose)                 # never stopped by the "already failed" gate
            self.assertEqual(bo.fails, {}, prose)

    async def test_structured_failures_are_counted_and_stop_the_tool_after_max_fails(self) -> None:
        for out in ('{"ok": false}', '{"success": false}', '{"error": "no such user"}'):
            executed, bo = await self._run("get_account", [out], 4)
            self.assertEqual(len(executed), backend.MAX_FAILS, out)
            self.assertEqual(bo.fails["get_account"], backend.MAX_FAILS)
            self.assertIn("Not called", bo.actions[-1])

    async def test_the_gate_does_not_depend_on_the_tool_name(self) -> None:
        for name in ("get_account", "find_user", "search_x", "update_account", "cancel_order", "x"):
            executed, _ = await self._run(name, ['{"ok": false}'], 4)
            self.assertEqual(len(executed), backend.MAX_FAILS, name)

    async def test_a_tool_that_succeeds_is_never_gated(self) -> None:
        executed, bo = await self._run("get_account", ['{"ok": true}'], 4)
        self.assertEqual((len(executed), bo.fails), (4, {}))

    async def test_the_no_result_timeout_envelope_is_structured(self) -> None:
        s = realtime.RealtimeSession.__new__(realtime.RealtimeSession)
        src = Path(realtime.__file__).read_text(encoding="utf-8")
        self.assertIn('"no_result_from_system"', src)
        self.assertNotIn('"Error: no result', src)
        from sophon_live import grounding
        self.assertTrue(grounding.output_failed({"output": json.dumps({"ok": False, "error": "no_result_from_system"})}))
        del s


# --------------------------------------------------------------------------- engine error -> session close (B-5)
class EngineErrorClosesOnlyWhenTyped(unittest.IsolatedAsyncioTestCase):
    async def _drive(self, cls, ev: dict) -> list[str]:
        s = cls.__new__(cls)
        s.closed = False
        class _Q:                                                  # yields the one event, then ends the loop
            def __init__(self): self.items = [ev]
            async def get(self):
                if self.items:
                    return self.items.pop(0)
                raise asyncio.TimeoutError
        s.engine = mock.MagicMock(); s.engine.q = _Q()
        calls: list[str] = []
        async def error(*a, **k): calls.append("error")
        async def close(*a, **k):
            calls.append("close"); s.closed = True
        s.error, s.close, s.id = error, close, "t"
        try:
            await cls._engine_events(s)
        except asyncio.TimeoutError:
            pass
        return calls

    async def test_words_in_the_message_do_not_close_the_session(self) -> None:
        for cls in (realtime.RealtimeSession, gateway.LiveSession):
            for msg in ("engine connection closed", "the window was closed by the user", "closed", '{"code": "closed"}'):
                self.assertEqual(await self._drive(cls, {"t": "error", "message": msg}), [], (cls.__name__, msg))

    async def test_a_fatal_flag_closes_the_session_whatever_the_message(self) -> None:
        for cls in (realtime.RealtimeSession, gateway.LiveSession):
            self.assertEqual(await self._drive(cls, {"t": "error", "message": "x", "fatal": True}), ["error", "close"], cls.__name__)
            self.assertEqual(await self._drive(cls, {"t": "error", "message": "x", "fatal": "yes"}), [], cls.__name__)   # only the typed True

    def test_the_adapter_flags_a_lost_connection_as_fatal_and_an_engine_error_as_not(self) -> None:
        src = Path(__file__).resolve().parents[1].joinpath("sophon_live", "adapters.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        flagged = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "fatal" for k in n.keys)]
        self.assertEqual(len(flagged), 1, flagged)
        self.assertIn("engine connection closed", flagged[0])


# --------------------------------------------------------------------------- orchestrator failure signals (I-25)
class OrchestratorFailureSignals(unittest.IsolatedAsyncioTestCase):
    async def _call(self, tool_out, *, tool: str = "get_weather", required=()):
        job = WorkerJob(task_id="t1", version=1, objective="随便聊聊，说说北京的天气", require_tools=list(required))
        sess = _FakeSession([_FakeResp(200, _tool_call(tool, json.dumps(ARGS[tool], ensure_ascii=False))), _FakeResp(200, _say("好的。"))])
        rt = tool_out if isinstance(tool_out, mock.AsyncMock) else mock.AsyncMock(return_value=tool_out)
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            return await orch.run_job(job)

    async def test_no_prose_is_a_failure_for_any_tool(self) -> None:
        """Kills: a new failure word (失败), an old one back, a lowercase/regex/any() probe, web_search judged by '没有搜到', raw '\"ok\": false'."""
        for tool in ("get_weather", "web_search", "get_news"):
            for prose in PROSE:
                res = await self._call(prose, tool=tool)
                self.assertTrue(res.evidence[0]["ok"], (tool, prose))
                self.assertNotIn("failure", res.evidence[0], (tool, prose))
                self.assertEqual(orch._tool_failure_code(prose, {}), None, prose)

    async def test_a_failure_key_inside_json_is_not_the_failure_signal(self) -> None:
        """Kills: any truthy ``failure`` key in the JSON counts. Only toolmeta's failure code and ``ok: false`` are signals."""
        for out in ('{"ok": true, "failure": "network"}', '{"failure": true}', '{"failure": "timeout", "text": "x"}', '{"error": "boom"}', '{"success": false}'):
            self.assertIsNone(orch._tool_failure_code(out, {}), out)
            res = await self._call(out)
            self.assertTrue(res.evidence[0]["ok"], out)

    async def test_web_search_is_a_failure_by_its_code_not_by_its_sentence(self) -> None:
        sentence = "没有搜到关于「北京」的可靠摘要。"
        res = await self._call(sentence, tool="web_search", required=["web_search"])
        self.assertTrue(res.evidence[0]["ok"])
        res = await self._call(failing_tool(sentence, failcodes.NO_RESULTS), tool="web_search", required=["web_search"])
        self.assertFalse(res.evidence[0]["ok"])
        self.assertEqual(res.evidence[0]["failure"], failcodes.NO_RESULTS)

    def test_a_toolmeta_failure_code_wins_and_nothing_else_does(self) -> None:
        self.assertEqual(orch._tool_failure_code("晴", {"failure": "network"}), "network")
        self.assertEqual(orch._tool_failure_code('{"ok": false}', {}), "error")
        self.assertIsNone(orch._tool_failure_code("晴", {"failure": ""}))
        self.assertIsNone(orch._tool_failure_code("晴", {"count": 3, "success": True}))


class WeatherErrorCodeComesFromTheExceptionType(unittest.TestCase):
    def test_the_message_of_an_exception_is_never_read(self) -> None:
        """Kills: error code derived from str(e). Same words, different types -> the type decides."""
        words = ("timed out", "timeout", "Connection refused", "bad data", "http 500")
        for w in words:
            self.assertEqual(weather._error_code(RuntimeError(w)), "error", w)
            self.assertEqual(weather._error_code(ValueError(w)), "bad_data", w)
            self.assertEqual(weather._error_code(KeyError(w)), "bad_data", w)
            self.assertEqual(weather._error_code(OSError(w)), "network", w)
            self.assertEqual(weather._error_code(urllib.error.URLError(w)), "network", w)
        self.assertEqual(weather._error_code(asyncio.TimeoutError()), "timeout")
        self.assertEqual(weather._error_code(TimeoutError("no words")), "timeout")
        self.assertEqual(weather._error_code(urllib.error.HTTPError("u", 500, "m", {}, None)), "http")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the AST guard (I-25), now real
def _uses(fn: ast.AST, name: str) -> list[tuple[ast.AST, ast.AST]]:
    parents = {c: p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
    return [(n, parents[n]) for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) and n in parents]


def _allowed_out_use(node: ast.AST, parent: ast.AST) -> bool:
    """The tool output may be: passed whole to the two helpers, stored whole as the tool message, sliced for the log line. Nothing compares, searches or aliases it."""
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in ("_tool_failure_code", "_tool_failure_text"):
        return True
    if isinstance(parent, ast.Dict):
        return True
    if isinstance(parent, ast.Subscript) and isinstance(parent.slice, ast.Slice) and parent.value is node:
        return True
    return False


class OrchestratorHasNoTextProbe(unittest.TestCase):
    """Replaces 2ed385b's ``test_the_orchestrator_has_no_text_probe_for_failure`` (its assertion was vacuous: it could never fail)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
        cls.fns = {n.name: n for n in ast.walk(cls.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_run_job_only_passes_the_tool_output_on(self) -> None:
        uses = _uses(self.fns["run_job"], "out")
        self.assertGreaterEqual(len(uses), 5)                      # the guard is looking at something
        bad = [ast.unparse(p) for n, p in uses if not _allowed_out_use(n, p)]
        self.assertEqual(bad, [], "tool output is read, compared or aliased in run_job")

    def test_tool_failure_code_reads_only_structure(self) -> None:
        fn = self.fns["_tool_failure_code"]
        uses = _uses(fn, "t") + _uses(fn, "out")
        self.assertGreaterEqual(len(uses), 3)
        for node, parent in uses:
            ok = (isinstance(parent, ast.BoolOp)                                                       # (out or "")
                  or (isinstance(parent, ast.Call) and ast.unparse(parent.func) == "json.loads")      # parsed as JSON
                  or (isinstance(parent, ast.Attribute) and parent.attr == "startswith"))             # t.startswith("{")
            self.assertTrue(ok, ast.unparse(parent))
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "startswith"):
            self.assertEqual([ast.unparse(a) for a in call.args], ["'{'"])                            # the only prefix looked at is the JSON brace
        text = ast.unparse(fn)
        for banned in (" in t", " in out", "re.", "__import__", "lower()", ".find(", ".count(", "any(", "all("):
            self.assertNotIn(banned, text, banned)

    def test_the_guard_catches_the_survivors(self) -> None:
        """The guard itself is tested: the five rewrites that survived on 2ed385b / 7cce424 are flagged."""
        for src in ('failed = fail_code is not None or "失败" in out', 'failed = x or any(k in out for k in ("a", "b"))',
                    'failed = x or __import__("re").search("a", out) is not None', "o = out", "failed = x or out.lower().startswith('error')"):
            fn = ast.parse(src)
            self.assertTrue([1 for n, p in _uses(fn, "out") if not _allowed_out_use(n, p)], src)


if __name__ == "__main__":
    unittest.main()
