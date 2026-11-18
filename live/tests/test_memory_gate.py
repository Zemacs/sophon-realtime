"""Round 1 / item 3: memory_gate. Long-term preferences != task-scoped slots; writes go through the gate.

Pure CPU; the JSON store lives in a temp file.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import memory, runner
from sophon_live.coordinator import Ledger, WorkerJob
from sophon_live.coordinator import memory_gate as mg
from sophon_live.coordinator.api import complete_job
from sophon_live.coordinator.schemas import TaskStatus
from tests.worker_stub import as_worker


def _run(coro):
    return asyncio.run(coro)


class GateCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "mem.json"
        self._env = os.environ.get("SOPHON_LIVE_MEMORY_PATH")
        os.environ["SOPHON_LIVE_MEMORY_PATH"] = str(self.path)
        self.ledger = Ledger()

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop("SOPHON_LIVE_MEMORY_PATH", None)
        else:
            os.environ["SOPHON_LIVE_MEMORY_PATH"] = self._env
        self._tmp.cleanup()

    def task(self, utterance: str):
        rec = self.ledger.create(utterance)
        self.ledger.mark_running(rec.task_id)
        return rec, mg.TaskContext(task_id=rec.task_id, version=rec.version, utterance=utterance, ledger=self.ledger)

    def stored(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}


class IntentTests(unittest.TestCase):
    def test_explicit_save_intent(self) -> None:
        for t in ("记住我喜欢美式", "以后默认查北京天气", "把这个记下来", "我喜欢听爵士"):
            self.assertTrue(mg.has_explicit_save_intent(t), t)
        for t in ("明天天气怎么样。用户补充：北京的", "现在几点了", "科技新闻", ""):
            self.assertFalse(mg.has_explicit_save_intent(t), t)


class GatedWriteTests(GateCase):
    def test_slot_fill_answer_is_not_saved_as_long_term_memory(self) -> None:
        rec, ctx = self.task("明天天气怎么样。用户补充：北京的")
        with mg.bind_task(ctx):
            out = _run(memory.run({"op": "save", "key": "preferred_city", "value": "北京"}))
        self.assertIn("只用于当前任务", out)
        self.assertEqual(self.stored(), {})
        self.assertEqual(self.ledger.get_slots(rec.task_id), {"preferred_city": "北京"})

    def test_explicit_remember_saves_long_term(self) -> None:
        rec, ctx = self.task("记住我喜欢美式")
        with mg.bind_task(ctx):
            out = _run(memory.run({"op": "save", "key": "coffee", "value": "美式"}))
        self.assertIn("记住", out)
        self.assertEqual(self.stored(), {"coffee": "美式"})
        self.assertEqual(self.ledger.get_slots(rec.task_id), {})

    def test_explicit_remember_of_a_task_slot_key_is_stored_as_preferred(self) -> None:
        rec, ctx = self.task("记住我在北京，以后默认查北京")
        with mg.bind_task(ctx):
            _run(memory.run({"op": "save", "key": "city", "value": "北京"}))
        self.assertEqual(self.stored(), {"preferred_city": "北京"})  # never shares a key with the task slot "city"

    def test_without_task_context_direct_tool_calls_keep_working(self) -> None:
        out = _run(memory.run({"op": "save", "key": "city", "value": "上海"}))
        self.assertIn("记住", out)
        self.assertEqual(self.stored(), {"city": "上海"})

    def test_recall_and_list_only_read_long_term(self) -> None:
        self.path.write_text(json.dumps({"coffee": "美式"}, ensure_ascii=False), encoding="utf-8")
        rec, ctx = self.task("我的咖啡偏好是什么")
        self.ledger.set_slot(rec.task_id, 1, "city", "北京")
        with mg.bind_task(ctx):
            self.assertIn("美式", _run(memory.run({"op": "recall", "key": "coffee"})))
            self.assertIn("还没有", _run(memory.run({"op": "recall", "key": "city"})))
            lst = _run(memory.run({"op": "list"}))
        self.assertIn("coffee", lst)
        self.assertNotIn("city", lst)

    def test_long_term_value_never_fills_a_task_slot(self) -> None:
        self.path.write_text(json.dumps({"preferred_city": "北京"}, ensure_ascii=False), encoding="utf-8")
        rec, ctx = self.task("明天天气怎么样")
        self.assertEqual(self.ledger.get_slots(rec.task_id), {})
        with mg.bind_task(ctx):
            _run(memory.run({"op": "recall", "key": "preferred_city"}))
        self.assertEqual(self.ledger.get_slots(rec.task_id), {})  # reading memory does not create slots

    def test_context_is_unbound_after_the_block(self) -> None:
        rec, ctx = self.task("x")
        with mg.bind_task(ctx):
            self.assertIs(mg.current_task(), ctx)
        self.assertIsNone(mg.current_task())


class SlotScopeTests(GateCase):
    def test_slots_are_per_task_and_version(self) -> None:
        a, _ = self.task("上海天气")
        b, _ = self.task("科技新闻")
        self.assertTrue(self.ledger.set_slot(a.task_id, 1, "city", "上海"))
        self.assertTrue(self.ledger.set_slot(b.task_id, 1, "topic", "科技"))
        self.assertEqual(self.ledger.get_slots(a.task_id), {"city": "上海"})
        self.assertEqual(self.ledger.get_slots(b.task_id), {"topic": "科技"})

    def test_refine_starts_with_empty_slots(self) -> None:
        a, _ = self.task("上海明天天气")
        self.ledger.set_slot(a.task_id, 1, "city", "上海")
        new = self.ledger.refine(a.task_id, "改成北京", version=1)
        assert new is not None
        self.assertEqual(self.ledger.get_slots(a.task_id), {})
        self.assertTrue(self.ledger.set_slot(a.task_id, 2, "city", "北京"))
        self.assertEqual(self.ledger.get_slots(a.task_id), {"city": "北京"})

    def test_stale_or_finished_task_cannot_take_slots(self) -> None:
        a, _ = self.task("上海明天天气")
        self.ledger.refine(a.task_id, "改成北京", version=1)
        self.assertFalse(self.ledger.set_slot(a.task_id, 1, "city", "上海"))  # stale version
        c, _ = self.task("讲个笑话")
        self.ledger.cancel(c.task_id, 1)
        self.assertFalse(self.ledger.set_slot(c.task_id, 1, "x", "y"))
        d, _ = self.task("现在几点了")
        complete_job(self.ledger, WorkerJob(task_id=d.task_id, version=1, objective="x"), "三点")
        self.assertFalse(self.ledger.set_slot(d.task_id, 1, "x", "y"))
        self.assertEqual(self.ledger.tasks[d.task_id].status, TaskStatus.COMPLETED)

    def test_tool_args_are_captured_as_slots_of_the_bound_task(self) -> None:
        rec, ctx = self.task("北京明天天气")

        async def fake_weather(args):
            return "北京明天晴，12度。"

        orig = runner._HANDLERS["get_weather"]
        runner._HANDLERS["get_weather"] = fake_weather
        try:
            with mg.bind_task(ctx):
                out = _run(runner.run_tool("get_weather", {"city": "Beijing", "when": "tomorrow"}))
        finally:
            runner._HANDLERS["get_weather"] = orig
        self.assertIn("12度", out)
        self.assertEqual(self.ledger.get_slots(rec.task_id), {"city": "Beijing", "when": "tomorrow"})

    def test_no_capture_without_bound_task(self) -> None:
        async def fake_weather(args):
            return "x"

        orig = runner._HANDLERS["get_weather"]
        runner._HANDLERS["get_weather"] = fake_weather
        try:
            _run(runner.run_tool("get_weather", {"city": "Beijing"}))
        finally:
            runner._HANDLERS["get_weather"] = orig
        self.assertEqual(self.ledger.tasks, {})


class GatewayBindingTests(GateCase):
    def _session(self):
        from tests.test_gateway_coord_control import _session

        return _session()

    def test_worker_memory_write_for_a_plain_task_becomes_a_slot(self) -> None:
        import unittest.mock as mock

        import sophon_live.assistant_tools as tools
        from sophon_live.coordinator import api
        from tests.test_r0_coord_boundary import _arm

        session = self._session()

        async def orch(task, instructions="", history=None):
            # what a sloppy worker LLM does after 「北京的」: tries to save the city as a preference
            await runner.run_tool("memory_note", {"op": "save", "key": "preferred_city", "value": "北京"})
            return "北京明天晴。"

        _arm(session, "d1", "北京明天天气")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(orch)):
            _run(session._coord_resolve("d1", "北京明天天气"))
        tid = session.coord_ledger.order[0]
        self.assertEqual(self.stored(), {})
        self.assertEqual(session.coord_ledger.tasks[tid].slots.get("preferred_city"), "北京")

    def test_worker_memory_write_after_explicit_remember_is_saved(self) -> None:
        import unittest.mock as mock

        import sophon_live.assistant_tools as tools
        from sophon_live.coordinator import api
        from tests.test_r0_coord_boundary import _arm

        session = self._session()

        async def orch(task, instructions="", history=None):
            await runner.run_tool("memory_note", {"op": "save", "key": "coffee", "value": "美式"})
            return "好的，记住了。"

        _arm(session, "d1", "记住我喜欢美式")
        with mock.patch.object(api, "USE_JEV", False), mock.patch.object(tools, "run_worker_job", as_worker(orch)):
            _run(session._coord_resolve("d1", "记住我喜欢美式"))
        self.assertEqual(self.stored(), {"coffee": "美式"})


if __name__ == "__main__":
    unittest.main()
