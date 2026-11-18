"""Round 4d / S-23 (failure echo: whole entity only, real error names, refine objective shapes) and S-24 (a dedicated 'busy' sentence)."""
from __future__ import annotations

import asyncio
import threading
import unittest
import urllib.error
from unittest import mock

from sophon_live.assistant_tools import inflight, news, weather, web_search
from sophon_live.coordinator.api import _refined_objective, finish_job
from sophon_live.coordinator.ledger import Ledger
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER, TOOL_FAILURE_TEMPLATES, WorkerJob, WorkerResult, is_tool_failure_sentence as ok

W = {"tool": "get_weather"}
ZH = "暂时查不到{}的天气，请稍后再试。"


def _spoken(said: str, tool: str, objective: str) -> str:
    led = Ledger()
    rec = led.create(objective)
    led.mark_running(rec.task_id)
    job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=objective, require_tools=[tool])
    res = WorkerResult(task_id=rec.task_id, version=rec.version, status="failed", spoken_answer=said,
                       evidence=[{"tool": tool, "ok": False, "failure_text": said}])
    return finish_job(led, job, res).text


class EchoIsAWholeEntity(unittest.TestCase):
    def test_single_characters_and_pieces_of_a_city_are_rejected(self) -> None:
        for piece, obj in (("京", "北京今天天气"), ("海", "上海明天天气怎么样"), ("北", "北京天气"), ("广", "广州天气"), ("I", "weather in Istanbul")):
            self.assertFalse(ok(ZH.format(piece), W, obj), piece)
        self.assertTrue(ok(ZH.format("北京"), W, "北京今天天气"))
        self.assertTrue(ok(ZH.format("上海"), W, "上海明天天气怎么样"))

    def test_a_longer_unknown_place_is_fine_but_not_a_fragment_of_a_known_city(self) -> None:
        self.assertTrue(ok(ZH.format("乌鲁木齐"), W, "乌鲁木齐的天气"))
        self.assertFalse(ok(ZH.format("京天"), W, "北京天气"))

    def test_search_query_echo_needs_two_characters_too(self) -> None:
        self.assertFalse(ok("没有搜到关于「金」的可靠摘要。", {"tool": "web_search"}, "今天黄金多少钱"))
        self.assertTrue(ok("没有搜到关于「黄金」的可靠摘要。", {"tool": "web_search"}, "今天黄金多少钱"))


class RealErrorSentencesAreAccepted(unittest.TestCase):
    def test_lowercase_exception_class_names_and_dotted_city_names(self) -> None:
        for reason in ("sojson:gaierror", "open-meteo:herror, wttr:timeout", "sojson:ConnectionResetError"):
            self.assertTrue(ok(f"Weather for Paris is unavailable right now ({reason}).", W, "weather in Paris"), reason)
        self.assertTrue(ok("Weather for St. Louis is unavailable right now (wttr:no_data).", W, "what's the weather in St. Louis"))
        self.assertTrue(ok("No weather data for Xi'an.", W, "weather in Xi'an today"))

    def test_free_text_is_still_rejected(self) -> None:
        for reason in ("sojson:gaierror sunny 25C", "sojson:", "sojson:g", "sojson:" + "a" * 60, "evil:gaierror"):
            self.assertFalse(ok(f"Weather for Paris is unavailable right now ({reason}).", W, "weather in Paris"), reason)


class RefineObjectiveShapes(unittest.TestCase):
    """api._refined_objective builds  <old> + marker + <new words>,  possibly nested; the echo must follow the NEW words."""

    def test_gaikou_new_city_wins(self) -> None:
        obj = _refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False)
        self.assertTrue(ok(ZH.format("北京"), W, obj))
        self.assertFalse(ok(ZH.format("上海"), W, obj))          # the stale city is not what the job is about any more

    def test_gaikou_without_a_city_keeps_the_old_city(self) -> None:
        obj = _refined_objective("明天上海会下雨吗", "改成后天", slot_fill=False)
        self.assertTrue(ok(ZH.format("上海"), W, obj))
        self.assertFalse(ok(ZH.format("北京"), W, obj))

    def test_nested_refinements_follow_the_last_one(self) -> None:
        obj = _refined_objective(_refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False), "广州吧", slot_fill=False)
        self.assertTrue(ok(ZH.format("广州"), W, obj))
        self.assertFalse(ok(ZH.format("北京"), W, obj))
        self.assertFalse(ok(ZH.format("上海"), W, obj))
        obj2 = _refined_objective(_refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False), "后天吧", slot_fill=False)
        self.assertTrue(ok(ZH.format("上海"), W, obj2) or ok(ZH.format("北京"), W, obj2))     # no city in the last words: an earlier one carries over

    def test_slot_fill_joins_the_original_ask_and_the_supplement(self) -> None:
        obj = _refined_objective("明天天气怎么样", "北京", slot_fill=True)
        self.assertTrue(ok(ZH.format("北京"), W, obj))
        self.assertFalse(ok(ZH.format("上海"), W, obj))

    def test_english_refine(self) -> None:
        obj = _refined_objective("weather in Paris tomorrow", "make it London", slot_fill=False)
        self.assertTrue(ok("No weather data for London.", W, obj))
        self.assertFalse(ok("No weather data for Paris.", W, obj))

    def test_finish_job_end_to_end(self) -> None:
        obj = _refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False)
        self.assertEqual(_spoken(ZH.format("北京"), "get_weather", obj), ZH.format("北京"))
        self.assertEqual(_spoken(ZH.format("上海"), "get_weather", obj), NO_TOOL_ANSWER)


class BusySentence(unittest.IsolatedAsyncioTestCase):
    def test_constants_match_the_whitelist_and_the_orchestrator_markers(self) -> None:
        for tool in ("get_weather", "get_news", "web_search"):
            self.assertTrue(ok(inflight.BUSY_ZH, {"tool": tool}, ""), tool)
            self.assertTrue(ok(inflight.BUSY_EN, {"tool": tool}, ""), tool)
        self.assertFalse(ok(inflight.BUSY_ZH, {"tool": "memory_note"}, ""))
        self.assertIn("同时在查的太多", inflight.BUSY_ZH)
        self.assertIn("too many lookups", inflight.BUSY_EN)
        from pathlib import Path
        from sophon_live.assistant_tools import orchestrator
        src = Path(orchestrator.__file__).read_text(encoding="utf-8")
        self.assertIn('"同时在查的太多" in out', src)             # a busy tool result is a FAILED tool result for the Worker
        self.assertIn('"too many lookups" in out', src)

    async def _full(self):
        full = threading.BoundedSemaphore(1)
        full.acquire()
        return full

    async def test_each_tool_says_busy_when_the_cap_is_full(self) -> None:
        full = await self._full()
        weather.CACHE.clear()
        with mock.patch.object(inflight, "SHARED", full), mock.patch.object(web_search, "_inflight", full):
            self.assertEqual(await web_search.run({"query": "今天黄金价格"}), inflight.BUSY_ZH)
            self.assertEqual(await web_search.run({"query": "latest gold price today"}), inflight.BUSY_EN)
            self.assertEqual(await news.run({"topic": "科技", "lang": "zh"}), inflight.BUSY_ZH)
            self.assertEqual(await news.run({"topic": "technology", "lang": "en"}), inflight.BUSY_EN)
            text, _ = await weather.run_detailed({"city": "北京"})
            self.assertEqual(text, inflight.BUSY_ZH)
            weather.CACHE.clear()
            text, _ = await weather.run_detailed({"city": "Beijing", "lang": "en"})
            self.assertEqual(text, inflight.BUSY_EN)

    async def test_a_real_network_failure_still_says_network(self) -> None:
        boom = urllib.error.URLError("down")
        with mock.patch.object(web_search, "_urlopen_unbounded", side_effect=boom):
            out = await web_search.run({"query": "今天黄金价格"})
        self.assertIn("网络暂时不可用", out)
        with mock.patch.object(news, "_urlopen", side_effect=boom):
            self.assertIn("连不上", await news.run({"topic": "科技", "lang": "zh"}))

    async def test_busy_spoken_by_finish_job(self) -> None:
        for tool in ("get_weather", "get_news", "web_search"):
            self.assertEqual(_spoken(inflight.BUSY_ZH, tool, "今天黄金价格"), inflight.BUSY_ZH)

    def test_templates_are_bound_per_tool(self) -> None:
        for tool, pats in TOOL_FAILURE_TEMPLATES.items():
            self.assertTrue(any(p.fullmatch(inflight.BUSY_ZH) for p in pats), tool)


if __name__ == "__main__":
    unittest.main()
