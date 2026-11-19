"""Round 4f / B-4 (``_ungrounded_city`` without a city list) and I-18 (the speak-now nudge reads a structured success report, not tool text).

Same principle as B-3 (user, 00:27): the model judges, code only enforces. Provenance = the value literally occurs, as a whole word, in the
user's words / a tool result (or is the same known place spelled the other way); "is it a city" is not decided by code.
"""
from __future__ import annotations

import ast
import asyncio
import unittest
from pathlib import Path
from unittest import mock

from sophon_live import failcodes
from sophon_live.assistant_tools import news, toolmeta, weather
from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.coordinator.schemas import WorkerJob
from tests.test_orch_worker import _say, _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession


def _msgs(user: str, tool: str = "") -> list[dict]:
    m = [{"role": "system", "content": "上海 sys prompt"}, {"role": "user", "content": user}]
    if tool:
        m.append({"role": "tool", "content": tool})
    return m


def ungrounded(city: str, user: str, tool: str = "") -> bool:
    return orch._ungrounded_city("get_weather", {"city": city}, _msgs(user, tool))


class UngroundedCityIsProvenanceOnly(unittest.TestCase):
    def test_a_city_the_user_said_is_grounded(self) -> None:
        self.assertFalse(ungrounded("北京", "北京明天天气"))
        self.assertFalse(ungrounded("北京市", "北京明天天气"))                      # the model added 市: the bare name is in the words
        self.assertFalse(ungrounded("乌鲁木齐", "乌鲁木齐的天气"))                    # not in any table: provenance does not need one
        self.assertFalse(ungrounded("Paris", "what's the weather in paris"))

    def test_the_same_known_place_spelled_the_other_way_is_grounded(self) -> None:
        self.assertFalse(ungrounded("Beijing", "北京明天天气"))
        self.assertFalse(ungrounded("北京", "weather in Beijing tomorrow"))

    def test_a_city_nobody_said_is_not_grounded_even_if_another_city_was(self) -> None:
        self.assertTrue(ungrounded("上海", "明天天气怎么样"))
        self.assertTrue(ungrounded("上海", "北京明天天气"))        # 4-and-before: "any known city anywhere" made this grounded; it is not the user's word
        self.assertTrue(ungrounded("Shanghai", "北京明天天气"))
        self.assertTrue(ungrounded("Urumqi", "明天天气怎么样"))

    def test_ascii_value_must_be_a_whole_word(self) -> None:
        self.assertTrue(ungrounded("Rome", "weather in Romeo's town"))
        self.assertFalse(ungrounded("Rome", "weather in Rome"))

    def test_a_tool_result_counts_as_the_words(self) -> None:
        self.assertFalse(ungrounded("广州", "那边呢", tool="广州明天多云"))
        self.assertTrue(ungrounded("广州", "那边呢", tool="北京明天多云"))

    def test_system_prompt_does_not_count(self) -> None:
        self.assertTrue(ungrounded("上海", "明天天气怎么样"))                       # _msgs puts 上海 in the system message on purpose

    def test_other_tools_and_empty_city_are_not_this_checks_business(self) -> None:
        self.assertFalse(orch._ungrounded_city("get_news", {"city": "上海"}, _msgs("x")))
        self.assertFalse(orch._ungrounded_city("get_weather", {"city": "  "}, _msgs("x")))
        self.assertFalse(orch._ungrounded_city("get_weather", {}, _msgs("x")))

    def test_outcome_does_not_depend_on_the_coordinators_city_list(self) -> None:
        """Mutation-style: the removed rule read slots.CITY_NAMES. Patch it to nothing / a hostile list: no answer may change."""
        cases = (("北京", "北京明天天气"), ("上海", "明天天气怎么样"), ("上海", "北京明天天气"), ("乌鲁木齐", "乌鲁木齐的天气"), ("火星", "天气"),
                 ("北京市", "北京天气"), ("Rome", "weather in Rome"), ("Urumqi", "明天天气怎么样"), ("天气", "明天天气怎么样"))
        real = [ungrounded(c, u) for c, u in cases]
        for table in ((), ("天气", "明天", "火星", "Rome", "乌鲁木齐")):
            self.assertEqual([ungrounded(c, u) for c, u in cases], real, table)       # (v0.1.1: coordinator/slots.py is deleted; nothing left to patch)

    def test_the_orchestrator_no_longer_touches_the_city_list(self) -> None:
        src = Path(orch.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            self.assertFalse(isinstance(n, ast.Attribute) and n.attr in ("CITY_NAMES", "CITIES"), "city table referenced")
            self.assertFalse(isinstance(n, ast.Name) and n.id in ("CITY_NAMES", "CITIES"), "city table referenced")
            if isinstance(n, ast.ImportFrom):
                self.assertFalse(any(a.name in ("slots", "CITY_NAMES") for a in n.names))


class UngroundedCityEndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_an_invented_city_becomes_needs_input_and_no_lookup_happens(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"上海"}'))])
        rt = mock.AsyncMock(return_value="should not run")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(WorkerJob(task_id="t1", version=1, objective="明天天气怎么样", require_tools=["get_weather"]))
        self.assertEqual((res.status, res.needs_slots), ("needs_input", ["city"]))
        rt.assert_not_awaited()

    async def test_a_city_in_the_users_words_is_looked_up(self) -> None:
        sess = _FakeSession([_FakeResp(200, _tool_call("get_weather", '{"city":"北京"}')), _FakeResp(200, _say("好的。"))])
        rt = mock.AsyncMock(return_value="北京 多云")
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            await orch.run_job(WorkerJob(task_id="t1", version=1, objective="北京明天天气", require_tools=["get_weather"]))
        rt.assert_awaited_once()


NUDGE = "Tool succeeded."


def _structured_ok(text: str, **fields):
    async def _run(name, args=None):  # noqa: ANN001, ARG001
        toolmeta.note(**{failcodes.SUCCESS_KEY: True}, **fields)
        return text
    return mock.AsyncMock(side_effect=_run)


class SpeakNowNudgeReadsStructure(unittest.IsolatedAsyncioTestCase):
    async def _nudged(self, tool: str, args: str, rt) -> bool:
        sess = _FakeSession([_FakeResp(200, _tool_call(tool, args)), _FakeResp(200, _say("好的。"))])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            await orch.run_job(WorkerJob(task_id="t1", version=1, objective="北京今天天气，科技新闻"))
        msgs = sess.posts[1]["json"]["messages"]
        return any(m.get("role") == "user" and str(m.get("content", "")).startswith(NUDGE) for m in msgs)

    async def test_structured_success_gets_the_nudge_whatever_the_text(self) -> None:
        self.assertTrue(await self._nudged("get_news", '{"topic":"tech"}', _structured_ok("some words with none of the old markers", count=2)))
        self.assertTrue(await self._nudged("get_weather", '{"city":"北京"}', _structured_ok("cloudy and mild", tmin=1, tmax=5)))

    async def test_success_looking_text_without_a_structured_report_gets_no_nudge(self) -> None:
        self.assertFalse(await self._nudged("get_news", '{"topic":"tech"}', mock.AsyncMock(return_value="关于科技，第1条，某某；第2条，某某。")))
        self.assertFalse(await self._nudged("get_news", '{"topic":"tech"}', mock.AsyncMock(return_value="Headlines on tech: a; b.")))
        self.assertFalse(await self._nudged("get_weather", '{"city":"北京"}', mock.AsyncMock(return_value="北京今天多云，气温1到5度，现在3°C。")))

    async def test_a_failed_call_never_gets_the_success_nudge_even_with_a_success_flag(self) -> None:
        async def _run(name, args=None):  # noqa: ANN001, ARG001
            toolmeta.note(**{failcodes.SUCCESS_KEY: True}, failure=failcodes.NETWORK)
            return "x"
        self.assertFalse(await self._nudged("get_weather", '{"city":"北京"}', mock.AsyncMock(side_effect=_run)))

    async def test_only_news_and_weather_get_the_nudge(self) -> None:
        self.assertFalse(await self._nudged("get_datetime", '{"what":"time"}', _structured_ok("现在是 14:00。")))

    async def test_the_orchestrator_has_no_text_probe_left(self) -> None:
        src = Path(orch.__file__).read_text(encoding="utf-8")
        for needle in ('"第1条" in out', '"Headlines on" in out', '"度" in out', '"°" in out', '"°C" in out'):
            self.assertNotIn(needle, src, needle)


class RealToolsReportSuccessInStructure(unittest.IsolatedAsyncioTestCase):
    async def test_news_reports_success_and_how_many_headlines(self) -> None:
        toolmeta.take()
        with mock.patch.object(news, "_fetch_rss", mock.AsyncMock(return_value="<rss/>")), \
                mock.patch.object(news, "_titles_from_rss", return_value=["甲公司发布新品", "乙公司获融资", "丙大学成果", "丁市地铁开通"]):
            out = await news.run({"topic": "tech", "lang": "zh", "max_items": 2})
        meta = toolmeta.take()
        self.assertIn("第1条", out)
        self.assertIs(meta.get(failcodes.SUCCESS_KEY), True)
        self.assertEqual(meta.get("count"), 2)

    async def test_news_failure_reports_no_success(self) -> None:
        toolmeta.take()
        with mock.patch.object(news, "_fetch_rss", mock.AsyncMock(side_effect=OSError("down"))):
            await news.run({"topic": "tech", "lang": "zh"})
        meta = toolmeta.take()
        self.assertFalse(meta.get(failcodes.SUCCESS_KEY))
        self.assertEqual(meta.get("failure"), failcodes.NETWORK)

    async def test_weather_reports_success_only_with_temperatures(self) -> None:
        weather.CACHE.clear()
        data = {"tmin": 1, "tmax": 5, "desc": "cloudy", "precip": None, "now": 3}
        toolmeta.take()
        with mock.patch.object(weather, "_first_ok", mock.AsyncMock(return_value=(data, "open-meteo", []))):
            await weather.run({"city": "Rome", "lang": "en"})
        self.assertIs(toolmeta.take().get(failcodes.SUCCESS_KEY), True)
        # a cache hit is still a success report
        with mock.patch.object(weather, "_first_ok", mock.AsyncMock(side_effect=AssertionError("cache expected"))):
            await weather.run({"city": "Rome", "lang": "en"})
        self.assertIs(toolmeta.take().get(failcodes.SUCCESS_KEY), True)
        weather.CACHE.clear()
        with mock.patch.object(weather, "_first_ok", mock.AsyncMock(return_value=(None, "", ["open-meteo:network"]))):
            await weather.run({"city": "Rome", "lang": "en"})
        meta = toolmeta.take()
        self.assertFalse(meta.get(failcodes.SUCCESS_KEY))
        self.assertEqual(meta.get("failure"), failcodes.NETWORK)
        weather.CACHE.clear()


if __name__ == "__main__":
    unittest.main()
