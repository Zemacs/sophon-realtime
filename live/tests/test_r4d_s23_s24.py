"""Round 4d / S-23 + S-24, reworked in Round 4e (Critic B-3, I-17) under the user's 00:27 principle: the model judges, code only enforces.

What the code may do with a failure sentence: check PROVENANCE (what it echoes literally occurs in the user's words; it is one of OUR tool's
templates; the parenthesis holds only members of a closed enum) and read STRUCTURED failure signals (toolmeta ``failure`` code / ``ok:false``).
What it may not do: decide from word lists or message text what a place name is, which place the user finally meant, or whether a tool failed.
The 4d tests that encoded those decisions (single-character / fragment-of-a-known-city rejection, "the new city wins" in a 改口) are replaced
by the provenance versions below; they were not deleted to get green: the behavior they pinned is exactly what B-3 removes.
"""
from __future__ import annotations

import ast
import asyncio
import json
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from sophon_live import failcodes
from sophon_live.assistant_tools import inflight, news, toolmeta, weather, web_search
from sophon_live.assistant_tools import orchestrator as orch
from sophon_live.coordinator import schemas
from sophon_live.coordinator.api import _refined_objective, finish_job
from sophon_live.coordinator.ledger import Ledger
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER, TOOL_FAILURE_TEMPLATES, WorkerJob, WorkerResult, is_tool_failure_sentence as ok
from tests._tool_fakes import failing_tool
from tests.test_orch_worker import _say, _tool_call
from tests.test_orchestrator import _FakeResp, _FakeSession

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


class EchoIsProvenanceOnly(unittest.TestCase):
    """B-3: the echo check is "does it literally occur, as a whole word, in what the user said". No place-name list decides anything."""

    BATTERY = (
        (ZH.format("北京"), "北京今天天气"), (ZH.format("上海"), "北京今天天气"), (ZH.format("京"), "北京今天天气"), (ZH.format("京天"), "北京天气"),
        (ZH.format("乌鲁木齐"), "乌鲁木齐的天气"), (ZH.format("火星"), "今天天气"),
        ("Weather for Paris is unavailable right now (sojson:network).", "weather in Paris"),
        ("Weather for Par is unavailable right now (sojson:network).", "weather in Paris"),
        ("No weather data for St. Louis.", "what's the weather in St. Louis"), ("No weather data for Xi'an.", "weather in Xi'an today"),
        ("No weather data for I.", "weather in Istanbul"), ("No weather data for Rome.", "weather in Romeo's town"),
    )

    def test_echo_must_literally_be_in_the_users_words(self) -> None:
        self.assertTrue(ok(ZH.format("北京"), W, "北京今天天气"))
        self.assertTrue(ok(ZH.format("乌鲁木齐"), W, "乌鲁木齐的天气"))
        self.assertFalse(ok(ZH.format("上海"), W, "北京今天天气"))               # a city nobody said
        self.assertFalse(ok(ZH.format("火星"), W, "今天天气"))
        self.assertFalse(ok("No weather data for Par.", W, "weather in Paris"))     # ASCII: whole word only
        self.assertFalse(ok("No weather data for I.", W, "weather in Istanbul"))
        self.assertTrue(ok("No weather data for St. Louis.", W, "what's the weather in St. Louis"))
        self.assertTrue(ok("No weather data for Xi'an.", W, "weather in Xi'an today"))
        self.assertFalse(ok("没有搜到关于「金」的可靠摘要。", {"tool": "web_search"}, "今天美元多少钱"))     # not in the words at all
        self.assertTrue(ok("没有搜到关于「黄金」的可靠摘要。", {"tool": "web_search"}, "今天黄金多少钱"))

    def test_whether_an_echo_is_a_whole_place_name_is_not_decided_by_code(self) -> None:
        """4d rejected 京 / 海 / 北 / 京天 because a 21-city list said they cut into 北京 / 上海. That judgment is gone: the Worker model chose the
        tool's argument from the same words, so a fragment there is a model error to be fixed in the model's prompt / evals, not guarded by a list.
        Provenance alone accepts them, and this test pins that on purpose (it is the documented behavior change of Round 4e)."""
        self.assertTrue(ok(ZH.format("京"), W, "北京今天天气"))
        self.assertTrue(ok(ZH.format("京天"), W, "北京天气"))

    def test_outcome_does_not_depend_on_any_place_list(self) -> None:
        """Mutation-style: run the same battery with the weather tool's name table replaced by nothing / by a hostile table. A rule that
        consults a city list (the removed _is_whole_entity / 改口 logic) would change some answer; provenance does not."""
        real = {c: ok(t, W, o) for c, (t, o) in enumerate(self.BATTERY)}
        for table in ({}, {"京天": "Jingtian", "北京": "Beijing", "天气": "Weather", "乌鲁木齐": "Urumqi", "Louis": "Louis"}):
            with mock.patch.object(weather, "CITIES", table):
                got = {c: ok(t, W, o) for c, (t, o) in enumerate(self.BATTERY)}
            for c, (t, o) in enumerate(self.BATTERY):
                self.assertEqual(got[c], real[c], (t, o, table))

    def test_the_city_decision_helpers_are_gone(self) -> None:
        for name in ("_known_cities", "_is_whole_entity", "_split_objective", "_REFINE_MARKERS"):
            self.assertFalse(hasattr(schemas, name), name)

    def test_place_names_are_consulted_in_one_place_only_the_alias_translation(self) -> None:
        """The only remaining use of a name table in the coordinator: translating a value (北京 <-> Beijing) so provenance can be checked."""
        for path in (Path(schemas.__file__), Path(orch.__file__)):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            users = set()
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for n in ast.walk(fn):
                        if (isinstance(n, ast.Name) and n.id in ("CITIES", "CITY_NAMES")) or (isinstance(n, ast.Attribute) and n.attr in ("CITIES", "CITY_NAMES")):
                            users.add(fn.name)
                        if isinstance(n, ast.ImportFrom) and any(a.name in ("CITIES", "CITY_NAMES") for a in n.names):
                            users.add(fn.name)
            if path.name == "schemas.py":
                self.assertEqual(users, {"_city_aliases"}, users)

    def test_known_city_in_either_language_is_still_the_users_words(self) -> None:
        self.assertTrue(ok(ZH.format("北京"), W, "what's the weather in Beijing"))        # the tool names it canonically, the user wrote it in English
        self.assertFalse(ok(ZH.format("上海"), W, "what's the weather in Beijing"))


class RefineIsJudgedByTheModelNotTheCode(unittest.TestCase):
    """api._refined_objective builds  <old> + marker + <new words>.  Which city the user finally means is the Worker model's call (it reads the
    whole objective, marker included, and picks the tool argument). The Coordinator only proves the echoed place is somewhere in those words."""

    def test_both_the_old_and_the_new_place_are_the_users_words(self) -> None:
        obj = _refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False)
        self.assertTrue(ok(ZH.format("北京"), W, obj))
        self.assertTrue(ok(ZH.format("上海"), W, obj))          # 4d rejected this by reading the marker and a city list; now it is the model's decision
        self.assertFalse(ok(ZH.format("广州"), W, obj))          # a place that is nowhere in the words is still rejected

    def test_marker_text_has_no_effect_on_the_outcome(self) -> None:
        for slot_fill in (False, True):
            obj = _refined_objective("明天上海会下雨吗", "北京呢", slot_fill=slot_fill)
            plain = obj.replace("。用户改口，以新说法为准：", " / ").replace("。用户补充：", " / ")
            for city in ("北京", "上海", "广州"):
                self.assertEqual(ok(ZH.format(city), W, obj), ok(ZH.format(city), W, plain), (city, slot_fill))

    def test_nested_and_slot_fill_and_english(self) -> None:
        obj = _refined_objective(_refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False), "广州吧", slot_fill=False)
        self.assertTrue(all(ok(ZH.format(c), W, obj) for c in ("上海", "北京", "广州")))
        self.assertFalse(ok(ZH.format("深圳"), W, obj))
        obj = _refined_objective("明天天气怎么样", "北京", slot_fill=True)
        self.assertTrue(ok(ZH.format("北京"), W, obj))
        self.assertFalse(ok(ZH.format("上海"), W, obj))
        obj = _refined_objective("weather in Paris tomorrow", "make it London", slot_fill=False)
        self.assertTrue(ok("No weather data for London.", W, obj))
        self.assertFalse(ok("No weather data for Rome.", W, obj))

    def test_finish_job_end_to_end(self) -> None:
        obj = _refined_objective("明天上海会下雨吗", "北京呢", slot_fill=False)
        self.assertEqual(_spoken(ZH.format("北京"), "get_weather", obj), ZH.format("北京"))
        self.assertEqual(_spoken(ZH.format("广州"), "get_weather", obj), NO_TOOL_ANSWER)        # provenance fails: the generic sentence is spoken


class ParenthesisHoldsOnlyTheClosedEnum(unittest.IsolatedAsyncioTestCase):
    """I-17: the English weather failure sentence's slot is <source>:<code> with <code> in failcodes.SOURCE_ERROR_CODES, nothing else."""

    def _sentence(self, reason: str) -> str:
        return f"Weather for Paris is unavailable right now ({reason})."

    def test_every_enum_member_and_the_real_shapes_pass(self) -> None:
        for code in failcodes.SOURCE_ERROR_CODES:
            self.assertTrue(ok(self._sentence(f"sojson:{code}"), W, "weather in Paris"), code)
        self.assertTrue(ok(self._sentence("sojson:network, open-meteo:timeout, wttr:no_data"), W, "weather in Paris"))
        self.assertTrue(ok(self._sentence("no source"), W, "weather in Paris"))

    def test_free_tokens_are_rejected_even_when_shaped_like_a_code(self) -> None:
        for reason in ("sojson:sunny_minus3_to_11", "open-meteo:sunny_minus3_to_11, wttr:cloudy_5_to_9", "sojson:OSError", "sojson:gaierror",
                       "wttr:TimeoutError", "sojson:Network", "sojson:network_", "sojson:network2",
                       ", ".join(["wttr:" + "a" * 40] * 8), "sojson:" + "x" * 40):
            self.assertFalse(ok(self._sentence(reason), W, "weather in Paris"), reason)

    def test_the_regex_alternatives_are_exactly_the_enum(self) -> None:
        self.assertEqual(set(schemas._WEATHER_SRC.split(":(?:")[1].rstrip(")").split("|")), set(failcodes.SOURCE_ERROR_CODES))

    def test_every_error_the_weather_tool_can_note_is_in_the_enum(self) -> None:
        samples = (asyncio.TimeoutError(), TimeoutError(), OSError("x"), ConnectionResetError(), urllib.error.URLError("x"),
                   urllib.error.HTTPError("u", 500, "m", {}, None), json.JSONDecodeError("m", "d", 0), UnicodeDecodeError("utf-8", b"", 0, 1, "r"),
                   KeyError("k"), IndexError(), TypeError(), RuntimeError("sunny 25C"), Exception("free text"))
        seen = {weather._error_code(e) for e in samples}
        self.assertLessEqual(seen, set(failcodes.SOURCE_ERROR_CODES))
        self.assertEqual({weather._error_code(RuntimeError("sunny 25C")), weather._error_code(Exception("zzz"))}, {"error"})     # message text never leaks
        self.assertEqual(weather._error_code(asyncio.TimeoutError()), "timeout")
        self.assertEqual(weather._error_code(urllib.error.HTTPError("u", 500, "m", {}, None)), "http")

    async def test_the_real_sentence_after_real_failures_passes_the_whitelist(self) -> None:
        weather.CACHE.clear()
        for exc in (OSError("down"), asyncio.TimeoutError(), ValueError("bad json"), RuntimeError("sunny 25C minus3")):
            async def boom(*a, _e=exc, **k):
                raise _e
            with mock.patch.object(weather, "_fetch_json", boom), mock.patch.dict("os.environ", {"SOPHON_WEATHER_TIMEOUT": "1", "SOPHON_WEATHER_ROUNDS": "1"}):
                text, meta = await weather.run_detailed({"city": "Paris", "lang": "en"})
            self.assertTrue(text.startswith("Weather for 巴黎 is unavailable"), text)        # the tool names a known city canonically
            self.assertNotIn("sunny", text)
            self.assertTrue(ok(text, W, "weather in Paris"), text)
            self.assertEqual(meta["failure"], failcodes.NETWORK)
            weather.CACHE.clear()


class FailureIsReadFromStructureNotText(unittest.IsolatedAsyncioTestCase):
    """B-3 / orchestrator: ``failed`` comes from the tool's structured report. Mutation-style: every keyword the removed text probes looked for
    is returned as plain text WITHOUT a structured signal and must not make the call a failure; the structured signals must."""

    OLD_PROBES = ("暂时连不上", "同时在查的太多了", "too many lookups", "暂时查不到", "暂时没有", "unreachable", "unavailable", "不可用",
                  "没有搜到", "couldn't find useful snippets", "ok: false", "OK: FALSE")

    async def _call(self, tool_out, *, tool: str = "get_weather", required=()):
        job = WorkerJob(task_id="t1", version=1, objective="随便聊聊，说说北京的天气", require_tools=list(required))
        sess = _FakeSession([_FakeResp(200, _tool_call(tool, '{"city":"北京"}')), _FakeResp(200, _say("好的。"))])
        rt = tool_out if isinstance(tool_out, mock.AsyncMock) else mock.AsyncMock(return_value=tool_out)
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            return await orch.run_job(job)

    async def test_failure_words_in_plain_text_are_not_a_failure(self) -> None:
        for probe in self.OLD_PROBES:
            res = await self._call(f"北京 {probe} 北京")
            self.assertTrue(res.evidence[0]["ok"], probe)
            self.assertNotIn("failure", {k for k, v in res.evidence[0].items() if v})
            self.assertEqual(res.status, "completed", probe)

    async def test_ok_false_json_is_a_failure_whatever_the_spacing(self) -> None:
        for out in ('{"ok": false, "error": "boom"}', '{"ok":false,"error":"boom"}', '{ "tool": "x", "ok" : false }'):
            res = await self._call(out)
            self.assertFalse(res.evidence[0]["ok"], out)
            self.assertEqual(res.evidence[0]["failure"], "error")

    async def test_json_that_is_not_an_ok_false_envelope_is_not_a_failure(self) -> None:
        for out in ('{"ok": true}', '{"note": "ok false"}', '["ok", false]', "{not json", '{"ok": "false"}', '{"ok": null}'):
            res = await self._call(out)
            self.assertTrue(res.evidence[0]["ok"], out)

    async def test_a_tool_failure_code_is_a_failure_whatever_the_text(self) -> None:
        for code in sorted(failcodes.FAILURE_CODES):
            res = await self._call(failing_tool("天气晴，二十度。", code))        # benign text, structured failure
            self.assertFalse(res.evidence[0]["ok"], code)
            self.assertEqual(res.evidence[0]["failure"], code)

    async def test_required_tool_with_a_failure_code_ends_the_job_with_the_tools_sentence(self) -> None:
        sentence = ZH.format("北京")
        res = await self._call(failing_tool(sentence, failcodes.NETWORK), required=["get_weather"])
        self.assertEqual((res.status, res.spoken_answer, res.evidence[0]["failure_text"]), ("failed", sentence, sentence))

    async def test_the_same_sentence_without_a_code_is_not_ended_as_a_failure(self) -> None:
        res = await self._call(ZH.format("北京"), required=["get_weather"])        # plain text only: the Worker does not read it
        self.assertTrue(res.evidence[0]["ok"])
        self.assertEqual(res.status, "completed")

    async def test_the_orchestrator_has_no_text_probe_for_failure(self) -> None:
        tree = ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_job")
        probes = [n for n in ast.walk(fn) if isinstance(n, ast.Compare) and any(isinstance(o, ast.In) for o in n.ops)
                  and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and any(isinstance(c, ast.Name) and c.id == "out" for c in n.comparators)]
        # the only text probes left in run_job pick the NUDGE sent to the model after a success; none of them decides ``failed``
        self.assertTrue(all("failed" not in ast.unparse(n) for n in probes))
        self.assertNotIn("failed = (", Path(orch.__file__).read_text(encoding="utf-8"))


class RealToolsReportTheirFailureInStructure(unittest.IsolatedAsyncioTestCase):
    async def test_busy_is_reported_by_all_three_tools(self) -> None:
        full = threading.BoundedSemaphore(1)
        full.acquire()
        weather.CACHE.clear()
        with mock.patch.object(inflight, "SHARED", full), mock.patch.object(web_search, "_inflight", full):
            for label, call in (("web_search", lambda: web_search.run({"query": "今天黄金价格"})), ("news", lambda: news.run({"topic": "tech", "lang": "zh"})),
                                ("weather", lambda: weather.run({"city": "北京"}))):
                toolmeta.take()
                weather.CACHE.clear()
                await call()
                self.assertEqual(toolmeta.take().get("failure"), failcodes.BUSY, label)

    async def test_network_failure_and_no_results_and_no_data(self) -> None:
        boom = urllib.error.URLError("down")
        toolmeta.take()
        with mock.patch.object(web_search, "_urlopen_unbounded", side_effect=boom):
            await web_search.run({"query": "今天黄金价格"})
        self.assertEqual(toolmeta.take().get("failure"), failcodes.NETWORK)
        with mock.patch.object(news, "_urlopen", side_effect=boom):
            await news.run({"topic": "tech", "lang": "zh"})
        self.assertEqual(toolmeta.take().get("failure"), failcodes.NETWORK)
        with mock.patch.object(web_search, "_bing", mock.AsyncMock(return_value=[])), mock.patch.object(web_search, "_ddg", mock.AsyncMock(return_value=[])):
            out = await web_search.run({"query": "今天黄金价格"})
        self.assertIn("没有搜到", out)
        self.assertEqual(toolmeta.take().get("failure"), failcodes.NO_RESULTS)
        weather.CACHE.clear()
        with mock.patch.object(weather, "_first_ok", mock.AsyncMock(return_value=(None, "", ["open-meteo:no_data", "wttr:no_data"]))):
            await weather.run({"city": "Rome", "lang": "en"})
        self.assertEqual(toolmeta.take().get("failure"), failcodes.NO_DATA)

    async def test_success_reports_no_failure(self) -> None:
        weather.CACHE.clear()
        data = {"tmin": 1, "tmax": 5, "desc": "cloudy", "precip": None, "now": 3}
        toolmeta.take()
        with mock.patch.object(weather, "_first_ok", mock.AsyncMock(return_value=(data, "open-meteo", []))):
            await weather.run({"city": "Rome", "lang": "en"})
        self.assertFalse(toolmeta.take().get("failure"))


class BusySentence(unittest.IsolatedAsyncioTestCase):
    def test_constants_match_the_whitelist_and_the_orchestrator_markers(self) -> None:
        for tool in ("get_weather", "get_news", "web_search"):
            self.assertTrue(ok(inflight.BUSY_ZH, {"tool": tool}, ""), tool)
            self.assertTrue(ok(inflight.BUSY_EN, {"tool": tool}, ""), tool)
        self.assertFalse(ok(inflight.BUSY_ZH, {"tool": "memory_note"}, ""))
        self.assertIn("同时在查的太多", inflight.BUSY_ZH)
        self.assertIn("too many lookups", inflight.BUSY_EN)

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
            self.assertEqual(await news.run({"topic": "tech", "lang": "zh"}), inflight.BUSY_ZH)
            self.assertEqual(await news.run({"topic": "tech", "lang": "en"}), inflight.BUSY_EN)
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
            self.assertIn("连不上", await news.run({"topic": "tech", "lang": "zh"}))

    async def test_busy_spoken_by_finish_job(self) -> None:
        for tool in ("get_weather", "get_news", "web_search"):
            self.assertEqual(_spoken(inflight.BUSY_ZH, tool, "今天黄金价格"), inflight.BUSY_ZH)

    def test_templates_are_bound_per_tool(self) -> None:
        for tool, pats in TOOL_FAILURE_TEMPLATES.items():
            self.assertTrue(any(p.fullmatch(inflight.BUSY_ZH) for p in pats), tool)


if __name__ == "__main__":
    unittest.main()
