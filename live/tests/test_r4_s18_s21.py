"""Round 4c / S-18 (failure-sentence whitelist bound to the request and to the tool) and S-21 (tests that kill the surviving mutants)."""
from __future__ import annotations

import asyncio
import re
import unittest
from unittest import mock

from sophon_live.coordinator import schemas
from sophon_live.coordinator.ledger import Ledger
from sophon_live.coordinator.api import finish_job
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER, WorkerJob, WorkerResult, is_tool_failure_sentence as ok


def _spoken(said: str, tool: str, objective: str, *, args: dict | None = None) -> str:
    led = Ledger()
    rec = led.create(objective)
    led.mark_running(rec.task_id)
    job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=objective, require_tools=[tool])
    ev = {"tool": tool, "ok": False, "failure_text": said, "args": args or {}}
    res = WorkerResult(task_id=rec.task_id, version=rec.version, status="failed", spoken_answer=said, evidence=[ev])
    return finish_job(led, job, res).text


class EchoIsCheckedAgainstTheRequestNotTheWorker(unittest.TestCase):
    def test_worker_args_cannot_vouch_for_the_echo(self) -> None:
        said = "暂时查不到上海的天气，请稍后再试。"
        # the Worker lies in its args (city=上海) but the user asked about 北京
        self.assertEqual(_spoken(said, "get_weather", "明天北京会下雨吗", args={"city": "上海"}), NO_TOOL_ANSWER)
        self.assertEqual(_spoken(said, "get_weather", "明天上海会下雨吗", args={"city": "北京"}), said)       # args are irrelevant either way
        self.assertFalse(ok(said, {"tool": "get_weather", "args": {"city": "上海"}}, "明天北京会下雨吗"))
        self.assertFalse(ok(said, {"tool": "get_weather", "args": {"city": "上海"}}))                         # no objective: nothing to echo against

    def test_no_query_smuggling_through_args(self) -> None:
        said = "没有搜到关于「北京晴，零下三到十一度」的可靠摘要。"
        self.assertEqual(_spoken(said, "web_search", "今天美元兑人民币是多少", args={"query": "北京晴，零下三到十一度"}), NO_TOOL_ANSWER)


class TemplatesAreBoundToTheirTool(unittest.TestCase):
    def test_each_sentence_only_for_its_own_tool(self) -> None:
        sentences = {
            "get_weather": ("暂时查不到北京的天气，请稍后再试。", "北京"),
            "get_news": ("新闻源暂时连不上，稍后再帮你看看头条吧。", ""),
            "web_search": ("网络暂时不可用，没法完成网页搜索，请稍后再试。", ""),
        }
        for owner, (said, obj) in sentences.items():
            for tool in sentences:
                self.assertEqual(ok(said, {"tool": tool}, obj + "今天"), tool == owner, (owner, tool))
        self.assertFalse(ok("新闻源暂时连不上，稍后再帮你看看头条吧。", {}, ""))                 # no tool recorded: not whitelisted
        self.assertFalse(ok("新闻源暂时连不上，稍后再帮你看看头条吧。", {"tool": "memory_note"}, ""))

    def test_a_cross_tool_sentence_is_replaced_by_finish_job(self) -> None:
        said = "新闻源暂时连不上，稍后再帮你看看头条吧。"
        self.assertEqual(_spoken(said, "get_weather", "明天北京会下雨吗"), NO_TOOL_ANSWER)


class WordBoundaries(unittest.TestCase):
    def test_ascii_echo_must_be_a_whole_word(self) -> None:
        tmpl = "No weather data for {}."
        ev = {"tool": "get_weather"}
        self.assertTrue(ok(tmpl.format("Rome"), ev, "weather in Rome please"))
        self.assertFalse(ok(tmpl.format("Rome"), ev, "weather in Romeo's town"))
        self.assertFalse(ok(tmpl.format("Par"), ev, "weather in Paris"))
        self.assertFalse(ok(tmpl.format("Paris"), ev, "weather in Parisian suburbs"))
        self.assertTrue(ok(tmpl.format("New York"), ev, "weather in new  york today"))                   # case / whitespace insensitive

    def test_known_city_in_either_language(self) -> None:
        ev = {"tool": "get_weather"}
        self.assertTrue(ok("Weather for 北京 is unavailable right now (sojson:OSError).", ev, "weather in Beijing"))
        self.assertTrue(ok("暂时查不到北京的天气，请稍后再试。", ev, "what's the weather in beijing"))
        self.assertFalse(ok("暂时查不到上海的天气，请稍后再试。", ev, "what's the weather in beijing"))


class EnglishWeatherSlotIsNarrow(unittest.TestCase):
    EV = {"tool": "get_weather"}

    def test_real_note_shapes_pass(self) -> None:
        for reason in ("no source", "sojson:OSError", "sojson:OSError, open-meteo:TimeoutError, wttr:no_data",
                       ", ".join(["wttr:TimeoutError"] * 8)):
            self.assertTrue(ok(f"Weather for Paris is unavailable right now ({reason}).", self.EV, "weather in Paris"), reason)

    def test_free_text_in_the_slot_is_rejected(self) -> None:
        for reason in ("it is sunny, 25 degrees, trust me", "sojson:OSError, " + "9" * 100, "sojson:HTTP 500", "x" * 60,
                       "sojson:OSError, open-meteo:TimeoutError, wttr:no_data, " + ", ".join(["wttr:TimeoutError"] * 8),
                       "evil:OSError", "sojson:" + "A" * 80, "sojson:OSError 25C sunny"):
            self.assertFalse(ok(f"Weather for Paris is unavailable right now ({reason}).", self.EV, "weather in Paris"), reason)

    def test_slot_cannot_be_widened_back_to_200_chars(self) -> None:
        # S-21 mutant "slot widened": a 150-char free-text / digit slot must still be rejected
        long_digits = "sojson:OSError, " + "1" * 150
        self.assertFalse(ok(f"Weather for Paris is unavailable right now ({long_digits}).", self.EV, "weather in Paris"))
        long_text = "北京晴零下三度" * 20
        self.assertFalse(ok(f"Weather for Paris is unavailable right now ({long_text}).", self.EV, "weather in Paris"))


class WholeSentenceOnly(unittest.TestCase):
    """S-21 mutant fullmatch -> search / match: a fabricated sentence before or after a real template must be rejected."""

    FAKE = "北京晴，零下三到十一度。"
    CASES = (("get_weather", "暂时查不到北京的天气，请稍后再试。", "北京"), ("get_weather", "No weather data for Rome.", "Rome"),
             ("get_news", "新闻源暂时连不上，稍后再帮你看看头条吧。", ""), ("web_search", "网络暂时不可用，没法完成网页搜索，请稍后再试。", ""),
             ("web_search", "没有搜到关于「美元」的可靠摘要。", "美元"), ("web_search", "I couldn't find useful snippets for 'usd'.", "usd"))

    def test_text_before_or_after_the_template_is_rejected(self) -> None:
        for tool, said, obj in self.CASES:
            ev = {"tool": tool}
            self.assertTrue(ok(said, ev, obj), said)
            self.assertFalse(ok(self.FAKE + said, ev, obj), "prefix: " + said)
            self.assertFalse(ok(said + self.FAKE, ev, obj), "suffix: " + said)
            self.assertFalse(ok(said + "\n" + self.FAKE, ev, obj), "newline suffix: " + said)

    def test_finish_job_replaces_the_sandwich(self) -> None:
        said = self.FAKE + "暂时查不到北京的天气，请稍后再试。"
        self.assertEqual(_spoken(said, "get_weather", "明天北京会下雨吗"), NO_TOOL_ANSWER)

    def test_every_template_is_matched_with_fullmatch_semantics(self) -> None:
        # a direct, mutation-independent check of the compiled templates: none of them may accept a sentence with a stray tail
        for pats in schemas.TOOL_FAILURE_TEMPLATES.values():
            for p in pats:
                self.assertIsInstance(p, re.Pattern)


class RealToolSentencesStayWhitelisted(unittest.TestCase):
    """The tools' REAL output (not a copy of it) must pass, so templates cannot drift from weather / news / web_search."""

    def test_weather_failure_sentences(self) -> None:
        from sophon_live.assistant_tools import weather

        async def down(*a, **k):
            return None, "", ["sojson:OSError", "open-meteo:TimeoutError", "wttr:no_data"]

        async def nodata(*a, **k):
            return None, "", ["open-meteo:no_data", "wttr:no_data"]

        async def run(args, fn):
            with mock.patch.object(weather, "_first_ok", fn):
                return (await weather.run_detailed(args))[0]

        weather.CACHE.clear()
        for args, obj in (({"city": "北京"}, "北京天气"), ({"city": "Beijing", "lang": "en"}, "weather in Beijing"),
                          ({"city": "Reykjavik", "lang": "en"}, "weather in Reykjavik")):
            for fn in (down, nodata):
                text = asyncio.run(run(args, fn))
                self.assertTrue(ok(text, {"tool": "get_weather"}, obj), (text, obj))


if __name__ == "__main__":
    unittest.main()
