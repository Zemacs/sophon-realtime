"""Round 4h: (I-20) grounding is provenance only, no placeholder / free-text word lists; (news) the topic is a closed-enum tool parameter the
orchestrator model fills, the code fetches by that parameter and never classifies the user's words.

User principle (00:27): the model judges, code only enforces (provenance, structured codes, schema enums)."""
from __future__ import annotations

import ast
import asyncio
import json
import sys
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding  # noqa: E402
from sophon_live.assistant_tools import news  # noqa: E402
from sophon_live.assistant_tools import orchestrator as orch  # noqa: E402
from sophon_live.assistant_tools.registry import TOOL_SCHEMAS  # noqa: E402
from sophon_live.coordinator.schemas import WorkerJob  # noqa: E402
from tests.test_orch_worker import _say, _tool_call  # noqa: E402
from tests.test_orchestrator import _FakeResp, _FakeSession  # noqa: E402


def _bad(args: dict, heard: list[str], seen: list[str] | None = None) -> list[tuple[str, str, str]]:
    return grounding.check("some_tool", args, heard, seen or [])


class GroundingIsProvenanceOnly(unittest.TestCase):
    def test_placeholder_looking_values_the_caller_actually_said_are_allowed(self) -> None:
        self.assertEqual(_bad({"zip": "12345"}, ["my zip code is 12345"]), [])
        self.assertEqual(_bad({"first_name": "John", "last_name": "Doe"}, ["my name is John Doe"]), [])
        self.assertEqual(_bad({"email": "tester@example.com"}, ["it is tester@example.com"]), [])
        self.assertEqual(_bad({"phone": "5551234567"}, ["call me on 5551234567"]), [])
        self.assertEqual(_bad({"domain": "example.com"}, ["the site is example.com"]), [])

    def test_the_same_values_invented_by_the_model_are_blocked(self) -> None:
        for args in ({"zip": "12345"}, {"first_name": "John", "last_name": "Doe"}, {"email": "tester@example.com"}, {"phone": "5551234567"},
                     {"domain": "example.com"}):
            got = _bad(args, ["hello, I would like some help"])
            self.assertTrue(got, args)
            self.assertTrue(all("not said by the caller" in why for _, _, why in got), got)

    def test_any_other_invented_value_is_blocked_the_same_way(self) -> None:
        self.assertTrue(_bad({"order_id": "A98231"}, ["my order is about a late parcel"]))
        self.assertFalse(_bad({"order_id": "A98231"}, ["order A98231 please"]))

    def test_a_tool_result_counts_as_provenance(self) -> None:
        self.assertEqual(_bad({"zip": "90210"}, ["look it up"], ["{'zip': '90210', 'city': 'Beverly Hills'}"]), [])
        self.assertTrue(_bad({"zip": "90210"}, ["look it up"], ["{'zip': '10001'}"]))

    def test_spoken_forms_of_a_said_value_still_count(self) -> None:
        self.assertEqual(_bad({"email": "jon@acme.com"}, ["jon at acme dot com"]), [])
        self.assertEqual(_bad({"zip": "19122"}, ["one nine one two two"]), [])

    def test_no_argument_key_is_exempt_by_name(self) -> None:
        """4g and before: values under summary / reason / query / note ... were never checked. A short model-invented value there is now caught."""
        for key in ("summary", "reason", "query", "note", "notes", "message", "description", "comment", "content"):
            self.assertTrue(_bad({key: "Quarterly Alpha"}, ["hello"]), key)
            self.assertEqual(_bad({key: "Quarterly Alpha"}, ["please file Quarterly Alpha"]), [], key)

    def test_long_text_is_no_longer_out_of_scope(self) -> None:
        """Round 4j / I-23 reverses the 4h rule "more than 3 words is out of scope": the old expectation (an invented long summary passes
        when the caller only said "hello") is replaced on purpose. Long text now needs token provenance; see test_r4j_grounding_i23.py."""
        self.assertTrue(_bad({"summary": "the caller wants a refund for the broken blender they bought last week"}, ["hello"]))
        # Round 4k: the token-overlap ratio is gone; a composed value needs the model's evidence span (see test_r4k_evidence.py)
        said = ["I want a refund for the blender I bought last week it is broken"]
        ev = [{"argument": "summary", "span": "I want a refund for the blender", "value_comes_from_span": True}]
        args = {"summary": "the caller wants a refund for the broken blender they bought last week"}
        self.assertTrue(grounding.check("t", args, said, [], evidence=ev))                          # N-2: the model's own evidence is not enough without the independent check
        self.assertEqual(grounding.check("t", args, said, [], evidence=ev, judge=lambda *a: True), [])
        self.assertTrue(grounding.check("t", args, said, [], evidence=ev, judge=lambda *a: False))

    def test_the_old_mode_switch_has_no_effect(self) -> None:
        """SOPHON_LIVE_GROUND_MODE=placeholder used to skip every check except the placeholder lists. The lists are gone and so is the mode."""
        with mock.patch.dict("os.environ", {"SOPHON_LIVE_GROUND_MODE": "placeholder"}):
            self.assertTrue(_bad({"order_id": "A98231"}, ["hello"]))
            self.assertTrue(_bad({"zip": "12345"}, ["hello"]))

    def test_the_lists_are_gone(self) -> None:
        for name in ("PLACEHOLDER_NAMES", "PLACEHOLDER_VALUES", "PLACEHOLDER_DOMAINS", "FREE_TEXT_KEYS"):
            self.assertFalse(hasattr(grounding, name), name)
        src = Path(grounding.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        strings = {n.value.lower() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        for needle in ("john", "doe", "jane", "example.com", "12345", "test.com", "summary", "SOPHON_LIVE_GROUND_MODE".lower()):
            self.assertNotIn(needle, strings, needle)
        self.assertNotIn("placeholder value", src)


def _rss(*titles: str) -> bytes:
    return ("<?xml version='1.0'?><rss><channel>" + "".join(f"<item><title>{t}</title></item>" for t in titles) + "</channel></rss>").encode()


def _run(coro):
    return asyncio.run(coro)


class NewsFetchesByTheTopicParameter(unittest.TestCase):
    def _calls(self, args: dict, bodies):
        urls: list[str] = []

        async def fetch(url: str) -> bytes:
            urls.append(url)
            b = bodies(url) if callable(bodies) else bodies
            if isinstance(b, Exception):
                raise b
            return b

        with mock.patch.object(news, "_fetch_rss", new=fetch):
            out = _run(news.run(args))
        return out, urls

    def test_the_schema_offers_exactly_the_categories_the_code_has(self) -> None:
        topic = next(t for t in TOOL_SCHEMAS if t["name"] == "get_news")["parameters"]["properties"]["topic"]
        self.assertEqual(topic["enum"], list(news.CATEGORIES))
        self.assertEqual(set(news.CATEGORIES), set(news._ZH_FEEDS), set(news._LABELS))
        self.assertEqual(set(news._ZH_FEEDS), set(news._LABELS))
        self.assertIn("web_search", next(t for t in TOOL_SCHEMAS if t["name"] == "get_news")["description"])      # specific subjects go to search
        for c in news.CATEGORIES:
            self.assertIn(c, topic["description"])

    def test_no_word_tables_or_classifiers_remain(self) -> None:
        for name in ("_bucket", "_filter_topic", "_TECH_KEYS", "_FIN_KEYS", "_WORLD_KEYS", "_is_zh"):
            self.assertFalse(hasattr(news, name), name)
        src = Path(news.__file__).read_text(encoding="utf-8")
        for w in ("互联网", "人工智能", "芯片", "股市", "金融", "cyber", "数码", "海外", "require_hit"):
            self.assertNotIn(w, src, w)

    def test_zh_fetches_the_categorys_own_feed_and_takes_the_headlines_unfiltered(self) -> None:
        for cat in news.CATEGORIES:
            out, urls = self._calls({"topic": cat, "lang": "zh", "max_items": 3}, _rss("明天多云转晴", "某地举办马拉松", "新品咖啡上市"))
            self.assertEqual(urls, [news._ZH_FEEDS[cat][0]], cat)
            self.assertIn("明天多云转晴", out)          # nothing here "matches" the topic: no keyword filter decides what is kept
            self.assertIn(f"关于{news._LABELS[cat][0]}，", out)

    def test_a_category_has_no_cross_feed_fallback_it_would_need_a_filter_for(self) -> None:
        out, urls = self._calls({"topic": "tech", "lang": "zh"}, lambda u: OSError("down") if "people.com.cn" in u else _rss())
        self.assertEqual(urls[0], "http://www.people.com.cn/rss/it.xml")
        self.assertEqual([u for u in urls if "people.com.cn" in u], ["http://www.people.com.cn/rss/it.xml"])      # not finance.xml / politics.xml
        self.assertIn("连不上", out)

    def test_zh_falls_back_to_the_google_search_for_the_categorys_label(self) -> None:
        out, urls = self._calls({"topic": "finance", "lang": "zh"}, lambda u: OSError("x") if "people.com.cn" in u else _rss("财经标题一", "财经标题二"))
        self.assertIn("财经标题一", out)
        self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlparse(urls[-1]).query)["q"], ["财经"])

    def test_en_asks_google_for_the_category_label_not_for_the_users_words(self) -> None:
        for cat in news.CATEGORIES:
            out, urls = self._calls({"topic": cat, "lang": "en"}, _rss("Headline One", "Headline Two"))
            self.assertEqual(len(urls), 1)
            self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)["q"], [news._LABELS[cat][1]], cat)
            self.assertIn(f"Headlines on {news._LABELS[cat][1]}: Headline One", out)

    def test_a_value_outside_the_enum_is_a_missing_topic_and_nothing_is_fetched(self) -> None:
        for bad in ("科技", "NVIDIA", "特朗普", "technology", "", "  ", None, "TECH"):
            out, urls = self._calls({"topic": bad, "lang": "zh"}, AssertionError("must not fetch"))
            self.assertEqual(urls, [], bad)
            self.assertIn("哪方面的新闻", out)
        out, urls = self._calls({"lang": "en"}, AssertionError("must not fetch"))
        self.assertEqual((urls, out), ([], "Please tell me which news topic you want."))

    def test_lang_defaults_to_zh_when_absent(self) -> None:
        out, urls = self._calls({"topic": "tech"}, _rss("某某标题"))
        self.assertIn("关于科技", out)


class OrchestratorEnforcesTheEnumAndSuppliesTheLanguage(unittest.IsolatedAsyncioTestCase):
    async def _run(self, args: str, *, language: str = "zh", tool_out: str = "ok"):
        sess = _FakeSession([_FakeResp(200, _tool_call("get_news", args)), _FakeResp(200, _say("好的。"))])
        rt = mock.AsyncMock(return_value=tool_out)
        job = WorkerJob(task_id="t1", version=1, objective="有什么科技新闻", language=language, require_tools=["get_news"])
        with mock.patch.object(orch.aiohttp, "ClientSession", return_value=sess), mock.patch.object(orch, "run_tool", new=rt):
            res = await orch.run_job(job)
        return res, rt

    async def test_a_topic_outside_the_enum_is_needs_input_and_no_tool_call(self) -> None:
        for bad in ('{"topic":"科技"}', '{"topic":"NVIDIA"}', '{"topic":""}', "{}"):
            res, rt = await self._run(bad)
            self.assertEqual((res.status, res.needs_slots), ("needs_input", ["topic"]), bad)
            rt.assert_not_awaited()

    async def test_a_category_is_passed_through_and_the_job_language_fills_a_missing_lang(self) -> None:
        res, rt = await self._run('{"topic":"tech"}', language="en")
        self.assertEqual(rt.await_args.args[1], {"topic": "tech", "lang": "en"})
        res, rt = await self._run('{"topic":"tech","lang":"zh"}', language="en")
        self.assertEqual(rt.await_args.args[1]["lang"], "zh")                  # the model's own value wins
        res, rt = await self._run('{"topic":"world","lang":"fr"}', language="zh")
        self.assertEqual(rt.await_args.args[1]["lang"], "zh")

    async def test_other_tools_enum_params_are_not_over_enforced(self) -> None:
        self.assertEqual(orch._missing_required("get_weather", {"city": "北京", "when": "后天"}), [])      # optional enum: left to the tool
        self.assertEqual(orch._missing_required("memory_note", {"op": "delete"}), ["op"])                  # required enum: enforced
        self.assertEqual(orch._missing_required("get_news", {"topic": "world"}), [])

    def test_the_prompt_tells_the_model_to_choose_the_category(self) -> None:
        self.assertIn("get_news takes ``topic`` from its closed list", orch.SYS)
        self.assertIn("web_search", orch.SYS)


if __name__ == "__main__":
    unittest.main()
