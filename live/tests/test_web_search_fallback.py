"""体验版 v1 / 5: web search works from the GPU box (Bing China) and fails honestly within the budget."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import web_search

# trimmed from a real cn.bing.com result page (2026-11-18): the structure the parser relies on
BING_HTML = """
<html><body><ol id="b_results">
<li class="b_algo" data-id iid=SERP.5335><link rel="stylesheet" href="/rp/x.css" type="text/css"/><div class="b_tpcn"><cite>https://www.boc.cn</cite></div>
<h2><a target="_blank" href="https://www.boc.cn/sourcedb/whpj/">中国银行网站_金融市场_<strong>外汇牌价</strong></a></h2>
<div class="b_caption"><p class="b_lineclamp2">1 天前&ensp;&#0183;&ensp;中国银行外汇牌价：美元 713.20，仅供参考。</p></div></li>
<li class="b_algo" data-id iid=SERP.5336><h2><a href="https://finance.sina.com.cn/money/forex/hq/USDCNY.shtml">美元兑人民币 (USDCNY)_新浪财经</a></h2>
<div class="b_caption"><p class="b_lineclamp3">新浪财经外汇频道提供美元兑人民币汇率、走势图。</p></div></li>
<li class="b_algo"><h2><a href="https://example.com">只有标题的结果</a></h2></li>
<li class="b_ad"><h2><a href="https://ad.example.com">广告</a></h2></li>
</ol></body></html>
""".encode()

DDG_INSTANT = json.dumps({"AbstractText": "Python is a language.", "Heading": "Python", "RelatedTopics": []}).encode()


def _fetch(routes: dict[str, object], delays: dict[str, float] | None = None):
    """Fake _fetch_bytes: by URL substring → bytes / Exception; optional delay."""
    delays = delays or {}

    async def fetch(url: str, timeout: float = 5.5) -> bytes:
        fetch.calls.append(url)
        for key, val in routes.items():
            if key in url:
                if delays.get(key):
                    await asyncio.sleep(delays[key])
                if isinstance(val, Exception):
                    raise val
                return val  # type: ignore[return-value]
        raise AssertionError(url)

    fetch.calls = []
    return fetch


class BingParseTests(unittest.TestCase):
    def test_titles_and_snippets_of_organic_results_only(self) -> None:
        out = web_search._snippets_from_bing(BING_HTML.decode(), 5)
        self.assertEqual(len(out), 3)                                  # the ad (b_ad) is not an organic result
        self.assertEqual(out[0], "中国银行网站_金融市场_外汇牌价 — 中国银行外汇牌价：美元 713.20，仅供参考。")   # tags and "1 天前 ·" stripped
        self.assertTrue(out[1].startswith("美元兑人民币 (USDCNY)_新浪财经 — 新浪财经外汇频道"))
        self.assertEqual(out[2], "只有标题的结果")

    def test_limit_and_garbage(self) -> None:
        self.assertEqual(len(web_search._snippets_from_bing(BING_HTML.decode(), 1)), 1)
        self.assertEqual(web_search._snippets_from_bing("<html>captcha</html>", 3), [])


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_query_is_answered_by_bing_while_duckduckgo_is_unreachable(self) -> None:
        fetch = _fetch({"cn.bing.com": BING_HTML, "duckduckgo": OSError("unreachable")})
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search.run({"query": "今天美元兑人民币汇率", "max_results": 2})
        self.assertIn("外汇牌价", out)
        self.assertIn("新浪财经", out)
        self.assertNotIn("网络暂时不可用", out)
        self.assertTrue(any("ensearch=0" in u for u in fetch.calls))

    async def test_chinese_query_prefers_bing_even_when_duckduckgo_answers_first(self) -> None:
        fetch = _fetch({"cn.bing.com": BING_HTML, "api.duckduckgo.com": DDG_INSTANT, "lite.duckduckgo.com": b""}, {"cn.bing.com": 0.2})
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search.run({"query": "汇率", "max_results": 1})
        self.assertIn("外汇牌价", out)
        self.assertNotIn("Python", out)

    async def test_english_query_prefers_duckduckgo_and_falls_back_to_bing(self) -> None:
        ok = _fetch({"api.duckduckgo.com": DDG_INSTANT, "lite.duckduckgo.com": b"", "cn.bing.com": BING_HTML})
        with mock.patch.object(web_search, "_fetch_bytes", ok):
            out = await web_search.run({"query": "python language", "max_results": 1})
        self.assertIn("Python is a language", out)
        down = _fetch({"duckduckgo": OSError("unreachable"), "cn.bing.com": BING_HTML})
        with mock.patch.object(web_search, "_fetch_bytes", down):
            out = await web_search.run({"query": "usd cny rate", "max_results": 1})
        self.assertIn("外汇牌价", out)
        self.assertTrue(any("ensearch=1" in u for u in down.calls))

    async def test_everything_unreachable_fails_honestly(self) -> None:
        fetch = _fetch({"cn.bing.com": OSError("down"), "duckduckgo": OSError("down")})
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search.run({"query": "今天美元兑人民币汇率"})
        self.assertIn("网络暂时不可用", out)
        self.assertFalse(any(ch.isdigit() for ch in out))

    async def test_hanging_providers_fail_within_the_budget(self) -> None:
        fetch = _fetch({"cn.bing.com": BING_HTML, "duckduckgo": DDG_INSTANT}, {"cn.bing.com": 30.0, "duckduckgo": 30.0})
        with mock.patch.dict(os.environ, {"SOPHON_SEARCH_BUDGET": "1.2"}), mock.patch.object(web_search, "_fetch_bytes", fetch):
            t0 = time.monotonic()
            out = await web_search.run({"query": "今天美元兑人民币汇率"})
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertIn("网络暂时不可用", out)

    async def test_default_budget_is_six_seconds(self) -> None:
        os.environ.pop("SOPHON_SEARCH_BUDGET", None)
        self.assertEqual(web_search._budget_s(), 6.0)

    async def test_a_captcha_page_is_no_results_not_an_outage(self) -> None:
        fetch = _fetch({"cn.bing.com": b"<html>verify</html>", "duckduckgo": OSError("down")})
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search.run({"query": "汇率"})
        self.assertIn("没有搜到", out)


if __name__ == "__main__":
    unittest.main()


class BingHedgeTests(unittest.IsolatedAsyncioTestCase):
    def _flaky(self, plan: list):
        """Attempts at the Bing URL behave as ``plan`` says in order: 'ok' / 'fail' / 'hang'."""
        calls: list[float] = []

        async def fetch(url: str, timeout: float = 5.5) -> bytes:
            if "cn.bing.com" not in url:
                raise OSError("down")
            n = len(calls); calls.append(time.monotonic())
            what = plan[n] if n < len(plan) else "hang"
            if what == "fail":
                raise OSError("reset")
            if what == "hang":
                await asyncio.sleep(30)
            return BING_HTML
        fetch.calls = calls
        return fetch

    async def test_a_failed_attempt_is_retried_at_once(self) -> None:
        fetch = self._flaky(["fail", "ok"])
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            t0 = time.monotonic()
            out = await web_search.run({"query": "汇率"})
        self.assertIn("外汇牌价", out)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(len(fetch.calls), 2)

    async def test_a_stalled_attempt_is_hedged_after_the_hedge_delay(self) -> None:
        fetch = self._flaky(["hang", "ok"])
        with mock.patch.object(web_search, "_BING_HEDGE_S", 0.3), mock.patch.object(web_search, "_fetch_bytes", fetch):
            t0 = time.monotonic()
            out = await web_search.run({"query": "汇率"})
        self.assertIn("外汇牌价", out)
        self.assertLess(time.monotonic() - t0, 1.5)

    async def test_three_failures_are_an_honest_outage(self) -> None:
        fetch = self._flaky(["fail", "fail", "fail"])
        with mock.patch.object(web_search, "_fetch_bytes", fetch):
            out = await web_search.run({"query": "汇率"})
        self.assertIn("网络暂时不可用", out)
        self.assertEqual(len(fetch.calls), 3)

    def test_gzip_pages_are_decoded(self) -> None:
        import gzip as _gz
        import io

        class Resp(io.BytesIO):
            headers = {"Content-Encoding": "gzip"}
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch.object(web_search.urllib.request, "urlopen", lambda req, timeout=0: Resp(_gz.compress(BING_HTML))):
            self.assertEqual(web_search._urlopen("https://cn.bing.com/search?q=x", 1.0), BING_HTML)
