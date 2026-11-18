"""Round 2 / A: two weather sources raced with per-source timeouts; honest failure, no fake success."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

_spec = importlib.util.spec_from_file_location(
    "sophon_live.assistant_tools.weather", _LIVE_ROOT / "sophon_live" / "assistant_tools" / "weather.py"
)
weather = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weather)  # type: ignore[union-attr]

OM = {
    "current": {"temperature_2m": 3.4},
    "daily": {
        "weathercode": [3, 63],
        "temperature_2m_max": [8.2, 11.7],
        "temperature_2m_min": [1.1, 4.4],
        "precipitation_probability_max": [10, 80],
    },
}
WTTR = {
    "current_condition": [{"temp_C": "5"}],
    "weather": [
        {"mintempC": "2", "maxtempC": "9", "hourly": [{"chanceofrain": "20", "lang_zh": [{"value": "多云"}]}]},
        {"mintempC": "3", "maxtempC": "10", "hourly": [{"chanceofrain": "40", "lang_zh": [{"value": "小雨"}]}]},
    ],
}
SOJSON = {
    "status": 200,
    "data": {"wendu": "4", "forecast": [
        {"high": "高温 9℃", "low": "低温 -1℃", "type": "多云"},
        {"high": "高温 12℃", "low": "低温 2℃", "type": "小雨"},
    ]},
}
ENV = {"SOPHON_WEATHER_TIMEOUT": "0.6", "SOPHON_WEATHER_ROUNDS": "1", "SOPHON_WEATHER_CACHE_TTL": "0"}  # cache off unless a test turns it on


def _router(om=None, wttr=None, om_delay=0.0, wttr_delay=0.0, sojson=None, sj_delay=0.0):
    """A fake _fetch_json: routes by URL, optional delay, an Exception instance is raised."""

    async def fetch(url, timeout=5.0):  # noqa: ANN001
        if "sojson" in url:
            src, delay, val = "sojson", sj_delay, sojson
        else:
            src, delay, val = ("om", om_delay, om) if "open-meteo" in url else ("wttr", wttr_delay, wttr)
        fetch.calls.append(src)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(val, Exception):
            raise val
        return val

    fetch.calls = []
    return fetch


class WeatherSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        weather.STATS.clear()
        weather.CACHE.clear()

    async def test_open_meteo_parse_and_wmo_text(self) -> None:
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=TimeoutError("x"))):
            text, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(text, "北京明天中雨，气温4到12度，降水概率80%。")
        self.assertEqual(meta["source"], "open-meteo")

    async def test_today_reports_current_temperature(self) -> None:
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=TimeoutError("x"))):
            text = await weather.run({"city": "北京", "when": "today"})
        self.assertEqual(text, "北京今天阴，气温1到8度，降水概率10%。现在3度。")

    async def test_slow_primary_loses_to_fast_fallback(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, om_delay=5.0)  # open-meteo would take 5 s; the 0.6 s budget cuts it
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            t0 = time.monotonic()
            text, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(meta["source"], "wttr")
        self.assertIn("小雨", text)

    async def test_first_success_wins_and_cancels_the_other(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, wttr_delay=5.0)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            t0 = time.monotonic()
            _, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertLess(time.monotonic() - t0, 0.5)  # did not wait for the slow one
        self.assertEqual(meta["source"], "open-meteo")

    async def test_all_sources_down_is_an_honest_failure_without_numbers(self) -> None:
        fetch = _router(om=OSError("down"), wttr=TimeoutError("slow"))
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            text, meta = await weather.run_detailed({"city": "上海", "when": "tomorrow"})
        self.assertEqual(text, "暂时查不到上海的天气，请稍后再试。")
        self.assertIsNone(meta["source"])
        self.assertFalse(any(ch.isdigit() for ch in text))
        self.assertEqual(weather.STATS["open-meteo"]["fail"], 1)
        self.assertEqual(weather.STATS["wttr"]["fail"], 1)

    async def test_both_hang_fail_within_the_per_source_budget(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, om_delay=9.0, wttr_delay=9.0)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            t0 = time.monotonic()
            text = await weather.run({"city": "上海"})
        self.assertLess(time.monotonic() - t0, 2.5)
        self.assertIn("暂时查不到", text)

    async def test_garbled_payloads_do_not_become_success(self) -> None:
        fetch = _router(om={"daily": {"weathercode": [], "temperature_2m_max": []}}, wttr={"weather": [{"hourly": []}]})
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            text = await weather.run({"city": "北京"})
        self.assertNotIn("气温", text)

    async def test_second_round_recovers_from_a_transient_failure(self) -> None:
        calls = {"n": 0}

        async def fetch(url, timeout=5.0):  # noqa: ANN001
            if "sojson" in url:
                raise OSError("sojson down")
            if "open-meteo" in url:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("blip")
                return OM
            raise TimeoutError("wttr down")

        with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_ROUNDS": "2"}), mock.patch.object(weather, "_fetch_json", fetch):
            text, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "open-meteo")
        self.assertIn("中雨", text)

    async def test_unknown_city_is_geocoded_first(self) -> None:
        urls: list[str] = []

        async def fetch(url, timeout=5.0):  # noqa: ANN001
            urls.append(url)
            if "geocoding" in url:
                return {"results": [{"latitude": 30.5, "longitude": 114.3}]}
            if "open-meteo" in url:
                return OM
            raise TimeoutError

        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            text = await weather.run({"city": "襄阳", "when": "tomorrow"})
        self.assertTrue(urls[0].startswith("https://geocoding-api.open-meteo.com") or "geocoding" in " ".join(urls))
        self.assertIn("襄阳明天中雨", text)
        self.assertTrue(any("latitude=30.5" in u for u in urls))

    async def test_city_nobody_knows_is_not_an_outage(self) -> None:
        fetch = _router(om={"results": []}, wttr={"weather": []})
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            text = await weather.run({"city": "Nowhereville", "lang": "en"})
        self.assertEqual(text, "No weather data for Nowhereville.")

    async def test_cn_hosted_source_answers_when_the_overseas_ones_hang(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, om_delay=9.0, wttr_delay=9.0, sojson=SOJSON)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            t0 = time.monotonic()
            text, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(meta["source"], "sojson")
        # this feed has no rain probability: say nothing about it rather than invent one
        self.assertEqual(text, "北京明天小雨，气温2到12度。")

    async def test_sojson_is_skipped_for_cities_it_has_no_code_for(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            await weather.run({"city": "东京", "when": "tomorrow"})
        self.assertNotIn("sojson", fetch.calls)

    async def test_sojson_error_status_is_not_success(self) -> None:
        self.assertIsNone(weather._normalize_sojson({"status": 304, "data": {}}, 1, zh=True))
        self.assertIsNone(weather._normalize_sojson({"status": 200, "data": {"forecast": [{"high": "x"}]}}, 0, zh=True))

    async def test_fresh_cache_answers_instantly_without_touching_the_network(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "600"}
        fetch = _router(om=OM, wttr=WTTR)
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", fetch):
            first, m1 = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
            calls = len(fetch.calls)
            second, m2 = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(first, second)
        self.assertEqual(len(fetch.calls), calls)  # nothing fetched the second time
        self.assertIn(m1["source"], ("open-meteo", "wttr"))  # both answer at once: either may win the race
        self.assertEqual((m2["source"], m2["orig_source"]), ("cache", m1["source"]))

    async def test_cache_is_per_city_and_day(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "600"}
        fetch = _router(om=OM, wttr=WTTR)
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", fetch):
            await weather.run({"city": "北京", "when": "tomorrow"})
            _, m = await weather.run_detailed({"city": "上海", "when": "tomorrow"})
            _, m2 = await weather.run_detailed({"city": "北京", "when": "today"})
        self.assertNotEqual(m["source"], "cache")
        self.assertNotEqual(m2["source"], "cache")

    async def test_blackout_falls_back_to_old_data_but_says_so(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "60"}
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=WTTR)):
            await weather.run({"city": "北京", "when": "today"})
        key = next(iter(weather.CACHE))
        ts, data, src = weather.CACHE[key]
        weather.CACHE[key] = (ts - 15 * 60, data, src)  # 15 min old: past the fresh TTL
        down = _router(om=OSError("down"), wttr=TimeoutError("down"), sojson=OSError("down"))
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", down):
            text, meta = await weather.run_detailed({"city": "北京", "when": "today"})
        self.assertEqual(meta["source"], "cache-stale")
        self.assertTrue(text.startswith("网络不太好，这是15分钟前查到的："))
        self.assertNotIn("现在", text)  # an old reading is not "now"

    async def test_ancient_or_missing_cache_is_not_used(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "60"}
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=WTTR)):
            await weather.run({"city": "北京", "when": "today"})
        key = next(iter(weather.CACHE))
        ts, data, src = weather.CACHE[key]
        weather.CACHE[key] = (ts - 3 * 3600, data, src)
        down = _router(om=OSError("down"), wttr=TimeoutError("down"), sojson=OSError("down"))
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", down):
            text = await weather.run({"city": "北京", "when": "today"})
        self.assertEqual(text, "暂时查不到北京的天气，请稍后再试。")
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", down):
            self.assertEqual(await weather.run({"city": "上海", "when": "today"}), "暂时查不到上海的天气，请稍后再试。")

    async def test_failures_are_not_cached(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "600"}
        down = _router(om=OSError("down"), wttr=TimeoutError("down"), sojson=OSError("down"))
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", down):
            await weather.run({"city": "北京", "when": "today"})
        self.assertEqual(weather.CACHE, {})

    async def test_sources_can_be_selected_for_ab_runs(self) -> None:
        fetch = _router(om=OM, wttr=WTTR)
        with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_SOURCES": "wttr"}), mock.patch.object(weather, "_fetch_json", fetch):
            _, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "wttr")
        self.assertEqual(set(fetch.calls), {"wttr"})


class PriorityTests(unittest.IsolatedAsyncioTestCase):
    """体验版 v1 / 2: the answer is chosen by source priority, not by who arrives first."""

    def setUp(self) -> None:
        weather.STATS.clear()
        weather.CACHE.clear()

    async def test_slower_top_priority_source_still_wins_over_a_faster_lower_one(self) -> None:
        # sojson (priority 1) needs 0.3 s, open-meteo and wttr answer at once: sojson's numbers are the answer
        fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON, sj_delay=0.3)
        with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_TIMEOUT": "2"}), mock.patch.object(weather, "_fetch_json", fetch):
            text, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "sojson")
        self.assertEqual(text, "北京明天小雨，气温2到12度。")

    async def test_same_question_gets_the_same_answer_every_time(self) -> None:
        answers = set()
        for delays in ((0.0, 0.2, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.2), (0.1, 0.0, 0.1)):
            weather.CACHE.clear()
            fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON, om_delay=delays[0], wttr_delay=delays[1], sj_delay=delays[2])
            with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_TIMEOUT": "2"}), mock.patch.object(weather, "_fetch_json", fetch):
                answers.add(await weather.run({"city": "北京", "when": "tomorrow"}))
        self.assertEqual(len(answers), 1, answers)

    async def test_fails_over_in_priority_order_when_the_top_source_is_down(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, sojson=OSError("down"))
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            _, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "open-meteo")     # not wttr: open-meteo ranks above it
        self.assertEqual(meta["notes"], ["sojson:network"])

    async def test_top_source_that_hangs_is_cut_by_its_timeout_then_the_next_one_answers(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON, sj_delay=9.0)
        with mock.patch.dict(os.environ, ENV), mock.patch.object(weather, "_fetch_json", fetch):
            t0 = time.monotonic()
            _, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(meta["source"], "open-meteo")

    async def test_non_mainland_city_prefers_open_meteo_over_wttr(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON, om_delay=0.2)
        with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_TIMEOUT": "2"}), mock.patch.object(weather, "_fetch_json", fetch):
            _, meta = await weather.run_detailed({"city": "东京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "open-meteo")
        self.assertNotIn("sojson", fetch.calls)

    async def test_source_order_is_configurable(self) -> None:
        fetch = _router(om=OM, wttr=WTTR, sojson=SOJSON)
        with mock.patch.dict(os.environ, {**ENV, "SOPHON_WEATHER_SOURCES": "wttr,sojson"}), mock.patch.object(weather, "_fetch_json", fetch):
            _, meta = await weather.run_detailed({"city": "北京", "when": "tomorrow"})
        self.assertEqual(meta["source"], "wttr")

    async def test_cache_is_keyed_by_city_and_target_date(self) -> None:
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "600"}
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=WTTR, sojson=SOJSON)):
            await weather.run({"city": "北京", "when": "tomorrow"})
            await weather.run({"city": "北京", "when": "today"})
        import datetime as dt
        today = dt.date.today()
        self.assertEqual(
            sorted(k[:2] for k in weather.CACHE),
            sorted([("北京", today.isoformat()), ("北京", (today + dt.timedelta(days=1)).isoformat())]),
        )

    async def test_cache_default_ttl_is_long_enough_for_a_conversation(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOPHON_WEATHER_CACHE_TTL", None)
            self.assertGreaterEqual(weather._cache_ttl(), 1800.0)

    async def test_source_and_cache_origin_reach_the_trace_side_channel(self) -> None:
        from sophon_live.assistant_tools import toolmeta
        env = {**ENV, "SOPHON_WEATHER_CACHE_TTL": "600"}
        with mock.patch.dict(os.environ, env), mock.patch.object(weather, "_fetch_json", _router(om=OM, wttr=WTTR, sojson=SOJSON)):
            toolmeta.take()
            await weather.run({"city": "北京", "when": "tomorrow"})
            m1 = toolmeta.take()
            await weather.run({"city": "北京", "when": "tomorrow"})
            m2 = toolmeta.take()
        self.assertEqual(m1["source"], "sojson")
        self.assertEqual((m2["source"], m2["orig_source"]), ("cache", "sojson"))
        self.assertIn("age_s", m2)


if __name__ == "__main__":
    unittest.main()
