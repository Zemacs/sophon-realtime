"""City weather from independent sources, raced: Open-Meteo, sojson (CN-hosted, mainland cities) and wttr.in.

All of them are flaky from the GPU box (measured: each answers only some of the time, 1-40 s), so they run
concurrently with a short per-source timeout; the first source that returns real data wins and the rest
are cancelled. If every source fails or times out the caller gets an honest failure sentence, never
invented numbers. ``SOPHON_WEATHER_SOURCES`` (default ``open-meteo,sojson,wttr``) selects sources for A/B runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

log = logging.getLogger("sophon_live.weather")

CITIES: dict[str, str] = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
    "南京": "Nanjing",
    "武汉": "Wuhan",
    "西安": "Xi'an",
    "重庆": "Chongqing",
    "天津": "Tianjin",
    "苏州": "Suzhou",
    "香港": "Hong Kong",
    "台北": "Taipei",
    "东京": "Tokyo",
    "纽约": "New York",
    "伦敦": "London",
    "巴黎": "Paris",
    "墨尔本": "Melbourne",
    "悉尼": "Sydney",
    "新加坡": "Singapore",
}

_UA = "sophon-live-assistant/1.0"
_WTTR = "https://wttr.in/{city}?format=j1"
_OM_FC = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
    "&current=temperature_2m&timezone=auto&forecast_days=2"
)
_OM_GEO = "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=1&language=zh"

# Known cities need no geocoding round trip.
COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.90, 116.41), "上海": (31.23, 121.47), "广州": (23.13, 113.26), "深圳": (22.54, 114.06),
    "杭州": (30.27, 120.15), "成都": (30.66, 104.07), "南京": (32.06, 118.80), "武汉": (30.59, 114.31),
    "西安": (34.34, 108.94), "重庆": (29.56, 106.55), "天津": (39.34, 117.36), "苏州": (31.30, 120.59),
    "香港": (22.32, 114.17), "台北": (25.03, 121.57), "东京": (35.68, 139.69), "纽约": (40.71, -74.01),
    "伦敦": (51.51, -0.13), "巴黎": (48.86, 2.35), "墨尔本": (-37.81, 144.96), "悉尼": (-33.87, 151.21),
    "新加坡": (1.35, 103.82),
}

# sojson (t.weather.sojson.com): CN-hosted, no key, mainland cities by weather.com.cn city code
_SOJSON = "http://t.weather.sojson.com/api/weather/city/{code}"
SOJSON_CODES: dict[str, str] = {
    "北京": "101010100", "上海": "101020100", "广州": "101280101", "深圳": "101280601", "杭州": "101210101",
    "成都": "101270101", "南京": "101190101", "武汉": "101200101", "西安": "101110101", "重庆": "101040100",
    "天津": "101030100", "苏州": "101190401",
}

# WMO weather interpretation codes (Open-Meteo) → (中文, English)
_WMO: dict[int, tuple[str, str]] = {
    0: ("晴", "clear"), 1: ("大部晴朗", "mostly clear"), 2: ("多云", "partly cloudy"), 3: ("阴", "overcast"),
    45: ("雾", "fog"), 48: ("雾凇", "rime fog"),
    51: ("小毛毛雨", "light drizzle"), 53: ("毛毛雨", "drizzle"), 55: ("大毛毛雨", "heavy drizzle"),
    56: ("冻毛毛雨", "freezing drizzle"), 57: ("冻毛毛雨", "freezing drizzle"),
    61: ("小雨", "light rain"), 63: ("中雨", "rain"), 65: ("大雨", "heavy rain"),
    66: ("冻雨", "freezing rain"), 67: ("冻雨", "freezing rain"),
    71: ("小雪", "light snow"), 73: ("中雪", "snow"), 75: ("大雪", "heavy snow"), 77: ("雪粒", "snow grains"),
    80: ("小阵雨", "light showers"), 81: ("阵雨", "showers"), 82: ("强阵雨", "violent showers"),
    85: ("小阵雪", "light snow showers"), 86: ("阵雪", "snow showers"),
    95: ("雷暴", "thunderstorm"), 96: ("雷暴伴冰雹", "thunderstorm with hail"), 99: ("强雷暴伴冰雹", "thunderstorm with hail"),
}

# per-source reachability counters since process start (read by the matrix / trace tooling)
STATS: dict[str, dict[str, Any]] = {}


def _note(source: str, ok: bool, ms: float, err: str = "") -> None:
    st = STATS.setdefault(source, {"ok": 0, "fail": 0, "ms_ok": [], "errors": {}})
    if ok:
        st["ok"] += 1
        st["ms_ok"].append(round(ms))
    else:
        st["fail"] += 1
        st["errors"][err or "error"] = st["errors"].get(err or "error", 0) + 1


def _timeout_s() -> float:
    try:
        return max(0.5, float(os.environ.get("SOPHON_WEATHER_TIMEOUT", "5")))
    except ValueError:
        return 5.0


def _rounds() -> int:
    try:
        return max(1, int(os.environ.get("SOPHON_WEATHER_ROUNDS", "2")))
    except ValueError:
        return 2


def _source_names() -> list[str]:
    raw = os.environ.get("SOPHON_WEATHER_SOURCES", "open-meteo,sojson,wttr")
    names = [x.strip() for x in raw.split(",") if x.strip() in ("open-meteo", "sojson", "wttr")]
    return names or ["open-meteo", "sojson", "wttr"]


def _resolve_city(city: str) -> tuple[str, str, bool]:
    raw = (city or "").strip() or "Beijing"
    for zh_name, en in CITIES.items():
        if zh_name in raw or en.lower() == raw.lower():
            return en, zh_name, True
    if re.search(r"[一-鿿]", raw):
        return raw, raw, True
    return raw, raw, False


def _desc(day: dict[str, Any], *, zh: bool) -> str:
    hourly = day.get("hourly") or [{}]
    mid = hourly[min(4, len(hourly) - 1)] if hourly else {}
    if zh:
        lang = mid.get("lang_zh") or day.get("lang_zh") or []
        if lang and isinstance(lang, list) and lang[0].get("value"):
            return str(lang[0]["value"])
    desc = mid.get("weatherDesc") or []
    if desc and isinstance(desc, list) and desc[0].get("value"):
        return str(desc[0]["value"])
    return "多云" if zh else "cloudy"


def _http_get_json(url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Accept": "application/json", "Accept-Language": "zh-CN,en;q=0.8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())



async def _fetch_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """One attempt, bounded by ``timeout`` (the thread's socket timeout and the awaiting side)."""
    return await asyncio.wait_for(asyncio.to_thread(_http_get_json, url, timeout=timeout), timeout + 0.5)


def _normalize_wttr(data: dict[str, Any], idx: int, *, zh: bool) -> dict[str, Any] | None:
    days = data.get("weather") or []
    if len(days) <= idx:
        return None
    day = days[idx]
    tmin, tmax = day.get("mintempC"), day.get("maxtempC")
    if tmin is None or tmax is None:
        return None
    rains = []
    for h in day.get("hourly") or []:
        try:
            rains.append(int(h.get("chanceofrain") or 0))
        except (TypeError, ValueError):
            pass
    cur = (data.get("current_condition") or [{}])[0]
    return {
        "tmin": str(tmin), "tmax": str(tmax), "desc": _desc(day, zh=zh),
        "precip": max(rains) if rains else 0, "now": cur.get("temp_C"),
    }


def _normalize_om(data: dict[str, Any], idx: int, *, zh: bool) -> dict[str, Any] | None:
    daily = data.get("daily") or {}
    try:
        tmax = daily["temperature_2m_max"][idx]
        tmin = daily["temperature_2m_min"][idx]
        code = int(daily["weathercode"][idx])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if tmax is None or tmin is None:
        return None
    pp = (daily.get("precipitation_probability_max") or [0] * (idx + 1))[idx]
    now = (data.get("current") or {}).get("temperature_2m")
    zh_d, en_d = _WMO.get(code, ("多云", "cloudy"))
    return {
        "tmin": str(round(tmin)), "tmax": str(round(tmax)), "desc": zh_d if zh else en_d,
        "precip": int(pp or 0), "now": None if now is None else str(round(now)),
    }


def _normalize_sojson(data: dict[str, Any], idx: int, *, zh: bool) -> dict[str, Any] | None:
    if int(data.get("status") or 0) != 200:
        return None
    fc = (data.get("data") or {}).get("forecast") or []
    if len(fc) <= idx:
        return None
    day = fc[idx]
    hi = re.search(r"-?\d+", str(day.get("high") or ""))
    lo = re.search(r"-?\d+", str(day.get("low") or ""))
    typ = str(day.get("type") or "").strip()
    if not (hi and lo and typ):
        return None
    now = (data.get("data") or {}).get("wendu")
    return {
        "tmin": lo.group(0), "tmax": hi.group(0), "desc": typ if zh else typ,  # the feed only speaks Chinese
        "precip": None, "now": str(now) if now not in (None, "") else None,
    }


async def _from_sojson(display_name: str, idx: int, zh: bool, timeout: float) -> dict[str, Any] | None:
    code = SOJSON_CODES.get(display_name)
    if code is None:
        return None
    return _normalize_sojson(await _fetch_json(_SOJSON.format(code=code), timeout), idx, zh=zh)


async def _from_wttr(query_name: str, idx: int, zh: bool, timeout: float) -> dict[str, Any] | None:
    data = await _fetch_json(_WTTR.format(city=urllib.parse.quote(query_name)), timeout)
    return _normalize_wttr(data, idx, zh=zh)


async def _from_open_meteo(display_name: str, idx: int, zh: bool, timeout: float) -> dict[str, Any] | None:
    t0 = time.monotonic()
    ll = COORDS.get(display_name)
    if ll is None:
        geo = await _fetch_json(_OM_GEO.format(name=urllib.parse.quote(display_name)), timeout)
        hit = (geo.get("results") or [None])[0]
        if not hit:
            return None
        ll = (float(hit["latitude"]), float(hit["longitude"]))
    left = max(0.5, timeout - (time.monotonic() - t0))
    data = await _fetch_json(_OM_FC.format(lat=ll[0], lon=ll[1]), left)
    return _normalize_om(data, idx, zh=zh)


async def _first_ok(query_name: str, display_name: str, idx: int, zh: bool) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Race the sources; per-source timeout; a couple of rounds. Returns (data, source, error notes)."""
    timeout = _timeout_s()
    notes: list[str] = []
    for rnd in range(_rounds()):
        tasks: dict[asyncio.Task, tuple[str, float]] = {}
        for name in _source_names():
            if name == "sojson" and display_name not in SOJSON_CODES:
                continue  # mainland cities only
            if name == "open-meteo":
                coro = _from_open_meteo(display_name, idx, zh, timeout)
            elif name == "sojson":
                coro = _from_sojson(display_name, idx, zh, timeout)
            else:
                coro = _from_wttr(query_name, idx, zh, timeout)
            tasks[asyncio.ensure_future(asyncio.wait_for(coro, timeout + 0.5))] = (name, time.monotonic())
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    name, t0 = tasks[t]
                    ms = (time.monotonic() - t0) * 1000
                    try:
                        res = t.result()
                    except Exception as e:  # noqa: BLE001
                        err = type(e).__name__
                        _note(name, False, ms, err)
                        notes.append(f"{name}:{err}")
                        continue
                    if res is None:
                        _note(name, False, ms, "no_data")
                        notes.append(f"{name}:no_data")
                        continue
                    _note(name, True, ms)
                    log.info("weather source=%s ms=%d round=%d", name, ms, rnd + 1)
                    return res, name, notes
        finally:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)  # reap: no "exception was never retrieved"
            for t in tasks:
                if t.done() and not t.cancelled():
                    t.exception()  # mark retrieved (a source that failed in the same tick as the winner)
    return None, "", notes


async def run_detailed(args: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Spoken weather text plus metadata (source, latency) for tracing. Never invents data."""
    args = args or {}
    when = (args.get("when") or "today").strip().lower()
    if when not in ("today", "tomorrow"):
        when = "today"
    idx = 1 if when == "tomorrow" else 0

    query_name, display_name, zh = _resolve_city(str(args.get("city") or "Beijing"))
    if args.get("lang") == "en":
        zh = False
    elif args.get("lang") == "zh":
        zh = True

    t0 = time.monotonic()
    name = display_name
    for zh_name, en in CITIES.items():
        if en == query_name:
            name = zh_name
            break
    data, source, notes = await _first_ok(query_name, name, idx, zh)
    meta = {"source": source or None, "ms": round((time.monotonic() - t0) * 1000), "notes": notes}
    if data is None and notes and all(n.endswith(":no_data") for n in notes):
        # every source answered but none knows this city/day: not an outage
        return (f"找不到{display_name}的天气信息。" if zh else f"No weather data for {display_name}."), meta
    if data is None:
        log.warning("weather all sources failed for %s: %s", display_name, notes)
        if zh:
            return f"暂时查不到{display_name}的天气，请稍后再试。", meta
        return f"Weather for {display_name} is unavailable right now ({', '.join(notes) or 'no source'}).", meta

    tmin, tmax, desc, precip, now = data["tmin"], data["tmax"], data["desc"], data["precip"], data.get("now")
    if not zh:
        day_word = "tomorrow" if when == "tomorrow" else "today"
        rain = f", {precip}% chance of rain" if precip is not None else ""
        msg = f"{name} {day_word}: {desc}, {tmin} to {tmax}°C{rain}."
        if idx == 0 and now is not None:
            msg += f" Now {now}°C."
        return msg, meta
    day_label = "明天" if when == "tomorrow" else "今天"
    rain = f"，降水概率{precip}%" if precip is not None else ""
    msg = f"{name}{day_label}{desc}，气温{tmin}到{tmax}度{rain}。"
    if idx == 0 and now is not None:
        msg += f"现在{now}度。"
    return msg, meta


async def run(args: dict[str, Any] | None = None) -> str:
    text, meta = await run_detailed(args)
    from . import toolmeta

    toolmeta.note(source=meta.get("source"), fallbacks=len(meta.get("notes") or []) or None)
    return text
