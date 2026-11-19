"""City weather from independent sources in a fixed priority order, with a per-(city, date) cache.

All sources are flaky from the GPU box (measured: each answers only some of the time, 0.2-40 s), so they are
all started at once with a short per-source timeout (latency hedge), but the ANSWER is chosen by priority, not by
who arrives first: the highest-priority source that returns real data wins, a lower one is used only after every
higher one has failed or timed out. Same question, same source, same numbers (they used to differ from run to run
because whoever answered first won). Default priority ``sojson,open-meteo,wttr`` (sojson is CN-hosted and covers
mainland cities only; it is skipped for other cities). If every source fails the caller gets an honest failure
sentence, never invented numbers. ``SOPHON_WEATHER_SOURCES`` sets sources and their order (also for A/B runs).
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
from .. import failcodes
from . import inflight

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

# Recent answers, so a repeated question is instant and a network blackout does not erase what we just learned.
# Fresh (<= SOPHON_WEATHER_CACHE_TTL, default 10 min) entries answer directly. Older ones (<= 2 h) are used ONLY
# when every source has just failed, and are spoken as old data ("网络不太好，这是N分钟前查到的"), never as live.
CACHE: dict[tuple, tuple[float, dict[str, Any], str]] = {}
_STALE_MAX_S = 2 * 3600.0


def _cache_ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("SOPHON_WEATHER_CACHE_TTL", "1800")))
    except ValueError:
        return 1800.0


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
        return max(0.5, float(os.environ.get("SOPHON_WEATHER_TIMEOUT", "4")))
    except ValueError:
        return 4.0


def _rounds() -> int:
    try:
        return max(1, int(os.environ.get("SOPHON_WEATHER_ROUNDS", "2")))
    except ValueError:
        return 2


def _source_names() -> list[str]:
    """Source names in priority order (highest first)."""
    raw = os.environ.get("SOPHON_WEATHER_SOURCES", "sojson,open-meteo,wttr")
    names: list[str] = []
    for x in (p.strip() for p in raw.split(",")):
        if x in ("open-meteo", "sojson", "wttr") and x not in names:
            names.append(x)
    return names or ["sojson", "open-meteo", "wttr"]


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
    return await asyncio.wait_for(inflight.run(_http_get_json, url, timeout=timeout), timeout + 0.5)      # S-19: under the shared cap, off the default pool


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


def _error_code(e: BaseException) -> str:
    """Map an exception to the closed enum ``failcodes.SOURCE_ERROR_CODES`` (the class name itself is never spoken: I-17)."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):      # before OSError: TimeoutError is an OSError
        return "timeout"
    if isinstance(e, urllib.error.HTTPError):                    # before URLError: HTTPError is a URLError
        return "http"
    if isinstance(e, (urllib.error.URLError, OSError)):
        return "network"
    if isinstance(e, (ValueError, KeyError, IndexError, TypeError)):   # JSONDecodeError / UnicodeDecodeError are ValueErrors
        return "bad_data"
    return "error"


async def _first_ok(query_name: str, display_name: str, idx: int, zh: bool) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Start every source at once (hedge), pick the answer by PRIORITY; per-source timeout; a couple of rounds.

    Returns (data, source, error notes). A lower-priority source that is already done is used only after every
    higher-priority one has failed / timed out, so the same question gets the same source whenever it is up."""
    timeout = _timeout_s()
    notes: list[str] = []
    for rnd in range(_rounds()):
        tasks: dict[asyncio.Task, tuple[str, float]] = {}
        order: list[asyncio.Task] = []
        for name in _source_names():
            if name == "sojson" and display_name not in SOJSON_CODES:
                continue  # mainland cities only
            if name == "open-meteo":
                coro = _from_open_meteo(display_name, idx, zh, timeout)
            elif name == "sojson":
                coro = _from_sojson(display_name, idx, zh, timeout)
            else:
                coro = _from_wttr(query_name, idx, zh, timeout)
            task = asyncio.ensure_future(asyncio.wait_for(coro, timeout + 0.5))
            tasks[task] = (name, time.monotonic())
            order.append(task)
        try:
            for t in order:                        # priority order: wait for the best source before looking at a worse one
                name, t0 = tasks[t]
                try:
                    res = await asyncio.shield(t)
                except Exception as e:  # noqa: BLE001
                    err = _error_code(e)
                    _note(name, False, (time.monotonic() - t0) * 1000, err)
                    notes.append(f"{name}:{err}")
                    continue
                ms = (time.monotonic() - t0) * 1000
                if res is None:
                    _note(name, False, ms, "no_data")
                    notes.append(f"{name}:no_data")
                    continue
                _note(name, True, ms)
                log.info("weather source=%s ms=%d round=%d priority=%d", name, ms, rnd + 1, order.index(t) + 1)
                return res, name, notes
        finally:
            for t in order:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*order, return_exceptions=True)  # reap: no "exception was never retrieved"
    return None, "", notes


def _facts(data: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """What the spoken text is made of (city + temperatures): kept in the trace so a spoken answer can be checked against it."""
    if not data:
        return {}
    return {"city": name, "tmin": data.get("tmin"), "tmax": data.get("tmax"), "now": data.get("now")}


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
    import datetime as _dt

    ckey = (name, (_dt.date.today() + _dt.timedelta(days=idx)).isoformat(), zh)   # per (city, target date)
    hit = CACHE.get(ckey)
    age_s = (time.time() - hit[0]) if hit else None
    if hit and age_s is not None and age_s <= _cache_ttl():
        return _compose(hit[1], name, when, idx, zh), {"source": "cache", "ms": 0, "notes": [], "age_s": round(age_s), "orig_source": hit[2], **_facts(hit[1], name)}

    busy = inflight.track_busy()
    data, source, notes = await _first_ok(query_name, name, idx, zh)
    meta = {"source": source or None, "ms": round((time.monotonic() - t0) * 1000), "notes": notes, **_facts(data, name)}
    if data is None and notes and all(n.endswith(":no_data") for n in notes):
        # every source answered but none knows this city/day: not an outage
        meta["failure"] = failcodes.NO_DATA
        return (f"找不到{display_name}的天气信息。" if zh else f"No weather data for {display_name}."), meta
    if data is None:
        log.warning("weather all sources failed for %s: %s", display_name, notes)
        if hit and age_s is not None and age_s <= _STALE_MAX_S:
            mins = max(1, round(age_s / 60))
            old = {**hit[1], "now": None}  # a "current temperature" that old is not current
            body = _compose(old, name, when, idx, zh)
            lead = f"网络不太好，这是{mins}分钟前查到的：" if zh else f"The network is poor; this is from {mins} min ago: "
            meta.update(source="cache-stale", age_s=round(age_s), orig_source=hit[2], **_facts(old, name))
            return lead + body, meta
        if busy:
            meta["failure"] = failcodes.BUSY
            return (inflight.BUSY_ZH if zh else inflight.BUSY_EN), meta               # S-24: our own cap refused, not a weather-source outage
        meta["failure"] = failcodes.NETWORK
        if zh:
            return f"暂时查不到{display_name}的天气，请稍后再试。", meta
        return f"Weather for {display_name} is unavailable right now ({', '.join(notes) or 'no source'}).", meta

    CACHE[ckey] = (time.time(), data, source)
    return _compose(data, name, when, idx, zh), meta


def _compose(data: dict[str, Any], name: str, when: str, idx: int, zh: bool) -> str:
    tmin, tmax, desc, precip, now = data["tmin"], data["tmax"], data["desc"], data["precip"], data.get("now")
    if not zh:
        day_word = "tomorrow" if when == "tomorrow" else "today"
        rain = f", {precip}% chance of rain" if precip is not None else ""
        msg = f"{name} {day_word}: {desc}, {tmin} to {tmax}°C{rain}."
        if idx == 0 and now is not None:
            msg += f" Now {now}°C."
        return msg
    day_label = "明天" if when == "tomorrow" else "今天"
    rain = f"，降水概率{precip}%" if precip is not None else ""
    msg = f"{name}{day_label}{desc}，气温{tmin}到{tmax}度{rain}。"
    if idx == 0 and now is not None:
        msg += f"现在{now}度。"
    return msg


async def run(args: dict[str, Any] | None = None) -> str:
    text, meta = await run_detailed(args)
    from . import toolmeta

    toolmeta.note(source=meta.get("source"), fallbacks=len(meta.get("notes") or []) or None,
                  orig_source=meta.get("orig_source"), age_s=meta.get("age_s"),
                  failure=meta.get("failure"),
                  **{failcodes.SUCCESS_KEY: True if (not meta.get("failure") and meta.get("tmin") is not None and meta.get("tmax") is not None) else None},
                  city=meta.get("city"), tmin=meta.get("tmin"), tmax=meta.get("tmax"), now=meta.get("now"))
    return text
