"""City weather via wttr.in (Open-Meteo forecast is often unreachable from CN GPU boxes)."""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

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


async def _fetch_json(url: str) -> dict[str, Any]:
    last: Exception | None = None
    for attempt, timeout in enumerate((12.0, 20.0, 25.0), start=1):
        try:
            return await asyncio.to_thread(_http_get_json, url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last = e
            if attempt < 3:
                await asyncio.sleep(0.4 * attempt)
    assert last is not None
    raise last


async def run(args: dict[str, Any] | None = None) -> str:
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

    url = _WTTR.format(city=urllib.parse.quote(query_name))
    try:
        data = await _fetch_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        if zh:
            return f"暂时查不到{display_name}的天气，请稍后再试。"
        return f"Weather for {display_name} is unavailable right now ({e})."

    days = data.get("weather") or []
    if len(days) <= idx:
        return f"找不到{display_name}的天气信息。" if zh else f"No weather data for {display_name}."
    day = days[idx]
    tmin = day.get("mintempC") or "?"
    tmax = day.get("maxtempC") or "?"
    desc = _desc(day, zh=zh)
    # chance of rain: average of hourly chances if present
    rains = []
    for h in day.get("hourly") or []:
        try:
            rains.append(int(h.get("chanceofrain") or 0))
        except (TypeError, ValueError):
            pass
    precip = max(rains) if rains else 0

    name = display_name
    for zh_name, en in CITIES.items():
        if en == query_name:
            name = zh_name
            break

    if not zh:
        day_word = "tomorrow" if when == "tomorrow" else "today"
        msg = f"{name} {day_word}: {desc}, {tmin} to {tmax}°C, {precip}% chance of rain."
        if idx == 0:
            cur = (data.get("current_condition") or [{}])[0]
            if cur.get("temp_C") is not None:
                msg += f" Now {cur['temp_C']}°C."
        return msg

    day_label = "明天" if when == "tomorrow" else "今天"
    msg = f"{name}{day_label}{desc}，气温{tmin}到{tmax}度，降水概率{precip}%。"
    if idx == 0:
        cur = (data.get("current_condition") or [{}])[0]
        if cur.get("temp_C") is not None:
            msg += f"现在{cur['temp_C']}度。"
    return msg
