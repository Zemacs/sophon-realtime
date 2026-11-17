"""City weather via Open-Meteo (geocoding + forecast)."""
from __future__ import annotations

import asyncio
import json
import re
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

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _resolve_city(city: str) -> tuple[str, str, bool]:
    raw = (city or "").strip() or "Beijing"
    for zh_name, en in CITIES.items():
        if zh_name in raw or en.lower() == raw.lower():
            return en, zh_name, True
    if re.search(r"[一-鿿]", raw):
        return raw, raw, True
    return raw, raw, False


def _weather_desc(code: int, *, zh: bool) -> str:
    if zh:
        if code <= 1:
            return "晴"
        if code <= 3:
            return "多云"
        if code <= 48:
            return "有雾"
        if code <= 67:
            return "有雨"
        if code <= 77:
            return "有雪"
        if code <= 82:
            return "阵雨"
        return "雷雨"
    if code <= 1:
        return "clear"
    if code <= 3:
        return "cloudy"
    if code <= 48:
        return "foggy"
    if code <= 67:
        return "rainy"
    if code <= 77:
        return "snowy"
    if code <= 82:
        return "showers"
    return "thunderstorms"


def _http_get(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "sophon-live-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


async def _fetch_json(url: str) -> dict[str, Any]:
    return await asyncio.to_thread(_http_get, url)


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    when = (args.get("when") or "today").strip().lower()
    if when not in ("today", "tomorrow"):
        when = "today"
    idx = 1 if when == "tomorrow" else 0

    geocode_name, display_name, zh = _resolve_city(str(args.get("city") or "Beijing"))
    if args.get("lang") == "en":
        zh = False
    elif args.get("lang") == "zh":
        zh = True

    geo_q = urllib.parse.urlencode({"name": geocode_name, "count": "1"})
    geo = await _fetch_json(f"{_GEO_URL}?{geo_q}")
    results = geo.get("results") or []
    if not results:
        return f"找不到{display_name}的天气信息。" if zh else f"No weather data for {display_name}."
    point = results[0]

    fc_params = urllib.parse.urlencode(
        {
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": "2",
        }
    )
    forecast = await _fetch_json(f"{_FORECAST_URL}?{fc_params}")
    daily = forecast.get("daily") or {}
    code = int((daily.get("weather_code") or [0])[idx])
    tmin = round(float((daily.get("temperature_2m_min") or [0])[idx]))
    tmax = round(float((daily.get("temperature_2m_max") or [0])[idx]))
    precip = int((daily.get("precipitation_probability_max") or [0])[idx])
    desc = _weather_desc(code, zh=zh)

    if not zh:
        day_word = "tomorrow" if when == "tomorrow" else "today"
        msg = (
            f"{display_name} {day_word}: {desc}, "
            f"{tmin} to {tmax}°C, {precip}% chance of rain."
        )
        if idx == 0 and forecast.get("current"):
            cur = round(float(forecast["current"]["temperature_2m"]))
            msg += f" Now {cur}°C."
        return msg

    name = display_name
    for zh_name, en in CITIES.items():
        if en == geocode_name:
            name = zh_name
            break
    day_label = "明天" if when == "tomorrow" else "今天"
    msg = f"{name}{day_label}{desc}，气温{tmin}到{tmax}度，降水概率{precip}%。"
    if idx == 0 and forecast.get("current"):
        cur = round(float(forecast["current"]["temperature_2m"]))
        msg += f"现在{cur}度。"
    return msg