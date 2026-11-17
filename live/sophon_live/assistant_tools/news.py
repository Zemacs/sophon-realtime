"""Recent headlines via public Google News RSS (no API key)."""
from __future__ import annotations

import asyncio
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

_RSS_BASE = "https://news.google.com/rss/search"
_UA = "sophon-live-assistant/1.0"


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _clamp_items(n: Any, default: int = 3) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = default
    return max(1, min(8, v))


def _rss_url(topic: str, lang: str) -> str:
    q = urllib.parse.quote_plus(topic)
    if lang == "zh":
        return f"{_RSS_BASE}?q={q}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    return f"{_RSS_BASE}?q={q}&hl=en-US&gl=US&ceid=US:en"


def _titles_from_rss(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []
    titles: list[str] = []
    for item in channel.findall("item"):
        title_el = item.find("title")
        if title_el is None or not (title_el.text or "").strip():
            continue
        title = (title_el.text or "").strip()
        if " - " in title:
            title = title.split(" - ", 1)[-1].strip()
        titles.append(title)
    return titles


def _urlopen(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


async def _fetch_rss(url: str) -> bytes:
    return await asyncio.to_thread(_urlopen, url)


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    topic = (args.get("topic") or "").strip()
    if not topic:
        return "请说一下想听哪方面的新闻。" if args.get("lang") == "zh" else "Please tell me which news topic you want."

    lang = (args.get("lang") or "en").strip().lower()
    if lang not in ("zh", "en"):
        lang = "zh" if _is_zh(topic) else "en"

    max_items = _clamp_items(args.get("max_items"), 3)
    speak_n = min(max_items, 3)
    if max_items >= 2:
        speak_n = max(2, speak_n)

    url = _rss_url(topic, lang)
    try:
        raw = await _fetch_rss(url)
        titles = _titles_from_rss(raw)
    except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError):
        if lang == "zh":
            return "新闻源暂时连不上，稍后再帮你看看头条吧。"
        return "News feeds are unreachable right now. Try again in a moment."

    if not titles:
        if lang == "zh":
            return f"暂时没有查到关于「{topic}」的最新头条。"
        return f"No recent headlines found for '{topic}'."

    picked = titles[:speak_n]
    if lang == "zh":
        parts = "；".join(f"第{i + 1}条，{t}" for i, t in enumerate(picked))
        return f"关于{topic}，{parts}。"
    joined = "; ".join(picked)
    return f"Headlines on {topic}: {joined}."