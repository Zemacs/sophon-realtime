"""Public web search via DuckDuckGo (Instant Answer API + lite HTML fallback)."""
from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

_INSTANT_URL = "https://api.duckduckgo.com/"
_LITE_URL = "https://lite.duckduckgo.com/lite/"
_UA = "sophon-live-assistant/1.0"


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _clamp_max(n: Any, default: int = 3) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = default
    return max(1, min(5, v))


def _network_error(zh: bool) -> str:
    if zh:
        return "网络暂时不可用，没法完成网页搜索，请稍后再试。"
    return "Web search is unavailable right now due to a network problem. Please try again later."


def _no_results(query: str, *, zh: bool) -> str:
    if zh:
        return f"没有搜到关于「{query}」的可靠摘要。"
    return f"I couldn't find useful snippets for '{query}'."


def _urlopen(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


async def _fetch_bytes(url: str) -> bytes:
    return await asyncio.to_thread(_urlopen, url)


def _snippets_from_instant(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        heading = (data.get("Heading") or "").strip()
        if heading and heading.lower() not in abstract.lower():
            out.append(f"{heading}: {abstract}")
        else:
            out.append(abstract)

    def walk(topics: list[Any]) -> None:
        for item in topics:
            if not isinstance(item, dict):
                continue
            if "Topics" in item:
                walk(item.get("Topics") or [])
                continue
            text = (item.get("Text") or "").strip()
            if text:
                out.append(text)

    walk(data.get("RelatedTopics") or [])
    return out


def _snippets_from_lite(html: str, limit: int) -> list[str]:
    out: list[str] = []
    link_re = re.compile(r"class=['\"]result-link['\"][^>]*>([^<]+)</a>", re.I)
    snip_re = re.compile(r"class=['\"]result-snippet['\"][^>]*>([^<]+)", re.I)
    titles = [unescape(t.strip()) for t in link_re.findall(html)]
    snippets = [unescape(s.strip()) for s in snip_re.findall(html)]
    for i, title in enumerate(titles):
        if len(out) >= limit:
            break
        snippet = snippets[i] if i < len(snippets) else ""
        if snippet:
            out.append(f"{title} — {snippet}")
        else:
            out.append(title)
    return out


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    query = (args.get("query") or "").strip()
    if not query:
        return "请告诉我要搜索什么。" if _is_zh(str(args.get("lang") or "")) else "Please tell me what to search for."

    max_results = _clamp_max(args.get("max_results"), 3)
    zh = _is_zh(query) or args.get("lang") == "zh"

    snippets: list[str] = []
    try:
        q = urllib.parse.quote_plus(query)
        raw = await _fetch_bytes(
            f"{_INSTANT_URL}?q={q}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        )
        instant = json.loads(raw.decode("utf-8", errors="replace"))
        snippets = _snippets_from_instant(instant)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError):
        return _network_error(zh)

    if len(snippets) < max_results:
        try:
            q = urllib.parse.quote_plus(query)
            lite_html = (await _fetch_bytes(f"{_LITE_URL}?q={q}")).decode("utf-8", errors="replace")
            for line in _snippets_from_lite(lite_html, max_results):
                if line not in snippets:
                    snippets.append(line)
        except (OSError, urllib.error.URLError, UnicodeDecodeError):
            if not snippets:
                return _network_error(zh)

    snippets = snippets[:max_results]
    if not snippets:
        return _no_results(query, zh=zh)

    intro = f"关于「{query}」，" if zh else f"On '{query}', "
    body = "；".join(snippets) if zh else "; ".join(snippets)
    return intro + body + ("。" if zh else ".")