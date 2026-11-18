"""Recent headlines via public RSS (CN GPU boxes: people.cn topic feeds; Google often times out)."""
from __future__ import annotations

import asyncio
import os
import time
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from . import inflight, toolmeta

_RSS_BASE = "https://news.google.com/rss/search"
_UA = "sophon-live-assistant/1.0"

# Topic → reachable CN feeds (ordered). Do NOT dump politics into "科技".
_ZH_FEEDS: dict[str, tuple[str, ...]] = {
    "tech": (
        "http://www.people.com.cn/rss/it.xml",
        "http://www.people.com.cn/rss/finance.xml",
    ),
    "finance": (
        "http://www.people.com.cn/rss/finance.xml",
        "http://www.people.com.cn/rss/politics.xml",
    ),
    "world": (
        "http://www.people.com.cn/rss/world.xml",
        "http://www.people.com.cn/rss/politics.xml",
    ),
    "general": (
        "http://www.people.com.cn/rss/politics.xml",
        "http://www.people.com.cn/rss/world.xml",
        "http://www.people.com.cn/rss/finance.xml",
    ),
}

# Which feed family to read is chosen by the orchestrator MODEL through the ``category`` argument (an enum in the tool schema);
# this module does not read the topic words to guess one. "search" = a specific person / company / event: query RSS only.
CATEGORIES = ("general", "tech", "finance", "world", "search")


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _clamp_items(n: Any, default: int = 3) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = default
    return max(1, min(8, v))


def _bucket(category: Any) -> str:
    c = str(category or "").strip().lower()
    return c if c in CATEGORIES else "general"


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
        titles.append(_strip_publisher(item, (title_el.text or "").strip()))
    return titles


def _budget_s() -> float:
    """Whole-tool time budget. 2026-11-18 on the GPU box one get_news took 77.8 s (feeds tried one after the other, each up to 12 s + stalls)."""
    try:
        return max(2.0, float(os.environ.get("SOPHON_NEWS_BUDGET", "9")))
    except ValueError:
        return 9.0


_PER_FEED_S = 5.0


def _strip_publisher(item: ET.Element, title: str) -> str:
    """Drop the publisher from a feed title without guessing which side of " - " it is on.

    Real feeds (Google News) carry the publisher in ``<source>``: remove exactly that, as suffix (「标题 - 来源」) or
    prefix (「来源 - 标题」), and leave the title alone when it does not match. Only an item with no ``<source>``
    falls back to the old rule (drop the last " - " segment), which assumes the 「标题 - 来源」 order."""
    src_el = item.find("source")
    src = (src_el.text or "").strip() if src_el is not None else ""
    if src:
        for cut in (f" - {src}", f" – {src}", f" | {src}"):
            if title.endswith(cut) and title[: -len(cut)].strip():
                return title[: -len(cut)].strip()
        for cut in (f"{src} - ", f"{src} – ", f"{src} | "):
            if title.startswith(cut) and title[len(cut):].strip():
                return title[len(cut):].strip()
        return title
    return _strip_publisher_guess(title)


_PUB_MAX_CJK, _PUB_MAX_LATIN = 8, 28            # 「澎湃新闻」「中国新闻网」 / 「The Wall Street Journal」
_SENTENCE_MARKS = set("，。！？；：,.!?;:、“”\"《》")


def _has_cjk(seg: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in seg)


def _publisher_like(seg: str) -> bool:
    """A feed publisher name is a short label: no sentence punctuation, no more than 8 CJK / 28 Latin characters."""
    seg = seg.strip()
    return 0 < len(seg) <= (_PUB_MAX_CJK if _has_cjk(seg) else _PUB_MAX_LATIN) and not any(c in _SENTENCE_MARKS for c in seg)


def _strip_publisher_guess(title: str) -> str:
    """No ``<source>`` to go by (Critic S-8). The order 「标题 - 来源」 vs 「来源 - 标题」 is unknown, so strip only when exactly
    ONE end looks like a publisher (short, no sentence punctuation) and what remains is clearly the longer part (>= 2x, >= 6 chars) or is the only CJK side (a Latin label on a Chinese headline); when both or neither
    do, keep the title whole. Worst case the publisher name stays in the headline; a headline is never cut away."""
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) < 2 or not all(parts):
        return title
    head, last = " - ".join(parts[:-1]), parts[-1]
    first, tail = parts[0], " - ".join(parts[1:])
    suffix_pub = _publisher_like(last) and (len(head) >= max(6, 2 * len(last)) or (_has_cjk(head) and not _has_cjk(last)))
    prefix_pub = _publisher_like(first) and (len(tail) >= max(6, 2 * len(first)) or (_has_cjk(tail) and not _has_cjk(first)))
    if suffix_pub and not prefix_pub:
        return head
    if prefix_pub and not suffix_pub:
        return tail
    return title


def _urlopen(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_PER_FEED_S) as resp:
        return resp.read()


async def _fetch_rss(url: str) -> bytes:
    return await asyncio.to_thread(inflight.guarded, _urlopen, url)      # S-14: same in-flight cap as web_search


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    topic = (args.get("topic") or "").strip()
    if not topic:
        return "请说一下想听哪方面的新闻。" if args.get("lang") == "zh" else "Please tell me which news topic you want."

    lang = (args.get("lang") or "").strip().lower()
    if lang not in ("zh", "en"):
        lang = "zh" if _is_zh(topic) else "en"
    if _is_zh(topic):
        lang = "zh"

    max_items = _clamp_items(args.get("max_items"), 3)
    speak_n = min(max_items, 3)
    if max_items >= 2:
        speak_n = max(2, speak_n)

    titles: list[str] = []
    bucket = _bucket(args.get("category"))
    deadline = time.monotonic() + _budget_s()

    async def fetch(url: str) -> bytes:
        left = deadline - time.monotonic()
        if left <= 0.3:
            raise asyncio.TimeoutError("news budget spent")
        return await asyncio.wait_for(_fetch_rss(url), min(_PER_FEED_S + 0.5, left))

    if lang == "zh" and bucket != "search":
        # Category CN feeds first (Google News usually times out on PAI-DSW). The channel itself is the relevance filter.
        for fb in _ZH_FEEDS.get(bucket, _ZH_FEEDS["general"]):
            try:
                titles = _titles_from_rss(await fetch(fb))
                if titles:
                    break
            except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError, asyncio.TimeoutError):
                continue
    if not titles:
        try:
            raw = await fetch(_rss_url(topic, lang))
            titles = _titles_from_rss(raw)
        except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError, asyncio.TimeoutError):
            titles = []

    if not titles:
        toolmeta.note(ok=False)         # structured outcome: no feed answered
        if lang == "zh":
            return "新闻源暂时连不上，稍后再帮你看看头条吧。"
        return "News feeds are unreachable right now. Try again in a moment."

    picked = titles[:speak_n]
    if lang == "zh":
        parts = "；".join(f"第{i + 1}条，{t}" for i, t in enumerate(picked))
        return f"关于{topic}，{parts}。"
    joined = "; ".join(picked)
    return f"Headlines on {topic}: {joined}."
