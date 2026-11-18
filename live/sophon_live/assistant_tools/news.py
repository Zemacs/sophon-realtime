"""Recent headlines via public RSS (CN GPU boxes: people.cn topic feeds; Google often times out)."""
from __future__ import annotations

import asyncio
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from .. import failcodes
from . import inflight, toolmeta

_RSS_BASE = "https://news.google.com/rss/search"
_UA = "sophon-live-assistant/1.0"

# The feed categories this tool has. The orchestrator model picks one (the ``topic`` parameter is a closed enum in the tool schema, see
# registry.py); the code only fetches by that value. It never reads the user's words to classify a topic (Round 4h: the keyword tables
# _TECH_KEYS / _FIN_KEYS / _WORLD_KEYS, _bucket() and _filter_topic() are gone).
CATEGORIES = ("tech", "finance", "world", "general")

# category -> reachable CN feeds (ordered). Each category lists only feeds that are ON that subject (no cross-feed fallback that would
# need a keyword filter to avoid passing politics off as 科技); "general" is the top-headlines mix.
_ZH_FEEDS: dict[str, tuple[str, ...]] = {
    "tech": ("http://www.people.com.cn/rss/it.xml",),
    "finance": ("http://www.people.com.cn/rss/finance.xml",),
    "world": ("http://www.people.com.cn/rss/world.xml",),
    "general": (
        "http://www.people.com.cn/rss/politics.xml",
        "http://www.people.com.cn/rss/world.xml",
        "http://www.people.com.cn/rss/finance.xml",
    ),
}
# category -> (zh, en) label: the Google News search term and the word spoken in the answer ("关于科技，…" / "Headlines on technology: …")
_LABELS: dict[str, tuple[str, str]] = {
    "tech": ("科技", "technology"),
    "finance": ("财经", "finance"),
    "world": ("国际", "world news"),
    "general": ("头条", "top stories"),
}


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
        titles.append(_strip_publisher(item, (title_el.text or "").strip()))
    return titles


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
    with urllib.request.urlopen(req, timeout=12) as resp:
        return resp.read()


async def _fetch_rss(url: str) -> bytes:
    return await inflight.run(_urlopen, url)      # S-14 / S-19: same in-flight cap and dedicated pool as web_search


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    lang = (args.get("lang") or "").strip().lower()
    if lang not in ("zh", "en"):
        lang = "zh"                                             # the orchestrator fills ``lang`` from the job's language; zh is the product default
    topic = str(args.get("topic") or "").strip()
    if topic not in CATEGORIES:                                 # enforcement of the schema's closed enum, not a judgment about the words
        return "请说一下想听哪方面的新闻。" if lang == "zh" else "Please tell me which news topic you want."

    max_items = _clamp_items(args.get("max_items"), 3)
    speak_n = min(max_items, 3)
    if max_items >= 2:
        speak_n = max(2, speak_n)

    busy = inflight.track_busy()
    titles: list[str] = []
    label = _LABELS[topic][0 if lang == "zh" else 1]
    if lang == "zh":
        # The category's own CN feeds first (Google News usually times out on PAI-DSW). Titles are taken as the feed gives them.
        for fb in _ZH_FEEDS[topic]:
            try:
                titles = _titles_from_rss(await _fetch_rss(fb))
                if titles:
                    break
            except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError):
                continue
    if not titles:
        try:
            raw = await _fetch_rss(_rss_url(label, lang))
            titles = _titles_from_rss(raw)
        except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError):
            titles = []

    if not titles:
        if busy:
            toolmeta.note(failure=failcodes.BUSY)
            return inflight.BUSY_ZH if lang == "zh" else inflight.BUSY_EN         # S-24: our own cap refused, not a feed outage
        toolmeta.note(failure=failcodes.NETWORK)
        if lang == "zh":
            return "新闻源暂时连不上，稍后再帮你看看头条吧。"
        return "News feeds are unreachable right now. Try again in a moment."

    picked = titles[:speak_n]
    toolmeta.note(**{failcodes.SUCCESS_KEY: True}, count=len(picked))          # I-18: success is reported in structure, not read from the text
    if lang == "zh":
        parts = "；".join(f"第{i + 1}条，{t}" for i, t in enumerate(picked))
        return f"关于{label}，{parts}。"
    joined = "; ".join(picked)
    return f"Headlines on {label}: {joined}."
