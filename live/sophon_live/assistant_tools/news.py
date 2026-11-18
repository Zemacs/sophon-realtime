"""Recent headlines via public RSS (CN GPU boxes: people.cn topic feeds; Google often times out)."""
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

_TECH_KEYS = ("科技", "技术", "数码", "互联网", "AI", "人工智能", "芯片", "IT", "互联网", "cyber", "tech")
_FIN_KEYS = ("财经", "金融", "股市", "经济", "finance")
_WORLD_KEYS = ("国际", "世界", "海外", "world")


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _clamp_items(n: Any, default: int = 3) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = default
    return max(1, min(8, v))


def _bucket(topic: str) -> str:
    t = topic.lower()
    if any(k.lower() in t for k in _TECH_KEYS):
        return "tech"
    if any(k.lower() in t for k in _FIN_KEYS):
        return "finance"
    if any(k.lower() in t for k in _WORLD_KEYS):
        return "world"
    return "general"


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
    return await asyncio.to_thread(_urlopen, url)


def _filter_topic(titles: list[str], topic: str, *, require_hit: bool) -> list[str]:
    """If require_hit, never return off-topic filler when keywords miss."""
    keys = [k for k in re.split(r"[\s,，、]+", topic) if len(k) >= 2]
    # Drop generic wrappers that appear in every request.
    keys = [k for k in keys if k not in ("新闻", "头条", "资讯", "news", "headlines")]
    if not keys:
        return titles
    hit = [t for t in titles if any(k.lower() in t.lower() for k in keys)]
    if hit:
        return hit
    return [] if require_hit else titles


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
    bucket = _bucket(topic)
    if lang == "zh":
        # Topic-matched CN feeds first (Google News usually times out on PAI-DSW).
        for fb in _ZH_FEEDS.get(bucket, _ZH_FEEDS["general"]):
            try:
                raw = await _fetch_rss(fb)
                # Dedicated topic feeds (it/finance) are already on-domain — take titles.
                # Cross feeds: require keyword hit so politics never masquerades as 科技.
                require = bucket == "general" or "it.xml" not in fb and bucket == "tech"
                # For tech→it.xml / finance→finance.xml, trust the channel.
                if bucket == "tech" and "it.xml" in fb:
                    require = False
                if bucket == "finance" and "finance.xml" in fb:
                    require = False
                if bucket == "world" and "world.xml" in fb:
                    require = False
                titles = _filter_topic(_titles_from_rss(raw), topic, require_hit=require)
                if not require and not titles:
                    titles = _titles_from_rss(raw)
                if titles:
                    break
            except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError):
                continue
    if not titles:
        try:
            raw = await _fetch_rss(_rss_url(topic, lang))
            titles = _titles_from_rss(raw)
        except (OSError, urllib.error.URLError, ET.ParseError, UnicodeDecodeError):
            titles = []

    if not titles:
        if lang == "zh":
            return "新闻源暂时连不上，稍后再帮你看看头条吧。"
        return "News feeds are unreachable right now. Try again in a moment."

    picked = titles[:speak_n]
    if lang == "zh":
        parts = "；".join(f"第{i + 1}条，{t}" for i, t in enumerate(picked))
        return f"关于{topic}，{parts}。"
    joined = "; ".join(picked)
    return f"Headlines on {topic}: {joined}."
