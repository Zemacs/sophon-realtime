"""Public web search: Bing China (reachable from the GPU box) and DuckDuckGo (Instant Answer API + lite HTML).

DuckDuckGo is unreachable from mainland-hosted GPU boxes (every request times out), so a Chinese query asks Bing
China first and DuckDuckGo is only the second choice; an English query is the other way round. Both providers are
started at once and the answer is taken by priority; the whole search has a hard budget (default 6 s,
``SOPHON_SEARCH_BUDGET``), after which it fails honestly instead of hanging for ~20 s.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

from . import inflight as _inflight_mod

_INSTANT_URL = "https://api.duckduckgo.com/"
_LITE_URL = "https://lite.duckduckgo.com/lite/"
_UA = "sophon-live-assistant/1.0"
_BING_URL = "https://cn.bing.com/search?q={q}&ensearch={en}"
_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def _budget_s() -> float:
    try:
        return max(1.0, float(os.environ.get("SOPHON_SEARCH_BUDGET", "6")))
    except ValueError:
        return 6.0


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


# urllib cannot be interrupted: a cancelled hedge leaves its thread running until its own socket timeout (<= the attempt's remaining
# budget, <= 5.5 s). The number of such threads in flight is capped (Critic S-9), and the cap is shared with the news tool
# (assistant_tools/inflight.py, S-14): beyond it an attempt fails at once instead of piling up sockets.
_INFLIGHT_MAX = _inflight_mod.MAX_INFLIGHT
_inflight = _inflight_mod.SHARED


def _urlopen(url: str, timeout: float = 20.0) -> bytes:
    if not _inflight.acquire(blocking=False):
        raise OSError("too many web-search requests in flight")
    try:
        return _urlopen_unbounded(url, timeout)
    finally:
        _inflight.release()


def _urlopen_unbounded(url: str, timeout: float) -> bytes:
    bing = "bing.com" in url
    headers = {"User-Agent": _BROWSER_UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5", "Accept-Encoding": "gzip"} if bing else {"User-Agent": _UA}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if (getattr(resp, "headers", None) or {}).get("Content-Encoding", "") == "gzip":
            raw = gzip.decompress(raw)             # 106 KB -> 37 KB: matters on this link
        return raw


async def _fetch_bytes(url: str, timeout: float = 5.5) -> bytes:
    """One attempt bounded by ``timeout`` (the socket timeout in the thread and the awaiting side)."""
    return await asyncio.wait_for(asyncio.to_thread(_urlopen, url, timeout), timeout + 0.5)


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


_TAG = re.compile(r"<[^>]+>")
_AGE = re.compile(r"^\s*\d+\s*(小时|分钟|天|周|个月|年)(之)?前\s*[·・]\s*")


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub("", fragment))).strip()


def _snippets_from_bing(html: str, limit: int) -> list[str]:
    """Title — snippet for each organic Bing result (``li.b_algo``)."""
    out: list[str] = []
    for block in re.findall(r'<li class="b_algo"[\s\S]*?</li>', html):
        m = re.search(r"<h2[^>]*>\s*<a[^>]*>([\s\S]*?)</a>", block)
        title = _text(m.group(1)) if m else ""
        p = re.search(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>([\s\S]*?)</p>', block) or re.search(
            r'<div class="b_caption"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)</p>', block
        )
        snippet = _AGE.sub("", _text(p.group(1))) if p else ""
        if not title:
            continue
        out.append(f"{title} — {snippet[:140]}" if snippet else title)
        if len(out) >= limit:
            break
    return out


_BING_HEDGE_S = 1.0      # a Bing attempt that has not answered by then gets another independent connection (max 4)
_BING_ATTEMPTS = 4


async def _bing(query: str, max_results: int, zh: bool, timeout: float) -> list[str]:
    """The GPU box's route to Bing drops or stalls roughly every other new connection (measured 2026-11-18: of 10 fetches
    ~4 failed outright, ok ones took 0.3-5 s), so attempts are hedged: first success wins, a failure starts the next at once."""
    url = _BING_URL.format(q=urllib.parse.quote_plus(query), en=0 if zh else 1)
    deadline = time.monotonic() + timeout
    pending: set[asyncio.Future] = set()
    last: BaseException = asyncio.TimeoutError()
    empty_pages = 0
    try:
        for attempt in range(_BING_ATTEMPTS):
            left = deadline - time.monotonic()
            if left <= 0.3:
                break
            pending.add(asyncio.ensure_future(_fetch_bytes(url, left)))
            wait_s = _BING_HEDGE_S if attempt < _BING_ATTEMPTS - 1 else max(0.0, deadline - time.monotonic())
            done, pending = await asyncio.wait(pending, timeout=wait_s, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.exception() is None:
                    got = _snippets_from_bing(d.result().decode("utf-8", errors="replace"), max_results)
                    if got:
                        return got
                    empty_pages += 1       # HTTP 200 but nothing parsed (captcha / consent / layout change): not an answer, keep hedging
                    continue
                last = d.exception()  # type: ignore[assignment]
        while pending and deadline - time.monotonic() > 0:
            done, pending = await asyncio.wait(pending, timeout=deadline - time.monotonic(), return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.exception() is None:
                    got = _snippets_from_bing(d.result().decode("utf-8", errors="replace"), max_results)
                    if got:
                        return got
                    empty_pages += 1
                    continue
                last = d.exception()  # type: ignore[assignment]
    finally:
        for f in pending:
            f.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    if empty_pages:
        return []                  # Bing was reached on every try that answered and had nothing usable: "no results", not an outage
    raise last


async def _ddg(query: str, max_results: int, zh: bool, timeout: float) -> list[str]:
    """Instant Answer then lite HTML. Raises OSError only when nothing could be fetched at all."""
    snippets: list[str] = []
    q = urllib.parse.quote_plus(query)
    errors: list[Exception] = []
    try:
        raw = await _fetch_bytes(f"{_INSTANT_URL}?q={q}&format=json&no_redirect=1&no_html=1&skip_disambig=1", timeout)
        snippets = _snippets_from_instant(json.loads(raw.decode("utf-8", errors="replace")))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError, asyncio.TimeoutError) as e:
        errors.append(e)
    if len(snippets) < max_results:
        try:
            lite_html = (await _fetch_bytes(f"{_LITE_URL}?q={q}", timeout)).decode("utf-8", errors="replace")
            for line in _snippets_from_lite(lite_html, max_results):
                if line not in snippets:
                    snippets.append(line)
        except (OSError, urllib.error.URLError, UnicodeDecodeError, asyncio.TimeoutError) as e:
            errors.append(e)
    if not snippets and len(errors) >= 2:
        raise OSError("duckduckgo unreachable")
    return snippets


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    query = (args.get("query") or "").strip()
    if not query:
        return "请告诉我要搜索什么。" if _is_zh(str(args.get("lang") or "")) else "Please tell me what to search for."

    max_results = _clamp_max(args.get("max_results"), 3)
    zh = _is_zh(query) or args.get("lang") == "zh"

    budget = _budget_s()
    t_end = time.monotonic() + budget
    per_source = max(0.5, budget - 0.5)
    providers = [("bing", _bing), ("duckduckgo", _ddg)] if zh else [("duckduckgo", _ddg), ("bing", _bing)]
    tasks = [(name, asyncio.ensure_future(fn(query, max_results, zh, per_source))) for name, fn in providers]
    snippets: list[str] = []
    errored = empty = 0
    try:
        for name, task in tasks:                     # priority order, all already running
            left = t_end - time.monotonic()
            try:
                if left > 0:
                    got = await asyncio.wait_for(asyncio.shield(task), left)
                elif task.done() and not task.cancelled():
                    got = task.result()
                else:
                    raise asyncio.TimeoutError
            except asyncio.TimeoutError:
                errored += 1
                continue
            except Exception:  # noqa: BLE001  (a provider that broke is just not available)
                errored += 1
                continue
            if got:
                snippets = list(got)
                break
            empty += 1
    finally:
        for _, task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)

    if not snippets:
        if empty:
            return _no_results(query, zh=zh)
        return _network_error(zh)

    snippets = snippets[:max_results]
    intro = f"关于「{query}」，" if zh else f"On '{query}', "
    body = "；".join(snippets) if zh else "; ".join(snippets)
    return intro + body + ("。" if zh else ".")