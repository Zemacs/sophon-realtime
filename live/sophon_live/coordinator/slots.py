"""Slot words the Coordinator can recognise without a model (cities, relative days).

Used for one narrow, deterministic judgement: is this utterance the *same request* as the latest task with
only a slot changed (上海→北京, 明天→后天)? That is a REFINE, whatever JEV thinks. Keep the city list in sync
with ``assistant_tools.weather.CITIES`` (a test enforces it).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

CITY_NAMES: tuple[str, ...] = (
    "北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", "西安", "重庆", "天津", "苏州",
    "香港", "台北", "东京", "纽约", "伦敦", "巴黎", "墨尔本", "悉尼", "新加坡",
)
DAY_WORDS: tuple[str, ...] = ("大后天", "后天", "明天", "今天", "今晚", "明早", "昨天")

_PUNCT = re.compile(r"[\s，,。.！!？?、；;：:\"'“”‘’（）()\-—…]+")
_FILLER = re.compile(r"[的呢呀啊吧嘛哦]")
_SLOT_PREFIX = re.compile(r"^(改成|换成|改为|那|就|还是|不对|不是)+")


def find(text: str, words: tuple[str, ...]) -> list[str]:
    """Distinct slot words in order of appearance (longest-first so 大后天 does not also hit 后天)."""
    hits: list[tuple[int, str]] = []
    taken = [False] * len(text)
    for w in sorted(words, key=len, reverse=True):
        for m in re.finditer(re.escape(w), text):
            if not any(taken[m.start() : m.end()]):
                for i in range(m.start(), m.end()):
                    taken[i] = True
                hits.append((m.start(), w))
    out: list[str] = []
    for _, w in sorted(hits):
        if w not in out:
            out.append(w)
    return out


def skeleton(text: str) -> str:
    """The request with its slot words, punctuation and softeners removed."""
    t = text or ""
    for w in sorted(CITY_NAMES + DAY_WORDS, key=len, reverse=True):
        t = t.replace(w, "")
    t = _PUNCT.sub("", t)
    t = _SLOT_PREFIX.sub("", t)
    return _FILLER.sub("", t)


def similarity(a: str, b: str) -> float:
    sa, sb = skeleton(a), skeleton(b)
    if not sa and not sb:
        return 1.0
    return SequenceMatcher(None, sa, sb).ratio()
