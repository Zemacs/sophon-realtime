"""Number-containment: a number the assistant SPEAKS must come from a source this job really had (code enforces, the model judges).

This is provenance checking, not a reading of what the user meant: every number token in the spoken answer is compared literally to the
numbers in (a) the tool results of this job, (b) the task text and the user's own words. A number found in none of them was not
read from a tool -> the answer is replaced by one fixed honest sentence (``schemas.NUMBERS_UNVERIFIED``).

What counts as "the same number": digits and Chinese numerals are parsed to values; ``1.2万`` also yields 12000; a spoken value matches a
source value when it is equal or equal after rounding the source to the spoken precision (7.1983 ~ 7.2 ~ 7). A lone 一 / 两 / 零 and the adverb 十分 are not checked
(「查一下」「一条新闻」「零下」 are not data); every other numeral is.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10_000, "亿": 100_000_000}
_ARABIC = re.compile(r"\d+(?:[.,]\d+)?")
_CN = re.compile(r"[零〇一二两三四五六七八九十百千]+(?:[万亿][零〇一二两三四五六七八九十百千]*)*")
_DEC_CN = re.compile(r"点[零〇一二三四五六七八九]+")
_ARABIC_BIG = re.compile(r"(\d+(?:\.\d+)?)\s*(万亿|万|亿)")

Num = tuple[float, int]      # (value, decimals the speaker used)


def _unit(u: str) -> float:
    r = 1.0
    for ch in u:
        r *= _BIG[ch]
    return r


def _cn_to_int(s: str) -> float | None:
    total = 0
    section = 0
    num = 0
    seen = False
    for ch in s:
        if ch in _DIGITS:
            num = _DIGITS[ch]
            seen = True
        elif ch in _UNITS:
            section += (num or 1) * _UNITS[ch]
            num = 0
            seen = True
        elif ch in _BIG:
            section += num
            total += (section or 1) * _BIG[ch]
            section = num = 0
            seen = True
        else:
            return None
    return float(total + section + num) if seen else None


def spoken_numbers(text: str) -> list[Num]:
    """Number tokens as a speaker wrote them (lone 一 / 两 / 零 and the adverb 十分 skipped)."""
    out: list[Num] = []
    t = (text or "").replace("百分之", "")
    for m in _ARABIC_BIG.finditer(t):
        base = m.group(1)
        out.append((float(base) * _unit(m.group(2)), 0))
    t = _ARABIC_BIG.sub(" ", t)               # "1.2万亿" was taken whole above
    for m in _ARABIC.finditer(t):
        raw = m.group(0).replace(",", "")
        dec = len(raw.split(".")[1]) if "." in raw else 0
        out.append((float(raw), dec))
    for m in _CN.finditer(t):
        tok = m.group(0)
        if tok in ("一", "两", "零", "〇") or (tok == "十" and t[m.end():m.end() + 1] == "分"):
            continue          # quantifier / 零下 / 十分 uses, not data
        v = _cn_to_int(tok)
        if v is None:
            continue
        dec = 0
        d = _DEC_CN.match(t, m.end())
        if d:
            frac = d.group(0)[1:]
            v += int("".join(str(_DIGITS[c]) for c in frac)) / (10 ** len(frac))
            dec = len(frac)
        out.append((v, dec))
    return out


def source_values(texts: Iterable[str]) -> list[float]:
    vals: list[float] = []
    for t in texts:
        t = t or ""
        vals.extend(v for v, _ in spoken_numbers(t))
        vals.extend(float(x.replace(",", "")) for x in _ARABIC.findall(t))
        # numbers such as "第一条" are in the source as Chinese numerals as well: spoken_numbers covers them (lone 一/两 aside)
        if "一" in t or "两" in t:
            vals.extend((1.0, 2.0))
    return vals


def contained(value: float, dec: int, sources: list[float]) -> bool:
    for s in sources:
        if math.isclose(s, value, rel_tol=1e-9, abs_tol=1e-9):
            return True
        if round(s, dec) == round(value, dec) and abs(s - value) < 0.5 * 10 ** (-dec) + 1e-9:
            return True
    return False


def unverified_numbers(spoken: str, source_texts: Iterable[str]) -> list[float]:
    """Numbers in ``spoken`` that appear in none of ``source_texts`` (empty list = every number is accounted for)."""
    src = source_values(source_texts)
    return [v for v, dec in spoken_numbers(spoken) if not contained(v, dec, src)]
