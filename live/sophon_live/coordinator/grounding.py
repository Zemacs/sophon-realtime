"""Number-containment: a number the assistant SPEAKS must come from a source this job really had (code enforces, the model judges).

This is provenance checking, not a reading of what the user meant: every number token in the spoken answer is compared literally to the
numbers in (a) the tool results of this job, (b) the task text and the user's own words. A number found in none of them was not
read from a tool -> the answer is replaced by one fixed honest sentence (``schemas.NUMBERS_UNVERIFIED``).

ONE normalisation entry (``numbers``) is used for the spoken answer AND for every source, so both sides are read the same way: Arabic
digits, Chinese numerals (``三十八点五`` = 38.5, ``1.2万`` = 12000, ``百分之五`` = 5) and English number words (``twenty`` = 20). A spoken
value matches a source value when it is equal or equal after rounding the source to the spoken precision (7.1983 ~ 7.2 ~ 7); for a
clock reading (a successful get_datetime) a 12-hour spoken hour also matches its 24-hour value (下午三点 ~ 15:10).

Deliberately NOT checked (they are not data in ordinary speech, and they are the same on both sides): a lone 一 (「查一下」「一条」
「第一」) and the adverb 十分; 零 directly before 下 is the sign of 零下, not a number. A lone 两 / 零 elsewhere IS checked against the
sources (RC2-7: 「比昨天高两度」 with no 2 in any source is a fabrication). The old rule "the source mentions 一 or 两 => 1 and 2 are
sourced" is gone.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10_000, "亿": 100_000_000}
_CNCH = "零〇一二两三四五六七八九十百千"
# one regex per token shape; the Chinese decimal part is consumed WITH its integer part (no second token for 点五)
_TOKEN = re.compile(
    r"(?P<abig>\d+(?:\.\d+)?)\s*(?P<bigu>万亿|万|亿)"
    r"|(?P<arab>\d+(?:,\d{3})*(?:\.\d+)?)"
    rf"|(?:百分之)?(?P<cn>[{_CNCH}]+(?:[万亿][{_CNCH}]*)*)(?P<dec>点[零〇一二三四五六七八九]+)?(?P<dunit>万亿|万|亿)?"
    r"|(?P<en>\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)(?:[-\s]+(?:and\s+)?)?)+)",
    re.I,
)
_EN_ATOMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_EN_ATOMS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
_EN_SCALE = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}

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


def _en_to_num(s: str) -> float | None:
    total, cur, seen = 0, 0, False
    for w in re.split(r"[-\s]+", s.lower().strip()):
        if not w or w == "and":
            continue
        if w in _EN_ATOMS:
            cur += _EN_ATOMS[w]
            seen = True
        elif w in _EN_SCALE:
            if w == "hundred":
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * _EN_SCALE[w]
                cur = 0
            seen = True
    return float(total + cur) if seen else None


def numbers(text: str) -> list[Num]:
    """Number tokens of ``text``, the same reading for an answer and for a source."""
    out: list[Num] = []
    t = text or ""
    for m in _TOKEN.finditer(t):
        if m.group("abig"):
            out.append((float(m.group("abig")) * _unit(m.group("bigu")), 0))
        elif m.group("arab"):
            raw = m.group("arab").replace(",", "")
            out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0))
        elif m.group("cn"):
            tok = m.group("cn")
            nxt = t[m.end():m.end() + 1]
            if tok == "一" and not m.group("dec"):
                continue                                   # 查一下 / 一条 / 第一: not data
            if tok == "十" and nxt == "分":
                continue                                   # the adverb 十分
            if tok in ("零", "〇") and nxt == "下":
                continue                                   # 零下: the sign, not a number (the digits after it are checked)
            v = _cn_to_int(tok)
            if v is None:
                continue
            dec = 0
            if m.group("dec"):
                frac = m.group("dec")[1:]
                v += int("".join(str(_DIGITS[c]) for c in frac)) / (10 ** len(frac))
                dec = len(frac)
                if m.group("dunit"):
                    v *= _unit(m.group("dunit"))              # 一点二万亿
                    dec = 0
            out.append((v, dec))
        elif m.group("en"):
            v = _en_to_num(m.group("en"))
            if v is not None:
                out.append((v, 0))
    return out


# kept as the public names the Coordinator and the tests already use
spoken_numbers = numbers


def source_values(texts: Iterable[str]) -> list[float]:
    vals: list[float] = []
    for t in texts:
        vals.extend(v for v, _ in numbers(t or ""))
    return vals


def contained(value: float, dec: int, sources: list[float], *, clock12: bool = False) -> bool:
    cands = [value]
    if clock12 and dec == 0 and 1 <= value <= 12:
        cands.append(value + 12)                           # 下午三点 ~ 15:xx
    for v in cands:
        for s in sources:
            if math.isclose(s, v, rel_tol=1e-9, abs_tol=1e-9):
                return True
            if round(s, dec) == round(v, dec) and abs(s - v) < 0.5 * 10 ** (-dec) + 1e-9:
                return True
    return False


def unverified_numbers(spoken: str, source_texts: Iterable[str], *, clock12: bool = False) -> list[float]:
    """Numbers in ``spoken`` that appear in none of ``source_texts`` (empty list = every number is accounted for)."""
    src = source_values(source_texts)
    return [v for v, dec in numbers(spoken) if not contained(v, dec, src, clock12=clock12)]
