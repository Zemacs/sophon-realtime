"""Argument grounding for tool calls (Realtime face): a short argument value (name, email, zip, id) must come from somewhere
the agent actually saw -- the caller's words (ASR), an earlier tool result, or the instructions/tool schema. Values the model
made up (or a name it misheard) are sent back to the model instead of executed.

PROVENANCE ONLY (Round 4h / I-20): there is no list of "placeholder" names, values, domains or "free-text" argument keys. A value is
fine when it occurs in what the agent saw (literally, as a whole word, or in a spoken form of it); a value that occurs nowhere is not,
whatever it looks like. So 12345 / John Doe / example.com are allowed when the caller actually said them and blocked when the model made
them up, exactly like any other invented value."""
from __future__ import annotations

import copy
import difflib
import json
import re
import unicodedata

from .coordinator.schemas import appears_in_words

NUM_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@._]", "", s.lower())


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALE = {"hundred": 100, "thousand": 1000, "million": 1000000}


def word_numbers(text: str) -> list[str]:
    """'four thousand' -> '4000', 'two hundred fifty' -> '250', '1.5 thousand' ignored; '$4,000' -> '4000'."""
    out = [re.sub(r"[,$]", "", m) for m in re.findall(r"\$?\d[\d,]*(?:\.\d+)?", text)]
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    cur = tot = 0; on = False
    for t in toks + ["<end>"]:
        if t in _UNITS: cur += _UNITS[t]; on = True
        elif t in _TENS: cur += _TENS[t]; on = True
        elif t in _SCALE and on:
            if _SCALE[t] == 100: cur *= 100
            else: tot += cur * _SCALE[t]; cur = 0
        elif t in _SCALE:                       # "a hundred", "thousand dollars"
            if _SCALE[t] == 100: cur = 100
            else: tot = _SCALE[t]
            on = True
        elif t == "and" and on: continue
        else:
            if on: out.append(str(tot + cur))
            cur = tot = 0; on = False
    return out


_MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
        "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}


_CURRENCY = {"dollar": "usd", "buck": "usd", "euro": "eur", "pound": "gbp", "sterling": "gbp", "quid": "gbp", "yen": "jpy",
             "yuan": "cny", "renminbi": "cny", "rmb": "cny", "canadian": "cad", "australian": "aud", "aussie": "aud",
             "swiss": "chf", "franc": "chf", "rupee": "inr", "peso": "mxn", "won": "krw", "kiwi": "nzd", "zealand": "nzd",
             "hong kong": "hkd", "singapore": "sgd", "krona": "sek", "krone": "nok", "real": "brl", "rand": "zar", "ruble": "rub", "lira": "try"}


def spoken_forms(text: str) -> list[str]:
    """Extra normalised candidates from conversational speech: dates -> YYYY-MM-DD / MM-DD, spoken emails, grouped digits."""
    t = text.lower().replace("-", " ")
    out = [c for w, c in _CURRENCY.items() if re.search(rf"\b{w}", t)]
    # dates: "december 14th, 2026" / "december fourteenth twenty twenty six" / "the 14th of december"
    t2 = re.sub(r"twenty twenty (one|two|three|four|five|six|seven|eight|nine)", lambda m: "202" + str(_UNITS[m.group(1)]), t)
    for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:the\s+)?(\d{1,2}|[a-z]+(?:\s[a-z]+)?)(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", t2):
        mo = _MONTHS[m.group(1)]; dtok = m.group(2)
        d = int(dtok) if dtok.isdigit() else None
        if d is None:
            parts = dtok.split()
            if parts[0] in _ORD: d = _ORD[parts[0]]
            elif parts[0] in _TENS and len(parts) > 1 and parts[1] in _ORD: d = _TENS[parts[0]] + _ORD[parts[1]]
        if d:
            out.append(f"{mo:02d}-{d:02d}")
            if m.group(3): out.append(f"{m.group(3)}-{mo:02d}-{d:02d}")
    for m in re.finditer(r"(?:the\s+)?(\d{1,2}|[a-z]+)(?:st|nd|rd|th)?\s+of\s+(january|february|march|april|may|june|july|august|september|october|november|december)", t2):
        dtok = m.group(1); d = int(dtok) if dtok.isdigit() else _ORD.get(dtok)
        if d: out.append(f"{_MONTHS[m.group(2)]:02d}-{d:02d}")
    # seats like "fourteen C" / "14 C"
    for m in re.finditer(r"\b(?:(twenty|thirty)\s+)?(\d{1,2}|[a-z]+)\s+([a-f])\b", t):
        n = m.group(2); v = int(n) if n.isdigit() else _UNITS.get(n, _TENS.get(n))
        if v is not None and m.group(1) and v < 10: v += _TENS[m.group(1)]
        if v: out.append(f"{v}{m.group(3)}")
    # spoken email: "name dot surname thirty one at gmail dot com"
    if " at " in t and " dot " in t:
        e = t
        for w, v in list(_UNITS.items()) + list(_TENS.items()): e = re.sub(rf"\b{w}\b", str(v), e)
        e = re.sub(r"(\d+)\s+(\d)\b", lambda m: str(int(m.group(1)) + int(m.group(2))) if int(m.group(1)) % 10 == 0 and int(m.group(1)) >= 20 else m.group(0), e)
        e = e.replace(" underscore ", "_").replace(" dash ", "-").replace(" at ", "@").replace(" dot ", ".")
        for m in re.finditer(r"[a-z0-9._ \-]+@[a-z0-9. ]+", e):
            out.append(re.sub(r"\s+", "", m.group(0)).strip("."))
    # grouped digits: "eight sixteen" -> 816, "nineteen one twenty two" -> 19122
    toks = re.findall(r"[a-z]+|\d+", t); run = []
    for tok in toks + ["<end>"]:
        v = _UNITS.get(tok, _TENS.get(tok)) if tok != "oh" else 0
        if tok.isdigit(): v = int(tok)
        if v is not None and tok != "<end>":
            if run and tok in _UNITS and run[-1] in _TENS and _UNITS[tok] < 10:
                run[-1] = str(int(_TENS[run[-1]]) + _UNITS[tok]) if False else run[-1]
            run.append(tok)
        else:
            if len(run) >= 2:
                vals, i = [], 0
                while i < len(run):
                    a_ = run[i]
                    if a_ in _TENS and i + 1 < len(run) and run[i + 1] in _UNITS and 0 < _UNITS[run[i + 1]] < 10:
                        vals.append(str(_TENS[a_] + _UNITS[run[i + 1]])); i += 2; continue
                    vals.append(str(_UNITS[a_] if a_ in _UNITS else _TENS[a_] if a_ in _TENS else 0 if a_ == "oh" else int(a_))); i += 1
                out.append("".join(vals))
            run = []
    return out


def spelled_joins(text: str) -> list[str]:
    """'Y, U, S, U, F' -> 'yusuf'; '1 9 1 2 2' / 'one nine one two two' -> '19122'; 'W 2 3 7 8' -> 'w2378'."""
    text = re.sub(r"(\w)['’](\w)", r"\1\2", text)              # it's -> its (not a spelled "s")
    toks = re.findall(r"[A-Za-z]+|\d+", text)
    out, run = [], []
    for t in toks + ["<end>"]:
        tl = t.lower()
        unit = (len(t) == 1 and t.isalpha()) or t.isdigit() and len(t) <= 3 or tl in NUM_WORDS
        if unit and t != "<end>":
            run.append(NUM_WORDS.get(tl, tl))
        else:
            if len(run) >= 3: out.append("".join(run))
            run = []
    return out


def _values(args, key=""):
    if isinstance(args, dict):
        for k, v in args.items(): yield from _values(v, k)
    elif isinstance(args, list):
        for v in args: yield from _values(v, key)
    elif isinstance(args, (str, int)) and not isinstance(args, bool):
        yield key, str(args)


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
EVIDENCE_KEY = "grounding_evidence"      # reserved argument the gateway adds to every tool schema it shows the engine, and strips before the client sees the call

# ---------------------------------------------------------------- format normaliser (Round 4k / R-1): FORMAT conversion only
# Chinese numerals -> Arabic digits (八十八 -> 88, 一号 -> 1号, 三零二 -> 302, 二零二六 -> 2026), full-width -> half-width and case folding (NFKC), then
# everything that is not a word character, @ . _ is dropped. Applied identically to a value, to the caller's words and to a span, so equality is
# decided on one format. It never decides what a number MEANS (十二点 stays 12点, not 12:00) and it knows no names, places or words.
_CN_DIGIT = {"零": 0, "〇": 0, "○": 0, "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2, "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
             "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9}
_CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CN_RUN = re.compile("[" + "".join(list(_CN_DIGIT) + list(_CN_UNIT) + ["万"]) + "]+")


def _cn_to_int(run: str) -> str:
    if not any(c in _CN_DIGIT for c in run):                 # a bare unit (百, 万) is a word, not a number
        return "10" if run in ("十", "拾") else run
    if not any(c in _CN_UNIT or c == "万" for c in run):     # positional digits: 二零二六 -> 2026
        return "".join(str(_CN_DIGIT[c]) for c in run)
    total = section = num = 0
    for c in run:
        if c in _CN_DIGIT:
            num = _CN_DIGIT[c]
        elif c == "万":
            if section + num == 0:                           # a 万 with no number before it ("万一") is a word, not a number
                return run
            total += (section + num) * 10000
            section = num = 0
        else:
            section += (num if num else 1) * _CN_UNIT[c]
            num = 0
    return str(total + section + num)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _CN_RUN.sub(lambda m: _cn_to_int(m.group(0)), s)
    return re.sub(r"[^\w@.]", "", s)


def occurs(value_n: str, corpus_n: str) -> bool:
    """``value_n`` occurs in ``corpus_n`` (both normalised). A number must not be a piece of a longer number: 88 is not in 1988 or 8800."""
    if not value_n:
        return False
    i = corpus_n.find(value_n)
    while i >= 0:
        j = i + len(value_n)
        if not ((value_n[0].isdigit() and i > 0 and corpus_n[i - 1].isdigit()) or (value_n[-1].isdigit() and j < len(corpus_n) and corpus_n[j].isdigit())):
            return True
        i = corpus_n.find(value_n, i + 1)
    return False


def _turns_n(turns) -> str:
    return "|".join(norm_text(x) for x in turns)            # "|" keeps a match from straddling two turns


# ---------------------------------------------------------------- evidence: the MODEL says where a value comes from, code checks the span exists
def _evidence_items() -> dict:
    return {"type": "object", "properties": {
        "argument": {"type": "string", "description": "Name of the argument this entry is about."},
        "span": {"type": "string", "description": "The caller's words (or a tool result), copied EXACTLY as transcribed, that the value comes from."},
        "value_comes_from_span": {"type": "boolean", "description": "true only if YOU judge the value to be what that span says: the same number, the same name, a "
                                  "homophone or traditional / simplified spelling of it, or a short paraphrase. Never true for something the caller did not say."},
        "read_back_to_caller": {"type": "boolean", "description": "true only if the span is the caller's own confirmation (their next words) after you read this value back to them."},
    }, "required": ["argument", "span", "value_comes_from_span"]}


def with_evidence_field(tools: list) -> list:
    """Copy of ``tools`` where every function (except ``delegate``) accepts the optional ``grounding_evidence`` argument."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            out.append(t); continue
        t = copy.deepcopy(t)
        f = t.get("function") if isinstance(t.get("function"), dict) else t
        if f.get("name") != "delegate":
            params = f.setdefault("parameters", {"type": "object", "properties": {}})
            if isinstance(params, dict) and isinstance(params.setdefault("properties", {}), dict):
                params["properties"].setdefault(EVIDENCE_KEY, {"type": "array", "description": "For every argument whose value comes from what the caller said (or a "
                                                               "tool returned), where it comes from. The system only checks that the span really was said.",
                                                               "items": _evidence_items()})
        out.append(t)
    return out


EVIDENCE_RULES_EN = ("\n\nTool arguments: never invent a value. When an argument comes from what the caller said, list it in `grounding_evidence` with the exact words "
                     "you took it from (the system only checks that those words were really said; you judge that they mean the value). If the caller has not said it, ask. "
                     "If a call is refused, read the value back to the caller, and when they confirm, call again with their confirming words as the span and read_back_to_caller=true.")
EVIDENCE_RULES_ZH = ("\n\n工具参数：不要编造任何值。参数来自来电者说的话时，在 `grounding_evidence` 里写明你依据的原话（逐字照抄转写；系统只检查这段话确实说过，"
                     "说的是不是这个值由你判断）。来电者没说过就先问。如果调用被拒绝，把这个值向来电者复述一遍；对方确认后再调用一次，把对方确认的原话作为 span，并设 read_back_to_caller=true。")


def parse_evidence(raw) -> dict[str, list[dict]]:
    """``grounding_evidence`` as the model wrote it -> {argument name: [entries]}; anything malformed is simply not evidence."""
    if isinstance(raw, dict):
        flat = []
        for k, v in raw.items():
            flat += [dict(x, argument=k) for x in (v if isinstance(v, list) else [v]) if isinstance(x, dict)]
        raw = flat
    out: dict[str, list[dict]] = {}
    for it in raw if isinstance(raw, list) else []:
        if isinstance(it, dict) and isinstance(it.get("argument"), str) and isinstance(it.get("span"), str):
            out.setdefault(it["argument"], []).append(it)
    return out


def check(name: str, args: dict, heard: list[str], seen: list[str], evidence=None, spoken=None, audit=None) -> list[tuple[str, str, str]]:
    """Returns [(key, value, why)] for ungrounded values; [] = OK.

    EVERY value is checked, whatever its length or script (a number is never exempt), against ONLY what the caller said (``heard``, ASR turns) and what
    tools returned (``seen``): never the instructions or tool schema (R-7). A value is grounded when
      1. it occurs, after format normalisation, in the caller's words or a tool result (or is a spoken form of them: digits / dates / emails / spellings), or
      2. the MODEL attached ``grounding_evidence`` for it: a ``span`` that occurs verbatim (after the same normalisation) in the caller's words or a tool result,
         and the model's own statement ``value_comes_from_span: true``. Whether span and value are equivalent (八十八 / 88, homophones, 繁体, a short
         paraphrase) is the model's judgment; the code only proves the span exists. Every such acceptance is appended to ``audit``.
      3. (``read_back_to_caller``) the span is the caller's words from a turn AFTER the assistant last said the value aloud (``spoken`` = [(text, len(heard) then)]).
    The only values not checked: nothing to look up (no digit and under 3 characters: 'TX', one CJK character)."""
    evidence = parse_evidence(evidence)
    spoken = spoken or []
    heard_c, seen_c = _turns_n(heard), _turns_n(seen)
    corpus = heard_c + "|" + seen_c
    joins = ([j for h in heard for j in spelled_joins(h)] + [n for h in heard for n in word_numbers(h)]
             + [x for h in heard for x in spoken_forms(h)])
    joins_n = {norm_text(j) for j in joins}
    words = set(re.findall(r"[a-z0-9@._]+", " ".join(list(heard) + list(seen)).lower())) | {j.lower() for j in joins}
    said = "\n".join(list(heard) + list(seen))
    bad = []

    def near(w: str) -> float:
        return max((difflib.SequenceMatcher(None, w, x).ratio() for x in words), default=0.0)

    def literal(v: str, vn: str) -> tuple[bool, float]:
        if occurs(vn, corpus) or vn in joins_n:
            return True, 1.0
        if appears_in_words(v, said):
            return True, 1.0
        dm = re.fullmatch(r"\d{4}-(\d{2})-(\d{2})", v.strip())
        if dm and (dm.group(1) + dm.group(2)) in joins_n:                  # year usually not said
            return True, 1.0
        if "@" in vn:                                            # emails are often said as "name at domain dot com"
            loc, _, dom = vn.partition("@")
            if loc and occurs(loc, corpus) and occurs(dom.split(".")[0], corpus):
                return True, 1.0
        has_digit = any(c.isdigit() for c in vn)
        ascii_ = vn.isascii()
        best = 0.0
        if ascii_ and not has_digit:                             # misheard LETTERS only; a digit string is never fuzzy-matched (4829137 is not 4829130)
            best = near(vn)
            if best >= 0.85:
                return True, best
        parts = [norm_text(p) for p in re.split(r"[\s,]+", v) if norm_text(p)]
        if len(parts) > 1 and all(p in words or p in joins_n or occurs(p, corpus) and (any(c.isdigit() for c in p) or not p.isascii())
                                  or (p.isascii() and not any(c.isdigit() for c in p) and near(p) >= 0.85) for p in parts):
            return True, best
        return False, best

    def by_evidence(k: str, v: str, vn: str) -> tuple[bool, str]:
        entries = evidence.get(k) or []
        if not entries:
            return False, "no evidence span was given"
        why = "its evidence span does not occur in what the caller said"
        for en in entries:
            span_n = norm_text(en.get("span", ""))
            if en.get("value_comes_from_span") is not True:
                why = "the model did not state that the value comes from its span"
                continue
            if not span_n:
                continue
            if en.get("read_back_to_caller") is True:
                j = max((i for i, (tx, _) in enumerate(spoken) if occurs(vn, norm_text(tx))), default=-1)
                if j < 0:
                    why = "the value was never read back to the caller"
                    continue
                if not occurs(span_n, _turns_n(heard[spoken[j][1]:])):
                    why = "the caller said nothing like that after the read-back"
                    continue
                if audit is not None:
                    audit.append({"argument": k, "value": v, "how": "read_back_confirmed", "span": en.get("span"), "value_comes_from_span": True})
                return True, ""
            if occurs(span_n, corpus):
                if audit is not None:
                    audit.append({"argument": k, "value": v, "how": "model_span", "span": en.get("span"), "value_comes_from_span": True})
                return True, ""
        return False, why

    for k, v in _values(args):
        if k == EVIDENCE_KEY:
            continue
        vn = norm_text(v)
        if not vn or (len(vn) < 3 and not any(c.isdigit() for c in vn) and not (len(vn) >= 2 and _CJK.search(vn))):
            continue                                             # nothing to look up
        if len(vn) < 2 and _CJK.search(vn):
            continue
        ok, best = literal(v, vn)
        if ok:
            if audit is not None:
                audit.append({"argument": k, "value": v, "how": "literal"})
            continue
        ok, why = by_evidence(k, v, vn)
        if ok:
            continue
        bad.append((k, v, f"not said by the caller ({why}" + (f"; closest match {best:.2f}" if best else "") + ")"))
    return bad


def guard_message(bad, heard: list[str]) -> str:
    vals = "; ".join(f"{k}={v!r} ({why})" for k, v, why in bad)
    recent = " | ".join(h.strip() for h in heard[-3:] if h.strip())
    joins = [j for h in heard[-3:] for j in spelled_joins(h)]
    s = (f"NOT EXECUTED. These argument values were not provided by the caller: {vals}. Do not guess or invent values. "
         f"Ask the caller for the missing information (for names and emails, ask them to spell it letter by letter), "
         f"read it back, then call the tool again with exactly what they said. If the caller DID say it in other words (a spoken number, a homophone, a "
         f"short paraphrase), call again and list it in {EVIDENCE_KEY} with their exact words as the span.")
    if recent:
        s += f" Caller's recent words (speech transcript): \"{recent}\"."
    if joins:
        s += f" Spelled-out sequences heard: {', '.join(joins)}."
    return s


def needs_input_message(bad, give_up: bool = False) -> str:
    """The answer to a call the model repeated unchanged after being told its values were not provided: the tool is NOT run for these values. The way
    forward is the caller: read the value back, and the caller's confirming words (their next turn) become the provenance (``read_back_to_caller``)."""
    vals = ", ".join(f"{k}={v!r}" for k, v, _ in bad)
    if give_up:
        return (f"NOT EXECUTED. Too many of your calls were refused in this conversation ({vals}). Stop calling tools with values the caller has not given. "
                "Tell the caller plainly that you cannot complete this on your own and ask what they would like to do (they may prefer to end the call).")
    return (f"NOT EXECUTED and will not be. You repeated a call whose values were never provided by the caller ({vals}). Read each value back to the caller "
            "(\"I have <value>, is that right?\", or ask them to spell it), wait for their answer, and only then call the tool again: put the caller's confirming "
            f"words, copied exactly, as the span in {EVIDENCE_KEY} with value_comes_from_span=true and read_back_to_caller=true. Do not call it again before they answer.")


def output_failed(item: dict) -> bool:
    """Did the client report this ``function_call_output`` as a failure? STRUCTURE only; the words of the output are never read.

    Contract (the same one our own tools use, ``failcodes`` / the ``{"ok": false}`` envelope): the item carries a truthy ``error``, or its
    ``status`` is ``failed`` / ``incomplete``; or ``output`` is a JSON object with ``ok: false``, ``success: false`` or a non-empty ``error``.
    An output that merely *says* "not found" or "error" in prose is NOT a reported failure (B-3 / I-20: no word probes)."""
    if not isinstance(item, dict):
        return False
    if item.get("error") or item.get("status") in ("failed", "incomplete"):
        return True
    out = item.get("output")
    if isinstance(out, str) and out.lstrip().startswith("{"):
        try:
            out = json.loads(out)
        except ValueError:
            return False
    if isinstance(out, dict):
        return out.get("ok") is False or out.get("success") is False or bool(out.get("error"))
    return False


def failure_hint(failed: bool, heard: list[str]) -> str:
    """The spell-it-out hint for a tool call the client reported as failed (``output_failed``); nothing for any other output."""
    if not failed:
        return ""
    recent = " | ".join(h.strip() for h in heard[-2:] if h.strip())
    joins = [j for h in heard[-3:] for j in spelled_joins(h)]
    if not recent:
        return ""
    s = f"\n[Caller's recent words (speech transcript): \"{recent}\"."
    if joins: s += f" Spelled-out sequences heard: {', '.join(joins)}."
    return s + " If a name or id may have been misheard, ask the caller to spell it letter by letter and retry with the spelled version.]"


class ToolText(str):
    """A client's ``function_call_output`` text that remembers whether the item reported failure by structure (``output_failed(item)``): the back
    office reads ``.failed`` so ``item.error`` / ``item.status`` count toward MAX_FAILS, not only a JSON ``ok:false`` inside the text (R-4)."""
    failed: bool = False


def tool_text(item: dict) -> ToolText:
    s = ToolText(str((item or {}).get("output", "")))
    s.failed = output_failed(item)
    return s
