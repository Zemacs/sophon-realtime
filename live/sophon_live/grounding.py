"""Argument grounding for tool calls (Realtime face): a short argument value (name, email, zip, id) must come from somewhere
the agent actually saw -- the caller's words (ASR), an earlier tool result, or the instructions/tool schema. Values the model
made up (or a name it misheard) are sent back to the model instead of executed.

PROVENANCE ONLY (Round 4h / I-20): there is no list of "placeholder" names, values, domains or "free-text" argument keys. A value is
fine when it occurs in what the agent saw (literally, as a whole word, or in a spoken form of it); a value that occurs nowhere is not,
whatever it looks like. So 12345 / John Doe / example.com are allowed when the caller actually said them and blocked when the model made
them up, exactly like any other invented value."""
from __future__ import annotations

import asyncio
import copy
import json
import re
import unicodedata
from typing import Callable

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

# ---------------------------------------------------------------- format normaliser: NFKC, case, punctuation. NOTHING ELSE.
# Full-width -> half-width and case folding (NFKC), then everything that is not a word character, @ . _ is dropped. Applied identically to a value, to the
# caller's words and to a span. There is NO numeral conversion (Critic N-1 / N-7): "is this character a number here?" is a judgment about meaning, and a
# table that makes it turns 四川 into 4 and 一起 into 1 -- a provenance the caller never gave. 八十八 versus 88 is the model's call: the value goes through
# ``grounding_evidence`` (span + the model's statement) and the independent check of (span, value) below.
def norm_text(s: str) -> str:
    return re.sub(r"[^\w@.]", "", unicodedata.normalize("NFKC", str(s)).lower())


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


# ---------------------------------------------------------------- evidence: the MODEL says where a value comes from; a SECOND model checks it; code enforces
def _evidence_items() -> dict:
    return {"type": "object", "properties": {
        "argument": {"type": "string", "description": "Name of the argument this entry is about."},
        "span": {"type": "string", "description": "The caller's words (or a tool result), copied EXACTLY as transcribed, that the value comes from."},
        "value_comes_from_span": {"type": "boolean", "description": "true only if YOU judge the value to be what that span says: the same number, the same name, a "
                                  "homophone or traditional / simplified spelling of it, or a short paraphrase. Never true for something the caller did not say."},
        "read_back_to_caller": {"type": "boolean", "description": "true only if the span is the caller's own reply (their very next words) after you read this value back to them."},
        "caller_reply": {"type": "string", "enum": list(CALLER_REPLIES), "description": "Only with read_back_to_caller: did the caller's reply confirm the value you read back "
                         "(confirm), refuse or correct it (reject), or say something else (other)?"},
    }, "required": ["argument", "span", "value_comes_from_span"]}


CALLER_REPLIES = ("confirm", "reject", "other")      # closed set the model fills; the code only checks for "confirm"
JUDGE_EQUIVALENT, JUDGE_CONFIRMS = "equivalent", "confirms"

Judge = Callable[[str, str, str, str, str], "bool | None"]
"""An independent model's yes/no on one claim: ``judge(kind, argument, value, span, context) -> True | False | None`` (None = unavailable / unusable
answer; treated as a refusal). ``equivalent``: does ``span`` say ``value``? ``confirms``: does ``span`` (the caller's reply) confirm ``context`` (what the
assistant read back)? The code never reads either text for meaning."""


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
                                                               "tool returned), where it comes from. The system checks that the span really was said and asks a second "
                                                               "model whether it says the value.",
                                                               "items": _evidence_items()})
        out.append(t)
    return out


EVIDENCE_RULES_EN = ("\n\nTool arguments: never invent a value, and do not fill an optional argument the caller did not mention (leave it out). When an argument comes from what "
                     "the caller said, list it in `grounding_evidence` with the exact words you took it from (the system checks those words were really said, and a second model "
                     "checks that they say the value). If the caller has not said it, ask. If a call is refused, read the value back to the caller, and when their very next "
                     "words confirm it, call again with those words as the span, read_back_to_caller=true and caller_reply=confirm.")
EVIDENCE_RULES_ZH = ("\n\n工具参数：不要编造任何值；来电者没提到的可选参数不要填（省略即可）。参数来自来电者说的话时，在 `grounding_evidence` 里写明你依据的原话（逐字照抄转写；"
                     "系统会核对这段话确实说过，并由另一个模型核对它说的就是这个值）。来电者没说过就先问。如果调用被拒绝，把这个值向来电者复述一遍；对方的下一句话确认后再调用一次，"
                     "把对方这句话作为 span，并设 read_back_to_caller=true、caller_reply=confirm。")


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


def tool_parameters(tools, name: str):
    """The parameter schema the developer declared for tool ``name`` in the session's tool list (either ``{name, parameters}`` or ``{function: {...}}``)."""
    for t in tools or []:
        f = t.get("function") if isinstance(t, dict) and isinstance(t.get("function"), dict) else t
        if isinstance(f, dict) and f.get("name") == name and isinstance(f.get("parameters"), dict):
            return f["parameters"]
    return None


def _property(schema, key: str):
    """The sub-schema the developer wrote for the argument called ``key`` (searched through nested objects and array items); None when there is none."""
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if isinstance(props, dict):
        if isinstance(props.get(key), dict):
            return props[key]
        for sub in props.values():
            hit = _property(sub, key)
            if hit is not None:
                return hit
    return _property(schema.get("items"), key)


def developer_exempt(schema, key: str, value: str) -> bool:
    """N-6(d): STRUCTURE the developer wrote on the schema, not a guess about the argument: the value equals the schema's own ``default`` (no new information),
    or the property is annotated ``"x-grounding": "none"``. Nothing here depends on the argument's name or type. A required argument with neither needs provenance."""
    prop = _property(schema, key)
    if not prop:
        return False
    if prop.get("x-grounding") == "none":
        return True
    return "default" in prop and str(prop["default"]) == str(value)


def check(name: str, args: dict, heard: list[str], seen: list[str], evidence=None, spoken=None, audit=None, *,
          judge: "Judge | None" = None, pending: "list | None" = None, schema=None, rejected: "dict | None" = None) -> list[tuple[str, str, str]]:
    """Returns [(key, value, why)] for ungrounded values; [] = OK.

    EVERY value is checked, whatever its length or script (a number is never exempt), against ONLY what the caller said (``heard``, ASR turns) and what
    tools returned (``seen``): never the instructions or tool schema (R-7). A value is grounded when
      1. it occurs, after format normalisation (NFKC / case / punctuation, no numeral conversion), in the caller's words or a tool result (or is a spoken
         form of them: digits / dates / emails / spellings), or
      2. the MODEL attached ``grounding_evidence`` for it: a ``span`` that occurs verbatim in the caller's words or a tool result, plus its own statement
         ``value_comes_from_span: true``, AND an independent model (``judge``) answers yes to "does the span say the value" (N-2). No judge, an unavailable
         one or a non-yes answer = refused; a claim waiting for the judge is appended to ``pending`` (``check_async`` resolves those),
      3. (``read_back_to_caller``) the value was refused before (``rejected``), the assistant then said it aloud (``spoken`` = [(text, len(heard) then)]), the
         span is inside the caller's FIRST turn after that, the model's closed-set ``caller_reply`` is ``confirm``, and the judge agrees the reply confirms (N-3).
    Not checked: values with nothing to look up (no digit and under 3 characters: 'TX', one CJK character), and values the DEVELOPER exempted in the tool's
    schema (``default`` equal to the value, or ``x-grounding: none``; N-6). Every acceptance is appended to ``audit``.
    There is no fuzzy matching at all (N-4): a name or identifier the ASR got slightly wrong takes the evidence / read-back path."""
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

    def literal(v: str, vn: str) -> bool:
        if re.fullmatch(r"(?=.*[A-Za-z])[A-Za-z0-9@._\-]+", v.strip()):      # one Latin token (name, email, identifier): the WHOLE token, never a piece of a longer one
            if appears_in_words(unicodedata.normalize("NFKC", v), unicodedata.normalize("NFKC", said)) or vn in joins_n:     # (Daniel is not Danielle, min@x.com is not admin@x.com)
                return True
        elif occurs(vn, corpus) or vn in joins_n:
            return True
        if appears_in_words(v, said):
            return True
        dm = re.fullmatch(r"\d{4}-(\d{2})-(\d{2})", v.strip())
        if dm and (dm.group(1) + dm.group(2)) in joins_n:                  # year usually not said
            return True
        if "@" in vn:                                            # emails are often said as "name at domain dot com"
            loc, _, dom = v.strip().partition("@")                # both halves as whole words ("jon at acme dot com"); never a piece of a longer address
            if loc and dom and appears_in_words(loc, said) and appears_in_words(dom.split(".")[0], said):
                return True
        parts = [norm_text(p) for p in re.split(r"[\s,]+", v) if norm_text(p)]
        return len(parts) > 1 and all(p in words or p in joins_n or occurs(p, corpus) and (any(c.isdigit() for c in p) or not p.isascii()) for p in parts)

    def verdict(kind: str, k: str, v: str, span: str, context: str) -> "bool | None":
        if judge is None:
            if pending is not None and not any(p["kind"] == kind and p["argument"] == k and p["value"] == v and p["span"] == span for p in pending):
                pending.append({"kind": kind, "argument": k, "value": v, "span": span, "context": context})
            return None
        return judge(kind, k, v, span, context)

    def by_evidence(k: str, v: str, vn: str) -> tuple[bool, str]:
        entries = evidence.get(k) or []
        if not entries:
            return False, "no evidence span was given"
        why = "its evidence span does not occur in what the caller said"
        for en in entries:
            span = en.get("span", "")
            span_n = norm_text(span)
            if en.get("value_comes_from_span") is not True:
                why = "the model did not state that the value comes from its span"
                continue
            if not span_n:
                continue
            if en.get("read_back_to_caller") is True:
                if en.get("caller_reply") != "confirm":
                    why = "the caller's reply was not marked as a confirmation (caller_reply must be confirm)"
                    continue
                if rejected is not None and v not in rejected.get(k, ()):
                    why = "this value was not the one that was refused (a read-back confirms the refused value, not another)"
                    continue
                j = max((i for i, (tx, _) in enumerate(spoken) if occurs(vn, norm_text(tx))), default=-1)
                if j < 0:
                    why = "the value was never read back to the caller"
                    continue
                first = spoken[j][1]
                if first >= len(heard):
                    why = "the caller has not replied after the read-back yet"
                    continue
                if not occurs(span_n, norm_text(heard[first])):
                    why = "the span is not the caller's first reply after the read-back"
                    continue
                ans = verdict(JUDGE_CONFIRMS, k, v, span, spoken[j][0])
                if ans is not True:
                    why = ("independent check pending" if judge is None else "the independent check could not confirm that the reply is a confirmation" if ans is None
                           else "the independent check says the reply does not confirm the value")
                    continue
                if audit is not None:
                    audit.append({"argument": k, "value": v, "how": "read_back_confirmed", "span": span, "value_comes_from_span": True, "caller_reply": "confirm", "judge": True})
                return True, ""
            if occurs(span_n, corpus):
                ans = verdict(JUDGE_EQUIVALENT, k, v, span, "")
                if ans is not True:
                    why = ("independent check pending" if judge is None else "the independent check could not tell whether the span says the value (treated as a refusal)" if ans is None
                           else "the independent check says the span does not say this value")
                    continue
                if audit is not None:
                    audit.append({"argument": k, "value": v, "how": "model_span", "span": span, "value_comes_from_span": True, "judge": True})
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
        if literal(v, vn):
            if audit is not None:
                audit.append({"argument": k, "value": v, "how": "literal"})
            continue
        if schema is not None and developer_exempt(schema, k, v):
            if audit is not None:
                audit.append({"argument": k, "value": v, "how": "developer_exempt"})
            continue
        ok, why = by_evidence(k, v, vn)
        if ok:
            continue
        bad.append((k, v, f"not said by the caller ({why})"))
    return bad


async def check_async(name: str, args: dict, heard: list[str], seen: list[str], evidence=None, spoken=None, audit=None, *,
                      judge, schema=None, rejected=None, on_verdict=None) -> list[tuple[str, str, str]]:
    """``check`` with an async independent judge: pass 1 collects every claim waiting for a verdict, the judge answers them concurrently (an exception or a
    timeout is an unavailable judge = refusal), pass 2 decides. ``judge(kind, argument, value, span, context)`` is awaited; ``on_verdict`` gets each result."""
    pending: list = []
    bad = check(name, args, heard, seen, evidence, spoken, None, pending=pending, schema=schema, rejected=rejected)
    if not pending:                                  # no claim needed the judge: decide again, this time recording the audit
        return check(name, args, heard, seen, evidence, spoken, audit, schema=schema, rejected=rejected, judge=lambda *a: None)
    results = await asyncio.gather(*(judge(p["kind"], p["argument"], p["value"], p["span"], p["context"]) for p in pending), return_exceptions=True)
    table = {}
    for p, r in zip(pending, results):
        table[(p["kind"], p["argument"], p["value"], p["span"])] = r if isinstance(r, bool) else None
        if on_verdict is not None:
            on_verdict(p, table[(p["kind"], p["argument"], p["value"], p["span"])])
    return check(name, args, heard, seen, evidence, spoken, audit, schema=schema, rejected=rejected,
                 judge=lambda kind, k, v, span, ctx: table.get((kind, k, v, span)))


def guard_message(bad, heard: list[str]) -> str:
    vals = "; ".join(f"{k}={v!r} ({why})" for k, v, why in bad)
    recent = " | ".join(h.strip() for h in heard[-3:] if h.strip())
    joins = [j for h in heard[-3:] for j in spelled_joins(h)]
    s = (f"NOT EXECUTED. These argument values were not provided by the caller: {vals}. Do not guess or invent values. "
         f"Ask the caller for the missing information (for names and emails, ask them to spell it letter by letter), "
         f"read it back, then call the tool again with exactly what they said. If the caller DID say it in other words (a spoken number, a homophone, a "
         f"short paraphrase), call again and list it in {EVIDENCE_KEY} with their exact words as the span (a second model checks that the span says the value).")
    if recent:
        s += f" Caller's recent words (speech transcript): \"{recent}\"."
    if joins:
        s += f" Spelled-out sequences heard: {', '.join(joins)}."
    return s


def needs_input_message(bad, give_up: bool = False) -> str:
    """The answer to a call the model repeated unchanged after being told its values were not provided: the tool is NOT run for these values. The way
    forward is the caller: read the value back, and the caller's confirming words (their next turn) become the provenance (``read_back_to_caller``)."""
    vals = ", ".join(f"{k}={v!r}" for k, v, _ in bad)
    way = ("Read each value back to the caller (\"I have <value>, is that right?\", or ask them to spell it), wait for their answer, and only then call the tool again: "
           f"put the caller's very next words, copied exactly, as the span in {EVIDENCE_KEY} with value_comes_from_span=true, read_back_to_caller=true and "
           "caller_reply=confirm (confirm / reject / other: what the caller's reply was).")
    if give_up:
        return (f"NOT EXECUTED. Many of your calls were refused in a row in this conversation ({vals}). Stop guessing values the caller has not given. "
                "Ask the caller plainly for the missing information; they may also prefer to end the call. " + way)
    return (f"NOT EXECUTED and will not be. You repeated a call whose values were never provided by the caller ({vals}). " + way + " Do not call it again before they answer.")


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
