"""Argument grounding for tool calls (Realtime face): a short argument value (name, email, zip, id) must come from somewhere
the agent actually saw -- the caller's words (ASR), an earlier tool result, or the instructions/tool schema. Values the model
made up (or a name it misheard) are sent back to the model instead of executed.

PROVENANCE ONLY (Round 4h / I-20): there is no list of "placeholder" names, values, domains or "free-text" argument keys. A value is
fine when it occurs, character for character after format normalisation, in what the agent saw (a Latin token as a whole word); a value that occurs
nowhere is not, whatever it looks like. EVERY other "same value" (a spoken number, a date, an email said aloud, a homophone, a translation) is the model's
claim plus an independent model's yes (``grounding_evidence`` + ``judge``); no yes, a timeout or no judge = not grounded. So 12345 / John Doe / example.com are allowed when the caller actually said them and blocked when the model made
them up, exactly like any other invented value."""
from __future__ import annotations

import asyncio
import copy
import json
import re
import unicodedata
from typing import Callable

from .coordinator.schemas import appears_in_words


def spelled_joins(text: str) -> list[str]:
    """DISPLAY ONLY (the spell-it-out hint of ``failure_hint`` / ``guard_message``): 'Y, U, S, U, F' -> 'yusuf'; '1 9 1 2 2' -> '19122'. Nothing decides
    on it: it is never a candidate for provenance (Round B / Critic G-1); a value the caller spelled out is grounded through the evidence + judge path."""
    text = re.sub(r"(\w)['’](\w)", r"\1\2", text)              # it's -> its (not a spelled "s")
    toks = re.findall(r"[A-Za-z]+|\d+", text)
    out, run = [], []
    for t in toks + ["<end>"]:
        if t != "<end>" and ((len(t) == 1 and t.isalpha()) or (t.isdigit() and len(t) <= 3)):
            run.append(t.lower())
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
      1. it occurs, after format normalisation (NFKC / case / punctuation; no numeral conversion, no spoken forms, no partial email / date / multi-word
         matches), in the caller's words or a tool result (a Latin token as a whole word), or
      2. the MODEL attached ``grounding_evidence`` for it: a ``span`` that occurs verbatim in the caller's words or a tool result, plus its own statement
         ``value_comes_from_span: true``, AND an independent model (``judge``) answers yes to "does the span say the value" (N-2). No judge, an unavailable
         one or a non-yes answer = refused; a claim waiting for the judge is appended to ``pending`` (``check_async`` resolves those),
      3. (``read_back_to_caller``) the value was refused before (``rejected``), the assistant then said it aloud (``spoken`` = [(text, len(heard) then)]), the
         span is inside the caller's FIRST turn after that, the model's closed-set ``caller_reply`` is ``confirm``, and the judge agrees the reply confirms (N-3).
    Not checked: values with nothing left after normalisation (punctuation only), and values the DEVELOPER exempted in the tool's schema (``default`` equal to
    the value, or ``x-grounding: none``; N-6). A short value ('TX', one CJK character) is checked like any other (Round B / G-1). Every acceptance is appended to ``audit``.
    There is no fuzzy matching at all (N-4): a name or identifier the ASR got slightly wrong takes the evidence / read-back path."""
    evidence = parse_evidence(evidence)
    spoken = spoken or []
    heard_c, seen_c = _turns_n(heard), _turns_n(seen)
    corpus = heard_c + "|" + seen_c
    said = "\n".join(list(heard) + list(seen))
    bad = []

    def literal(v: str, vn: str) -> bool:
        """Exact provenance only: the value, after format normalisation, occurs in what was heard / returned. A Latin token must be a WHOLE word
        (Daniel is not Danielle, min@x.com is not admin@x.com); anything else is a plain occurrence with a digit boundary (88 is not in 1988).
        No spoken forms, no number words, no partial email / date / multi-word matches: every other kind of "same value" is the judge's call."""
        if re.fullmatch(r"(?=.*[A-Za-z])[A-Za-z0-9@._\-]+", v.strip()):
            return appears_in_words(unicodedata.normalize("NFKC", v), unicodedata.normalize("NFKC", said))
        return occurs(vn, corpus) or appears_in_words(v, said)

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
        if not vn:
            continue                                             # nothing but punctuation: there is nothing to compare
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
