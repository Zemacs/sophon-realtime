"""Engineering rule (deplux AGENTS.md hard constraint): no keyword / regex / word list decides what a user utterance MEANS.

The model judges (JEV choice questions, the orchestrator's tool calls); code only enforces (provenance, structured return codes,
state-machine versions, number containment). rc3 (Critic RC2-4) enforces this over the WHOLE ``sophon_live`` package with an AST scan.

What the scan reports (``kind``), per (file, enclosing function):
  IMPORT_RE        ``import re`` / ``from re import`` / ``regex`` / ``importlib`` / ``__import__``
  RE_CALL          any ``re.*`` call, or a ``compile/search/match/...`` call whose pattern has CJK in it (f-strings and ``+`` included)
  CJK_COLLECTION   a tuple/list/set/dict literal with >= 2 CJK string constants (word lists, keyword tables)
  CJK_IN           ``"中文" in x`` / ``x in ("中文", ...)``
  CJK_STR_METHOD   ``x.startswith("中文")``, ``.find``, ``.count``, ``.replace`` ... with a CJK literal
  LOWER_IN         ``"error" in x.lower()`` (English keyword test of a text)
  GETATTR_DYNAMIC  ``getattr(obj, <not a literal>)`` (building a legacy name at runtime)
Every hit must be on ALLOW with a reason and a removal version ("permanent" = it is not a semantic judgment: script detection, digit
containment, our own fixed sentences compared for provenance, prompt text, scraping of an external page). ALLOW is keyed by the expression itself
(a digest of it), not by function: a second regex in the same function, or one more word in an allowed table, is a new hit.
A new hit fails the test: route the decision through the model, or add an ALLOW line that a reviewer accepts.

This is an ALARM, not acceptance evidence (Critic rc3 G-1, 112a48b I-1): it is a name-based, not scope-aware tripwire and cannot see every way to hide a word list
(e.g. a word list that arrives from a file or the environment). No PR may cite it as proof that M-1 / G-1 are done.
The older wording: it cannot see every way to hide a word list. The evidence that no rule decides meaning is behavioural:
tests/release_gate.py and the model-judgment sets, not this file.
The scanner itself is tested against the bypass shapes the Critic used on rc2 (``ScannerCatchesBypasses``).
"""
from __future__ import annotations

import ast
import collections
import hashlib
import re
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

_PKG = _LIVE_ROOT / "sophon_live"
_CJK = re.compile(r"[\u3400-\u9fff]")
_WORD = re.compile(r"^[A-Za-z][A-Za-z ]{2,}$")
_RE_FUNCS = {"compile", "search", "match", "fullmatch", "sub", "subn", "findall", "finditer", "split"}
_STR_METHODS = {"startswith", "endswith", "find", "rfind", "count", "index", "rindex", "replace", "removeprefix", "removesuffix", "partition", "rpartition",
                "split", "rsplit", "__contains__", "translate", "join", "strip", "lstrip", "rstrip"}
_RE_MODULES = {"re", "regex", "importlib", "fnmatch", "difflib", "operator", "codecs", "base64", "binascii"}
_OBFUSCATION = {"chr", "fromhex", "b64decode", "unhexlify", "literal_eval", "eval", "exec"}


def _has_cjk(n: ast.AST) -> bool:
    if isinstance(n, ast.Constant):
        return isinstance(n.value, str) and bool(_CJK.search(n.value))
    if isinstance(n, ast.JoinedStr):
        return any(_has_cjk(v) for v in n.values)
    if isinstance(n, ast.BinOp):
        return _has_cjk(n.left) or _has_cjk(n.right)
    return False


def _is_word(n: ast.AST | None) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str) and bool(_WORD.match(n.value))


def _str_consts(n: ast.AST | None):
    """Every string constant that a value is made of: a constant, the elements of a tuple / list / set, the keys of a dict (so a one-word table is seen too)."""
    if isinstance(n, ast.Constant):
        if isinstance(n.value, str):
            yield n.value
    elif isinstance(n, (ast.Tuple, ast.List, ast.Set)):
        for e in n.elts:
            yield from _str_consts(e)
    elif isinstance(n, ast.Dict):
        for e in [*n.keys, *n.values]:
            if e is not None:
                yield from _str_consts(e)
    elif isinstance(n, ast.BinOp):
        yield from _str_consts(n.left)
        yield from _str_consts(n.right)
    elif isinstance(n, ast.JoinedStr):
        for v in n.values:
            yield from _str_consts(v)


def _wordy(n: ast.AST | None) -> bool:
    """A value holding at least ONE CJK or plain-English-word string constant (a singleton table counts: I-1)."""
    return any(_CJK.search(c) or _WORD.match(c) for c in _str_consts(n))


def _name_ids(n: ast.AST | None) -> set[str]:
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} if n is not None else set()


def _tainted_names(tree: ast.AST) -> set[str]:
    """Names that hold a CJK / English-word string constant (or a table of them, even a one-word one), or iterate over one: ``W = "天气"``, ``K = ["天气"]``,
    ``for k in K``, ``for k, v in K.items()``, ``any(c in t for c in "天气")``. A tripwire, not a data-flow engine: it is name-based and not scope-aware."""
    t: set[str] = set()

    def targets(x: ast.AST) -> set[str]:
        return {y.id for y in ast.walk(x) if isinstance(y, ast.Name)}

    def iterates_wordy(it: ast.AST) -> bool:
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("items", "keys", "values"):
            it = it.func.value
        return _wordy(it) or (isinstance(it, ast.Name) and it.id in t)

    for _ in range(4):
        before = len(t)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                if _wordy(n.value) or (isinstance(n.value, ast.Name) and n.value.id in t):
                    for x in tg:
                        t |= targets(x)
            elif isinstance(n, (ast.For, ast.comprehension)) and iterates_wordy(n.iter):
                t |= targets(n.target)
        if len(t) == before:
            break
    return t


def _digest(n: ast.AST) -> str:
    """The identity of one expression: any change to it (a word added to a table, a pattern edited) is a NEW hit that needs a reviewer (Critic rc3 G-1)."""
    return hashlib.sha1(ast.unparse(n).encode("utf-8")).hexdigest()[:8]


def scan_source(rel: str, source: str) -> dict[tuple[str, str, str, str], list[int]]:
    """(file, enclosing function, kind, expression digest) -> line numbers. Kinds:

    IMPORT_RE        import of re / regex / importlib / fnmatch / difflib / operator / codecs / base64 / binascii, or ``__import__`` / ``import_module``
    RE_CALL          any ``re.*`` (or fnmatch / difflib) call, or compile/search/match/... with CJK in the pattern
    CJK_COLLECTION   tuple / list / set / dict (keys OR values) with >= 2 CJK string constants
    WORD_COLLECTION  tuple / list / set with >= 2 plain ASCII-letter string constants, or a dict whose keys are such words and whose values are all constants (an English word list / table)
    CJK_IN / CJK_EQ  ``"中文" in x`` / ``x == "中文"``
    LOWER_IN         ``"error" in x.lower()``
    CJK_STR_METHOD   ``x.startswith("中文")`` / ``"中文 中文".split()`` / ``.partition`` / ``.__contains__`` ... with a CJK literal as argument or receiver
    LOWER_METHOD     ``x.lower().find("weather")`` and the like
    OBFUSCATION      ``chr(...)``, ``bytes.fromhex``, ``b64decode``, ``literal_eval``, ``eval``, ``json.loads("<literal>")``: building a word at run time
    MATCH_STR        a ``match`` statement with a string pattern
    GETATTR_DYNAMIC  ``getattr(obj, <not a literal>)``
    """
    tree = ast.parse(source)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def where(n: ast.AST) -> str:
        names = []
        while n in parents:
            n = parents[n]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(n.name)
        return ".".join(reversed(names)) or "<module>"

    tainted = _tainted_names(tree)
    hits: dict[tuple[str, str, str, str], list[int]] = collections.defaultdict(list)
    for n in ast.walk(tree):
        k = None
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] in _RE_MODULES for a in n.names):
            k = "IMPORT_RE"
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in _RE_MODULES:
            k = "IMPORT_RE"
        elif isinstance(n, (ast.Tuple, ast.List, ast.Set)):
            if sum(_has_cjk(e) for e in n.elts) >= 2:
                k = "CJK_COLLECTION"
            elif sum(_is_word(e) for e in n.elts) >= 2:
                k = "WORD_COLLECTION"
        elif isinstance(n, ast.Dict):
            if sum(_has_cjk(e) for e in [*n.keys, *n.values] if e is not None) >= 2:
                k = "CJK_COLLECTION"
            elif len(n.keys) >= 3 and all(_is_word(e) for e in n.keys) and all(isinstance(v, ast.Constant) for v in n.values):
                k = "WORD_COLLECTION"
        elif isinstance(n, ast.Compare):
            if any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) and any(_has_cjk(x) for x in [n.left, *n.comparators]):
                k = "CJK_IN"
            elif any(isinstance(o, (ast.Eq, ast.NotEq)) for o in n.ops) and any(_has_cjk(x) for x in [n.left, *n.comparators]):
                k = "CJK_EQ"
            elif any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("lower", "casefold") for c in n.comparators):
                k = "LOWER_IN"
            elif any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops) and _is_word(n.left) and any(not isinstance(c, ast.Constant) for c in n.comparators):
                k = "WORD_IN"                               # ``"weather" in t`` (no .lower() needed: I-1)
            elif any(isinstance(o, (ast.In, ast.NotIn, ast.Eq, ast.NotEq)) for o in n.ops) and any(_name_ids(x) & tainted for x in [n.left, *n.comparators]):
                k = "CONST_FLOW"                            # a word constant reached a comparison through a variable / loop target
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Attribute) and n.value.attr in _STR_METHODS:
            k = "METHOD_ALIAS"                              # ``g = t.startswith`` then ``g("天气")``
        elif isinstance(n, ast.Match) and any(isinstance(c, ast.Constant) and isinstance(c.value, str) for case in n.cases for c in ast.walk(case.pattern)):
            k = "MATCH_STR"
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            base = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
            recv = f.value if isinstance(f, ast.Attribute) else None
            if name in ("split", "rsplit") and isinstance(recv, ast.Constant) and isinstance(recv.value, str) and (
                    len(re.findall(r"[A-Za-z]{3,}", recv.value)) >= 3 or _is_word(recv)):
                k = "WORD_COLLECTION"                       # "weather rain temperature".split(), "weather".split()
            elif base in ("re", "regex", "fnmatch", "difflib", "operator") or (name in _RE_FUNCS and n.args and _has_cjk(n.args[0])):
                k = "RE_CALL"
            elif name in _STR_METHODS and ((n.args and _has_cjk(n.args[0])) or (recv is not None and _has_cjk(recv))):
                k = "CJK_STR_METHOD"
            elif name in _STR_METHODS and isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr in ("lower", "casefold") \
                    and n.args and _is_word(n.args[0]):
                k = "LOWER_METHOD"
            elif name in _STR_METHODS and n.args and _is_word(n.args[0]):
                k = "WORD_STR_METHOD"                       # ``t.find("weather")`` (no .lower() needed: I-1)
            elif name in _STR_METHODS and any(_name_ids(x) & tainted for x in [*n.args, *([recv] if recv is not None else [])]):
                k = "CONST_FLOW"
            elif name in ("set", "frozenset", "list", "tuple", "sorted") and isinstance(f, ast.Name) and n.args and isinstance(n.args[0], ast.Constant) and _wordy(n.args[0]):
                k = "WORD_SET_OP"                           # ``set("天气") & set(t)``
            elif name in _OBFUSCATION or any(isinstance(a, ast.Name) and a.id in _OBFUSCATION for a in n.args) or (name == "loads" and base == "json" and n.args and isinstance(n.args[0], ast.Constant)):
                k = "OBFUSCATION"
            elif name == "getattr" and len(n.args) >= 2 and not isinstance(n.args[1], ast.Constant):
                k = "GETATTR_DYNAMIC"
            elif name == "getattr" and len(n.args) >= 2 and isinstance(n.args[1].value, str) and n.args[1].value in _STR_METHODS:
                k = "GETATTR_METHOD"                        # ``getattr(t, "find")("天气")``
            elif name in ("__import__", "import_module"):
                k = "IMPORT_RE"
        if k:
            hits[(rel, where(n), k, _digest(n))].append(n.lineno)
    return hits


def scan_package() -> dict[tuple[str, str, str, str], list[int]]:
    hits: dict[tuple[str, str, str, str], list[int]] = {}
    for p in sorted(_PKG.rglob("*.py")):
        hits.update(scan_source(p.relative_to(_PKG).as_posix(), p.read_text(encoding="utf-8")))
    return hits


_ENUMCMP = ("comparison with a developer-declared enum / protocol field / state name (route, breaker state, tool or source name, event type, schema value) -- one side is "
            "OUR data or the model's closed-option answer, never the user's words (I-1 hit, reviewed)")
_CFGKEY = "protocol / schema / config KEY lookup (``\"format\" in obj``, XML element name, env-var name, request field): a field name, not a word of an utterance (I-1 hit, reviewed)"
_OUT = "chooses OUR fixed output text by a closed flag (language flag, a model enum); no word of the user is read (I-1 hit, reviewed)"
_PROTO = "HTTP header NAMES that mean \"this request was forwarded\"; compared against the request's header names, never against anything the user said (I-5 hit, reviewed)"
_TZ = "timezone / locale NAME from the system configuration to pick a language; not the user's words (I-1 hit, reviewed)"
_CITYTBL = "lookup in the weather tool's own city table (data, listed in wordlist-inventory 2d); the model chose the city (I-1 hit, reviewed)"
_PUNCT = "sentence splitting on punctuation characters of our own output (format) (I-1 hit, reviewed)"
_NOISE = "false positive of the name-based taint (no word constant is compared with text: an id / counter / state comparison); kept on the list so a change is re-read (I-1 hit, reviewed)"
_IMP = "import of a module whose every use below is a hit of its own, listed with its expression digest"
_B64 = "base64 import: decoding an audio payload (bytes), not text"
_B64D = "decoding an audio payload (bytes), not text"
_TORCH = "torch ``module.eval()``: inference mode, not Python eval"
_FMT = "output formatting (strip markup / control characters from a spoken line), not meaning"
_SCRIPT = "script detection (is this text Chinese?) to pick a language for output or a search locale; no word is recognised"
_SCRAPE = "parsing an external page or feed (tool output), not the user's words"
_FIXED = "our own fixed sentences, compared for PROVENANCE (is this exactly a sentence one of our tools wrote)"
_SCHEMA_RE = "patterns of OUR OWN tools' failure sentences (is_tool_failure_sentence): provenance of tool output, not of anything a user said"
_SAID = "whole-word provenance of an entity the tool echoes (S-18/S-23): literal comparison, no meaning"
_PROMPT = "prompt / question text sent to a model; nothing matches against it"
_TABLE = "data table (city names for translation provenance); the model chooses the city, the table only translates / validates it"
_CITY = "strip the administrative suffix 市 before comparing a city NAME with the words the user said (provenance), not a decision about meaning"
_ENUM = "closed set of developer-declared identifiers / enum values / type literals / stage names; nothing matches the user's words against it"
_FORMAT = "format normalisation / tokenising for an exact comparison (NFKC, case, punctuation); no conversion of meaning"
_WHOLE = "whole-token test for exact provenance: a Latin token must occur as a whole word (Daniel is not Danielle); no meaning is read"
_HINT = "DISPLAY ONLY: the spell-it-out hint shows letter / digit runs the caller said; nothing decides on it (Round B: never a provenance candidate)"
# (file, enclosing function, kind, digest of the expression) -> (reason, removal). The digest is of the hit EXPRESSION: editing a table or a pattern,
# or adding a second one beside it, is a new hit and fails the test until a reviewer reads it (Critic rc3 G-1: the old key was (file, function, kind),
# so one allowed regex let any other regex in the same function through). The trailing comment is the first line of the expression, for the reader.
ALLOW: dict[tuple[str, str, str, str], tuple[str, str]] = {
    ('coordinator/api.py', '_admit', 'WORD_IN', '87bdad92'): (_CFGKEY, "permanent"),   # control_key = _pick(resp, "control", meta, asked="control" in questions)
    ('assistant_tools/datetime_tool.py', 'run', 'WORD_IN', '11b19496'): (_TZ, "permanent"),   # zh = (not tz_name) or tz_name.startswith("Asia/") or "Shanghai" in tz_name or "Hong_Kong" 
    ('assistant_tools/news.py', '_strip_publisher', 'WORD_STR_METHOD', '316b3c39'): (_CFGKEY, "permanent"),   # src_el = item.find("source")
    ('assistant_tools/news.py', '_titles_from_rss', 'WORD_STR_METHOD', '56068760'): (_CFGKEY, "permanent"),   # title_el = item.find("title")
    ('assistant_tools/news.py', '_titles_from_rss', 'WORD_STR_METHOD', 'f935fd59'): (_CFGKEY, "permanent"),   # channel = root.find("channel")
    ('assistant_tools/news.py', 'run', 'CONST_FLOW', '2bb67eac'): (_ENUMCMP, "permanent"),   # if topic not in CATEGORIES:                                 # enforcement of the schema's 
    ('assistant_tools/orchestrator.py', '_cfg', 'CONST_FLOW', 'df9c854e'): (_CFGKEY, "permanent"),   # key = os.environ.get("SOPHON_LIVE_ORCH_KEY", _DEFAULT_KEY).strip() or _DEFAULT_KEY
    ('assistant_tools/orchestrator.py', '_ungrounded_city', 'CONST_FLOW', '7ecba9c2'): (_ENUMCMP, "permanent"),   # seen = "\\n".join(str(m.get("content") or "") for m in msgs if m.get("role") in ("user", "
    ('assistant_tools/orchestrator.py', '_ungrounded_city', 'CONST_FLOW', 'c8a578f6'): (_ENUMCMP, "permanent"),   # seen = "\\n".join(str(m.get("content") or "") for m in msgs if m.get("role") in ("user", "
    ('assistant_tools/orchestrator.py', 'run_job._step', 'CONST_FLOW', '482779b6'): (_CFGKEY, "permanent"),   # if status == 400 and ("chat_template_kwargs" in body or "tool_choice" in body):
    ('assistant_tools/orchestrator.py', 'run_job._step', 'CONST_FLOW', 'f2bbbdd8'): (_CFGKEY, "permanent"),   # if status == 400 and ("chat_template_kwargs" in body or "tool_choice" in body):
    ('assistant_tools/weather.py', '_compose', 'CONST_FLOW', '6bb70d3b'): (_ENUMCMP, "permanent"),   # day_label = "明天" if when == "tomorrow" else "今天"
    ('assistant_tools/weather.py', '_first_ok', 'CONST_FLOW', '3c683832'): (_ENUMCMP, "permanent"),   # if name == "open-meteo":
    ('assistant_tools/weather.py', '_first_ok', 'CONST_FLOW', 'f599a544'): (_ENUMCMP, "permanent"),   # if name == "sojson" and display_name not in SOJSON_CODES:
    ('assistant_tools/weather.py', '_first_ok', 'CONST_FLOW', 'f5f1e061'): (_ENUMCMP, "permanent"),   # if name == "sojson" and display_name not in SOJSON_CODES:
    ('assistant_tools/weather.py', '_resolve_city', 'CONST_FLOW', '7401e2d6'): (_CITYTBL, "permanent"),   # if zh_name in raw or en.lower() == raw.lower():
    ('assistant_tools/weather.py', '_resolve_city', 'CONST_FLOW', '9ed221c7'): (_CITYTBL, "permanent"),   # if zh_name in raw or en.lower() == raw.lower():
    ('assistant_tools/weather.py', 'run_detailed', 'CONST_FLOW', '35104f60'): (_CITYTBL, "permanent"),   # if en == query_name:
    ('assistant_tools/weather.py', 'run_detailed', 'CONST_FLOW', '6bb70d3b'): (_ENUMCMP, "permanent"),   # idx = 1 if when == "tomorrow" else 0
    ('assistant_tools/weather.py', 'run_detailed', 'CONST_FLOW', '77c3bb2b'): (_ENUMCMP, "permanent"),   # if when not in ("today", "tomorrow"):
    ('assistant_tools/web_search.py', '_snippets_from_instant.walk', 'WORD_IN', 'ed6342d0'): (_SCRAPE, "permanent"),   # if "Topics" in item:
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '1bf26259'): (_ENUMCMP, "permanent"),   # if route == "FAST":
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '2af6c0bb'): (_ENUMCMP, "permanent"),   # if route == "CLARIFY":
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '3034ea0c'): (_ENUMCMP, "permanent"),   # if route in _ROUTES:
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '604cfb86'): (_ENUMCMP, "permanent"),   # if route in ("CONTROL", "WORKER"):
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '818c5b56'): (_ENUMCMP, "permanent"),   # if route == "STOP_SPEAKING":
    ('coordinator/api.py', '_admit', 'CONST_FLOW', '8faf5c71'): (_ENUMCMP, "permanent"),   # if route == "CONTROL":
    ('coordinator/clarify.py', 'clarify_for_slots', 'CONST_FLOW', '6e51330e'): (_ENUMCMP, "permanent"),   # if s in _ASK:
    ('coordinator/clarify.py', 'slots_from_answer', 'CONST_FLOW', 'be5e7f22'): (_ENUMCMP, "permanent"),   # return [choice] if choice in _ASK else []
    ('coordinator/intent.py', 'explicit_save_from_answer', 'CONST_FLOW', '355cf403'): (_ENUMCMP, "permanent"),   # if choice is None or choice not in TOOL_CRITERIA:
    ('coordinator/intent.py', 'tools_from_answer', 'CONST_FLOW', '355cf403'): (_ENUMCMP, "permanent"),   # if choice is None or choice not in TOOL_CRITERIA:
    ('coordinator/intent.py', 'tools_from_answer', 'CONST_FLOW', 'd551ee40'): (_ENUMCMP, "permanent"),   # if choice == NONE:
    ('gateway.py', 'ASR.__call__', 'CONST_FLOW', '9a1a9c6a'): (_NOISE, "permanent"),   # return self.proc.decode(o[:, inp["input_ids"].shape[1]:], return_format="transcription_onl
    ('gateway.py', 'LiveSession._engine_events', 'CONST_FLOW', '402e4b39'): (_NOISE, "permanent"),   # if rid == self.cur_rid:
    ('gateway.py', 'LiveSession._engine_events', 'CONST_FLOW', '52f9b5c3'): (_NOISE, "permanent"),   # if rid != self.cur_rid or rid not in self.resp:
    ('gateway.py', 'LiveSession._engine_events', 'CONST_FLOW', '5d8db4de'): (_NOISE, "permanent"),   # if rid != self.cur_rid or rid not in self.resp:
    ('gateway.py', 'LiveSession._engine_events', 'CONST_FLOW', '75fa8a0b'): (_NOISE, "permanent"),   # if rid in self.resp:
    ('gateway.py', 'LiveSession._orch_resolve', 'CONST_FLOW', 'd125dc34'): (_OUT, "permanent"),   # text = (text or "").strip() or "好的，我这边查过了。"
    ('gateway.py', 'LiveSession._post_update_locked', 'CONST_FLOW', '9efe89f3'): (_OUT, "permanent"),   # gap_s = ("还没开过口" if self.lang == "zh" else "never") if gap is None else (f"{gap:.0f} 秒" if
    ('gateway.py', 'LiveSession._release_text', 'CONST_FLOW', 'b16315ae'): (_NOISE, "permanent"),   # finished_audio = r["done"] and not any(p[1] == rid for p in self.play) and self.play_buf.s
    ('gateway.py', 'LiveSession._release_text', 'CONST_FLOW', 'fca8e611'): (_NOISE, "permanent"),   # finished_audio = r["done"] and not any(p[1] == rid for p in self.play) and self.play_buf.s
    ('gateway.py', 'LiveSession.on_event', 'CONST_FLOW', '62b3af16'): (_NOISE, "permanent"),   # self.instructions = (self.instructions + "\\n" + str(ev.get("content") or "")).strip()
    ('gateway.py', 'main.process_request', 'WORD_STR_METHOD', '2aeb3941'): (_CFGKEY, "permanent"),   # key = auth[7:].strip() if auth.startswith("Bearer ") else next(
    ('gateway.py', 'split_pieces', 'CONST_FLOW', '01932a7d'): (_PUNCT, "permanent"),   # if text[i] in "。，,、！!？?;；.":
    ('gateway.py', 'split_pieces', 'CONST_FLOW', 'cef5d727'): (_NOISE, "permanent"),   # if text[j - 1] == " " and j - i > 1:
    ('gateway.py', 'split_pieces', 'CONST_FLOW', 'f66debce'): (_PUNCT, "permanent"),   # while j < len(text) and not CJK.match(text[j]) and text[j] not in "。，,、！!？?;；":
    ('grounding.py', 'developer_exempt', 'WORD_IN', '88d53da2'): (_CFGKEY, "permanent"),   # return "default" in prop and str(prop["default"]) == str(value)
    ('grounding.py', 'guard_message', 'CONST_FLOW', '9cbaa21a'): (_NOISE, "permanent"),   # vals = "; ".join(f"{k}={v!r} ({why})" for k, v, why in bad)
    ('model_client.py', 'Breaker.allow', 'CONST_FLOW', '424bd74e'): (_ENUMCMP, "permanent"),   # if self.state == "half_open":
    ('model_client.py', 'Breaker.allow', 'CONST_FLOW', 'b9e79266'): (_ENUMCMP, "permanent"),   # if self.state == "open":
    ('model_client.py', 'Breaker.failure', 'CONST_FLOW', '35ecf629'): (_ENUMCMP, "permanent"),   # if self.state != "open":
    ('model_client.py', 'Breaker.failure', 'CONST_FLOW', '424bd74e'): (_ENUMCMP, "permanent"),   # if self.state == "half_open" or self.fails >= self.threshold:
    ('model_client.py', 'Breaker.snapshot', 'CONST_FLOW', 'b9e79266'): (_ENUMCMP, "permanent"),   # remaining = max(0.0, self.open_s - (self._clock() - self.opened_at)) if self.state == "ope
    ('model_client.py', 'Breaker.success', 'CONST_FLOW', 'd45e05fe'): (_ENUMCMP, "permanent"),   # if self.state != "closed":
    ('realtime.py', 'RealtimeSession._engine_events', 'CONST_FLOW', '3c03cec7'): (_ENUMCMP, "permanent"),   # if self.bo is not None and e.get("name") == "delegate":
    ('realtime.py', 'RealtimeSession._guard', 'CONST_FLOW', '56d5cdd1'): (_NOISE, "permanent"),   # log.info("realtime %s: heard so far: %s", self.id, " || ".join(self.heard[-4:])[-600:])
    ('realtime.py', 'RealtimeSession._guard', 'CONST_FLOW', 'efcc7f3d'): (_CFGKEY, "permanent"),   # if isinstance(args, dict) and grounding.EVIDENCE_KEY in args:
    ('realtime.py', 'RealtimeSession._sys', 'CONST_FLOW', '894260d1'): (_ENUMCMP, "permanent"),   # if tools and self.cfg.get("tool_choice") != "none" and self.bo is not None:
    ('realtime.py', 'RealtimeSession._sys', 'CONST_FLOW', '9efe89f3'): (_OUT, "permanent"),   # rule = ("\\n\\n你的回复会被直接合成为语音说出来：用自然的口语；只写要说出口的话，不要写括号说明、表情或格式符号。" if self.lang == "zh" els
    ('realtime.py', 'RealtimeSession.on_event', 'CONST_FLOW', 'ea32897a'): (_ENUMCMP, "permanent"),   # if item.get("type") == "function_call_output":
    ('realtime.py', 'RealtimeSession.on_update', 'CONST_FLOW', '2028b16a'): (_ENUMCMP, "permanent"),   # if self.cfg.get("tool_choice") == "none":
    ('realtime.py', 'RealtimeSession.on_update', 'CONST_FLOW', '59ef9fba'): (_CFGKEY, "permanent"),   # if k in s and s[k] is not None:
    ('realtime.py', 'RealtimeSession.on_update', 'CONST_FLOW', '628557bf'): (_CFGKEY, "permanent"),   # if k in ain: self.cfg["audio"]["input"][k] = ain[k]
    ('realtime.py', 'RealtimeSession.on_update', 'WORD_IN', '1df2f0d6'): (_CFGKEY, "permanent"),   # if "modalities" in s:                                  # beta name
    ('realtime.py', 'RealtimeSession.on_update', 'WORD_IN', '21a07f4c'): (_CFGKEY, "permanent"),   # if "format" in aout: self.cfg["audio"]["output"]["format"] = aout["format"]
    ('realtime.py', 'RealtimeSession.on_update', 'WORD_IN', 'c5c0e060'): (_CFGKEY, "permanent"),   # if "format" in ain: self.cfg["audio"]["input"]["format"] = ain["format"]
    ('grounding.py', '<module>', 'WORD_COLLECTION', 'decc6564'): (_ENUM, "permanent"),   # CALLER_REPLIES = ("confirm", "reject", "other")
    ('grounding.py', '<module>', 'WORD_COLLECTION', 'fb455303'): (_ENUM, "permanent"),   # JUDGE_EQUIVALENT, JUDGE_CONFIRMS = "equivalent", "confirms"
    ('grounding.py', '_evidence_items', 'WORD_COLLECTION', 'e4dd7f33'): (_ENUM, "permanent"),   # }, "required": ["argument", "span", "value_comes_from_span"]}
    ('grounding.py', 'check.literal', 'RE_CALL', 'd392e94a'): (_WHOLE, "permanent"),   # if re.fullmatch(r"(?=.*[A-Za-z])[A-Za-z0-9@._\-]+", v.strip()):
    ('grounding.py', 'output_failed', 'WORD_COLLECTION', '7f7bf678'): (_ENUM, "permanent"),   # item.get("status") in ("failed", "incomplete"): the client's structured status enum
    ('grounding.py', 'spelled_joins', 'RE_CALL', '1700b69c'): (_HINT, "permanent"),   # text = re.sub(r"(\w)['’](\w)", r"\1\2", text)
    ('grounding.py', 'spelled_joins', 'RE_CALL', 'cfb90f5e'): (_HINT, "permanent"),   # toks = re.findall(r"[A-Za-z]+|\d+", text)
    ('adapters.py', '<module>', 'IMPORT_RE', '5269d420'): (_B64, "permanent"),   # import base64
    ('adapters.py', 'VllmOmniAdapter._recv', 'OBFUSCATION', '8bad184c'): (_B64D, "permanent"),   # b = base64.b64decode(e.get("delta") or "")
    ('adapters.py', 'VllmOmniAdapter.start', 'WORD_COLLECTION', 'acccab84'): (_ENUM, "permanent"),   # self.cfg = {"modalities": ["audio", "text"], "input_audio_format": "pc
    ('assistant_tools/device.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('assistant_tools/device.py', 'run', 'RE_CALL', '8b8027c5'): (_SCRIPT, "permanent"),   # zh = bool(re.search(r"[一-鿿]", device + action + (value_s or "")))
    ('assistant_tools/news.py', '<module>', 'WORD_COLLECTION', '4d4a2736'): (_ENUM, "permanent"),   # CATEGORIES = ("tech", "finance", "world", "general")
    ('assistant_tools/orchestrator.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('assistant_tools/orchestrator.py', '_say', 'RE_CALL', '56eb8f1a'): (_FMT, "permanent"),   # t = re.sub(r"^<think>[\s\S]*?</think>", "", t, flags=re.I).strip()
    ('assistant_tools/orchestrator.py', '_ungrounded_city', 'CJK_STR_METHOD', '79bbd9b1'): (_CITY, "permanent"),   # bare = city.rstrip("市")
    ('assistant_tools/orchestrator.py', '_ungrounded_city', 'WORD_COLLECTION', '84ff218b'): (_ENUM, "permanent"),   # seen = "\n".join(str(m.get("content") or "") for m in msgs if m.get("r
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', '211ec45a'): (_ENUM, "permanent"),   # "max_results": {"type": "integer", "description": "Max snippets (1-5).
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', '26ef3c72'): (_ENUM, "permanent"),   # "topic": {"type": "string", "enum": ["tech", "finance", "world", "gene
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', '270cba04'): (_ENUM, "permanent"),   # "required": ["device", "action"],
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', '30082eb8'): (_ENUM, "permanent"),   # "what": {"type": "string", "enum": ["time", "date", "both"], "descript
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', '5523bc10'): (_ENUM, "permanent"),   # "when": {"type": "string", "enum": ["today", "tomorrow"], "description
    ('assistant_tools/registry.py', '<module>', 'WORD_COLLECTION', 'bf610a44'): (_ENUM, "permanent"),   # "op": {"type": "string", "enum": ["save", "recall", "list"], "descript
    ('assistant_tools/runner.py', '<module>', 'WORD_COLLECTION', '36e41a86'): (_ENUM, "permanent"),   # "get_news": ("topic", "lang"),
    ('assistant_tools/runner.py', '<module>', 'WORD_COLLECTION', '6a517695'): (_ENUM, "permanent"),   # "device_control": ("device", "action"),
    ('assistant_tools/runner.py', '<module>', 'WORD_COLLECTION', '9fcaa764'): (_ENUM, "permanent"),   # "get_weather": ("city", "when"),
    ('assistant_tools/weather.py', '<module>', 'CJK_COLLECTION', '0fba6cc2'): (_TABLE, "permanent"),   # SOJSON_CODES: dict[str, str] = {
    ('assistant_tools/weather.py', '<module>', 'CJK_COLLECTION', '3fc3f3d8'): (_TABLE, "permanent"),   # CITIES: dict[str, str] = {
    ('assistant_tools/weather.py', '<module>', 'CJK_COLLECTION', '647a39ca'): (_TABLE, "permanent"),   # COORDS: dict[str, tuple[float, float]] = {
    ('assistant_tools/weather.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('assistant_tools/weather.py', '_normalize_sojson', 'RE_CALL', 'af58464d'): (_SCRAPE, "permanent"),   # hi = re.search(r"-?\d+", str(day.get("high") or ""))
    ('assistant_tools/weather.py', '_normalize_sojson', 'RE_CALL', 'c1094668'): (_SCRAPE, "permanent"),   # lo = re.search(r"-?\d+", str(day.get("low") or ""))
    ('assistant_tools/weather.py', '_resolve_city', 'RE_CALL', '8c8ecc7d'): (_SCRIPT, "permanent"),   # if re.search(r"[一-鿿]", raw):
    ('assistant_tools/weather.py', '_source_names', 'WORD_COLLECTION', 'b4ee6bcf'): (_ENUM, "permanent"),   # return names or ["sojson", "open-meteo", "wttr"]
    ('assistant_tools/weather.py', '_source_names', 'WORD_COLLECTION', 'e82ab4db'): (_ENUM, "permanent"),   # if x in ("open-meteo", "sojson", "wttr") and x not in names:
    ('assistant_tools/weather.py', 'run_detailed', 'WORD_COLLECTION', '64e6e72f'): (_ENUM, "permanent"),   # if when not in ("today", "tomorrow"):
    ('assistant_tools/web_search.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('assistant_tools/web_search.py', '<module>', 'RE_CALL', '384f4ab2'): (_SCRAPE, "permanent"),   # _TAG = re.compile(r"<[^>]+>")
    ('assistant_tools/web_search.py', '_is_zh', 'RE_CALL', '67b7589d'): (_SCRIPT, "permanent"),   # return bool(re.search(r"[一-鿿]", text))
    ('assistant_tools/web_search.py', '_snippets_from_bing', 'RE_CALL', '184800e1'): (_SCRAPE, "permanent"),   # p = re.search(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>([\s\S]*?)</
    ('assistant_tools/web_search.py', '_snippets_from_bing', 'RE_CALL', '85a86df2'): (_SCRAPE, "permanent"),   # p = re.search(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>([\s\S]*?)</
    ('assistant_tools/web_search.py', '_snippets_from_bing', 'RE_CALL', 'c8f4f1dd'): (_SCRAPE, "permanent"),   # m = re.search(r"<h2[^>]*>\s*<a[^>]*>([\s\S]*?)</a>", block)
    ('assistant_tools/web_search.py', '_snippets_from_bing', 'RE_CALL', 'e5744759'): (_SCRAPE, "permanent"),   # for block in re.findall(r'<li class="b_algo"[\s\S]*?</li>', html):
    ('assistant_tools/web_search.py', '_snippets_from_lite', 'RE_CALL', '2878e0f8'): (_SCRAPE, "permanent"),   # snip_re = re.compile(r"class=['\"]result-snippet['\"][^>]*>([^<]+)", r
    ('assistant_tools/web_search.py', '_snippets_from_lite', 'RE_CALL', 'c4225f23'): (_SCRAPE, "permanent"),   # link_re = re.compile(r"class=['\"]result-link['\"][^>]*>([^<]+)</a>", 
    ('assistant_tools/web_search.py', '_text', 'RE_CALL', '88658718'): (_SCRAPE, "permanent"),   # return re.sub(r"\s+", " ", unescape(_TAG.sub("", fragment))).strip()
    ('coordinator/__init__.py', '<module>', 'WORD_COLLECTION', '6f3760e7'): (_ENUM, "permanent"),   # __all__ = [
    ('coordinator/api.py', '<module>', 'WORD_COLLECTION', '9617a4f6'): (_ENUM, "permanent"),   # _ROUTES = frozenset({"STOP_SPEAKING", "CLARIFY", "FAST", "CONTROL", "W
    ('coordinator/api.py', '<module>', 'WORD_COLLECTION', 'd2f70700'): (_ENUM, "permanent"),   # USE_JEV = os.environ.get("SOPHON_LIVE_JEV", "1").strip().lower() in ("
    ('coordinator/api.py', '_admit', 'WORD_COLLECTION', '16a42bc1'): (_ENUM, "permanent"),   # if route in ("CONTROL", "WORKER"):
    ('coordinator/api.py', '_parse_control_key', 'WORD_COLLECTION', '4d4da6a0'): (_ENUM, "permanent"),   # if len(parts) != 3 or parts[0] not in ("CANCEL", "STATUS", "REFINE"):
    ('coordinator/clarify.py', '<module>', 'CJK_COLLECTION', '3d487e79'): (_PROMPT, "permanent"),   # _ASK = {
    ('coordinator/clarify.py', '<module>', 'CJK_COLLECTION', 'a6bcaf37'): (_PROMPT, "permanent"),   # _MISSING_CRITERIA = {
    ('coordinator/intent.py', '<module>', 'CJK_COLLECTION', '0233333a'): (_PROMPT, "permanent"),   # TOOL_CRITERIA: dict[str, str] = {
    ('coordinator/intent.py', '<module>', 'CJK_COLLECTION', '09750ce7'): (_PROMPT, "permanent"),   # _PAD_BY_TOOL = {
    ('coordinator/memory_gate.py', '<module>', 'WORD_COLLECTION', 'ca2e6dd3'): (_ENUM, "permanent"),   # TASK_SLOT_KEYS = frozenset({"city", "location", "when", "date", "time"
    ('coordinator/route.py', '<module>', 'CJK_COLLECTION', '780bbd7e'): (_PROMPT, "permanent"),   # ROUTE_CRITERIA = {
    ('coordinator/schemas.py', '<module>', 'CJK_COLLECTION', 'eb2ab528'): (_SCHEMA_RE, "permanent"),   # FIXED_FAILURES = frozenset({
    ('coordinator/schemas.py', '<module>', 'CJK_COLLECTION', 'e8807f1d'): (_SCHEMA_RE, "permanent"),   # "web_search": tuple(re.compile(p) for p in (
    ('coordinator/schemas.py', '<module>', 'CJK_COLLECTION', 'f6c3720b'): (_SCHEMA_RE, "permanent"),   # "get_weather": tuple(re.compile(p) for p in (
    ('coordinator/schemas.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('coordinator/schemas.py', '<module>', 'RE_CALL', 'df461e63'): (_SCHEMA_RE, "permanent"),   # "get_weather": tuple(re.compile(p) for p in (
    ('coordinator/schemas.py', 'ControlDecision', 'WORD_COLLECTION', 'e6943c36'): (_ENUM, "permanent"),   # via: Literal["jev", "guard"] = "jev"
    ('coordinator/schemas.py', 'ControlDecision', 'WORD_COLLECTION', '4d4da6a0'): (_ENUM, "permanent"),   # op: Literal["CANCEL", "STATUS", "REFINE"]
    ('coordinator/schemas.py', 'PlaybackReceipt', 'WORD_COLLECTION', 'd292af62'): (_ENUM, "permanent"),   # outcome: Literal["delivered", "interrupted", "abandoned"]
    ('coordinator/schemas.py', 'SpeakPlan', 'WORD_COLLECTION', 'cf99c929'): (_ENUM, "permanent"),   # kind: Literal["pad", "clarify", "answer", "stop_speaking", "noop"]
    ('coordinator/schemas.py', 'WorkerResult', 'WORD_COLLECTION', 'd7d8c25b'): (_ENUM, "permanent"),   # status: Literal["completed", "needs_input", "failed", "cancelled"]
    ('coordinator/schemas.py', '_said_in', 'RE_CALL', '91f72f4b'): (_SAID, "permanent"),   # return re.search(pat, r) is not None
    ('coordinator/schemas.py', '_said_in', 'RE_CALL', 'a0061935'): (_SAID, "permanent"),   # if re.match(r"[a-z0-9]", e[0]):
    ('coordinator/schemas.py', '_said_in', 'RE_CALL', 'd7665f03'): (_SAID, "permanent"),   # if re.match(r"[a-z0-9]", e[-1]):
    ('coordinator/schemas.py', '_said_in', 'RE_CALL', 'f3ca4aa9'): (_SAID, "permanent"),   # pat = re.escape(e)
    ('coordinator/span_judge.py', 'build_question', 'CJK_COLLECTION', '595b6cac'): (_PROMPT, "permanent"),   # "criteria": {
    ('coordinator/span_judge.py', 'build_question', 'CJK_COLLECTION', '8b05e5f1'): (_PROMPT, "permanent"),   # "criteria": {
    ('coordinator/trace.py', '<module>', 'WORD_COLLECTION', 'cdd947db'): (_ENUM, "permanent"),   # STAGES = ("vad_end", "pad_start", "deleg", "route", "worker_start", "w
    ('failcodes.py', '<module>', 'WORD_COLLECTION', 'fdd94390'): (_ENUM, "permanent"),   # SOURCE_ERROR_CODES = ("no_data", "timeout", "network", "http", "bad_da
    ('gateway.py', '<module>', 'CJK_COLLECTION', '678c112d'): (_PROMPT, "permanent"),   # RESULT_FMT = {"zh": ("后端结果（只能根据这段内容回答，用你自己的口语自然地告诉用户，不要提到后端或工具；内容里没有的信
    ('gateway.py', '<module>', 'CJK_COLLECTION', 'd11a05c1'): (_PROMPT, "permanent"),   # UPDATE_PARTS = {"zh": ("要转达给用户的内容：", "后端的进度 / 思考（不用照读）：", "（还没有新内容）"),
    ('gateway.py', '<module>', 'IMPORT_RE', '5269d420'): (_B64, "permanent"),   # import base64
    ('gateway.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('gateway.py', '<module>', 'RE_CALL', '6b6fba6f'): (_SCRIPT, "permanent"),   # CJK = re.compile(r"[㐀-鿿]")
    ('gateway.py', '<module>', 'RE_CALL', 'dc3dd4c0'): (_FMT, "permanent"),   # EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍～~*#`_>]+")
    ('gateway.py', 'ASR.__init__', 'OBFUSCATION', 'ff8da25d'): (_TORCH, "permanent"),   # self.model = transformers.AutoModelForMultimodalLM.from_pretrained(pat
    ('gateway.py', 'LiveSession._coord_resolve', 'WORD_COLLECTION', 'f2457d74'): (_ENUM, "permanent"),   # if result.speak.kind in ("clarify", "answer") and result.speak.text an
    ('gateway.py', 'LiveSession._delegate', 'WORD_COLLECTION', 'd2f70700'): (_ENUM, "permanent"),   # use_coord = os.environ.get("SOPHON_LIVE_COORD", "").strip().lower() in
    ('gateway.py', 'LiveSession._engine_events', 'WORD_COLLECTION', '3a7d3cfd'): (_ENUM, "permanent"),   # if t in ("audio", "text"):
    ('gateway.py', 'LiveSession.on_event', 'OBFUSCATION', '5330d5e1'): (_B64D, "permanent"),   # pcm = np.frombuffer(base64.b64decode(ev.get("audio") or ""), "<i2").as
    ('grounding.py', 'norm_text', 'RE_CALL', '5cf0b726'): (_FORMAT, "permanent"),   # return re.sub(r"[^\w@.]", "", unicodedata.normalize("NFKC", str(s)).lower())
    ('gateway.py', '<module>', 'WORD_COLLECTION', 'e4f7042e'): (_PROTO, "permanent"),   # _PROXY_HEADERS = ("x-forwarded-for", ..., "via", ...)
    ('gateway.py', '_is_local_peer', 'CONST_FLOW', 'dce06465'): (_PROTO, "permanent"),   # any(hdr in headers for hdr in _PROXY_HEADERS)
    ('gateway.py', 'LiveSession.on_start', 'WORD_COLLECTION', '5583496c'): (_ENUM, "permanent"),   # if role in ("user", "assistant"):
    ('gateway.py', '_flag', 'WORD_COLLECTION', 'd2f70700'): (_ENUM, "permanent"),   # return (env if env is not None else os.environ).get(name, "").strip().
    ('gateway.py', 'clean_text', 'RE_CALL', 'e07b1597'): (_FMT, "permanent"),   # t = re.sub(r"\s*\n\s*", "" if lang == "zh" else " ", t)
    ('gateway.py', 'clean_text', 'RE_CALL', 'fcf4a16f'): (_FMT, "permanent"),   # return re.sub(r" {2,}", " ", t)
    ('grounding.py', '<module>', 'IMPORT_RE', 'c307405b'): (_IMP, "permanent"),   # import re
    ('model_client.py', '<module>', 'WORD_COLLECTION', '99027f96'): (_ENUM, "permanent"),   # DEFAULT_BUDGET_S = {"jev": 1.5, "orch": 120.0, "tag": 15.0}
    ('realtime.py', '<module>', 'IMPORT_RE', '5269d420'): (_B64, "permanent"),   # import base64
    ('realtime.py', 'RealtimeSession._engine_events', 'WORD_COLLECTION', '3a7d3cfd'): (_ENUM, "permanent"),   # if t in ("audio", "text"):
    ('realtime.py', 'RealtimeSession.on_update', 'WORD_COLLECTION', '3a7d3cfd'): (_ENUM, "permanent"),   # self.cfg["output_modalities"] = [m for m in s["modalities"] if m in ("
    ('realtime.py', 'decode_audio', 'OBFUSCATION', '7fff1e58'): (_B64D, "permanent"),   # raw = base64.b64decode(b64 or "")
}

_REMOVED_NAMES = ("heuristic_control", "heuristic_route", "slot_edit", "missing_slots_for_text", "has_explicit_save_intent",
                  "EXPLICIT_SAVE_HINTS", "REFINE_GUARD", "run_task", "_builtin_resolve", "_route_task", "CITY_NAMES")


class NoSemanticRulesInTheWholePackage(unittest.TestCase):
    def test_every_hit_is_allowlisted_with_a_reason_and_a_removal_version(self) -> None:
        unexplained = {k: v for k, v in scan_package().items() if k not in ALLOW}
        self.assertEqual(unexplained, {}, "a keyword / regex / word list is a semantic rule: let the model judge, or allowlist it with a "
                                          "reason (see this file's docstring)")

    def test_the_allowlist_has_no_dead_entries_and_every_entry_has_a_reason(self) -> None:
        hits = scan_package()
        self.assertEqual([k for k in ALLOW if k not in hits], [], "remove allowlist entries that no longer match anything")
        for k, (reason, removal) in ALLOW.items():
            self.assertTrue(reason.strip() and removal.strip(), k)

    def test_removed_legacy_names_stay_removed_by_ast_not_by_file_name(self) -> None:
        for p in _PKG.rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                names = []
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(n.name)
                elif isinstance(n, ast.Name):
                    names.append(n.id)
                elif isinstance(n, ast.Attribute):
                    names.append(n.attr)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    names += [a.name.split(".")[-1] for a in n.names] + [a.asname for a in n.names if a.asname]
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in _REMOVED_NAMES:
                    names.append(n.value)             # getattr(x, "slot_edit")
                for name in names:
                    self.assertNotIn(name, _REMOVED_NAMES, f"{p.relative_to(_PKG)}:{getattr(n, 'lineno', '?')}")
        self.assertFalse((_PKG / "coordinator" / "slots.py").exists())

    def test_production_never_imports_the_test_double(self) -> None:
        for p in _PKG.rglob("*.py"):
            self.assertNotIn("jev_stub", p.read_text(encoding="utf-8"), p.name)

    def test_orchestrator_decides_tool_failure_from_the_structured_outcome_not_from_the_sentence(self) -> None:
        src = (_PKG / "assistant_tools/orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("_tool_failure_code(out, tool_meta)", src)


class NoSemanticRulesInIntent(unittest.TestCase):
    def test_intent_has_no_regex_and_no_pattern_tables(self) -> None:
        # intent.py holds two tables of OUTPUT text, nothing that is matched against the user's words: the criteria text of the tool question
        # (sent to the model) and the fixed filler sentence per tool (T-2). Any other hit, or an edit to either table, fails here.
        got = {k[2:] for k in scan_package() if k[0] == "coordinator/intent.py" and k[2] != "CONST_FLOW"}
        self.assertEqual(got, {("CJK_COLLECTION", "0233333a"), ("CJK_COLLECTION", "09750ce7")})
        # I-1: the flow hits are the three comparisons of the model's option KEY with our own option table (``choice in TOOL_CRITERIA`` / ``choice == NONE``)
        flows = sorted(k[1] for k in scan_package() if k[0] == "coordinator/intent.py" and k[2] == "CONST_FLOW")
        self.assertEqual(flows, ["explicit_save_from_answer", "tools_from_answer", "tools_from_answer"])


class ScannerCatchesBypasses(unittest.TestCase):
    """Every shape the Critic used to get past the guard (rc2 ``guard_bypass.py`` and rc3 G-1 ``guard_bypass_rc3.py``) is reported by the scanner."""

    CASES = {
        "from re import search": "IMPORT_RE",
        "import importlib\nimportlib.import_module('re')": "IMPORT_RE",
        "_KEYS = ('天气', '气温', '下雨')": "CJK_COLLECTION",
        "_PATTERNS: dict = {'天气': 1, '气温': 2}": "CJK_COLLECTION",
        "def f(t):\n    return '天气' in t": "CJK_IN",
        "def f(t):\n    return t in ('天气', '气温')": "CJK_COLLECTION",
        "import re\ndef f(t):\n    return re.search('x', t)": "RE_CALL",
        "import regex as re\ndef f(t):\n    return re.search(f'天{1}', t)": "RE_CALL",
        "def f(t):\n    return t.startswith('取消')": "CJK_STR_METHOD",
        "def f(t):\n    return t.count('天' + '气')": "CJK_STR_METHOD",
        "def f(m):\n    return getattr(m, 'slot' + '_edit')": "GETATTR_DYNAMIC",
        "def f(out):\n    return 'error' in out.lower()": "LOWER_IN",
        # rc3 G-1 (14 shapes that passed)
        "WORDS = '天气 气温 下雨'.split()": "CJK_STR_METHOD",
        "TOOLS = {'a': '天气', 'b': '气温'}": "CJK_COLLECTION",
        "W = {'weather', 'rain', 'temperature'}": "WORD_COLLECTION",
        "def g(t):\n    return set(t.split()) & {'weather', 'rain'}": "WORD_COLLECTION",
        "W = 'weather rain temperature'.split()": "WORD_COLLECTION",
        "def g(t):\n    return t.__contains__('天气')": "CJK_STR_METHOD",
        "_W = chr(22825) + chr(27668)": "OBFUSCATION",
        "_W = bytes.fromhex('e5a4a9e6b094').decode()": "OBFUSCATION",
        "def g(t):\n    return t.partition('天气')[1]": "CJK_STR_METHOD",
        "import operator\ndef g(t):\n    return operator.contains(t, '天气')": "RE_CALL",
        "def g(t):\n    match t:\n        case '天气':\n            return 1": "MATCH_STR",
        "import fnmatch\ndef g(t):\n    return fnmatch.fnmatch(t, '*天气*')": "RE_CALL",
        "import json\n_L = json.loads('[\"天气\",\"气温\"]')": "OBFUSCATION",
        "def g(t):\n    return t.lower().find('weather') >= 0": "LOWER_METHOD",
        "def g(t):\n    return t == '天气'": "CJK_EQ",
        "import base64\n_W = base64.b64decode('5aSp5rCU').decode()": "OBFUSCATION",
        # 112a48b I-1 (Critic's bypass112.py: shapes that passed in R-A)
        "W = '天气'\ndef f(t):\n    return W in t": "CONST_FLOW",
        "W = '天气'\ndef f(t):\n    return t.find(W) >= 0": "CONST_FLOW",
        "def f(t):\n    return 'weather' in t": "WORD_IN",
        "def f(t):\n    return t.find('weather')": "WORD_STR_METHOD",
        "K = ['天气']\ndef f(t):\n    return any(k in t for k in K)": "CONST_FLOW",
        "K = {'天气': 'get_weather'}\ndef f(t):\n    return [v for k, v in K.items() if k in t]": "CONST_FLOW",
        "def f(t):\n    return any(c in t for c in '天气')": "CONST_FLOW",
        "def f(t):\n    return bool(set(t) & set('天气'))": "WORD_SET_OP",
        "def f(t):\n    return getattr(t, 'find')('天气')": "GETATTR_METHOD",
        "def f(t):\n    m = getattr(t, 'startswith')\n    return m('天')": "GETATTR_METHOD",
        "def f(t):\n    g = t.startswith\n    return g('天气')": "METHOD_ALIAS",
        "W = '天气'\ndef f(t):\n    return t == W": "CONST_FLOW",
        "K = 'weather'.split()": "WORD_COLLECTION",
        "def f(t):\n    return ''.join(map(chr, [22825, 27668])) in t": "OBFUSCATION",
    }

    def test_each_bypass_is_reported(self) -> None:
        for src, kind in self.CASES.items():
            kinds = {k[2] for k in scan_source("x.py", src)}
            self.assertIn(kind, kinds, src)

    def test_a_word_that_arrives_from_outside_the_source_is_not_seen_and_nobody_claims_it_is(self) -> None:
        """Critic's ``keyword_in_prompt_file`` sample (``t in K`` with K read from a file / the environment): an AST scan has nothing to see. The guard is an ALARM,
        not acceptance evidence (module docstring); the evidence is the whole-package grep and the behaviour tests."""
        self.assertEqual(dict(scan_source("x.py", "def f(t, K):\n    return t in K\n")), {})

    def test_an_ordinary_loop_over_non_word_data_is_not_a_hit(self) -> None:
        for src in ("def f(xs):\n    return [x for x in xs if x in (1, 2)]\n", "N = 3\ndef f(a):\n    return a == N\n"):
            self.assertEqual(dict(scan_source("x.py", src)), {}, src)

    def test_a_clean_module_reports_nothing(self) -> None:
        self.assertEqual(dict(scan_source("x.py", "def f(a, b):\n    return a + b\n")), {})

    def test_a_new_hit_in_an_existing_file_would_fail_the_allowlist(self) -> None:
        hits = scan_source("coordinator/api.py", "def judge(text):\n    return '天气' in text\n")
        self.assertTrue(all(k not in ALLOW for k in hits) and hits)

    def test_a_second_pattern_beside_an_allowed_one_is_a_new_hit_G1(self) -> None:
        """rc3 G-1 / rc2#7: the old key was (file, function, kind), so one allowed regex let ``WEATHER = re.compile('天气|气温')`` through."""
        path = _PKG / "grounding.py"
        src = path.read_text(encoding="utf-8") + "\nimport re as _r\nWEATHER = _r.compile('天气|气温|下雨')\n"
        new = [k for k in scan_source("grounding.py", src) if k not in ALLOW]
        self.assertTrue(new, "the extra regex in an allowlisted file must be reported")
        self.assertTrue(any(k[2] == "RE_CALL" for k in new))

    def test_editing_an_allowed_table_or_pattern_is_a_new_hit_G1(self) -> None:
        """A word added to an allowed closed set or a widened regex changes the expression digest: it is not the allowed expression any more."""
        for rel, old, new_ in (("grounding.py", '("confirm", "reject", "other")', '("confirm", "reject", "other", "maybe")'),
                                                              ("grounding.py", '[A-Za-z0-9@._\\-]+", v.strip()', '[A-Za-z0-9@._\\-]+|\\d+", v.strip()')):
            src = (_PKG / rel).read_text(encoding="utf-8")
            self.assertIn(old, src, (rel, old))
            new = [k for k in scan_source(rel, src.replace(old, new_, 1)) if k not in ALLOW]
            self.assertTrue(new, (rel, new_))

    def test_the_grounding_literal_path_holds_no_word_list_or_number_table_G1_round_b(self) -> None:
        """Round B: the grounding module's only allowed hits are closed sets of our own identifiers and format / whole-token patterns; a number-word, month or
        currency table (or a pattern that rebuilds spoken numbers, dates or e-mails) would be a new, unallowed hit."""
        for k in scan_package():
            if k[0] == "grounding.py":
                self.assertIn(k, ALLOW, k)
                if k[2] != "IMPORT_RE":
                    self.assertIn(ALLOW[k][0], (_ENUM, _WHOLE, _FORMAT, _HINT, _NOISE, _CFGKEY, _ENUMCMP), k)
        src = (_PKG / "grounding.py").read_text(encoding="utf-8")
        for planted in ('_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}\n',
                        '_M = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")\n', 'import re as _r\n_E = _r.compile(r"\\sat\\s")\n'):
            self.assertTrue([k for k in scan_source("grounding.py", src + "\n" + planted) if k not in ALLOW], planted)

    def test_the_allowlist_is_keyed_by_the_expression_not_the_function(self) -> None:
        self.assertTrue(all(len(k) == 4 and len(k[3]) == 8 for k in ALLOW))

    def test_an_aliased_import_of_a_removed_name_is_seen_by_the_ast_check(self) -> None:
        tree = ast.parse("from x import heuristic_control as _hc")
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
        self.assertIn("heuristic_control", names)


if __name__ == "__main__":
    unittest.main()
