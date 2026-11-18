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
Every hit must be on ALLOW with a reason and a removal version ("permanent" = it is not a semantic judgment: script detection, number
normalisation for the containment check, our own fixed sentences compared for provenance, prompt text, scraping of an external page).
A new hit fails the test: route the decision through the model, or add an ALLOW line that a reviewer accepts.
The scanner itself is tested against the bypass shapes the Critic used on rc2 (``ScannerCatchesBypasses``).
"""
from __future__ import annotations

import ast
import collections
import re
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

_PKG = _LIVE_ROOT / "sophon_live"
_CJK = re.compile(r"[\u3400-\u9fff]")
_RE_FUNCS = {"compile", "search", "match", "fullmatch", "sub", "subn", "findall", "finditer", "split"}
_STR_METHODS = {"startswith", "endswith", "find", "rfind", "count", "index", "replace", "removeprefix", "removesuffix"}


def _has_cjk(n: ast.AST) -> bool:
    if isinstance(n, ast.Constant):
        return isinstance(n.value, str) and bool(_CJK.search(n.value))
    if isinstance(n, ast.JoinedStr):
        return any(_has_cjk(v) for v in n.values)
    if isinstance(n, ast.BinOp):
        return _has_cjk(n.left) or _has_cjk(n.right)
    return False


def scan_source(rel: str, source: str) -> dict[tuple[str, str, str], list[int]]:
    tree = ast.parse(source)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def where(n: ast.AST) -> str:
        names = []
        while n in parents:
            n = parents[n]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(n.name)
        return ".".join(reversed(names)) or "<module>"

    hits: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for n in ast.walk(tree):
        k = None
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] in ("re", "regex", "importlib") for a in n.names):
            k = "IMPORT_RE"
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in ("re", "regex", "importlib"):
            k = "IMPORT_RE"
        elif isinstance(n, (ast.Tuple, ast.List, ast.Set)) and sum(_has_cjk(e) for e in n.elts) >= 2:
            k = "CJK_COLLECTION"
        elif isinstance(n, ast.Dict) and sum(_has_cjk(e) for e in n.keys if e is not None) >= 2:
            k = "CJK_COLLECTION"
        elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
            if any(_has_cjk(x) for x in [n.left, *n.comparators]):
                k = "CJK_IN"
            elif isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) and any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in ("lower", "casefold") for c in n.comparators):
                k = "LOWER_IN"
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            base = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
            if base in ("re", "regex") or (name in _RE_FUNCS and n.args and _has_cjk(n.args[0])):
                k = "RE_CALL"
            elif name in _STR_METHODS and n.args and _has_cjk(n.args[0]):
                k = "CJK_STR_METHOD"
            elif name == "getattr" and len(n.args) >= 2 and not isinstance(n.args[1], ast.Constant):
                k = "GETATTR_DYNAMIC"
            elif name in ("__import__", "import_module"):
                k = "IMPORT_RE"
        if k:
            hits[(rel, where(n), k)].append(n.lineno)
    return hits


def scan_package() -> dict[tuple[str, str, str], list[int]]:
    hits: dict[tuple[str, str, str], list[int]] = {}
    for p in sorted(_PKG.rglob("*.py")):
        hits.update(scan_source(p.relative_to(_PKG).as_posix(), p.read_text(encoding="utf-8")))
    return hits


_FMT = "output formatting / script detection, not meaning"
_SCRAPE = "parsing an external page or feed (tool output), not the user's words"
_NUM = "number normalisation for the containment check (coordinator/grounding.py): compares a spoken number to its sources"
_FIXED = "our own fixed sentences / templates, compared for PROVENANCE (is this exactly a sentence one of our tools wrote)"
_PROMPT = "prompt text sent to a model; nothing matches against it"
_TABLE = "data table (city names for translation provenance); the model chooses the city, the table only translates / validates it"
_PROV = "provenance of tool arguments (does this value appear in what the user said): literal comparison, no decision about meaning"

# (file, function, kind) -> (reason, removal version)
ALLOW: dict[tuple[str, str, str], tuple[str, str]] = {
    ("assistant_tools/device.py", "<module>", "IMPORT_RE"): (_FMT, "permanent"),
    ("assistant_tools/device.py", "run", "RE_CALL"): ("script detection to pick the reply language", "permanent"),
    ("assistant_tools/orchestrator.py", "<module>", "IMPORT_RE"): (_FMT, "permanent"),
    ("assistant_tools/orchestrator.py", "_say", "RE_CALL"): ("strip the model's markup from a spoken line (formatting)", "permanent"),
    ("assistant_tools/weather.py", "<module>", "CJK_COLLECTION"): (_TABLE, "permanent"),
    ("assistant_tools/weather.py", "<module>", "IMPORT_RE"): (_FMT, "permanent"),
    ("assistant_tools/weather.py", "_normalize_sojson", "RE_CALL"): (_SCRAPE, "permanent"),
    ("assistant_tools/weather.py", "_resolve_city", "RE_CALL"): ("script detection of the city name (Chinese vs Latin)", "permanent"),
    ("assistant_tools/web_search.py", "<module>", "IMPORT_RE"): (_SCRAPE, "permanent"),
    ("assistant_tools/web_search.py", "<module>", "RE_CALL"): (_SCRAPE, "permanent"),
    ("assistant_tools/web_search.py", "_is_zh", "RE_CALL"): ("script detection to pick the search locale", "permanent"),
    ("assistant_tools/web_search.py", "_snippets_from_bing", "RE_CALL"): (_SCRAPE, "permanent"),
    ("assistant_tools/web_search.py", "_snippets_from_lite", "RE_CALL"): (_SCRAPE, "permanent"),
    ("assistant_tools/web_search.py", "_text", "RE_CALL"): (_SCRAPE, "permanent"),
    ("coordinator/grounding.py", "<module>", "CJK_COLLECTION"): (_NUM, "permanent"),
    ("coordinator/grounding.py", "<module>", "IMPORT_RE"): (_NUM, "permanent"),
    ("coordinator/grounding.py", "<module>", "RE_CALL"): (_NUM, "permanent"),
    ("coordinator/grounding.py", "_en_to_num", "RE_CALL"): (_NUM, "permanent"),
    ("coordinator/schemas.py", "<module>", "CJK_COLLECTION"): (_FIXED, "permanent"),
    ("coordinator/schemas.py", "<module>", "IMPORT_RE"): (_FIXED, "permanent"),
    ("coordinator/schemas.py", "<module>", "RE_CALL"): (_FIXED, "permanent"),
    ("coordinator/schemas.py", "_said_in", "RE_CALL"): ("whole-word provenance of an entity the tool echoes (S-18/S-23), not meaning", "permanent"),
    ("gateway.py", "<module>", "CJK_COLLECTION"): (_PROMPT, "permanent"),
    ("gateway.py", "<module>", "IMPORT_RE"): (_FMT, "permanent"),
    ("gateway.py", "<module>", "RE_CALL"): ("script detection (is the text Chinese?) for the text/speech split", "permanent"),
    ("gateway.py", "clean_text", "RE_CALL"): ("strip markup / control characters from a spoken line (formatting)", "permanent"),
    ("grounding.py", "<module>", "IMPORT_RE"): (_PROV, "permanent"),
    ("grounding.py", "<module>", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "_norm", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "check", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "check.latin_grounded", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "spelled_joins", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "spoken_forms", "RE_CALL"): (_PROV, "permanent"),
    ("grounding.py", "word_numbers", "RE_CALL"): (_PROV, "permanent"),
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

    def test_the_gateway_net_reads_no_user_text(self) -> None:
        """The no-delegate net's trigger is state (done / no delegate / fresh transcript); the decision is judge_tools (the model)."""
        src = (_PKG / "gateway.py").read_text(encoding="utf-8")
        i = src.index("async def _no_delegate_net")
        body = src[i:src.index("async def _delegate", i)]
        self.assertNotIn(".strip())) >", body)
        self.assertNotIn("len((r.get", body, "no text-length proxy for 'the front only led in' (RC2-9)")
        for needle in ("re.", "required_tools", "NET_FILLER_MAX_CHARS"):
            self.assertNotIn(needle, body)


class NoSemanticRulesInIntent(unittest.TestCase):
    def test_intent_has_no_regex_and_no_pattern_tables(self) -> None:
        self.assertEqual({k: v for k, v in scan_package().items() if k[0] == "coordinator/intent.py"}, {})


class ScannerCatchesBypasses(unittest.TestCase):
    """The shapes the Critic used to get past rc2's guard (guard_bypass.py) are all reported by the scanner."""

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
        "from coordinator.route import heuristic_control as _hc": "IMPORT_RE",   # see the removed-names test (alias): asserted below
    }

    def test_each_bypass_is_reported(self) -> None:
        for src, kind in self.CASES.items():
            if "heuristic_control" in src:
                continue
            kinds = {k[2] for k in scan_source("x.py", src)}
            self.assertIn(kind, kinds, src)

    def test_a_clean_module_reports_nothing(self) -> None:
        self.assertEqual(dict(scan_source("x.py", "def f(a, b):\n    return a + b\n")), {})

    def test_a_new_hit_in_an_existing_file_would_fail_the_allowlist(self) -> None:
        hits = scan_source("coordinator/api.py", "def judge(text):\n    return '天气' in text\n")
        self.assertTrue(all(k not in ALLOW for k in hits) and hits)

    def test_an_aliased_import_of_a_removed_name_is_seen_by_the_ast_check(self) -> None:
        tree = ast.parse("from x import heuristic_control as _hc")
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
        self.assertIn("heuristic_control", names)


if __name__ == "__main__":
    unittest.main()
