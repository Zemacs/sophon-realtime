"""Engineering rule (deplux AGENTS.md hard constraint): no keyword / regex / word list decides what a user utterance MEANS.

The model judges (JEV choice questions, the orchestrator's tool calls); code only enforces (provenance, structured return codes,
state-machine versions, number containment). v0.1.1 enforces this over the WHOLE ``sophon_live`` package, not just intent.py:

* an AST scan finds every Chinese word list (>=3 CJK string literals in one tuple/list/set), every ``"中文" in x`` test and every regex
  / ``re`` call whose pattern has CJK in it;
* each hit must be on the ALLOWLIST below, with the reason it is not a semantic judgment (script detection, number normalisation for
  the containment check, our own fixed sentences compared for provenance, prompt text, display hygiene);
* the removed legacy routers must stay removed; nothing in production may import the test double ``tests/jev_stub``.

A new hit fails the test: either route the decision through the model, or add an allowlist line with a reason a reviewer accepts.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

_PKG = _LIVE_ROOT / "sophon_live"
_CJK = re.compile(r"[\u3400-\u9fff]")
_RE_FUNCS = {"compile", "search", "match", "fullmatch", "sub", "findall", "finditer", "split"}

# (file relative to sophon_live, kind) -> why it is allowed. Kinds: LIST / IN / RE.
ALLOW: dict[tuple[str, str], str] = {
    ("assistant_tools/device.py", "RE"): "script detection (is the text Chinese?) to pick the reply language, not meaning",
    ("assistant_tools/weather.py", "RE"): "script detection to pick the reply language",
    ("assistant_tools/web_search.py", "RE"): "script detection + stripping a '3小时前 ·' date prefix off a snippet (display hygiene of tool output)",
    ("gateway.py", "RE"): "script detection (language of the utterance)",
    ("gateway.py", "LIST"): "prompt text fragments sent to the front model, not matched against anything",
    ("coordinator/grounding.py", "RE"): "number-containment check: normalises spoken Chinese numerals so they can be compared to tool output",
    ("coordinator/grounding.py", "IN"): "number-containment check (numeral parsing)",
    ("coordinator/grounding.py", "LIST"): "number-containment check (numeral parsing)",
    ("coordinator/schemas.py", "LIST"): "our own fixed failure sentences / tool-sentence templates, compared for PROVENANCE (is this exactly a sentence a tool of ours wrote)",
}

_REMOVED_NAMES = ("heuristic_control", "heuristic_route", "slot_edit", "missing_slots_for_text", "has_explicit_save_intent",
                  "EXPLICIT_SAVE_HINTS", "REFINE_GUARD", "run_task", "_builtin_resolve")


def _scan() -> dict[tuple[str, str], list[int]]:
    hits: dict[tuple[str, str], list[int]] = {}
    for p in sorted(_PKG.rglob("*.py")):
        rel = p.relative_to(_PKG).as_posix()
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            kind = None
            if isinstance(n, (ast.Tuple, ast.List, ast.Set)) and len(n.elts) >= 3:
                cjk = [e for e in n.elts if isinstance(e, ast.Constant) and isinstance(e.value, str) and _CJK.search(e.value)]
                if len(cjk) >= 3:
                    kind = "LIST"
            elif isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
                for side in [n.left, *n.comparators]:
                    if isinstance(side, ast.Constant) and isinstance(side.value, str) and _CJK.search(side.value):
                        kind = "IN"
            elif isinstance(n, ast.Call):
                f = n.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in _RE_FUNCS and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) \
                        and _CJK.search(n.args[0].value):
                    kind = "RE"
            if kind:
                hits.setdefault((rel, kind), []).append(n.lineno)
    return hits


class NoSemanticRulesInTheWholePackage(unittest.TestCase):
    def test_every_cjk_word_list_in_test_or_regex_is_allowlisted_with_a_reason(self) -> None:
        unexplained = {k: v for k, v in _scan().items() if k not in ALLOW}
        self.assertEqual(unexplained, {}, "a keyword / regex / word list over Chinese text is a semantic rule: let the model judge, "
                                          "or allowlist it with a reason (see this file's docstring)")

    def test_the_allowlist_has_no_dead_entries(self) -> None:
        hits = _scan()
        self.assertEqual([k for k in ALLOW if k not in hits], [], "remove allowlist entries that no longer match anything")

    def test_removed_legacy_routers_stay_removed(self) -> None:
        for p in _PKG.rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                name = (n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        else n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else "")
                self.assertNotIn(name, _REMOVED_NAMES, f"{p.name}:{getattr(n, 'lineno', '?')}")
        self.assertFalse((_PKG / "coordinator" / "slots.py").exists())

    def test_production_never_imports_the_test_double(self) -> None:
        for p in _PKG.rglob("*.py"):
            self.assertNotIn("jev_stub", p.read_text(encoding="utf-8"), p.name)


class NoSemanticRulesInIntent(unittest.TestCase):
    def test_intent_imports_no_regex_and_has_no_pattern_tables(self) -> None:
        src = (_PKG / "coordinator/intent.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("re", imported)
        names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        for forbidden in ("_PATTERNS", "_REALTIME_RE", "_STRONG_VETO", "_WEAK_VETO", "_NOW_WIDE", "_ENTITY", "_QUESTION", "_PERSONAL", "_NARRATIVE", "_STABLE"):
            self.assertNotIn(forbidden, names)

    def test_orchestrator_decides_tool_failure_from_the_structured_outcome_not_from_the_sentence(self) -> None:
        src = (_PKG / "assistant_tools/orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("_tool_failure_code(out, tool_meta)", src)


if __name__ == "__main__":
    unittest.main()
