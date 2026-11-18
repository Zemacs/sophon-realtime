"""v0.1.0-rc2 engineering rule: no keyword / regex / word list decides what a user utterance MEANS (deplux AGENTS.md hard constraint).

A static guard on the modules that used to hold such judgements: ``coordinator/intent.py`` must stay free of regular expressions and
word lists; the gateway's no-delegate net must not read the user's words. (Legacy ``route.heuristic_*`` / ``slot_edit`` /
``memory_gate.EXPLICIT_SAVE_HINTS`` are a separate, listed removal item: see docs/v0.1.0-critic-response.md.)
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))


class NoSemanticRulesInIntent(unittest.TestCase):
    def test_intent_imports_no_regex_and_has_no_pattern_tables(self) -> None:
        src = (_LIVE_ROOT / "sophon_live/coordinator/intent.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("re", imported)
        names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        for forbidden in ("_PATTERNS", "_REALTIME_RE", "_STRONG_VETO", "_WEAK_VETO", "_NOW_WIDE", "_ENTITY", "_QUESTION", "_PERSONAL", "_NARRATIVE", "_STABLE"):
            self.assertNotIn(forbidden, names)
        for gone in ("required_tools", "is_realtime_fact", "is_tool_scoped", "pad_for"):
            self.assertFalse(hasattr(__import__("sophon_live.coordinator.intent", fromlist=["x"]), gone), gone)

    def test_the_only_callers_of_legacy_keyword_routers_are_the_known_ones(self) -> None:
        """Nothing new may call heuristic_control / heuristic_route / slot_edit (they are legacy, scheduled for removal)."""
        callers = {}
        for p in (_LIVE_ROOT / "sophon_live").rglob("*.py"):
            t = p.read_text(encoding="utf-8")
            for fn in ("heuristic_control(", "heuristic_route(", "slot_edit("):
                if fn in t and p.name != "route.py":
                    callers.setdefault(p.name, set()).add(fn)
        self.assertEqual(callers, {"api.py": {"heuristic_control(", "slot_edit("}})


if __name__ == "__main__":
    unittest.main()
