"""Round 4k / R-3 static guards: nothing in the tool / failure code may judge text by a pattern or a literal.

* Every regular expression in web_search / news / weather / orchestrator is pinned to the exact set below (HTML parsing, whitespace, a ``<think>`` block, a
  CJK-script test for the output language, digits of a temperature). A new pattern (say, one that drops snippets containing 广告) fails this test; adding
  one on purpose means changing the pin in review.
* No ``<Chinese literal> in <something>`` anywhere in the judging modules.
* Behaviour: a snippet is never dropped for what it says (广告 / 推广 snippets are kept as written).
* The AST whitelist covers EVERY function of orchestrator.py that reads a tool's text or toolmeta: ``_tool_failure_code``, ``_tool_failure_text``, ``run_job``.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import web_search  # noqa: E402

PKG = _LIVE_ROOT / "sophon_live"
PINS = {'web_search': ['<[^>]+>', '<div class="b_caption"[^>]*>[\\s\\S]*?<p[^>]*>([\\s\\S]*?)</p>', '<h2[^>]*>\\s*<a[^>]*>([\\s\\S]*?)</a>', '<li class="b_algo"[\\s\\S]*?</li>', '<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>([\\s\\S]*?)</p>', '[一-鿿]', '\\s+', 'class=[\'\\"]result-link[\'\\"][^>]*>([^<]+)</a>', 'class=[\'\\"]result-snippet[\'\\"][^>]*>([^<]+)'], 'news': [], 'weather': ['-?\\d+', '[一-鿿]'], 'orchestrator': ['^<think>[\\s\\S]*?</think>']}


def _tree(rel: str) -> ast.AST:
    return ast.parse((PKG / rel).read_text(encoding="utf-8"))


def _re_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "re"]


class RegexPatternsArePinned(unittest.TestCase):
    def test_each_module_uses_exactly_its_documented_patterns(self) -> None:
        for mod, want in PINS.items():
            calls = _re_calls(_tree(f"assistant_tools/{mod}.py"))
            self.assertTrue(all(c.args and isinstance(c.args[0], ast.Constant) for c in calls), f"{mod}: a pattern that is not a literal")
            self.assertEqual(sorted({ast.literal_eval(c.args[0]) for c in calls}), want, mod)

    def test_no_pattern_is_built_or_compiled_somewhere_else(self) -> None:
        for mod in PINS:
            src = (PKG / "assistant_tools" / f"{mod}.py").read_text(encoding="utf-8")
            self.assertNotIn("__import__", src, mod)
            self.assertNotIn("importlib", src, mod)

    def test_the_guard_notices_an_added_pattern(self) -> None:
        tree = ast.parse('import re\nif re.search("广告|推广", snippet): pass')
        self.assertEqual([ast.literal_eval(c.args[0]) for c in _re_calls(tree)], ["广告|推广"])
        self.assertNotIn("广告|推广", PINS["web_search"])


class CorePatternsAndTablesArePinned(unittest.TestCase):
    """N-8: the regular expressions and the literal tables of grounding.py (the module that decides whether a value was said) are pinned the same way.
    A new pattern, or a changed spoken-form table, fails here until the pin is changed in review. Nothing else in the judging modules may use ``re`` at all."""
    GROUNDING_PATTERNS = ["(?:the\\s+)?(\\d{1,2}|[a-z]+)(?:st|nd|rd|th)?\\s+of\\s+(january|february|march|april|may|june|july|august|september|october|november|december)", "(?=.*[A-Za-z])[A-Za-z0-9@._\\-]+", "(\\d+)\\s+(\\d)\\b", "(\\w)['’](\\w)", "(january|february|march|april|may|june|july|august|september|october|november|december)\\s+(?:the\\s+)?(\\d{1,2}|[a-z]+(?:\\s[a-z]+)?)(?:st|nd|rd|th)?(?:,?\\s*(\\d{4}))?", "[,$]", "[A-Za-z]+|\\d+", "[\\s,]+", "[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]+", "[^\\w@.]", "[^a-z0-9@._]", "[a-z0-9._ \\-]+@[a-z0-9. ]+", "[a-z0-9@._]+", "[a-z]+", "[a-z]+|\\d+", "\\$?\\d[\\d,]*(?:\\.\\d+)?", "\\b(?:(twenty|thirty)\\s+)?(\\d{1,2}|[a-z]+)\\s+([a-f])\\b", "\\d{4}-(\\d{2})-(\\d{2})", "\\s+", "f'\\\\b{w}'", "f'\\\\b{w}\\\\b'", "twenty twenty (one|two|three|four|five|six|seven|eight|nine)"]
    GROUNDING_TABLES = {"NUM_WORDS": "e990b103d7c257cf", "_UNITS": "3beaa25643d36892", "_TENS": "60bde3ed5a1d8140", "_SCALE": "a88a1c029caecf4f", "_MONTHS": "ee80feb56c06ba76", "_ORD": "23396ee0f0dc8289", "_CURRENCY": "fc0f099115ad13cf"}

    @staticmethod
    def _patterns(rel: str) -> list[str]:
        out = set()
        for c in _re_calls(_tree(rel)):
            out.add(ast.literal_eval(c.args[0]) if c.args and isinstance(c.args[0], ast.Constant) else ast.unparse(c.args[0]) if c.args else "<no pattern>")
        return sorted(out)

    def test_grounding_uses_exactly_these_patterns(self) -> None:
        self.assertEqual(self._patterns("grounding.py"), self.GROUNDING_PATTERNS)

    def test_the_spoken_form_tables_are_unchanged(self) -> None:
        import hashlib
        import json
        from sophon_live import grounding
        got = {n: hashlib.sha256(json.dumps(getattr(grounding, n), sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16] for n in self.GROUNDING_TABLES}
        self.assertEqual(got, self.GROUNDING_TABLES)

    def test_no_other_judging_module_uses_re(self) -> None:
        for rel in ("realtime.py", "backend.py", "failcodes.py", "coordinator/span_judge.py", "coordinator/jev_client.py", "model_client.py"):
            if (PKG / rel).exists():
                self.assertEqual(self._patterns(rel), [], rel)

    def test_the_guard_notices_an_added_grounding_pattern(self) -> None:
        tree = ast.parse('import re\nWEATHER = re.compile("天气|气温|下雨")')
        self.assertEqual([ast.literal_eval(c.args[0]) for c in _re_calls(tree)], ["天气|气温|下雨"])
        self.assertNotIn("天气|气温|下雨", self.GROUNDING_PATTERNS)         # the old allowlist waved anything in an allowed file through (rc3 G-1 #7)

    def test_the_span_judge_never_reads_the_span_or_the_value_for_meaning(self) -> None:
        """span_judge only builds a question from them: the only operations allowed on span / value / context are dict construction and passing them on."""
        tree = _tree("coordinator/span_judge.py")
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "judge")
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare):
                for side in [node.left, *node.comparators]:
                    self.assertFalse(isinstance(side, ast.Name) and side.id in ("span", "value", "context", "argument"), ast.unparse(node))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(isinstance(node.func.value, ast.Name) and node.func.value.id in ("span", "value", "context", "argument"), ast.unparse(node))


class NoChineseLiteralIsSearchedFor(unittest.TestCase):
    MODULES = ("assistant_tools/web_search.py", "assistant_tools/news.py", "assistant_tools/weather.py", "assistant_tools/orchestrator.py",
               "backend.py", "grounding.py", "failcodes.py", "realtime.py", "coordinator/span_judge.py")

    def test_no_in_comparison_with_a_chinese_literal(self) -> None:
        for rel in self.MODULES:
            for n in ast.walk(_tree(rel)):
                if isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
                    for side in [n.left, *n.comparators]:
                        if isinstance(side, ast.Constant) and isinstance(side.value, str):
                            self.assertFalse(any("\u4e00" <= ch <= "\u9fff" for ch in side.value), f"{rel}: {ast.unparse(n)}")


class SnippetsAreNeverDroppedForWhatTheySay(unittest.TestCase):
    def test_advert_looking_and_other_snippets_are_kept_as_written(self) -> None:
        html = ('<li class="b_algo"><h2><a href="x">推广 标题</a></h2><div class="b_caption"><p class="b_lineclamp2">广告：立即购买，推广价</p></div></li>'
                '<li class="b_algo"><h2><a href="y">Sponsored result</a></h2><div class="b_caption"><p class="b_lineclamp2">ad ad ad</p></div></li>'
                '<li class="b_algo"><h2><a href="z">正常结果</a></h2><div class="b_caption"><p class="b_lineclamp2">3小时前 · 内容</p></div></li>')
        got = web_search._snippets_from_bing(html, 5)
        self.assertEqual(got, ["推广 标题 — 广告：立即购买，推广价", "Sponsored result — ad ad ad", "正常结果 — 3小时前 · 内容"])


def _uses(fn: ast.AST, name: str):
    parents = {c: p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
    return [(n, parents[n]) for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) and n in parents]


class OrchestratorFunctionsReadOnlyStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fns = {n.name: n for n in ast.walk(_tree("assistant_tools/orchestrator.py")) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_tool_failure_text_reads_only_shape_and_the_shape_checks_are_these_three(self) -> None:
        """What to say after a failed required tool: a JSON-looking output or a long / multi-line one falls back to the generic sentence. These are the
        registered text-SHAPE checks (length, newline, a JSON brace); nothing reads what the text says."""
        fn = self.fns["_tool_failure_text"]
        uses = _uses(fn, "t") + _uses(fn, "out")
        self.assertGreaterEqual(len(uses), 5)
        shapes = set()
        for node, parent in uses:
            if isinstance(parent, ast.BoolOp):                                                       # (out or "")
                shapes.add("or")
            elif isinstance(parent, ast.Attribute) and parent.attr == "startswith":
                shapes.add("startswith")
            elif isinstance(parent, ast.Call) and ast.unparse(parent) == "len(t)":
                shapes.add("len")
            elif isinstance(parent, ast.Compare) and ast.unparse(parent) == "'\\n' in t":
                shapes.add("newline")
            elif isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):                  # `not t`: empty
                shapes.add("empty")
            elif isinstance(parent, ast.Return):
                shapes.add("return")
            else:
                self.fail("unexpected read of the tool text: " + ast.unparse(parent))
        self.assertEqual(shapes, {"or", "startswith", "len", "newline", "empty", "return"})
        for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "startswith"):
            self.assertEqual([ast.unparse(a) for a in call.args], ["('{', '[')"])

    def test_tool_failure_code_reads_toolmeta_only_for_its_failure_code(self) -> None:
        fn = self.fns["_tool_failure_code"]
        uses = _uses(fn, "tool_meta")
        self.assertEqual(len(uses), 1)
        node, parent = uses[0]
        self.assertIsInstance(parent, ast.BoolOp)                                                    # (tool_meta or {})
        gets = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "get" and c.func.value is parent]
        self.assertEqual([[ast.unparse(a) for a in c.args] for c in gets], [["'failure'"]])

    def test_the_guard_catches_the_survivors_critic_found(self) -> None:
        """Rewrites that got past the 4j guard: a word probe in _tool_failure_text, a toolmeta text read, str(obj) / json.dumps(obj) probes, len(out), split."""
        t_ok = lambda src: all(isinstance(p, (ast.BoolOp, ast.Return)) or ast.unparse(p) in ("t.startswith", "len(t)", "'\\n' in t")  # noqa: E731
                               for _, p in _uses(ast.parse(src), "t") + _uses(ast.parse(src), "out"))
        self.assertTrue(t_ok("t = (out or '').strip()"))
        for src in ('if "error" in t.lower(): return 1', "if t.split(' ')[0] == 'x': return 1", "x = str(t)", "o = out", "if len(out) < 3: pass"):
            self.assertFalse(t_ok(src), src)


if __name__ == "__main__":
    unittest.main()
