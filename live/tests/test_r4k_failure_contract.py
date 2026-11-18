"""Round 4k / R-3: the exact structured contract of ``grounding.output_failed`` (what the 4j survivors changed without a test noticing)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding  # noqa: E402


def failed(**item) -> bool:
    return grounding.output_failed(item)


class OutputFailedContract(unittest.TestCase):
    def test_status_failed_and_incomplete_only(self) -> None:
        for st in ("failed", "incomplete"):
            self.assertTrue(failed(output="x", status=st), st)
        for st in ("error", "cancelled", "completed", "in_progress", "FAILED", "", None, 1):
            self.assertFalse(failed(output="x", status=st), st)

    def test_error_field_truthy_only(self) -> None:
        self.assertTrue(failed(output="x", error={"code": "e"}))
        self.assertTrue(failed(output="x", error="boom"))
        for e in (None, "", {}, [], 0, False):
            self.assertFalse(failed(output="x", error=e), e)

    def test_ok_false_is_exactly_false(self) -> None:
        self.assertTrue(failed(output=json.dumps({"ok": False})))
        self.assertTrue(failed(output='{"success": false}'))
        for ok in (None, 0, "", "false", "no", [], {}, True, 1):
            self.assertFalse(failed(output=json.dumps({"ok": ok})), ok)
            self.assertFalse(failed(output=json.dumps({"success": ok})), ok)
        self.assertFalse(failed(output="{}"))
        self.assertFalse(failed(output=json.dumps({"x": 1})))

    def test_json_error_key_must_be_non_empty(self) -> None:
        self.assertTrue(failed(output='{"error": "no such user"}'))
        for e in ('{"error": ""}', '{"error": null}', '{"error": []}', '{"errors": "x"}', '{"status": "error"}', '{"failure": "network"}'):
            self.assertFalse(failed(output=e), e)

    def test_prose_never_fails_and_non_dicts_are_ignored(self) -> None:
        for out in ("Error: not found", "failed", "不存在", "ok: false", '"ok": false', "[1, 2]", "[false]", "{not json", ""):
            self.assertFalse(failed(output=out), out)
        for item in (None, "x", 3, []):
            self.assertFalse(grounding.output_failed(item))

    def test_tool_text_carries_the_item_level_flag(self) -> None:
        self.assertTrue(grounding.tool_text({"output": "plain", "status": "failed"}).failed)
        self.assertFalse(grounding.tool_text({"output": "User not found"}).failed)
        self.assertEqual(str(grounding.tool_text({"output": 5})), "5")


if __name__ == "__main__":
    unittest.main()
