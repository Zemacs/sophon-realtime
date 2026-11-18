"""Round 4g / grounding.failure_hint: the spell-it-out hint follows a STRUCTURED failure report (item error / status / JSON ok:false), never
the words of the tool output ("not found", "error", ...). Same principle as B-3 (user, 00:27): the model judges, code only enforces."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import grounding  # noqa: E402

OLD_WORDS = ("not found", "no user", "could not find", "cannot find", "does not exist", "invalid", "error", "Error: lookup failed", "ERROR")


def _item(output, **kw):
    return {"type": "function_call_output", "call_id": "c1", "output": output, **kw}


class OutputFailedIsStructural(unittest.TestCase):
    def test_prose_with_failure_words_is_not_a_reported_failure(self) -> None:
        for w in OLD_WORDS:
            self.assertFalse(grounding.output_failed(_item(f"The account {w} for that name")), w)
            self.assertFalse(grounding.output_failed(_item(json.dumps({"message": w}))), w)          # a JSON key that is not the contract
            self.assertFalse(grounding.output_failed(_item(json.dumps({"ok": True, "note": w}))), w)

    def test_structured_failures_are_reported(self) -> None:
        for item in (_item("x", error={"message": "boom"}), _item("x", error="boom"), _item("x", status="failed"), _item("x", status="incomplete"),
                     _item(json.dumps({"ok": False})), _item(json.dumps({"ok": False, "error": "no user"})), _item(json.dumps({"success": False})),
                     _item(json.dumps({"error": "boom"})), _item('{ "ok" : false }'), _item({"ok": False})):
            self.assertTrue(grounding.output_failed(item), item)

    def test_non_failures(self) -> None:
        for item in (_item("done"), _item(""), _item("{not json"), _item("[1, 2]"), _item(json.dumps({"ok": True})), _item(json.dumps({"error": ""})),
                     _item(json.dumps({"error": None})), _item("x", status="completed"), _item("x", error=None), _item('{"ok": "false"}'), None, "str"):
            self.assertFalse(grounding.output_failed(item), item)


class FailureHintFollowsTheFlag(unittest.TestCase):
    HEARD = ["my name is Jon Smith", "J O N"]

    def test_no_hint_unless_failed(self) -> None:
        for out in OLD_WORDS:
            self.assertEqual(grounding.failure_hint(False, self.HEARD), "")
        self.assertEqual(grounding.failure_hint(True, []), "")                    # nothing heard: nothing to repeat back

    def test_hint_for_a_failed_call_carries_the_callers_words(self) -> None:
        h = grounding.failure_hint(True, self.HEARD)
        self.assertIn("Jon Smith", h)
        self.assertIn("spell it letter by letter", h)

    def test_the_function_no_longer_takes_the_output_text(self) -> None:
        import inspect
        self.assertEqual(list(inspect.signature(grounding.failure_hint).parameters), ["failed", "heard"])
        src = Path(grounding.__file__).read_text(encoding="utf-8")
        self.assertNotIn("not found|no user", src)


class RealtimeSessionPath(unittest.IsolatedAsyncioTestCase):
    """The real RealtimeSession.on_event: function_call_output items, then response.create, and what the engine is handed."""

    def _session(self):
        import sophon_live.realtime as rt

        s = rt.RealtimeSession.__new__(rt.RealtimeSession)
        s.id = "sess_t"
        s.pending_outputs, s.seen, s.heard, s.bk_wait = [], [], list(FailureHintFollowsTheFlag.HEARD), {}
        s.resp = None
        s.sent = []

        async def send(ev):
            s.sent.append(ev)

        async def begin(meta=None, erid=None):
            return {"done": False, "audio": False}

        s.send, s._begin, s._track = send, begin, lambda r: None
        s.engine = mock.MagicMock()
        s.engine.tool_result = mock.AsyncMock()
        return s

    async def _outputs(self, items: list[dict]) -> list[str]:
        s = self._session()
        for it in items:
            await s.on_event({"type": "conversation.item.create", "item": it})
        await s.on_event({"type": "response.create", "response": {}})
        handed = [c.args[1] for c in s.engine.tool_result.await_args_list]
        s.resp and s.resp.get("empty_timer") and s.resp["empty_timer"].cancel()
        return handed

    async def test_prose_error_gets_no_hint_but_a_structured_failure_does(self) -> None:
        import sophon_live.realtime as rt

        with mock.patch.object(rt, "GROUND", True):
            prose = await self._outputs([_item("Error: user not found")])
            structured = await self._outputs([_item("Error: user not found", status="failed")])
            as_json = await self._outputs([_item(json.dumps({"ok": False, "error": "no user"}))])
            ok_json = await self._outputs([_item(json.dumps({"ok": True}))])
        self.assertEqual(prose, ["Error: user not found"])                       # the old regex appended the hint here
        self.assertTrue(structured[0].startswith("Error: user not found\n[Caller's recent words"))
        self.assertIn("Jon Smith", structured[0])
        self.assertIn("Jon Smith", as_json[0])
        self.assertEqual(ok_json, [json.dumps({"ok": True})])

    async def test_flag_is_per_item_when_several_outputs_are_queued(self) -> None:
        import sophon_live.realtime as rt

        with mock.patch.object(rt, "GROUND", True):
            outs = await self._outputs([_item("a error", call_id="c1"), _item("b", call_id="c2", error="boom"), _item("c invalid", call_id="c3")])
        self.assertEqual([("Caller's recent words" in o) for o in outs], [False, True, False])

    async def test_ground_off_never_hints(self) -> None:
        import sophon_live.realtime as rt

        with mock.patch.object(rt, "GROUND", False):
            outs = await self._outputs([_item("x", status="failed")])
        self.assertEqual(outs, ["x"])


if __name__ == "__main__":
    unittest.main()
