"""Round 4k / R-1, R-2, R-4, R-5, R-6, R-7.

R-1 every number needs provenance whatever its length; Chinese numerals and Arabic digits are compared in ONE format (format conversion only).
R-2 the 0.75 / 0.5 overlap ratios are gone. The MODEL attaches ``grounding_evidence`` (a span of the caller's words it took the value from, plus its own
    statement that the value comes from the span); the code only verifies that the span occurs verbatim (after the same format normalisation). A refused
    call can be unblocked by a read-back the caller confirms. Every model-judged acceptance is audited.
R-4 item.error / item.status reach the back office; R-5 the degraded path is explicit, rejections are capped per session and the table is cleared;
R-6 digit strings are never fuzzy, invented words in long values are caught; R-7 instructions / tool schema are not provenance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live import backend, grounding, realtime  # noqa: E402
from tests.test_orch_worker import _say, _tool_call  # noqa: E402
from tests.test_orchestrator import _FakeResp, _FakeSession  # noqa: E402

K = grounding.EVIDENCE_KEY


def bad(args, heard, seen=None, evidence=None, spoken=None, audit=None):
    return grounding.check("t", args, heard, seen or [], evidence=evidence, spoken=spoken, audit=audit)


def ev(argument, span, derives=True, read_back=None):
    d = {"argument": argument, "span": span, "value_comes_from_span": derives}
    if read_back is not None:
        d["read_back_to_caller"] = read_back
    return [d]


class Normaliser(unittest.TestCase):
    """Format conversion only: the same function is applied to the value, to the caller's words and to a span."""

    def test_chinese_numerals_become_digits(self) -> None:
        for src, want in (("八十八", "88"), ("一号", "1号"), ("二十三号", "23号"), ("三零二", "302"), ("二零二六年十一月", "2026年11月"), ("十二", "12"), ("十", "10"),
                          ("一百零五", "105"), ("两千", "2000"), ("一万二千", "12000"), ("九十", "90"), ("壹佰贰拾叁", "123"), ("三千零八", "3008")):
            self.assertEqual(grounding.norm_text(src), want, src)

    def test_a_bare_unit_or_other_text_is_left_alone(self) -> None:
        self.assertEqual(grounding.norm_text("万一"), "万一")            # a 万 with no number before it is a word, so the whole run is left alone
        self.assertEqual(grounding.norm_text("百货"), "百货")
        self.assertEqual(grounding.norm_text("十二点"), "12点")           # no meaning is added: 12点 is not turned into 12:00

    def test_width_case_and_punctuation(self) -> None:
        self.assertEqual(grounding.norm_text("ＡＢＣ １２３"), "abc123")
        self.assertEqual(grounding.norm_text("张江路 88 号!"), "张江路88号")
        self.assertEqual(grounding.norm_text("Bob@Gmail.com"), "bob@gmail.com")

    def test_a_number_is_never_a_piece_of_a_longer_number(self) -> None:
        self.assertTrue(grounding.occurs("88", grounding.norm_text("张江路88号")))
        self.assertFalse(grounding.occurs("88", "1988"))
        self.assertFalse(grounding.occurs("88", "8800"))
        self.assertTrue(grounding.occurs("88号", "张江路88号"))
        self.assertFalse(grounding.occurs("8号", "张江路88号"))


class EveryNumberNeedsProvenance(unittest.TestCase):
    ZJ = ["我住在上海市浦东新区张江路"]

    def test_the_house_number_the_caller_never_said_is_blocked(self) -> None:
        for v in ("上海市浦东新区张江路 88 号", "上海市浦东新区张江路88号", "张江路88号", "88号", "88"):
            self.assertTrue(bad({"address": v}, self.ZJ), v)

    def test_it_passes_when_said_in_digits_or_in_chinese_numerals(self) -> None:
        for heard in (["我住在上海市浦东新区张江路88号"], ["我住在上海市浦东新区张江路八十八号"], ["我住在上海市浦东新区张江路 88 号"]):
            for v in ("上海市浦东新区张江路88号", "上海市浦东新区张江路八十八号", "张江路 88 号"):
                self.assertEqual(bad({"address": v}, heard), [], (heard, v))

    def test_building_numbers_both_directions(self) -> None:
        self.assertTrue(bad({"room": "9号楼"}, ["一号楼"]))
        self.assertEqual(bad({"room": "1号楼"}, ["一号楼"]), [])
        self.assertEqual(bad({"room": "一号楼"}, ["1号楼"]), [])
        self.assertEqual(bad({"n": "二十三号"}, ["23号"]), [])
        self.assertEqual(bad({"n": "23号"}, ["二十三号"]), [])

    def test_short_numbers_are_checked_too(self) -> None:
        self.assertTrue(bad({"age": "37"}, ["我想查一下账单"]))
        self.assertTrue(bad({"floor": "12"}, ["张伟"]))
        self.assertTrue(bad({"qty": 5}, ["hello"]))
        self.assertEqual(bad({"age": "37"}, ["我37岁"]), [])
        self.assertEqual(bad({"age": "37"}, ["我三十七岁"]), [])
        self.assertEqual(bad({"floor": 12}, ["十二楼"]), [])
        self.assertEqual(bad({"qty": 2}, ["two tickets please"]), [])

    def test_a_number_cannot_be_assembled_from_two_turns(self) -> None:
        self.assertTrue(bad({"n": "88"}, ["张江路8", "8号"]))
        self.assertTrue(bad({"n": "张江路88"}, ["张江路8", "8号"]))

    def test_digits_inside_a_longer_number_do_not_count(self) -> None:
        self.assertTrue(bad({"n": "88"}, ["我是1988年生的"]))
        self.assertTrue(bad({"n": "88"}, ["8800"]))

    def test_only_a_value_with_nothing_to_look_up_is_skipped(self) -> None:
        self.assertEqual(bad({"state": "TX"}, ["hello"]), [])
        self.assertEqual(bad({"c": "京"}, ["hello"]), [])
        self.assertTrue(bad({"c": "李明"}, ["hello"]))
        self.assertTrue(bad({"n": "4"}, ["hello"]))

    def test_a_number_from_a_tool_result_passes(self) -> None:
        self.assertEqual(bad({"order": "73301"}, ["look it up"], ["{'order': 73301}"]), [])

    def test_digit_strings_are_never_fuzzy_matched(self) -> None:
        """R-6: 4829137 for a heard 4829130 differs in one digit and must be blocked."""
        self.assertTrue(bad({"id": "4829137"}, ["it is 4829130"]))
        self.assertEqual(bad({"id": "4829130"}, ["it is 4829130"]), [])

    def test_invented_words_mixed_into_a_long_value_are_caught(self) -> None:
        said = ["call back about the order please to someone"]
        self.assertTrue(bad({"note": "call back to John Smith about the order"}, said))
        self.assertTrue(bad({"note": "please book a table for seven tonight"}, ["please book a table for four tonight"]))
        self.assertEqual(bad({"note": "please book a table for four tonight"}, ["please book a table for four tonight"]), [])


class TheModelsEvidenceSpan(unittest.TestCase):
    """False blocks the old overlap ratios produced; the model judges equivalence, the code checks the span exists."""

    def test_a_short_paraphrase_passes_with_a_span_and_not_without(self) -> None:
        heard = ["我住在浦东新区张江路"]
        self.assertTrue(bad({"address": "浦东张江"}, heard))
        audit: list = []
        self.assertEqual(bad({"address": "浦东张江"}, heard, evidence=ev("address", "浦东新区张江路"), audit=audit), [])
        self.assertEqual([(a["how"], a["span"], a["value_comes_from_span"]) for a in audit], [("model_span", "浦东新区张江路", True)])

    def test_homophone_and_traditional_characters(self) -> None:
        heard = ["我叫张伟"]
        for v in ("张炜", "章伟", "張偉"):
            self.assertTrue(bad({"name": v}, heard), v)
            self.assertEqual(bad({"name": v}, heard, evidence=ev("name", "张伟")), [], v)

    def test_the_span_must_really_occur(self) -> None:
        heard = ["我叫张伟"]
        got = bad({"name": "李四"}, heard, evidence=ev("name", "李四"))
        self.assertTrue(got)
        self.assertIn("evidence span does not occur", got[0][2])
        self.assertTrue(bad({"name": "李四"}, heard, evidence=ev("name", "我叫李四")))

    def test_the_model_must_state_the_value_comes_from_the_span(self) -> None:
        got = bad({"name": "张炜"}, ["我叫张伟"], evidence=ev("name", "张伟", derives=False))
        self.assertTrue(got)
        self.assertIn("did not state", got[0][2])
        for flag in ("true", 1, None, "yes"):
            self.assertTrue(bad({"name": "张炜"}, ["我叫张伟"], evidence=[{"argument": "name", "span": "张伟", "value_comes_from_span": flag}]), flag)

    def test_malformed_evidence_is_just_not_evidence(self) -> None:
        for e in (None, "张伟", 7, [None], [{"argument": 3, "span": "张伟"}], [{"span": "张伟", "value_comes_from_span": True}], {"name": "张伟"}, [[]]):
            self.assertTrue(bad({"name": "张炜"}, ["我叫张伟"], evidence=e), e)

    def test_evidence_is_per_argument(self) -> None:
        self.assertTrue(bad({"name": "张炜"}, ["我叫张伟"], evidence=ev("nickname", "张伟")))

    def test_the_span_is_compared_after_format_normalisation(self) -> None:
        self.assertEqual(bad({"address": "88号"}, ["我住张江路八十八号"], evidence=ev("address", "张江路 八十八号")), [])
        self.assertEqual(bad({"t": "2026-11"}, ["二零二六年十一月"], evidence=ev("t", "二零二六年十一月")), [])

    def test_a_span_may_come_from_a_tool_result(self) -> None:
        self.assertEqual(bad({"customer": "Ms Zhang"}, ["look it up"], ["{'name': '张伟'}"], evidence=ev("customer", "张伟")), [])

    def test_the_instructions_and_the_tool_schema_are_not_provenance(self) -> None:
        """R-7: an example value in the system prompt is not something the caller said; only caller words and tool results count."""
        self.assertTrue(bad({"name": "张伟"}, ["hello"], seen=[]))
        s = _session(["hello"])
        s.cfg = {"instructions": "例如：张伟、李娜、北京市朝阳区建国路", "tools": [{"name": "lookup_account", "description": "e.g. 张伟"}]}
        self.assertTrue(asyncio.run(s._guard({"name": "lookup_account", "call_id": "c", "arguments": {"name": "张伟"}})))

    def test_the_12_character_value_with_two_invented_characters_is_blocked(self) -> None:
        heard = ["我住在上海市浦东新区世纪大道"]                    # the address, 12 characters, ends here
        full = "上海市浦东新区世纪大道"
        self.assertEqual(len(full), 11)
        self.assertEqual(bad({"a": full}, heard), [])
        self.assertTrue(bad({"a": full[:-2] + "路段"}, heard))      # last two characters invented: blocked (old pair-ratio let this through)
        self.assertTrue(bad({"a": full + "88号"}, heard))
        self.assertTrue(bad({"a": full[:-1] + "路"}, heard))        # one invented character: blocked too, no tolerance ratio any more


# ----------------------------------------------------------------- the guard in a session
def _session(heard, instructions: str = "") -> realtime.RealtimeSession:
    s = realtime.RealtimeSession.__new__(realtime.RealtimeSession)
    s.id = "t"; s.heard = list(heard); s.seen = []; s.asr = object(); s.asr_tasks = []; s.guard_counts = {}
    s.guard_total = 0; s.audit = []; s.spoken = []; s.degraded_warned = False; s.closed = False; s.tasks = []
    s.cfg = {"instructions": instructions, "tools": [{"name": "lookup_account"}]}
    s.engine = mock.MagicMock(); s.engine.tool_result = mock.AsyncMock(); s.engine.close = mock.AsyncMock()
    return s


def call(args: dict, cid: str = "c") -> dict:
    return {"name": "lookup_account", "call_id": cid, "arguments": dict(args)}


class GuardWithEvidence(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_releases_the_call_and_is_stripped_before_the_client_sees_it(self) -> None:
        s = _session(["我叫张伟"])
        e = call({"name": "张炜", K: ev("name", "张伟")})
        self.assertFalse(await s._guard(e))
        self.assertEqual(e["arguments"], {"name": "张炜"})                    # the reserved argument never reaches the client's tool
        self.assertEqual([(a["how"], a["argument"]) for a in s.audit], [("model_span", "name")])
        s.engine.tool_result.assert_not_awaited()

    async def test_evidence_that_does_not_check_out_is_stripped_and_refused(self) -> None:
        s = _session(["我叫张伟"])
        e = call({"name": "李四", K: ev("name", "李四")})
        self.assertTrue(await s._guard(e))
        self.assertNotIn(K, e["arguments"])
        self.assertIn("evidence span does not occur", s.engine.tool_result.await_args.args[1])
        self.assertIn(K, s.engine.tool_result.await_args.args[1])             # the refusal tells the model about the evidence path

    async def test_a_plain_refusal_tells_the_model_about_the_evidence_path(self) -> None:
        s = _session(["我叫张伟"])
        self.assertTrue(await s._guard(call({"name": "李四"})))
        self.assertIn(K, s.engine.tool_result.await_args.args[1])

    async def test_the_tool_schema_in_the_session_config_is_not_provenance(self) -> None:
        s = _session(["你好"])
        s.cfg["tools"] = [{"name": "lookup_account", "description": "e.g. account 99817", "parameters": {"properties": {"acct": {"enum": ["99817"]}}}}]
        self.assertTrue(await s._guard(call({"acct": "99817"})))

    async def test_model_judged_acceptances_are_logged(self) -> None:
        s = _session(["我叫张伟"])
        with self.assertLogs("sophon_live", level="INFO") as cm:
            await s._guard(call({"name": "张炜", K: ev("name", "张伟")}))
        self.assertTrue(any("on the model's evidence" in m and "张伟" in m and "model_span" in m for m in cm.output), cm.output)

    async def test_refused_call_then_read_back_then_confirmation_releases_it(self) -> None:
        s = _session(["我姓张"])
        a = {"name": "张炜"}
        for i in range(1, 4):
            self.assertTrue(await s._guard(call(a, f"r{i}")))
        sent = json.loads(s.engine.tool_result.await_args.args[1])
        self.assertEqual(sent["status"], "needs_input")
        for needle in ("Read each value back", K, "read_back_to_caller=true", "confirming words"):
            self.assertIn(needle, sent["message"])
        # the assistant reads it back, the caller confirms
        s.spoken.append(("我这边记的是张炜，对吗？", len(s.heard)))
        s.heard.append("对，就是这个")
        e = call({**a, K: ev("name", "对，就是这个", read_back=True)}, "r4")
        self.assertFalse(await s._guard(e))
        self.assertEqual(e["arguments"], a)
        self.assertEqual(s.audit[-1]["how"], "read_back_confirmed")

    async def test_a_confirmation_needs_a_read_back_and_must_come_after_it(self) -> None:
        a = {"name": "张炜"}
        s = _session(["我姓张", "对，就是这个"])                                   # the "confirmation" is older than any read-back
        s.spoken.append(("请问是张炜吗", 2))
        self.assertTrue(await s._guard(call({**a, K: ev("name", "对，就是这个", read_back=True)})))
        s = _session(["我姓张"])                                                    # nothing was ever read back
        s.heard.append("对，就是这个")
        self.assertTrue(await s._guard(call({**a, K: ev("name", "对，就是这个", read_back=True)})))
        s = _session(["我姓张"])                                                    # a different value was read back
        s.spoken.append(("请问是李四吗", 1)); s.heard.append("对，就是这个")
        self.assertTrue(await s._guard(call({**a, K: ev("name", "对，就是这个", read_back=True)})))
        s = _session(["我姓张"])                                                    # the caller's words after the read-back do not contain the span
        s.spoken.append(("请问是张炜吗", 1)); s.heard.append("不对")
        self.assertTrue(await s._guard(call({**a, K: ev("name", "对，就是这个", read_back=True)})))

    async def test_a_plain_span_cannot_borrow_the_read_back_flag(self) -> None:
        s = _session(["我叫张伟"])
        s.spoken.append(("请问是张炜吗", 1))
        self.assertTrue(await s._guard(call({"name": "李四", K: ev("name", "李四", read_back=True)})))

    async def test_needs_input_is_spoken(self) -> None:
        s = _session(["hello"])
        for i in range(4):
            await s._guard(call({"name": "张伟"}, f"c{i}"))
        self.assertIs(s.engine.tool_result.await_args.kwargs["speak"], True)
        self.assertEqual(json.loads(s.engine.tool_result.await_args.args[1])["status"], "needs_input")


class SpokenTextIsTrackedForReadBacks(unittest.IsolatedAsyncioTestCase):
    async def test_a_finished_reply_is_recorded_with_the_number_of_caller_turns_so_far(self) -> None:
        s = _session(["我姓张", "好的"])
        s.send = mock.AsyncMock(); s.bo = None; s.resp = None
        r = {"id": "resp_1", "item_id": "item_1", "text": "请问是张炜吗", "calls": [], "audio": False, "done": False, "metadata": None}
        await s._finish(r)
        self.assertEqual(s.spoken, [("请问是张炜吗", 2)])
        r2 = {"id": "resp_2", "item_id": "item_2", "text": "", "calls": [], "audio": False, "done": False, "metadata": None}
        await s._finish(r2)
        self.assertEqual(len(s.spoken), 1)                                       # silence is not a read-back


class DegradedPathSessionCapAndCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_no_asr_is_an_explicit_labelled_degraded_path(self) -> None:
        s = _session(["hello"]); s.asr = None
        e = call({"name": "张伟", K: ev("name", "张伟")})
        with self.assertLogs("sophon_live", level="INFO") as cm:
            self.assertFalse(await s._guard(e))
            self.assertFalse(await s._guard(call({"zip": "12345"}, "c2")))
        self.assertEqual(e["arguments"], {"name": "张伟"})
        self.assertEqual(sum("GROUNDING UNAVAILABLE (no_asr_transcript)" in m and "WARNING" in m for m in cm.output), 1, cm.output)   # warned once
        self.assertEqual(sum("grounding degraded" in m for m in cm.output), 2)                                                          # every call is labelled
        self.assertEqual([(a["grounding"], a["reason"]) for a in s.audit], [("unavailable", "no_asr_transcript")] * 2)

    async def test_ground_off_is_labelled_too(self) -> None:
        s = _session(["hello"])
        with mock.patch.object(realtime, "GROUND", False), self.assertLogs("sophon_live", level="INFO") as cm:
            self.assertFalse(await s._guard(call({"zip": "12345"})))
        self.assertTrue(any("grounding_off" in m for m in cm.output))
        self.assertEqual(s.audit[0]["reason"], "grounding_off")

    async def test_a_model_that_changes_one_argument_each_time_is_capped_per_session(self) -> None:
        s = _session(["hello"])
        msgs = []
        for i in range(30):
            self.assertTrue(await s._guard(call({"name": f"张{i}"}, f"c{i}")), i)
            msgs.append(s.engine.tool_result.await_args.args[1])
        self.assertEqual(realtime.GUARD_SESSION_MAX_REJECTIONS, 12)
        self.assertTrue(all(m.startswith("NOT EXECUTED. These") for m in msgs[:12]))
        for m in msgs[12:]:
            body = json.loads(m)
            self.assertEqual(body["status"], "needs_input")
            self.assertIn("Too many of your calls were refused", body["message"])
        self.assertLessEqual(len(s.guard_counts), realtime.GUARD_SESSION_MAX_REJECTIONS)       # the table cannot grow without bound
        self.assertIs(s.engine.tool_result.await_args.kwargs["speak"], True)

    async def test_the_counter_table_is_cleared_when_the_session_ends(self) -> None:
        s = _session(["hello"])
        await s._guard(call({"name": "张伟"}))
        self.assertTrue(s.guard_counts and s.audit == [] or s.guard_counts)
        s.audit.append({"x": 1}); s.spoken.append(("a", 0))
        await s.close()
        self.assertEqual((s.guard_counts, s.audit, s.spoken), ({}, [], []))


class SchemaAndPrompt(unittest.TestCase):
    TOOL = {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"zip": {"type": "string"}}}}

    def test_every_function_gets_the_optional_evidence_argument_and_the_original_is_untouched(self) -> None:
        orig = json.dumps([self.TOOL, {"function": {"name": "x", "parameters": {"type": "object", "properties": {}}}}], sort_keys=True)
        tools = json.loads(orig)
        out = grounding.with_evidence_field(tools)
        self.assertEqual(json.dumps(tools, sort_keys=True), orig)
        self.assertIn(K, out[0]["parameters"]["properties"])
        self.assertIn(K, out[1]["function"]["parameters"]["properties"])
        self.assertNotIn(K, out[0]["parameters"].get("required", []))                    # optional
        item = out[0]["parameters"]["properties"][K]["items"]
        self.assertEqual(set(item["required"]), {"argument", "span", "value_comes_from_span"})

    def test_delegate_and_an_existing_property_are_left_alone(self) -> None:
        d = {"type": "function", "name": "delegate", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}}}
        own = {"type": "function", "name": "t", "parameters": {"type": "object", "properties": {K: {"type": "string"}}}}
        out = grounding.with_evidence_field([d, own])
        self.assertNotIn(K, out[0]["parameters"]["properties"])
        self.assertEqual(out[1]["parameters"]["properties"][K], {"type": "string"})

    def test_the_engine_is_shown_the_evidence_field_and_the_client_session_keeps_its_own_tools(self) -> None:
        s = realtime.RealtimeSession(mock.MagicMock(), argparse.Namespace(engine="x", bridge="y"), object(), None, None, "m")
        s.engine = mock.MagicMock(); s.engine.cfg = {"extra_body": {}}; s.engine.set_instructions = mock.AsyncMock(); s.send = mock.AsyncMock()
        asyncio.run(s.on_update({"session": {"instructions": "hi", "tools": [self.TOOL]}}))
        shown = s.engine.cfg["extra_body"]["tools"]
        self.assertIn(K, shown[0]["parameters"]["properties"])
        self.assertNotIn(K, s.cfg["tools"][0]["parameters"]["properties"])
        s.asr = None                                                                       # without ASR there is nothing to ground with: no extra field
        asyncio.run(s.on_update({"session": {"tools": [self.TOOL]}}))
        self.assertNotIn(K, s.engine.cfg["extra_body"]["tools"][0]["parameters"]["properties"])

    def test_the_system_prompt_explains_the_evidence_rule_only_when_it_applies(self) -> None:
        s = realtime.RealtimeSession(mock.MagicMock(), argparse.Namespace(engine="x", bridge="y"), object(), None, None, "m")
        s.cfg["tools"] = [self.TOOL]; s.bo = None
        for lang, rules in (("en", grounding.EVIDENCE_RULES_EN), ("zh", grounding.EVIDENCE_RULES_ZH)):
            s.lang = lang
            self.assertIn(rules, s._sys())
            self.assertIn(K, rules)
        s.asr = None
        self.assertNotIn(K, s._sys())


# ----------------------------------------------------------------- R-4: item-level structure on the delegation path
class DelegationPathKeepsItemStructure(unittest.IsolatedAsyncioTestCase):
    async def _item(self, item: dict) -> grounding.ToolText:
        s = _session(["hi"]); s.bk_wait = {}; s.pending_outputs = []
        s.send = mock.AsyncMock()
        fut = asyncio.get_running_loop().create_future(); s.bk_wait["c1"] = fut
        await s.on_event({"type": "conversation.item.create", "item": dict(item, type="function_call_output", call_id="c1")})
        return fut.result()

    async def test_item_error_and_status_are_carried_with_the_text(self) -> None:
        for item in ({"output": "User not found", "error": {"code": "x"}}, {"output": "plain text", "status": "failed"}, {"output": "plain", "status": "incomplete"},
                     {"output": '{"ok": false}'}):
            out = await self._item(item)
            self.assertTrue(out.failed, item)
            self.assertEqual(str(out), item["output"])
        for item in ({"output": "User not found"}, {"output": "fine", "status": "completed"}, {"output": "fine", "status": "error"}):
            self.assertFalse((await self._item(item)).failed, item)

    async def test_a_failure_reported_by_item_counts_toward_max_fails(self) -> None:
        steps = [_FakeResp(200, _tool_call("update_account", "{}")) for _ in range(4)] + [_FakeResp(200, _say("ok"))]
        executed = []

        async def call_tool(name, args):
            executed.append(name)
            t = grounding.ToolText("plain prose, no structure in the text"); t.failed = True
            return t
        bo = backend.BackOffice("t")
        with mock.patch.object(backend.aiohttp, "ClientSession", return_value=_FakeSession(steps)), mock.patch.object(backend, "_key", return_value="k"):
            await bo.run("", [{"name": "update_account", "parameters": {"type": "object", "properties": {}}}], "task", call_tool)
        self.assertEqual(len(executed), backend.MAX_FAILS)
        self.assertEqual(bo.fails["update_account"], backend.MAX_FAILS)

    async def test_plain_text_without_a_failed_flag_is_not_counted(self) -> None:
        steps = [_FakeResp(200, _tool_call("update_account", "{}")) for _ in range(4)] + [_FakeResp(200, _say("ok"))]
        executed = []

        async def call_tool(name, args):
            executed.append(name)
            return grounding.ToolText("User not found")
        bo = backend.BackOffice("t")
        with mock.patch.object(backend.aiohttp, "ClientSession", return_value=_FakeSession(steps)), mock.patch.object(backend, "_key", return_value="k"):
            await bo.run("", [{"name": "update_account", "parameters": {"type": "object", "properties": {}}}], "task", call_tool)
        self.assertEqual((len(executed), bo.fails), (4, {}))


if __name__ == "__main__":
    unittest.main()
