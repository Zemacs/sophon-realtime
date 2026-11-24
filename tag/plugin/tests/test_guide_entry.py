"""Guidance entry tests. No GPU and no engine: generation runs against a stub tokenizer.

  python3 -m pytest tag/plugin/tests/test_guide_entry.py -p no:cacheprovider
  (or: python3 -m unittest discover -s tag/plugin/tests)

Contract tests read the duplex-zh examples from the folder in DZH_CONTRACTS_ROOT (the
one that contains contracts/); without it they skip. Nothing from duplex-zh is copied
here. The schema-wide tests also need jsonschema. Assertions look at structure only
(fields, types, enums), never at generated or free text.
"""

import base64
import copy
import importlib
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sophon_realtime import bridge, guide

ROOT = os.environ.get("DZH_CONTRACTS_ROOT")
FRONT = Path(ROOT or ".") / "contracts" / "examples" / "front"
SCHEMAS = Path(ROOT or ".") / "contracts" / "schemas" / "front"
NO_ROOT = "DZH_CONTRACTS_ROOT is not set (see the module docstring)"
LOG = ("sophon_realtime.guide", "WARNING")
REF = {"guidance_id": "g1", "version": 1}
_STUBBED: list[str] = []


def good():
    return {
        "schema_version": "0.1.0",
        "guidance_id": "g1",
        "version": 1,
        "session_id": "s1",
        "emitted_at": "2026-11-22T01:00:03.050+08:00",
        "seq": 7,
        "run_fingerprint": "3f" * 32,
        "host_id": "decision-host-1",
        "readout_id": "ro1",
        "snapshot_last_entry_id": "tl3",
        "turn_kind": "event_turn",
        "release": [
            {
                "item_id": "P1",
                "item_version": 1,
                "kind": "result",
                "released_by": "immediate",
                "evidence": [{"id": "ev1", "version": 1}],
                "facts": [
                    {"key": "task_state", "value": "OK"},
                    {"key": "temperature", "value": 24, "unit": "celsius"},
                ],
            }
        ],
    }


def setUpModule():
    """Stand in for vllm, vllm_omni, pybase64 and the torch-bound proc module."""
    if importlib.util.find_spec("vllm_omni") is not None:
        return

    class Base:
        def __init__(self, *a, **k):
            pass

    ns = types.SimpleNamespace
    duplex = "vllm_omni.engine.duplex"
    stubs = {
        "vllm": {},
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.sampling_params": {
            "RequestOutputKind": ns(DELTA=1),
            "StructuredOutputsParams": lambda **k: ns(**k),
        },
        "vllm_omni": {},
        "vllm_omni.engine": {},
        duplex: {},
        duplex + ".config": {"DuplexCapabilities": Base},
        duplex + ".contracts": {
            "DuplexAppendPlan": lambda prompt: ns(prompt=prompt),
            "DuplexOutputAction": Base,
            "DuplexOutputDecision": Base,
        },
        duplex + ".plugin": {
            "DefaultDuplexModelSessionState": Base,
            "DuplexDataPlane": Base,
            "DuplexModelPlugin": Base,
            "DuplexRuntimeConfigError": RuntimeError,
            "PcmAppendBuffer": Base,
            "PcmAppendReservation": Base,
        },
        "pybase64": {"b64encode": base64.b64encode, "b64decode": base64.b64decode},
        "sophon_realtime.proc": {"strip_tags": str, "tag_clean": lambda s, _: s},
    }
    for name, attrs in stubs.items():
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__dict__.update(attrs)
        _STUBBED.append(name)


def tearDownModule():
    for name in [*_STUBBED, "sophon_realtime.duplex"]:
        sys.modules.pop(name, None)


class StubTok:
    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages, ensure_ascii=False)


def plugin_and_buf():
    duplex = importlib.import_module("sophon_realtime.duplex")
    p = duplex.SophonRealtimeDuplexPlugin.__new__(duplex.SophonRealtimeDuplexPlugin)
    p._bufs, p._reqs, p._spec_rid, p._spec_pending, p._keys = {}, {}, {}, {}, {}
    p._spec_lock, p.tok = threading.Lock(), StubTok()
    buf = duplex.SophonBuffer()
    p._bufs[id(buf)], p._keys["k"] = buf, buf
    return p, buf


def emit(buf):
    """One silent audio append; the buffer may answer with a reply append."""
    pcm = base64.b64encode(np.zeros(320, np.float32).tobytes()).decode()
    audio = {"audio": pcm, "format": "pcm_f32le", "sample_rate_hz": 16000}
    kw = {"operation_id": "op", "chunk_period_ms": 100, "allow_emit": True}
    return buf.prepare_append(audio, **kw)


def generate(plugin, buf, rid, sampling):
    """Plan the emitted append: (payload, prompt messages, request info)."""
    res = emit(buf)
    config = {"lang": "zh", "instructions": "SYS", "tools": None}
    plan = plugin.plan_append(
        request_id=rid,
        fence=None,
        session_config=None,
        seq=0,
        turn_seq=0,
        final=True,
        sampling_params=[sampling],
        runtime_config=config,
        payload=res.payload,
    )
    res.commit()
    buf.inflight_t = 0.0
    return res.payload, json.loads(plan.prompt["prompt"]), plugin._reqs[rid]


def drop(g, *path):
    node = g
    for step in path[:-1]:
        node = node[step]
    del node[path[-1]]
    return g


def examples(kind):
    return sorted((FRONT / kind).glob("*.json"))


def load(path):
    return json.loads(path.read_text())


class Validation(unittest.TestCase):
    def test_a_complete_guidance_is_valid(self):
        self.assertEqual(guide.validate(good()), [])

    def test_each_missing_required_field_is_rejected_and_logged(self):
        for k in guide.TOP_REQUIRED:
            g, q = drop(good(), k), []
            with self.assertLogs(*LOG) as logs:
                code, body = guide.accept(q, g)
            self.assertEqual((code, q), (400, []), k)
            self.assertIn(f"$.{k}: required", body["errors"])
            self.assertIn("rejected", logs.output[0])

    def test_missing_required_fields_inside_nested_objects_are_rejected(self):
        item = ("release", 0)
        fact = (*item, "facts", 0)
        for path in (
            (*item, "facts"),
            (*item, "item_id"),
            (*item, "released_by"),
            (*fact, "key"),
            (*fact, "value"),
        ):
            self.assertTrue(guide.validate(drop(good(), *path)), path)

    def test_structure_rules_from_the_schema(self):
        def broken(fn):
            g = good()
            fn(g)
            return guide.validate(g)

        def item(g):
            return g["release"][0]

        def fact(g):
            return g["release"][0]["facts"][0]

        turn = {"H1": "to_system", "H7": "answer", "H8": "normal"}
        self.assertTrue(broken(lambda g: item(g).update(evidence=[])))  # needs evidence
        self.assertTrue(broken(lambda g: item(g).update(text="x")))  # no extra fields
        self.assertTrue(broken(lambda g: g.update(turn_kind="chat_turn")))  # enum
        self.assertTrue(broken(lambda g: g.update(version=True)))  # not an integer
        self.assertTrue(broken(lambda g: fact(g).update(value="x" * 65)))  # too long
        self.assertTrue(broken(lambda g: fact(g).update(value=["a"])))
        self.assertTrue(broken(lambda g: g.update(current_turn=turn)))  # user_turn only
        self.assertEqual(
            broken(lambda g: item(g).update(kind="question", evidence=[])), []
        )
        self.assertTrue(guide.validate([]) and guide.validate(None))

    def test_refusal_cannot_open_tasks_and_facts_need_a_refusal(self):
        g = good()
        g.update(
            turn_kind="user_turn",
            current_turn={"H1": "to_system", "H7": "answer", "H8": "escalate"},
            opened_tasks=[{"task_id": "T9", "task_version": 1}],
            disposition_facts=[{"key": "helpline_ref", "value": "hl.cn.1"}],
        )
        err = "$.opened_tasks: must be empty when H8 is refuse or escalate"
        self.assertIn(err, guide.validate(g))
        g["opened_tasks"] = []
        self.assertEqual(guide.validate(g), [])
        g["current_turn"]["H8"] = "normal"
        self.assertTrue(guide.validate(g))


class Accept(unittest.TestCase):
    def test_valid_guidance_is_queued_with_id_and_version_and_facts_as_given(self):
        g, q = good(), []
        g["release"][0]["facts"][0]["value"] = "任意 free text"
        self.assertEqual(guide.accept(q, g), (200, {"ok": True, **REF}))
        self.assertEqual((q[0]["guidance_id"], q[0]["version"]), ("g1", 1))
        self.assertEqual(q[0]["release"], g["release"])
        self.assertNotIn("run_fingerprint", q[0])

    def test_bridge_guide_route_answers_with_the_entry_result(self):
        q = []
        br = bridge.Bridge(lambda *a: False, on_guide=lambda key, j: guide.accept(q, j))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = bridge.PORT = s.getsockname()[1]
        br.start()
        try:

            def post(obj):
                url = f"http://127.0.0.1:{port}/bridge/k/guide"
                data = json.dumps(obj).encode()
                req = urllib.request.Request(url, data, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.load(r)
                except urllib.error.HTTPError as e:
                    return e.code, json.load(e)

            self.assertEqual(post(good())[0], 200)
            code, body = post(drop(good(), "host_id"))
            self.assertEqual((code, body["ok"]), (400, False))
            self.assertEqual(len(q), 1)
        finally:
            br._srv.shutdown()
            br._srv.server_close()
            bridge.PORT = 0


class Generation(unittest.TestCase):
    def test_guidance_goes_into_this_generation_not_a_verbatim_say(self):
        plugin, buf = plugin_and_buf()
        self.assertEqual(plugin._bridge_guide("k", good())[0], 200)
        sampling = types.SimpleNamespace(structured_outputs=None, max_tokens=120)
        payload, msgs, info = generate(plugin, buf, "rid1", sampling)
        self.assertIn("guide", payload)
        self.assertNotIn("say", payload)
        self.assertNotIn("say", info)
        self.assertIsNone(sampling.structured_outputs)  # say pins the thinker to a text
        self.assertEqual(sampling.max_tokens, 120)
        self.assertEqual(msgs[0]["content"], "SYS\n\n" + guide.SYS_GUIDE["zh"])
        self.assertEqual(msgs[-1]["role"], "tool")
        ctx = json.loads(msgs[-1]["content"])
        self.assertEqual((ctx["guidance_id"], ctx["version"]), ("g1", 1))
        self.assertEqual(ctx["release"][0]["facts"], good()["release"][0]["facts"])
        self.assertEqual(info["guidance_ref"], REF)
        self.assertEqual(buf.guide_q, [])

    def test_the_guidance_stays_in_history_next_to_its_reply(self):
        plugin, buf = plugin_and_buf()
        plugin._bridge_guide("k", good())
        _, _, info = generate(plugin, buf, "rid1", None)
        info["a_text"] = "reply"
        plugin._finish_reply("rid1", True)
        kept = buf.history[-1]["guide"]
        self.assertEqual((kept["guidance_id"], kept["version"]), ("g1", 1))
        plugin._bridge_guide("k", dict(good(), guidance_id="g2", version=3))
        _, msgs, _ = generate(plugin, buf, "rid2", None)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["system", "tool", "assistant", "tool"])
        self.assertEqual(json.loads(msgs[1]["content"])["guidance_id"], "g1")
        self.assertEqual(json.loads(msgs[3]["content"])["guidance_id"], "g2")

    def test_a_rejected_guidance_queues_nothing_and_nothing_is_emitted(self):
        plugin, buf = plugin_and_buf()
        with self.assertLogs(level="WARNING"):
            self.assertEqual(plugin._bridge_guide("k", drop(good(), "release"))[0], 400)
        self.assertEqual(plugin._bridge_guide("unknown", good())[0], 404)
        self.assertEqual(buf.guide_q, [])
        self.assertIsNone(emit(buf))


@unittest.skipUnless(ROOT, NO_ROOT)
class ContractExamples(unittest.TestCase):
    def test_examples_directory_is_complete(self):
        counts = (len(examples("valid")), len(examples("invalid")))
        self.assertEqual(counts, (13, 16))

    def test_guidance_examples_through_the_entry(self):
        for name in ("guidance-escalate", "guidance-user-turn"):
            g = load(FRONT / "valid" / f"{name}.json")
            plugin, buf = plugin_and_buf()
            self.assertEqual(plugin._bridge_guide("k", copy.deepcopy(g))[0], 200, name)
            _, msgs, info = generate(plugin, buf, "rid1", None)
            ctx = json.loads(msgs[-1]["content"])
            ref = {"guidance_id": g["guidance_id"], "version": g["version"]}
            self.assertEqual((ctx["guidance_id"], ctx["version"]), tuple(ref.values()))
            for k in ("release", "disposition_facts", "current_turn", "opened_tasks"):
                self.assertEqual(ctx.get(k), g.get(k), f"{name}.{k}")
            self.assertEqual(info["guidance_ref"], ref)

    def test_guidance_invalid_examples_are_refused_by_the_entry(self):
        names = [p.stem for p in examples("invalid") if p.stem.startswith("guidance-")]
        self.assertEqual(
            names,
            [
                "guidance-refuse-with-new-task",
                "guidance-result-without-evidence",
                "guidance-wording-forbidden",
            ],
        )
        for name in names:
            q = []
            with self.assertLogs(*LOG):
                code, _ = guide.accept(q, load(FRONT / "invalid" / f"{name}.json"))
            self.assertEqual((code, q), (400, []), name)


@unittest.skipUnless(ROOT, NO_ROOT)
@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is missing")
class ContractSchemas(unittest.TestCase):
    def validator(self, name):
        import jsonschema

        schema = load(SCHEMAS / f"{name.split('-')[0]}.schema.json")
        return jsonschema.Draft202012Validator(schema)

    def test_13_valid_examples_pass_and_16_invalid_are_refused(self):
        valid, invalid = examples("valid"), examples("invalid")
        self.assertEqual((len(valid), len(invalid)), (13, 16))
        for p in valid:
            self.assertTrue(self.validator(p.stem).is_valid(load(p)), p.name)
        for p in invalid:
            self.assertFalse(self.validator(p.stem).is_valid(load(p)), p.name)

    def test_entry_and_schema_agree_on_the_guidance_examples(self):
        for p in sorted(FRONT.glob("*/guidance-*.json")):
            g = load(p)
            ok = self.validator(p.stem).is_valid(g)
            self.assertEqual(not guide.validate(g), ok, p.name)


if __name__ == "__main__":
    unittest.main()
