"""Guidance entry tests. No GPU, no engine: the generation path runs against a stub tokenizer and stubbed vllm modules.

  python3 -m pytest tag/plugin/tests/test_guide_entry.py -p no:cacheprovider      (or: python3 -m unittest discover -s tag/plugin/tests)

Contract tests read the duplex-zh examples from the directory in DZH_CONTRACTS_ROOT (the folder that contains contracts/);
without it they skip. Nothing from duplex-zh is copied into this repository. The schema-wide tests also need jsonschema.
They only look at structure (fields, types, enums); no assertion looks at generated or free text."""
import base64
import copy
import importlib
import importlib.util
import json
import os
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
FRONT = Path(ROOT) / "contracts" / "examples" / "front" if ROOT else None
SCHEMAS = Path(ROOT) / "contracts" / "schemas" / "front" if ROOT else None
NO_ROOT = "DZH_CONTRACTS_ROOT is not set (point it at the folder that contains contracts/ from duplex-zh@233b2c41)"
_STUBBED: list[str] = []


def good():
    return {"schema_version": "0.1.0", "guidance_id": "g1", "version": 1, "session_id": "s1", "emitted_at": "2026-11-22T01:00:03.050+08:00", "seq": 7,
            "run_fingerprint": "3f" * 32, "host_id": "decision-host-1", "readout_id": "ro1", "snapshot_last_entry_id": "tl3", "turn_kind": "event_turn",
            "release": [{"item_id": "P1", "item_version": 1, "kind": "result", "released_by": "immediate", "evidence": [{"id": "ev1", "version": 1}],
                         "facts": [{"key": "task_state", "value": "OK"}, {"key": "temperature", "value": 24, "unit": "celsius"}]}]}


def setUpModule():
    """Stand in for vllm / vllm_omni / pybase64 / the torch-bound proc module so duplex.py can be imported without a GPU stack."""
    if importlib.util.find_spec("vllm_omni") is not None:
        return
    def mod(name, **attrs):
        m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; _STUBBED.append(name)
    class Base:
        def __init__(self, *a, **k): pass
    import logging
    mod("vllm"); mod("vllm.logger", init_logger=logging.getLogger)
    mod("vllm.sampling_params", RequestOutputKind=types.SimpleNamespace(DELTA=1), StructuredOutputsParams=lambda **k: types.SimpleNamespace(**k))
    mod("vllm_omni"); mod("vllm_omni.engine"); mod("vllm_omni.engine.duplex")
    mod("vllm_omni.engine.duplex.config", DuplexCapabilities=Base)
    mod("vllm_omni.engine.duplex.contracts", DuplexAppendPlan=lambda prompt: types.SimpleNamespace(prompt=prompt), DuplexOutputAction=Base, DuplexOutputDecision=Base)
    mod("vllm_omni.engine.duplex.plugin", DefaultDuplexModelSessionState=Base, DuplexDataPlane=Base, DuplexModelPlugin=Base,
        DuplexRuntimeConfigError=RuntimeError, PcmAppendBuffer=Base, PcmAppendReservation=Base)
    mod("pybase64", b64encode=base64.b64encode, b64decode=base64.b64decode)
    mod("sophon_realtime.proc", strip_tags=lambda s: s, tag_clean=lambda s, final: s)


def tearDownModule():
    for name in _STUBBED + ["sophon_realtime.duplex"]:
        sys.modules.pop(name, None)


class StubTok:
    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages, ensure_ascii=False)


def plugin_and_buf():
    duplex = importlib.import_module("sophon_realtime.duplex")
    p = duplex.SophonRealtimeDuplexPlugin.__new__(duplex.SophonRealtimeDuplexPlugin)
    p._bufs, p._reqs, p._spec_rid, p._spec_pending, p._keys = {}, {}, {}, {}, {}
    p._spec_lock = threading.Lock(); p.tok = StubTok()
    buf = duplex.SophonBuffer(); p._bufs[id(buf)] = buf; p._keys["k"] = buf
    return duplex, p, buf


def generate(plugin, buf, rid, sampling):
    """What the framework does after the plugin emits an append: plan it. Returns (the emitted payload, the prompt messages, the request info)."""
    pcm = base64.b64encode(np.zeros(320, np.float32).tobytes()).decode()
    res = buf.prepare_append({"audio": pcm, "format": "pcm_f32le", "sample_rate_hz": 16000}, operation_id="op", chunk_period_ms=100, allow_emit=True)
    plan = plugin.plan_append(request_id=rid, fence=None, session_config=None, seq=0, turn_seq=0, final=True, sampling_params=[sampling],
                              runtime_config={"lang": "zh", "instructions": "SYS", "tools": None}, payload=res.payload)
    res.commit(); buf.inflight_t = 0.0                                       # the framework accepted the append and the reply started
    return res.payload, json.loads(plan.prompt["prompt"]), plugin._reqs[rid]


class Validation(unittest.TestCase):
    def test_a_complete_guidance_is_valid(self):
        self.assertEqual(guide.validate(good()), [])

    def test_each_missing_required_field_is_rejected_and_logged(self):
        for k in guide.TOP_REQUIRED:
            g = good(); del g[k]; q = []
            with self.assertLogs("sophon_realtime.guide", "WARNING") as logs:
                code, body = guide.accept(q, g)
            self.assertEqual((code, q), (400, []), k)
            self.assertIn(f"$.{k}: required", body["errors"])
            self.assertIn("rejected", logs.output[0])

    def test_missing_required_fields_inside_nested_objects_are_rejected(self):
        for path in (("release", 0, "facts"), ("release", 0, "item_id"), ("release", 0, "released_by"), ("release", 0, "facts", 0, "key"), ("release", 0, "facts", 0, "value")):
            g = good(); node = g
            for step in path[:-1]: node = node[step]
            del node[path[-1]]
            self.assertTrue(guide.validate(g), path)

    def test_structure_rules_from_the_schema(self):
        def broken(fn):
            g = good(); fn(g); return guide.validate(g)
        self.assertTrue(broken(lambda g: g["release"][0].update(evidence=[])))                    # result needs evidence
        self.assertTrue(broken(lambda g: g["release"][0].update(text="x")))                       # no extra fields
        self.assertTrue(broken(lambda g: g.update(turn_kind="chat_turn")))                        # closed enum
        self.assertTrue(broken(lambda g: g.update(version=True)))                                 # booleans are not integers
        self.assertTrue(broken(lambda g: g["release"][0]["facts"][0].update(value="x" * 65)))     # a value is not a sentence
        self.assertTrue(broken(lambda g: g["release"][0]["facts"][0].update(value=["a"])))
        self.assertTrue(broken(lambda g: g.update(current_turn={"H1": "to_system", "H7": "answer", "H8": "normal"})))   # only a user_turn has it
        self.assertEqual(broken(lambda g: g["release"][0].update(kind="question", evidence=[])), [])
        self.assertTrue(guide.validate([]) and guide.validate(None))

    def test_refusal_cannot_open_tasks_and_facts_need_a_refusal(self):
        g = good(); g.update(turn_kind="user_turn", current_turn={"H1": "to_system", "H7": "answer", "H8": "escalate"}, opened_tasks=[{"task_id": "T9", "task_version": 1}],
                             disposition_facts=[{"key": "helpline_ref", "value": "hl.cn.1"}])
        self.assertIn("$.opened_tasks: must be empty when H8 is refuse or escalate", guide.validate(g))
        g["opened_tasks"] = []; self.assertEqual(guide.validate(g), [])
        g["current_turn"]["H8"] = "normal"; self.assertTrue(guide.validate(g))


class Accept(unittest.TestCase):
    def test_valid_guidance_is_queued_with_id_and_version_and_facts_as_given(self):
        g = good(); g["release"][0]["facts"][0]["value"] = "任意 free text"; q = []
        self.assertEqual(guide.accept(q, g), (200, {"ok": True, "guidance_id": "g1", "version": 1}))
        self.assertEqual((q[0]["guidance_id"], q[0]["version"]), ("g1", 1))
        self.assertEqual(q[0]["release"], g["release"])
        self.assertNotIn("run_fingerprint", q[0])

    def test_bridge_guide_route_answers_with_the_entry_result(self):
        q = []; br = bridge.Bridge(lambda *a: False, on_guide=lambda key, j: guide.accept(q, j))
        with socket_free_port() as port:
            bridge.PORT = port
        br.start()
        try:
            def post(obj):
                req = urllib.request.Request(f"http://127.0.0.1:{port}/bridge/k/guide", data=json.dumps(obj).encode(), method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r: return r.status, json.load(r)
                except urllib.error.HTTPError as e:
                    return e.code, json.load(e)
            self.assertEqual(post(good())[0], 200)
            bad = good(); del bad["host_id"]
            code, body = post(bad)
            self.assertEqual((code, body["ok"]), (400, False))
            self.assertEqual(len(q), 1)
        finally:
            br._srv.shutdown(); br._srv.server_close(); bridge.PORT = 0


def socket_free_port():
    import contextlib, socket
    @contextlib.contextmanager
    def cm():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); yield s.getsockname()[1]
    return cm()


class Generation(unittest.TestCase):
    def test_guidance_goes_into_this_generation_not_a_verbatim_say(self):
        duplex, plugin, buf = plugin_and_buf()
        self.assertEqual(plugin._bridge_guide("k", good())[0], 200)
        sampling = types.SimpleNamespace(structured_outputs=None, max_tokens=120)
        payload, msgs, info = generate(plugin, buf, "rid1", sampling)
        self.assertIn("guide", payload); self.assertNotIn("say", payload); self.assertNotIn("say", info)
        self.assertIsNone(sampling.structured_outputs)                        # say would pin the thinker to one allowed text
        self.assertEqual(sampling.max_tokens, 120)
        self.assertEqual(msgs[0]["content"], "SYS\n\n" + guide.SYS_GUIDE["zh"])
        self.assertEqual(msgs[-1]["role"], "tool")
        ctx = json.loads(msgs[-1]["content"])
        self.assertEqual((ctx["guidance_id"], ctx["version"]), ("g1", 1))
        self.assertEqual(ctx["release"][0]["facts"], good()["release"][0]["facts"])
        self.assertEqual(info["guidance_ref"], {"guidance_id": "g1", "version": 1})
        self.assertEqual(buf.guide_q, [])

    def test_the_guidance_stays_in_history_next_to_its_reply(self):
        duplex, plugin, buf = plugin_and_buf()
        plugin._bridge_guide("k", good())
        _, _, info = generate(plugin, buf, "rid1", None)
        info["a_text"] = "reply"; plugin._finish_reply("rid1", True)
        self.assertEqual((buf.history[-1]["guide"]["guidance_id"], buf.history[-1]["guide"]["version"]), ("g1", 1))
        second = good(); second.update(guidance_id="g2", version=3)
        plugin._bridge_guide("k", second)
        _, msgs, _ = generate(plugin, buf, "rid2", None)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["system", "tool", "assistant", "tool"])
        self.assertEqual(json.loads(msgs[1]["content"])["guidance_id"], "g1")
        self.assertEqual(json.loads(msgs[3]["content"])["guidance_id"], "g2")

    def test_a_rejected_guidance_queues_nothing_and_nothing_is_emitted(self):
        duplex, plugin, buf = plugin_and_buf()
        bad = good(); del bad["release"]
        with self.assertLogs(level="WARNING"):
            self.assertEqual(plugin._bridge_guide("k", bad)[0], 400)
        self.assertEqual(plugin._bridge_guide("unknown", good())[0], 404)
        self.assertEqual(buf.guide_q, [])
        pcm = base64.b64encode(np.zeros(320, np.float32).tobytes()).decode()
        self.assertIsNone(buf.prepare_append({"audio": pcm, "format": "pcm_f32le", "sample_rate_hz": 16000}, operation_id="op", chunk_period_ms=100, allow_emit=True))


@unittest.skipUnless(ROOT, NO_ROOT)
class ContractExamples(unittest.TestCase):
    def test_examples_directory_is_complete(self):
        self.assertEqual((len(list((FRONT / "valid").glob("*.json"))), len(list((FRONT / "invalid").glob("*.json")))), (13, 16))

    def test_guidance_examples_through_the_entry(self):
        for name in ("guidance-escalate", "guidance-user-turn"):
            g = json.loads((FRONT / "valid" / f"{name}.json").read_text())
            duplex, plugin, buf = plugin_and_buf()
            self.assertEqual(plugin._bridge_guide("k", copy.deepcopy(g))[0], 200, name)
            _, msgs, info = generate(plugin, buf, "rid1", None)
            ctx = json.loads(msgs[-1]["content"])
            self.assertEqual((ctx["guidance_id"], ctx["version"]), (g["guidance_id"], g["version"]), name)
            for k in ("release", "disposition_facts", "current_turn", "opened_tasks"):
                self.assertEqual(ctx.get(k), g.get(k), f"{name}.{k}")
            self.assertEqual(info["guidance_ref"], {"guidance_id": g["guidance_id"], "version": g["version"]})

    def test_guidance_invalid_examples_are_refused_by_the_entry(self):
        names = sorted(p.stem for p in (FRONT / "invalid").glob("guidance-*.json"))
        self.assertEqual(names, ["guidance-refuse-with-new-task", "guidance-result-without-evidence", "guidance-wording-forbidden"])
        for name in names:
            q = []
            with self.assertLogs("sophon_realtime.guide", "WARNING"):
                code, _ = guide.accept(q, json.loads((FRONT / "invalid" / f"{name}.json").read_text()))
            self.assertEqual((code, q), (400, []), name)


@unittest.skipUnless(ROOT, NO_ROOT)
@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is not installed")
class ContractSchemas(unittest.TestCase):
    def validator(self, name):
        import jsonschema
        schema = json.loads((SCHEMAS / f"{name.split('-')[0]}.schema.json").read_text())
        return jsonschema.Draft202012Validator(schema)

    def test_13_valid_examples_pass_and_16_invalid_are_refused(self):
        valid = sorted((FRONT / "valid").glob("*.json")); invalid = sorted((FRONT / "invalid").glob("*.json"))
        self.assertEqual((len(valid), len(invalid)), (13, 16))
        for p in valid:
            self.assertTrue(self.validator(p.stem).is_valid(json.loads(p.read_text())), p.name)
        for p in invalid:
            self.assertFalse(self.validator(p.stem).is_valid(json.loads(p.read_text())), p.name)

    def test_entry_and_schema_agree_on_the_guidance_examples(self):
        for p in sorted(FRONT.glob("*/guidance-*.json")):
            g = json.loads(p.read_text())
            self.assertEqual(not guide.validate(g), self.validator(p.stem).is_valid(g), p.name)


if __name__ == "__main__":
    unittest.main()
