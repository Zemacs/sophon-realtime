"""Contract consistency of the fork's F2 turn event with duplex-zh r7.1 (FrontPort).

The contract files are NOT part of this repository. Point DZH_CONTRACTS_ROOT at the
directory that contains contracts/ (a read-only copy of
duplex-zh@233b2c415fabb16746f2a19400bda1a4714ab680: contracts/examples/front/
{valid,invalid}, contracts/schemas/front). Without it the tests skip.
Needs the jsonschema package (an import error is a failure, not a skip).
Structure only: field presence, types, enums."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fork_harness import Call, Readouts, load_duplex

ROOT = os.environ.get("DZH_CONTRACTS_ROOT")
SKIP_REASON = (
    "DZH_CONTRACTS_ROOT is not set: point it at the directory containing contracts/ "
    "(duplex-zh@233b2c41 r7.1 examples + schemas)"
)
NEEDS_ROOT = unittest.skipUnless(ROOT, SKIP_REASON)
ADAPTER_FIELDS = (
    "schema_version",
    "event_id",
    "session_id",
    "emitted_at",
    "seq",
    "run_fingerprint",
    "host_id",
    "at_ms",
)


def _load(*parts):
    with open(os.path.join(ROOT, "contracts", *parts), encoding="utf-8") as f:
        return json.load(f)


def _examples(kind):
    d = os.path.join(ROOT, "contracts", "examples", "front", kind)
    return sorted(n for n in os.listdir(d) if n.endswith(".json"))


@NEEDS_ROOT
class R71Examples(unittest.TestCase):
    """The validator is only trusted once it passes every valid example and rejects
    every invalid one."""

    @classmethod
    def setUpClass(cls):
        import jsonschema

        cls.validators = {}
        schema_dir = os.path.join(ROOT, "contracts", "schemas", "front")
        for name in sorted(os.listdir(schema_dir)):
            schema = _load("schemas", "front", name)
            jsonschema.Draft202012Validator.check_schema(schema)
            validator = jsonschema.Draft202012Validator(schema)
            cls.validators[name[: -len(".schema.json")]] = validator

    def _errors(self, kind, name):
        schema_name = name.split("-", 1)[0]
        example = _load("examples", "front", kind, name)
        return list(self.validators[schema_name].iter_errors(example))

    def test_13_valid_examples_pass(self):
        names = _examples("valid")
        self.assertEqual(len(names), 13)
        for n in names:
            with self.subTest(n):
                self.assertEqual([e.message for e in self._errors("valid", n)], [])

    def test_16_invalid_examples_are_rejected(self):
        names = _examples("invalid")
        self.assertEqual(len(names), 16)
        for n in names:
            with self.subTest(n):
                self.assertNotEqual(self._errors("invalid", n), [])


def _fork_events():
    dx = load_duplex()
    take = Call(dx, Readouts((0.99, 0.0)))
    take.speech(500)
    take.silence(300)
    no_take = Call(dx, Readouts((0.1, 0.0)))
    no_take.speech(500)
    no_take.silence(2200)
    old, dx.IGNORE_SHORT = dx.IGNORE_SHORT, True
    try:
        reject = Call(dx, Readouts((0.1, 0.9)))
        reject.speech(300)
        reject.silence(2200)
    finally:
        dx.IGNORE_SHORT = old
    return {
        "take": take.events,
        "no_take": no_take.events,
        "acoustic_reject": reject.events,
    }


@NEEDS_ROOT
class ForkF2AgainstTurnEventSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jsonschema

        schema = _load("schemas", "front", "turn_event.schema.json")
        cls.validator = jsonschema.Draft202012Validator(schema)
        sample = _load("examples", "front", "valid", "turn_event-take.json")
        cls.adapter = {k: sample[k] for k in ADAPTER_FIELDS}
        cls.events = _fork_events()

    def test_each_case_reports_exactly_one_f2(self):
        self.assertEqual(
            {k: [e["turn_action"] for e in v] for k, v in self.events.items()},
            {"take": ["take"], "no_take": ["no_take"], "acoustic_reject": ["no_take"]},
        )

    def test_fork_payload_plus_adapter_fields_validates(self):
        for case, evs in self.events.items():
            for ev in evs:
                with self.subTest(case):
                    record = {k: v for k, v in ev.items() if k != "type"}
                    record.update(self.adapter)
                    errors = [e.message for e in self.validator.iter_errors(record)]
                    self.assertEqual(errors, [])

    def test_fork_payload_has_no_probability_reason_or_timestamp(self):
        for case, evs in self.events.items():
            for ev in evs:
                with self.subTest(case):
                    fields = {k for k in ev if k != "type"}
                    self.assertEqual(
                        fields, {"endpoint_id", "turn_action", "turn_check_ms"}
                    )
                    self.assertTrue(
                        all(
                            isinstance(v, (str, int)) and not isinstance(v, float)
                            for v in ev.values()
                        )
                    )

    def test_adapter_added_probability_is_still_rejected(self):
        ev = dict(self.events["take"][0])
        record = {k: v for k, v in ev.items() if k != "type"}
        record.update(self.adapter)
        record["take_prob"] = 0.91
        self.assertNotEqual(list(self.validator.iter_errors(record)), [])


if __name__ == "__main__":
    unittest.main()
