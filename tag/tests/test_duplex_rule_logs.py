"""Tests for the structured rule logs in tag/plugin/sophon_realtime/duplex.py.

Run from the repository root: python3 -m unittest discover -s tag/tests -q

vllm / vllm_omni / pybase64 are not needed: tiny stand-ins are put in sys.modules while the module is loaded, and removed afterwards.

1. Equivalence: the module as it is now and the baseline (commit 6b17135, blob 7464e7cb) are driven with the same synthetic audio, clock, turn-check
   results and framework callbacks; the decision sequences (what each prepare_append returns, the turn state after it, owner_drop / owner_release
   calls) must be identical. A logger that raises must not change them either.
2. The new log lines carry the required fields and point at the line that emitted them.
"""
import base64 as _b64
import logging
import os
import random
import subprocess
import sys
import threading
import types
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PLUGIN = os.path.join(REPO, "tag", "plugin")
DUPLEX = os.path.join(PLUGIN, "sophon_realtime", "duplex.py")
BASE_BLOB = "7464e7cbbea0e35a97ba39e7d1ddc60eb1006a67"          # duplex.py at 6b1713518eb75f8495415662870f682357372d09
SR = 16000
CHUNK = 1600                                                      # 100 ms

_saved_modules = {}
_stub_names = []


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    if name not in _saved_modules:
        _saved_modules[name] = sys.modules.get(name)
    sys.modules[name] = m
    _stub_names.append(name)
    return m


def setUpModule():
    if np is None:
        raise unittest.SkipTest("numpy is not installed")

    class Base:
        pass

    _stub("pybase64", b64decode=_b64.b64decode, b64encode=_b64.b64encode)
    _stub("vllm")
    _stub("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    _stub("vllm.sampling_params", RequestOutputKind=types.SimpleNamespace(DELTA="delta"), StructuredOutputsParams=Base)
    _stub("vllm_omni")
    _stub("vllm_omni.engine")
    _stub("vllm_omni.engine.duplex")
    _stub("vllm_omni.engine.duplex.config", DuplexCapabilities=Base)
    _stub("vllm_omni.engine.duplex.contracts", DuplexAppendPlan=Base, DuplexOutputAction=Base, DuplexOutputDecision=Base)
    _stub("vllm_omni.engine.duplex.plugin", DefaultDuplexModelSessionState=Base, DuplexDataPlane=Base, DuplexModelPlugin=Base,
          DuplexRuntimeConfigError=Exception, PcmAppendBuffer=Base, PcmAppendReservation=Base)
    pkg = _stub("sophon_realtime")
    pkg.__path__ = [os.path.join(PLUGIN, "sophon_realtime")]
    _stub("sophon_realtime.proc", strip_tags=lambda s: s, tag_clean=lambda s, final=False: s)


def tearDownModule():
    for name in set(_stub_names) | {n for n in sys.modules if n.startswith("sophon_realtime.")}:
        if _saved_modules.get(name) is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = _saved_modules[name]


class Clock:
    def __init__(self, t=1000.0, w=1_800_000_000.0):
        self.t = t; self.w = w

    def advance(self, dt):
        self.t += dt; self.w += dt


class CapLogger:
    """Collects every formatted message; `broken=True` makes only the new "sophon rule:" lines raise (the pre-existing log calls are not exception-safe and are not touched)."""
    def __init__(self, broken=False):
        self.lines = []; self.broken = broken

    def _log(self, msg, *args):
        if self.broken and msg.startswith("sophon rule:"):
            raise RuntimeError("logger exploded")
        self.lines.append(msg % args if args else msg)

    info = warning = error = debug = _log


class FakeThread:
    pending = None

    def __init__(self, target=None, daemon=None, **kw):
        self.target = target

    def start(self):
        FakeThread.pending.append(self.target)


def load(source, name, clock):
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("SOPHON_")}
    try:
        mod = types.ModuleType("sophon_realtime." + name)
        mod.__package__ = "sophon_realtime"; mod.__file__ = DUPLEX
        sys.modules[mod.__name__] = mod
        exec(compile(source, DUPLEX, "exec"), mod.__dict__)
    finally:
        os.environ.update(saved)
    mod.time = types.SimpleNamespace(monotonic=lambda: clock.t, time=lambda: clock.w, sleep=lambda s: None)
    mod.threading = types.SimpleNamespace(Lock=threading.Lock, Thread=FakeThread)
    return mod


def current_source():
    with open(DUPLEX, encoding="utf-8") as f:
        return f.read()


def baseline_source():
    r = subprocess.run(["git", "cat-file", "blob", BASE_BLOB], cwd=REPO, capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8")


def chunk(rng, amp):
    return (np.random.RandomState(rng.randrange(2 ** 31)).standard_normal(CHUNK).astype(np.float32) * np.float32(amp)).astype("<f4")


def snapshot(buf):
    return (round(buf.speech_ms, 6), round(buf.silence_ms, 6), round(buf.voiced_ms, 6), buf.next_check, buf.pending_reply, buf.check_inflight,
            buf.resume_pending, buf.fallback, buf.spec_id is not None, buf.gen, round(buf.noise, 12),
            None if buf.cut is None else (round(buf.cut["t"], 6), buf.cut["said"]), buf.merge_pending, buf.mute_rid, len(buf.utt),
            buf.last_play is None, buf.prev_user is None, buf.last_said, len(buf.reservations), buf.retry is None, round(buf.inflight_t, 6))


def describe(res):
    if res is None:
        return None
    p = res.payload
    if p is None:
        return ("empty",)
    return (p.get("sophon_kind"), p.get("resume"), "spec_id" in p, p.get("say"), p.get("tool_followup"), len(p.get("audio", "")))


def run(mod, clock, ops, seed, last_turn=7):
    """Drive one SophonBuffer through `ops`; return (trace, log lines). All randomness is seeded and independent of the module under test."""
    FakeThread.pending = []
    rng = random.Random(seed)
    trace = []
    buf = mod.SophonBuffer()
    buf.last_turn = last_turn
    results = []                                             # scripted turn-check answers (p, pb)
    owner = {"act": None}
    buf.owner_drop = lambda sid: trace.append(("drop",))
    buf.owner_release = lambda sid: (trace.append(("release",)), results_rng.random() < 0.5)[1]
    buf.owner_active = lambda key: owner["act"]
    results_rng = random.Random(seed + 1)
    mod.turn_prob_http = lambda audio, lang, said: results.pop(0) if results else (0.1, 0.0)
    log = CapLogger()
    mod.logger = log
    reservation = None
    for op in ops:
        kind = op[0]
        if kind == "cut":
            buf.cut = {"t": clock.t - op[1], "said": "so the thing is"}
        elif kind == "last_play":
            buf.last_play = {"t0": clock.t - op[1], "dur": op[2], "entry": {"a_text": "", "a_audio": None}, "text": "hello there friend", "prefix": ""}
        elif kind == "last_play_edge":                       # a reply whose playback position at the next speech chunk sits `delta` s from the barge-in boundary
            dur, delta = op[1], op[2]
            buf.last_play = {"t0": clock.t + 0.1 - 0.25 - (dur - 0.3 + delta), "dur": dur, "entry": {"a_text": "", "a_audio": None}, "text": "hello there friend", "prefix": ""}
        elif kind == "owner":
            owner["act"] = ("rid1", "hello there friend", op[2], clock.t - op[1], "")
        elif kind == "owner_off":
            owner["act"] = None
        elif kind == "check":
            results.append((op[1], op[2]))
        elif kind == "audio":
            _, amp, n, allow_emit = op
            for _ in range(n):
                clock.advance(0.1)
                while FakeThread.pending:
                    FakeThread.pending.pop(0)()
                if reservation is not None:
                    (reservation.commit if rng.random() < 0.8 else reservation.rollback)()
                    buf.inflight_t = 0.0
                    reservation = None
                payload = {"format": "pcm_f32le", "sample_rate_hz": SR, "audio": base64_encode(chunk(rng, amp))}
                res = buf.prepare_append(payload, operation_id="op", chunk_period_ms=100, allow_emit=allow_emit)
                if res is not None:
                    reservation = res
                trace.append((describe(res), snapshot(buf)))
    return trace, log.lines


def base64_encode(arr):
    return _b64.b64encode(arr.tobytes()).decode()


def random_ops(seed):
    rng = random.Random(seed * 7919 + 13)
    ops = []
    for _ in range(rng.randint(3, 7)):
        r = rng.random()
        if r < 0.35:
            ops.append(("last_play", rng.uniform(-0.5, 4.0), rng.uniform(0.5, 4.0)))
        elif r < 0.5:
            ops.append(("owner", rng.uniform(0.0, 3.0), rng.uniform(0.5, 4.0)))
        elif r < 0.6:
            ops.append(("owner_off",))
        if rng.random() < 0.4:
            ops.append(("last_play_edge", rng.uniform(1.0, 4.0), rng.choice([-0.15, -0.05, 0.05, 0.15])))
        if rng.random() < 0.6:
            ops.append(("cut", rng.uniform(0.0, 6.0)))
        for _ in range(rng.randint(0, 3)):
            ops.append(("check", rng.choice([None, 0.05, 0.3, 0.9, 0.99]), rng.choice([0.0, 0.2, 0.6, 0.9])))
        ops.append(("audio", rng.choice([0.011, 0.0125, 0.0135, 0.05, 0.1, 0.25]), rng.randint(1, 20), rng.random() < 0.9))
        ops.append(("audio", rng.choice([0.001, 0.003, 0.0115, 0.0125, 0.02]), rng.randint(1, 30), rng.random() < 0.9))
    for _ in range(rng.randint(1, 3)):                        # resume-shaped blocks: a cut reply, then a short sound, then silence with back-channel-looking check answers
        pb = rng.choice([0.45, 0.5, 0.55, 0.6, 0.9]); p = rng.choice([0.05, 0.3, 0.4, 0.5])
        n = rng.randint(2, 20)
        age = rng.uniform(3.7, 4.0) - 0.1 * n if rng.random() < 0.6 else rng.uniform(2.0, 4.2)
        ops += [("audio", 0.001, 3, True), ("cut", max(age, 0.0))] + [("check", p, pb)] * 6 + [("audio", 0.2, n, True), ("audio", 0.001, 30, True)]
    return ops


# hand-written scenarios that reach each rule; times are in seconds on the fake clock
S_ONSET_QUIET = [("audio", 0.001, 5, True), ("audio", 0.2, 4, True)]
S_ONSET_BARGE_IN = [("audio", 0.001, 3, True), ("last_play", 1.5, 6.0), ("audio", 0.2, 4, True)]
S_ONSET_LAST_PLAY_FINISHED = [("audio", 0.001, 3, True), ("last_play", 9.0, 2.0), ("audio", 0.2, 4, True)]
S_ONSET_WHILE_GENERATING = [("audio", 0.001, 3, True), ("owner", 2.0, 3.0), ("audio", 0.2, 4, True)]
S_RESUME_CHECK = [("audio", 0.001, 3, True), ("cut", 0.5), ("check", 0.1, 0.8), ("audio", 0.2, 4, True), ("audio", 0.001, 25, True)]
# check answers arrive after BC_WINDOW (4 s) from the cut, so the check does not resume; the 2 s fallback (window 4 + 2 s) does
S_RESUME_FALLBACK = [("audio", 0.001, 3, True), ("cut", 3.5)] + [("check", 0.1, 0.8)] * 6 + [
                     ("audio", 0.2, 4, True), ("audio", 0.001, 30, True)]


def kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def rule_lines(lines, rule=None):
    got = [l for l in lines if l.startswith("sophon rule:")]
    return [l for l in got if kv(l)["rule"] == rule] if rule else got


class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_src = baseline_source()
        if cls.base_src is None:
            raise unittest.SkipTest("baseline blob %s is not in this clone (fetch 6b1713518eb7)" % BASE_BLOB[:12])
        cls.new_src = current_source()

    def both(self, ops, seed, broken_logger=False):
        c1 = Clock(); base = load(self.base_src, "dup_base", c1)
        t1, _ = run(base, c1, ops, seed)
        c2 = Clock(); new = load(self.new_src, "dup_new", c2)
        t2, lines = run(new, c2, ops, seed) if not broken_logger else run_with_broken_logger(new, c2, ops, seed)
        return t1, t2, lines

    def test_baseline_is_the_expected_blob(self):
        self.assertNotIn("sophon rule:", self.base_src)

    def test_hand_written_scenarios_give_the_same_decisions(self):
        for name, ops in [("quiet", S_ONSET_QUIET), ("barge", S_ONSET_BARGE_IN), ("finished", S_ONSET_LAST_PLAY_FINISHED),
                          ("generating", S_ONSET_WHILE_GENERATING), ("check", S_RESUME_CHECK), ("fallback", S_RESUME_FALLBACK)]:
            with self.subTest(name):
                t1, t2, _ = self.both(ops, seed=1)
                self.assertGreater(len(t1), 5)
                self.assertEqual(t1, t2)

    def test_random_scenarios_give_the_same_decisions_and_reach_every_rule(self):
        seen = {"energy_onset": 0, "bc_resume_check": 0, "bc_resume_fallback": 0, "bc_resume_emitted": 0}
        emitted = 0
        for seed in range(100):
            t1, t2, lines = self.both(random_ops(seed), seed)
            self.assertEqual(t1, t2, "seed %d" % seed)
            emitted += sum(1 for e in t1 if len(e) == 2 and e[0] is not None and len(e[0]) > 1 and e[0][1] is not None)
            for r in seen:
                seen[r] += len(rule_lines(lines, r))
        self.assertGreater(emitted, 0, "no resume payload was ever produced: the random scenarios are vacuous")
        for r, n in seen.items():
            self.assertGreater(n, 0, "rule %s never fired in the random scenarios" % r)

    def test_a_failing_rule_log_does_not_change_decisions(self):
        for name, ops in [("barge", S_ONSET_BARGE_IN), ("check", S_RESUME_CHECK), ("fallback", S_RESUME_FALLBACK)]:
            with self.subTest(name):
                t1, t2, _ = self.both(ops, seed=3, broken_logger=True)
                self.assertEqual(t1, t2)
        for seed in range(30):
            t1, t2, _ = self.both(random_ops(seed), seed, broken_logger=True)
            self.assertEqual(t1, t2, "seed %d" % seed)


def run_with_broken_logger(mod, clock, ops, seed):
    # run() installs a fresh CapLogger; swap in a broken one for the same drive
    orig = CapLogger.__init__
    CapLogger.__init__ = lambda self, broken=True: orig(self, broken=True)
    try:
        return run(mod, clock, ops, seed)
    finally:
        CapLogger.__init__ = orig


class RuleLogFields(unittest.TestCase):
    COMMON = ("rule", "src", "mono", "wall", "buf", "turn", "gen")

    @classmethod
    def setUpClass(cls):
        cls.src = current_source()
        cls.src_lines = cls.src.split("\n")

    def drive(self, ops):
        clock = Clock()
        mod = load(self.src, "dup_logs", clock)
        trace, lines = run(mod, clock, ops, seed=1)
        return clock, trace, lines

    def check_common(self, line, rule, clock):
        self.assertNotIn("\n", line)
        f = kv(line)
        for k in self.COMMON:
            self.assertIn(k, f, "%s missing from: %s" % (k, line))
        self.assertEqual(f["rule"], rule)
        name, no = f["src"].rsplit(":", 1)
        self.assertEqual(name, "duplex.py")
        self.assertIn('_rule_log("%s"' % rule, self.src_lines[int(no) - 1])
        self.assertGreaterEqual(float(f["mono"]), 1000.0)
        self.assertLessEqual(float(f["mono"]), clock.t)
        self.assertGreaterEqual(float(f["wall"]), 1_800_000_000.0)
        self.assertEqual(f["turn"], "7")
        self.assertTrue(f["gen"].lstrip("-").isdigit())
        self.assertNotEqual(f["buf"], "None")
        return f

    def onset(self, ops):
        clock, _, lines = self.drive(ops)
        got = rule_lines(lines, "energy_onset")
        self.assertEqual(len(got), 1, got)
        f = self.check_common(got[0], "energy_onset", clock)
        for k in ("rms", "thr", "noise", "playing", "barge_in"):
            self.assertIn(k, f)
        self.assertAlmostEqual(float(f["thr"]), 0.012, places=4)
        self.assertGreater(float(f["rms"]), float(f["thr"]))
        return f

    def test_energy_onset_with_nothing_playing(self):
        f = self.onset(S_ONSET_QUIET)
        self.assertEqual((f["playing"], f["barge_in"]), ("False", "False"))

    def test_energy_onset_cutting_a_playing_reply(self):
        f = self.onset(S_ONSET_BARGE_IN)
        self.assertEqual((f["playing"], f["barge_in"]), ("True", "True"))

    def test_energy_onset_after_the_reply_finished_playing(self):
        f = self.onset(S_ONSET_LAST_PLAY_FINISHED)
        self.assertEqual((f["playing"], f["barge_in"]), ("False", "False"))

    def test_energy_onset_while_the_reply_is_still_generating(self):
        f = self.onset(S_ONSET_WHILE_GENERATING)
        self.assertEqual((f["playing"], f["barge_in"]), ("True", "True"))

    def test_bc_resume_check_and_the_shared_emitted_line_name_the_check_path(self):
        clock, trace, lines = self.drive(S_RESUME_CHECK)
        chk = rule_lines(lines, "bc_resume_check"); fb = rule_lines(lines, "bc_resume_fallback"); em = rule_lines(lines, "bc_resume_emitted")
        self.assertEqual((len(chk), len(fb), len(em)), (1, 0, 1), lines)
        f = self.check_common(chk[0], "bc_resume_check", clock)
        for k in ("check_idx", "p_finished", "p_backchannel", "thr", "speech_ms", "cut_age_s"):
            self.assertIn(k, f)
        self.assertEqual(self.check_common(em[0], "bc_resume_emitted", clock)["via"], "check")
        old = [l for l in lines if l.startswith("sophon timing: resume emitted")]
        self.assertEqual(len(old), 1)
        self.assertTrue(old[0].endswith("via=check"), old[0])

    def test_bc_resume_fallback_and_the_shared_emitted_line_name_the_fallback_path(self):
        clock, trace, lines = self.drive(S_RESUME_FALLBACK)
        chk = rule_lines(lines, "bc_resume_check"); fb = rule_lines(lines, "bc_resume_fallback"); em = rule_lines(lines, "bc_resume_emitted")
        self.assertEqual((len(chk), len(fb), len(em)), (0, 1, 1), lines)
        f = self.check_common(fb[0], "bc_resume_fallback", clock)
        for k in ("silence_ms", "speech_ms", "p_backchannel", "max_p_finished", "cut_age_s"):
            self.assertIn(k, f)
        self.assertGreaterEqual(float(f["silence_ms"]), 2000.0)
        self.assertEqual(self.check_common(em[0], "bc_resume_emitted", clock)["via"], "fallback")
        old = [l for l in lines if l.startswith("sophon timing: resume emitted")]
        self.assertTrue(old[0].endswith("via=fallback"), old[0])

    def test_log_sites_are_where_the_spec_says_relative_to_the_code_they_observe(self):
        def line_of(rule):
            return [i for i, l in enumerate(self.src_lines) if '_rule_log("%s"' % rule in l]
        for rule in ("energy_onset", "bc_resume_check", "bc_resume_fallback", "bc_resume_emitted"):
            self.assertEqual(len(line_of(rule)), 1, rule)
        s = self.src_lines
        self.assertIn("self.speech_ms = self.voiced_ms", s[line_of("energy_onset")[0] - 1])
        self.assertIn("self.resume_pending = True", s[line_of("bc_resume_fallback")[0] - 2])
        self.assertIn("self.resume_pending = True", s[line_of("bc_resume_check")[0] - 2])
        self.assertIn("resume emitted", s[line_of("bc_resume_emitted")[0] - 1])


if __name__ == "__main__":
    unittest.main()
