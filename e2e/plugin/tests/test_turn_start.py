"""Turn start from the Silero speech probability. Offline: audio is synthesized here, Silero is the repo's ONNX file, the framework is stubbed.
Run network-isolated:  unshare -rn sh -c 'ip link set lo up; python -m pytest e2e/plugin/tests'"""
import numpy as np
import pytest

import duplex_harness as h

pytest.importorskip("onnxruntime")
LEVELS = (0.003, 0.006, 0.01)          # peak RMS over 20 ms windows
SEEDS = range(6)


def stream(seed, peak, lead=0.8, tail=3.0, floor=0.0004):
    """[mic noise floor] + speech scaled to `peak` + tail, noise under everything; returns (audio, true speech end in s)."""
    sp = h.scaled(h.speech(seed), peak); n_l, n_t = int(lead * h.SR), int(tail * h.SR)
    x = np.concatenate([np.zeros(n_l, np.float32), sp, np.zeros(n_t, np.float32)])
    k = len(sp) // 320; rms = np.sqrt((sp[:k * 320].reshape(k, 320) ** 2).mean(1)); end = lead + (np.flatnonzero(rms > 0.05 * rms.max())[-1] + 1) * 0.02
    return x + h.noise("white", len(x), floor, seed), end


def drive(m, x, key, vad=True):
    buf = h.new_buf(m); spans = []; buf.span_sink = spans.append; t0 = m.time.t; opened = None
    for c, p in h.chunks(x, key):
        h.append(m, buf, c, p if vad else None)
        if opened is None and buf.speech_ms > 0: opened = m.time.t - t0
    return buf, spans, opened


@pytest.mark.parametrize("peak", LEVELS)
def test_quiet_speech_opens_a_turn(new, peak):
    for s in SEEDS:
        x, _ = stream(s, peak); probs = [p for _, p in h.chunks(x, ("q", s, peak))]
        assert max(probs) >= 0.5, "premise: Silero reports speech for this input"
        buf, spans, opened = drive(new, x, ("q", s, peak))
        assert opened is not None, f"no turn opened for utterance {s} at peak RMS {peak}"
        st = [r for r in spans if r["name"] == "turn_check.start"]; ac = [r for r in spans if r["name"] == "turn_check.accepted"]
        assert st and ac and st[0]["span_id"] == ac[0]["span_id"] and ac[0]["speech_prob"] >= 0.5 and ac[0]["path"] == "silero"
        assert buf.turn_id >= 1


@pytest.mark.parametrize("peak", LEVELS)
def test_old_code_never_opens_on_the_same_quiet_speech(old, peak):
    for s in SEEDS:
        x, _ = stream(s, peak); _, _, opened = drive(old, x, ("q", s, peak))
        assert opened is None, f"old code unexpectedly opened a turn for utterance {s} at peak RMS {peak}"


@pytest.mark.parametrize("kind", ["white", "pink", "cabin"])
@pytest.mark.parametrize("rms", [0.001, 0.003, 0.01, 0.03])
def test_noise_opens_no_turn(new, kind, rms):
    for s in range(2):
        x = h.noise(kind, 8 * h.SR, rms, 100 + s); buf, spans, opened = drive(new, x, (kind, rms, s))
        assert opened is None and not spans and buf.turn_id == 0, f"{kind} noise rms {rms} seed {s}: {spans[:2]}"


def test_digital_silence_and_dither_open_no_turn(new):
    for name, x in (("zeros", np.zeros(5 * h.SR, np.float32)), ("dither", np.random.default_rng(1).normal(0, 3e-5, 5 * h.SR).astype(np.float32))):
        buf, spans, opened = drive(new, x, (name,)); assert opened is None and not spans, name


def test_short_burst_is_rejected_with_a_structured_reason(new):
    buf = h.new_buf(new); spans = []; buf.span_sink = spans.append; buf.session_id = "sess-1"
    burst = h.scaled(h.speech(1), 0.05)[4000:4960]                      # 60 ms: below the 100 ms minimum
    h.append(new, buf, burst, 0.9, ms=60); h.append(new, buf, np.zeros(1600, np.float32), 0.05)
    assert buf.speech_ms == 0 and buf.turn_id == 0
    rej = [r for r in spans if r["name"] == "turn_check.rejected"]; st = [r for r in spans if r["name"] == "turn_check.start"]
    assert len(st) == 1 and len(rej) == 1 and rej[0]["span_id"] == st[0]["span_id"]
    r = rej[0]
    assert r["reason"] == "voiced_ms_below_min" and r["broken_by"] == "prob_below_threshold" and r["voiced_ms"] == 60
    for k in ("session_id", "turn_id", "span_id", "parent_span_id", "t_mono", "t_wall", "speech_prob", "rms", "noise_floor", "thr_prob", "thr_energy", "min_voiced_ms"):
        assert k in r, k
    assert r["session_id"] == "sess-1" and r["turn_id"] == 1 and len(r["span_id"]) == 16 and r["thr_prob"] == new.TURN_START_PROB


def test_energy_floor_is_configurable(new, monkeypatch):
    x, _ = stream(0, 0.01); monkeypatch.setattr(new, "ENERGY_FLOOR", 0.05)
    _, spans, opened = drive(new, x, ("q", 0, 0.01)); assert opened is None
    gate = [r for r in spans if r["name"] == "turn_check.rejected"]
    assert len(gate) == 1 and gate[0]["reason"] == "energy_below_floor" and gate[0]["energy_floor"] == 0.05 and gate[0]["speech_prob"] >= 0.5


def test_without_a_vad_block_the_energy_rule_still_opens_on_normal_speech(new, old):
    x, _ = stream(0, 0.1)
    for m in (new, old): assert drive(m, x, ("q", 0, 0.1), vad=False)[2] is not None


def test_barge_in_is_unchanged(new, old):
    out = []
    for m in (old, new):
        buf = h.new_buf(m); entry = {"a_text": "", "a_audio": None}
        buf.last_play = {"t0": m.time.t - 1.0, "dur": 5.0, "entry": entry, "text": "hello there, how are you today", "prefix": ""}
        x, _ = stream(0, 0.1, lead=0.4, tail=0.5)
        for c, p in h.chunks(x[:int(1.5 * h.SR)], ("q", 0, 0.1, "bi")): h.append(m, buf, c, p)
        out.append((entry["a_text"], buf.last_said, buf.cut["said"], buf.last_play is None, buf.speech_ms > 0))
    assert out[0] == out[1] and "被用户打断" in out[0][0] and out[0][3] and out[0][4]
