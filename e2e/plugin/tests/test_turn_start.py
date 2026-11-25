"""Turn start from the host framework's Silero verdict (payload['is_speech']). Offline: audio is synthesized here; the host's server VAD is re-created in
duplex_harness.FrameworkVAD on the repo's Silero ONNX file; the rest of the framework is stubbed. A turn counts as opened when the plugin emits `turn_started`
(the previous code emits no events, so for it opened = the buffer holds speech).
Run network-isolated:  unshare -rn sh -c 'ifconfig lo up; python -m pytest e2e/plugin/tests'  (the bridge test needs loopback only)"""
import types

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


def drive(m, x, fw=True):
    """Feed a stream; returns (buffer, span records, seconds until the first turn_started (new code) or until the buffer held speech (old code))."""
    buf = h.new_buf(m); spans = []; buf.span_sink = spans.append; t0 = m.time.t; opened = None
    for c, verdict in h.chunks(x):
        h.append(m, buf, c, verdict if fw else None)
        if opened is None and (any(r["name"] == "turn_started" for r in spans) if hasattr(buf, "span_sink") and hasattr(m.SophonBuffer, "_span") else buf.speech_ms > 0): opened = m.time.t - t0
    return buf, spans, opened


@pytest.mark.parametrize("peak", LEVELS)
def test_quiet_speech_opens_a_turn(new, peak):
    for s in SEEDS:
        x, _ = stream(s, peak); assert h.framework_heard_speech(x), "premise: the host's Silero VAD reports speech for this input"
        buf, spans, opened = drive(new, x)
        assert opened is not None, f"no turn_started for utterance {s} at peak RMS {peak}"
        st = [r for r in spans if r["name"] == "turn_check.start"]; ts = [r for r in spans if r["name"] == "turn_started"]
        assert st and ts and st[0]["span_id"] == ts[0]["span_id"] and ts[0]["framework_is_speech"] is True and ts[0]["source"] == "framework_vad"
        assert buf.turn_id >= 1


@pytest.mark.parametrize("peak", LEVELS)
def test_old_code_never_opens_on_the_same_quiet_speech(old, peak):
    for s in SEEDS:
        x, _ = stream(s, peak); _, _, opened = drive(old, x)
        assert opened is None, f"old code unexpectedly opened a turn for utterance {s} at peak RMS {peak}"


@pytest.mark.parametrize("kind", ["white", "pink", "cabin"])
@pytest.mark.parametrize("rms", [0.001, 0.003, 0.01, 0.03])
def test_noise_opens_no_turn(new, kind, rms):
    for s in range(2):
        buf, spans, opened = drive(new, h.noise(kind, 8 * h.SR, rms, 100 + s))
        assert opened is None and not spans and buf.turn_id == 0, f"{kind} noise rms {rms} seed {s}: {spans[:2]}"


def test_digital_silence_and_dither_open_no_turn(new):
    for name, x in (("zeros", np.zeros(5 * h.SR, np.float32)), ("dither", np.random.default_rng(1).normal(0, 3e-5, 5 * h.SR).astype(np.float32))):
        buf, spans, opened = drive(new, x); assert opened is None and not spans, name


def test_short_burst_is_rejected_with_a_structured_reason(new):
    buf = h.new_buf(new); spans = []; buf.span_sink = spans.append; buf.session_id = "sess-1"
    h.append(new, buf, h.scaled(h.speech(1), 0.05)[4000:4960], True, ms=60); h.append(new, buf, np.zeros(1600, np.float32), False)     # 60 ms voiced: below the minimum
    assert buf.speech_ms == 0 and buf.turn_id == 0 and not any(r["name"] == "turn_started" for r in spans)
    rej = [r for r in spans if r["name"] == "turn_check.rejected"]; st = [r for r in spans if r["name"] == "turn_check.start"]
    assert len(st) == 1 and len(rej) == 1 and rej[0]["span_id"] == st[0]["span_id"]
    r = rej[0]
    assert r["reason"] == "voiced_ms_below_min" and r["broken_by"] == "framework_not_speech" and r["voiced_ms"] == 60
    for k in ("session_id", "turn_id", "span_id", "parent_span_id", "t_mono", "t_wall", "framework_is_speech", "rms", "noise_floor", "thr_energy", "min_voiced_ms"):
        assert k in r, k
    assert r["session_id"] == "sess-1" and r["turn_id"] == 1 and len(r["span_id"]) == 16


def test_turn_started_event_carries_the_span_fields(new):
    x, _ = stream(0, 0.05); buf, spans, _ = drive(new, x); buf2 = None
    ev = [r for r in spans if r["name"] == "turn_started"]
    assert len(ev) == 1 and ev[0]["turn_id"] == 1 and ev[0]["status"] == "started" and ev[0]["parent_span_id"] and ev[0]["t_wall"] > 1e9 and ev[0]["voiced_ms"] >= 100


def test_the_tail_of_the_framework_flag_does_not_reopen_a_turn_after_the_reply(new):
    x, _ = stream(0, 0.1); buf, spans, _ = drive(new, x)
    assert buf.turn_id == 1 and buf.t_reply > 0, "one utterance -> exactly one turn, answered, even though the host flag stays true ~500 ms after speech"


def test_energy_fallback_when_the_host_sends_no_verdict(new, old):
    x, _ = stream(0, 0.1)
    buf, spans, opened = drive(new, x, fw=False)
    assert opened is not None and spans[0]["source"] == "energy" and drive(old, x, fw=False)[2] is not None


def test_energy_floor_is_configurable_for_the_fallback(new, monkeypatch):
    x, _ = stream(0, 0.1); monkeypatch.setattr(new, "ENERGY_FLOOR", 0.5)
    assert drive(new, x, fw=False)[2] is None


def test_barge_in_is_unchanged(new, old):
    out = []
    for m in (old, new):
        buf = h.new_buf(m); entry = {"a_text": "", "a_audio": None}
        buf.last_play = {"t0": m.time.t - 1.0, "dur": 5.0, "entry": entry, "text": "hello there, how are you today", "prefix": ""}
        x, _ = stream(0, 0.1, lead=0.4, tail=0.5)
        for c, fw in h.chunks(x[:int(1.5 * h.SR)]): h.append(m, buf, c, fw)
        out.append((entry["a_text"], buf.last_said, buf.cut["said"], buf.last_play is None, buf.speech_ms > 0))
    (o_txt, o_said, _, *o_f), (n_txt, n_said, _, *n_f) = out           # same handling; the new code opens one framework confirmation (~100 ms, a char or two of this reply) later
    full, cut = "hello there, how are you today", "……（被用户打断）"
    assert o_f == n_f == [True, True] and o_txt == o_said + cut and n_txt == n_said + cut and full.startswith(n_said) and 0 <= len(n_said) - len(o_said) <= 3
