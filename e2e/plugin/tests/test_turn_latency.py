"""Offline latency proxy, old vs new, on normal-level synthetic speech (peak RMS 0.1; 0.05 as a second level) with mic noise at RMS 0.001.
The model is stubbed (a turn question answers instantly, P(finished)=0.99) and time is virtual, so these numbers isolate the plugin's own VAD /
silence logic; real model and TTS time is NOT included. "first audio" is the proxy: the moment the plugin decides to emit / release the reply."""
import numpy as np
import pytest

import duplex_harness as h
from test_turn_start import stream

pytest.importorskip("onnxruntime")
OFFS = np.random.default_rng(7).uniform(0.5, 1.0, 20)         # random lead-in so speech ends fall at every phase of the 100 ms append grid


def measure(m, peak):
    check, audio = [], []
    for s, lead in enumerate(OFFS):
        x, end = stream(s, peak, lead=float(lead), tail=3.0, floor=0.001); buf = h.new_buf(m); t0 = m.time.t; rec = {}
        orig = buf._start_check
        def wrapped(thr, idx, _o=orig, _r=rec): _r.setdefault("check", m.time.t - t0); _o(thr, idx)
        buf._start_check = wrapped; tr = buf.t_reply
        for c, p in h.chunks(x, ("lat", s, peak, float(lead))):
            h.append(m, buf, c, p)
            if buf.t_reply != tr: rec.setdefault("reply", m.time.t - t0); tr = buf.t_reply
        assert "check" in rec and "reply" in rec, (s, peak)
        check.append(rec["check"] - end); audio.append(rec["reply"] - end)
    return np.array(check) * 1000, np.array(audio) * 1000


@pytest.mark.parametrize("peak", [0.1, 0.05])
def test_latency_not_worse_than_50ms(new, old, variant, peak, capsys):
    (c_old, a_old), (c_new, a_new) = measure(old, peak), measure(new, peak)
    pc = lambda a, q: float(np.percentile(a, q))
    rows = [("speech end -> turn-check start", c_old, c_new), ("speech end -> reply emit (first-audio proxy)", a_old, a_new)]
    with capsys.disabled():
        print(f"\n[{variant}] peak RMS {peak}, n={len(OFFS)} utterances, ms; model stubbed, virtual clock")
        print("| metric | old P50 | new P50 | old P90 | new P90 |\n|---|---|---|---|---|")
        for n, o, nw in rows: print(f"| {n} | {pc(o, 50):.0f} | {pc(nw, 50):.0f} | {pc(o, 90):.0f} | {pc(nw, 90):.0f} |")
    for _, o, nw in rows:
        assert pc(nw, 50) - pc(o, 50) <= 50 and pc(nw, 90) - pc(o, 90) <= 50
