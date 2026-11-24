"""Test harness for tag/plugin/sophon_realtime/duplex.py (no GPU, vLLM, vllm_omni).

The engine modules are replaced by empty stand-ins, and the turn readout
(turn_prob_http, the call into stage 0) is replaced by a scripted function.
Everything else in SophonBuffer (silence timing, checkpoints, thresholds, fallback)
is the real code."""

import base64
import importlib
import logging
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.environ.get("SOPHON_PLUGIN_DIR") or os.path.join(HERE, "..", "plugin")
TURN_ENV = (
    "SOPHON_TURN_CHECKS",
    "SOPHON_TURN_THRESHOLDS",
    "SOPHON_TURN_FALLBACK",
    "SOPHON_PERIODIC_CHECK",
    "SOPHON_IGNORE_SHORT",
    "SOPHON_BC_RESUME",
    "SOPHON_SPEC_AT",
)
SR = 16000


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def load_duplex():
    """Import duplex.py with the turn-taking environment variables unset.

    The module constants are then the defaults."""
    if "sophon_realtime.duplex" in sys.modules:
        return sys.modules["sophon_realtime.duplex"]
    for k in TURN_ENV:
        os.environ.pop(k, None)
    os.environ["SOPHON_SPEC_AT"] = "0"
    sys.path.insert(0, os.path.abspath(PLUGIN_DIR))
    _module("vllm")
    _module("vllm.logger", init_logger=logging.getLogger)
    _module(
        "vllm.sampling_params",
        RequestOutputKind=object,
        StructuredOutputsParams=object,
    )
    for n in ("vllm_omni", "vllm_omni.engine", "vllm_omni.engine.duplex"):
        _module(n)
    _module("vllm_omni.engine.duplex.config", DuplexCapabilities=object)
    _module(
        "vllm_omni.engine.duplex.contracts",
        DuplexAppendPlan=object,
        DuplexOutputAction=object,
        DuplexOutputDecision=object,
    )
    _module(
        "vllm_omni.engine.duplex.plugin",
        DefaultDuplexModelSessionState=object,
        DuplexDataPlane=object,
        DuplexModelPlugin=object,
        DuplexRuntimeConfigError=RuntimeError,
        PcmAppendBuffer=object,
        PcmAppendReservation=object,
    )
    sys.modules["pybase64"] = base64
    pkg = importlib.import_module("sophon_realtime")
    _module(
        "sophon_realtime.proc",
        strip_tags=lambda s: s,
        tag_clean=lambda s, final: s,
    )
    pkg.proc = sys.modules["sophon_realtime.proc"]
    return importlib.import_module("sophon_realtime.duplex")


class Readouts:
    """Scripted stand-in for turn_prob_http.

    Returns (P(finished), P(backchannel)) from the script, the last entry repeats;
    None raises (a failed readout)."""

    def __init__(self, *script, delay_s=0.02):
        self.script, self.delay_s, self.calls = list(script), delay_s, 0

    def __call__(self, audio, lang, said):
        time.sleep(self.delay_s)
        v = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if v is None:
            raise RuntimeError("scripted readout failure")
        return v


class Call:
    """One session buffer fed with 16 kHz float32 audio (100 ms appends)."""

    def __init__(self, duplex, readouts):
        self.dx = duplex
        self.dx.turn_prob_http = readouts
        self.buf = duplex.SophonBuffer()
        self.events, self.outputs = [], []
        self.buf.owner_event = lambda buf, ev: self.events.append(ev)
        self.buf.owner_drop = lambda sid: None
        self.buf.owner_release = lambda sid: False
        self.buf.owner_active = lambda bid: None

    def feed(self, kind, ms=100):
        n = SR * ms // 1000
        if kind == "speech":
            pcm = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype("<f4")
        else:
            pcm = np.zeros(n, "<f4")
        payload = {
            "audio": base64.b64encode(pcm.tobytes()).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": SR,
        }
        out = self.buf.prepare_append(
            payload, operation_id="op", chunk_period_ms=100, allow_emit=True
        )
        self.outputs.append(out)
        t_end = time.monotonic() + 3
        while self.buf.check_inflight and time.monotonic() < t_end:
            time.sleep(0.002)
        return out

    def speech(self, ms=500):
        for _ in range(ms // 100):
            self.feed("speech")

    def silence(self, ms):
        for _ in range(ms // 100):
            self.feed("silence")

    def reply_appends(self):
        return [
            o
            for o in self.outputs
            if o is not None and (o.payload or {}).get("sophon_kind") == "reply"
        ]
