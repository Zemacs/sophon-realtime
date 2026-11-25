"""Offline harness for the duplex plugins: stubs the serving framework, loads a plugin's duplex.py (working tree or a pinned older blob),
synthesizes speech / noise with numpy, and runs real Silero (the repo's ONNX file) the way the framework's server VAD reports it."""
from __future__ import annotations

import base64
import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SR = 16000
BASE_SHA = "0eb549fe3c30dc853666b4f9c7301df8c8d2fe1a"        # the commit before the Silero turn-start change
VARIANTS = ("e2e", "tag", "flash")
SILERO = ROOT / "e2e" / "config" / "silero_vad.onnx"


def _stub_framework():
    if "vllm_omni" in sys.modules:
        return
    def mod(name, **kw):
        m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
    class _Base:
        def __init__(self, *a, **k): self.__dict__.update(k)
    class _Delta: DELTA = "delta"
    mod("pybase64", b64decode=base64.b64decode, b64encode=base64.b64encode)
    mod("vllm"); mod("vllm.logger", init_logger=logging.getLogger); mod("vllm.sampling_params", RequestOutputKind=_Delta)
    mod("vllm_omni"); mod("vllm_omni.engine"); mod("vllm_omni.engine.duplex")
    mod("vllm_omni.engine.duplex.config", DuplexCapabilities=_Base)
    mod("vllm_omni.engine.duplex.contracts", DuplexAppendPlan=_Base, DuplexOutputAction=_Base, DuplexOutputDecision=_Base)
    mod("vllm_omni.engine.duplex.plugin", DefaultDuplexModelSessionState=_Base, DuplexDataPlane=object, DuplexModelPlugin=object,
        DuplexRuntimeConfigError=RuntimeError, PcmAppendBuffer=object, PcmAppendReservation=object)


class VirtualTime:
    """Deterministic clock the plugin sees instead of `time`."""
    def __init__(self): self.t = 1000.0
    def monotonic(self): return self.t
    def time(self): return 1.7e9 + self.t
    def sleep(self, s): self.t += s


class InlineThreads:
    """threading shim: Thread.start() only queues the target; the harness runs it outside the buffer lock (the plugin starts it while holding it)."""
    Lock = staticmethod(__import__("threading").Lock)
    Thread_queue: list = []
    class Thread:
        def __init__(self, target=None, daemon=None, **_): self.target = target
        def start(self): InlineThreads.Thread_queue.append(self.target)


def load(variant: str, *, old: bool = False):
    """Import <variant>/plugin/sophon_realtime/duplex.py (and bridge.py) as a fresh package. old=True swaps in the files as of BASE_SHA."""
    _stub_framework()
    src = ROOT / variant / "plugin" / "sophon_realtime"
    name = f"sophon_rt_{variant}_{'old' if old else 'new'}_{os.getpid()}"
    if name in sys.modules:
        return sys.modules[name + ".duplex"]
    d = Path(tempfile.mkdtemp(prefix="sophon_rt_")) / "pkg"
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("deploy", "__pycache__"))
    if old:
        for f in ("duplex.py", "bridge.py"):
            blob = subprocess.run(["git", "show", f"{BASE_SHA}:{variant}/plugin/sophon_realtime/{f}"], cwd=ROOT, capture_output=True)
            if blob.returncode: raise FileNotFoundError(f"base commit {BASE_SHA[:7]} not available (shallow clone?)")
            (d / f).write_bytes(blob.stdout)
    pkg = types.ModuleType(name); pkg.__path__ = [str(d)]; sys.modules[name] = pkg
    sys.modules[name + ".proc"] = types.SimpleNamespace(strip_tags=lambda s: s, tag_clean=lambda s, final=False: s)
    m = importlib.import_module(name + ".duplex")
    m.time = VirtualTime(); m.threading = InlineThreads; m.turn_prob_http = lambda audio, lang, said: (0.99, 0.0)
    return m


def new_buf(m):
    b = m.SophonBuffer(); b.owner_release = lambda sid: True; b.owner_drop = lambda sid: None; b.owner_active = lambda i: None
    return b


# ---------------------------------------------------------------- Silero as the framework reports it
class Silero:
    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(SILERO), providers=["CPUExecutionProvider"]); self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32); self.ctx = np.zeros(64, np.float32); self.rem = np.zeros(0, np.float32); self.last = 0.0

    def chunk_prob(self, a):
        """max speech probability over the 512-sample frames completed by this chunk (what payload['vad']['speech_probability'] carries)."""
        x = np.concatenate([self.rem, a]); n = x.size // 512; self.rem = x[n * 512:].copy(); ps = []
        for i in range(n):
            fr = x[i * 512:(i + 1) * 512]
            p, self.state = self.sess.run(None, {"input": np.concatenate([self.ctx, fr])[None].astype(np.float32), "state": self.state, "sr": np.array(16000, np.int64)})
            self.ctx = fr[-64:]; ps.append(float(p[0][0]))
        self.last = max(ps) if ps else self.last
        return self.last


_VOWELS = np.array([(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410), (660, 1720, 2410), (440, 1020, 2240), (850, 1220, 2810)], float)


def speech(seed: int, dur: float = 2.2) -> np.ndarray:
    """Synthetic voiced speech: glottal pulses (declining f0), 3 moving formants, syllable envelope, fricative-like onsets. Peak-normalised later."""
    rng = np.random.default_rng(seed); n = int(dur * SR); tt = np.arange(n); f0b = rng.uniform(100, 200)
    t = int(0.2 * SR); syl = []
    while t < n - int(0.35 * SR):
        L = int(rng.uniform(0.14, 0.26) * SR); syl.append((t, L, _VOWELS[rng.integers(len(_VOWELS))])); t += L + int(rng.uniform(0.0, 0.05) * SR)
    cen = np.array([s[0] + s[1] // 2 for s in syl]); F = np.stack([np.interp(tt, cen, [s[2][i] for s in syl]) for i in range(3)], 1)
    f0 = f0b * (1 - 0.12 * tt / n) * (1 + 0.05 * np.sin(2 * np.pi * np.interp(tt, cen, rng.uniform(-1, 1, len(syl))))) * np.interp(tt, cen, 1 + rng.uniform(-.08, .12, len(syl)))
    amp = np.zeros(n)
    for s, L, _ in syl: amp[s:s + L] = np.maximum(amp[s:s + L], np.hanning(L) ** 0.6)
    ph = np.cumsum(f0) / SR; fr_ = ph - np.floor(ph); oq = 0.6
    g = np.where(fr_ < oq * 0.7, 0.5 * (1 - np.cos(np.pi * fr_ / (oq * 0.7))), np.where(fr_ < oq, np.cos(0.5 * np.pi * (fr_ - oq * 0.7) / (oq * 0.3)), 0.0))
    d = np.diff(g, prepend=0) * SR / f0.mean(); src = (d + rng.normal(0, 0.06, n) * np.abs(d).max() * 0.3) * amp
    for s, L, _ in syl:
        k = int(0.04 * SR); b = np.diff(rng.normal(0, 1, k), prepend=0) * np.hanning(k); a0 = max(s - k // 2, 0); src[a0:a0 + k] += b[:n - a0] * 0.2 * np.abs(src).max()
    N, H = 640, 160; win = np.hanning(N); out = np.zeros(n + N); ws = np.zeros(n + N); fr = np.fft.rfftfreq(N, 1 / SR)
    for st in range(0, n - N + 1, H):
        X = np.fft.rfft(src[st:st + N] * win); c = st + N // 2
        M = sum(1.0 / np.sqrt(1 + ((fr - F[c, i]) / bw) ** 2) for i, bw in enumerate((90, 120, 180))) + 0.15 / np.sqrt(1 + ((fr - 3500) / 300) ** 2)
        out[st:st + N] += np.fft.irfft(X * M, N) * win; ws[st:st + N] += win ** 2
    return (out[:n] / np.maximum(ws[:n], 1e-3)).astype(np.float32)


def peak_rms(x) -> float:
    k = len(x) // 320; f = x[:k * 320].reshape(k, 320); return float(np.sqrt((f * f).mean(1)).max())


def scaled(x, peak):  # peak = max RMS over 20 ms windows
    return (x * (peak / peak_rms(x))).astype(np.float32)


def noise(kind: str, n: int, rms: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed); X = np.fft.rfft(r.normal(0, 1, n)); f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    if kind == "pink": X = X / np.sqrt(f)
    elif kind == "cabin":                        # low-frequency road/engine rumble: ~1/f below 180 Hz, steeply rolled off above, plus slowly wobbling 38/76/114 Hz hum
        X = X / f / np.sqrt(1 + (f / 180) ** 4)
    y = np.fft.irfft(X, n)
    if kind == "cabin":
        t = np.arange(n) / SR; hum = sum(np.sin(2 * np.pi * h * (1 + 0.02 * np.sin(2 * np.pi * 0.3 * t)) * t + r.uniform(0, 6)) / (i + 1) for i, h in enumerate((38, 76, 114)))
        y = y / y.std() + 0.8 * hum / hum.std()
    return (y / y.std() * rms).astype(np.float32)


# ---------------------------------------------------------------- driving a buffer
_PROBS: dict = {}


def chunks(audio: np.ndarray, key=None, ms: int = 100):
    """[(chunk, silero_prob)] for 100 ms appends; Silero runs once per distinct `key` and is cached."""
    n = SR * ms // 1000; parts = [audio[i:i + n] for i in range(0, len(audio), n)]
    if key is None or key not in _PROBS:
        v = Silero(); ps = [v.chunk_prob(c) for c in parts]
        if key is not None: _PROBS[key] = ps
    else: ps = _PROBS[key]
    return list(zip(parts, ps))


def payload(chunk: np.ndarray, prob=None) -> dict:
    p = {"audio": base64.b64encode(chunk.astype("<f4").tobytes()).decode(), "format": "pcm_f32le", "sample_rate_hz": SR}
    if prob is not None: p["vad"] = {"speech_probability": prob, "backend": "silero"}
    return p


def append(m, buf, chunk, prob=None, *, ms=100):
    """One framework append at virtual time +ms; then runs the turn-question threads the plugin queued. Returns the reservation (committed)."""
    m.time.t += ms / 1000
    r = buf.prepare_append(payload(chunk, prob), operation_id="op", chunk_period_ms=ms, allow_emit=True)
    if r is not None: r.commit()
    while InlineThreads.Thread_queue: InlineThreads.Thread_queue.pop(0)()
    return r
