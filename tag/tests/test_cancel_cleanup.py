"""Reply cleanup when a response is cancelled (sophon_realtime.duplex).

Run from the repository root:  python -m unittest discover -s tag/tests -v

The plugin imports vLLM / vLLM-Omni / torch at module level. When they are not installed, minimal stand-ins are put in
sys.modules so the pure-Python bookkeeping can still be exercised; tests that push audio through prepare_append /
plan_append need the real numpy and are skipped without it.
"""
import base64
import importlib
import logging
import os
import sys
import threading
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../tag
sys.path.insert(0, os.path.join(ROOT, "plugin"))
for k, v in {"SOPHON_WARMUP": "0", "SOPHON_VISION": "0", "SOPHON_FENCE_FLOOR": "0", "SOPHON_PORT": ""}.items():
    os.environ.setdefault(k, v)


def _have(name):
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_NUMPY = _have("numpy")


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _install_stand_ins():
    if not HAVE_NUMPY:
        class _Arr(list):
            @property
            def size(self): return len(self)
        _mod("numpy", float32="float32", ndarray=_Arr, zeros=lambda n, dtype=None: _Arr([0.0] * int(n)),
             concatenate=lambda xs: _Arr([y for x in xs for y in x]))
    if not _have("pybase64"):
        _mod("pybase64", b64decode=base64.b64decode, b64encode=base64.b64encode)
    if not _have("torch"):
        _mod("torch")
    if not _have("vllm"):
        class _Kind:
            CUMULATIVE = "cumulative"; DELTA = "delta"; FINAL_ONLY = "final_only"

        class _SO:
            def __init__(self, **kw): self.__dict__.update(kw)
        _mod("vllm")
        _mod("vllm.logger", init_logger=logging.getLogger)
        _mod("vllm.sampling_params", RequestOutputKind=_Kind, StructuredOutputsParams=_SO)
    if not _have("vllm_omni"):
        class _Any:
            def __init__(self, *a, **kw): self.args = a; self.__dict__.update(kw)

        class _State:
            def __init__(self, audio_buffer=None, **kw): self.audio_buffer = audio_buffer

        class _Err(Exception):
            pass
        for n in ("vllm_omni", "vllm_omni.engine", "vllm_omni.engine.duplex", "vllm_omni.model_executor",
                  "vllm_omni.model_executor.stage_input_processors"):
            _mod(n)
        _mod("vllm_omni.engine.duplex.config", DuplexCapabilities=_Any)
        _mod("vllm_omni.engine.duplex.contracts", DuplexAppendPlan=_Any, DuplexOutputAction=_Any, DuplexOutputDecision=_Any)
        _mod("vllm_omni.engine.duplex.plugin", DefaultDuplexModelSessionState=_State, DuplexDataPlane=object,
             DuplexModelPlugin=object, DuplexRuntimeConfigError=_Err, PcmAppendBuffer=object, PcmAppendReservation=object)
        _mod("vllm_omni.data_entry_keys", IdsStruct=_Any, MetaStruct=_Any, OmniPayloadStruct=_Any)
        _mod("vllm_omni.model_executor.stage_input_processors.tts_utils",
             extract_language_from_request=lambda *a, **k: None, extract_speaker_from_request=lambda *a, **k: None)


_install_stand_ins()
D = importlib.import_module("sophon_realtime.duplex")
CUT = "……（被用户打断）"


def make():
    plugin = D.SophonRealtimeDuplexPlugin(encode_audio=lambda *a, **k: "")
    buf = plugin.create_session_state().audio_buffer
    return plugin, buf


def reply(plugin, buf, rid, *, text="今天天气不错，适合出门散步", released=True, t_out=None, audio=None):
    info = {"kind": "reply", "buf": buf, "user": None, "a_audio": list(audio or []), "a_text": text, "t_plan": time.monotonic(),
            "spec": not released, "released": released, "dropped": False, "held": [],
            "t_out": time.monotonic() if t_out is None else t_out}
    plugin._reqs[rid] = info
    return info


class FakeClock:
    """Replace the plugin module's clock (time.monotonic) with a manual one."""
    def __init__(self, t0=1000.0):
        self.t = t0; self.real = None

    def __enter__(self):
        self.real = D.time
        D.time = types.SimpleNamespace(monotonic=lambda: self.t, time=self.real.time, sleep=self.real.sleep)
        return self

    def __exit__(self, *exc):
        D.time = self.real


UPPER = "服务端上界"


class ClientCancel(unittest.TestCase):
    """A client response.cancel closes the reply's stream: the reply is ended at once and no longer blocks a queued say."""

    def test_cancel_ends_released_reply(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        self.assertFalse(buf._quiet_out())                       # the reply counts as playing
        plugin.data_plane.close_stream("resp-a")
        self.assertNotIn("resp-a", plugin._reqs)
        self.assertTrue(buf._quiet_out())                        # a queued say may be emitted now
        self.assertEqual(len(buf.history), 1)
        t = buf.history[0]["a_text"]
        self.assertTrue(t.startswith("今天天气不错，适合出门散步"))   # generated text kept whole
        self.assertIn("被取消", t); self.assertIn("听到多少未知", t)    # no audio sent: amount heard unknown

    def test_cancel_marker_gives_server_upper_bound(self):
        plugin, buf = make()
        with FakeClock() as c:
            reply(plugin, buf, "resp-a", t_out=c.t - 0.8, audio=[[0.0] * (2 * D.SR)])   # 2 s generated, sent 0.8 s ago
            plugin.data_plane.close_stream("resp-a")
        t = buf.history[0]["a_text"]
        self.assertTrue(t.startswith("今天天气不错，适合出门散步"))
        self.assertIn("已播出约 800 ms", t); self.assertIn(UPPER, t)
        self.assertNotIn(CUT, t)

    def test_cancel_by_framework_output_id(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        plugin.data_plane.close_stream("resp-a-stage0")           # output ids may carry a suffix (same matching as outputs)
        self.assertNotIn("resp-a", plugin._reqs)
        self.assertTrue(buf._quiet_out())

    def test_cancel_drops_unreleased_draft(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a", released=False)
        plugin.data_plane.close_stream("resp-a")
        self.assertNotIn("resp-a", plugin._reqs)
        self.assertEqual(buf.history, [])

    def test_unknown_or_check_request_untouched(self):
        plugin, buf = make()
        plugin._reqs["chk"] = {"kind": "check", "buf": buf, "perm": [0, 1, 2, 3], "thr": 0.5, "top": {}}
        plugin.data_plane.close_stream("nope")
        plugin.data_plane.close_stream("chk")
        self.assertIn("chk", plugin._reqs)


class UserCut(unittest.TestCase):
    """Barge-in / input cancel: the framework clears input first. The reply is left for the next turn, exactly as before."""

    def test_force_listen_clear_keeps_reply(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        buf.clear_force_listen()                                 # overlap barge-in path
        plugin.data_plane.close_stream("resp-a")
        self.assertIn("resp-a", plugin._reqs)
        self.assertFalse(buf._quiet_out())

    def test_input_clear_keeps_reply(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        buf.clear()                                              # barge_in / input.cancel event path
        plugin.data_plane.close_stream("resp-a")
        self.assertIn("resp-a", plugin._reqs)

    @unittest.skipUnless(HAVE_NUMPY, "needs numpy (plan_append decodes audio)")
    def test_next_turn_ends_it_through_same_path(self):
        plugin, buf = make()
        plugin._render = lambda msgs, tools=None: ""
        reply(plugin, buf, "resp-a")
        buf.clear_force_listen()
        plugin.data_plane.close_stream("resp-a")
        plugin.plan_append(request_id="resp-b", fence=None, session_config=None, runtime_config={}, seq=1, turn_seq=1,
                           payload=dict(_pcm(1.0, 0.1), sophon_kind="reply", sophon_buf=id(buf)), final=True, sampling_params=None)
        self.assertNotIn("resp-a", plugin._reqs)
        self.assertIn("resp-b", plugin._reqs)
        self.assertEqual(len(buf.history), 1)
        self.assertTrue(buf.history[0]["a_text"].endswith(CUT))   # the next-turn path keeps its marker as before

    def test_cut_reply_ended_after_fb_without_voice(self):
        plugin, buf = make()
        with FakeClock() as c:
            reply(plugin, buf, "resp-a", t_out=c.t - 0.5, audio=[[0.0] * D.SR])
            buf.clear_force_listen(); plugin.data_plane.close_stream("resp-a")
            fb = D.FALLBACK_S
            c.t += fb - 0.02
            plugin._retire_closed(id(buf), buf.t_voice, fb)
            self.assertIn("resp-a", plugin._reqs)                # not yet
            c.t += 0.04
            plugin._retire_closed(id(buf), buf.t_voice, fb)
        self.assertNotIn("resp-a", plugin._reqs)
        t = buf.history[0]["a_text"]
        self.assertIn("被打断", t); self.assertIn("已播出约 500 ms", t); self.assertIn(UPPER, t)

    def test_voice_after_cut_restarts_the_wait(self):
        plugin, buf = make()
        with FakeClock() as c:
            reply(plugin, buf, "resp-a")
            buf.clear_force_listen(); plugin.data_plane.close_stream("resp-a")
            fb = D.FALLBACK_S
            c.t += fb - 0.02; buf.t_voice = c.t                  # a voiced frame 20 ms before the wait would end
            c.t += 0.04
            plugin._retire_closed(id(buf), buf.t_voice, fb)
            self.assertIn("resp-a", plugin._reqs)
            c.t += fb
            plugin._retire_closed(id(buf), buf.t_voice, fb)
        self.assertNotIn("resp-a", plugin._reqs)


class Idempotent(unittest.TestCase):
    """Cancel racing the reply's own end: the reply is finished exactly once."""

    def test_cancel_then_finish(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        plugin.data_plane.close_stream("resp-a")
        plugin._finish_reply("resp-a", completed=True)
        plugin.data_plane.close_stream("resp-a")
        self.assertEqual(len(buf.history), 1)

    def test_finish_then_cancel(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        plugin._finish_reply("resp-a", completed=True)
        plugin.data_plane.close_stream("resp-a")
        self.assertEqual(len(buf.history), 1)
        self.assertFalse(buf.history[0]["a_text"].endswith(CUT))

    def test_concurrent(self):
        for i in range(300):
            plugin, buf = make()
            reply(plugin, buf, "resp-a")
            go = threading.Barrier(2)

            def a():
                go.wait(); plugin.data_plane.close_stream("resp-a")

            def b():
                go.wait(); plugin._finish_reply("resp-a", completed=True)
            ts = [threading.Thread(target=a), threading.Thread(target=b)]
            for t in ts: t.start()
            for t in ts: t.join()
            self.assertEqual(len(buf.history), 1, f"iteration {i}")
            self.assertNotIn("resp-a", plugin._reqs)


def _pcm(seconds, level=0.0):
    import numpy as np
    x = np.full(int(seconds * D.SR), level, dtype=np.float32)
    return {"audio": base64.b64encode(x.tobytes()).decode(), "format": "pcm_f32le", "sample_rate_hz": D.SR}


@unittest.skipUnless(HAVE_NUMPY, "needs numpy (audio path)")
class AudioPath(unittest.TestCase):
    def test_say_after_client_cancel_is_emitted(self):
        plugin, buf = make()
        reply(plugin, buf, "resp-a")
        buf.say_q.append("这是一条测试消息。")
        self.assertIsNone(buf.prepare_append(_pcm(0.1), operation_id="op1", chunk_period_ms=100, allow_emit=True))   # waits for the reply
        plugin.data_plane.close_stream("resp-a")
        r = buf.prepare_append(_pcm(0.1), operation_id="op2", chunk_period_ms=100, allow_emit=True)
        self.assertIsNotNone(r)
        self.assertEqual(r.payload.get("say"), "这是一条测试消息。")
        self.assertEqual(buf.say_q, [])

    def test_say_still_waits_without_cleanup(self):          # reference: the stale reply blocks the say when the stream close is ignored
        plugin, buf = make()
        plugin.data_plane.close_stream = lambda rid: None
        reply(plugin, buf, "resp-a")
        buf.say_q.append("这是一条测试消息。")
        plugin.data_plane.close_stream("resp-a")
        for i in range(3):
            self.assertIsNone(buf.prepare_append(_pcm(0.1), operation_id=f"op{i}", chunk_period_ms=100, allow_emit=True))
        self.assertEqual(buf.say_q, ["这是一条测试消息。"])

    def _barge_then_next_turn(self, ignore_close):
        import numpy as np
        plugin, buf = make()
        plugin._render = lambda msgs, tools=None: ""
        if ignore_close:
            plugin.data_plane.close_stream = lambda rid: None
        reply(plugin, buf, "resp-a", text="一二三四五六七八九十", t_out=time.monotonic() - 1.0, audio=[np.zeros(2 * D.SR, np.float32)])
        buf.say_q.append("排队的消息")
        buf.clear_force_listen(); plugin.data_plane.close_stream("resp-a")        # framework order on an overlap barge-in
        for i in range(3):
            buf.prepare_append(_pcm(0.1, 0.1), operation_id=f"op{i}", chunk_period_ms=100, allow_emit=True)
        snap_mid = (sorted(plugin._reqs), buf._quiet_out(), list(buf.say_q))
        plugin.plan_append(request_id="resp-b", fence=None, session_config=None, runtime_config={}, seq=1, turn_seq=1,
                           payload=dict(_pcm(1.0, 0.1), sophon_kind="reply", sophon_buf=id(buf)), final=True, sampling_params=None)
        return (snap_mid, sorted(plugin._reqs), [h["a_text"] for h in buf.history], (buf.cut or {}).get("said"),
                (buf.cut or {}).get("rid"), buf.last_said, list(buf.say_q))

    def test_server_vad_cut_then_new_turn_unchanged(self):
        new = self._barge_then_next_turn(ignore_close=False)
        old = self._barge_then_next_turn(ignore_close=True)
        self.assertEqual(new, old)
        self.assertEqual(new[1], ["resp-b"])                     # the cut reply was ended by the new turn, once
        self.assertEqual(len(new[2]), 1)

    def test_say_not_held_after_vad_cut_and_short_sound(self):
        """Server VAD cuts the reply, the user only makes a short sound that never becomes a turn (under the 100 ms onset),
        then stays silent: no new reply is planned. The queued say must still go out within FB."""
        plugin, buf = make()
        clock = [time.monotonic()]; t0 = clock[0]
        real_time = D.time
        D.time = types.SimpleNamespace(monotonic=lambda: clock[0], time=real_time.time, sleep=real_time.sleep)
        try:
            reply(plugin, buf, "resp-a", t_out=clock[0] - 0.5)
            buf.say_q.append("排队的消息")
            buf.clear_force_listen(); plugin.data_plane.close_stream("resp-a")        # framework order on an overlap barge-in
            buf.prepare_append(_pcm(0.06, 0.1), operation_id="op0", chunk_period_ms=60, allow_emit=True)   # 60 ms sound
            got = None
            for i in range(50):                                                      # 5 s of silence
                clock[0] += 0.1
                got = buf.prepare_append(_pcm(0.1), operation_id=f"op{i + 1}", chunk_period_ms=100, allow_emit=True) or got
                if got is not None: break
            self.assertEqual(buf.speech_ms, 0)                                       # the sound never became a turn
            self.assertIsNotNone(got)
            self.assertEqual(got.payload.get("say"), "排队的消息")
            self.assertLessEqual(clock[0] - t0, D.FALLBACK_S + 0.2)
            self.assertIn("听到多少未知", buf.history[0]["a_text"])
        finally:
            D.time = real_time


def _feed(buf, clock, seconds, level, step=0.1, check=None):
    """Push audio in 100 ms chunks, advancing the fake clock; stop at and return the first reservation that carries a say."""
    got = None
    for _ in range(int(round(seconds / step))):
        clock.t += step
        r = buf.prepare_append(_pcm(step, level), operation_id=f"op{clock.t:.2f}", chunk_period_ms=int(step * 1000), allow_emit=True)
        if check is not None: check()
        if r is not None and isinstance(getattr(r, "payload", None), dict) and r.payload.get("say") is not None:
            got = r; break
    return got


@unittest.skipUnless(HAVE_NUMPY, "needs numpy (audio path)")
class BoundedWait(unittest.TestCase):
    def setUp(self):
        self.saved = (D.SPEC_AT, D.IGNORE_SHORT)
        D.SPEC_AT = 0.0                                            # no speculative drafts: they would hold reservations and hide the conditions

    def tearDown(self):
        D.SPEC_AT, D.IGNORE_SHORT = self.saved

    def _session(self, cut):
        plugin, buf = make()
        buf._start_check = lambda thr, idx: None                   # no turn-question requests in unit tests
        buf.say_q.append("排队的消息")
        if cut:
            import numpy as np
            reply(plugin, buf, "resp-a", t_out=time.monotonic() - 0.5, audio=[np.zeros(D.SR, np.float32)])
            buf.clear_force_listen(); plugin.data_plane.close_stream("resp-a")
        return plugin, buf

    def _speech_100(self, cut):
        with FakeClock(time.monotonic()) as c:
            plugin, buf = self._session(cut)
            _feed(buf, c, 0.1, 0.0)
            self.assertIsNone(_feed(buf, c, 0.1, 0.1))            # exactly 5 voiced frames: onset with speech_ms = 100
            self.assertEqual(buf.speech_ms, 100)
            t_last = buf.t_voice
            got = _feed(buf, c, D.FALLBACK_S - 0.2, 0.0)
            self.assertIsNone(got); self.assertEqual(buf.speech_ms, 100)    # before FB: still waiting
            got = _feed(buf, c, 0.5, 0.0)
            self.assertIsNotNone(got); self.assertEqual(buf.speech_ms, 0)
            self.assertLessEqual(c.t - t_last, D.FALLBACK_S + 0.25)
        return plugin, buf

    def test_speech_ms_100_without_cancel(self):
        self._speech_100(cut=False)

    def test_speech_ms_100_after_vad_cut(self):
        plugin, buf = self._speech_100(cut=True)
        self.assertNotIn("resp-a", plugin._reqs)
        self.assertIn(UPPER, buf.history[0]["a_text"]); self.assertIn("被打断", buf.history[0]["a_text"])

    def _long_sentence(self, fb_s, pause1, pause2):
        with FakeClock(time.monotonic()) as c:
            plugin, buf = self._session(cut=True)
            buf.fb_s = fb_s

            def still():
                self.assertIn("resp-a", plugin._reqs)
                self.assertEqual(buf.say_q, ["排队的消息"])
            for sec, lvl in ((1.0, 0.1), (pause1, 0.0), (1.0, 0.1), (pause2, 0.0)):
                self.assertIsNone(_feed(buf, c, sec, lvl, check=still))
                if lvl: self.assertGreaterEqual(buf.speech_ms, 120)
            self.assertGreaterEqual(buf.speech_ms, 120)               # the turn is intact

    def test_long_sentence_with_pauses_not_cleared(self):
        self._long_sentence(None, 1.5, 1.0)

    def test_long_sentence_session_fb(self):
        self._long_sentence(1.2, 1.0, 1.0)

    def test_ignore_short_then_say(self):
        D.IGNORE_SHORT = True
        with FakeClock(time.monotonic()) as c:
            plugin, buf = self._session(cut=True)
            _feed(buf, c, 0.2, 0.1)                                  # 200 ms sound: becomes a short turn
            self.assertGreater(buf.speech_ms, 0)
            t_last = buf.t_voice
            got = _feed(buf, c, D.FALLBACK_S + 0.5, 0.0)
            self.assertIsNotNone(got); self.assertEqual(buf.speech_ms, 0)
            self.assertLessEqual(c.t - t_last, D.FALLBACK_S + 0.5)
            self.assertNotIn("resp-a", plugin._reqs)

    def test_speech_20ms_before_fb_keeps_reply(self):
        with FakeClock(time.monotonic()) as c:
            plugin, buf = self._session(cut=True)
            _feed(buf, c, D.FALLBACK_S - 0.1, 0.0)                    # silence up to FB - 100 ms after the cut
            c.t += 0.08                                              # the user starts talking 20 ms before FB
            self.assertIsNone(buf.prepare_append(_pcm(0.02, 0.1), operation_id="near", chunk_period_ms=20, allow_emit=True))
            self.assertIn("resp-a", plugin._reqs)
            t_voice = buf.t_voice
            self.assertGreater(t_voice, 0.0)
            _feed(buf, c, 1.0, 0.1, check=lambda: self.assertIn("resp-a", plugin._reqs))   # the user keeps talking
            self.assertGreater(buf.speech_ms, 0)
            self.assertEqual(buf.say_q, ["排队的消息"])

    def test_blip_20ms_before_fb_restarts_wait(self):
        with FakeClock(time.monotonic()) as c:
            plugin, buf = self._session(cut=True)
            _feed(buf, c, D.FALLBACK_S - 0.1, 0.0)
            c.t += 0.08                                              # 20 ms before FB
            buf.prepare_append(_pcm(0.02, 0.1), operation_id="near", chunk_period_ms=20, allow_emit=True)   # 20 ms blip, no onset
            t_blip = buf.t_voice
            self.assertIsNone(_feed(buf, c, D.FALLBACK_S - 0.2, 0.0, check=lambda: self.assertIn("resp-a", plugin._reqs)))
            got = _feed(buf, c, 0.5, 0.0)
            self.assertIsNotNone(got)
            self.assertLessEqual(c.t - t_blip, D.FALLBACK_S + 0.25)


if __name__ == "__main__":
    unittest.main()
