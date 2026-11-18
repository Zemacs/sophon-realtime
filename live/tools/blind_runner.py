"""Blind-test runner for Critic's I-10 routing test (one self-contained file; touches no production code).

It feeds sentences, one per run, through the Coordinator (``coordinator.admit``: the JEV route decision, the guard, the heuristic
fallback) and, for a job, through the Worker, and RECORDS what the system did. It never scores, never compares to an expected
route, and contains no word list or regex about meaning.

Usage (from ``live/``)::

    python -m tools.blind_runner SENTENCES.txt OUT.jsonl                       # fake mode (default): nothing leaves the process
    python -m tools.blind_runner SENTENCES.jsonl OUT.jsonl --fake-route FAST   # a different constant fake JEV answer
    BLIND_RUNNER_ALLOW_REAL=1 python -m tools.blind_runner S.txt OUT.jsonl --mode real --jev-url http://HOST:8110 --orch-url http://HOST:8101/v1/chat/completions

Input: a text file (one sentence per line, blank lines skipped; ids ``s0001``...) or JSONL ``{"id": "<opaque>", "text": "..."}`` per line
(a line starting with ``{`` is JSON). Output, one JSON object per sentence, written ONLY to OUT.jsonl::

    {"id", "route"  (the Coordinator's route: WORKER|FAST|CLARIFY|CONTROL|STOP_SPEAKING),
     "decision" (NEW / REFINE / CANCEL / STATUS / the route), "jev_route", "jev_control" (what the JEV answered, even if a guard overrode it),
     "via" (jev | guard | heuristic), "degraded" (JEV was expected but the heuristic decided), "handled_by" (fast | clarify | control | worker | stop),
     "tools" (names of the tools the Worker called), "tool_ok" (their ok flags), "worker_status", "reply", "error" (exception class name only)}

Modes
  fake (default)  a constant fake JEV (``--fake-route``, default WORKER) and a fake Worker that "calls" the tools the Coordinator required
                  and answers with a fixed string. No network, no model. This is what the tests use.
  real            the real JEV (``--jev-url``) and the real Worker/orchestrator (``--orch-url``/``--orch-model``). Refused unless the
                  environment has ``BLIND_RUNNER_ALLOW_REAL=1``. Real runs are for AFTER the server freeze is lifted; do not run them before.
  gateway URL     NOT available: the GPT-Live gateway and the Realtime face take speech only (text items are ignored), so there is no text
                  turn interface to drive. ``--mode gateway`` exits with code 3 and says so.

Privacy (what the runner guarantees, and the test that pins it: ``tests/test_blind_runner_privacy.py``)
  * sentence text is written to no place except the reply field of OUT.jsonl (the sentence itself is not even echoed there);
  * all logging is switched off for the run (``logging.disable``), warnings are ignored, and the runner prints only counts;
  * the memory store is pointed at the null device, the device hook is unset, and errors are recorded as a class name only;
  * REAL mode necessarily sends sentences to the JEV / orchestrator URLs and to the tools those use (search, weather, news): that is the
    system under test's own traffic, not the runner's; point it only at servers you control.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import warnings
from typing import Any, Iterable

ALLOW_REAL_ENV = "BLIND_RUNNER_ALLOW_REAL"
FAKE_REPLY = "[fake worker reply]"
_HOOK_ENV_NAMES = ("SOPHON_LIVE_DEVICE_HOOK",)


def read_sentences(path: str) -> list[tuple[str, str]]:
    """[(id, text)]. A line that starts with ``{`` is JSON ({"id", "text"}); anything else is a plain sentence."""
    out: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    o = json.loads(line)
                    sid, text = str(o.get("id") if o.get("id") is not None else f"s{n:04d}"), str(o["text"])
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise SystemExit(f"input line {n}: expected JSON with a 'text' field")   # never echo the line
            else:
                sid, text = f"s{n:04d}", line
            out.append((sid, text))
    ids = [s for s, _ in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("input ids are not unique")
    return out


@contextlib.contextmanager
def quiet_and_sealed(real: bool, orch_url: str | None = None, orch_model: str | None = None):
    """No logging, no warnings, no memory writes, no device hook, for the duration of the run (everything restored after)."""
    saved_env = {k: os.environ.get(k) for k in ("SOPHON_LIVE_MEMORY_PATH", *_HOOK_ENV_NAMES, "SOPHON_LIVE_ORCH_URL", "SOPHON_LIVE_ORCH_MODEL")}
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["SOPHON_LIVE_MEMORY_PATH"] = os.devnull
            for k in _HOOK_ENV_NAMES:
                os.environ.pop(k, None)
            if orch_url:
                os.environ["SOPHON_LIVE_ORCH_URL"] = orch_url
            if orch_model:
                os.environ["SOPHON_LIVE_ORCH_MODEL"] = orch_model
            yield
    finally:
        logging.disable(prev_disable)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Recorder:
    """Remembers what the JEV answered for the current sentence (the Coordinator may then override it)."""
    def __init__(self) -> None:
        self.route: str | None = None
        self.control: str | None = None


async def _fake_jev(route: str):
    async def system_one(state, questions, *, url=None, model=None):   # noqa: ANN001, ARG001
        return {"answers": {"route": {"type": "choice", "choice": route}}}
    return system_one


async def _fake_worker(job, *, instructions="", history=None):   # noqa: ANN001, ARG001
    from sophon_live.coordinator.schemas import WorkerResult
    ev = [{"tool": t, "ok": True} for t in (job.require_tools or [])]
    return WorkerResult(task_id=job.task_id, version=job.version, status="completed", spoken_answer=FAKE_REPLY,
                        evidence=ev, freshness_token=job.freshness_token)


async def run_one(sid: str, text: str, *, ledger, jev, worker, language: str = "zh") -> dict[str, Any]:
    from sophon_live import coordinator as C
    from sophon_live.coordinator import api, jev_client
    rec = _Recorder()
    row: dict[str, Any] = {"id": sid}

    async def spy(state, questions, **kw):   # noqa: ANN001
        resp = await jev(state, questions, **kw)
        rec.route = jev_client.choice_answer(resp, "route")
        rec.control = jev_client.choice_answer(resp, "control")
        return resp

    orig = jev_client.system_one
    jev_client.system_one = spy
    try:
        res = await C.admit(C.AdmitRequest(session_id="blind", utterance_id=sid, text=text, delegate_task=text, language=language,
                                           raw_text=text), ledger)
    finally:
        jev_client.system_one = orig
    decision = res.control.op if res.control is not None else ("NEW" if res.job is not None else res.route)
    row.update(route=res.route, decision=decision, jev_route=rec.route, jev_control=rec.control, via=res.via,
               degraded=bool(api.USE_JEV and res.via == "heuristic"), handled_by=None, tools=[], tool_ok=[], worker_status=None, reply="")
    if res.job is None:
        row["handled_by"] = {"STOP_SPEAKING": "stop", "FAST": "fast", "CLARIFY": "clarify"}.get(res.route, "control" if res.control else res.route.lower())
        row["reply"] = res.speak.text or ""
        return row
    job = res.job
    row["handled_by"] = "worker"
    with C.bind_task(C.TaskContext(task_id=job.task_id, version=job.version, utterance=job.objective, ledger=ledger)):
        wr = await worker(job, instructions="", history=[])
    row["worker_status"] = wr.status
    row["tools"] = [str(e.get("tool")) for e in wr.evidence]
    row["tool_ok"] = [e.get("ok") for e in wr.evidence]
    out = C.finish_job(ledger, job, wr)
    row["reply"] = out.text
    return row


async def run(sentences: Iterable[tuple[str, str]], out_path: str, *, mode: str = "fake", fake_route: str = "WORKER",
              jev_url: str | None = None, shared_ledger: bool = False) -> int:
    from sophon_live import coordinator as C
    from sophon_live.coordinator import api, jev_client
    if mode == "fake":
        jev = await _fake_jev(fake_route)
        worker = _fake_worker
        use_jev = True
    else:
        real_system_one = jev_client.system_one

        async def jev(state, questions, **kw):   # noqa: ANN001
            return await real_system_one(state, questions, url=jev_url, **kw)
        from sophon_live.assistant_tools import run_worker_job as worker
        use_jev = api.USE_JEV
    n = 0
    ledger = C.Ledger()
    prev_use = api.USE_JEV
    api.USE_JEV = use_jev
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for sid, text in sentences:
                if not shared_ledger:
                    ledger = C.Ledger()
                try:
                    row = await run_one(sid, text, ledger=ledger, jev=jev, worker=worker)
                except Exception as e:   # noqa: BLE001   class name only: a message may quote the sentence
                    row = {"id": sid, "error": type(e).__name__}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
    finally:
        api.USE_JEV = prev_use
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="blind_runner", description="Record what the system does with each sentence (no scoring).")
    ap.add_argument("sentences"); ap.add_argument("out")
    ap.add_argument("--mode", choices=("fake", "real", "gateway"), default="fake")
    ap.add_argument("--fake-route", default="WORKER", help="fake mode: the constant route the fake JEV answers")
    ap.add_argument("--jev-url"); ap.add_argument("--orch-url"); ap.add_argument("--orch-model")
    ap.add_argument("--shared-ledger", action="store_true", help="keep one ledger across the sentences (they then see each other as earlier turns)")
    a = ap.parse_args(argv)
    if a.mode == "gateway":
        print("blind_runner: no text-turn interface exists on the gateway or the Realtime face (speech only); use fake or real mode", file=sys.stderr)
        return 3
    if a.mode == "real":
        if os.environ.get(ALLOW_REAL_ENV) != "1":
            print(f"blind_runner: real mode needs {ALLOW_REAL_ENV}=1 (only after the server freeze is lifted)", file=sys.stderr)
            return 4
        if not a.jev_url:
            print("blind_runner: real mode needs --jev-url", file=sys.stderr)
            return 4
    sents = read_sentences(a.sentences)
    with quiet_and_sealed(a.mode == "real", a.orch_url, a.orch_model):
        n = asyncio.run(run(sents, a.out, mode=a.mode, fake_route=a.fake_route, jev_url=a.jev_url, shared_ledger=a.shared_ledger))
    print(f"blind_runner: {n} sentences -> output file ({a.mode} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
