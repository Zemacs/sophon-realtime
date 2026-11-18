"""System One client for Karpo / TypeSafe-compatible decision servers.

Stage 0 (JEV reliability): typed failure reasons, a connection-only retry, a circuit breaker probed through ``GET /healthz``,
and counters. Nothing here judges meaning: when JEV cannot answer, the *caller* decides how to degrade (``api.py``), and that
decision never reads the user's words.

Defaults: 0.3 s to connect, 1.5 s for the whole call. Measured read-only on the real Karpo (2026-11-19, ``s0/jev_conc_probe.py``,
84 requests): the server decides one request at a time, so latency grows linearly with simultaneous callers (1 question ~154 ms
each, 2 questions ~262 ms each; throughput flat at ~6.5 / ~3.8 per second). 1.5 s therefore holds up to ~9 simultaneous
1-question calls or ~5 simultaneous 2-question calls; at 8 simultaneous 2-question calls the slowest took 2.1 s and would be
degraded. That is fine for the current 1-3 sessions; raise ``SOPHON_JEV_TIMEOUT`` (3.0 s covers ~11) if more are expected.
Retries happen only on connection errors (refused / reset), never on a timeout (a timed-out call may still be queued
server-side; retrying would double the load on the one thing that is slow).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

import aiohttp

log = logging.getLogger("sophon_live.coordinator.jev")

DEFAULT_URL = os.environ.get("SOPHON_JEV_URL", "http://127.0.0.1:8110/v1/systemone")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class Config:
    """Read at call time (tests patch the attributes)."""
    timeout = _f("SOPHON_JEV_TIMEOUT", 1.5)                 # total budget of one call, retry included
    connect_timeout = _f("SOPHON_JEV_CONNECT_TIMEOUT", 0.3)
    probe_interval = _f("SOPHON_JEV_PROBE_INTERVAL", 15.0)  # background /healthz cadence (gateway)
    startup_wait = _f("SOPHON_JEV_STARTUP_WAIT", 30.0)      # bounded wait for /healthz at gateway start
    connect_retries = int(_f("SOPHON_JEV_CONNECT_RETRIES", 1))   # connection errors only; never timeouts
    retry_delay = _f("SOPHON_JEV_RETRY_DELAY", 0.1)
    breaker_fails = int(_f("SOPHON_JEV_BREAKER_FAILS", 3))       # consecutive failed calls that open the breaker
    breaker_open_s = _f("SOPHON_JEV_BREAKER_OPEN_S", 10.0)
    probe_timeout = _f("SOPHON_JEV_PROBE_TIMEOUT", 1.0)


CFG = Config
DEFAULT_TIMEOUT = CFG.timeout          # kept for older imports; the live value is CFG.timeout

_clock: Callable[[], float] = time.monotonic   # tests replace this

# Typed failure reasons (the only values ``JevError.reason`` / the counters use).
TIMEOUT, CONNECT_TIMEOUT, REFUSED, HTTP_5XX, HTTP_4XX, BAD_JSON, NOT_CHOICE, BREAKER_OPEN, ERROR = (
    "timeout", "connect_timeout", "refused", "http_5xx", "http_4xx", "bad_json", "not_choice", "breaker_open", "error")
# Reasons that say "the service is not usable right now" (they count towards the breaker).
_BREAKER_REASONS = {TIMEOUT, CONNECT_TIMEOUT, REFUSED, HTTP_5XX, BAD_JSON, ERROR}


_CONNECT_TIMEOUT_EXC = getattr(aiohttp, "ConnectionTimeoutError", ())   # older aiohttp: no such class


class JevError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"jev {reason}: {detail}"[:240])
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------------------------- counters
class Stats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ok = 0
        self.fail: dict[str, int] = {}
        self.degraded: dict[str, int] = {}
        self.answers: dict[str, int] = {}      # "<question>:<type>" -> n

    @property
    def n_fail(self) -> int:
        return sum(self.fail.values())

    @property
    def n_degraded(self) -> int:
        return sum(self.degraded.values())

    def line(self) -> str:
        return f"jev_ok={self.ok} jev_fail={self.n_fail} jev_degraded={self.n_degraded}"

    def snapshot(self) -> dict[str, Any]:
        return {"jev_ok": self.ok, "jev_fail": dict(self.fail), "jev_degraded": dict(self.degraded), "answers": dict(self.answers)}


stats = Stats()


def note_ok() -> None:
    stats.ok += 1


def note_fail(reason: str) -> None:
    stats.fail[reason] = stats.fail.get(reason, 0) + 1


def note_degraded(reason: str) -> None:
    stats.degraded[reason] = stats.degraded.get(reason, 0) + 1


def answer_type(resp: dict[str, Any], key: str) -> str:
    """The type JEV gave for one question: ``choice`` / ``noul`` / ... or ``missing``."""
    ans = (resp.get("answers") or {}).get(key)
    if not isinstance(ans, dict):
        return "missing"
    return str(ans.get("type") or "missing")


def confidence(resp: dict[str, Any], key: str) -> float | None:
    """The model's own uncertainty signal for one answer, as given (``answers.<q>.confidence``). Observability only: no
    routing decision reads it, and nothing in this package builds word lists around it."""
    ans = (resp.get("answers") or {}).get(key)
    v = ans.get("confidence") if isinstance(ans, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def record_answers(resp: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    """Count (and return) the answer type per asked question; one INFO line carries type (and the model's confidence)."""
    out = {k: answer_type(resp, k) for k in keys}
    for k, t in out.items():
        stats.answers[f"{k}:{t}"] = stats.answers.get(f"{k}:{t}", 0) + 1
    log.info("jev answers %s", " ".join(
        f"{k}={t}" + ("" if confidence(resp, k) is None else f"(conf={confidence(resp, k):.2f})") for k, t in out.items()))
    return out


# --------------------------------------------------------------------------------------------- breaker
class Breaker:
    """closed -> (N consecutive failed calls) -> open -> (after open_s, ``/healthz`` ok) -> half_open -> one trial call
    -> closed on success / open again on failure. While open (or while the single probe / trial is in flight) callers get
    ``JevError(breaker_open)`` at once, with no network I/O."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = "closed"
        self.consecutive = 0
        self.opened_at = 0.0
        self.busy = False            # a probe or the half-open trial is in flight

    async def before(self, endpoint: str) -> None:
        if self.state == "closed":
            return
        if self.busy:
            raise JevError(BREAKER_OPEN, f"{self.state}, probe/trial in flight")
        if self.state == "open":
            if _clock() - self.opened_at < CFG.breaker_open_s:
                raise JevError(BREAKER_OPEN, "open")
            self.busy = True
            try:
                ok, ms, detail = await probe(endpoint)
            except BaseException:
                self.busy = False
                raise
            if not ok:
                self.busy = False
                self.opened_at = _clock()
                log.warning("jev breaker: probe failed (%s); staying open", detail)
                raise JevError(BREAKER_OPEN, f"probe failed: {detail}")
            self.state = "half_open"          # busy stays True: this call is the one trial
            log.info("jev breaker: /healthz ok (%.0f ms); half-open, trying one call", ms)
            return
        # half_open and nobody in flight: this call becomes the trial
        self.busy = True

    def success(self) -> None:
        if self.state != "closed":
            log.info("jev breaker: closed again")
        self.reset()

    def failure(self, reason: str) -> None:
        self.busy = False
        if self.state == "half_open":
            self.state, self.opened_at = "open", _clock()
            log.warning("jev breaker: trial failed (%s); open for %.0fs", reason, CFG.breaker_open_s)
            return
        self.consecutive += 1
        if self.state == "closed" and self.consecutive >= CFG.breaker_fails:
            self.state, self.opened_at = "open", _clock()
            log.warning("jev breaker: OPEN after %d consecutive failures (last: %s); no JEV calls for %.0fs",
                        self.consecutive, reason, CFG.breaker_open_s)

    def release(self) -> None:
        """The call ended without saying anything about availability (e.g. a 4xx or a cancelled await)."""
        if self.state == "half_open":
            self.busy = False


breaker = Breaker()


# --------------------------------------------------------------------------------------------- HTTP
def _endpoint(url: str | None) -> str:
    endpoint = (url or DEFAULT_URL).rstrip("/")
    if not endpoint.endswith("/v1/systemone"):
        endpoint = endpoint + "/v1/systemone"
    return endpoint


def healthz_url(endpoint: str) -> str:
    base = endpoint[: -len("/v1/systemone")] if endpoint.endswith("/v1/systemone") else endpoint
    return base.rstrip("/") + "/healthz"


async def probe(url: str | None = None, timeout: float | None = None) -> tuple[bool, float, str]:
    """``GET /healthz`` on the JEV server. Returns (ok, elapsed_ms, detail). Never raises."""
    hz = healthz_url(_endpoint(url))
    t0 = _clock()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout or CFG.probe_timeout, sock_connect=CFG.connect_timeout)) as sess:
            async with sess.get(hz) as resp:
                text = await resp.text()
                ms = (_clock() - t0) * 1000
                if resp.status != 200:
                    return False, ms, f"HTTP {resp.status}"
                try:
                    data = await resp.json(content_type=None)
                except Exception:  # noqa: BLE001
                    return False, ms, "healthz: bad json"
                if not (isinstance(data, dict) and data.get("ok") is True):
                    return False, ms, f"healthz not ok: {text[:80]}"
                return True, ms, "ok"
    except asyncio.TimeoutError:
        return False, (_clock() - t0) * 1000, "timeout"
    except aiohttp.ClientConnectionError as e:
        return False, (_clock() - t0) * 1000, f"refused/reset: {type(e).__name__}"
    except Exception as e:  # noqa: BLE001
        return False, (_clock() - t0) * 1000, f"{type(e).__name__}: {e}"[:120]


async def _post_once(endpoint: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout, sock_connect=min(CFG.connect_timeout, timeout))) as sess:
            async with sess.post(endpoint, json=body) as resp:
                text = await resp.text()
                if resp.status >= 500:
                    log.warning("jev HTTP %s: %s", resp.status, text[:300])
                    raise JevError(HTTP_5XX, f"HTTP {resp.status}: {text[:160]}")
                if resp.status >= 400:
                    log.warning("jev HTTP %s: %s", resp.status, text[:300])
                    raise JevError(HTTP_4XX, f"HTTP {resp.status}: {text[:160]}")
                try:
                    data = await resp.json(content_type=None)
                except Exception as e:  # noqa: BLE001
                    raise JevError(BAD_JSON, text[:160]) from e
                if not isinstance(data, dict):
                    raise JevError(BAD_JSON, f"not an object: {text[:120]}")
                return data
    except JevError:
        raise
    except _CONNECT_TIMEOUT_EXC as e:        # could not even connect within connect_timeout (not retried)
        raise JevError(CONNECT_TIMEOUT, f"no connection within {CFG.connect_timeout:.1f}s") from e
    except asyncio.TimeoutError as e:
        raise JevError(TIMEOUT, f"no answer within {timeout:.1f}s") from e
    except aiohttp.ClientConnectionError as e:        # refused, reset, server closed the connection
        raise JevError(REFUSED, f"{type(e).__name__}: {e}"[:160]) from e
    except Exception as e:  # noqa: BLE001
        raise JevError(ERROR, f"{type(e).__name__}: {e}"[:160]) from e


async def system_one(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """POST /v1/systemone; return parsed JSON (answers map). Raises ``JevError`` (a ``RuntimeError``) with a typed reason."""
    endpoint = _endpoint(url)
    body: dict[str, Any] = {"state": state, "questions": questions}
    if model:
        body["model"] = model
    try:
        await breaker.before(endpoint)
    except JevError as e:
        note_fail(e.reason)
        raise
    deadline = _clock() + CFG.timeout
    attempt = 0
    try:
        while True:
            remaining = max(0.05, deadline - _clock())
            try:
                data = await _post_once(endpoint, body, remaining)
                breaker.success()
                return data
            except JevError as e:
                if e.reason == REFUSED and attempt < CFG.connect_retries and deadline - _clock() > CFG.retry_delay + 0.05:
                    attempt += 1
                    log.info("jev connection error, retry %d/%d: %s", attempt, CFG.connect_retries, e.detail)
                    await asyncio.sleep(CFG.retry_delay)
                    continue
                raise
    except JevError as e:
        note_fail(e.reason)
        if e.reason in _BREAKER_REASONS:
            breaker.failure(e.reason)
        else:
            breaker.release()
        raise
    except BaseException:       # cancelled: say nothing about availability
        breaker.release()
        raise


_PING = {"state": {"utterance": "ping"},
         "questions": {"route": {"type": "choice", "instructions": "health check", "criteria": {"A": "a", "B": "b"}}}}


async def deep_probe(url: str | None = None) -> tuple[bool, str]:
    """``/healthz`` and then one tiny real System One request (a loaded-but-broken model passes /healthz, not this)."""
    endpoint = _endpoint(url)
    ok, ms, detail = await probe(endpoint)
    if not ok:
        return False, f"healthz: {detail}"
    try:
        data = await _post_once(endpoint, _PING, CFG.timeout)
    except JevError as e:
        return False, f"ping: {e.reason}: {e.detail}"[:160]
    return (answer_type(data, "route") == "choice"), f"healthz {ms:.0f} ms, ping route={answer_type(data, 'route')}"


async def wait_ready(url: str | None = None, *, max_wait: float | None = None, interval: float = 1.0,
                     sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> tuple[bool, float, str]:
    """Bounded wait for ``/healthz`` (gateway start). Returns (ready, waited_s, last_detail); never raises, never waits
    longer than ``max_wait`` (default ``CFG.startup_wait``)."""
    limit = CFG.startup_wait if max_wait is None else max_wait
    t0 = _clock()
    detail = "not tried"
    while True:
        ok, _ms, detail = await probe(url)
        waited = _clock() - t0
        if ok:
            return True, waited, detail
        if waited + interval > limit:
            return False, waited, detail
        await sleep(interval)


async def probe_loop(url: str | None = None, *, interval: float | None = None, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
                     rounds: int | None = None) -> None:
    """Background prober (gateway). Closed breaker: ``/healthz`` only, a failure counts like a failed call so the breaker
    opens before a user turn pays for it. Open / half-open: after the open window, ``/healthz`` plus one tiny real request;
    success closes the breaker, failure keeps it open. ``rounds`` is for tests."""
    endpoint = _endpoint(url)
    n = 0
    while rounds is None or n < rounds:
        n += 1
        await sleep(CFG.probe_interval if interval is None else interval)
        try:
            if breaker.busy:
                continue
            if breaker.state == "closed":
                ok, _ms, detail = await probe(endpoint)
                if ok:
                    breaker.consecutive = 0
                else:
                    log.warning("jev probe: /healthz failed (%s)", detail)
                    breaker.failure("probe")
            elif _clock() - breaker.opened_at >= CFG.breaker_open_s:
                breaker.busy = True
                try:
                    ok, detail = await deep_probe(endpoint)
                finally:
                    breaker.busy = False
                if ok:
                    breaker.success()
                else:
                    breaker.opened_at = _clock()
                    breaker.state = "open"
                    log.warning("jev probe: still down (%s)", detail)
        except Exception as e:  # noqa: BLE001  (a prober must never die)
            log.warning("jev probe loop error: %r", e)


def choice_answer(resp: dict[str, Any], key: str) -> str | None:
    ans = (resp.get("answers") or {}).get(key) or {}
    if ans.get("type") == "choice":
        return ans.get("choice")
    return None


def noul_prob(resp: dict[str, Any], key: str) -> float | None:
    ans = (resp.get("answers") or {}).get(key) or {}
    if ans.get("type") == "noul":
        v = ans.get("noul")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------------------------- secondary hook
# A second decision model with the same System One contract (e.g. an emulator in front of another LLM). NOT configured here:
# it stays ``None`` unless code registers one, so no new paid GPU / API is ever used by default.
SecondaryFn = Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]
SECONDARY: SecondaryFn | None = None


def register_secondary(fn: SecondaryFn | None) -> None:
    global SECONDARY
    SECONDARY = fn
