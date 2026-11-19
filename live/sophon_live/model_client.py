"""Shared client discipline for the model services Sophon depends on (v0.1.0).

Every downstream model (Tag front ``tag``, orchestrator ``orch``, decision model ``jev``) goes through here so that a
dead or slow service is handled the same way: a bounded time budget, retries only for *connection* errors (never for
timeouts: a slow server is not helped by more load), a circuit breaker that turns repeated failures into an instant
failure, and a snapshot the gateway publishes on ``/health?detail=1``.

There is deliberately no semantic fallback in here. What the caller does when a service is down must be model-free and
safe: a fixed sentence, an honest failure, or handing the request to the next stage as-is.

Budgets (override with ``SOPHON_LIVE_TIMEOUT_<NAME>`` seconds; evidence for the defaults):
  jev   1.5 s  Karpo Decision 31B answers the route question in p50 103 ms / p99 514 ms (P1-G01 probe, server run 11-18);
                the old 8 s only added dead air when the service was down.
  orch  120 s  whole tool-backed job (<= 6 LLM steps + tools). Unchanged from before this round: the measured Worker p50 0.66 s / p95 2.23 s
                is a relayed sample and says nothing about the TAIL or about truncating a legitimately long job, so the final value is to be
                set from the latency / truncation distribution recorded in the unfreeze window (override: ``SOPHON_LIVE_ORCH_TIMEOUT``, the one the
                orchestrator reads; ``SOPHON_LIVE_TIMEOUT_ORCH`` only changes the value ``/health?detail=1`` reports).
  tag   15 s   websocket open + session.created; unchanged (cold vLLM-Omni session setup is a few seconds).
Breaker (``SOPHON_LIVE_BREAKER_<NAME>_N`` consecutive failures, ``..._OPEN_S`` seconds): default N=3, open 15 s, then one
half-open probe call decides.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Awaitable, Callable

log = logging.getLogger("sophon_live.model_client")

DEFAULT_BUDGET_S = {"jev": 1.5, "orch": 120.0, "tag": 15.0}
DEFAULT_BREAKER_N = 3
DEFAULT_BREAKER_OPEN_S = 15.0
RETRY_BASE_S = 0.1


class HttpStatusError(RuntimeError):
    """A downstream answered with an HTTP error status. Carries ONLY the status code: the response body is never kept in the exception, so it cannot reach
    ``last_error`` / ``/health`` / a ModelUnavailable message (R-A' / Critic I-5)."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


def error_label(e: BaseException) -> str:
    """What may be remembered / shown about a failure: the exception CLASS, or ``http <status>``. Never the message (it can echo a downstream body)."""
    return f"http {e.status}" if isinstance(e, HttpStatusError) else type(e).__name__


class ModelUnavailable(Exception):
    """The service cannot be used right now. ``reason`` is one of: breaker_open, timeout, connection, error."""

    def __init__(self, name: str, reason: str, detail: str = "") -> None:
        super().__init__(f"{name} unavailable: {reason} {detail}".strip())
        self.name, self.reason, self.detail = name, reason, detail


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def budget(name: str) -> float:
    """Total seconds one call (all attempts together) may take. Read per call so env changes apply without re-import."""
    return max(0.2, _env_float(f"SOPHON_LIVE_TIMEOUT_{name.upper()}", DEFAULT_BUDGET_S.get(name, 10.0)))


class Breaker:
    """closed -> (N consecutive failures) -> open for T s -> half_open (one probe) -> closed on success, open on failure."""

    def __init__(self, name: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self._clock = name, clock
        self.state = "closed"
        self.fails = 0
        self.opened_at = 0.0
        self.probing = False
        self.total_ok = self.total_fail = self.total_rejected = self.opened_count = 0
        self.last_error = ""
        self.last_ok_ts = self.last_fail_ts = 0.0

    @property
    def threshold(self) -> int:
        return max(1, int(_env_float(f"SOPHON_LIVE_BREAKER_{self.name.upper()}_N", DEFAULT_BREAKER_N)))

    @property
    def open_s(self) -> float:
        return max(0.0, _env_float(f"SOPHON_LIVE_BREAKER_{self.name.upper()}_OPEN_S", DEFAULT_BREAKER_OPEN_S))

    def allow(self) -> bool:
        if self.state == "open":
            if self._clock() - self.opened_at >= self.open_s:
                self.state, self.probing = "half_open", False
            else:
                self.total_rejected += 1
                return False
        if self.state == "half_open":
            if self.probing:                       # one probe at a time; everyone else fails fast
                self.total_rejected += 1
                return False
            self.probing = True
        return True

    def success(self) -> None:
        self.total_ok += 1
        self.last_ok_ts = time.time()
        if self.state != "closed":
            log.warning("model %s recovered: breaker closed", self.name)
        self.state, self.fails, self.probing = "closed", 0, False

    def release_probe(self) -> None:
        """The caller was cancelled before the service gave a verdict: that is not a success and not a failure, but a half-open probe
        slot must not stay taken (RC2-3: otherwise the breaker stays half_open and rejects every call until the process restarts)."""
        self.probing = False

    def failure(self, reason: str) -> None:
        self.total_fail += 1
        self.last_fail_ts = time.time()
        self.last_error = reason
        self.fails += 1
        self.probing = False
        if self.state == "half_open" or self.fails >= self.threshold:
            if self.state != "open":
                self.opened_count += 1
                log.warning("model %s breaker OPEN for %.0fs after %d failures (last: %s)", self.name, self.open_s, self.fails, reason)
            self.state, self.opened_at = "open", self._clock()

    def snapshot(self) -> dict[str, Any]:
        remaining = max(0.0, self.open_s - (self._clock() - self.opened_at)) if self.state == "open" else 0.0
        return {"state": self.state, "consecutive_failures": self.fails, "open_remaining_s": round(remaining, 1),
                "ok": self.total_ok, "failed": self.total_fail, "rejected": self.total_rejected, "opened": self.opened_count,
                "last_error": self.last_error, "last_ok_ts": self.last_ok_ts or None, "last_fail_ts": self.last_fail_ts or None,
                "budget_s": budget(self.name)}


_BREAKERS: dict[str, Breaker] = {}


def breaker(name: str) -> Breaker:
    b = _BREAKERS.get(name)
    if b is None:
        b = _BREAKERS[name] = Breaker(name)
    return b


def reset() -> None:
    """Forget all breaker state (tests, operator restart of a downstream)."""
    _BREAKERS.clear()


def is_connection_error(e: BaseException) -> bool:
    """Connection-level failures only: the request never reached a working server. Timeouts are not retried."""
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return False
    try:
        import aiohttp
        if isinstance(e, aiohttp.ClientConnectionError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(e, (ConnectionError, OSError))


async def call(name: str, fn: Callable[[], Awaitable[Any]], *, retries: int = 1, total_s: float | None = None) -> Any:
    """Run ``fn`` under the service's budget and breaker. Raises :class:`ModelUnavailable`; never returns a fallback.

    ``retries`` extra attempts are made only after a connection error, with full-jitter exponential backoff, and only
    while the total budget still has room. A timeout or an application error is final for this call.
    """
    br = breaker(name)
    if not br.allow():
        raise ModelUnavailable(name, "breaker_open")
    total = budget(name) if total_s is None else total_s
    deadline = time.monotonic() + total
    attempt = 0
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                br.failure("timeout")
                raise ModelUnavailable(name, "timeout", f"budget {total:.1f}s")
            try:
                res = await asyncio.wait_for(fn(), left)
                br.success()
                return res
            except ModelUnavailable:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                br.failure("timeout")
                raise ModelUnavailable(name, "timeout", f"budget {total:.1f}s") from None
            except Exception as e:  # noqa: BLE001
                if is_connection_error(e) and attempt < retries:
                    attempt += 1
                    pause = random.uniform(0, RETRY_BASE_S * (2 ** attempt))
                    if time.monotonic() + pause < deadline:
                        log.info("model %s connection error %r, retry %d after %.0f ms", name, e, attempt, pause * 1000)
                        await asyncio.sleep(pause)
                        continue
                reason = "connection" if is_connection_error(e) else "error"
                br.failure(f"{reason}: {error_label(e)}")
                raise ModelUnavailable(name, reason, error_label(e)) from e
    except BaseException as e:
        if not isinstance(e, Exception):          # CancelledError / KeyboardInterrupt: no verdict from the service, free the probe slot
            br.release_probe()
        raise


def snapshot() -> dict[str, Any]:
    """Breaker state of every service this process has talked to (always lists the three known ones)."""
    for n in DEFAULT_BUDGET_S:
        breaker(n)
    return {n: b.snapshot() for n, b in sorted(_BREAKERS.items())}


async def probe(name: str, url: str, *, timeout_s: float = 1.5) -> dict[str, Any]:
    """Active health probe (GET). Reported in /health?detail=1; does not touch the breaker."""
    import aiohttp
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as s:
            async with s.get(url) as r:
                return {"up": r.status < 500, "status": r.status, "ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"up": False, "error": type(e).__name__, "ms": round((time.monotonic() - t0) * 1000)}
