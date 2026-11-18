"""One cap + one dedicated thread pool for outbound web requests, shared by web_search, news and weather (Critic S-9 / S-14 / S-19).

``urllib`` cannot be interrupted: when an ``asyncio`` caller gives up (hedge cancelled, budget spent) the thread runs on until its
own socket timeout. Two things bound the damage:

* the cap (``MAX_INFLIGHT`` slots) is taken on the EVENT-LOOP side, before the work is submitted, and the slot is given back only when
  the thread has really finished (or its work item was cancelled before it started). Over the cap a request raises at once and the
  tool reports its normal, honest failure. Nothing waits in a queue for a straggler's socket timeout (S-19: the 13th request used to
  queue behind 12 stuck threads for up to ~20 s because the slot was taken INSIDE the thread, after queueing);
* the threads come from a DEDICATED pool of exactly ``MAX_INFLIGHT`` workers, not from asyncio's default ``to_thread`` pool
  (``min(32, cpu + 4)``, 12 on an 8-core box), which ASR (``to_thread(self.asr, ...)``) and the device tool also use. Stuck outbound
  sockets can therefore never starve speech recognition, and (slots == workers) a slot holder never has to wait for a worker.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

BUSY_ZH = "我这边同时在查的太多了，稍等再问我一次。"
BUSY_EN = "I'm running too many lookups at once right now. Please ask me again in a moment."


class Busy(OSError):
    """The cap is full: a refusal OF OURS, not a network problem (S-24: the tools say so instead of 'network unavailable')."""


# one list per tool call; ``run`` appends to it when it refuses, the tool reads it when it has to report a failure
_SEEN: contextvars.ContextVar[list | None] = contextvars.ContextVar("sophon_outbound_busy", default=None)


def track_busy() -> list:
    """Start a fresh busy record for the current tool call (tasks it spawns share it through the copied context)."""
    seen: list = []
    _SEEN.set(seen)
    return seen


MAX_INFLIGHT = max(1, int(os.environ.get("SOPHON_SEARCH_MAX_INFLIGHT", "12")))
SHARED = threading.BoundedSemaphore(MAX_INFLIGHT)
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_INFLIGHT, thread_name_prefix="sophon-outbound")


async def run(fn: Callable[..., T], *args, sem: threading.BoundedSemaphore | None = None, **kw) -> T:  # noqa: ANN002, ANN003
    """Run ``fn(*args, **kw)`` in the outbound pool if a slot is free (non-blocking); raise OSError at once if not.

    The slot is released exactly once, when the work finishes, raises, or is cancelled before it started, NOT when the awaiting side
    is cancelled (the thread is still holding a socket then)."""
    s = sem if sem is not None else SHARED
    if not s.acquire(blocking=False):
        seen = _SEEN.get()
        if seen is not None:
            seen.append(1)
        raise Busy("too many outbound requests in flight")
    try:
        fut: Future = EXECUTOR.submit(functools.partial(fn, *args, **kw))
    except BaseException:
        s.release()
        raise
    fut.add_done_callback(lambda _f: s.release())
    return await asyncio.wrap_future(fut)
