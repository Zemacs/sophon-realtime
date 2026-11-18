"""One cap on outbound web requests in flight, shared by web_search and news (Critic S-9 / S-14).

``urllib`` cannot be interrupted: when an ``asyncio`` caller gives up (hedge cancelled, budget spent) the thread runs on until
its own socket timeout. This cap bounds how many such threads can exist at once across ALL tools, so slow stragglers cannot
pile up sockets or starve the shared ``to_thread`` pool. Over the cap a request fails at once (the tool then reports its normal,
honest failure) instead of queueing.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, TypeVar

T = TypeVar("T")

MAX_INFLIGHT = int(os.environ.get("SOPHON_SEARCH_MAX_INFLIGHT", "12"))
SHARED = threading.BoundedSemaphore(MAX_INFLIGHT)


def guarded(fn: Callable[..., T], *args, sem: threading.BoundedSemaphore | None = None, **kw) -> T:  # noqa: ANN002, ANN003
    """Run ``fn`` if a slot is free (non-blocking), release the slot when it returns or raises."""
    s = sem if sem is not None else SHARED
    if not s.acquire(blocking=False):
        raise OSError("too many outbound requests in flight")
    try:
        return fn(*args, **kw)
    finally:
        s.release()
