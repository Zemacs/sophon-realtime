"""Delivery bookkeeping (pure, no audio / network): which spoken answer belongs to which (task_id, version).

The gateway arms a delivery when it hands a result to the voice front, attaches it to the first
response the engine starts afterwards, and reports how that response ended:
  played to the end -> delivered      cut off by barge-in -> interrupted      session closed -> abandoned
Receipts go to the Ledger via ``coordinator.api.on_playback_receipt``. An interruption never cancels work.
"""
from __future__ import annotations

from collections import deque

from .schemas import PlaybackReceipt


class DeliveryTracker:
    def __init__(self) -> None:
        self._armed: deque[tuple[str, int]] = deque()  # handed to the front, response not started yet
        self._attached: dict[str, tuple[str, int]] = {}  # response id -> (task_id, version)

    def arm(self, task_id: str, version: int) -> None:
        self._armed.append((task_id, version))

    def attach(self, rid: str) -> bool:
        """A new response started: it carries the oldest armed delivery (if any)."""
        if rid in self._attached or not self._armed:
            return False
        self._attached[rid] = self._armed.popleft()
        return True

    def on_played(self, rid: str) -> PlaybackReceipt | None:
        item = self._attached.pop(rid, None)
        if item is None:
            return None
        return PlaybackReceipt(item[0], item[1], "delivered")

    def on_interrupted(self) -> list[PlaybackReceipt]:
        """Barge-in: every response that was speaking is cut. Deliveries not yet speaking stay expected."""
        out = [PlaybackReceipt(t, v, "interrupted") for t, v in self._attached.values()]
        self._attached.clear()
        return out

    def on_close(self) -> list[PlaybackReceipt]:
        out = [PlaybackReceipt(t, v, "abandoned") for t, v in self._attached.values()]
        out += [PlaybackReceipt(t, v, "abandoned") for t, v in self._armed]
        self._attached.clear()
        self._armed.clear()
        return out
