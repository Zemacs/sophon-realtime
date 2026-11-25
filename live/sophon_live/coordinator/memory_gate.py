"""memory_gate: long-term preferences are NOT task-scoped slots.

Two scopes, one gate:
  * long-term preferences — persisted (JSON store), only written when the user explicitly asked to remember;
  * task slots — live on the Ledger task version (city, date, topic ...), die with it, never persisted.

Every ``memory_note`` write made on behalf of a task goes through ``MemoryGate.handle``:
a value that is only an answer to *this* task (e.g. 「北京的」 after 「明天天气怎么样」) becomes a task slot, not memory.
Reading memory never creates slots, and a stored preference never silently fills a missing slot
(missing slots are asked via CLARIFY).

Pure policy: no tool I/O here. The store is injected (see ``assistant_tools.memory``).
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

from .ledger import Ledger

# The user asked to keep something: 记住/记下/保存/以后/默认/别忘/我喜欢/我的偏好/我习惯.
EXPLICIT_SAVE_HINTS = ("记住", "记下", "记得", "保存", "以后", "默认", "别忘", "我喜欢", "我的偏好", "我习惯")
# Keys that describe *this* request. As long-term memory they get the preferred_ prefix, so a stored
# value can never share a name with a task slot.
TASK_SLOT_KEYS = frozenset({"city", "location", "when", "date", "time", "day", "device", "action", "topic", "query", "target", "value"})
PREFERENCE_PREFIXES = ("preferred_", "default_", "favorite_")


def has_explicit_save_intent(text: str | None) -> bool:
    t = (text or "").strip()
    return bool(t) and any(h in t for h in EXPLICIT_SAVE_HINTS)


def long_term_key(key: str) -> str:
    k = key.strip()
    if k.startswith(PREFERENCE_PREFIXES) or k not in TASK_SLOT_KEYS:
        return k
    return "preferred_" + k


class PrefStore(Protocol):
    def load(self) -> dict[str, str]: ...

    def save(self, store: dict[str, str]) -> None: ...


@dataclass
class TaskContext:
    """Who is asking: the task (id, version), the user's words behind it, and the Ledger that owns its slots."""

    task_id: str
    version: int
    utterance: str
    ledger: Ledger


_CURRENT: ContextVar[TaskContext | None] = ContextVar("memory_gate_task", default=None)


def current_task() -> TaskContext | None:
    return _CURRENT.get()


@contextlib.contextmanager
def bind_task(ctx: TaskContext | None) -> Iterator[None]:
    token = _CURRENT.set(ctx)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class MemoryGate:
    def __init__(self, store: PrefStore, *, context: Callable[[], TaskContext | None] = current_task) -> None:
        self.store = store
        self._context = context

    # ---- reads: long-term only (never slots, and they never create slots)
    def recall(self, key: str) -> str:
        if not key:
            return "要回忆哪一条？请说一下键名。"
        data = self.store.load()
        if key not in data:
            return f"还没有记录过「{key}」。"
        return f"「{key}」是：{data[key]}。"

    def list(self) -> str:
        data = self.store.load()
        if not data:
            return "目前还没有保存任何记忆。"
        return f"当前记住了这些项：{'、'.join(sorted(data.keys()))}。"

    # ---- writes
    def save(self, key: str, value: Any) -> str:
        key = (key or "").strip()
        if not key:
            return "保存记忆需要提供一个键名。"
        if value is None or str(value).strip() == "":
            return "要保存什么内容？请再说一下值。"
        val = str(value).strip()
        ctx = self._context()
        if ctx is not None and not has_explicit_save_intent(ctx.utterance):
            # Only an answer for this task: keep it as a task slot, not as memory.
            ctx.ledger.set_slot(ctx.task_id, ctx.version, key, val)
            return f"这次的「{key}」只用于当前任务，没有存成长期记忆。想让我长期记住的话，请说「记住……」。"
        store_key = long_term_key(key) if ctx is not None else key
        data = self.store.load()
        data[store_key] = val
        try:
            self.store.save(data)
        except OSError:
            return "记忆文件写入失败，请检查磁盘或权限。"
        return f"好的，我已经记住「{store_key}」了。"

    def handle(self, args: dict[str, Any] | None) -> str:
        args = args or {}
        op = (args.get("op") or "").strip().lower()
        key = (args.get("key") or "").strip()
        if op == "save":
            return self.save(key, args.get("value"))
        if op == "recall":
            return self.recall(key)
        if op == "list":
            return self.list()
        return "不支持的记忆操作，请用 save、recall 或 list。"
