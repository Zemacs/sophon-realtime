"""Local JSON memory for short user preferences and facts (store only; policy lives in coordinator.memory_gate)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_LIVE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FILE = _LIVE_ROOT / "logs" / "assistant_memory.json"
MEMORY_FILE = _DEFAULT_FILE


def _memory_path() -> Path:
    override = os.environ.get("SOPHON_LIVE_MEMORY_PATH", "").strip()
    return Path(override) if override else MEMORY_FILE


def _load_store() -> dict[str, str]:
    path = _memory_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def _save_store(store: dict[str, str]) -> None:
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


class _JsonStore:
    """PrefStore over the local JSON file (path resolved on every call, so tests and env overrides stay honest)."""

    def load(self) -> dict[str, str]:
        return _load_store()

    def save(self, store: dict[str, str]) -> None:
        _save_store(store)


async def run(args: dict[str, Any] | None = None) -> str:
    """memory_note tool. Every op goes through the C-layer memory_gate (long-term prefs vs task slots)."""
    from ..coordinator.memory_gate import MemoryGate

    return MemoryGate(_JsonStore()).handle(args)
