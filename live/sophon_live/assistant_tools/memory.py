"""Local JSON memory for short user preferences and facts."""
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


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    op = (args.get("op") or "").strip().lower()
    key = (args.get("key") or "").strip()
    value = args.get("value")

    if op == "save":
        if not key:
            return "保存记忆需要提供一个键名。"
        if value is None or str(value).strip() == "":
            return "要保存什么内容？请再说一下值。"
        store = _load_store()
        store[key] = str(value).strip()
        try:
            _save_store(store)
        except OSError:
            return "记忆文件写入失败，请检查磁盘或权限。"
        return f"好的，我已经记住「{key}」了。"

    if op == "recall":
        if not key:
            return "要回忆哪一条？请说一下键名。"
        store = _load_store()
        if key not in store:
            return f"还没有记录过「{key}」。"
        return f"「{key}」是：{store[key]}。"

    if op == "list":
        store = _load_store()
        if not store:
            return "目前还没有保存任何记忆。"
        keys = "、".join(sorted(store.keys()))
        return f"当前记住了这些项：{keys}。"

    return "不支持的记忆操作，请用 save、recall 或 list。"