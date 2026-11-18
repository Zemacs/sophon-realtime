"""Simulated smart-home control with optional external hook."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from typing import Any

_HOOK_ENV = "SOPHON_LIVE_DEVICE_HOOK"


def _simulate(device: str, action: str, value: str | None, *, zh: bool) -> str:
    val = f"，设为{value}" if value else ""
    if zh:
        return f"（模拟设备）已帮你把{device}{action}{val}。"
    extra = f" to {value}" if value else ""
    return f"(simulated device) {device}: {action}{extra}."


def _run_hook(hook: str, payload: dict[str, Any]) -> str:
    path = os.path.abspath(os.path.expanduser(hook))
    if os.path.isfile(path):
        cmd = [path]
    elif shutil.which(path):
        cmd = [path]
    else:
        raise FileNotFoundError(f"device hook not found: {path}")

    # Prefer argv-style (device, action, [value]); fall back to JSON on stdin for scripts without args.
    device = str(payload.get("device") or "")
    action = str(payload.get("action") or "")
    value = payload.get("value")
    if device or action:
        cmd = cmd + [device, action] + ([str(value)] if value else [])

    proc = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False) if not (device or action) else None,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.stdout.strip():
        return proc.stdout.strip()
    if proc.returncode != 0 and proc.stderr.strip():
        return proc.stderr.strip()[:300]
    return ""


async def run(args: dict[str, Any] | None = None) -> str:
    args = args or {}
    device = str(args.get("device") or "device").strip()
    action = str(args.get("action") or "do").strip()
    value = args.get("value")
    value_s = str(value).strip() if value is not None and str(value).strip() else None

    zh = bool(re.search(r"[一-鿿]", device + action + (value_s or "")))
    if args.get("lang") == "en":
        zh = False
    elif args.get("lang") == "zh":
        zh = True

    hook = os.environ.get(_HOOK_ENV, "").strip()
    if hook:
        try:
            msg = await asyncio.to_thread(
                _run_hook,
                hook,
                {"device": device, "action": action, "value": value_s},
            )
            if msg:
                return msg
            return _simulate(device, action, value_s, zh=zh)
        except Exception as e:  # noqa: BLE001
            from . import toolmeta
            toolmeta.note(ok=False)
            if zh:
                return f"设备控制失败：{e}"
            return f"Device control failed: {e}"

    return _simulate(device, action, value_s, zh=zh)