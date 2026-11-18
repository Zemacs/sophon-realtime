"""System One client for Karpo / TypeSafe-compatible decision servers."""
from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

log = logging.getLogger("sophon_live.coordinator.jev")

DEFAULT_URL = os.environ.get("SOPHON_JEV_URL", "http://127.0.0.1:8110/v1/systemone")
DEFAULT_TIMEOUT = float(os.environ.get("SOPHON_JEV_TIMEOUT", "8"))


async def system_one(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """POST /v1/systemone; return parsed JSON (answers map)."""
    endpoint = (url or DEFAULT_URL).rstrip("/")
    if not endpoint.endswith("/v1/systemone"):
        endpoint = endpoint + "/v1/systemone"
    body: dict[str, Any] = {"state": state, "questions": questions}
    if model:
        body["model"] = model
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(endpoint, json=body) as resp:
            text = await resp.text()
            if resp.status >= 400:
                log.warning("jev HTTP %s: %s", resp.status, text[:300])
                raise RuntimeError(f"jev HTTP {resp.status}: {text[:200]}")
            try:
                return await resp.json(content_type=None)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"jev bad json: {text[:200]}") from e


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
