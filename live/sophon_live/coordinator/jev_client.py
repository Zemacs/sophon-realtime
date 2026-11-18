"""System One client for Karpo / TypeSafe-compatible decision servers."""
from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

from .. import model_client

log = logging.getLogger("sophon_live.coordinator.jev")

DEFAULT_URL = os.environ.get("SOPHON_JEV_URL", "http://127.0.0.1:8110/v1/systemone")
# Budget now lives in model_client.budget("jev") (SOPHON_LIVE_TIMEOUT_JEV, default 1.5 s). SOPHON_JEV_TIMEOUT still wins if set.
DEFAULT_TIMEOUT = float(os.environ["SOPHON_JEV_TIMEOUT"]) if os.environ.get("SOPHON_JEV_TIMEOUT") else None


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
    async def _once() -> dict[str, Any]:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(endpoint, json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    log.warning("jev HTTP %s: %s", resp.status, text[:300])
                    raise RuntimeError(f"jev HTTP {resp.status}: {text[:200]}")
                try:
                    return await resp.json(content_type=None)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"jev bad json: {text[:200]}") from e

    # Raises model_client.ModelUnavailable (breaker open / timeout / connection / error); the caller degrades model-free.
    return await model_client.call("jev", _once, retries=1, total_s=DEFAULT_TIMEOUT)


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
