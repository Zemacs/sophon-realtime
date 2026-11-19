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
                    raise model_client.HttpStatusError(resp.status)             # the body stays in the log line above, never in the exception
                try:
                    return await resp.json(content_type=None)
                except Exception as e:  # noqa: BLE001
                    log.warning("jev bad json (HTTP %s): %s", resp.status, text[:200])
                    raise RuntimeError("jev bad json") from e

    # Raises model_client.ModelUnavailable (breaker open / timeout / connection / error); the caller degrades model-free.
    return await model_client.call("jev", _once, retries=1, total_s=DEFAULT_TIMEOUT)


def raw_choice(resp: Any, key: str) -> Any:
    """The ``choice`` of a ``choice`` answer to question ``key`` exactly as the router sent it (any JSON type), or None. Never raises on a wrongly shaped response
    (a list where a dict belongs, a bare string answer, ...): a bad field is unusable, it must not take the other answers down (Critic I-2)."""
    answers = resp.get("answers") if isinstance(resp, dict) else None
    ans = answers.get(key) if isinstance(answers, dict) else None
    if isinstance(ans, dict) and ans.get("type") == "choice":
        return ans.get("choice")
    return None


def choice_answer(resp: Any, key: str) -> str | None:
    """The router's option key (a string) for question ``key``; None when the answer is missing or not a string (the caller counts it as invalid)."""
    c = raw_choice(resp, key)
    return c if isinstance(c, str) else None


def noul_prob(resp: Any, key: str) -> float | None:
    answers = resp.get("answers") if isinstance(resp, dict) else None
    ans = answers.get(key) if isinstance(answers, dict) else None
    if isinstance(ans, dict) and ans.get("type") == "noul":
        try:
            return float(ans.get("noul"))
        except (TypeError, ValueError):
            return None
    return None
