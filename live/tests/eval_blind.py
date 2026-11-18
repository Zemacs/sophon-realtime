"""Runner for a BLIND, one-shot labeled set (Critic RC2-6 / I-10). It ships no sentences: the reviewer (or the user) supplies the file.

    python -m tests.eval_blind SET.json [--out result.json] [--conc 3]

SET.json = {"realtime": [text, ...],   # requests that need a tool/lookup  -> a "miss" is the model answering NONE (or nothing usable)
            "free":     [text, ...],   # requests that need no tool        -> an "over" is the model naming a tool
            "tools":    {"get_weather": [text, ...], ...},   # optional: exact option expected
            "stress":   [text, ...]}   # optional: colloquial "now/today" chat that must stay NONE (same scoring as "free")
The question asked is exactly production's (``intent.build_tool_question`` through the real JEV client, with ``user_said`` = the text), so
this measures the router model, not a rule. Output: counts, rates and 95% Wilson upper bounds (0 misses of 38 is still ~7.9%).
Needs a reachable JEV (SOPHON_JEV_URL). Do NOT tune prompts against the file you score with; use a fresh set each time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

_LIVE = Path(__file__).resolve().parents[1]
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from sophon_live.coordinator import intent, jev_client  # noqa: E402
from tests import jev_stub  # noqa: E402  (importing ``tests`` installs the fake router; we talk to the real one)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


async def judge(text: str) -> str:
    state = {"utterance": text, "user_said": text, "playback_active": False, "tasks": []}
    resp = await jev_stub.real_system_one(state, {intent.TOOL_KEY: intent.build_tool_question()})
    return str(jev_client.choice_answer(resp, intent.TOOL_KEY))


async def run_many(texts: list[str], conc: int) -> list[str]:
    sem = asyncio.Semaphore(conc)

    async def one(t: str) -> str:
        async with sem:
            err = ""
            for _ in range(2):
                try:
                    return await judge(t)
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
            return "ERR:" + err

    return list(await asyncio.gather(*(one(t) for t in texts)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default="")
    ap.add_argument("--conc", type=int, default=3)
    a = ap.parse_args()
    data = json.loads(Path(a.file).read_text(encoding="utf-8"))
    out: dict = {}
    if data.get("realtime"):
        ans = await run_many(data["realtime"], a.conc)
        miss = [(t, g) for t, g in zip(data["realtime"], ans) if g in ("None", intent.NONE) or g.startswith("ERR")]
        out["realtime"] = {"n": len(ans), "miss": miss, "miss_rate": len(miss) / len(ans), "miss_upper95": wilson_upper(len(miss), len(ans))}
    for key in ("free", "stress"):
        if data.get(key):
            ans = await run_many(data[key], a.conc)
            over = [(t, g) for t, g in zip(data[key], ans) if g != intent.NONE]
            out[key] = {"n": len(ans), "over": over, "over_rate": len(over) / len(ans), "over_upper95": wilson_upper(len(over), len(ans))}
    if data.get("tools"):
        items = [(t, tool) for tool, ts in data["tools"].items() for t in ts]
        ans = await run_many([t for t, _ in items], a.conc)
        wrong = [(t, want, got) for (t, want), got in zip(items, ans) if got != want]
        out["tools"] = {"n": len(items), "wrong": wrong, "wrong_rate": len(wrong) / len(items), "wrong_upper95": wilson_upper(len(wrong), len(items))}
    for k, v in out.items():
        rate = [x for x in v if x.endswith("_rate")][0]
        up = [x for x in v if x.endswith("_upper95")][0]
        print(f"[{k}] n={v['n']} {rate}={v[rate]:.3f} 95%-upper={v[up]:.3f}")
    json.dump(out, open(a.out, "w", encoding="utf-8") if a.out else sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
