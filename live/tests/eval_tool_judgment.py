"""Measure (not enforce): how often does the router model call a request tool-backed / not, on the labeled sets?

    python -m tests.eval_tool_judgment [--sets dev,critic,held,frozen,tools] [--out file.json]

Needs a live JEV (SOPHON_JEV_URL, default http://127.0.0.1:8110/v1/systemone). Not a unit test and not a rule: the labeled
sentences are only ever sent TO the model; no code path uses them to decide anything. Targets (v0.1.0-rc2): miss <= 5%,
false trigger <= 5% on each set. ``miss`` = a realtime/tool request the model answered NONE (or a wrong tool class);
``over`` = a free request the model sent to a tool.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_LIVE = Path(__file__).resolve().parents[1]
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from sophon_live.coordinator import intent, jev_client  # noqa: E402
from tests import jev_stub  # noqa: E402  (importing ``tests`` installs the fake router; the eval talks to the real one)
from tests import intent_critic92 as critic  # noqa: E402
from tests import intent_dev as dev  # noqa: E402
from tests import intent_frozen as frozen  # noqa: E402
from tests import intent_rc1_heldout as held  # noqa: E402
from tests import intent_tools_set as toolset  # noqa: E402

SETS = {
    "dev": (dev.DEV_REALTIME, dev.DEV_FREE),
    "critic": (critic.CRITIC_REALTIME, critic.CRITIC_FREE),
    "held": (held.HELDOUT_REALTIME, held.HELDOUT_FREE),
    "frozen": (frozen.FROZEN_REALTIME, frozen.FROZEN_FREE),
}


async def judge(text: str) -> str | None:
    state = {"utterance": text, "user_said": text, "playback_active": False, "tasks": [], "language": "zh"}
    resp = await jev_stub.real_system_one(state, {intent.TOOL_KEY: intent.build_tool_question()})
    return jev_client.choice_answer(resp, intent.TOOL_KEY)


async def run_many(texts, conc: int = 6):
    sem = asyncio.Semaphore(conc)

    async def one(t):
        async with sem:
            for _ in range(2):
                try:
                    return await judge(t)
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
            return "ERR:" + err

    return await asyncio.gather(*(one(t) for t in texts))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="dev,critic,held,frozen,tools")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out: dict = {}
    for name in a.sets.split(","):
        if name == "tools":
            items = [(t, tool) for tool, ts in toolset.TOOL_SET.items() for t in ts]
            ans = await run_many([t for t, _ in items])
            wrong = [(t, want, got) for (t, want), got in zip(items, ans) if got != want]
            out[name] = {"n": len(items), "wrong": wrong, "rate": round(len(wrong) / len(items), 3)}
            print(f"[{name}] n={len(items)} wrong={len(wrong)} ({len(wrong)/len(items):.1%})")
            for w in wrong:
                print("   ", w)
            continue
        pos, neg = SETS[name]
        ap_, an_ = await run_many(pos), await run_many(neg)
        miss = [(t, g) for t, g in zip(pos, ap_) if g in (None, intent.NONE) or str(g).startswith("ERR")]
        over = [(t, g) for t, g in zip(neg, an_) if g not in (intent.NONE,)]
        out[name] = {"pos": len(pos), "neg": len(neg), "miss": miss, "over": over,
                     "miss_rate": round(len(miss) / len(pos), 3), "over_rate": round(len(over) / len(neg), 3)}
        print(f"[{name}] realtime n={len(pos)} miss={len(miss)} ({len(miss)/len(pos):.1%})  free n={len(neg)} over={len(over)} ({len(over)/len(neg):.1%})")
        for m in miss:
            print("   MISS", m)
        for o in over:
            print("   OVER", o)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
