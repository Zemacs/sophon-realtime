"""Measure (not enforce) the router model on the v0.1.1 judgments that replaced the last keyword rules.

    python -m tests.eval_control_judgment [--out file.json]

* ``control``: given live tasks in the ledger, is the utterance a REFINE / CANCEL / STATUS of one of them, or NEW? (replaces
  ``route.heuristic_control`` and ``route.slot_edit``). The effective decision is read the way production reads it: the ``control``
  answer is honoured under route WORKER or CONTROL.
* ``missing_slot``: on a CLARIFY route, which slot is missing? (replaces ``slots.missing_slots_for_text``)
* ``memory_save`` vs ``memory_note``: did the user ask to keep something long-term? (replaces ``EXPLICIT_SAVE_HINTS``)

Needs a live JEV (SOPHON_JEV_URL). The labeled sentences are only sent TO the model; nothing here is a rule.
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
from sophon_live.coordinator.clarify import MISSING_SLOT_KEY, build_missing_slot_question  # noqa: E402
from sophon_live.coordinator.ledger import Ledger  # noqa: E402
from sophon_live.coordinator.route import build_route_questions, build_state  # noqa: E402
from sophon_live.coordinator.schemas import TaskStatus  # noqa: E402
from tests import jev_stub  # noqa: E402  (importing ``tests`` installs the fake router; the eval talks to the real one)

# (task objective, task status, [(utterance, expected op)])
CONTROL_CASES = [
    ("明天上海会下雨吗", "running", [
        ("明天北京会下雨吗", "REFINE"), ("改成北京的", "REFINE"), ("换成广州", "REFINE"), ("不对，我是说杭州", "REFINE"),
        ("后天呢", "REFINE"), ("深圳明天会下雨吗", "REFINE"),
        ("别查了", "CANCEL"), ("取消刚才那个", "CANCEL"), ("算了，不用了", "CANCEL"),
        ("查到了吗", "STATUS"), ("刚才那个查得怎么样了", "STATUS"),
        ("现在几点了", "NEW"), ("给我讲个笑话", "NEW"), ("今天有什么科技新闻", "NEW"), ("把客厅的灯打开", "NEW"), ("帮我记一下我对花生过敏", "NEW"),
    ]),
    ("明天上海会下雨吗", "completed", [
        ("那北京呢", "REFINE"), ("明天成都会下雨吗", "REFINE"), ("换成后天", "REFINE"),
        ("再说一遍结果", "STATUS"), ("刚才查的结果是什么", "STATUS"),
        ("现在几点了", "NEW"), ("给我讲个笑话", "NEW"), ("来点财经新闻", "NEW"),
    ]),
    ("讲个笑话", "completed", [
        ("再讲一个", "REFINE"), ("换一个不一样的", "REFINE"), ("不好笑，重新讲", "REFINE"),
        ("明天天气怎么样", "NEW"), ("现在几点了", "NEW"), ("打开卧室空调", "NEW"),
    ]),
    ("明天天气怎么样", "needs_input", [
        ("北京的", "REFINE"), ("上海", "REFINE"), ("杭州吧", "REFINE"),
        ("现在几点了", "NEW"), ("讲个笑话", "NEW"),
    ]),
]
MISSING_CASES = [("明天天气怎么样", "city"), ("会下雨吗", "city"), ("今天冷不冷", "city"), ("帮我查一下", "query"), ("搜一下", "query"),
                 ("给我念几条新闻", "topic"), ("来点新闻", "topic")]


def _ledger(obj: str, status: str):
    led = Ledger()
    rec = led.create(obj)
    if status == "running":
        led.mark_running(rec.task_id)
    elif status == "completed":
        led.mark_running(rec.task_id)
        led.mark_completed(rec.task_id, "好的。")
    elif status == "needs_input":
        led.mark_needs_input(rec.task_id)
    return led, rec


async def _control(obj: str, status: str, text: str):
    led, rec = _ledger(obj, status)
    state = build_state(text, led, playback_active=False)
    state["user_said"] = text
    questions = build_route_questions(led)
    questions[intent.TOOL_KEY] = intent.build_tool_question()
    resp = await jev_stub.real_system_one(state, questions)
    route, key = jev_client.choice_answer(resp, "route"), jev_client.choice_answer(resp, "control")
    if route not in ("CONTROL", "WORKER") or not key or key == "NEW":
        return "NEW", route, key
    return key.split(":")[0], route, key


async def _missing(text: str):
    resp = await jev_stub.real_system_one({"utterance": text, "language": "zh"}, {MISSING_SLOT_KEY: build_missing_slot_question()})
    return jev_client.choice_answer(resp, MISSING_SLOT_KEY)


async def _retry(fn, *a):
    err = ""
    for _ in range(2):
        try:
            return await fn(*a)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    return ("ERR:" + err, None, None) if fn is _control else "ERR:" + err


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sem = asyncio.Semaphore(3)

    async def lim(fn, *args):
        async with sem:
            return await _retry(fn, *args)

    jobs, meta = [], []
    for obj, st, cases in CONTROL_CASES:
        for text, want in cases:
            jobs.append(lim(_control, obj, st, text)); meta.append((obj, st, text, want))
    res = await asyncio.gather(*jobs)
    wrong = [(m, r) for m, r in zip(meta, res) if r[0] != m[3]]
    by_op: dict[str, list[int]] = {}
    for m, r in zip(meta, res):
        by_op.setdefault(m[3], [0, 0])[1] += 1
        by_op[m[3]][0] += r[0] == m[3]
    print(f"[control] n={len(meta)} wrong={len(wrong)} ({len(wrong)/len(meta):.1%})  per-op correct/total: {by_op}")
    for m, r in wrong:
        print("   WRONG", m[1], repr(m[0]), repr(m[2]), "want", m[3], "got", r)
    mres = await asyncio.gather(*(lim(_missing, t) for t, _ in MISSING_CASES))
    mw = [(t, w, g) for (t, w), g in zip(MISSING_CASES, mres) if g != w]
    print(f"[missing_slot] n={len(MISSING_CASES)} wrong={len(mw)}")
    for w in mw:
        print("   WRONG", w)
    json.dump({"control": {"n": len(meta), "wrong": [[list(m), list(r)] for m, r in wrong], "by_op": by_op},
               "missing_slot": {"n": len(MISSING_CASES), "wrong": mw}}, open(a.out, "w") if a.out else sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
