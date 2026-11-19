"""Coordinator-owned CLARIFY wording: the Worker reports missing slots, the C layer decides what to ask."""
from __future__ import annotations

_ASK = {
    "city": "请问您想查哪个城市的天气？",
    "query": "您想搜什么内容？",
    "topic": "您想听哪方面的新闻？",
}
_GENERIC = "还缺一点信息，方便再说具体一点吗？"


def clarify_for_slots(slots: list[str] | None) -> str:
    """One short question for the first missing slot; unknown slots get the generic ask."""
    for s in slots or []:
        if s in _ASK:
            return _ASK[s]
    return _GENERIC


# Which slot is missing is the ROUTER MODEL's judgment (one extra, rare question: only when the route is CLARIFY).
MISSING_SLOT_KEY = "missing_slot"
_MISSING_CRITERIA = {
    "city": "缺城市或地点（比如问天气但没说哪里）",
    "query": "缺要搜索或查询的具体内容",
    "topic": "缺想听哪方面的新闻",
    "other": "缺别的信息，或说不清缺什么",
}


def build_missing_slot_question() -> dict:
    return {"type": "choice", "instructions": "这句话还缺哪一项关键信息，没有它就没法办？选最缺的一项。", "criteria": dict(_MISSING_CRITERIA)}


def slots_from_answer(choice: str | None) -> list[str]:
    """The model's enumerated answer -> slot names ([] = unknown / other -> the generic ask)."""
    return [choice] if choice in _ASK else []
