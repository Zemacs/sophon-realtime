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


def missing_slots_for_text(text: str) -> list[str]:
    """Slots the C layer already knows are missing from the raw request (used by admit's CLARIFY route)."""
    return ["city"] if "天气" in (text or "") else []
