"""Build System One questions and interpret route/control decisions."""
from __future__ import annotations

from typing import Any

from .ledger import Ledger

# Interactive routes (Jarvis-inspired, Path-1 subset)
ROUTE_CRITERIA = {
    "FAST": "纯致谢、确认、或无需任何生成/工具的极短应答（例如「好的」「谢谢」）。",
    "CLARIFY": "缺关键槽位（如天气缺城市），应追问用户，暂不启动后台工具。",
    "WORKER": "需要后台执行：查时间/天气/新闻、开关设备、记偏好、讲笑话/段子等娱乐生成，一律委派 Worker。",
    "CONTROL": "针对已有任务：改口（含把同一件事换了地点、日期或条件重说一遍）、取消、查进度、重听结果，或同一笑话再来一个。",
    "STOP_SPEAKING": "只要停下当前播报，不要取消后台任务。",
}


def build_route_questions(ledger: Ledger) -> dict[str, Any]:
    q: dict[str, Any] = {
        "route": {
            "type": "choice",
            "instructions": "根据用户这句话，选择下一步协调动作。",
            "criteria": dict(ROUTE_CRITERIA),
        }
    }
    active = ledger.snapshot_for_jev()
    if active:
        criteria: dict[str, str] = {"NEW": "与这些任务都无关的新目标（换了一件完全不同的事）"}
        for t in active:
            tid, ver, obj = t["task_id"], t["version"], t["objective"][:80]
            criteria[f"REFINE:{tid}:{ver}"] = (
                f"改口/细化/补充这个任务：{obj}（包括用户把同一件事换了地点、日期或条件再说一遍，哪怕说成了完整的新句子；"
                "也包括回答后台追问的缺项）")
            criteria[f"CANCEL:{tid}:{ver}"] = f"取消任务：{obj}"
            criteria[f"STATUS:{tid}:{ver}"] = f"询问进度/结果：{obj}"
        q["control"] = {
            "type": "choice",
            "instructions": "这句话是不是在改口、取消或追问上面某个已有任务？是就选对应任务和动作；说的是另一件事就选 NEW。",
            "criteria": criteria,
        }
    return q


def build_state(text: str, ledger: Ledger, *, playback_active: bool) -> dict[str, Any]:
    return {
        "utterance": text,
        "playback_active": playback_active,
        "tasks": ledger.snapshot_for_jev(),
        "language": "zh",
    }

