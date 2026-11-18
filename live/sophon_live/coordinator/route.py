"""Build System One questions and interpret route/control decisions."""
from __future__ import annotations

from typing import Any

from .ledger import Ledger

# Interactive routes (Jarvis-inspired, Path-1 subset)
ROUTE_CRITERIA = {
    "FAST": "纯致谢、确认、或无需任何生成/工具的极短应答（例如「好的」「谢谢」）。",
    "CLARIFY": "缺关键槽位（如天气缺城市），应追问用户，暂不启动后台工具。",
    "WORKER": "需要后台执行：查时间/天气/新闻、开关设备、记偏好、讲笑话/段子等娱乐生成，一律委派 Worker。",
    "CONTROL": "针对已有任务：改口、取消、查进度、重听结果，或同一笑话再来一个。",
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
        criteria: dict[str, str] = {"NEW": "与进行中任务无关的新目标"}
        for t in active:
            tid, ver, obj = t["task_id"], t["version"], t["objective"][:80]
            criteria[f"REFINE:{tid}:{ver}"] = f"改口/细化任务：{obj}"
            criteria[f"CANCEL:{tid}:{ver}"] = f"取消任务：{obj}"
            criteria[f"STATUS:{tid}:{ver}"] = f"询问进度/结果：{obj}"
        q["control"] = {
            "type": "choice",
            "instructions": "若 route=CONTROL，选对哪个任务做什么；否则可忽略。",
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


# Heuristic fallback when JEV is down — keep Path-1 usable.
_CLARIFY_HINTS = ("天气怎么样", "明天天气", "今天天气", "会不会下雨")
_WORKER_HINTS = (
    "几点",
    "什么时间",
    "天气",
    "气温",
    "新闻",
    "头条",
    "打开",
    "关掉",
    "空调",
    "记住",
    "我喜欢",
    "笑话",
    "讲个",
)
_CONTROL_HINTS = ("再讲", "换一个", "不好笑", "取消", "别查了", "北京的", "上海的", "改成")


def heuristic_route(text: str, ledger: Ledger) -> str:
    t = (text or "").strip()
    if ledger.latest() and any(h in t for h in _CONTROL_HINTS):
        return "CONTROL"
    if any(h in t for h in _CLARIFY_HINTS) and not any(
        c in t for c in ("北京", "上海", "广州", "深圳", "杭州", "成都", "城市")
    ):
        return "CLARIFY"
    if any(h in t for h in _WORKER_HINTS):
        return "WORKER"
    if "别说了" in t or "先停" in t or "闭嘴" in t:
        return "STOP_SPEAKING"
    return "FAST"
