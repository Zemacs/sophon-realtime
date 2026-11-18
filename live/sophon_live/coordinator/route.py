"""Build System One questions and interpret route/control decisions."""
from __future__ import annotations

import re
from typing import Any

from .ledger import Ledger
from .schemas import ControlDecision, TaskStatus

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


# Heuristic fallback when JEV is down (or cannot name a target) — keep Path-1 usable.
_CLARIFY_HINTS = ("天气怎么样", "明天天气", "今天天气", "会不会下雨")
_CITIES = ("北京", "上海", "广州", "深圳", "杭州", "成都", "城市")
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
# Words that make a short utterance a *new* request, never the answer to a pending question.
_NEW_REQUEST_HINTS = ("几点", "什么时间", "新闻", "头条", "打开", "关掉", "空调", "记住", "笑话", "讲个")
_CANCEL_HINTS = ("取消", "别查了", "不用查了", "不要查了", "算了", "不用了", "别办了")
_STATUS_HINTS = (
    "查得怎么样",
    "查到了吗",
    "查好了吗",
    "查完了吗",
    "结果呢",
    "结果怎么样",
    "进度",
    "刚才那个",
    "刚才的结果",
    "刚才查的",
    "再说一遍",
    "重新说一遍",
    "没听清",
    "再听一下",
)
_REFINE_HINTS = ("改成", "换成", "改为", "不对", "再讲", "再来一个", "换一个", "另一个", "不一样的", "不好笑", "重新讲")
_SWAP_RE = re.compile(r"(那|换成|改成)?[一-鿿]{2,4}(的呢|的|呢)")
_SLOT_MAX_CHARS = 10
_ACK_WORDS = ("好的", "好吧", "好嘞", "好", "嗯", "哦", "行", "可以", "谢谢", "多谢", "知道了", "明白", "收到", "ok", "OK")


def _is_ack(t: str) -> bool:
    """「好的」「谢谢」「好的谢谢」: an acknowledgement is never the answer to a pending question."""
    rest = t
    for w in sorted(_ACK_WORDS, key=len, reverse=True):
        rest = rest.replace(w, "")
    return rest.strip("，,。.！!？? ") == ""


def heuristic_control(text: str, ledger: Ledger) -> ControlDecision | None:
    """Pick (op, task_id, version) from the Ledger when JEV is off or names no target.

    The Ledger decides which task exists; the words only choose the operation.
    Returns None when the utterance is not about an existing task.
    """
    t = (text or "").strip()
    latest = ledger.latest()
    if not t or latest is None:
        return None
    pending = [r for r in ledger.active() if r.status in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.NEEDS_INPUT)]

    def pick(op: str, rec) -> ControlDecision:  # noqa: ANN001
        return ControlDecision(op=op, task_id=rec.task_id, version=rec.version, via="heuristic")  # type: ignore[arg-type]

    if any(h in t for h in _CANCEL_HINTS):
        return pick("CANCEL", pending[-1] if pending else latest)
    if any(h in t for h in _STATUS_HINTS):
        return pick("STATUS", latest)
    waiting = ledger.pending_input()
    if waiting is not None and len(t) <= _SLOT_MAX_CHARS and not _is_ack(t) and not any(h in t for h in _NEW_REQUEST_HINTS):
        return pick("REFINE", waiting)
    if any(h in t for h in _REFINE_HINTS):
        return pick("REFINE", latest)
    if "天气" in latest.objective and _SWAP_RE.fullmatch(t):
        return pick("REFINE", latest)
    return None


def heuristic_route(text: str, ledger: Ledger) -> str:
    t = (text or "").strip()
    if heuristic_control(t, ledger) is not None:
        return "CONTROL"
    if any(h in t for h in _CLARIFY_HINTS) and not any(c in t for c in _CITIES):
        return "CLARIFY"
    if any(h in t for h in _WORKER_HINTS):
        return "WORKER"
    if "别说了" in t or "先停" in t or "闭嘴" in t:
        return "STOP_SPEAKING"
    return "FAST"
