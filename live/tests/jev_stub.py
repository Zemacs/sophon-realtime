"""Test-only stand-in for the JEV router.

Ledger/control/trace tests exercise the Coordinator's ledger behaviour, not Karpo. They get an explicit fake JEV.

THIS FILE IS A TEST DOUBLE FOR THE MODEL: its keyword tables stand in for what Karpo would answer, so the Coordinator can be
tested around a scripted router. Nothing under ``sophon_live`` may import it or hold such tables
(``tests/test_no_semantic_rules.py`` enforces that for the whole package; v0.1.1 removed the keyword router from production).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sophon_live.coordinator import api  # noqa: E402
import re  # noqa: E402

from sophon_live.coordinator.route import build_state as _real_build_state  # noqa: E402
from sophon_live.coordinator.ledger import Ledger  # noqa: E402
from sophon_live.coordinator.schemas import ControlDecision, TaskStatus  # noqa: E402

# ---- scripted fake model (former production keyword router, moved here in v0.1.1) ----
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




_last: dict = {}
real_system_one = api.jev_client.system_one   # the production client, for tests that exercise it


TOOL = {"choice": "NONE"}       # what the fake router answers to the tool question; a test sets it per case (scripted model answer)
FAITHFUL = {"choice": "SAME"}


def tool_is(choice: str) -> None:
    TOOL["choice"] = choice


def _build_state(text, ledger, **kw):  # noqa: ANN001
    _last["args"] = (text, ledger)
    return _real_build_state(text, ledger, **kw)


async def fake_system_one(state, questions, **kw):  # noqa: ANN001
    if "missing_slot" in questions:                       # the CLARIFY follow-up question (standalone call)
        return {"answers": {"missing_slot": {"type": "choice", "choice": "city" if "天气" in str((state or {}).get("utterance")) else "other"}}}
    text, ledger = _last["args"]
    route = heuristic_route(text, ledger)
    ans: dict = {"route": {"type": "choice", "choice": route}}
    if "tool" in questions:
        ans["tool"] = {"type": "choice", "choice": TOOL["choice"]}
    if "task_faithful" in questions:
        ans["task_faithful"] = {"type": "choice", "choice": FAITHFUL["choice"]}
    if route == "CONTROL":
        d = heuristic_control(text, ledger)
        if d is not None:
            ans["control"] = {"type": "choice", "choice": f"{d.op}:{d.task_id}:{d.version}"}
    return {"answers": ans}


def install() -> None:
    api.build_state = _build_state
    api.jev_client.system_one = fake_system_one
