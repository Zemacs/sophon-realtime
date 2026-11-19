"""Stable contracts between Frontend ↔ Coordinator ↔ Worker ↔ Delivery.

Keep this module free of gateway/audio/tool I/O so layers stay decoupled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from ..failcodes import SOURCE_ERROR_CODES


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    NEEDS_INPUT = "NEEDS_INPUT"
    COMPLETED = "COMPLETED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


@dataclass
class AdmitRequest:
    session_id: str
    utterance_id: str
    text: str
    language: str = "zh"
    delegate_task: str | None = None
    playback_active: bool = False
    now_ms: int = 0
    raw_text: str = ""  # what the user actually said (ASR), when known; `text` may be the front's rewrite


@dataclass
class SpeakPlan:
    kind: Literal["pad", "clarify", "answer", "stop_speaking", "noop"]
    text: str = ""
    barge_policy: Literal["keep", "drop_playback"] = "keep"
    task_id: str | None = None
    version: int | None = None


# The only sentences a *failed* WorkerResult may carry to the user's ears (besides a failed required tool's own sentence,
# which the Worker records as evidence ``failure_text``). Model free text is never spoken for a failed job (Critic I-5).
NO_TOOL_ANSWER = "这个我这边没查到准确结果，你再问我一次吧。"
FAILED_GENERIC = "这个我这边没办成，你再问我一次吧。"
FIXED_FAILURES = frozenset({
    NO_TOOL_ANSWER,
    FAILED_GENERIC,
    "后台模型暂时不可用，请稍后再试。",
    "后台模型连接失败，请稍后再试。",
    "我还在查，请稍等一下再问一次。",
    "后台编排出错了，请稍后再试。",
})


# The failure sentences OUR tools produce themselves (assistant_tools: weather / news / web_search). The Coordinator speaks a Worker-reported
# ``failure_text`` only if it is exactly one of these AND belongs to the tool that failed (Critic S-13, S-18): a Worker cannot launder a
# fabricated sentence by writing it into evidence as the "tool's own text", nor pass a web_search sentence off as the weather tool's.
# Group 1, where present, is what the tool echoes back (city / query); it must appear, as a whole word, in the user's ORIGINAL request
# (``job.objective``), never in the Worker's own ``args`` (the Worker is the party being checked).
_CITY = r"[\u4e00-\u9fffA-Za-z·\-.' ]"          # (St. Louis, Xi'an)
# I-17: the parenthesis carries <source>:<code> pairs and the code is a member of the CLOSED enum ``failcodes.SOURCE_ERROR_CODES`` (what
# weather._error_code maps an exception to), never a free token: a Worker cannot smuggle text such as ``sojson:sunny_minus3_to_11`` through it.
_WEATHER_SRC = r"(?:sojson|open-meteo|wttr):(?:" + "|".join(SOURCE_ERROR_CODES) + r")"
# S-24: the "I am saturated" sentence, one text for every outbound tool (assistant_tools.inflight.BUSY_ZH / BUSY_EN; a test keeps them equal)
BUSY_ZH_RE = r"我这边同时在查的太多了，稍等再问我一次。"
BUSY_EN_RE = r"I'm running too many lookups at once right now\. Please ask me again in a moment\."
TOOL_FAILURE_TEMPLATES: dict[str, tuple[re.Pattern[str], ...]] = {
    "get_weather": tuple(re.compile(p) for p in (
        rf"暂时查不到({_CITY}{{1,24}})的天气，请稍后再试。",
        rf"Weather for ({_CITY}{{1,40}}) is unavailable right now \((?:no source|{_WEATHER_SRC}(?:, {_WEATHER_SRC}){{0,7}})\)\.",
        rf"找不到({_CITY}{{1,24}})的天气信息。",
        rf"No weather data for ({_CITY}{{1,40}})\.",
        BUSY_ZH_RE, BUSY_EN_RE,
    )),
    "get_news": tuple(re.compile(p) for p in (
        r"新闻源暂时连不上，稍后再帮你看看头条吧。",
        BUSY_ZH_RE, BUSY_EN_RE,
        r"News feeds are unreachable right now\. Try again in a moment\.",
    )),
    "web_search": tuple(re.compile(p) for p in (
        r"网络暂时不可用，没法完成网页搜索，请稍后再试。",
        BUSY_ZH_RE, BUSY_EN_RE,
        r"Web search is unavailable right now due to a network problem\. Please try again later\.",
        r"没有搜到关于「([^」\n]{1,60})」的可靠摘要。",
        r"I couldn't find useful snippets for '([^'\n]{1,60})'\.",
    )),
}


def _said_in(echoed: str, request: str) -> bool:
    """``echoed`` occurs in ``request`` as a whole word (an ASCII edge must not sit inside a longer ASCII word; CJK has no word boundary)."""
    e = " ".join(echoed.split()).lower()
    r = " ".join((request or "").split()).lower()
    if not e:
        return False
    pat = re.escape(e)
    if re.match(r"[a-z0-9]", e[0]):
        pat = r"(?<![a-z0-9])" + pat
    if re.match(r"[a-z0-9]", e[-1]):
        pat += r"(?![a-z0-9])"
    return re.search(pat, r) is not None


def appears_in_words(value: str, words: str, *, translate: bool = False) -> bool:
    """Provenance, the one check the code may make about a model-produced value: ``value`` literally occurs, as a whole word, in ``words``
    (the user's own words or a tool's output). ``translate`` also accepts the same known place spelled the other way (北京 <-> Beijing).
    Whether ``value`` is a city, a whole name or the right place is NOT decided here: that is the model's call (B-3, B-4)."""
    names = {value} | (_city_aliases(value) if translate else set())
    return any(_said_in(n, words or "") for n in names)


def _echo_ok(echoed: str, objective: str, *, city: bool) -> bool:
    """Provenance only: what the failed tool echoes back must literally occur, as a whole word, in the user's own words (``job.objective``,
    which also holds an earlier ask and a later 改口 / 补充 of the same task). The code does NOT judge what the user meant: whether the echo
    is the whole place name, or which of several places in the words the user finally wants, was decided by the Worker model when it chose
    the tool's arguments (it reads the same objective); this check only proves the sentence is built from the user's words (B-3)."""
    return appears_in_words(echoed, objective, translate=city)


def is_tool_failure_sentence(text: str, ev: dict, objective: str = "") -> bool:
    """``text`` is one of the failed tool's own failure sentences, and what it echoes (city / query) is in the user's original request."""
    tool = str((ev or {}).get("tool") or "")
    for pat in TOOL_FAILURE_TEMPLATES.get(tool, ()):
        m = pat.fullmatch(text or "")
        if not m:
            continue
        return True if not m.groups() else _echo_ok(m.group(1), objective, city=tool == "get_weather")
    return False


def _city_aliases(name: str) -> set[str]:
    """The weather tool names a city by its canonical (Chinese) name whatever the user wrote, so the same place spelled the other way (北京 /
    Beijing) is also "the user's words". This is the weather tool's own name table used only to translate a value for the provenance check;
    it decides nothing about what the user wants (B-3)."""
    try:
        from ..assistant_tools.weather import CITIES
    except Exception:  # noqa: BLE001
        return set()
    n = " ".join(name.split()).lower()
    out: set[str] = set()
    for zh, en in CITIES.items():
        if n in (zh.lower(), en.lower()):
            out.update((zh, en))
    out.discard(name)
    return out


def tool_succeeded(ev: dict) -> bool:
    """A tool call counts as evidence only when the Worker says ``ok`` is exactly True (a missing ``ok`` is NOT success)."""
    return ev.get("ok") is True and not ev.get("missing") and not ev.get("refused")


def freshness_token_for(task_id: str, version: int) -> str:
    """The one place that spells the token: it names exactly one (task_id, version) of the Ledger."""
    return f"{task_id}:{version}"


@dataclass
class WorkerJob:
    task_id: str
    version: int
    objective: str
    language: str = "zh"
    tool_scope: list[str] = field(default_factory=lambda: ["*"])
    constraints: list[dict[str, Any]] = field(default_factory=list)
    freshness_token: str = ""
    # Tool-backed request (weather / time / news / device / search / memory): the answer MUST come from a tool result.
    # Set by the Coordinator (coordinator.intent); the Worker forces the first call, finish_job rejects a tool-less answer.
    require_tools: list[str] = field(default_factory=list)
    # Who decided ``require_tools``: "model" (the router answered; [] = it said no tool) or "unknown" (the router was unavailable or
    # gave a non-option answer). With "unknown" nothing is forced (and it is traced as such).
    tool_judgment: str = "model"
    # The router model's judgment: did the user explicitly ask to keep this long-term? ``memory_gate`` writes long-term memory only on True.
    explicit_save: bool | None = None


@dataclass
class WorkerResult:
    task_id: str
    version: int
    status: Literal["completed", "needs_input", "failed", "cancelled"]
    spoken_answer: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    needs_slots: list[str] = field(default_factory=list)
    freshness_token: str = ""  # the Worker echoes the job's token; the Coordinator refuses a result whose token is not the job's


@dataclass
class PlaybackReceipt:
    task_id: str
    version: int
    outcome: Literal["delivered", "interrupted", "abandoned"]


@dataclass
class ControlDecision:
    """A CONTROL operation aimed at one (task_id, version) of the Ledger."""

    op: Literal["CANCEL", "STATUS", "REFINE"]
    task_id: str
    version: int
    via: Literal["jev", "guard"] = "jev"

    @property
    def key(self) -> str:
        return f"{self.op}:{self.task_id}:{self.version}"


@dataclass
class TaskRecord:
    task_id: str
    version: int
    objective: str
    status: TaskStatus = TaskStatus.QUEUED
    spoken_answer: str = ""
    constraints: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)  # task-scoped slots (this task version only; never long-term)
    require_tools: list[str] = field(default_factory=list)   # what the router said this task needs (a REFINE inherits it)
    tool_judgment: str = "model"
    explicit_save: bool | None = None
    interrupted: int = 0  # how many times playback of the answer was cut off (answer stays re-playable)
