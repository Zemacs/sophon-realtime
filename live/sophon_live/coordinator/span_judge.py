"""Independent check of a grounding claim (Critic N-2 / N-3): a SECOND model says yes or no, the code only enforces the answer.

The front model that fills a tool call can write ``grounding_evidence`` for any value it invented, so its own statement proves nothing. Before
that evidence releases a call, the decision model (JEV, a different model from the front) is asked one closed question per claim:

  equivalent  does the caller's span say this value? (same number / name, homophone, traditional spelling, short paraphrase; 八十八 and 88 are
              the same number, 张江路 does not say 88)
  confirms    is the caller's reply a confirmation of what the assistant read back? (「不对」 and an unrelated sentence are not)

Failure is a refusal: the service down, slow (``model_client`` budget + breaker), an answer that is not one of the options, or an exception all give
``None``, and ``grounding.check`` refuses the call. Nothing in this module reads the span or the value for meaning; it only builds the question.
"""
from __future__ import annotations

import logging

from . import jev_client

log = logging.getLogger("sophon_live.coordinator.span_judge")

KEY = "span_check"
STATS = {"yes": 0, "no": 0, "unavailable": 0}      # published on /health?detail=1; an "unavailable" rate that climbs means the evidence path is closed


def build_question(kind: str) -> dict:
    if kind == "confirms":
        return {
            "type": "choice",
            "instructions": "assistant_read_back 是助手刚向来电者复述的内容，caller_reply 是来电者紧接着说的话。来电者是不是在确认助手复述的内容是对的？",
            "criteria": {
                "YES": "来电者明确认可（对、是的、没错、就是这个……）",
                "NO": "来电者否认、纠正、没听清、问别的事，或者没有表态",
            },
        }
    return {
        "type": "choice",
        "instructions": "caller_words 是来电者（或工具结果）里的一段原话，value 是助手准备填进工具参数的值。这段原话说的就是这个值吗？"
                        "同一个数（八十八和88）、同一个名字或它的同音字/繁体写法、意思一致的简短转述都算 YES；原话里没有这个值的信息算 NO。",
        "criteria": {
            "YES": "原话说的就是这个值（只是写法或说法不同）",
            "NO": "原话没有说这个值，或者说的是另一个值",
        },
    }


async def judge(kind: str, argument: str, value: str, span: str, context: str = "") -> bool | None:
    """True / False = the model's yes / no; None = no usable answer (treated as a refusal by the caller)."""
    state: dict = {"argument": argument, "value": value}
    if kind == "confirms":
        state.update({"assistant_read_back": context, "caller_reply": span})
    else:
        state["caller_words"] = span
    try:
        resp = await jev_client.system_one(state, {KEY: build_question(kind)})
    except Exception as e:  # noqa: BLE001  (breaker open, timeout, connection, HTTP error: all the same to the guard)
        STATS["unavailable"] += 1
        log.warning("span check unavailable (%s): %r", kind, e)
        return None
    ans = jev_client.choice_answer(resp, KEY)
    if ans == "YES":
        STATS["yes"] += 1
        return True
    if ans == "NO":
        STATS["no"] += 1
        return False
    STATS["unavailable"] += 1
    log.warning("span check: not an option: %r", ans)
    return None
