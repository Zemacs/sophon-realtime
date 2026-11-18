"""Which requests are *tool-backed* (their answer must come from a tool result, never from the model's head)?

The Coordinator decides this, not the Worker: a weather / time / news / device / search / memory request that is
answered without the matching tool call is fabricated. ``required_tools`` is a deterministic, keyword-level
judgement (zh + en, no model), used to
  - stamp ``WorkerJob.require_tools`` (the Worker must call one of them; its first call is forced), and
  - reject a Worker answer that carries no tool evidence for a tool-scoped job (``finish_job``).
Jokes, chit-chat and general knowledge need no tool and return an empty list.
Keep the tool names in sync with ``assistant_tools.registry`` (a test enforces it).
"""
from __future__ import annotations

import re

# Realtime facts (Critic I-8): prices, quotes, FX, whether a market is open, scores, box office, traffic, "who is the current X".
# Their answer changes by the hour, so the model must not give one from its head: they are tool-backed (web_search) and,
# if the tool fails or finds nothing, the user hears an honest failure. Deliberately wide: a chit-chat question wrongly
# sent to the tool costs a lookup; a fact answered without it is a fabrication.
_MONEY = r"美元|美金|欧元|日元|英镑|港币|港元|澳元|加元|韩元|卢布|比特币|以太坊|狗狗币|BTC|ETH|USDT|黄金|白银|金价|银价|油价|原油|汽油|柴油|铜价|白糖|猪肉价"
_MARKET = r"A股|a股|港股|美股|沪指|深指|上证|深证|创业板|科创板|恒生|纳斯达克|纳指|道指|道琼斯|标普|大盘|指数|基金|净值|涨停|跌停|开盘|收盘|开市|休市|交易所|市值|股价|股票|股市|行情|涨了|跌了|涨幅|跌幅"
_SPORT_ETC = r"比分|比赛结果|谁赢了|赢了没|赛程|战绩|积分榜|世界杯|欧冠|英超|西甲|NBA|CBA|奥运|票房|热搜|榜单|航班|晚点|延误|路况|堵车|限行|地震|台风|疫情"
_PRICE = r"多少钱|什么价|价格|报价|价位|售价|汇率|兑换|换算|兑人民币|兑美元|兑[^，。！？]{1,3}(多少|汇率|换)"
_FRESH_Q = r"(今天|今日|现在|目前|当前|最新|实时|刚刚|此刻|这会儿)[^，。！？]{0,10}(多少|几点开|几多|多高|多低|涨|跌|开盘|收盘|开了吗|开门|营业|比分|排名|第几|结果|怎么样了|什么情况|发生了什么|是谁|谁)"
_CURRENT_OFFICE = r"现任|在任的|最新一[任届期]|当前的?(总统|首相|总理|主席|冠军|纪录)"
_REALTIME_FACTS = "|".join((
    rf"(?:{_MONEY})[^，。！？]{{0,8}}(?:多少|几|价|汇|涨|跌|行情|报价|兑)",    # 「比特币现在多少钱」「美元兑人民币是多少」「金价」
    rf"(?:兑|换)[^，。！？]{{0,4}}(?:{_MONEY}|人民币)", r"人民币[^，。！？]{0,6}(?:汇|兑|多少|换)",
    _MARKET, _SPORT_ETC, _PRICE, _FRESH_Q, _CURRENT_OFFICE,
    r"exchange rate|stock price|share price|bitcoin|\bbtc\b|\bnasdaq\b|dow jones|s&p 500|\bscore\b|box office|\bprice of\b|how much is|live score|market open",
))

# (tool name, pattern) in priority order; the first matching pattern wins for the *primary* tool.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("get_weather", re.compile(r"天气|下雨|下雪|下不下|气温|温度|冷不冷|热不热|带伞|穿什么|weather|rain|snow|temperature|forecast", re.I)),
    ("get_news", re.compile(r"新闻|头条|快讯|headline|\bnews\b", re.I)),
    ("get_datetime", re.compile(r"几点|现在时间|当前时间|什么时间|几号|星期几|周几|礼拜几|今天日期|今天的日期|what time|the time|what day|today'?s date|\bdate\b", re.I)),
    ("device_control", re.compile(r"(打开|关掉|关闭|关上|调高|调低|调亮|调暗|开|关)[^，。！？]{0,6}(灯|空调|窗帘|电视|风扇|插座)|(灯|空调|窗帘|电视|风扇|插座)[^，。！？]{0,3}(打开|关掉|关闭|关上|开|关)|音量|闹钟|设个提醒|设置提醒|turn (on|off)|\blights?\b|volume|alarm|curtain|\btv\b", re.I)),
    ("web_search", re.compile(r"搜索|搜一下|搜一搜|百度一下|google|\bsearch\b|" + _REALTIME_FACTS, re.I)),
    ("memory_note", re.compile(r"记住|记下|别忘了|你还?记得|上次说|之前说|我的偏好|喜欢吃什么|喜欢喝什么|remember|recall", re.I)),
)


def required_tools(text: str) -> list[str]:
    """Tool names a request needs, primary first. Empty for chit-chat / jokes / general knowledge."""
    t = (text or "").strip()
    if not t:
        return []
    out: list[str] = []
    for name, pat in _PATTERNS:
        if pat.search(t) and name not in out:
            out.append(name)
    return out


def is_tool_scoped(text: str) -> bool:
    return bool(required_tools(text))


# What the user hears while the Worker runs (the front says it before calling delegate): by intent, minimal.
_PAD_BY_TOOL = {
    "get_weather": "我帮你查一下。",
    "get_news": "我帮你查一下。",
    "web_search": "我帮你查一下。",
    "memory_note": "",
    "device_control": "好的，马上。",
    "get_datetime": "",
}


def pad_for(text: str) -> str:
    """Filler for a request by intent. 查询类 查一下; 设备 好的，马上; 时间 / 笑话 / 闲聊 none."""
    tools = required_tools(text)
    if not tools:
        return ""
    return _PAD_BY_TOOL.get(tools[0], "")
