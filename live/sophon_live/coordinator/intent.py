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

# Realtime facts (Critic I-8, narrowed by I-9): prices, quotes, FX, market status, scores, box office, traffic, "who is the current X".
# Their answer changes by the hour, so the model must not give one from its head: they are tool-backed (web_search) and, if the
# tool fails or finds nothing, the user hears an honest failure. A topic word alone is NOT a realtime signal (地震 / 世界杯 / 基金 /
# 汇率 / NBA ... turn up in definitions, history and chit-chat): it takes a topic + a "what is it NOW" ask (price / quote / status /
# score / alert) or a price noun, and an explanatory sentence (什么意思, 为什么, 的历史 ...) is not a lookup.
# Misclassification is measured, not guessed: tests/test_r4_intent_gate.py (labeled set in tests/intent_cases.py).
_TIME = r"今天|今日|现在|目前|当前|最新|实时|刚才|刚刚|此刻|这会儿|昨晚|昨天|今晚|最近|明天|本周|这周|周末|上一场|那场"
_ASSET = (r"美元|美金|欧元|日元|英镑|港币|港元|澳元|加元|韩元|卢布|人民币|比特币|以太坊|狗狗币|BTC|ETH|USDT|黄金|白银|原油|汽油|柴油|"
          r"基金|股票|股市|A股|a股|港股|美股|大盘|沪指|深指|上证|深证|创业板|科创板|恒生|恒指|纳斯达克|纳指|道指|道琼斯|标普|沪深300|日经|综指|成指")
_QTY = (r"多少(?:钱|点|元|块|美元|人民币|日元|欧元|英镑|港币|盎司|一[克斤升吨桶个枚股]|[了吗呢啊？?]*$)|什么价|价格|涨|跌|点位|开盘|收盘|开市|休市|"
        r"行情|走势|怎么样|情况|牌价|兑|换|汇")
_ASSET_QUOTE = rf"(?:{_ASSET})[^，。！？]{{0,10}}(?:{_QTY})"
_PRICE_NOUN = r"股价|金价|银价|油价|汇率|牌价|报价|售价|价位|开盘价|收盘价|市值|涨停|跌停|净值|多少钱|什么价|价格"
_CHARTS = r"票房|热搜|热榜|比分|赛果|比赛结果|积分榜|赛程"
_SPORT = r"NBA|CBA|欧冠|英超|西甲|德甲|意甲|法甲|中超|世界杯|奥运|F1|网球公开赛|温网|大满贯"
_SPORT_NOW = rf"(?:{_SPORT})[^，。！？]{{0,12}}(?:{_TIME})|(?:{_TIME})[^，。！？]{{0,12}}(?:{_SPORT})"
_SPORT_ASK = r"怎么样|打得|比分|战况|结果|几比几|比赛|赢|输|开赛|几点|直播|赛程"
_OUTCOME_NOW = rf"(?:{_TIME})[^，。！？]{{0,10}}(?:赢了|输了|赢没赢|谁赢|谁输|打赢|夺冠|晋级)|(?:谁赢了|谁输了|赢了吗|赢了没)[^，。！？]{{0,6}}(?:{_SPORT})|(?:谁赢了|谁输了)[^，。！？]{{0,6}}(?:{_TIME})"
_EVENT = rf"(?:{_TIME})[^，。！？]{{0,8}}(?:地震|台风|海啸|暴雨|山火|疫情)|(?:地震|台风|海啸|疫情|山火)[^，。！？]{{0,6}}(?:了吗|了没|预警|几级|多少级|登陆|最新|动态|路径)"
_TRANSPORT = (r"(?:航班|高铁|火车|列车|地铁|飞机)[^，。！？]{0,12}(?:晚点|延误|正点|取消了|到哪了|几点到|起飞了吗|还有票)|[A-Za-z]{2}\d{3,4}[^，。！？]{0,8}(?:晚点|延误|起飞|到)|"
              rf"路况|拥堵|堵车吗|堵不堵|堵吗|限不限行|限行(?:吗|尾号|规定)|(?:{_TIME})[^，。！？]{{0,6}}限行")
_OFFICE = (r"现任[^，。！？]{0,10}(?:总统|首相|总理|主席|部长|市长|省长|CEO|总裁|主帅|教练|冠军|首富|董事长|院长|校长|书记)|最新一[任届期]|"
           rf"(?:{_TIME})的?[^，。！？]{{0,6}}(?:首富|冠军|总统|首相|总理|主席|纪录保持者|榜首)")
_FRESH = r"最新(?:的)?(?:消息|动态|报道|数据|价格|行情|汇率|战况|比分)|实时(?:价格|行情|汇率|数据|比分|路况)"
_EN = (r"exchange rate|stock price|share price|box office|live score|market (?:is )?open|stock market (?:is )?open|price of|\bscore\b|who won|did the [^.?!]{0,20} win|"
       r"(?:bitcoin|btc|ethereum|eth|gold|oil|crude|dollar|euro|yen|pound)\b[^.?!]{0,20}\b(?:price|rate|worth|trading)\b|"
       r"how much is [^.?!]{0,20}(?:bitcoin|btc|dollar|euro|yen|pound|gold|oil|tesla|nvidia|stock|share)")
_REALTIME_RE = re.compile("|".join((_ASSET_QUOTE, _PRICE_NOUN, _CHARTS, rf"(?:{_SPORT_NOW})(?=.*(?:{_SPORT_ASK}))", _OUTCOME_NOW, _EVENT, _TRANSPORT, _OFFICE, _FRESH, _EN)), re.I)
# an explanatory / definitional / how-to sentence is knowledge, not a lookup, even when it carries a topic word
_STRONG_VETO = re.compile(r"什么意思|什么是|有什么(?:区别|不同|风险|影响)|区别|的历史|原理|由来|换算成|stand for|哪一年|多久[办开]|多长时间|怎么(?:换算|统计|排)")
# ...softer ones are cancelled by a "now" word ("为什么今天A股跌这么多" is still a lookup)
_WEAK_VETO = re.compile(r"为什么|如何|怎么(?!样)|是怎么|怎样(?!了)")
_NOW = re.compile(_TIME)


def is_realtime_fact(text: str) -> bool:
    t = (text or "").strip()
    if not t or not _REALTIME_RE.search(t) or _STRONG_VETO.search(t):
        return False
    return not (_WEAK_VETO.search(t) and not _NOW.search(t))


# (tool name, pattern) in priority order; the first matching pattern wins for the *primary* tool.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("get_weather", re.compile(r"天气|下雨|下雪|下不下|气温|温度|冷不冷|热不热|带伞|穿什么|weather|rain|snow|temperature|forecast", re.I)),
    ("get_news", re.compile(r"新闻|头条|快讯|headline|\bnews\b", re.I)),
    ("get_datetime", re.compile(r"几点(?!开|到|起|出|下班|上班|吃|睡|见|结束|放学|出发|降落)|现在时间|当前时间|什么时间|几号|星期几|周几|礼拜几|今天日期|今天的日期|what time|the time|what day|today'?s date|\bdate\b", re.I)),
    ("device_control", re.compile(r"(打开|关掉|关闭|关上|调高|调低|调亮|调暗|开|关)[^，。！？]{0,6}(灯|空调|窗帘|电视|风扇|插座)|(灯|空调|窗帘|电视|风扇|插座)[^，。！？]{0,3}(打开|关掉|关闭|关上|开|关)|音量|闹钟|设个提醒|设置提醒|turn (on|off)|\blights?\b|volume|alarm|curtain|\btv\b", re.I)),
    ("web_search", re.compile(r"搜索|搜一下|搜一搜|百度一下|google|\bsearch\b", re.I)),
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
    if "web_search" not in out and is_realtime_fact(t):
        out.append("web_search")
    return out


def is_tool_scoped(text: str) -> bool:
    return bool(required_tools(text))


# What the user hears while the Worker runs (the front says it before calling delegate): by intent, minimal.
_PAD_BY_TOOL = {
    "get_weather": "我帮你查一下。",
    "get_news": "我帮你查一下。",
    "web_search": "我帮你查一下。",
    "memory_note": "我帮你查一下。",
    "device_control": "好的，马上。",
    "get_datetime": "",
}


def pad_for(text: str) -> str:
    """Filler for a request by intent. 查询类 查一下; 设备 好的，马上; 时间 / 笑话 / 闲聊 none."""
    tools = required_tools(text)
    if not tools:
        return ""
    return _PAD_BY_TOOL.get(tools[0], "")
