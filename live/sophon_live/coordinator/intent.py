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
_CCY_CODE = r"(?<![A-Za-z])(?:USD|CNY|RMB|CNH|EUR|JPY|GBP|HKD|AUD|CAD|KRW|CHF|SGD)(?![A-Za-z])"
_ASSET = (_CCY_CODE + r"|美元|美金|欧元|日元|英镑|港币|港元|澳元|加元|韩元|卢布|人民币|比特币|以太坊|狗狗币|BTC|ETH|USDT|黄金|白银|原油|汽油|柴油|"
          r"基金|股票|股市|A股|a股|港股|美股|大盘|沪指|深指|上证|深证|创业板|科创板|恒生|恒指|纳斯达克|纳指|道指|道琼斯|标普|沪深300|日经|综指|成指")
_QTY = (r"多少(?:钱|点|元|块|美元|人民币|日元|欧元|英镑|港币|盎司|一[克斤升吨桶个枚股]|[了吗呢啊？?]*$)|什么价|价格|涨|跌|点位|开盘|收盘|开市|休市|"
        r"行情|走势|怎么样|情况|牌价|兑|换|汇|折合|折算|折成")
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
_EN_CCY = r"usd|cny|rmb|eur|jpy|gbp|hkd|dollars?|euros?|yen|pounds?|yuan|renminbi|bitcoin|btc"
_EN = (r"exchange rate|convert [^.?!]{0,40}\b(?:" + _EN_CCY + r")\b|\bin (?:" + _EN_CCY + r")\b|stock price|share price|box office|live score|market (?:is )?open|stock market (?:is )?open|price of|\bscore\b|who won|did the [^.?!]{0,20} win|"
       r"(?:bitcoin|btc|ethereum|eth|gold|oil|crude|dollar|euro|yen|pound)\b[^.?!]{0,20}\b(?:price|rate|worth|trading)\b|"
       r"how much is [^.?!]{0,20}(?:bitcoin|btc|dollar|euro|yen|pound|gold|oil|tesla|nvidia|stock|share)")
_REALTIME_RE = re.compile("|".join((_ASSET_QUOTE, _PRICE_NOUN, _CHARTS, rf"(?:{_SPORT_NOW})(?=.*(?:{_SPORT_ASK}))", _OUTCOME_NOW, _EVENT, _TRANSPORT, _OFFICE, _FRESH, _EN)), re.I)
# an explanatory / definitional / how-to sentence is knowledge, not a lookup, even when it carries a topic word
_STRONG_VETO = re.compile(r"什么意思|什么是|有什么(?:区别|不同|风险|影响)|区别|的历史|原理|由来|stand for|哪一年|我[^，。！？]{0,4}有多少钱|多久[办开]|多长时间|怎么(?:统计|排)")
# unit conversion (厘米→英寸, 斤→克, 一盎司黄金→克) is arithmetic, not a lookup. It stays one only when a CURRENCY / coin (美元 / USD /
# 比特币 ...) or an explicit price word (多少钱 / 价格) is in the sentence: 黄金 / 白银 / 原油 alone are goods with a weight or volume,
# not a currency (S-20): 把一盎司黄金换算成克 is free, 一克黄金折合多少人民币 / 一克黄金多少钱 is a lookup.
_UNIT_CONVERT = re.compile(r"换算|折合|折算|折成|等于(?:多少|几)")
_MONEYISH = re.compile(_CCY_CODE + r"|美元|美金|欧元|日元|英镑|港币|港元|澳元|加元|韩元|卢布|人民币|比特币|以太坊|狗狗币|BTC|ETH|USDT|多少钱|价格|什么价|值多少钱|卖多少")
# ...softer ones are cancelled by a "now" word ("为什么今天A股跌这么多" is still a lookup)
_WEAK_VETO = re.compile(r"为什么|如何|怎么(?!样)|是怎么|怎样(?!了)")
_NOW = re.compile(_TIME)

# I-10 fallback (the topic lists above can never be complete: Critic's independent sentences missed 5/38 and a held-out set of our own
# 22/57). Two broader routes that PREFER to search when the sentence is realtime-ish but not recognised by a topic pattern. A wrong
# search costs ~2-4 s and an answer the user can verify; a wrong "from memory" answer is a confident stale number, so the cost is lopsided.
#   (E) a realtime-prone ENTITY (market / sports / traffic / alert / office-holder / chart / product words) + a value or status QUESTION;
#   (F) a "now" word + a QUANTITY / status question about anything that is not the user themself, arithmetic or a physical constant.
# Both are cancelled by: an explanatory / opinion / creative / personal sentence, a stable-fact marker (哪年 / 成立 / 最先 / 总部 ...),
# the veto above. The primary routes (topic + ask) are unchanged and are not subject to the new vetoes.
_NOW_WIDE = _TIME + r"|今年|本月|这个月|本季度|近期|眼下|当下|这次|这期|这一站|上周|上个月|上月|至今|最近一"
_ENTITY = (r"汇率|利率|中间价|买入价|卖出价|LPR|降息|加息|准备金|国债|收益率|股价|股指|成交额|成交量|两市|沪深|北向|南向|涨停|跌停|净值|市值|财报|营收|ETF|"
           r"港元|离岸|在岸|金价|银价|油价|汽油|柴油|原油|布伦特|期货|比特币|以太坊|美股|港股|A股|道指|纳指|标普|恒指|沪指|上证|深证|大盘|基金|股票|"
           r"比分|赛程|比赛|开赛|联赛|杆位|排位赛|决赛|夺冠|冠军|晋级|战绩|积分榜|射手榜|世界杯|欧冠|英超|西甲|中超|NBA|CBA|F1|温网|公开赛|奥运|球赛|对阵|"
           r"限号|限行|尾号|路况|堵车|拥堵|晚点|延误|航班|高铁|火车|动车|地铁|班次|停运|余票|车票|机票|"
           r"地震|台风|寒潮|暴雨|暴雪|预警|海啸|山火|疫情|"
           r"总统|首相|总理|主席|秘书长|部长|市长|省长|总裁|CEO|得主|首富|榜首|纪录保持者|"
           r"奥斯卡|诺贝尔|格莱美|金球奖|艾美奖|金鸡奖|金马奖|最佳影片|最佳男|最佳女|热搜|热榜|票房|上映|开奖|彩票|双色球|大乐透|发售|售价|零售价|报价|多少钱|销量|发布会|失业率|CPI|PPI|GDP|PMI|通胀")
_QUESTION = r"多少|几[点号位个只条场度步名分级轮]?|吗|了没|是谁|谁|哪[个位只队场部家款里儿]|怎么样|什么价|什么结果|什么时候|有没有|有票|是涨是跌|涨还是跌|第几|是什么|什么"
_QTY_VALUE = r"多少|几[点号位个只条场度分级轮名]?|了吗|了没|第几|什么价|是涨是跌|涨还是跌|怎么样|谁赢|赢了|有票|有没有"   # asks for a value / a status
_QTY_NOW = r"多少|几[点号位个只条场度分级轮名]?|了吗|吗|了没|第几|什么价|是涨是跌|涨还是跌|谁赢|赢了"   # (F) no entity: only a quantity / yes-no (a bare 现在怎么样 is chit-chat)
_PERSONAL = re.compile(r"你(?:觉得|认为|喜欢|会|能|知道|想|的|在|是|有|吗)|(?<![帮告诉给替跟让请烦])我(?!想知道|想问|想查|想看看|想了解|要知道|要问|要查|问|查|想确认)|咱|心情|开心|难过|过得|累|困|饿|无聊|"
                       r"帮我(?:写|想|编|翻译)|教我|给我(?:讲|写|推荐)|推荐|故事|顺口溜|造句|诗|歌词|笑话|走了|睡了|吃了|喝了|花了|听到|听见|"
                       r"write|poem|story|joke|song|recommend|my |i am|i'm|do you", re.I)
_STABLE = re.compile(r"哪年|哪一年|什么时候(?:成立|开始|发明|出现|诞生|创立)|成立|创立|创始|设立|发明|诞生|起源|由来|历史|总部|首都|最先|最早|第一届|第一个|吉祥物|全称|缩写|简称|"
                     r"组成|分为|几种|原理|概率|计算|代表什么|叫什么|长什么样|颜色|任期|一般|通常|为什么|有哪些|"
                     r"光速|音速|圆周率|沸点|冰点|熔点|常数|半径|直径|周长|体积|why|history|called|stand for|invented|founded|how many players", re.I)
_ARITH = re.compile(r"[0-9一二三四五六七八九十百千万两]+[^，。！？]{0,3}(?:加|减|乘|除|等于|\+|-|\*|/)|摄氏|华氏|公里|厘米|公斤|千克|英寸|英里|盎司|桶|小时|分钟|秒")
_ENTITY_RE = re.compile(_ENTITY, re.I)
_QUESTION_RE = re.compile(_QUESTION)
_NOW_WIDE_RE = re.compile(_NOW_WIDE)
_QTY_NOW_RE = re.compile(_QTY_NOW)
_QTY_VALUE_RE = re.compile(_QTY_VALUE)
_COUNT_WEAK = re.compile(r"有多少|共有|一共有|总共有|总共")      # "how many are there" is a stable count unless a "now" word is present
_EN_NOW = re.compile(r"\b(?:right now|today|tonight|currently|current|latest|now|this (?:week|month|year)|last night)\b", re.I)
_EN_ENTITY = re.compile(r"\b(?:price|score|rate|rates|stock|stocks|market|nasdaq|dow|s&p|bitcoin|btc|ethereum|president|prime minister|champion|standings|forecast|traffic|flight|open|closed|trading|winning|won|results?)\b", re.I)


def _fallback_realtime(t: str) -> bool:
    """I-10: realtime-ish but not matched by a topic pattern: prefer to search (see the block comment above)."""
    if _PERSONAL.search(t) or _STABLE.search(t) or _STRONG_VETO.search(t):
        return False
    if _WEAK_VETO.search(t) and not _NOW_WIDE_RE.search(t):
        return False
    if _UNIT_CONVERT.search(t) or _ARITH.search(t):
        return False
    if _COUNT_WEAK.search(t) and not _NOW_WIDE_RE.search(t):
        return False
    if _ENTITY_RE.search(t) and _QUESTION_RE.search(t) and (_NOW_WIDE_RE.search(t) or _QTY_VALUE_RE.search(t)):
        return True                                                                  # (E)
    if _NOW_WIDE_RE.search(t) and _QTY_NOW_RE.search(t):
        return True                                                                  # (F)
    return bool(_EN_NOW.search(t) and _EN_ENTITY.search(t))                          # English, same two halves


def _primary_realtime(t: str) -> bool:
    if not _REALTIME_RE.search(t) or _STRONG_VETO.search(t):
        return False
    if _UNIT_CONVERT.search(t) and not _MONEYISH.search(t):
        return False                                    # unit conversion is arithmetic, never a lookup
    currency_conversion = bool(_UNIT_CONVERT.search(t) and _MONEYISH.search(t))     # 「美元怎么换算成人民币」 wants today's rate
    return not (_WEAK_VETO.search(t) and not _NOW.search(t) and not currency_conversion)


def is_realtime_fact(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and (_primary_realtime(t) or _fallback_realtime(t))


# (tool name, pattern) in priority order; the first matching pattern wins for the *primary* tool.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("get_weather", re.compile(r"天气|下雨|下雪|下不下|气温|温度|冷不冷|热不热|带伞|穿什么|weather|rain|snow|temperature|forecast", re.I)),
    ("get_news", re.compile(r"新闻|头条|快讯|headline|\bnews\b", re.I)),
    ("get_datetime", re.compile(r"几点(?!开|到|起|出|有|比|赛|播|直|下班|上班|吃|睡|见|结束|放学|出发|降落)|现在时间|当前时间|什么时间|几号|星期几|周几|礼拜几|今天日期|今天的日期|what time|the time|what day|today'?s date|\bdate\b", re.I)),
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
    if "web_search" not in out and (_primary_realtime(t) or (not out and _fallback_realtime(t))):   # the fallback never adds to a request a specific tool already owns (明天会下雨吗 is weather)
        out.append("web_search")
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
