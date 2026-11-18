"""Labeled utterances for the tool-routing classifier (intent.required_tools): what needs a realtime lookup, what does not.

Used by test_no_fabrication (named lists, zero misses allowed) and the misclassification-rate gate in test_r4_intent_gate.
The lists mix three sources, so the numbers in the gate mean something:
  OFFICIAL_*   the 29 + 19 cases of Round 4b / I-8
  CRITIC_*     the 12 realtime extras and 15 over-trigger probes of Critic incr-2 (runs/incr-2/i8_classifier.txt)
  HARDER_*     written BEFORE tuning the classifier for I-9 (first-pass numbers are in the commit message / PR body):
               negatives that carry the trigger topics (地震/世界杯/基金/汇率/油价/黄金/NBA...) in a definition / history /
               how-it-works / chit-chat sentence, and realtime asks phrased differently from the official ones.
"""
from __future__ import annotations

OFFICIAL_REALTIME = (
    "今天美元兑人民币是多少", "比特币现在多少钱", "A股开盘了吗",
    "现在金价多少", "今天油价多少", "上证指数今天涨了吗", "昨晚湖人比分是多少", "这周末英超有什么比赛", "现在日元汇率",
    "一百美元能换多少人民币", "特斯拉股价", "今天的票房冠军是谁", "北京到上海的航班晚点吗", "现在堵车吗", "现任美国总统是谁",
    "纳斯达克收盘了没有", "iPhone 17 多少钱", "苹果最新市值多少", "今天有什么热搜", "港股今天开市吗", "明天限行吗",
    "今天欧元兑人民币多少", "美元现在对人民币什么价", "黄金今天跌了没有", "茅台现在股价", "最新一任奥运冠军是谁",
    "what's the bitcoin price", "dollar to yuan exchange rate today", "did the Lakers win, what's the score",
)
OFFICIAL_FREE = (
    "讲个笑话", "你好", "你叫什么名字", "给我唱首歌", "现在心情怎么样", "今天过得怎么样", "帮我翻译一下这句话", "一加一等于几",
    "李白是哪个朝代的", "水的沸点是多少度", "光速是多少", "地球到月亮多远", "帮我写首诗", "你会做什么", "谢谢你", "再说一遍",
    "谁发明了电灯", "北京是中国的首都吗", "灯为什么会亮",
)
CRITIC_REALTIME = (
    "现在比特币什么价了", "人民币兑美元今天什么价", "今天美股收盘了吗", "黄金实时价格", "沪指现在多少点", "今晚NBA打得怎么样了",
    "特斯拉现在什么价", "一百人民币能换多少日元", "油价现在涨了没有", "今天恒生指数多少", "谁赢了昨晚那场欧冠", "北京今天限行尾号是多少",
)
CRITIC_FREE = (
    "今天是谁的生日", "今天谁来吃饭", "现在心情怎么样了", "地震是怎么形成的", "世界杯的历史是什么", "台风是怎样形成的",
    "厘米怎么换算成英寸", "疫情是什么时候开始的", "幸福指数是什么意思", "光速现在是多少", "今天过得怎么样了", "帮我把英寸换算成厘米",
    "NBA是什么意思", "基金和股票有什么区别", "你会不会下棋谁赢了都行",
)
HARDER_REALTIME = (
    "现在一美元能换多少人民币", "欧元对人民币汇率今天多少", "比特币今天涨了还是跌了", "以太坊现在多少钱一个", "今天白银价格怎么样",
    "92号汽油今天多少钱一升", "现在上证综指多少点", "创业板今天涨了吗", "深证成指今天收盘多少", "美股现在开盘了没有",
    "港股今天涨跌怎么样", "英伟达股价现在多少", "阿里巴巴今天股价", "这只基金今天净值多少", "沪深300今天怎么样",
    "昨天欧冠决赛比分是多少", "湖人今天赢了吗", "世界杯今天有哪些比赛", "中超今晚几点开赛", "今天电影票房排第几",
    "今天微博热搜第一是什么", "现在去机场路上堵不堵", "CA1501航班今天晚点吗", "今天北京限行吗", "刚才哪里地震了",
    "最近有台风预警吗", "现任日本首相是谁", "现在的世界首富是谁", "今天黄金多少钱一克", "苹果现在的股价",
    "what is the dollar to euro rate today", "price of bitcoin right now", "who won the NBA game last night", "is the stock market open today",
)
HARDER_FREE = (
    "地震的时候应该躲在哪里", "台风为什么有名字", "世界杯是哪一年开始的", "基金是什么", "股票是怎么回事", "股市有什么风险",
    "汇率是什么意思", "什么是通货膨胀", "油价为什么会涨", "黄金为什么值钱", "比特币是什么", "以太坊和比特币有什么不同",
    "NBA有多少支球队", "足球比赛一场多长时间", "奥运会多久办一次", "谁是历史上最伟大的篮球运动员", "中国有多少个省", "美元是哪个国家的货币",
    "一公里等于多少米", "一斤是多少克", "摄氏度怎么换成华氏度", "一年有多少天", "今天我想吃点好的",
    "现在我有点累", "你今天开心吗", "帮我想个今晚吃什么的主意", "你知道苹果为什么是红色的吗", "教我怎么炒股", "怎么看K线图",
    "指数函数是什么", "指数基金定投是什么意思", "疫情对经济有什么影响", "航班延误怎么索赔", "高铁为什么这么快",
    "路况信息是怎么收集的", "热搜是怎么排的", "票房是怎么统计的", "总统是怎么选出来的", "首相和总理有什么区别",
    "what is bitcoin", "how do earthquakes happen", "what does NBA stand for", "how much is five plus five", "tell me a joke",
)

REALTIME = OFFICIAL_REALTIME + CRITIC_REALTIME + HARDER_REALTIME
FREE = OFFICIAL_FREE + CRITIC_FREE + HARDER_FREE

# Written after the classifier was tuned on the sets above, before looking at its answers (first pass on them: realtime 0/20
# missed, free 2/20 over-triggered: 「今天天气好适合打球」 -> get_weather (the old weather keyword, not I-8/I-9) and
# 「限行对环境有什么好处」 -> web_search, which was then fixed, so this set is no longer blind).
HELDOUT_REALTIME = (
    "今天人民币对日元汇率是多少", "现在美元涨了吗", "今天茅台收盘价多少", "沪指今天收在多少点", "波音股价现在怎么样", "今晚欧冠有哪些比赛",
    "勇士昨晚赢了吗", "刚才有地震吗", "今天北京限不限行", "现在三环堵不堵", "今天最新的疫情数据", "最新比特币行情", "现任英国首相是谁",
    "现在汽油多少钱一升", "今天电影票房第一是谁", "MU5137航班现在晚点了吗", "今天港股收盘了吗", "一万日元现在合多少人民币",
    "今天金价是涨是跌", "纳指现在多少点",
)
HELDOUT_FREE = (
    "地震波是怎么传播的", "你喜欢看NBA吗", "世界杯的冠军奖杯长什么样", "股票代码是什么意思", "黄金的化学符号是什么", "为什么人民币叫元",
    "美元上的头像是谁", "我今天想去看球赛", "今天天气好适合打球", "汽油和柴油有什么不同", "给我讲讲比特币的原理", "限行对环境有什么好处",
    "航班上能带充电宝吗", "火车上怎么睡觉比较舒服", "我想学英语怎么提高", "现在我想听歌", "今天的作业怎么写", "谁是你的创造者",
    "帮我算一下100乘以3", "基金经理是做什么的",
)
