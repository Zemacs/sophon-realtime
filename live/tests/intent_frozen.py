"""FROZEN held-out set for the realtime-fact router (Critic I-10). Do not tune the classifier on this file, and do not edit it.

Written and committed BEFORE any rule change made for I-10 (commit message of the commit that added it carries the classifier's
numbers on it, taken from the unchanged 4b5ec1b classifier). tests/intent_frozen.sha256 pins its content: a test fails if this
file changes, so it cannot quietly be "fixed" to match the rules. Labels: REALTIME = answer changes by the day/hour or needs a
current value, must be routed to web_search; FREE = knowledge, chit-chat, arithmetic, creative or personal, must NOT be.
The sentences were written for breadth (many families, ordinary phrasing, each family on both sides of the line), without running
them through the classifier first.
"""
from __future__ import annotations

FROZEN_REALTIME = (
    # FX / rates / commodities
    "港币对人民币的中间价今天是多少", "今天招商银行美元现汇买入价多少", "现在一欧元兑多少英镑", "离岸人民币今天收盘多少", "LPR利率这个月是多少",
    "央行这次降息了吗", "美联储昨晚的利率决议是什么结果", "现在一克黄金卖多少钱", "布伦特原油现在报多少", "今天国内汽油价格调整了没有",
    "现在一盎司白银多少美元", "今天的国债收益率是多少", "银行现在的一年期定期利率多少",
    # equities / funds / crypto
    "今天沪深两市成交额是多少", "北向资金今天净流入多少", "宁德时代今天收盘价是多少", "腾讯现在的股价是多少港元", "标普500昨晚收了多少点",
    "今天哪些股票涨停了", "狗狗币现在什么价", "易方达蓝筹今天净值多少", "纳斯达克指数今天是涨还是跌", "今天A股成交量怎么样", "苹果公司最新财报营收是多少",
    # sports / lottery
    "昨晚皇马比赛的比分是多少", "今天中超有哪几场比赛", "湖人最近一场比赛赢了没有", "世界杯决赛是谁拿了冠军", "今晚中国队的比赛几点开始",
    "双色球昨晚开奖号码是多少", "F1这一站的排位赛谁拿了杆位", "今天网球公开赛决赛谁赢了",
    # city / traffic / transport / weather-like alerts
    "杭州今天限号是多少", "成都今天尾号限行吗", "现在北京到上海高铁G1还有票吗", "首都机场现在大面积延误吗", "东三环现在堵得厉害吗", "今天有没有台风要登陆广东",
    "刚才是不是哪里发生地震了", "最近一周有没有寒潮预警", "今天地铁二号线正常运营吗",
    # people / rankings / charts / products
    "现在的联合国秘书长是谁", "今年的诺贝尔和平奖得主是谁", "这周电影票房榜第一是哪部", "今天抖音热榜第一条是什么", "现在日本首相是哪位", "最新一期全球富豪榜第一是谁",
    "iPhone最新款现在官网卖多少钱", "特斯拉Model Y现在的售价是多少", "今天茅台飞天53度的零售价多少", "这个月的猪肉价格是涨还是跌",
    # english
    "what's the euro to dollar exchange rate right now", "how is the Nasdaq doing today", "what was the score of last night's Warriors game",
    "is the Tokyo stock exchange open right now", "what's the current price of gold per ounce", "who is the current prime minister of the UK",
)

FROZEN_FREE = (
    # definitions / explanations
    "中间价是什么意思", "LPR是怎么形成的", "央行为什么要降息", "什么是国债收益率", "成交额和成交量有什么区别", "涨停板制度是怎么来的",
    "黄金为什么能保值", "原油是怎么炼成汽油的", "基金定投适合什么样的人", "股票的市盈率怎么计算", "比特币挖矿的原理是什么", "期权和期货有什么不同",
    # history / stable facts
    "世界杯是哪个国家最先举办的", "第一届奥运会在哪里举行", "足球比赛为什么有加时赛", "F1赛车为什么那么快", "双色球是哪年开始发行的", "纳斯达克是哪年成立的",
    "联合国总部设在哪个城市", "诺贝尔奖是谁设立的", "日本首都是哪里", "英国有几个组成部分", "珠穆朗玛峰有多高", "太阳到地球的距离是多少",
    "台风和飓风是同一种东西吗", "地震波分为哪几种", "高铁的最高设计时速是多少", "飞机为什么能飞起来", "寒潮是怎么形成的", "票房分账是怎么算的",
    # how-to / personal / chit-chat with topic words
    "我想学炒股应该先看什么书", "你觉得买黄金划算还是买基金划算", "我昨天梦到我中了双色球", "我今天打算去看足球比赛", "帮我写一段关于股市的顺口溜",
    "给我讲讲美联储的故事", "我爸爸喜欢看世界杯", "我现在有点堵车的烦躁", "今天心情不错想听音乐", "你现在在做什么", "现在是不是该睡觉了呢", "今天好无聊啊",
    # arithmetic / unit / language / creative
    "一百加两百等于多少", "三公里等于多少米", "把华氏一百度换成摄氏度", "一盎司等于多少克", "二十乘以十五是多少", "帮我翻译成英文：我爱你", "用成语造一个句子",
    "给我推荐几部好看的电影", "夸夸我吧", "你的名字是谁取的", "你会唱歌吗", "春天适合做什么运动", "怎么煮一碗好吃的面条", "感冒了怎么办",
    # english
    "what does exchange rate mean", "why do stock markets crash", "how do airplanes stay in the air", "who invented the telephone", "tell me a bedtime story",
)
