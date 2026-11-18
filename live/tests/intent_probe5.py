"""PROBE set (incr-5), written after the DEV2-driven rule change was finished and run exactly once on it (counts only) before being pinned
by tests/intent_probe5.sha256. Builder-authored, so "independent" only in the sense that no rule was tuned on it; it is NOT Critic-blind."""
from __future__ import annotations

PROBE_REALTIME = (
    "纳斯达克这会儿涨了还是跌了", "人民币对美元今天中间价报多少", "上海这会儿的地铁是不是又停了", "刚才首都机场有航班取消吗", "沪深300今天收盘多少点",
    "今儿的油价调了没", "湖人和勇士今晚几点开打", "刚才那场球谁赢了", "高速上这会儿堵不堵", "淘宝现在是不是登录不了",
    "这两天金价咋样", "A股今天成交额破万亿了吗", "最新一期大乐透开奖号码是啥", "欧元对人民币眼下啥价", "今天的票房冠军是哪部电影",
    "刚才北京是不是地震了", "现在去虹桥的路上堵吗", "这阵子比特币涨得咋样", "今晚欧冠有哪几场", "高铁G101今天晚点了没",
    "what is the euro trading at this afternoon", "is the metro running late today", "how is the stock market doing this morning",
    "is the bridge closed right now", "what's the latest score of the match", "who is ahead in the game tonight", "is there a flight delay at the airport now",
    "what's bitcoin at right now", "has the storm hit the coast yet", "what time does the game start tonight",
)
PROBE_FREE = (
    "今天几点下班我都忘了", "现在你想聊点什么", "你最近有什么新鲜事吗", "今天心情怎么样啊", "这周末你有什么计划", "咱们现在去哪吃饭",
    "他今天还来吗", "今天作业多不多", "我最近是不是胖了", "现在可以关电视了吗", "明天要不要早起", "你知道今天是星期几吗",
    "这个月的房租交了吗", "今天有几节课来着", "现在几点该吃药了", "孩子最近乖不乖", "咱家今天来了几个客人", "刚才你说什么",
    "地铁怎么刷码进站", "怎么买火车票最便宜", "股票开户需要什么材料", "机场安检能带多少液体", "为什么飞机会晚点一般是什么原因", "票房和口碑有关系吗",
    "give me a pep talk for today", "how do I get to the airport by subway", "what makes the stock market go up in general", "can you tell me a story about a train",
    "I am feeling tired right now", "is it rude to be late to a meeting",
)
