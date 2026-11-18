"""DEV set 2 (incr-5, I-10 residual / I-11): colloquial and English realtime phrasing, plus casual / task sentences that carry a time word
and a quantity or yes/no question. Written by Builder before the rule change, from phrasing families (spoken-Chinese 啥/咋/这阵子/刚才,
English "trading at / is X down / how is X doing"), not from Critic's example sentences. The earlier frozen set and Critic's 92 sentences
are NO LONGER blind (their words were used while writing rules, and Critic's 92 became a regression set): they stay as regression numbers
only. This is the set the rules are tuned on; it is not an acceptance number either."""
from __future__ import annotations

DEV2_REALTIME = (
    "黄金这阵子啥价", "美元今儿啥价", "眼下油价多少一升", "这两天恒指咋样", "昨儿夜里美股咋收的", "今早A股开盘咋样", "刚才有没有地震",
    "10号线这会儿还通着吗", "刚刚高铁是不是又晚点了", "机场现在是不是还在停飞", "杭州东站这会儿人多不多", "二环路眼下堵得厉害吗",
    "微信是不是崩了", "支付宝现在能用吗", "刚才网站是不是打不开了", "这会儿服务宕机了没",
    "湖人昨晚赢了没", "皇马今晚几点踢", "国足刚才踢得咋样", "这场比赛谁领先",
    "新款手机上市了没", "这部电影票房破十亿了吗", "最近一期双色球出了啥号", "热搜第一是啥",
    "英镑兑人民币现在啥价", "100欧元这会儿能换多少人民币", "港币今天能换多少人民币",
    "what is gold trading at right now", "what's tesla stock at today", "how is the nasdaq doing", "is the subway down right now",
    "is the highway to the airport jammed", "who is leading the match", "did the lakers win last night", "is the bank open today",
    "what's the dollar to yuan rate this morning", "has the earthquake been reported yet", "is the website down for everyone",
    "when does the match kick off tonight", "latest bitcoin price",
)
DEV2_FREE = (
    "现在能开始上课了吗", "你现在忙不忙", "今天能不能提前走", "最近睡得好吗", "这个月我花了多少钱", "今天吃了几顿饭", "我们现在几个人",
    "明天几点叫我起床", "这会儿说话方便吗", "你这阵子心情咋样", "今儿要不要带伞啊我忘了", "刚才那个笑话好笑吗", "现在是不是该休息了",
    "今天要开几个会", "帮我看看今天有几个待办", "现在有多少人在线聊天", "这次作业我拿了多少分", "咱们今天几点吃饭", "孩子今天乖不乖",
    "航班延误险是什么", "地铁是谁发明的", "动车组是哪家工厂造的", "票房是怎么计算的", "榜单一般多久更新一次呢", "股票和基金有什么区别",
    "黄金为什么贵", "欧元是哪年开始流通的", "足球场有多大", "怎么才能看懂K线图", "写一首关于地铁的诗",
    "把100加50再乘2", "一公斤等于几斤", "三百欧元加两百欧元是多少", "把摄氏30度换成华氏",
    "what time should we have dinner", "why do birds sing in the morning", "write a short poem about the subway", "how do airplanes fly",
    "what does GDP stand for", "can you help me plan my day", "do you like football", "how many days are in a leap year",
)
