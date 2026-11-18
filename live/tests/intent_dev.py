"""DEV set for the I-10 rule changes: the only labeled sentences that were looked at while tuning (written after the frozen set,
before the rule change; the frozen set and Critic's independent 92 were only run once, at the end, for aggregate numbers)."""
from __future__ import annotations

DEV_REALTIME = (
    "今天上证指数收盘多少", "现在美元对日元是多少", "今晚几点有足球比赛", "今天深圳限号是几", "现在高速上堵不堵", "最新的LPR报价是多少",
    "今天晚上的欧冠谁对谁", "上周六的NBA总决赛谁赢了", "这个月的汽油价格涨了吗", "今天的黄金价格是多少一克", "港股今天成交额多少", "现在比特币是涨是跌",
    "今年的奥斯卡最佳影片是什么", "现在的美国总统是谁", "今天的头条热搜是什么", "新款iPhone现在多少钱", "刚刚有没有发生地震", "今天有台风吗",
    "航班MU5101现在到哪了", "这趟高铁今天晚点多久", "现在去杭州的火车还有票吗", "今天的彩票开奖号码是多少", "最新的失业率是多少", "上个月的CPI是多少",
    "今天人民币汇率中间价是多少", "现在欧元兑美元多少", "今天恒指涨了多少点", "现在原油价格是多少", "最近一场世界杯比赛的比分是多少", "今天的火箭队比赛赢了没",
    "what's the price of ethereum now", "is the New York stock exchange open today", "what is the latest score of the Lakers game", "who is the current president of France",
    "convert 100 USD to CNY at today's rate", "how much is 50 euros in dollars right now",
    "一百欧元折合成人民币是多少", "USD兑CNY今天多少", "把五百港币兑换成人民币是多少",
)
DEV_FREE = (
    "今天几号心情最好", "现在你能听到我说话吗", "今天晚上吃什么好呢", "你今天过得开心吗", "我现在有多少钱啊我也不知道", "今天走了多少步我都没数",
    "上证指数是哪家公司编制的", "美元为什么叫美元", "足球比赛一般有几个裁判", "限号是为了解决什么问题", "高速公路是谁发明的", "LPR是哪几个字母的缩写",
    "世界杯的吉祥物有哪些", "NBA的全称是什么", "黄金是什么颜色的", "比特币的创始人是谁", "国债是国家借的钱吗", "奥斯卡奖杯长什么样",
    "总统的任期一般是几年", "地震发生时人该怎么办", "台风眼里为什么是晴天", "航班晚点了应该怎么投诉", "火车票怎么抢比较容易", "彩票中奖的概率是多少",
    "失业率是怎么计算的", "CPI代表什么", "帮我写首关于股票的诗", "我想学习外汇交易有什么书推荐", "推荐一部好看的足球电影", "我爷爷以前是火车司机",
    "把一盎司黄金换算成克", "一吨原油等于多少桶", "一桶汽油折合多少升", "五公斤白银是多少克", "一英寸等于几厘米", "一小时等于多少秒",
    "what is the dollar sign called", "why is the sky blue", "write a poem about football", "how many players are on a basketball team",
)
