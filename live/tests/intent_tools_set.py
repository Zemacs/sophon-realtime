"""Per-tool labeled requests (written for rc2, labels first). Used only by tests/eval_tool_judgment.py to MEASURE whether the router
model picks the right tool class; never used as a rule. TOOL_SET[option key] = utterances that should get that option."""
from __future__ import annotations

TOOL_SET: dict[str, tuple[str, ...]] = {
    "get_weather": ("明天上海会下雨吗", "北京今天多少度", "后天要带伞吗", "广州这周末热不热", "成都现在冷不冷", "杭州明天天气怎么样", "出门需要穿外套吗今天", "深圳今晚会不会下大雨"),
    "get_news": ("今天有什么科技新闻", "给我念几条头条", "来点财经快讯", "最近有什么国际新闻", "播报一下今天的体育新闻", "有什么新鲜事报给我听听"),
    "get_datetime": ("现在几点了", "今天几号", "今天是星期几", "现在是什么时间", "今天的日期是多少", "明天礼拜几"),
    "device_control": ("把客厅的灯关掉", "打开卧室空调", "把窗帘拉上", "电视调大声一点", "关掉书房的灯", "风扇开一下", "把灯调暗一些", "打开插座"),
    "memory_note": ("帮我记住我下周三要体检", "你还记得我喜欢吃什么吗", "我上次跟你说过我的生日是哪天来着", "记一下我对花生过敏", "别忘了提醒我周五交周报"),
    "web_search": ("搜索一下最近的显卡价格", "今天美元兑人民币多少", "现在比特币什么价", "上证指数今天收盘多少", "今晚湖人比赛比分多少", "上海今天限行尾号是多少", "帮我上网搜一下附近好评的火锅店", "今天沪深两市开盘了吗"),
    "NONE": ("给我讲个笑话", "你好呀", "谢谢你", "三加五等于几", "李白是哪个朝代的人", "光速是多少", "我今天心情不太好", "帮我想一个周末的活动", "为什么天空是蓝色的", "用一句话解释什么是通货膨胀"),
}
