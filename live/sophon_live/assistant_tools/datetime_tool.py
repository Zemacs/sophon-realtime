"""Current date/time tool."""
from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


async def run(args: dict[str, Any]) -> str:
    what = (args.get("what") or "both").lower()
    tz_name = (args.get("timezone") or "").strip()
    now: datetime
    if tz_name and ZoneInfo is not None:
        try:
            now = datetime.now(ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001
            now = datetime.now().astimezone()
            tz_name = ""
    else:
        now = datetime.now().astimezone()
    zh = (not tz_name) or tz_name.startswith("Asia/") or "Shanghai" in tz_name or "Hong_Kong" in tz_name
    wd = "一二三四五六日"[now.weekday()]
    if what == "time":
        return f"现在是{now.hour}点{now.minute:02d}分。" if zh else f"It's {now.strftime('%I:%M %p').lstrip('0')}."
    if what == "date":
        return f"今天是{now.month}月{now.day}日，星期{wd}。" if zh else now.strftime("Today is %A, %B %d.")
    if zh:
        return f"现在是{now.month}月{now.day}日星期{wd}，{now.hour}点{now.minute:02d}分。"
    return now.strftime("It's %A, %B %d, ") + now.strftime("%I:%M %p").lstrip("0")
