"""v0.1.0 / Critic S-13: failure_text is only spoken when it has the shape of a real tool failure sentence."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))

from sophon_live.assistant_tools import news, web_search
from sophon_live.coordinator import Ledger, WorkerJob, WorkerResult, api, finish_job
from sophon_live.coordinator.schemas import NO_TOOL_ANSWER, TaskStatus

FAKE = "北京晴，零下三到十一度。"


def _finish(tool: str, said: str, ev_text: str | None, task: str = "明天北京会下雨吗"):
    led = Ledger()
    rec = led.create(task)
    led.mark_running(rec.task_id)
    job = WorkerJob(task_id=rec.task_id, version=rec.version, objective=rec.objective, require_tools=[tool])
    ev = {"tool": tool, "ok": False}
    if ev_text is not None:
        ev["failure_text"] = ev_text
    return led, rec, finish_job(led, job, WorkerResult(rec.task_id, rec.version, "failed", said, evidence=[ev]))


class WhitelistTests(unittest.TestCase):
    def test_a_dishonest_worker_cannot_launder_the_fake_through_failure_text(self) -> None:
        led, rec, out = _finish("get_weather", FAKE, FAKE)          # evidence says "failure_text == what I said"
        self.assertEqual((out.text, out.rejected), (NO_TOOL_ANSWER, "failed_text"))
        self.assertEqual(led.tasks[rec.task_id].status, TaskStatus.FAILED)
        self.assertNotIn("零下", led.tasks[rec.task_id].spoken_answer)

    def test_real_tool_failure_sentences_still_pass(self) -> None:
        for tool, s in (
            ("get_weather", "暂时查不到北京的天气，请稍后再试。"),
            ("get_weather", "Weather for Paris is unavailable right now (sojson:timeout, wttr:no_data)."),
            ("web_search", "网络暂时不可用，没法完成网页搜索，请稍后再试。"),
            ("web_search", "没有搜到关于「今天美元兑人民币汇率」的可靠摘要。"),
            ("web_search", "Web search is unavailable right now due to a network problem. Please try again later."),
            ("get_news", "新闻源暂时连不上，稍后再帮你看看头条吧。"),
            ("get_news", "没有找到相关头条。"),
        ):
            _, _, out = _finish(tool, s, s)
            self.assertEqual((out.text, out.rejected), (s, ""), s)

    def test_a_tool_sentence_with_extra_text_does_not_pass(self) -> None:
        s = "暂时查不到北京的天气，请稍后再试。不过我猜零下三到十一度。"
        _, _, out = _finish("get_weather", s, s)
        self.assertEqual(out.text, NO_TOOL_ANSWER)
        s2 = "暂时查不到北京的天气，请稍后再试。\n北京晴"
        self.assertEqual(_finish("get_weather", s2, s2)[2].text, NO_TOOL_ANSWER)

    def test_shapes_match_what_the_tools_really_say(self) -> None:
        """The whitelist is only useful if it tracks the tools: feed it their real failure outputs."""
        real = [web_search._network_error(True), web_search._network_error(False), web_search._no_results("汇率", zh=True), web_search._no_results("fx rate", zh=False)]

        async def down(url: str) -> bytes:
            raise OSError("down")

        with mock.patch.object(news, "_fetch_rss", down):
            real.append(asyncio.run(news.run({"topic": "科技", "lang": "zh"})))
            real.append(asyncio.run(news.run({"topic": "technology", "lang": "en"})))
        for s in real:
            self.assertTrue(api._is_tool_failure_sentence(s), s)


if __name__ == "__main__":
    unittest.main()
