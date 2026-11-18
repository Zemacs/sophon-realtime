"""Offline-safe tests for datetime, weather, and device assistant tools."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

_LIVE_ROOT = Path(__file__).resolve().parents[1]
if str(_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVE_ROOT))


def _load_tool(name: str):
    path = _LIVE_ROOT / "sophon_live" / "assistant_tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sophon_live.assistant_tools.{name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


datetime_tool = _load_tool("datetime_tool")
weather = _load_tool("weather")
device = _load_tool("device")


class DatetimeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_shanghai(self) -> None:
        fixed = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        with (
            mock.patch.object(datetime_tool, "ZoneInfo", return_value=timezone.utc),
            mock.patch.object(datetime_tool, "datetime") as mdt,
        ):
            mdt.now.return_value = fixed
            out = await datetime_tool.run({"timezone": "Asia/Shanghai", "what": "time"})
        self.assertIn("14点30分", out)
        self.assertTrue(out.endswith("。"))

    async def test_date_english(self) -> None:
        fixed = datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
        with (
            mock.patch.object(datetime_tool, "ZoneInfo", return_value=timezone.utc),
            mock.patch.object(datetime_tool, "datetime") as mdt,
        ):
            mdt.now.return_value = fixed
            out = await datetime_tool.run(
                {"timezone": "America/New_York", "what": "date", "lang": "en"}
            )
        self.assertIn("March", out)
        self.assertIn("Today is", out)


class WeatherToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_weather_today_zh_mocked(self) -> None:
        payload = {
            "current_condition": [{"temp_C": "18"}],
            "weather": [
                {
                    "mintempC": "5",
                    "maxtempC": "12",
                    "hourly": [
                        {"chanceofrain": "10", "lang_zh": [{"value": "晴"}], "weatherDesc": [{"value": "Sunny"}]},
                        {"chanceofrain": "10", "lang_zh": [{"value": "晴"}], "weatherDesc": [{"value": "Sunny"}]},
                        {"chanceofrain": "10", "lang_zh": [{"value": "晴"}], "weatherDesc": [{"value": "Sunny"}]},
                        {"chanceofrain": "10", "lang_zh": [{"value": "晴"}], "weatherDesc": [{"value": "Sunny"}]},
                        {"chanceofrain": "10", "lang_zh": [{"value": "晴"}], "weatherDesc": [{"value": "Sunny"}]},
                    ],
                },
                {"mintempC": "6", "maxtempC": "14", "hourly": [{"chanceofrain": "30", "lang_zh": [{"value": "多云"}]}]},
            ],
        }

        with mock.patch.object(weather, "_fetch_json", return_value=payload):
            out = await weather.run({"city": "北京", "when": "today"})

        self.assertIn("北京", out)
        self.assertIn("今天", out)
        self.assertIn("晴", out)
        self.assertIn("现在18度", out)

    async def test_weather_empty_days(self) -> None:
        with mock.patch.object(weather, "_fetch_json", return_value={"weather": []}):
            out = await weather.run({"city": "Nowhereville", "when": "today", "lang": "en"})
        self.assertIn("No weather data", out)


class DeviceToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_simulate_zh(self) -> None:
        with mock.patch.dict(os.environ, {"SOPHON_LIVE_DEVICE_HOOK": ""}, clear=False):
            out = await device.run({"device": "客厅灯", "action": "打开"})
        self.assertIn("模拟设备", out)
        self.assertIn("客厅灯", out)

    async def test_hook_script(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as f:
            f.write("@echo hook ok\n")
            path = f.name

        try:
            with mock.patch.dict(os.environ, {"SOPHON_LIVE_DEVICE_HOOK": path}, clear=False):
                out = await device.run({"device": "light", "action": "on", "lang": "en"})
            self.assertIn("hook ok", out)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()