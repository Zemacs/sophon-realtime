"""OpenAI-style function schemas for the built-in assistant tool pack."""
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "name": "get_datetime",
        "description": "Get the current local date and/or time for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Shanghai. Default: system local."},
                "what": {"type": "string", "enum": ["time", "date", "both"], "description": "Which part to return."},
            },
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather and a short forecast for a city (Open-Meteo, no API key).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name in Chinese or English, e.g. 上海 / Shanghai."},
                "when": {"type": "string", "enum": ["today", "tomorrow"], "description": "Day to report."},
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the public web for recent information; return short snippets for spoken summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in the user's language."},
                "max_results": {"type": "integer", "description": "Max snippets (1-5).", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_news",
        "description": "Fetch current headlines for ONE of a fixed set of news categories via public RSS. Headlines only: for news about a specific person, company or event use web_search instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": ["tech", "finance", "world", "general"], "description": "News category: tech (technology / internet / AI), finance (economy / markets), world (international), general (top headlines, or when no category is named)."},
                "lang": {"type": "string", "enum": ["zh", "en"], "description": "The language the caller speaks: zh or en."},
                "max_items": {"type": "integer", "default": 3},
            },
            "required": ["topic"],
        },
    },
    {
        "type": "function",
        "name": "device_control",
        "description": "Control a simulated smart-home / media device (light, AC, volume, TV, alarm). Safe mock unless SOPHON_LIVE_DEVICE_HOOK is set.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "description": "Device id or type, e.g. light / 空调 / volume."},
                "action": {"type": "string", "description": "Action, e.g. on / off / set / increase."},
                "value": {"type": "string", "description": "Optional value, e.g. 26 for temperature or 50 for volume."},
            },
            "required": ["device", "action"],
        },
    },
    {
        "type": "function",
        "name": "memory_note",
        "description": "Save or recall a short user preference / fact in a local JSON memory store.",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["save", "recall", "list"], "description": "Memory operation."},
                "key": {"type": "string", "description": "Memory key, e.g. preferred_city."},
                "value": {"type": "string", "description": "Value to save (for op=save)."},
            },
            "required": ["op"],
        },
    },
]


def list_tool_names() -> list[str]:
    return [t["name"] for t in TOOL_SCHEMAS]
