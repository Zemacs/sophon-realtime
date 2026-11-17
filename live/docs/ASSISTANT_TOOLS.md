# Sophon Live 通用语音助手工具包

> 分支：`feat/assistant-tools` · Fork：`github.com/Zemacs/sophon-realtime`  
> 更新：2026-11-17

## 1. 当前「已注册」工具（盘点结论）

| 层 | 模型可见工具 | 真正执行方 |
|---|---|---|
| GPT-Live `/v1/live/sessions` | 仅 **`delegate(task)`**（`delegation.type=client` 时） | **推荐** `SOPHON_LIVE_ORCH=1`（本机 LLM 选型+工具）；或浏览器 demoBackend / 手动 / 遗留 `BUILTIN` 启发式 |
| OpenAI Realtime `/v1/realtime` | 客户端传入的 `tools[]`，或 BackOffice 模式下的 **`delegate`** | 客户端 function_call 或 `SOPHON_LIVE_BACKEND` |
| 引擎 Tag/Flash/E2E | **无内置领域工具**；`extra_body.tools` 透传 | `tool_mode=mock` 本地假结果，或 `client` + HTTP bridge |

演示页原能力：天气（Open-Meteo）、时间/日期、设备模拟；**新闻/搜索/记忆缺失**（用户实测「查特朗普/黄仁勋」会落到「查不到」）。

## 2. 本包新增（P0）

包路径：`live/sophon_live/assistant_tools/`

| 工具名 | 作用 |
|---|---|
| `get_datetime` | 本地/时区 时间日期 |
| `get_weather` | Open-Meteo 天气 |
| `web_search` | DuckDuckGo Instant Answer（无 key） |
| `get_news` | Google News RSS 头条 |
| `device_control` | 模拟设备（可选 `SOPHON_LIVE_DEVICE_HOOK`） |
| `memory_note` | 本地 JSON 记忆 |

前端仍只调用 `delegate`。

- **推荐：** `SOPHON_LIVE_ORCH=1` → `orchestrator.py` 调本地 vLLM，用上表 schemas 选型，再 `run_tool` 执行并 `commentary` 回灌（`backend=orch`）。  
- **遗留演示：** `SOPHON_LIVE_BUILTIN_TOOLS=1` → 启发式 `run_task`（**不是**正式架构）。

## 3. 启用（路径 1 共置编排）

```bash
# 先起本地编排 LLM，例如 :8101 served-model-name=qwen3.8-27b-bo
export SOPHON_LIVE_ORCH=1
export SOPHON_LIVE_ORCH_URL=http://127.0.0.1:8101/v1/chat/completions
export SOPHON_LIVE_ORCH_MODEL=qwen3.8-27b-bo
BUILD=tag PORT=8000 LIVE_PORT=8780 bash start_live.sh
```

测试页收到 `delegation.backend === "orch"`（或 `"builtin"`）时不再跑浏览器 demoBackend。总方案见 deplux `docs/sophon-path1-colocated-backend.md`。

## 4. 架构选择（通用助手）

```text
用户语音 → LiveSession → 引擎（仅 delegate）
                ↓
        assistant_tools.run_task(task)
                ↓
        commentary → tool_result → 口语转述
```

远期 P1（未做）：日历、邮件、地图导航、真实 IoT、鉴权搜索 API；Realtime 面可直接暴露 `TOOL_SCHEMAS` 给 BackOffice。

## 5. 测试

```bash
cd live
python -m unittest tests.test_assistant_tools -v
```

## 6. 安全

- 不提交 API key；默认搜索/新闻为公开只读源。  
- 设备控制默认真模拟；生产需自建 hook 与鉴权。  
- 记忆文件在 `live/logs/`（已 gitignore 惯例，勿提交用户数据）。
