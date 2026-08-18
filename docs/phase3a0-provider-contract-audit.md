# Phase 3A-0 Provider 契约盘点 -- Anthropic-specific 依赖全景

> 状态：盘点完成，未改任何代码。本文档是 3A-1（AnthropicAdapter）的施工图。
> 原则：Phase 3 只抽「模型调用纵向链路」，不碰已封板的
> ToolRegistry / Profile / Extension / Artifact / OutputPolicy。

## 1. 核心发现：Anthropic 依赖不止在 harness_core

`client.messages.create()` 共有 **5 个调用点**，分布在 4 个模块：

| # | 调用点 | 位置 | 用途 | Anthropic 方言深度 |
|---|--------|------|------|--------------------|
| 1 | **Agent Loop 主调用** | `harness_core.py:1351` | 带全部 8 个 Extension hooks、tools、system、event_callback | 深（content blocks / stop_reason / usage） |
| 2 | **Subagent 循环** | `subagent.py:73` | 30 轮子循环，硬编码 4 工具 | 深（content / stop_reason / tool_use） |
| 3 | **Team member `_loop`** | `team_manager.py:246` | 50 轮 member 循环，硬编码 7 工具 | 深（content / stop_reason / tool_use / 异常兜底） |
| 4 | **auto_compact 摘要** | `compression.py:43` | 无 tools、max_tokens=2000、取 `content[0].text` | 浅 |
| 5 | **token_budget 摘要** | `token_budget.py:275`（`summarize_with_anthropic`） | 无 tools 摘要调用 | 浅 |

另有两处非调用但依赖 Anthropic 概念：

| 位置 | 依赖 | 说明 |
|------|------|------|
| `config.py:12,42` | `from anthropic import Anthropic` + `client = Anthropic(base_url=...)` | 全局唯一 client 构造点（含 `ANTHROPIC_BASE_URL` 时清 `ANTHROPIC_AUTH_TOKEN` 的兼容 hack） |
| `frontend/server.py:91` | 解析「Anthropic response.content」的辅助 | 从 event_callback/捕获 stdout 提取，不直接调 SDK |

**消息格式耦合**（Anthropic wire format 深入各处）：`harness_core`、`subagent`、`team_manager`、`compression`、`token_budget`、`tool_output_policy`、`frontend/server.py`（session 存 Anthropic message dicts）全部直接构造/解析 `{"role","content":[blocks]}`、`tool_use`/`tool_result` block、`tool_use_id`。

**测试替身**：`tests/test_harness_background.py` 用 `types.SimpleNamespace(stop_reason=..., content=[...])` 模拟响应；全库测试无一处 import anthropic SDK（fake 注入 `module.client.messages.create`），3A 改造**不需要**网络。

## 2. 归属分类表（3A-1 施工目标）

| Anthropic-specific | 当前位置 | Phase 3 后归属 | 3A 处理 |
|--------------------|----------|----------------|---------|
| `Anthropic(...)` client 构造 + BASE_URL/AUTH_TOKEN 兼容 | `config.py` | AnthropicAdapter 内部（client 仍可由 config 注入） | 3A-1 |
| `client.messages.create(**kwargs)`（主循环） | `harness_core.py:1351` | AnthropicAdapter.complete() | 3A-1 |
| `response.content` block 遍历（text/tool_use） | `harness_core.py:1352,1377,1396` | AnthropicAdapter -> ModelResponse.text/tool_calls | 3A-1 |
| `response.stop_reason != "tool_use"` | `harness_core.py:1389` | Adapter -> StopReason.TOOL_CALL | 3A-1 |
| `response.usage.input/output_tokens`（tokens 事件） | `harness_core.py:1366-1376` | Adapter -> TokenUsage | 3A-1 |
| `messages.append({"role":"assistant","content": response.content})` | `harness_core.py:1352` | Adapter 需提供 `raw_content`（回写 wire format）**或** Loop 改写统一 Message 追加 -- 见 §4 决策点 | 3A-1 |
| `tool_result` block 构造（`tool_use_id`） | `harness_core.py:1497` 等 | **暂留 Agent Loop**（消息层统一放 3B/3C，见 §4） | 3A 先不动 |
| subagent `client.messages.create` 循环 | `subagent.py:73` | AnthropicAdapter（Runtime 边界不变） | 3A-1 |
| team member `client.messages.create` 循环 | `team_manager.py:246` | AnthropicAdapter（Runtime 边界不变，异常兜底语义保持） | 3A-1 |
| compression / token_budget 摘要调用 | `compression.py:43`、`token_budget.py:275` | AnthropicAdapter.complete（无 tools 场景） | 3A-1 |
| tools schema（canonical 即 Anthropic 格式） | ToolRegistry.resolve() | **暂不动**。canonical -> provider wire 转换放 Adapter（3B OpenAI 时才需要） | 不动 |
| tool 执行 / OutputPolicy / Artifact / hooks | `harness_core` | Agent Loop，不动 | 不动 |
| BEFORE_MODEL_REQUEST 的 `request_kwargs` | `harness_core.py:1326` | 仍传 dict（Extension 契约已封板）；3A-1 让 Adapter 接收该 dict 形状 | 保持形状 |
| AFTER_MODEL_RESPONSE 的 `response` 对象 | `harness_core.py:1357` | 传统一 ModelResponse（含 provider_metadata/raw_response）；Extension 若需 Anthropic 对象可取 raw_response | 3A-1（附迁移说明） |
| frontend session 存储 Anthropic messages | `frontend/server.py` | 暂不动（3D 统一 Message 时再理） | 不动 |

## 3. 统一类型（Phase 3 目标形状，3A-1 落地于 `agents/provider/types.py`）

```python
class StopReason(Enum):
    END / TOOL_CALL / LENGTH / STOP_SEQUENCE / ERROR / UNKNOWN

@dataclass TokenUsage:   input_tokens / output_tokens / cache_read_tokens / cache_write_tokens
@dataclass ToolCall:      id / name / arguments
@dataclass ModelRequest:  model / messages / tools / system / max_tokens / temperature / metadata
@dataclass ModelResponse: text / tool_calls / usage / stop_reason / provider / model /
                          provider_metadata / raw_response
```

映射表（Anthropic -> 统一）：

| Anthropic | 统一 |
|-----------|------|
| `stop_reason="end_turn"` | `StopReason.END` |
| `stop_reason="tool_use"` | `StopReason.TOOL_CALL` |
| `stop_reason="max_tokens"` | `StopReason.LENGTH` |
| `stop_reason="stop_sequence"` | `StopReason.STOP_SEQUENCE` |
| `content[].type=="text"` | 拼接进 `ModelResponse.text`（多 text block 保留顺序拼接） |
| `content[].type=="tool_use"` | `ToolCall(id=block.id, name=block.name, arguments=dict(block.input))` |
| `usage.input_tokens` / `output_tokens` | `TokenUsage`（cache 字段 getattr 容错，现版本 SDK 可能为 None） |
| 其他 content block 类型（thinking 等） | 进 `provider_metadata["unknown_blocks"]`，不丢数据 |

## 4. 关键决策点（3A-1 开工前需确认）

**D-1：wire-format 消息历史怎么办？**

5 个调用点共享同一条 Anthropic 格式的 `messages` 历史（`[{"role","content":[...]}]`）。
两个方案：

- **方案甲（推荐）**：3A 只包「调用+解析」，`ModelRequest.messages` 仍传 Anthropic wire 格式 dict 列表；Adapter 原样透传。消息历史格式统一推迟到 3B/3C（届时 canonical Message + 各 Adapter 双向转换）。优点：零行为变化、改动面最小、subagent/team/compression 四个调用点可无痛切换。
- **方案乙**：3A 同时定义统一 Message 格式并双向转换。缺点：触碰所有消息构造点 + 压缩/token_budget 对历史结构的遍历（`tool_result` 识别）+ frontend session 存储，等于重开一个大面，违背「先证明零变化」。

**D-2：`{"role":"assistant","content": response.content}` 回写**

方案甲下 Adapter 需暴露 `response.raw_response.content`（或 ModelResponse.provider_metadata["assistant_content"]）供 Loop 原样回写历史。3A-1 采用：`ModelResponse.raw_response` 保留 SDK 原对象，Loop 回写逻辑不变。

**D-3：异常语义**

Anthropic SDK 异常（APIError/Timeout 等）3A 不吞不转：Adapter 让其原样抛出，各调用点现有 try/except 行为（如 team member 捕获一切后 shutdown）完全不变。统一 ProviderError 体系放 3B。

## 5. 3A-1 验收标准（golden/snapshot）

现有 `tests/test_harness_background.py` 的 fake client 模式（替换 `module.client.messages.create`）是天然的 golden 测试基座。3A-1 完成后必须：

1. 全部现有测试**不修改断言**通过（fake 直接打到 Adapter 底下的 client 层）；
2. 新增 `tests/test_provider_anthropic_adapter.py`：
   - 纯文本回答 / 单 tool call / 多 tool call / text+tool_call 混合
   - tool args 透传 / tool_call id / stop_reason 四种映射
   - input/output token 提取 / cache token 容错（None）
   - max_tokens / system / tools 参数透传（请求形状快照）
   - 空 content / 未知 block 类型不丢数据
   - 异常穿透（不包装）
   - **等价性对照**：同一 fake response 分别走「旧直连路径」与「Adapter 路径」，Agent Loop 可观察结果（事件序列、消息历史、工具执行）逐项一致；
3. subagent / team_manager / compression / token_budget 四个调用点切换后各自测试套件全绿；
4. `harness_core.py` 中不再出现 `block.type == "tool_use"`、`stop_reason != "tool_use"`、`response.usage` 等 Anthropic 方言（grep 验证）。

## 6. 明确不做（Phase 3A 边界）

- 不加任何新 Provider（OpenAI-compatible 是 3B）
- 不动 streaming（现无 streaming，3C/3D 再议）
- 不统一 Tool Schema wire format（canonical 保持现状，3B 才做转换）
- 不动 ToolRegistry / Profile / Extension hooks 契约 / Artifact / OutputPolicy
- 不做 ProviderRouter / Model Registry / ModelSpec / Capabilities（3B/3C）
- 不做 `/model` 切换与 MODEL_CHANGED（3D）
- 不动 frontend session 消息存储格式
