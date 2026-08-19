# Stage 2 架构快照 —— Extension/Tool 迁移总收口（2D-D3）

> 本文档是 Pi 风格工具系统改造（Stage 2A → 2D-D3）的最终架构快照。
> D3 之后 Stage 2 正式封板，进入 Phase 3（Provider 抽象）。
> 如果后续阶段（Provider / Runtime Unification）有意改变本文任何契约，
> **必须在同一次提交中更新本文档与
> `tests/test_d3_final_composition_contract.py`**。

## 1. 迁移成果总览

```text
最初（Stage 2A 之前）：          现在（2D-D3 之后）：
Base = 25                       Base（kernel）= 12
Extension = 0                   TodoExtension      = 1  (TodoWrite)
                                TaskExtension      = 4  (task_create/get/update/list)
                                SubagentExtension  = 1  (task)
                                TeamExtension      = 7  (spawn_teammate/list_teammates/
                                                         send_message/read_inbox/broadcast/
                                                         shutdown_request/plan_approval)
                                -------------------------
                                默认组合            = 25（名称、顺序、schema、行为不变）
```

13/25 原内置工具已迁出 Base，默认用户仍看到原来的 25 个工具。

## 2. 架构图

```text
                    Agent Loop
                        │
          ┌─────────────┴─────────────┐
          │                           │
 Extension Lifecycle             Tool Runtime
          │                           │
 ExtensionRegistry          BASE_TOOL_REGISTRY (12, immutable)
                                      │
                              per-call ToolRegistryOverlay
                                      │
                       (DEFAULT_TOOL_CONTRIBUTORS)
                    ┌────────┬────────┴────────┬─────────┐
                    │        │                 │         │
             TodoExtension TaskExtension SubagentExt. TeamExtension
                (1)         (4)             (1)         (7)
```

工具执行链路（单次工具调用）：

```text
ToolRegistry Overlay
-> Profile resolve
-> BEFORE_TOOL_CALL
-> Kernel security（权限/沙箱/安全策略）
-> handler
-> ToolOutputPolicy
-> ArtifactStore
-> AFTER_TOOL_RESULT
-> FinalOutputGuard
-> Model
```

## 3. Tool Ownership 表（D3-2，source of truth）

Registry 是工具所有权的唯一事实来源；本表是其快照，由
`tests/test_d3_final_composition_contract.py` 锁定。

| Tool | Owner | Source | Default(None) | coding | planning | readonly | team |
|------|-------|--------|:---:|:---:|:---:|:---:|:---:|
| bash | kernel | builtin | ✓ | ✓ | – | – | ✓ |
| read_file | kernel | builtin | ✓ | ✓ | ✓ | ✓ | ✓ |
| write_file | kernel | builtin | ✓ | ✓ | – | – | ✓ |
| edit_file | kernel | builtin | ✓ | ✓ | – | – | ✓ |
| grep_search | kernel | builtin | ✓ | ✓ | ✓ | ✓ | ✓ |
| glob_search | kernel | builtin | ✓ | ✓ | ✓ | ✓ | ✓ |
| load_skill | kernel | builtin | ✓ | – | – | – | – |
| compress | kernel | builtin | ✓ | – | – | – | – |
| background_run | kernel | builtin | ✓ | – | – | – | – |
| check_background | kernel | builtin | ✓ | – | – | – | – |
| idle | kernel | builtin | ✓ | – | – | – | – |
| claim_task | kernel | builtin | ✓ | – | – | – | – |
| TodoWrite | todo-extension | extension | ✓ | – | ✓ | – | – |
| task_create | task-extension | extension | ✓ | – | ✓ | – | – |
| task_get | task-extension | extension | ✓ | – | ✓ | – | – |
| task_update | task-extension | extension | ✓ | – | ✓ | – | – |
| task_list | task-extension | extension | ✓ | – | ✓ | – | – |
| task | subagent-extension | extension | ✓ | – | – | – | ✓ |
| spawn_teammate | team-extension | extension | ✓ | – | – | – | ✓ |
| list_teammates | team-extension | extension | ✓ | – | – | – | ✓ |
| send_message | team-extension | extension | ✓ | – | – | – | ✓ |
| read_inbox | team-extension | extension | ✓ | – | – | – | ✓ |
| broadcast | team-extension | extension | ✓ | – | – | – | ✓ |
| shutdown_request | team-extension | extension | ✓ | – | – | – | ✓ |
| plan_approval | team-extension | extension | ✓ | – | – | – | ✓ |

## 4. Base 12 分类（D3-1 盘点结论）

| 类别 | 工具 | 决策 |
|------|------|------|
| Coding 原子能力（6） | bash、read_file、write_file、edit_file、grep_search、glob_search | **永久留 Base** |
| 生命周期/基础设施（4） | load_skill、compress、background_run、check_background | 暂留 Base；候选迁出方向：SkillSystem/Background Extension（Stage 4+），不在本阶段处理 |
| Team member 硬编码工具（2） | idle、claim_task | 暂留 Base（D0 决策不动 member Runtime）；随 Team member 工具集在 Runtime Unification 阶段一起迁移 |

明确不做的：**不为了追求 Base=6 而硬迁**。D3 只盘点、只记录。

## 5. 关键契约

- `LEGACY_25_TOOL_NAMES`：25 个工具的权威顺序；默认组合 resolve(None)
  必须逐项等于该列表。
- `LEGACY_TOOL_ORDER`（D3-3 新增）：`{name: slot}` 统一字典，25 个槽位
  唯一事实来源；旧的 `LEGACY_*_ORDER` 逐工具常量保留为别名（数值与
  D3 前完全一致，纯机械收口）。
- Extension 无状态：四个 Extension 均不持有 per-agent 数据；状态留在
  Runtime（TODO/SKILLS/TASK_MGR/BG/BUS/TEAM 等模块级对象）。
- `tool_contributors=()` 显式禁用全部可选扩展工具（全部 Unknown）。
- Profile 白名单允许可选 Extension 工具不存在（如 team profile 在
  TeamExtension 禁用时仍可启动）。

## 6. Runtime 边界（迁移中守住、继续保持）

Extension 层**只管注册可见性**，不碰以下 Runtime 语义：

- **Subagent**：同一 asyncio Task 内同步执行，30 轮独立循环，硬编码
  子工具集，继承 Parent SecureBashContext（ContextVar 同 Task 可见）。
- **Team member**：独立 daemon thread，自己的 50 轮 `_loop`，不调
  agent_loop、不用 ToolRegistry、不继承 Parent Profile/Contributors，
  使用自己的 7 工具硬编码集（bash/read_file/write_file/edit_file/
  send_message/idle/claim_task），max_tokens=8000，共享
  Client/Model/Sandbox/BUS/TASK_MGR。
- **Secure Bash 不对称（钉死）**：Subagent 继承 Parent grant；
  Team member 新线程不继承 ContextVar，secure 模式下 Bash 被拒。

## 7. 登记的技术债（不在 Extension 阶段处理）

| # | 技术债 | 登记位置 | 处理阶段 |
|---|--------|----------|----------|
| 1 | Team Runtime 缺少完整 team-scoped 状态隔离：BUS/TASK_MGR 为 Harness 级全局共享（inbox/shutdown/claim 仅按名字域隔离） | `test_team_extension_migration.py` 场景 22d | Runtime Unification / Multi-Agent |
| 2 | Secure Bash Parent→Team-member 不对称（新线程不继承 grant） | D0 契约快照 + 迁移测试场景 20 | Runtime Unification |
| 3 | `MessageBus.send()` 用 `open('a')` 无文件锁，Windows 并发写可能丢行 —— 即当前全量回归中唯一的 `1 xfailed`（`test_team_manager.py::TestMessageBusConcurrency::test_concurrent_send_no_data_loss`，strict=False） | `tests/test_team_manager.py:187` | Runtime Unification（与 #1 同族） |
| 4 | Base 中 `idle`/`claim_task` 属 Team member 硬编码工具，注册归属错位 | 本文 §4 | Runtime Unification（随 member 工具集迁移） |

## 8. Stage 2 验收状态

- D0 契约快照（Parent→Child 继承矩阵）✅
- D1 SubagentExtension ✅
- D2 TeamExtension + 测试债修复 ✅（已验收）
- D3 总收口 ✅（本文档 + 20 项组合契约测试）
  - D3-1 Base 12 盘点 ✅
  - D3-2 Ownership 表 ✅
  - D3-3 LEGACY_TOOL_ORDER 收口 ✅（机械，无数值变化）
  - D3-4 最终组合快照 ✅
  - D3-5 xfailed 定位登记 ✅（见 §7 #3）
  - D3-6 本文档 ✅

## 9. 下一步：Phase 3 Provider 抽象

```text
Anthropic 现有实现      ->  AnthropicAdapter
OpenAI-compatible       ->  OpenAICompatibleAdapter
                    ↓
                  ProviderRouter
                    ↓
          ModelResponse / ModelCapabilities
```

工具系统、Extension、Output Policy、Artifact、Sandbox、Profile 均已有
清晰边界与契约测试护栏，Provider 改造只需处理模型调用这一条纵向链路，
不再与工具架构纠缠。

## 10. Phase 3 Provider 抽象 —— 进度与验证记录

### 阶段状态

```text
3A-0 Provider 契约盘点 + golden ✅
3A-1 AnthropicAdapter + 5 调用点迁移 ✅
3B-0 OpenAI wire 契约 ✅
3B-1 OpenAICompatibleAdapter ✅
3B-2 DeepSeek 真机 ✅
3B-3 OpenRouter 真机 ✅
3B 封板 ✅
3C-0 ModelSpec / ModelCapabilities / ModelRegistry ✅
3C-1 ProviderRouter ⏸（下一步）
3D /model 会话内切换 ⏸
```

### 真机验证记录（OpenAICompatibleAdapter 零修改）

**3B-2 DeepSeek**（model=deepseek-chat, base=https://api.deepseek.com）：
- plain chat: `'OK'`，stop=end，usage(9, 1)
- tool call: `get_weather {city: Beijing}`，stop=tool_call
- continuation: `'Beijing 22°C sunny'`，usage(349, 13)，**cache_read=256**（来自 prompt_tokens_details.cached_tokens）

**3B-3 OpenRouter**（model=qwen/qwen-2.5-72b-instruct, base=https://openrouter.ai/api/v1，非 OpenAI 原生模型，验证 Adapter 通用性）：
- plain chat: `'OK'`，stop=end，usage(13, 1)
- tool call: `get_weather {city: Beijing}`，stop=tool_call
- continuation: `'Beijing 22°C sunny'`，stop=end，usage(387, 13)

同一 OpenAICompatibleAdapter 连接两个不同 Provider 端点，三段全链路通过，
无任何 `if provider` 特判 → 证明抽象是通用 Adapter 而非 DeepSeekAdapter。

### Provider 抽象边界（3C 核心建模）

```text
Model    -> ModelSpec.model_id（发给 provider 的字符串）
Provider -> ModelSpec.provider（anthropic / deepseek / openrouter）
Adapter  -> 协议类型（AnthropicAdapter / OpenAICompatibleAdapter）
```

provider 绝不退化为 "openai-compatible"（服务商 ≠ 协议）。ModelSpec 不含
base_url/api_key/client（归属 3C-1 Provider binding）。

### 登记技术债

| # | 技术债 | 处理阶段 |
|---|--------|----------|
| 5 | OpenAICompatibleAdapter 用标准库 urllib 传输层（无外部依赖，3B 阶段正确）；streaming/connection pooling/retry/HTTP2/async 引入时再换 | streaming/async transport |

### 3C-1 ProviderRouter（计划）

```text
ModelSpec.provider
        ↓
ProviderRouter
        ↓
Provider binding
        ↓
Adapter
  anthropic  -> AnthropicAdapter
  deepseek   -> OpenAICompatibleAdapter
  openrouter -> OpenAICompatibleAdapter
```

binding 持有 endpoint 配置（base_url / api_key_env），ModelSpec 不持有。
