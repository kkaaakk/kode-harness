# Kode Harness

一个参考级 AI Agent Harness 实现 —— 为 AI 模型提供工具链、知识检索、上下文管理、团队协作和权限边界的完整基础设施。

```
Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions
```

模型负责决策，Harness 负责执行。模型负责推理，Harness 提供上下文。模型是驾驶员，Harness 是载具。

## 架构总览

```
Kode Harness = Agent Loop（核心循环）
             + 内置工具集（bash、文件读写编辑、代码搜索...）
             + MCP 协议工具扩展（数据库、文档转换、飞书、社交媒体...）
             + 按需技能加载（Skill System）
             + 上下文压缩（Token Budget + Micro/Macro Compaction）
             + 子 Agent 派发（Subagent Spawning）
             + 持久化任务系统（带依赖图）
             + 多 Agent 团队协作（异步邮箱 + 自动认领）
             + 可插拔沙箱隔离（NoOp / Docker）
             + 执行追踪（Distributed Tracing）
             + 持久化记忆（文件级 Memory）
             + 权限治理
```

## 技术栈

### 运行时 & 语言

| 技术 | 用途 |
|------|------|
| **Python 3.11+** | 核心实现语言，Agent 主循环、工具调度、MCP 集成等全部用纯 Python 编写 |
| **标准库 `re` / `pathlib` / `fnmatch`** | 代码搜索引擎（grep_search / glob_search），零外部依赖，跨平台 |
| **标准库 `threading` / `queue`** | 后台任务并发、团队 Agent 多线程执行、消息通知队列 |
| **标准库 `contextvars`** | 分布式追踪上下文的 Trace ID / Span ID 传播 |
| **标准库 `json` / `uuid` / `time` / `datetime`** | 序列化、任务/消息唯一标识、时间戳、transcript 持久化 |
| **标准库 `subprocess`** | 沙箱命令执行、Docker 容器交互（exec / create / start / stop） |

### AI & LLM 集成

| 技术 | 用途 |
|------|------|
| **Anthropic SDK (`anthropic >= 0.25.0`)** | 与 Claude 系列模型交互的核心 SDK，支持 tool_use / tool_result 多轮对话，支持自定义 base_url 接入兼容 API 提供商 |
| **MCP SDK (`mcp >= 1.9.4`)** | Model Context Protocol 原生 SDK，用于外部工具服务器的连接、工具发现与调用，支持 stdio / streamable_http / SSE 三种传输协议 |
| **tiktoken（可选）** | OpenAI 兼容的 token 计数器，用于精确估算上下文 token 消耗；不可用时自动降级为字符估算 |

### 数据 & 配置

| 技术 | 用途 |
|------|------|
| **PyYAML (`pyyaml >= 6.0`)** | 记忆系统（Memory Manager）中 YAML frontmatter 的解析与写入 |
| **python-dotenv (`>= 1.0.0`)** | 环境变量加载，支持 `.env` 文件管理 API Key、模型 ID、沙箱配置等 |
| **JSONL** | 消息邮箱（inbox）、任务持久化（tasks）、执行追踪（traces）、对话转录（transcripts）的存储格式 |

### Web 前端

| 技术 | 用途 |
|------|------|
| **FastAPI** | HTTP API 服务层，封装 Agent Loop 为 REST 接口（`/api/chat`、`/api/state`、`/api/health`），支持会话管理、CORS |
| **Pydantic** | 请求/响应模型校验（ChatRequest / ChatResponse / StateResponse） |
| **Uvicorn** | ASGI 服务器，支持热重载开发模式 |
| **原生 HTML / CSS / JS** | 前端界面，暗色/亮色双主题，纯静态文件，无框架依赖 |

### 沙箱 & 容器化

| 技术 | 用途 |
|------|------|
| **Docker** | 可选的容器级沙箱隔离后端，每个 Agent 会话一个持久容器 |
| **Ubuntu 24.04（sandbox 镜像）** | 沙箱容器基础镜像，内置 Bash、Git、Python 3、Node.js 22、Rust、build-essential 等开发工具链 |
| **非 root 用户（sandbox）** | Docker 容器内以非特权用户运行，安全隔离 |

### 测试 & CI

| 技术 | 用途 |
|------|------|
| **pytest** | 测试框架，覆盖 Agent 冒烟测试、代码搜索、记忆注入、Token 预算压缩、沙箱、追踪等核心模块 |
| **pytest-benchmark** | 性能基准测试 |
| **GitHub Actions** | CI 流水线，push/PR 触发自动测试（Python 3.11 + Ubuntu） |

### MCP 工具生态

通过 MCP 协议可动态接入以下外部工具服务器（即插即用，故障隔离）：

| MCP 服务器 | 传输协议 | 能力 |
|-----------|---------|------|
| **Bytebase DBHub** | stdio / HTTP | 数据库查询工具（SQL 执行、表列举等），需要 Node.js >= 22.5.0 |
| **Microsoft MarkItDown** | stdio | 文件到 Markdown 转换（PDF、DOCX、PPTX 等） |
| **Feishu / Lark MCP** | stdio | 飞书 API 工具（文档、消息、日历等），支持 OAuth |
| **社交媒体 MCP** | stdio / HTTP | 内置的社交媒体舆情分析工具集（帖子搜索、情感分析、投诉检测、风险评级、去重持久化） |
| **自定义 MCP 服务器** | stdio / HTTP / SSE | 通过 `MCP_SERVERS` 环境变量以 JSON 格式注册任意 MCP 服务器 |

### 兼容 API 提供商

通过 `ANTHROPIC_BASE_URL` 环境变量可接入以下 Anthropic 兼容提供商：

| 提供商 | 模型 |
|--------|------|
| Anthropic（默认） | claude-sonnet-4-6 |
| MiniMax | MiniMax-M2.5 |
| GLM（智谱） | glm-5 |
| Kimi（月之暗面） | kimi-k2.5 |
| DeepSeek | deepseek-chat (V3.2) |

## 核心模块

### Agent Loop (`agents/harness_core.py`)

组合入口，将各子系统组装为完整的 Agent 主循环：

- **工具调度**：25+ 内置工具的注册与分发（bash、文件操作、搜索、任务管理、团队协作、子 Agent 等）
- **REPL 交互**：支持 `/compact`（手动压缩）、`/tasks`（任务看板）、`/team`（团队状态）、`/inbox`（收件箱）
- **生命周期管理**：atexit 自动清理沙箱资源

### 代码搜索 (`agents/code_search.py`)

纯 Python 实现的工作区代码搜索引擎，替代外部 MCP RAG 服务器：

- **`grep_search()`**：基于 `re` 的正则内容搜索，返回 `file:line:content` 格式（类似 ripgrep）
- **`glob_search()`**：基于 `fnmatch` + 自定义 `**` glob 转正则的文件名模式匹配
- 内置目录排除（.venv、__pycache__、node_modules 等）和文件类型过滤（二进制、图片、压缩包等）
- 文件大小上限（1MB）、结果数量上限、输出截断保护
- 路径逃逸防护（所有路径必须在工作区内）

### Token 预算管理 (`agents/token_budget.py`)

多层级上下文压缩系统：

- **Micro Compact**：逐轮清理旧的工具调用结果（保留最近 N 条），释放 token 空间
- **Auto Compact**：当 token 估算超过阈值（默认 100K × 85% = 85K），自动触发 LLM 摘要压缩
- **Transcript 持久化**：压缩前将完整对话历史保存为 JSONL 转录文件
- **记忆保护层**：压缩时自动识别并保留 protected / user_preference / long_term 类型的记忆消息
- **Token 估算**：优先使用 tiktoken 精确计数，不可用时降级为字符估算

### 持久化记忆 (`agents/memory_manager.py`)

基于文件系统的 CRUD 记忆管理：

- **按日分组存储**：每天一个 `.memory/YYYY-MM-DD.md` 文件，内含多条记忆段落
- **YAML 元数据**：每条记忆携带 `memory_type`（preference / user_preference / long_term / protected）
- **索引文件**：`MEMORY.md` 全局索引，支持按名称快速查找
- **消息注入**：可将全部记忆加载为消息上下文，按优先级排序（protected > long_term > preference）
- **关键词搜索**：支持按名称、描述、正文的加权关键词检索

### 任务系统 (`agents/task_manager.py`)

基于文件的持久化任务管理：

- 每个任务存储为独立 JSON 文件（`task_*.json`）
- 支持状态流转：pending → in_progress → completed / deleted
- 依赖图：`blockedBy` 字段定义任务间阻塞关系，完成后自动解除
- 所有权：任务可被 claim 给特定 Agent
- 团队场景下支持自动认领（idle Agent 自动领取未分配的 pending 任务）

### 多 Agent 团队协作 (`agents/team_manager.py`)

完整的多 Agent 协作框架：

- **Teammate 生命周期**：spawn → working → idle → auto-claim → shutdown
- **异步邮箱**：基于 JSONL 文件的消息总线（message / broadcast / shutdown_request / plan_approval_response）
- **工作阶段**：每个 Teammate 在独立线程中运行，最多 50 轮工具调用
- **空闲阶段**：轮询收件箱 + 扫描未认领任务，超时后自动关闭
- **计划审批**：lead Agent 可审批/驳回 Teammate 提交的计划
- **优雅关闭**：基于 request_id 握手的 shutdown 协议

### 子 Agent 派发 (`agents/subagent.py`)

轻量级子 Agent 机制：

- 支持 Explore（只读：bash + read_file）和 general-purpose（读写：+ write_file + edit_file）两种类型
- 独立对话上下文，最多 30 轮工具调用
- 结果汇总后返回给主 Agent

### 技能系统 (`agents/skill_loader.py`)

按需加载的技能模块：

- 每个技能是一个 `SKILL.md` 文件，包含 YAML frontmatter 元数据和 Markdown 正文
- 启动时自动扫描 `skills/` 目录，生成技能清单注入系统提示词
- 运行时按需 load 特定技能，将完整技能内容注入对话上下文
- 当前内置技能：agent-builder、code-review、mcp-builder、pdf

### 沙箱隔离 (`agents/sandbox.py`)

可插拔的命令执行后端：

- **NoOpSandbox**（默认）：直接在宿主机执行命令和文件操作
- **DockerSandbox**：每个 Agent 会话一个 Docker 容器，所有操作通过 `docker exec` 执行
  - 路径映射：自动处理宿主机路径到容器路径的转换（含 Windows WSL 兼容）
  - 资源限制：可配置内存（`--memory`）、CPU（`--cpus`）、网络（`--network`）
  - 危险命令拦截：`rm -rf /`、`sudo`、`shutdown`、`reboot` 等黑名单
  - 超时控制：默认 120 秒
- **工厂模式**：通过 `AGENT_SANDBOX_BACKEND` 环境变量选择后端，自动检测 Docker 可用性

### 执行追踪 (`agents/tracer.py` + `agents/trace_context.py`)

类 OpenTelemetry 的分布式追踪系统：

- **Trace Context**：基于 `contextvars` 的 Trace ID / Span ID 管理，跨线程传播
- **TraceSpan**：上下文管理器，自动记录 span 的起止时间、耗时、输入/输出摘要、错误信息
- **持久化**：追踪事件以 JSONL 格式按日写入 `.tmp/runtime/team/traces/`
- **线程安全**：写入操作通过 `threading.Lock` 保护

### MCP 集成 (`mcp/`)

完整的 MCP 工具链集成层：

- **Tool Loader** (`tools.py`)：多服务器并行加载，每个服务器独立故障隔离
- **Tool Registry** (`registry.py`)：三维索引（by_name / by_server / by_domain），O(1) 查找
- **Tool Executor** (`executor.py`)：connect → execute → disconnect 模式，每次调用重新连接目标服务器
- **Tool Assembler** (`tool_assembler.py`)：启动时一次性发现、注册、组装所有 MCP 工具
- **Domain Filter** (`domain_filter.py`)：基于场景的工具域过滤系统，支持 LLM 域分类 + 关键词回退
- **Tool Wrapper** (`tool_wrapper.py`)：命名空间前缀冲突解决
- **Social Media MCP** (`social_media_mcp.py`)：内置的社交媒体舆情分析 MCP 服务器（15+ 工具），支持帖子搜索、情感分析、投诉检测、风险评级、去重持久化

### 前端 (`frontend/`)

轻量 Web 界面：

- **FastAPI 后端**：REST API 封装 Agent Loop，支持多会话管理
- **原生前端**：HTML + CSS + JavaScript，无框架依赖
- **暗色/亮色主题**：CSS 变量驱动的主题切换
- **状态面板**：实时展示任务看板、团队状态、后台任务、收件箱、Todo 列表

## 项目结构

```
kode-harness/
├── agents/
│   ├── harness_core.py        # 组合入口（Agent 主循环 + REPL）
│   ├── config.py              # 全局常量、环境配置、Sandbox 初始化
│   ├── base_tools.py          # 基础工具函数（bash、文件读写、代码搜索）
│   ├── todo_manager.py        # Todo 清单管理
│   ├── skill_loader.py        # 按需技能加载系统
│   ├── compression.py         # 上下文压缩（microcompact / auto_compact）
│   ├── task_manager.py        # 持久化文件任务管理（带依赖图）
│   ├── subagent.py            # 轻量级子 Agent 派发
│   ├── team_manager.py        # 多 Agent 团队协作（MessageBus + TeammateManager）
│   ├── code_search.py         # 纯 Python grep/glob 代码搜索
│   ├── memory_manager.py      # 持久化文件记忆管理
│   ├── token_budget.py        # Token 预算与多层压缩
│   ├── sandbox.py             # 可插拔沙箱（NoOp / Docker）
│   ├── tracer.py              # 分布式执行追踪
│   └── trace_context.py       # Trace/Span 上下文管理
├── mcp/
│   ├── tools.py               # 多服务器 MCP 工具加载器
│   ├── registry.py            # 三维索引工具注册表
│   ├── executor.py            # MCP 工具执行器
│   ├── tool_assembler.py      # 工具组装入口
│   ├── domain_filter.py       # 场景域过滤系统
│   ├── tool_wrapper.py        # 命名空间冲突处理
│   └── social_media_mcp.py    # 社交媒体舆情分析 MCP 服务器
├── skills/                    # 按需加载技能模块
│   ├── agent-builder/         # Agent 构建技能
│   ├── code-review/           # 代码审查技能
│   ├── mcp-builder/           # MCP 服务器构建技能
│   └── pdf/                   # PDF 处理技能
├── frontend/
│   ├── server.py              # FastAPI 服务层
│   └── index.html             # Web 界面
├── docker/
│   └── sandbox.Dockerfile     # 沙箱容器镜像定义
├── tests/                     # 测试套件
├── .github/workflows/         # CI 流水线
├── requirements.txt           # Python 依赖
└── .env.example               # 环境变量模板
```

## 快速开始

```sh
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env 填入 ANTHROPIC_API_KEY

# CLI 模式
python agents/harness_core.py

# Web 模式
python -m uvicorn frontend.server:app --reload --port 8765
```

## License

MIT

