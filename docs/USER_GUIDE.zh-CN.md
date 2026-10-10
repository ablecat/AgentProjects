# Repo Maintainer Agent 简易用户说明书

[项目首页](../README.zh-CN.md) | [演示流程](DEMO.md) | [评测协议](../benchmarks/README.md)

## 1. 功能简介

Repo Maintainer Agent 接收本地 Git 仓库和自然语言 Bug 描述，使用
OpenAI-compatible 模型分析代码并生成修改计划；计划经批准后，它只在临时候选
clone 中修改代码、补充回归测试，并在隔离 Docker 容器中运行 Python pytest
验证。最终交付 patch、报告、运行记录、trace 和测试日志，不会把修改自动应用、
提交或推送到原仓库。当前 v1 只支持 Python/pytest 仓库。

当前仓库已经接通仓库工具、安全补丁、Docker 检查、LangGraph 状态机、SQLite
checkpoint、CLI、FastAPI 队列、可视化工作台、独立评测执行器和离线安全验收。
没有模型配置时，系统会明确报错，不会用伪造的模型结果冒充成功；仍可显式使用
无密钥只读 Demo。仓库文档不预设真实模型评测成绩，最终结论以锁定评测集产出的
逐题结果为准。

## 2. 使用前准备

- Docker Desktop 已启动，并使用 `desktop-linux` context。
- 已构建 `repo-agent-python:0.1`。
- 待维护目录是至少包含一次提交的 clean Git 仓库。
- 宿主安装了 Python 3.11 或更高版本和 Git。

首次使用在项目根目录执行：

```powershell
./scripts/start-docker.ps1
./scripts/build-sandboxes.ps1
./scripts/verify-docker.ps1
python -m pip install -e ".[dev]"
python ./scripts/smoke-day2.py
```

Agent 不会读取当前仓库的未提交内容。真实维护任务发现 dirty worktree 时会返回
`policy_denied`；请先自行审核并提交目标仓库的修改。

当前 v1 采用完整文件读取和替换：已有文件必须能被完整读取为仅使用 LF 换行的
UTF-8 文本，且单文件不超过 200 个逻辑行和 16 KiB。`read_files` 每次只能读取
1-4 个互不重复的已有路径，完整批次输出不得超过 16 KiB；任何超限都会拒绝整个
读取，不会把截断前缀交给模型继续编辑。新建或替换内容也受每文件 200 行、16 KiB
限制。因此仓库虽可包含更大、CRLF 或二进制文件，当前版本不能修改这些文件；请在
创建任务前确认目标实现文件及所需测试文件符合限制。

`build-sandboxes.ps1` 和 `verify-docker.ps1` 默认只处理 Python 镜像。旧 Maven
镜像和相应验收只用于复现早期 Python/Java 混合证据；需要这一历史兼容路径时必须
显式选择：

```powershell
./scripts/build-sandboxes.ps1 -IncludeLegacyMaven
./scripts/verify-docker.ps1 -IncludeLegacyMaven
```

前者会刷新 `docker/image-lock.json` 中与本机相关的 image ID，且仅在指定开关时
包含 Maven 条目。Maven 开关不是 Python-only 正常使用的前置条件。

## 3. 配置并验证模型

当前使用 DeepSeek 官方 `https://api.deepseek.com` 作为 API Base URL，模型为
`deepseek-v4-pro`。首次配置或轮换 Key 时，在项目根目录运行：

```powershell
.\scripts\start-relay.ps1 -Reconfigure -ConfigureOnly -Model deepseek-v4-pro
```

脚本会通过显示星号遮罩的 PowerShell 提示读取 API Key，不把密钥写入命令历史、`.env`
或仓库；随后从已认证的 `/models` 响应列出可用模型，并执行
`doctor --allow-remote-model` 的两轮原生 function call 验证。只有验证成功，Key、
Base URL 和模型才会作为一个整体由 Windows DPAPI 加密并原子保存至当前用户的
`%LOCALAPPDATA%\RepoMaintainerAgent\config\relay.json`。该文件不含明文配置，且
只能由本机同一 Windows 用户解密。

以后直接启动，不再输入 Key 或选择模型：

```powershell
.\scripts\start-relay.ps1 -Repository D:\path\to\clean-repository
```

需要重新执行真实模型握手时加 `-Verify`；这会产生少量模型调用费用。脚本默认
选择 `deepseek-v4-pro`；如需显式选择其他已发现的模型，可在首次配置或
`-Reconfigure` 时加 `-Model MODEL_ID`。配置损坏、
换机、换 Windows 用户或管理员强制重置账户凭据后，需要重新配置。脚本退出时仍
会清除它设置的进程级模型环境变量。

只针对 4 个 development fixture 做在线调优时，可运行：

```powershell
.\scripts\start-relay.ps1 -Evaluate
```

该模式使用已保存的 DPAPI 配置，执行固定的 `full` single-mode 评测，并为每次执行
创建新的 `evaluation-results/development-v1/` 子目录。它不接受自定义评测路径，
也不会把 Key 放入参数、日志或评测产物。`-Evaluate` 必须单独使用，不能与启动、
配置、复验或正式评测参数组合。只有至少解决 3/4、四项均未超预算且
`actionable_tool_error_rate <= 5%` 时，命令才以成功状态退出。

代码、prompt、模型和预算冻结后，运行正式候选验收：

```powershell
.\scripts\start-relay.ps1 -FormalEvaluate
```

该入口固定使用 `repo-agent-python-bugfix-v1`，只运行
`py-bugfix-001` 至 `py-bugfix-006` 六题的 `full` trial 1；不接受仓库、端口、
模型、任务、variant 或输出路径参数，也不能和启动、配置、复验或 `-Evaluate`
组合。只有固定 6/6 任务全部完成、至少解决 4/6、每题 usage 完整且不超过
30,000 token、无超时或预算失败且 p95 不超过 480 秒时才通过。结果必须称为
Python v1 regression acceptance；该模式不运行或改写早期 Python/Java 混合
44-job 矩阵与 `benchmarks/results/v1`。

DPAPI 防止其他普通账户或离线复制者直接读取配置，但不防同一已登录账户中的恶意
程序、管理员读取进程内存、配置文件删除或旧密文回放。若怀疑 Key 泄露，应先在
DeepSeek 控制台撤销旧 Key，再使用 `-Reconfigure -ConfigureOnly` 保存新 Key。

非本机回环地址必须使用 HTTPS，并在 `doctor`、`run` 或网页任务中显式允许远程
模型。`doctor` 会优先探测 Responses API，在不支持时尝试 Chat Completions，并
完成两轮原生 function call 校验。两种接口都不可用时直接失败，不做文本 JSON
回退，也不会输出 API Key。

## 4. 启动可视化工作台

```powershell
repo-agent serve --repo D:\path\to\clean-repository --open
```

默认地址为 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)，且只监听本机回环
地址。端口冲突时可加 `--port 9000`。依赖 bootstrap 需要在启动服务时明确授权：

```powershell
repo-agent serve --repo D:\path\to\clean-repository --allow-bootstrap --open
```

工作台有两种模式：

- `Maintenance`：创建持久化 Agent 任务，查看队列与状态，审批或拒绝计划，恢复、
  取消任务，并下载所有运行产物。
- `Read-only demo`：无需模型密钥，只读检查已提交的 `HEAD`，用于验证 Docker、
  repo map 和工具执行链。

页面中的 `Remote model` 必须由用户主动勾选；未勾选时，非 loopback 模型地址会
被策略拒绝。页面不会接受另一个宿主路径或任意 Docker 镜像。

DeepSeek API 会收到任务提示和完成任务所需的代码片段。只对允许发送至该服务的
仓库使用 `Remote model`。

## 5. 使用 CLI 运行任务

交互式审批：

```powershell
repo-agent run `
  --repo D:\path\to\clean-repository `
  --task "修复分页边界错误并增加回归测试" `
  --allow-remote-model
```

从 UTF-8 文件读取任务并自动批准计划：

```powershell
repo-agent run `
  --repo D:\path\to\clean-repository `
  --task-file .\issue.txt `
  --base-ref HEAD `
  --yes `
  --allow-remote-model `
  --format json
```

若仓库声明了需要下载的 Python 依赖，增加 `--allow-bootstrap`。该选项只
授权预注册依赖命令的 bootstrap 容器使用 `bridge` 网络，后续 verify 仍强制
`network=none`。

无密钥只读 Demo 必须显式选择：

```powershell
repo-agent run --repo . --task "Inspect TODO markers" --provider demo
```

## 6. 审批、查询、恢复和取消

`run` 会输出 32 位 `RUN_ID`。常用命令如下：

```powershell
repo-agent show RUN_ID --format json
repo-agent resume RUN_ID --approve --reason "已审核修改范围"
repo-agent resume RUN_ID --reject --reason "范围过大"
repo-agent resume RUN_ID
repo-agent cancel RUN_ID
```

- `--approve` 和 `--reject` 只适用于 `awaiting_approval`。
- 不带审批参数的 `resume` 只适用于 `interrupted`。
- 服务重启会把尚未完成的任务标记为 `interrupted`，恢复时从 SQLite checkpoint
  继续，不重复已经完成的节点和已记录副作用。
- 同一服务只有一个 worker，等待队列最多四项。

Windows 默认把数据库和产物放在 `%LOCALAPPDATA%\repo-agent`。可通过
`--data-dir` 或 `REPO_AGENT_DATA_DIR` 指定其他目录。

同一数据目录同一时间只允许一个 `RunService` 进程持有。可视化服务或 REST API
运行期间，应通过该服务的 UI/API 完成审批、恢复和取消；不要再用 CLI 的
`run`、`resume` 或 `cancel` 打开同一数据目录。`show` 只读查询不创建服务实例，
仍可使用。

## 7. 启动 REST API

```powershell
$env:REPO_AGENT_BEARER_TOKEN = "choose-a-local-token"
$repoRoot = (Get-Location).Path
$allowedRoot = Split-Path -Parent $repoRoot
repo-agent serve --api `
  --repo $repoRoot `
  --allowed-root $allowedRoot `
  --host 127.0.0.1 `
  --port 8080
```

除 `/healthz` 和 `/readyz` 外，请求都必须包含：

```text
Authorization: Bearer choose-a-local-token
```

主要接口：

| 接口 | 用途 |
|---|---|
| `POST /v1/runs` | 创建异步任务，返回 `202` 和 `run_id` |
| `GET /v1/runs?status=&limit=` | 按状态查询近期任务、排队位置和调度摘要 |
| `GET /v1/service/status` | 查询队列深度、容量、活跃任务和并发上限 |
| `GET /v1/runs/{id}` | 查询状态、计划、检查和指标 |
| `GET /v1/runs/{id}/events` | 订阅支持 `Last-Event-ID` 续传的 SSE 活动流 |
| `POST /v1/runs/{id}/decision` | 批准或拒绝计划 |
| `POST /v1/runs/{id}/resume` | 恢复中断任务 |
| `POST /v1/runs/{id}/cancel` | 请求取消任务 |
| `GET /v1/runs/{id}/artifacts/{kind}` | 下载 patch、报告、结果、trace 或测试日志 |

REST 提交的 `repo_path` 必须是绝对路径，并位于服务启动时声明的
`allowed_repo_roots` 内。历史任务的读取和 artifact 查询也应用同一边界。

服务固定为单 worker，最多允许四个任务等待；正在运行的任务不占等待槽位。队列满
时创建任务返回 `429`，取消等待任务会立即释放容量。SSE 同时最多四条连接，客户端
断线后可使用最后收到的事件序号重连；游标来自 SQLite 持久化事件，而不是进程内
计数。事件流只包含白名单化的结构化活动摘要，不包含原始 detail、工具错误文本或
凭据。需要完整检查输出时下载 `checks/*.log`。原生浏览器 `EventSource` 不支持自定义
Authorization 请求头，因此 Bearer API 的 SSE 应由可设置请求头的客户端消费；内置
可视化工作台使用同源、仅回环地址的 `/api/agent/.../events` 接口。

## 8. 状态与产物

| 状态 | 含义 |
|---|---|
| `queued` | 等待单 worker |
| `planning` | 准备仓库、跑基线或生成计划 |
| `awaiting_approval` | 等待人工批准计划 |
| `running` | 实现、验证、修复或审查中 |
| `interrupted` | 服务停止，可手动恢复 |
| `succeeded` | 修改、验证和审查全部完成 |
| `unverified` | 有候选修改，但确定性检查未通过 |
| `policy_denied` | 仓库或请求违反安全策略 |
| `failed` | 模型、工具或工作流失败 |
| `cancelled` / `rejected` | 已取消或计划被拒绝 |

每次运行固定生成：

- `patch.diff`：供人工审阅且保持原字节可应用的 unified diff；若候选包含已配置
  密钥或高置信凭据格式，任务会在写入前失败，不会用脱敏文本改写补丁。
- `report.md`：任务、计划、检查和风险摘要。
- `run.json`：最终结构化状态与指标。
- `trace.jsonl`：追加式、已脱敏的执行事件。
- `checks/*.log`：各次基线和验证日志。

## 9. 安全边界

- 原仓库永不以读写方式挂载，不会自动 apply、commit 或 push。
- Agent 只修改无 hardlink 的临时候选 clone；每次验证使用新的副本。
- 工具按阶段最小化暴露：持久化修改阶段先仅允许 `read_files`，读取成功后仅允许
  提交完整文件内容的 `write_files`；不提供 shell、网络、Git push 或 Docker
  socket。检查由工作流在独立 Docker 阶段执行，模型不能直接调用检查命令。
- 每次任务的模型预算固定为规划 4,000 token、实现与最多一次修复合计 23,500、
  独立审查 2,500；阶段间不可借用，总量硬上限为 30,000 token。审查只有一次
  响应，拒绝后任务失败但候选 patch 和证据会保留。
  该硬上限适用于当前 DeepSeek durable 工作流：provider usage 不完整时任务会
  fail closed。不返回 usage 的 loopback provider 与归档 baseline 仅用于开发或
  历史复现，不属于 Python-only v1 正式路径。
- 路径策略拒绝绝对路径、`..`、`.git`、敏感文件、symlink、submodule、LFS、
  二进制、rename、mode change 和越界路径。
- verify 容器使用非 root UID/GID `10001`、只读 rootfs、`network=none`、2 CPU、
  4 GB 内存、256 PID、`cap-drop ALL` 与 `no-new-privileges`。
- 宿主 HOME、SSH、模型密钥和 Docker socket不会挂载到容器。
- trace、报告、结果和检查日志执行凭据脱敏。

## 10. 验收与评测复现

以下命令覆盖代码质量、CLI/FastAPI、队列、审批、checkpoint、artifact、fake-model
集成、冻结任务结构以及默认 Python Docker 路径：

```powershell
python -m pytest
python ./scripts/smoke-day1.py
python ./scripts/smoke-day2.py
python ./scripts/smoke-day3.py
python ./scripts/smoke-day6.py
python ./benchmarks/tasks/python/bugfix/validate.py
python ./benchmarks/tasks/python/oss/validate.py
python ./benchmarks/validate.py --structure-only
```

其中 Python-only validator 默认执行六题完整生命周期；根目录 validator 的
`--structure-only` 仅审计旧混合套件锁。Day 1/3/6 默认同样只运行 Python case。
OSS validator 只验证三个公开 issue 的裁剪快照、内容锁和独立回归生命周期，不会
调用模型。模型修复率、耗时和 token 必须来自后续 executed single-mode 结果；
`run.json` 中的首次 `checks` 结果和 `metrics.repair_attempts` 可区分首验与 repair。
若需追加旧 Maven 兼容 case，再显式执行：

```powershell
./scripts/build-sandboxes.ps1 -IncludeLegacyMaven
./scripts/verify-docker.ps1 -IncludeLegacyMaven
python ./scripts/smoke-day1.py --include-legacy-maven
python ./scripts/smoke-day3.py --include-legacy-maven
python ./scripts/smoke-day6.py --include-legacy-maven
```

这些 Maven case 只验证历史证据兼容性，不表示当前 Agent 支持 Java 仓库。

在线门禁依次使用固定 development 和正式 Python 验收入口：

```powershell
./scripts/start-relay.ps1 -Evaluate
./scripts/start-relay.ps1 -FormalEvaluate
```

早期 44 次 Python/Java 混合矩阵、重复题规则和固定结果仅作为只读审计资料保留，
见 [benchmarks/README.md](../benchmarks/README.md)。不要将它作为当前发布门禁重跑，
也不要覆盖 `benchmarks/results/v1`。完整可视化演示见 [DEMO.md](DEMO.md)。
