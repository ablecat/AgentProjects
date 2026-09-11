# Repo Maintainer Agent 可复现实操演示

[项目首页](../README.zh-CN.md) | [详细用户说明](USER_GUIDE.zh-CN.md) | [评测协议](../benchmarks/README.md)

本演示先验证无需模型的只读路径，再使用冻结 benchmark 中的一个 Python fixture
创建独立、clean 的 Bug 仓库，演示真实模型规划、人工审批、Docker 验证和产物
审阅。整个过程不会修改 Agent 项目仓库，也不会把候选补丁写回示例原仓库。

真实模型阶段会产生 API 调用并把任务文本及为完成任务所选取的代码片段发送到所
配置的服务。只在接受该数据边界时继续。

## 1. 准备环境

在 `AgentProjects` 根目录打开 PowerShell：

```powershell
./scripts/start-docker.ps1
./scripts/build-sandboxes.ps1
./scripts/verify-docker.ps1
py -3.11 -m pip install -e ".[dev]"
python ./benchmarks/validate.py --structure-only
```

验收重点：Docker Server 是 Linux/amd64，两个镜像存在，镜像默认用户为
UID/GID `10001`，结构校验返回成功。任一步失败时先停止，不进入真实模型演示。

## 2. 创建独立 Bug 仓库

下面的命令把 `py-bugfix-002` 的健康 baseline 复制到系统临时目录，应用冻结的
`setup.patch` 注入 Bug，然后提交为一个 clean 仓库。生成的绝对路径保存在当前
PowerShell 会话的 `$demoRepo` 中。

```powershell
$demoRoot = Join-Path `
  ([IO.Path]::GetTempPath()) `
  ("repo-agent-demo-" + [Guid]::NewGuid().ToString("N"))
$demoRepo = Join-Path $demoRoot "target-repository"
$fixture = Resolve-Path `
  ".\benchmarks\tasks\python\bugfix\end-offset"

New-Item -ItemType Directory -Path $demoRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $fixture "baseline") `
  -Destination $demoRepo `
  -Recurse

git -C $demoRepo init
git -C $demoRepo config user.name "Repo Agent Demo"
git -C $demoRepo config user.email "demo@example.invalid"
git -C $demoRepo add --all
git -C $demoRepo commit -m "healthy baseline"
git -C $demoRepo apply (Join-Path $fixture "setup.patch")
git -C $demoRepo add --all
git -C $demoRepo commit -m "reproduce end-offset bug"

git -C $demoRepo status --short
git -C $demoRepo log --oneline -2
$demoHead = git -C $demoRepo rev-parse HEAD
```

`git status --short` 应无输出；日志应包含刚创建的两个提交。不要把
`hidden_tests`、`gold.patch` 或 `metadata.json` 复制进 `$demoRepo`，这些内容属于
独立评分边界，不应提供给 Agent。

## 3. 运行无 Key 只读路径

```powershell
repo-agent run `
  --repo $demoRepo `
  --task "Inspect the repository structure and TODO markers" `
  --provider demo `
  --format json
```

检查 JSON 中的 `status`、工具调用和仓库地图。这个模式只能读取已提交快照，不会
生成或应用修复。

也可以先启动可视化只读工作台：

```powershell
$demoData = Join-Path $demoRoot "read-only-data"
repo-agent serve `
  --repo $demoRepo `
  --data-dir $demoData `
  --port 8765 `
  --open
```

在页面选择 **Read-only demo** 并运行任务。观察环境 readiness、仓库 commit、
工具 trace 和 dirty-worktree 提示。完成后在启动服务的终端按 `Ctrl+C`，确保服务
退出后再继续，避免端口或 data directory 被两个进程同时占用。

## 4. 配置并验证真实模型

当前 Windows 中转站辅助脚本会以星号遮罩输入 Key，发现模型，执行原生 function
calling doctor，并将配置用 DPAPI CurrentUser 加密保存：

```powershell
./scripts/start-relay.ps1 -Reconfigure -ConfigureOnly
```

若配置已经保存，无需再次输入；需要主动复验时运行：

```powershell
./scripts/start-relay.ps1 -Repository $demoRepo -Verify -OpenBrowser
```

普通启动不会重复要求 Key 或模型：

```powershell
./scripts/start-relay.ps1 -Repository $demoRepo -OpenBrowser
```

`-Verify` 和真实任务会产生模型调用。脚本占用前台终端直到 UI 退出，并在退出时
恢复它设置的进程级模型环境变量。

## 5. 在 UI 中完成维护流程

1. 选择 **Maintenance** 模式。
2. 输入任务：`Fix the end-offset boundary bug and add a focused regression test.`
3. 对远程中转站任务显式启用 **Remote model** 授权，再创建 run。
4. 等待状态进入 `awaiting_approval`，阅读计划中的目标文件、测试与风险。
5. 计划范围合理时批准；范围错误时拒绝，不要为了演示强行通过。
6. 观察 `implement -> verify -> review -> finalize`。若检查失败，工作流可进入有界
   repair；模型或检查失败也应以结构化终态和产物呈现。
7. 下载 `patch.diff`、`report.md`、`run.json`、`trace.jsonl` 和 `checks/*.log`。

本演示不预设模型一定修复成功。可信的演示结论应来自页面状态、确定性 check
记录和实际补丁内容，而不是自然语言总结。

## 6. 检查安全不变量

停止 UI 服务后，在原 PowerShell 会话运行：

```powershell
git -C $demoRepo status --short
$currentDemoHead = git -C $demoRepo rev-parse HEAD
if ($currentDemoHead -ne $demoHead) {
  throw "The source repository HEAD changed during the demo."
}
docker ps `
  --filter "name=repo-agent" `
  --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
```

预期结果：

- `$demoRepo` 仍为 clean，HEAD 仍是 `reproduce end-offset bug` 对应提交；
- Agent 没有把候选补丁应用、提交或推送到 `$demoRepo`；
- 当前运行拥有的短生命周期容器已经回收；
- 补丁只存在于运行产物中，等待人工审阅。

进一步运行 Day 6 离线安全验收：

```powershell
python ./scripts/smoke-day6.py
```

该命令输出机器可读 JSON，并检查容器身份、只读 rootfs、写入边界、网络隔离、
资源限制、无宿主 Key/Docker socket、canary 隔离和精确超时清理。以进程退出码和
JSON 中每项检查为准，不要只截取一行日志作为结论。

## 7. 在独立副本中复核补丁

从 UI 下载 `patch.diff` 后，把其绝对路径赋给 `$patchPath`。只在新的 review clone
中做 apply 和验证：

```powershell
$reviewRepo = Join-Path $demoRoot "review-repository"
git clone --no-local $demoRepo $reviewRepo
git -C $reviewRepo apply --check $patchPath
git -C $reviewRepo apply $patchPath
git -C $reviewRepo diff --check
git -C $reviewRepo diff --stat
git -C $reviewRepo config user.name "Repo Agent Demo Reviewer"
git -C $reviewRepo config user.email "reviewer@example.invalid"
git -C $reviewRepo add --all
git -C $reviewRepo commit -m "apply candidate in isolated review clone"
repo-agent check --repo $reviewRepo --format json
```

若该 fixture 需要 bootstrap，先审核依赖描述，再为最后一条命令增加
`--allow-bootstrap`。上述 commit 只发生在临时 review clone 中，用于满足 checker 的
clean-worktree 前置条件。只有补丁内容、回归测试和独立检查均符合预期后，才由人
决定是否在真实项目中应用。

## 8. 演示结束

临时目录由 `$demoRoot` 标识。确认其中没有还需保留的 patch 或日志后，可由用户
自行删除。项目不会自动删除用户下载的产物，也不会运行全局 Docker prune。
