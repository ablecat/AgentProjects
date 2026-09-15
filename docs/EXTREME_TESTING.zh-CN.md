# 极端测试与故障注入

本文记录 Repo Maintainer Agent 的高风险边界、自动化入口和验收标准。极端测试只对
临时仓库、测试进程和带唯一名称的测试容器执行；不得挂载宿主 HOME、Docker socket、
SSH 配置或原始待维护仓库。

## 测试矩阵

| ID | 场景 | 预期结果 | 自动化位置 |
|---|---|---|---|
| HTTP-01 | `Content-Length` 大于实际 JSON 字节数，客户端提前结束发送 | 返回 400，不能创建任务或调用检查器 | `tests/test_web_edges.py` |
| HTTP-02 | 已认证请求体超过 64 KiB | 返回 413，业务 service 不被调用 | `tests/test_api_edges.py` |
| HTTP-03 | 未认证的超大请求 | 先返回 401，不读取和缓存攻击正文 | `tests/test_api_edges.py` |
| HTTP-04 | 分块正文累计越界、重复或畸形长度头、正文总超时 | 分别返回 413、400、408 | `tests/test_api_edges.py` |
| HTTP-05 | 正文恰好等于上限并被拆成多个 ASGI message | 完整重放给应用，字节不丢失 | `tests/test_api_edges.py` |
| HTTP-06 | 旧 Web API 收到重复或冲突的 `Content-Length` | 返回 400，不能创建任务或调用检查器 | `tests/test_web_edges.py` |
| INPUT-01 | 任务或审批理由含 NUL | 所有入口在持久化前拒绝 | `tests/test_web_edges.py`、`tests/test_run_service.py` |
| TIME-01 | `NaN`、正负无穷、超大有限数、不可转为浮点的整数、负数、布尔或字符串 timeout | 立即抛出 `ValueError`，不能等待或改变 lease | `tests/test_run_service.py`、`tests/test_sandbox.py` |
| PROC-01 | 父命令超时但普通派生子进程继续运行 | 终止完整 OS ownership boundary，限定时间内返回 | `tests/test_sandbox.py` |
| PROC-02 | 父命令退出，子进程仍继承 stdout 管道 | 终止子进程，输出 reader 不得无限 join | `tests/test_sandbox.py` |
| PROC-03 | Windows Job 配置失败 | suspended 子进程在执行用户代码前终止，启动必须 fail closed | `tests/test_processes.py`、`tests/test_sandbox.py` |
| PROC-04 | 输出线程构造/启动失败、stdout/stderr 超量、stdin 或 pipe I/O 失败、直接子进程无法 reap | 已启动的 ownership boundary 必须终止并回收；`limit+1` 字节立即触发终止；清理失败必须显式报错 | `tests/test_processes.py` |
| PROC-05 | `git -z` 输出在路径中间被截断 | 丢弃未以 NUL 结束的记录，不得把路径前缀当作真实文件 | `tests/test_repository_map_edges.py` |
| RACE-01 | 审批发布、主动取消、终态 artifact 写入和 close 并发 | 终态不可被旧状态覆盖，artifact 与数据库一致 | `tests/test_run_service.py`、`tests/test_artifacts.py` |
| MODEL-01 | 重复 JSON key、多 tool call、未知工具、参数超限、响应截断 | fail closed，不执行未经验证的工具 | `tests/test_openai_provider.py` |
| MODEL-02 | Responses 不支持后回退 Chat Completions，usage 缺失或畸形 | 只按明确能力错误回退；远程模型 token 统计不完整即失败 | `tests/test_doctor.py`、`tests/test_workflow_runtime_edges.py` |
| STATE-01 | SQLite/checkpoint 损坏、并发 CAS、服务中断后恢复 | 拒绝损坏状态；活动任务发布为可恢复的 `interrupted` | `tests/test_checkpoint_edges.py`、`tests/test_run_service.py` |
| DOCKER-01 | 无网络、只读 rootfs、非 root、资源限制、超时清理 | 策略全部生效，不遗留容器，不修改宿主仓库 | `scripts/smoke-day6.py` |

## 快速回归

```powershell
python -m pytest -q `
  tests/test_api_edges.py `
  tests/test_web_edges.py `
  tests/test_run_service.py `
  tests/test_sandbox.py `
  tests/test_openai_provider.py `
  tests/test_workflow_runtime_edges.py
```

并发竞态应重复执行，而不是只依赖单次通过：

```powershell
$targets = @(
  "tests/test_run_service.py::test_approval_pause_is_not_published_before_worker_finishes_sync",
  "tests/test_run_service.py::test_active_cancel_cannot_overwrite_terminal_artifact",
  "tests/test_run_service.py::test_close_notifies_an_unbounded_wait_after_publishing_interrupted_state",
  "tests/test_run_service.py::test_close_serializes_a_concurrent_mutation",
  "tests/test_artifacts.py::test_trace_append_atomically_replaces_a_complete_jsonl_snapshot"
)
1..20 | ForEach-Object {
  python -m pytest -q @targets
  if ($LASTEXITCODE -ne 0) { throw "race iteration $_ failed" }
}
```

## 真实容器验收

```powershell
python .\scripts\smoke-day6.py
```

Linux 进程树还应在正式 Python 沙箱镜像中验证，并保留与生产容器相同的 `--init`、
`network=none`、只读文件系统、UID、CPU、内存、PID、capability 和
`no-new-privileges` 限制。`smoke-day6.py` 会在该容器中终止一个继承输出管道的后代，
并确认 `/proc/<pid>` 最终消失，而不是只把 zombie 当作成功。运行结束后，以报告中的
`run_label` 精确确认本轮容器已经清理：

```powershell
$runLabel = "REPORT_RUN_LABEL"
docker ps -a `
  --filter "label=$runLabel" `
  --format "{{.ID}}"
```

只要求该 label 的输出为空；机器上无关容器不影响验收，也不得被停止或删除。报告还
必须给出 `cleanup.containers_removed=true`。禁止用 `docker system prune` 掩盖资源泄漏。

宿主侧只执行固定构造的 Git、Docker、Python 验证命令。Windows 进程先以 suspended
状态创建，成功加入带 `KILL_ON_JOB_CLOSE` 的 Job Object 后才恢复执行；Job 建立失败时
不得降级为普通进程。POSIX 使用独立 session/process group，它只承诺终止正常派生的
后代；Python 只能 `wait` 直接子进程，孤儿 zombie 必须由宿主 init、subreaper 或容器
`--init` 回收。进程组也不是针对恶意 `setsid()` 逃逸的容器边界；不受信任的仓库命令
必须始终留在 Docker 容器内。若 ownership boundary 或 pipe 无法清理，调用必须显式失败。

Windows 专属行为由 CI 的 `windows-process-lifecycle` job 定向覆盖；Docker 隔离仍由
Linux `docker-e2e` job 验证。

## 本轮故障定位

极端测试曾复现并锁定六类缺陷：短 HTTP 正文被接受、REST 正文无总量限制、重复
`Content-Length` 被接受、异常 timeout 导致无限等待或平台异常、Windows Job 挂接
存在启动竞态，以及超时命令的后代和输出 reader 未被有界回收。对应回归用例必须与
修复同存，后续修改不得仅延长 timeout 或在测试后执行全局清理来规避失败。
