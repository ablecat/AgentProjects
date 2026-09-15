# Frozen Python and Java benchmark suite

This directory contains twelve frozen bug-fix fixtures: six Python tasks from
Day 2 and six single-module Maven tasks added on Day 3. The suite is independent
from the model runtime and never needs a model key. Python lifecycle validation
uses only the standard library; Java lifecycle validation uses the pinned local
Maven sandbox image and an explicitly authorized dependency bootstrap.

## Layout

```text
benchmarks/
  manifest.json                 suite index and task content locks
  task.schema.json              machine-readable task metadata contract
  validate.py                   structural and behavioral validator
  tasks/python/bugfix/<slug>/
    baseline/                   healthy repository snapshot
    setup.patch                 injects the issue into the snapshot
    issue.md                    prompt shown to the evaluated agent
    hidden_tests/               tests withheld from the agent
    gold.patch                  reference repair, applied after setup.patch
    metadata.json               task contract instance
  tasks/python/development/     reserved locations for four later tasks
  tasks/java/bugfix/<slug>/
    baseline/                   healthy Java 17 single-module Maven project
    setup.patch                 injects the issue into production code
    issue.md                    prompt shown to the evaluated agent
    hidden_tests/               JUnit tests withheld from the agent
    gold.patch                  production repair plus a new *RegressionTest
    metadata.json               Maven lifecycle contract instance
```

The healthy baseline must pass both public and hidden tests. After applying
`setup.patch`, public tests must still pass while the named hidden tests fail.
After applying `gold.patch`, both suites must pass again. Python gold patches
restore the healthy baseline byte for byte. Java gold patches restore production
sources and the `pom.xml`, and also add at least one focused `*RegressionTest`.
The validator applies only that new test to the buggy setup and proves it fails
before accepting the repaired state.

## Validate

Run the complete validation from the repository root. The flag is an explicit
authorization for a fixed `maven-dependency-plugin:go-offline` command followed
by allowlisted `dependency:get` calls for the pinned Surefire and JUnit Platform
runtime artifacts:

```powershell
python ./benchmarks/validate.py --allow-bootstrap-network
```

Each bootstrap step runs in a short-lived hardened Maven container with network
access but does not execute repository tests. Before any network access, the
validator parses each fixture POM and requires the exact project, dependency,
plugin, and bootstrap-profile structure; parent POMs, custom repositories,
extensions, and extra build inputs cannot enter `go-offline`. A timed-out step,
or the specific Maven transport error for a truncated `Content-Length` download,
is retried once with the same run-scoped cache. Other failures are deterministic
and fail immediately. Every healthy, setup, hidden, regression, and gold
lifecycle test runs afterward in a fresh copy with `--network none` and Maven
`--offline`. Containers use the same non-root, read-only-rootfs, CPU, memory,
PID, capability, and privilege restrictions as the runtime sandbox and are
removed by exact generated identity.

For a quick contract and lock check without Docker, network, or task execution:

```powershell
python ./benchmarks/validate.py --structure-only
```

`manifest.json` locks each of the twelve task directories with a canonical SHA-256
digest.  When a fixture is intentionally edited, print the newly calculated
digests with `--print-digests`, review the diff, and update the corresponding
manifest entry explicitly. Text files are hashed with LF line endings, and
generated Python/test cache directories, Maven `target` directories, and compiled
Python artifacts are not part of the frozen task content.

## Evaluation boundary

An evaluator should copy `baseline/`, apply `setup.patch`, and expose only that
working copy, its public tests, and `issue.md` to the candidate Agent.  It must
not expose `hidden_tests/`, `gold.patch`, or the expected failure signatures in
`metadata.json`.  Evaluation then applies the candidate patch to a fresh setup
state and runs both public and hidden tests with network access disabled. For
Java, it must also verify that a candidate-added regression test fails against
the buggy setup before running that test against the repaired candidate.

## Independent scoring contract

A workflow terminal state is not a benchmark verdict. The trusted scorer
reconstructs a fresh buggy setup, validates and applies the candidate patch, and
requires all of the following before marking a task solved:

- the original tests and supported build descriptors were not weakened;
- a focused regression test was added where required;
- that regression test fails on the buggy setup and passes on the candidate;
- public and hidden tests pass with `network=none`;
- patch, tool-call, token, time, resource, and security policies pass.

Candidate patches, per-task state, model usage, failure categories, and scorer
evidence are stored outside this frozen task tree. Unknown token prices must not
be guessed: omit both price options so raw token usage is retained while cost
remains unavailable.

## Evaluation CLI

Inspect one variant or the complete locked matrix without calling a model:

```powershell
repo-agent eval --variant full --format json
$projectParent = Split-Path -Parent (Get-Location).Path
$evaluationRoot = Join-Path $projectParent ".repo-agent-eval-day7-v1"
repo-agent eval --matrix-only `
  --output-dir $evaluationRoot `
  --format json
```

The first command validates a single 12-task selection and reports
`execution_started=false`. The second emits all 44 unique `(variant, trial,
task_id)` jobs, the fixed five-shard count, and the worker limit. Capture and
review that matrix before starting the formal run. It also atomically writes or
validates `$evaluationRoot\matrix.json`; a mismatched existing manifest is
rejected.

Run one task through all three variants as a paid canary. Canary state lives
under a separate directory and is never merged into the formal 44 jobs:

```powershell
repo-agent eval `
  --canary --execute `
  --allow-remote-model `
  --allow-bootstrap `
  --workers 2 `
  --output-dir $evaluationRoot `
  --format json
```

Each task has a 20-minute hard Agent deadline, at most 30 repository tool calls,
and at most 30,000 reported tokens. Single-variant `--variant ... --execute`
runs remain available for focused debugging but are not a replacement for the
locked formal matrix.

The four fixtures under `tasks/python/development` are explicitly non-formal.
They may be run only through single-variant mode for prompt and workflow tuning;
matrix, canary, shard, and merge paths reject them. On Windows, the restricted
`scripts/start-relay.ps1 -Evaluate` entry point loads the DPAPI-protected model
configuration and runs the fixed four-task `full` command. It never places the
key in command arguments or evaluation artifacts.

## Locked 44-run experiment

The formal comparison is fixed before execution:

| Variant | Trial 1 | Trials 2 and 3 | Total |
|---|---:|---:|---:|
| `baseline` | all 12 tasks | not allowed | 12 |
| `no-review` | all 12 tasks | not allowed | 12 |
| `full` | all 12 tasks | four locked tasks per trial | 20 |
| **Total** | **36** | **8** | **44** |

`baseline` uses the basic tool loop without repository-map, repair, independent
review, or checkpoints. `no-review` keeps the durable workflow, repository-map,
and repair but disables independent review. `full` enables all of those
features.

The repeated full tasks are locked to:

- `py-bugfix-003`
- `py-bugfix-004`
- `java-bugfix-003`
- `java-bugfix-006`

After model configuration, fixture lifecycle validation, and the canary,
execute the matrix into one dedicated output root. The five shards are run
sequentially here so total model concurrency never exceeds two:

```powershell
0..4 | ForEach-Object {
  repo-agent eval `
    --matrix --execute `
    --shard-index $_ --shard-count 5 --workers 2 `
    --allow-remote-model --allow-bootstrap `
    --output-dir $evaluationRoot `
    --format json
  if ($LASTEXITCODE -ne 0) { throw "Evaluation shard $_ failed." }
}
```

Completed per-task state is durable. Rerunning a shard with the same suite,
model, pricing policy, and output root skips completed jobs and continues
incomplete work; incompatible metadata is rejected rather than mixed. Do not
delete or rerun failures and timeouts under the same identity merely to improve
the score.

Only after all five shards complete, run the strict merge into a separate,
sanitized publish directory:

```powershell
repo-agent eval `
  --merge `
  --output-dir $evaluationRoot `
  --report-dir .\benchmarks\results\v1 `
  --format json
```

Merge rejects missing, duplicate, unexpected, misplaced, or lock-mismatched
records. It writes `results.jsonl`, reads the file back, requires exactly 44
rows in matrix order, and derives `report.json` and `failure-analysis.json`
from those persisted rows. Trials 2 and 3 contain exactly the four locked
repeat tasks.

The output root is intentionally outside both the application data directory
and the Git repository. It contains the canary, five shard directories,
resumable per-task state, candidate workspaces and patches, traces, scorer
evidence, and intermediate summaries; none of it may be committed. The
repository publish directory `benchmarks/results/v1` contains only the merged
`results.jsonl`, `report.json`, and `failure-analysis.json`, with private state
and machine-local paths excluded. Treat `results.jsonl` as the source of truth for pass@1, per-variant
and per-language success, test-restoration rate, tool error rate, patch size,
token use, p50/p95 latency, and failure classification.

Do not pass `--input-cost-per-million` or `--output-cost-per-million` unless the
price source has been independently verified. Without them, every result still
records input, cached-input, output, and total tokens while `cost_usd` remains
`null` and `price_source` is `unavailable`.

## Claims policy

The immutable v1 result has a documented pytest `SUBFAILED` attribution issue.
See [`ERRATA.md`](ERRATA.md). New acceptance output must live in a new directory
and be labeled v1 regression acceptance rather than an independent holdout.

Publishing raw results does not automatically justify a positive architecture
claim. The predeclared thresholds are:

- `full` solves at least 8/12 trial-one tasks, including at least 4/6 in each
  language, before the project claims the target quality bar was met;
- `full` solves at least two more trial-one tasks than `baseline` before it
  claims the complete architecture improved success;
- `full` solves at least one more trial-one task than `no-review` before it
  claims independent review helped; otherwise the neutral or negative result is
  reported;
- all six security probes pass with zero host mutations and zero canary leaks,
  and all three checkpoint-boundary recovery probes pass without repeating the
  protected side effect;
- p95 latency of at most eight minutes is a target, while 20 minutes is the
  mandatory hard stop for every run.

Failed, timed-out, and policy-denied trials remain in `results.jsonl`. Any
number used in a README, report, resume, or interview should be recomputable
from that file rather than copied from an ad hoc console summary.
