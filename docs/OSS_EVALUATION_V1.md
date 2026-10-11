# Public Python Issue Evaluation

This page records the first and only executed trial of the frozen
`repo-agent-python-oss-v1` suite. It is evidence for the Python-only v1
portfolio release, not a claim about general software-engineering ability.

## Evaluation identity

| Field | Value |
| --- | --- |
| Date | 2026-10-11 (Asia/Shanghai) |
| Source commit | `394fb21e467efdd701570686b8afefeb20c95bd2` |
| Suite | `repo-agent-python-oss-v1` |
| Manifest SHA-256 | `d2f50704618f598cee5f97349b72ee9a4b2cd905a24f6dffa3076f04a7d23d98` |
| Variant / trial | `full` / `1` |
| Model | `deepseek-v4-pro`, official DeepSeek provider |
| Approval | automatic benchmark approval; no manual patch edits |
| Task timeout | 1,200 seconds per task |
| Token budget | 30,000 per task: 4,000 planning, 23,500 implementation/repair, 2,500 review |
| Raw summary SHA-256 | `afe317479cceb586245b7f650e8dca846eb7c8b9b717eb4a450e3e3b037e76ba` |

The raw run directory is intentionally Git-ignored because it contains
workspaces, SQLite service state, and verbose traces. The sanitized,
machine-readable aggregate committed with this report is
[`benchmarks/results/oss-v1/report.json`](../benchmarks/results/oss-v1/report.json).
A bounded credential scan covered 40 text artifacts and found zero matches for
API-key, bearer-token, or DPAPI-ciphertext patterns.

## Measured result

All three tasks completed. One task was solved, giving pass@1 of **1/3
(33.3%)**. There were no task timeouts and no budget failures. Aggregate usage
was 43,132 tokens across the three tasks; provider pricing was unavailable, so
cost is not reported. Median task latency was 437.668 seconds and p95 was
546.199 seconds. The actionable tool error rate was 10% (1/10 tool calls).

| Task | Public issue | Outcome | First verify | Repair | Tokens | Duration |
| --- | --- | --- | --- | --- | ---: | ---: |
| `py-oss-001` | PySnooper #2 | Failed public/hidden/regression candidate checks | Failed | One repair, still failed | 13,639 | 558.258 s |
| `py-oss-002` | HTTPie #235 | Failed public/hidden/regression candidate checks | Failed | Repair response rejected as incomplete | 22,539 | 437.668 s |
| `py-oss-003` | The Fuck #271 | Solved: public, hidden, regression, and policy checks passed | Passed | None | 6,954 | 296.200 s |

The failures are retained rather than retried:

- `py-oss-001` fixed the undefined path variable, but the generated regression
  test called an internal function through a package export that did not exist.
  The candidate therefore failed its own public test after the one allowed
  repair.
- `py-oss-002` made Host detection case-insensitive, but its generated test
  imported a dependency module absent from the curated snapshot. Verification
  failed during collection; the repair response then hit
  `max_output_tokens` and was rejected by the protocol layer.

These results show a working bounded workflow and one independently scored
repair, but also show that model-authored tests are a material failure mode.

## Release-candidate checks

The documentation commit was prepared only after these local gates passed on
2026-10-11:

- Ruff: passed;
- Mypy: passed for 29 source files;
- full test suite: `1483 passed, 9 skipped`;
- combined line/branch coverage report: `84.92%` (80% gate);
- frozen Python bugfix lifecycles: `6/6`;
- frozen public-issue lifecycles: `3/3`;
- Day 6 Docker security probes: `6/6`, with zero host mutations, zero canary
  leaks, and zero run-labeled residual containers.

The Day 6 run also inspected the non-root UID, `network=none`, read-only root
filesystem, CPU/memory/PID limits, dropped capabilities,
`no-new-privileges`, `--init` descendant reaping, exact timeout cleanup, and
absence of Docker socket or model-provider environment variables. This is a
bounded test result, not a comprehensive security certification.

## Showcase: The Fuck #271

The upstream regression was a substring match in the `ls_lah` rule. Commands
such as `pacman -S binutils` or `lsof` were incorrectly treated as `ls`
commands. The Agent:

1. completed the clean-repository and baseline checks;
2. proposed a two-file plan and received benchmark auto-approval;
3. read the target and tests through the bounded tool surface;
4. changed the rule to require the first command token to be `ls`;
5. added regression cases for unrelated commands, valid `ls` invocations, and
   existing option exclusions;
6. passed Docker verification on the first attempt (`11 passed`);
7. passed independent review and finalization; and
8. passed withheld public, hidden, regression-lifecycle, and policy scoring.

The production change was:

```diff
-    return 'ls' in command.script and not ('ls -' in command.script)
+    return command.script.split(' ')[0] == 'ls' and not ('ls -' in command.script)
```

The candidate changed two files and 36 lines in total, including its new test
file. Its patch SHA-256 is
`48d341bf043a3b7d3a894dbc519b8c4ed3e36246b06f940d44b1156c52bf1d26`.
No repair or human patch modification occurred.

## Evidence boundaries

- The three tasks are curated source snapshots with exact upstream target files
  and licenses, not complete historical checkouts of the upstream projects.
- Public and hidden tests were authored for this project. They do not represent
  the full upstream test suites.
- This is one frozen trial on three previously resolved public issues, not an
  uncontaminated holdout and not a statistically stable success estimate.
- Workflow `succeeded` and benchmark `solved` are separate. Only `py-oss-003`
  satisfied every independent scoring condition.
- Cost is unmeasured because the provider response did not supply pricing and
  the evaluator does not estimate it.
- The run was not repeated after observing failures, and no candidate patch was
  manually edited.
