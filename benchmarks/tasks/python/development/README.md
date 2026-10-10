# Python development tasks

These four small repositories are the only prompt-development fixtures for the
active Python/pytest v1 profile. They exercise feature work that is broader
than the frozen bug-fix suite while remaining fast enough for local iteration:

| ID | Directory | Intended behavior |
|---|---|---|
| `py-development-001` | `configurable-path-exclusions/` | Add validated repository scan exclusions with deterministic matching. |
| `py-development-002` | `json-report-exporter/` | Add a stable structured report without changing the text report. |
| `py-development-003` | `tool-timeout-budget/` | Add per-tool timeout budgeting and explicit timeout results. |
| `py-development-004` | `severity-summary/` | Aggregate findings by severity while preserving source order. |

Each task contains a clean `baseline/`, the prompt in `issue.md`, withheld
acceptance tests in `hidden_tests/`, a reviewed `gold.patch`, and metadata. The
independent `manifest.json` locks all four task directories. They are not part
of the active `repo-agent-python-bugfix-v1` formal selection and never count
toward its 4/6 release threshold. They are also not part of the retained legacy
12-task manifest or its read-only 44-run result set.

The restricted `scripts/start-relay.ps1 -Evaluate` entry point runs exactly
these four tasks and accepts the development gate only at 3/4 solved, complete
usage, no timeout or budget failure, and `actionable_tool_error_rate <= 5%`.
After the code, prompt, model, and budgets are frozen,
`scripts/start-relay.ps1 -FormalEvaluate` selects only `py-bugfix-001` through
`py-bugfix-006`; it never promotes these development fixtures into formal data.

Run the complete development lifecycle validation from this directory or the
repository root:

```powershell
python ./benchmarks/tasks/python/development/validate.py
```

For a fast structure and content-lock check that does not execute task code:

```powershell
python ./benchmarks/tasks/python/development/validate.py --structure-only
```

When intentionally changing a fixture, review newly calculated hashes before
updating the development manifest:

```powershell
python ./benchmarks/tasks/python/development/validate.py --print-digests
```

The validator proves that each baseline's public tests pass, its hidden
acceptance test fails, and both public and hidden tests pass after applying the
reference patch. Hidden tests and reference patches must never be shown to a
candidate during prompt development.
