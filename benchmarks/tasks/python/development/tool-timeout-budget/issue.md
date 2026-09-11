# Add per-tool timeout budgeting

`tasklib.tools.run_tool` currently lets a timeout exception escape and reports
an operation as completed even when it consumed more than its assigned budget.
Callers need an explicit result for both cases.

Keep the existing API and add a keyword-only `timeout_seconds` argument with a
default of 30 seconds plus an injectable keyword-only `clock` for deterministic
tests. The action still receives the validated timeout value. Reject booleans,
zero, negative, and non-finite budgets with `ValueError`. Catch `TimeoutError`
and return a `ToolResult` whose status is `timed_out`, output is empty, and error
is `tool timed out`. Also return that explicit timeout result if elapsed time is
greater than the budget, even when the action returned normally. Preserve the
existing completed result inside budget, and do not add threads or dependencies.
