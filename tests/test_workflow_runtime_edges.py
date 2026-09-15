from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import repo_agent.workflow_runtime as runtime
from repo_agent.artifacts import ArtifactStore
from repo_agent.models import CandidateArtifact, FinalAnswer, ToolCall, ToolResult
from repo_agent.openai_provider import OpenAIConfig
from repo_agent.persistence import RunDatabase
from repo_agent.processes import CapturedProcess
from repo_agent.run_models import ChangePlan, CheckSummary, RunRecord
from repo_agent.workflow import (
    NodeOutcome,
    OperationContext,
    VerificationOutcome,
    WorkflowSnapshot,
)


RUN_ID = "c" * 32
NOW = "2026-09-11T00:00:00.000Z"


def _record(repository: Path, **updates: object) -> RunRecord:
    values = {
        "run_id": RUN_ID,
        "repo_path": str(repository),
        "task": "Repair the repository",
        "status": "running",
        "current_node": "implement",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(updates)
    return RunRecord.model_validate(values)


def _context(
    record: RunRecord,
    node: str,
    *,
    attempt: int = 0,
    suffix: str = "edge",
) -> OperationContext:
    return OperationContext(
        run=record,
        node=node,
        attempt=attempt,
        idempotency_key=f"{record.run_id}:{node}:{attempt}:{suffix}",
    )


@pytest.fixture
def operations_bundle(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    record = _record(repository)
    database = RunDatabase(tmp_path / "runs.sqlite3")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    artifact_dir = artifacts.initialize(record.run_id)
    database.create(record, artifact_dir)
    operations = runtime.RepositoryWorkflowOperations(
        record,
        database=database,
        artifacts=artifacts,
        cancel_event=threading.Event(),
    )
    return operations, database, artifacts, record


def test_runtime_constructors_require_boolean_review_flag(
    operations_bundle,
) -> None:
    operations, database, artifacts, record = operations_bundle
    del operations
    with pytest.raises(ValueError, match="review_enabled"):
        runtime.RepositoryWorkflowOperations(
            record,
            database=database,
            artifacts=artifacts,
            cancel_event=threading.Event(),
            review_enabled=1,
        )
    with pytest.raises(ValueError, match="review_enabled"):
        runtime.ServiceWorkflowRunner(database, artifacts, review_enabled="yes")


def test_default_provider_factory_uses_remote_model_policy(monkeypatch) -> None:
    config = SimpleNamespace(base_url="http://127.0.0.1:9")
    calls: list[object] = []
    monkeypatch.setattr(
        runtime.OpenAIConfig,
        "from_env",
        lambda *, allow_remote_model: calls.append(allow_remote_model) or config,
    )
    monkeypatch.setattr(
        runtime,
        "OpenAIProvider",
        lambda provided, **kwargs: (provided, kwargs),
    )

    provider = runtime.default_provider_factory(
        (runtime.PLAN_TOOL,),
        "system",
        "idempotent",
        allow_remote_model=True,
    )

    assert calls == [True]
    assert provider[0] is config
    assert provider[1]["tool_definitions"] == (runtime.PLAN_TOOL,)
    assert provider[1]["idempotency_key"] == "idempotent"
    assert provider[1]["reasoning_effort"] is None


@pytest.mark.parametrize(
    "base_url",
    ("https://api.deepseek.com", "https://API.DEEPSEEK.COM:443/v1/"),
)
def test_official_deepseek_workflow_uses_phase_reasoning_effort(
    base_url: str,
) -> None:
    config = OpenAIConfig(
        api_key="fake-deepseek-key",
        base_url=base_url,
        model="deepseek-v4-pro",
        allow_remote_model=True,
    )

    assert runtime._workflow_reasoning_effort(config, (runtime.PLAN_TOOL,)) == "none"
    apply_patch = next(
        definition
        for definition in runtime.AGENT_TOOL_DEFINITIONS
        if definition.name == "apply_patch"
    )
    assert runtime._workflow_reasoning_effort(config, (apply_patch,)) == "low"


@pytest.mark.parametrize(
    ("method_name", "node", "payload", "expected_type"),
    [
        (
            "prepare",
            "prepare",
            NodeOutcome(summary="cached prepare").model_dump(mode="json"),
            NodeOutcome,
        ),
        (
            "baseline_check",
            "baseline_check",
            NodeOutcome(summary="cached baseline").model_dump(mode="json"),
            NodeOutcome,
        ),
        (
            "inspect_and_plan",
            "inspect_and_plan",
            ChangePlan(
                goal="cached plan",
                files=("app.py",),
                steps=("repair",),
                checks=("pytest",),
            ).model_dump(mode="json"),
            ChangePlan,
        ),
        (
            "verify",
            "verify",
            VerificationOutcome(ok=True, status="passed").model_dump(mode="json"),
            VerificationOutcome,
        ),
        (
            "review",
            "review",
            NodeOutcome(summary="cached review").model_dump(mode="json"),
            NodeOutcome,
        ),
        (
            "finalize",
            "finalize",
            NodeOutcome(summary="cached final").model_dump(mode="json"),
            NodeOutcome,
        ),
    ],
)
def test_nodes_restore_cached_results_without_repeating_side_effects(
    operations_bundle,
    method_name: str,
    node: str,
    payload: dict[str, object],
    expected_type: type,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    context = _context(record, node)
    database.record_side_effect(
        record.run_id,
        f"workflow:{context.idempotency_key}",
        payload,
    )

    result = getattr(operations, method_name)(context)

    assert isinstance(result, expected_type)
    assert result.model_dump(mode="json") == payload


def test_change_node_ignores_legacy_patch_embedded_in_cached_side_effect(
    operations_bundle,
) -> None:
    operations, database, artifacts, record = operations_bundle
    context = _context(record, "implement", suffix="cached-patch")
    cached = NodeOutcome(
        summary="cached change",
        detail={"patch": "cached diff\n"},
    ).model_dump(mode="json")
    database.record_side_effect(
        record.run_id,
        f"workflow:{context.idempotency_key}",
        cached,
    )

    result = operations.implement(context)

    assert result.summary == "cached change"
    assert "patch" not in result.detail
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == ""


def test_prepare_rejects_dirty_mismatched_and_linked_repositories(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    monkeypatch.setattr(runtime, "effective_core_autocrlf", lambda _repo: "false")

    scenarios = [
        (["?? untracked.txt"], "worktree must be clean"),
        (["", "a" * 40, "b" * 40], "base_ref must currently resolve"),
        (["", "a" * 40, "a" * 40, "120000 abc 0\tlink"], "symbolic links"),
    ]
    for index, (responses, message) in enumerate(scenarios):
        iterator = iter(responses)
        monkeypatch.setattr(runtime, "_git", lambda *_args: next(iterator))
        context = _context(record, "prepare", suffix=str(index))

        result = operations.prepare(context)

        assert result.ok is False
        assert message in (result.error or "")
        assert result.detail["failure_status"] == "policy_denied"


def test_finalize_rejects_empty_candidate(operations_bundle) -> None:
    operations, _database, _artifacts, record = operations_bundle

    result = operations.finalize(_context(record, "finalize", suffix="empty"))

    assert result.ok is False
    assert result.error == "candidate patch is empty"


class _DecisionProvider:
    def __init__(self, *decisions, usage=None) -> None:
        self.decisions = list(decisions)
        self.usage = usage
        self.response_token_budgets: list[int] = []
        self.forced_tool_choices: list[str] = []

    def next_step(self, _task, _results):
        return self.decisions.pop(0)

    def set_response_token_budget(self, total_tokens: int) -> None:
        self.response_token_budgets.append(total_tokens)

    def set_forced_tool_choice(self, tool_name: str) -> None:
        self.forced_tool_choices.append(tool_name)


class _FailingProvider:
    def __init__(self, *decisions: ToolCall, total_tokens: int = 17) -> None:
        self.decisions = list(decisions)
        self.usage = SimpleNamespace(
            response_count=len(decisions),
            reported_response_count=len(decisions),
            input_tokens=total_tokens - 1,
            cached_input_tokens=0,
            output_tokens=1,
            total_tokens=total_tokens,
            complete=True,
        )

    def next_step(self, _task, _results):
        if self.decisions:
            return self.decisions.pop(0)
        raise RuntimeError("provider failed after reporting usage")


class _BrokenUsage:
    @property
    def total_tokens(self):
        raise RuntimeError("usage getter failed")


class _Executor:
    def __init__(self, *, advance_revision: bool = False) -> None:
        self.workspace_revision = 0
        self.advance_revision = advance_revision
        self.calls: list[ToolCall] = []

    def execute(self, decision: ToolCall) -> ToolResult:
        self.calls.append(decision)
        if self.advance_revision:
            self.workspace_revision += 1
        return ToolResult(decision.id, decision.name, True, "ok")


class _ParallelViolationProvider(_DecisionProvider):
    def __init__(self, decision: ToolCall, violations: int) -> None:
        super().__init__(decision)
        self.parallel_tool_call_violations = 0
        self._violations = violations

    def next_step(self, task, results):
        self.parallel_tool_call_violations += self._violations
        return super().next_step(task, results)


def test_planning_parallel_violations_consume_repository_tool_budget() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]}),
        violations=2,
    )
    executor = _Executor()

    with pytest.raises(RuntimeError, match="exceeded 2 repository tool calls"):
        runtime._run_until_named_tool(
            provider,
            "inspect",
            executor,
            target="submit_change_plan",
            max_calls=2,
        )

    assert executor.calls == []


def test_change_parallel_violations_consume_repository_tool_budget() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]}),
        violations=8,
    )
    executor = _Executor()

    outcome = runtime._run_model_loop(
        provider,
        "implement",
        executor,
        max_calls=8,
    )

    assert outcome.ok is False
    assert outcome.tool_budget_exhausted is True
    assert outcome.error == "model exceeded 8 repository tool calls"
    assert executor.calls == []


def test_planning_reentry_counts_existing_results_and_parallel_violations() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read-2", "read_files", {"paths": ["other.py"]}),
        violations=0,
    )
    provider.parallel_tool_call_violations = 1
    executor = _Executor()
    results = [ToolResult("read-1", "read_files", True, "first")]

    with pytest.raises(RuntimeError, match="exceeded 2 repository tool calls"):
        runtime._run_until_named_tool(
            provider,
            "inspect",
            executor,
            target="submit_change_plan",
            max_calls=2,
            tool_results=results,
        )

    assert executor.calls == []


def test_change_reentry_counts_existing_results_and_parallel_violations() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read-2", "read_files", {"paths": ["other.py"]}),
        violations=0,
    )
    provider.parallel_tool_call_violations = 1
    executor = _Executor()
    results = [ToolResult("read-1", "read_files", True, "first")]

    outcome = runtime._run_model_loop(
        provider,
        "implement",
        executor,
        max_calls=2,
        tool_results=results,
    )

    assert outcome.ok is False
    assert outcome.tool_budget_exhausted is True
    assert outcome.error == "model exceeded 2 repository tool calls"
    assert executor.calls == []


def test_named_tool_loop_accepts_target_after_repository_tool() -> None:
    inspect = ToolCall("read", "read_file", {"path": "app.py"})
    submit = ToolCall("plan", "submit_change_plan", {})
    executor = _Executor(advance_revision=True)

    decision, results = runtime._run_until_named_tool(
        _DecisionProvider(inspect, submit),
        "task",
        executor,
        target="submit_change_plan",
        max_calls=1,
    )

    assert decision is submit
    assert len(results) == 1


def test_named_tool_loop_rejects_text_repetition_and_budget_exhaustion() -> None:
    call = ToolCall("read", "read_file", {"path": "app.py"})
    with pytest.raises(RuntimeError, match="returned text"):
        runtime._run_until_named_tool(
            _DecisionProvider(FinalAnswer("done")),
            "task",
            _Executor(),
            target="submit_change_plan",
            max_calls=1,
        )
    with pytest.raises(RuntimeError, match="exceeded 0"):
        runtime._run_until_named_tool(
            _DecisionProvider(call),
            "task",
            _Executor(),
            target="submit_change_plan",
            max_calls=0,
        )
    with pytest.raises(RuntimeError, match="repeated"):
        runtime._run_until_named_tool(
            _DecisionProvider(call, call),
            "task",
            _Executor(),
            target="submit_change_plan",
            max_calls=2,
        )


def test_model_loop_supports_text_and_valid_finish() -> None:
    text = runtime._run_model_loop(
        _DecisionProvider(FinalAnswer("complete")),
        "task",
        _Executor(),
        max_calls=0,
    )
    finished = runtime._run_model_loop(
        _DecisionProvider(
            ToolCall(
                "finish",
                "finish",
                {"summary": "  repaired  ", "risks": ["small"]},
            )
        ),
        "task",
        _Executor(),
        max_calls=0,
    )

    assert text.ok is True
    assert text.summary == "complete"
    assert finished.ok is True
    assert finished.summary == "repaired"
    assert finished.risks == ("small",)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"summary": "ok"}, "finish arguments are invalid"),
        ({"summary": "", "risks": []}, "finish arguments have invalid types"),
        ({"summary": "ok", "risks": "none"}, "finish arguments have invalid types"),
        ({"summary": "ok", "risks": [1]}, "finish arguments have invalid types"),
    ],
)
def test_model_loop_rejects_invalid_finish(arguments, error: str) -> None:
    outcome = runtime._run_model_loop(
        _DecisionProvider(ToolCall("finish", "finish", arguments)),
        "task",
        _Executor(),
        max_calls=0,
    )
    assert outcome.ok is False
    assert outcome.error == error


def test_model_loop_reports_tool_budget_and_repetition() -> None:
    call = ToolCall("read", "read_file", {"path": "app.py"})
    exhausted = runtime._run_model_loop(
        _DecisionProvider(call), "task", _Executor(), max_calls=0
    )
    repeated = runtime._run_model_loop(
        _DecisionProvider(call, call), "task", _Executor(), max_calls=2
    )
    progressed = runtime._run_model_loop(
        _DecisionProvider(call, FinalAnswer("done")),
        "task",
        _Executor(advance_revision=True),
        max_calls=1,
    )

    assert exhausted.error == "model exceeded 0 repository tool calls"
    assert repeated.error == "model repeated an identical tool call"
    assert len(repeated.tool_results) == 1
    assert progressed.ok is True
    assert len(progressed.tool_results) == 1


def test_run_budget_aggregates_only_valid_model_events(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(record.run_id, {"event": "unrelated", "tool_calls": 100})
    database.append_event(
        record.run_id,
        {
            "event": "planning_tool_summary",
            "phase": "planning",
            "tool_calls": 5,
            "model_usage": {"complete": True, "total_tokens": 100},
        },
    )
    assert operations._model_budget(_context(record, "implement")) == (
        runtime.MAX_RUN_TOOL_CALLS - 5,
        100,
    )

    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": 2,
            "model_usage": {"complete": False, "total_tokens": -1},
        },
    )
    assert operations._model_budget(_context(record, "implement")) == (
        runtime.MAX_RUN_TOOL_CALLS - 7,
        None,
    )


@pytest.mark.parametrize(
    "event",
    (
        {
            "event": "model_tool_summary",
            "tool_calls": 1,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
        {
            "event": "planning_tool_summary",
            "phase": "implement",
            "tool_calls": 1,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": True,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": -1,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
    ),
)
def test_run_budget_rejects_malformed_summary_events(
    operations_bundle, event: dict[str, object]
) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(record.run_id, event)

    with pytest.raises(RuntimeError, match="budget event phase|tool usage"):
        operations._model_budget(_context(record, "implement"))


def test_run_budget_stops_at_global_tool_and_token_limits(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": runtime.MAX_RUN_TOOL_CALLS,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
    )
    assert operations._model_budget(_context(record, "implement"))[0] == 0

    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "review",
            "tool_calls": 0,
            "model_usage": {
                "complete": True,
                "total_tokens": runtime.MAX_RUN_TOKENS,
            },
        },
    )
    with pytest.raises(RuntimeError, match="token budget exhausted"):
        operations._model_budget(_context(record, "implement"))


@pytest.mark.parametrize(
    ("prior", "current"),
    [(None, 1), (0, None), (0, True), (0, -1)],
)
def test_token_budget_fails_closed_unless_provider_is_explicitly_non_remote(
    prior: int | None, current: object
) -> None:
    usage = {"complete": False, "total_tokens": current}
    with pytest.raises(RuntimeError, match="usage is incomplete"):
        runtime._enforce_token_budget(prior, usage)
    runtime._enforce_token_budget(prior, usage, require_complete=False)


def test_token_budget_accepts_the_exact_boundary() -> None:
    runtime._enforce_token_budget(
        runtime.MAX_RUN_TOKENS - 1,
        {"complete": True, "total_tokens": 1},
    )


def test_token_budget_rejects_node_that_crosses_run_limit() -> None:
    with pytest.raises(RuntimeError, match="token budget exceeded"):
        runtime._enforce_token_budget(
            29_999,
            {"complete": True, "total_tokens": 2},
        )


def test_model_loop_checks_usage_before_tool_and_before_another_response() -> None:
    call = ToolCall("read", "read_file", {"path": "app.py"})
    executor = _Executor()
    over_budget = _DecisionProvider(
        call,
        usage=SimpleNamespace(complete=True, total_tokens=runtime.MAX_RUN_TOKENS + 1),
    )

    with pytest.raises(RuntimeError, match="token budget exceeded"):
        runtime._run_model_loop(
            over_budget,
            "task",
            executor,
            max_calls=1,
            after_response=lambda: runtime._enforce_token_budget(
                0, runtime._provider_usage(over_budget)
            ),
        )
    assert executor.calls == []

    executor = _Executor()
    at_limit = _DecisionProvider(
        call,
        FinalAnswer("done"),
        usage=SimpleNamespace(complete=True, total_tokens=runtime.MAX_RUN_TOKENS),
    )
    with pytest.raises(RuntimeError, match="token budget exhausted"):
        runtime._run_model_loop(
            at_limit,
            "task",
            executor,
            max_calls=1,
            before_response=lambda: runtime._enforce_next_model_response(
                0, runtime._provider_usage(at_limit)
            ),
            after_response=lambda: runtime._enforce_token_budget(
                0, runtime._provider_usage(at_limit)
            ),
        )
    assert len(executor.calls) == 1


def test_named_tool_loop_checks_usage_before_tool_and_before_another_response() -> None:
    call = ToolCall("read", "read_file", {"path": "app.py"})
    submit = ToolCall("plan", "submit_change_plan", {})
    executor = _Executor()
    over_budget = _DecisionProvider(
        call,
        usage=SimpleNamespace(complete=True, total_tokens=runtime.MAX_RUN_TOKENS + 1),
    )

    with pytest.raises(RuntimeError, match="token budget exceeded"):
        runtime._run_until_named_tool(
            over_budget,
            "task",
            executor,
            target="submit_change_plan",
            max_calls=1,
            after_response=lambda: runtime._enforce_token_budget(
                0, runtime._provider_usage(over_budget)
            ),
        )
    assert executor.calls == []

    executor = _Executor()
    at_limit = _DecisionProvider(
        call,
        submit,
        usage=SimpleNamespace(complete=True, total_tokens=runtime.MAX_RUN_TOKENS),
    )
    with pytest.raises(RuntimeError, match="token budget exhausted"):
        runtime._run_until_named_tool(
            at_limit,
            "task",
            executor,
            target="submit_change_plan",
            max_calls=1,
            before_response=lambda: runtime._enforce_next_model_response(
                0, runtime._provider_usage(at_limit)
            ),
            after_response=lambda: runtime._enforce_token_budget(
                0, runtime._provider_usage(at_limit)
            ),
        )
    assert len(executor.calls) == 1
    assert at_limit.decisions == [submit]


def test_model_loop_rejects_missing_remote_usage_before_tool_execution() -> None:
    call = ToolCall("read", "read_file", {"path": "app.py"})
    provider = _DecisionProvider(call)
    executor = _Executor()

    with pytest.raises(RuntimeError, match="usage is incomplete"):
        runtime._run_model_loop(
            provider,
            "task",
            executor,
            max_calls=1,
            after_response=lambda: runtime._enforce_token_budget(
                0, runtime._provider_usage(provider)
            ),
        )
    assert executor.calls == []


def test_remote_run_rejects_prior_incomplete_usage_and_tool_overage(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    remote = record.model_copy(update={"allow_remote_model": True})
    operations.record = remote
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": 0,
            "model_usage": {"complete": False, "total_tokens": None},
        },
    )
    with pytest.raises(RuntimeError, match="usage is incomplete"):
        operations._model_budget(_context(remote, "implement"))


def test_repository_tool_overage_is_rejected(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "tool_calls": runtime.MAX_RUN_TOOL_CALLS + 1,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
    )
    with pytest.raises(RuntimeError, match="repository tool budget exceeded"):
        operations._model_budget(_context(record, "implement"))


def test_fixed_phase_budgets_do_not_borrow_between_phases(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle

    assert runtime.MAX_PLANNING_TOKENS == 4_000
    assert runtime.MAX_CHANGE_TOKENS == 23_500
    assert runtime.MAX_REVIEW_TOKENS == 2_500
    assert runtime.MAX_RUN_TOKENS == 30_000
    assert runtime.MAX_PLANNING_TOOL_CALLS == 2
    assert runtime.MAX_CHANGE_TOOL_CALLS == 8
    change_tool_names = {definition.name for definition in runtime._change_tools()}
    assert "read_files" in change_tool_names
    assert "run_check" not in change_tool_names

    database.append_event(
        record.run_id,
        {
            "event": "planning_tool_summary",
            "phase": "planning",
            "tool_calls": 2,
            "model_usage": {"complete": True, "total_tokens": 4_000},
        },
    )
    with pytest.raises(RuntimeError, match="planning token budget exhausted"):
        operations._phase_model_budget(
            _context(record, "implement"),
            phases=frozenset({"planning"}),
            tool_limit=runtime.MAX_PLANNING_TOOL_CALLS,
            token_limit=runtime.MAX_PLANNING_TOKENS,
            label="planning",
        )

    remaining, tokens = operations._phase_model_budget(
        _context(record, "implement"),
        phases=frozenset({"implement", "repair"}),
        tool_limit=runtime.MAX_CHANGE_TOOL_CALLS,
        token_limit=runtime.MAX_CHANGE_TOKENS,
        label="change",
    )
    assert (remaining, tokens) == (8, 0)


def test_planning_exposes_only_batched_reads_before_plan_submission() -> None:
    planning_tool_names = [
        definition.name
        for definition in (*runtime._read_only_tools(), runtime.PLAN_TOOL)
    ]

    assert planning_tool_names == ["read_files", "submit_change_plan"]


def test_implement_and_repair_share_budget_while_review_remains_isolated(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    for phase, calls, tokens in (
        ("implement", 5, 20_000),
        ("repair", 3, 3_500),
    ):
        database.append_event(
            record.run_id,
            {
                "event": "model_tool_summary",
                "phase": phase,
                "tool_calls": calls,
                "model_usage": {"complete": True, "total_tokens": tokens},
            },
        )

    with pytest.raises(RuntimeError, match="change token budget exhausted"):
        operations._phase_model_budget(
            _context(record, "repair"),
            phases=frozenset({"implement", "repair"}),
            tool_limit=runtime.MAX_CHANGE_TOOL_CALLS,
            token_limit=runtime.MAX_CHANGE_TOKENS,
            label="change",
        )
    assert operations._phase_model_budget(
        _context(record, "review"),
        phases=frozenset({"review"}),
        tool_limit=0,
        token_limit=runtime.MAX_REVIEW_TOKENS,
        label="review",
    ) == (0, 0)


def test_next_response_estimate_includes_provider_replay_bytes(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    result = ToolResult("read", "read_file", True, "多字节 output")
    response_budgets: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=1_000),
        last_response_usage=SimpleNamespace(complete=True, total_tokens=100),
        pending_internal_tool_result_bytes=37,
        set_response_token_budget=response_budgets.append,
    )
    estimate = runtime._next_response_token_estimate(
        provider, (result,), require_complete=True
    )
    assert estimate == (
        100
        + len(runtime._json(runtime.asdict(result)).encode("utf-8"))
        + 37
        + runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )

    operations._enforce_next_model_response(
        0,
        provider,
        phase_prior_tokens=0,
        phase_token_limit=1_000 + estimate,
        new_tool_results=(result,),
    )
    assert response_budgets == [estimate]
    with pytest.raises(runtime._PhaseBudgetExhausted, match="predicted next"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=1_000 + estimate - 1,
            soft_phase_boundary=True,
            new_tool_results=(result,),
        )


def test_tool_result_fit_reserves_provider_replay_bytes() -> None:
    result = ToolResult("read", "read_files", True, "x" * 1_000)
    maximum_result_bytes = 200
    current_tokens = 100
    last_tokens = 10
    pending_bytes = 50
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=current_tokens),
        last_response_usage=SimpleNamespace(complete=True, total_tokens=last_tokens),
        pending_internal_tool_result_bytes=pending_bytes,
    )
    phase_limit = (
        current_tokens
        + last_tokens
        + pending_bytes
        + runtime.NEXT_RESPONSE_TOKEN_MARGIN
        + maximum_result_bytes
    )

    fitted = runtime._fit_tool_result_to_next_response_budget(
        result,
        provider,
        prior_tokens=0,
        phase_prior_tokens=0,
        phase_token_limit=phase_limit,
        phase_label="planning",
        reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
        require_complete=True,
    )

    assert fitted.truncated is True
    assert len(runtime._json(runtime.asdict(fitted)).encode("utf-8")) <= 200


def test_observed_planning_usage_and_batch_read_leave_capacity_for_plan(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    result = ToolResult(
        "read",
        "read_files",
        True,
        "x" * 1_225,
        exit_code=0,
    )
    result_bytes = len(runtime._json(runtime.asdict(result)).encode("utf-8"))
    assert 1_300 <= result_bytes <= 1_350
    first_response_tokens = 600
    response_budgets: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(
            response_count=1,
            complete=True,
            total_tokens=first_response_tokens,
        ),
        last_response_usage=SimpleNamespace(
            complete=True,
            total_tokens=first_response_tokens,
        ),
        set_response_token_budget=response_budgets.append,
    )

    operations._enforce_next_model_response(
        0,
        provider,
        phase_prior_tokens=0,
        phase_token_limit=runtime.MAX_PLANNING_TOKENS,
        phase_label="planning",
        new_tool_results=(result,),
        reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
    )

    assert response_budgets == [
        runtime.MAX_PLANNING_TOKENS - first_response_tokens
    ]


def test_planning_truncates_observed_batch_result_before_provider_history(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle

    class ObservedPlanningProvider:
        def __init__(self) -> None:
            self.usage = SimpleNamespace(
                response_count=0,
                reported_response_count=0,
                input_tokens=None,
                cached_input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                complete=False,
            )
            self.last_response_usage = None
            self.response_token_budgets: list[int] = []
            self.forced_tool_choices: list[str] = []
            self.result_histories: list[tuple[ToolResult, ...]] = []

        def set_response_token_budget(self, total_tokens: int) -> None:
            self.response_token_budgets.append(total_tokens)

        def set_forced_tool_choice(self, tool_name: str) -> None:
            self.forced_tool_choices.append(tool_name)

        def next_step(
            self, _task: str, results: tuple[ToolResult, ...]
        ) -> ToolCall:
            self.result_histories.append(results)
            if len(self.result_histories) == 1:
                self.usage = SimpleNamespace(
                    response_count=1,
                    reported_response_count=1,
                    input_tokens=840,
                    cached_input_tokens=0,
                    output_tokens=33,
                    total_tokens=873,
                    complete=True,
                )
                self.last_response_usage = SimpleNamespace(
                    total_tokens=873,
                    complete=True,
                )
                return ToolCall("read", "read_files", {"paths": ["app.py"]})

            self.usage = SimpleNamespace(
                response_count=2,
                reported_response_count=2,
                input_tokens=1_800,
                cached_input_tokens=0,
                output_tokens=200,
                total_tokens=2_000,
                complete=True,
            )
            self.last_response_usage = SimpleNamespace(
                total_tokens=1_127,
                complete=True,
            )
            return ToolCall(
                "plan",
                "submit_change_plan",
                {
                    "goal": "repair",
                    "files": ["app.py"],
                    "steps": ["change the implementation"],
                    "checks": ["pytest"],
                    "risks": [],
                },
            )

    provider = ObservedPlanningProvider()
    executor = _Executor()
    executor.execute = lambda decision: ToolResult(  # type: ignore[method-assign]
        decision.id,
        decision.name,
        True,
        "x" * 2_000,
        exit_code=0,
    )
    operations._provider_factory = lambda *_args: provider
    operations._repository_map = lambda _context: "repository map"
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._tool_executor = lambda *_args, **_kwargs: executor

    plan = operations.inspect_and_plan(
        _context(record, "inspect_and_plan", suffix="trim-observed-result")
    )

    delivered = provider.result_histories[1][0]
    maximum_result_bytes = (
        runtime.MAX_PLANNING_TOKENS
        - 873
        - 873
        - runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )
    assert plan.goal == "repair"
    assert delivered.truncated is True
    assert delivered.output.endswith(runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER)
    assert len(runtime._json(runtime.asdict(delivered)).encode("utf-8")) == (
        maximum_result_bytes
    )
    assert provider.response_token_budgets == [
        runtime.MAX_PLANNING_TOKENS,
        runtime.MAX_PLANNING_TOKENS - 873,
    ]
    assert provider.forced_tool_choices == ["submit_change_plan"]


def test_tool_result_truncation_counts_unicode_and_json_escaping() -> None:
    result = ToolResult(
        "read",
        "read_files",
        True,
        ("界\"\\\n" * 200),
        exit_code=0,
    )
    marker_only = replace(
        result,
        output=runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER,
        truncated=True,
    )
    maximum_bytes = (
        len(runtime._json(runtime.asdict(marker_only)).encode("utf-8")) + 37
    )

    truncated = runtime._truncate_tool_result_to_json_bytes(
        result,
        maximum_bytes,
        phase_label="planning",
    )

    retained = truncated.output[: -len(runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER)]
    assert result.output.startswith(retained)
    assert len(runtime._json(runtime.asdict(truncated)).encode("utf-8")) <= (
        maximum_bytes
    )
    next_character = replace(
        result,
        output=(
            result.output[: len(retained) + 1]
            + runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER
        ),
        truncated=True,
    )
    assert len(runtime._json(runtime.asdict(next_character)).encode("utf-8")) > (
        maximum_bytes
    )


def test_tool_result_truncation_fails_when_fixed_envelope_cannot_fit() -> None:
    result = ToolResult("read", "read_files", True, "x" * 1_000, exit_code=0)
    marker_only = replace(
        result,
        output=runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER,
        truncated=True,
    )
    fixed_envelope_bytes = len(
        runtime._json(runtime.asdict(marker_only)).encode("utf-8")
    )

    with pytest.raises(RuntimeError, match="cannot fit the tool result envelope"):
        runtime._truncate_tool_result_to_json_bytes(
            result,
            fixed_envelope_bytes - 1,
            phase_label="planning",
        )


def test_tool_result_within_budget_is_returned_unchanged() -> None:
    result = ToolResult("read", "read_files", True, "unchanged", exit_code=0)
    serialized_bytes = len(runtime._json(runtime.asdict(result)).encode("utf-8"))

    fitted = runtime._truncate_tool_result_to_json_bytes(
        result,
        serialized_bytes,
        phase_label="planning",
    )

    assert fitted is result


def test_tool_result_budget_preserves_review_reservation() -> None:
    result = ToolResult("read", "read_files", True, "x" * 1_000, exit_code=0)
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=100),
        last_response_usage=SimpleNamespace(complete=True, total_tokens=100),
    )

    without_reservation = runtime._fit_tool_result_to_next_response_budget(
        result,
        provider,
        prior_tokens=26_000,
        phase_prior_tokens=0,
        phase_token_limit=runtime.MAX_PLANNING_TOKENS,
        phase_label="planning",
        reserved_run_tokens=0,
        require_complete=True,
    )
    with_reservation = runtime._fit_tool_result_to_next_response_budget(
        result,
        provider,
        prior_tokens=26_000,
        phase_prior_tokens=0,
        phase_token_limit=runtime.MAX_PLANNING_TOKENS,
        phase_label="planning",
        reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
        require_complete=True,
    )

    reserved_result_bytes = (
        runtime.MAX_RUN_TOKENS
        - runtime.MAX_REVIEW_TOKENS
        - 26_000
        - 100
        - 100
        - runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )
    assert without_reservation is result
    assert with_reservation.truncated is True
    assert len(runtime._json(runtime.asdict(with_reservation)).encode("utf-8")) == (
        reserved_result_bytes
    )


def test_planning_and_review_set_initial_fixed_response_budgets(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    planning_provider = _DecisionProvider(
        ToolCall(
            "plan",
            "submit_change_plan",
            {
                "goal": "repair",
                "files": ["app.py"],
                "steps": ["change the implementation"],
                "checks": ["pytest"],
                "risks": [],
            },
        )
    )
    operations._provider_factory = lambda *_args: planning_provider
    operations._repository_map = lambda _context: "repository map"
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    operations.inspect_and_plan(
        _context(record, "inspect_and_plan", suffix="initial-planning-budget")
    )

    assert planning_provider.response_token_budgets == [
        runtime.MAX_PLANNING_TOKENS
    ]

    review_record = record.model_copy(update={"run_id": "d" * 32})
    review_database = RunDatabase(artifacts.root.parent / "review-runs.sqlite3")
    review_artifacts = ArtifactStore(artifacts.root.parent / "review-artifacts")
    review_directory = review_artifacts.initialize(review_record.run_id)
    review_database.create(review_record, review_directory)
    review_artifacts.write_patch(review_record.run_id, "diff\n")
    review_operations = runtime.RepositoryWorkflowOperations(
        review_record,
        database=review_database,
        artifacts=review_artifacts,
        cancel_event=threading.Event(),
    )
    review_provider = _DecisionProvider(
        ToolCall(
            "review",
            "submit_review",
            {"approved": True, "summary": "approved", "risks": []},
        )
    )
    review_operations._provider_factory = lambda *_args: review_provider

    review_operations.review(
        _context(review_record, "review", suffix="initial-review-budget")
    )

    assert review_provider.response_token_budgets == [runtime.MAX_REVIEW_TOKENS]
    assert review_provider.forced_tool_choices == ["submit_review"]


def test_change_phase_sets_initial_shared_response_budget(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), "durable diff\n")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()
    provider = _DecisionProvider(FinalAnswer("done"))
    operations._provider_factory = lambda *_args: provider

    outcome = operations.implement(
        _context(record, "implement", suffix="initial-change-budget")
    )

    assert outcome.ok is True
    assert provider.response_token_budgets == [runtime.MAX_CHANGE_TOKENS]


@pytest.mark.parametrize(
    (
        "prior_tokens",
        "current_tokens",
        "phase_prior_tokens",
        "phase_token_limit",
        "reserved_run_tokens",
        "expected_budget",
    ),
    (
        (700, 300, 700, runtime.MAX_PLANNING_TOKENS, 2_500, 3_000),
        (26_000, 500, 21_000, runtime.MAX_CHANGE_TOKENS, 2_500, 1_000),
        (15_000, 0, 12_000, runtime.MAX_CHANGE_TOKENS, 2_500, 11_500),
        (29_000, 0, 0, runtime.MAX_REVIEW_TOKENS, 0, 1_000),
    ),
)
def test_response_budget_uses_run_phase_and_review_reservations(
    operations_bundle,
    prior_tokens: int,
    current_tokens: int,
    phase_prior_tokens: int,
    phase_token_limit: int,
    reserved_run_tokens: int,
    expected_budget: int,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    provider = _DecisionProvider(
        usage=SimpleNamespace(
            response_count=1,
            total_tokens=current_tokens,
            complete=True,
        )
    )

    operations._configure_provider_response_budget(
        prior_tokens,
        provider,
        phase_prior_tokens=phase_prior_tokens,
        phase_token_limit=phase_token_limit,
        phase_label="test",
        reserved_run_tokens=reserved_run_tokens,
    )

    assert provider.response_token_budgets == [expected_budget]


def test_next_response_estimate_fails_closed_for_remote_missing_usage(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    operations.record = record.model_copy(update={"allow_remote_model": True})
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=100),
        last_response_usage=None,
    )

    with pytest.raises(RuntimeError, match="last model response token usage"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=runtime.MAX_CHANGE_TOKENS,
            new_tool_results=(ToolResult("r", "read_file", True, "ok"),),
        )


def test_tool_outcomes_distinguish_expected_checks_from_actionable_errors() -> None:
    results = (
        ToolResult("ok", "read_file", True, "ok"),
        ToolResult(
            "check", "run_check", False, '{"status":"failed"}', exit_code=1
        ),
        ToolResult(
            "policy",
            "run_check",
            False,
            '{"status":"policy_denied"}',
            exit_code=1,
        ),
        ToolResult("args", "apply_patch", False, "", error="ValueError: requires diff"),
        ToolResult("patch", "apply_patch", False, "", error="Patch rejected"),
        ToolResult("transport", "read_file", False, "", error="Docker timed out"),
        ToolResult(
            "persist",
            "apply_patch",
            False,
            "",
            error="candidate patch persistence failed: disk full",
        ),
    )

    assert runtime._tool_outcome_counts(results) == {
        "ok": 1,
        "check_failed": 1,
        "invalid_args": 1,
        "policy_denied": 1,
        "transport_error": 2,
        "patch_rejected": 1,
        "other_error": 0,
    }


def test_model_summary_keeps_legacy_counts_and_adds_tool_outcomes(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    results = (
        ToolResult("ok", "read_file", True, "ok"),
        ToolResult("bad", "apply_patch", False, "", error="Patch rejected"),
    )

    provider = SimpleNamespace(
        usage=None,
        parallel_tool_call_violations=2,
    )
    operations._append_model_summary(
        _context(record, "implement"),
        event="model_tool_summary",
        phase="implement",
        provider=provider,
        tool_results=results,
    )

    summary = _only_model_summary(database, record.run_id)
    assert summary["tool_calls"] == 4
    assert summary["tool_errors"] == 3
    assert summary["parallel_tool_call_violations"] == 2
    assert summary["tool_outcomes"]["ok"] == 1
    assert summary["tool_outcomes"]["invalid_args"] == 2
    assert summary["tool_outcomes"]["patch_rejected"] == 1


def test_model_metrics_preserve_actual_failure_usage_beyond_policy_caps(
    operations_bundle,
) -> None:
    _operations, _database, _artifacts, record = operations_bundle
    synchronized = runtime._synchronize_model_metrics(
        record,
        (
            {
                "event": "planning_tool_summary",
                "tool_calls": 2,
                "model_usage": {"total_tokens": 4_000},
            },
            {
                "event": "model_tool_summary",
                "tool_calls": 9,
                "model_usage": {"total_tokens": 27_000},
            },
        ),
    )

    assert synchronized.metrics.tool_calls == 11
    assert synchronized.metrics.tokens == 31_000


def _only_model_summary(database: RunDatabase, run_id: str) -> dict[str, object]:
    summaries = [
        event
        for event in database.events(run_id)
        if event.get("event") in {"planning_tool_summary", "model_tool_summary"}
    ]
    assert len(summaries) == 1
    return summaries[0]


def test_planning_provider_failure_preserves_tool_and_usage_summary(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = _FailingProvider(
        ToolCall("read", "read_file", {"path": "app.py"}), total_tokens=19
    )
    operations._provider_factory = lambda *_args: provider
    operations._repository_map = lambda _context: "repository map"
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.inspect_and_plan(_context(record, "inspect_and_plan", suffix="failure"))

    summary = _only_model_summary(database, record.run_id)
    assert summary["event"] == "planning_tool_summary"
    assert summary["phase"] == "planning"
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 0
    assert summary["model_usage"]["total_tokens"] == 19


def test_implementation_provider_failure_preserves_tool_and_usage_summary(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = _FailingProvider(
        ToolCall("read", "read_file", {"path": "app.py"}), total_tokens=23
    )
    operations._provider_factory = lambda *_args: provider
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.implement(_context(record, "implement", suffix="failure"))

    summary = _only_model_summary(database, record.run_id)
    assert summary["event"] == "model_tool_summary"
    assert summary["phase"] == "implement"
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 0
    assert summary["model_usage"]["total_tokens"] == 23


def test_review_provider_failure_preserves_usage_summary(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    operations._provider_factory = lambda *_args: _FailingProvider(total_tokens=29)

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.review(_context(record, "review", suffix="failure"))

    summary = _only_model_summary(database, record.run_id)
    assert summary["event"] == "model_tool_summary"
    assert summary["phase"] == "review"
    assert summary["tool_calls"] == 0
    assert summary["model_usage"]["total_tokens"] == 29


def test_summary_write_failure_does_not_mask_provider_failure(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    operations._provider_factory = lambda *_args: _FailingProvider()
    operations._append_model_summary = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("summary write failed")
    )

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.review(_context(record, "review", suffix="summary-failure"))


def test_summary_write_failure_propagates_without_provider_failure(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    operations._provider_factory = lambda *_args: _DecisionProvider(
        ToolCall(
            "review",
            "submit_review",
            {"approved": True, "summary": "approved", "risks": []},
        )
    )
    operations._append_model_summary = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("summary write failed")
    )

    with pytest.raises(RuntimeError, match="summary write failed"):
        operations.review(_context(record, "review", suffix="summary-only-failure"))


def test_broken_usage_fields_are_recorded_as_incomplete(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = _FailingProvider()
    provider.usage = _BrokenUsage()
    operations._provider_factory = lambda *_args: provider

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.review(_context(record, "review", suffix="broken-usage"))

    usage = _only_model_summary(database, record.run_id)["model_usage"]
    assert usage["complete"] is False
    assert usage["total_tokens"] is None


def test_provider_usage_reads_optional_usage_fields() -> None:
    usage = SimpleNamespace(
        response_count=2,
        input_tokens=10,
        cached_input_tokens=4,
        output_tokens=3,
        total_tokens=13,
        complete=True,
    )

    payload = runtime._provider_usage(SimpleNamespace(usage=usage))

    assert payload == {
        "response_count": 2,
        "reported_response_count": None,
        "input_tokens": 10,
        "cached_input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 13,
        "complete": True,
    }


def test_non_remote_unknown_history_still_enforces_known_current_usage() -> None:
    over_limit = {
        "complete": True,
        "total_tokens": runtime.MAX_RUN_TOKENS + 1,
    }
    at_limit = {"complete": True, "total_tokens": runtime.MAX_RUN_TOKENS}

    with pytest.raises(RuntimeError, match="token budget exceeded"):
        runtime._enforce_token_budget(None, over_limit, require_complete=False)
    with pytest.raises(RuntimeError, match="token budget exhausted"):
        runtime._enforce_next_model_response(None, at_limit, require_complete=False)


@pytest.mark.parametrize(
    ("decision", "error"),
    [
        (FinalAnswer("text"), "did not return submit_review"),
        (ToolCall("wrong", "finish", {}), "did not return submit_review"),
        (
            ToolCall("review", "submit_review", {"approved": True}),
            "unexpected fields",
        ),
        (
            ToolCall(
                "review",
                "submit_review",
                {"approved": 1, "summary": "ok", "risks": []},
            ),
            "invalid field types",
        ),
    ],
)
def test_review_rejects_invalid_model_verdicts(
    operations_bundle, decision, error: str
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, "diff\n")
    operations._provider_factory = lambda *_args: _DecisionProvider(decision)

    with pytest.raises((RuntimeError, ValueError), match=error):
        operations.review(_context(record, "review", suffix=error))


def test_review_preserves_rejection_as_node_failure(operations_bundle) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, "diff\n")
    decision = ToolCall(
        "review",
        "submit_review",
        {"approved": False, "summary": "risk remains", "risks": ["scope"]},
    )
    operations._provider_factory = lambda *_args: _DecisionProvider(decision)

    result = operations.review(_context(record, "review", suffix="rejected"))

    assert result.ok is False
    assert result.error == "risk remains"
    assert result.detail["risks"] == ["scope"]


def test_review_capacity_failure_keeps_full_patch_and_uses_zero_tokens(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    patch_lines = [
        "diff --git a/app.py b/app.py",
        "--- a/app.py",
        "+++ b/app.py",
        "@@ -1 +1,1000 @@",
        "-old",
        *(f"+{index:04d}-" + ("x" * 34) for index in range(1000)),
    ]
    patch = "\n".join(patch_lines) + "\n"
    assert 40_000 <= len(patch.encode("utf-8")) <= 64 * 1024
    assert runtime.validate_patch(patch).changed_lines == 1001
    artifacts.write_patch(record.run_id, patch)

    http_calls: list[tuple[object, ...]] = []
    providers: list[runtime.OpenAIProvider] = []

    def provider_factory(definitions, system_prompt, idempotency_key):
        provider = runtime.OpenAIProvider(
            runtime.OpenAIConfig(
                api_key="fake-provider-key",
                base_url="http://127.0.0.1:9",
                model="fake-model",
            ),
            tool_definitions=definitions,
            system_prompt=system_prompt,
            idempotency_key=idempotency_key,
        )

        def unexpected_http(*args, **kwargs):
            http_calls.append((args, kwargs))
            raise AssertionError("capacity failure must precede HTTP")

        monkeypatch.setattr(provider._client, "post_json", unexpected_http)
        providers.append(provider)
        return provider

    operations._provider_factory = provider_factory
    context = _context(record, "review", suffix="capacity")

    result = operations.review(context)

    assert result.ok is False
    assert result.summary == "Candidate exceeds the fixed full-review capacity"
    assert result.error is not None
    assert result.error.startswith("review_capacity_exceeded:")
    assert "split the task" in result.error
    assert result.detail == {
        "failure_status": "policy_denied",
        "failure_code": "review_capacity_exceeded",
        "tool_calls": 0,
        "tokens": 0,
    }
    assert http_calls == []
    assert len(providers) == 1
    assert providers[0].usage.response_count == 0
    assert providers[0].usage.total_tokens is None
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch

    stored = database.side_effect(
        record.run_id, f"workflow:{context.idempotency_key}"
    )
    assert stored == result.model_dump(mode="json")
    summary = _only_model_summary(database, record.run_id)
    assert summary["model_usage"]["response_count"] == 0
    assert summary["model_usage"]["total_tokens"] is None
    synchronized = runtime._synchronize_model_metrics(
        record, database.events(record.run_id)
    )
    assert synchronized.metrics.tokens == 0


class _CandidateSandbox:
    workspace_revision = 1

    def __init__(
        self,
        candidate: CandidateArtifact | None,
        *,
        restore_result: ToolResult | None = None,
    ) -> None:
        self.candidate = candidate
        self.restore_result = restore_result or ToolResult(
            "restore", "apply_patch", True, "restored"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, _call):
        return self.restore_result

    def candidate_artifact(self):
        return self.candidate


def _stub_change_execution(monkeypatch, operations, sandbox) -> None:
    operations._sandbox = lambda *_args, **_kwargs: sandbox
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: runtime.ModelLoopOutcome(True, "done", (), ()),
    )


def test_provider_failure_after_patch_keeps_atomically_persisted_candidate(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), "durable diff\n")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._provider_factory = lambda *_args: _FailingProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
        total_tokens=23,
    )

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.implement(_context(record, "implement", suffix="persist-on-failure"))

    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == (
        candidate.patch
    )


def test_patch_persistence_failure_stops_before_next_model_response(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), "durable diff\n")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    provider = _DecisionProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
        FinalAnswer("must not be requested"),
    )
    operations._provider_factory = lambda *_args: provider
    monkeypatch.setattr(
        artifacts,
        "write_patch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(RuntimeError, match="candidate patch persistence failed"):
        operations.implement(
            _context(record, "implement", suffix="persistence-fail-closed")
        )

    assert provider.decisions == [FinalAnswer("must not be requested")]


def test_candidate_capture_failure_stops_before_next_model_response(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    sandbox = _CandidateSandbox(None)
    sandbox.candidate_artifact = lambda: (_ for _ in ()).throw(
        RuntimeError("candidate capture failed")
    )
    operations._sandbox = lambda *_args, **_kwargs: sandbox
    provider = _DecisionProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
        FinalAnswer("must not be requested"),
    )
    operations._provider_factory = lambda *_args: provider

    with pytest.raises(RuntimeError, match="candidate patch persistence failed"):
        operations.implement(
            _context(record, "implement", suffix="capture-fail-closed")
        )

    assert provider.decisions == [FinalAnswer("must not be requested")]


def test_credential_candidate_is_rejected_before_artifact_or_side_effect(
    operations_bundle,
) -> None:
    operations, database, artifacts, record = operations_bundle
    secret = "sk-this-is-a-real-looking-secret"
    secured = ArtifactStore(artifacts.root.parent / "secured", secrets=(secret,))
    secured.initialize(record.run_id)
    operations.artifacts = secured
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("app.py",),
        f"diff --git a/app.py b/app.py\n+api_key={secret}\n",
    )
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    provider = _DecisionProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
        FinalAnswer("must not be requested"),
    )
    operations._provider_factory = lambda *_args: provider
    context = _context(record, "implement", suffix="credential-candidate")

    with pytest.raises(RuntimeError, match="candidate patch persistence failed"):
        operations.implement(context)

    assert secured.path(record.run_id, "patch").read_text(encoding="utf-8") == ""
    assert database.side_effect(
        record.run_id, f"workflow:{context.idempotency_key}"
    ) is None
    summary = _only_model_summary(database, record.run_id)
    assert summary["tool_outcomes"]["policy_denied"] == 1
    assert provider.decisions == [FinalAnswer("must not be requested")]


def test_soft_budget_boundary_advances_only_candidate_with_new_regression_test(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_issue_regression.py "
        "b/tests/test_issue_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_issue_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_issue_regression.py",), patch
    )
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime._PhaseBudgetExhausted("change token budget exhausted")
        ),
    )

    context = _context(record, "implement", suffix="soft-budget-candidate")
    outcome = operations.implement(context)

    assert outcome.ok is True
    assert outcome.detail["soft_budget_boundary"] is True
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch
    stored = database.side_effect(
        record.run_id, f"workflow:{context.idempotency_key}"
    )
    assert isinstance(stored, dict)
    assert "patch" not in stored["detail"]


def test_tool_call_boundary_advances_a_verifiable_persisted_candidate(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_tool_regression.py "
        "b/tests/test_tool_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_tool_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_tool_regression.py",), patch
    )
    _stub_change_execution(monkeypatch, operations, _CandidateSandbox(candidate))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: runtime.ModelLoopOutcome(
            False,
            "",
            (),
            (),
            "model exceeded 8 repository tool calls",
            tool_budget_exhausted=True,
        ),
    )

    outcome = operations.implement(
        _context(record, "implement", suffix="tool-budget-candidate")
    )

    assert outcome.ok is True
    assert outcome.detail["soft_budget_boundary"] is True
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch


def test_soft_budget_boundary_rejects_candidate_without_new_regression_test(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), "invalid diff\n")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime._PhaseBudgetExhausted("change token budget exhausted")
        ),
    )

    with pytest.raises(RuntimeError, match="no verifiable candidate"):
        operations.implement(
            _context(record, "implement", suffix="soft-budget-no-test")
        )


def test_repair_receives_only_redacted_eight_kibibyte_log_tail(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_check(
        record.run_id,
        "verify-0.log",
        "discard-me\n" + ("x" * 9_000) + "\napi_key=plain-secret\n",
    )
    failed = CheckSummary(
        attempt=0,
        check_id="python-pytest",
        status="failed",
        ok=False,
        duration_ms=1,
        log_artifact="checks/verify-0.log",
    )
    with_check = record.model_copy(update={"checks": (failed,)})
    context = _context(with_check, "repair", suffix="log-tail")

    tail = operations._repair_log_tail(context)

    assert len(tail.encode("utf-8")) <= runtime.MAX_REPAIR_LOG_BYTES
    assert "discard-me" not in tail
    assert "plain-secret" not in tail
    assert "[REDACTED]" in tail
    assert "Sanitized failing-check log tail" in runtime._change_prompt(
        context, "repair", "diff", tail
    )
    assert "Sanitized failing-check log tail" not in runtime._change_prompt(
        context, "implement", "diff"
    )

    unsafe = with_check.model_copy(
        update={
            "checks": (
                failed.model_copy(update={"log_artifact": "checks/..\\result.json"}),
            )
        }
    )
    with pytest.raises(RuntimeError, match="invalid verification log"):
        operations._repair_log_tail(_context(unsafe, "repair", suffix="unsafe-log"))


def test_change_candidate_reports_missing_and_unchanged_patch(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    missing_context = _context(record, "implement", suffix="missing")
    _stub_change_execution(monkeypatch, operations, _CandidateSandbox(None))
    missing = operations.implement(missing_context)
    assert missing.ok is False
    assert missing.error == "model finished without producing a candidate patch"

    prior_patch = "same diff\n"
    artifacts.write_patch(record.run_id, prior_patch)
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), prior_patch)
    _stub_change_execution(monkeypatch, operations, _CandidateSandbox(candidate))
    unchanged = operations.repair(_context(record, "repair", suffix="unchanged"))
    assert unchanged.ok is False
    assert unchanged.error == "repair did not change the candidate patch"


def test_change_candidate_stops_when_prior_patch_cannot_be_restored(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, "prior diff\n")
    failed = ToolResult(
        "restore",
        "apply_patch",
        False,
        "",
        error="restore rejected",
    )
    sandbox = _CandidateSandbox(None, restore_result=failed)
    _stub_change_execution(monkeypatch, operations, sandbox)

    with pytest.raises(RuntimeError, match="restore rejected"):
        operations.repair(_context(record, "repair", suffix="restore-failed"))


def test_repository_map_propagates_sandbox_error(operations_bundle) -> None:
    operations, _database, _artifacts, record = operations_bundle
    failed = ToolResult("map", "repo_map", False, "", error="map rejected")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(
        None, restore_result=failed
    )

    with pytest.raises(RuntimeError, match="map rejected"):
        operations._repository_map(_context(record, "inspect_and_plan"))


def test_planning_repository_map_is_bounded_before_provider_prompt(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    mapped = ToolResult("map", "repo_map", True, "界" * 20_000)
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(
        None, restore_result=mapped
    )

    result = operations._repository_map(_context(record, "inspect_and_plan"))

    assert len(result.encode("utf-8")) <= runtime.MAX_PLANNING_REPOSITORY_MAP_BYTES
    assert result.endswith("... repository map truncated for planning budget ...\n")


def test_cancel_closes_active_resource_and_contains_close_error(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    closed: list[bool] = []
    operations._active_resource = SimpleNamespace(close=lambda: closed.append(True))
    operations.cancel()
    assert operations.cancel_event.is_set()
    assert closed == [True]

    operations.cancel_event.clear()
    operations._active_resource = SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    )
    operations.cancel()
    assert operations.cancel_event.is_set()


def test_active_context_supports_plain_and_context_managed_resources() -> None:
    owner = SimpleNamespace(
        _check_cancelled=lambda: None,
        _active_lock=threading.RLock(),
        _active_resource=None,
    )
    plain = object()
    with runtime._ActiveContext(owner, plain) as entered:
        assert entered is plain
        assert owner._active_resource is plain
    assert owner._active_resource is None

    class Managed:
        def __enter__(self):
            return "entered"

        def __exit__(self, *_args):
            owner._active_resource = "replacement"
            return True

    context = runtime._ActiveContext(owner, Managed())
    assert context.__enter__() == "entered"
    assert context.__exit__(None, None, None) is True
    assert owner._active_resource == "replacement"


def test_cancelled_operation_stops_at_node_boundary(operations_bundle) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    operations.cancel_event.set()
    with pytest.raises(runtime.WorkflowCancelled, match="cancelled"):
        operations._check_cancelled()


def test_operations_default_provider_delegates_record_policy(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    expected = object()
    captured: list[dict[str, object]] = []

    def factory(definitions, system_prompt, idempotency_key, **kwargs):
        captured.append(
            {
                "definitions": definitions,
                "system_prompt": system_prompt,
                "idempotency_key": idempotency_key,
                **kwargs,
            }
        )
        return expected

    monkeypatch.setattr(runtime, "default_provider_factory", factory)
    result = operations._provider((runtime.PLAN_TOOL,), "system", "key")

    assert result is expected
    assert captured[0]["allow_remote_model"] == record.allow_remote_model


def test_service_runner_cancel_delegates_only_for_active_run(operations_bundle) -> None:
    _operations, database, artifacts, _record_value = operations_bundle
    runner = runtime.ServiceWorkflowRunner(database, artifacts)
    calls: list[str] = []
    runner._active[RUN_ID] = SimpleNamespace(cancel=lambda: calls.append(RUN_ID))

    runner.cancel("d" * 32)
    runner.cancel(RUN_ID)

    assert calls == [RUN_ID]


@pytest.mark.parametrize(
    ("status", "resume", "expected"),
    [("awaiting_approval", True, "approve"), ("interrupted", False, "resume")],
)
def test_service_runner_resumes_existing_checkpoints(
    operations_bundle, monkeypatch, status: str, resume: bool, expected: str
) -> None:
    _operations, database, artifacts, base_record = operations_bundle
    record = base_record.model_copy(
        update={
            "status": status,
            "current_node": "approval"
            if status == "awaiting_approval"
            else "implement",
            "approval_reason": "approved" if resume else None,
        }
    )
    snapshot = WorkflowSnapshot(record=record, next_node=record.current_node, version=1)
    calls: list[str] = []

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load_run(self, _run_id):
            return snapshot

        def list_checkpoints(self, _run_id):
            return ()

    class FakeEngine:
        def approve(self, *_args, **_kwargs):
            calls.append("approve")
            return record.model_copy(
                update={"status": "running", "current_node": "implement"}
            )

        def resume(self, _run_id):
            calls.append("resume")
            return record.model_copy(
                update={"status": "succeeded", "current_node": None}
            )

    store = FakeStore()
    monkeypatch.setattr(runtime, "SQLiteCheckpointStore", lambda _path: store)
    monkeypatch.setattr(runtime, "WorkflowEngine", lambda *_args: FakeEngine())
    checkpointed: list[dict[str, object]] = []
    service_context = SimpleNamespace(
        cancel_event=threading.Event(),
        artifacts=artifacts,
        checkpoint=lambda result, **kwargs: checkpointed.append(kwargs) or result,
    )
    runner = runtime.ServiceWorkflowRunner(database, artifacts)

    result = runner.execute(record, service_context, resume=resume)

    assert calls == [expected]
    assert result.created_at == record.created_at
    assert checkpointed[0]["event"] == "workflow_state_synchronized"
    assert RUN_ID not in runner._active


def test_checkpoint_trace_sync_skips_existing_and_records_new_entries() -> None:
    checkpoints = [
        SimpleNamespace(
            sequence=1,
            node="prepare",
            phase="completed",
            attempt=0,
            idempotency_key="prepare-0",
            created_at=NOW,
        ),
        SimpleNamespace(
            sequence=2,
            node="baseline_check",
            phase="started",
            attempt=0,
            idempotency_key="baseline-0",
            created_at=NOW,
        ),
    ]

    class Database:
        path = Path("unused")

        def side_effect(self, _run_id, key):
            return {"written": True} if key.endswith(":1") else None

        def record_side_effect(self, run_id, key, value):
            recorded.append((run_id, key, value))

    recorded: list[tuple[object, ...]] = []
    traces: list[dict[str, object]] = []
    context = SimpleNamespace(
        artifacts=SimpleNamespace(
            append_trace=lambda _run_id, event: traces.append(event)
        )
    )
    runner = runtime.ServiceWorkflowRunner(Database(), SimpleNamespace())

    runner._sync_checkpoint_trace(
        SimpleNamespace(list_checkpoints=lambda _run_id: checkpoints),
        context,
        RUN_ID,
    )

    assert [event["checkpoint_sequence"] for event in traces] == [2]
    assert recorded == [(RUN_ID, "trace:workflow-checkpoint:2", {"written": True})]


def test_git_helper_uses_stderr_then_exit_code(monkeypatch, tmp_path: Path) -> None:
    responses = iter(
        [
            CapturedProcess(2, b"", b"git denied", False, False, False),
            CapturedProcess(3, b"", b"", False, False, False),
        ]
    )
    monkeypatch.setattr(
        runtime, "run_isolated_capture", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(RuntimeError, match="git denied"):
        runtime._git(tmp_path, "status")
    with pytest.raises(RuntimeError, match="Git exited 3"):
        runtime._git(tmp_path, "status")
