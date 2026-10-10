from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import repo_agent.openai_provider as provider_module
import repo_agent.workflow_runtime as runtime
from repo_agent.artifacts import ArtifactStore
from repo_agent.models import CandidateArtifact, FinalAnswer, ToolCall, ToolResult
from repo_agent.openai_provider import (
    OpenAIConfig,
    OpenAIHTTPError,
    OpenAIProvider,
    OpenAIToolArgumentsError,
    OpenAIUnknownToolError,
    _apply_response_token_budget,
    _encode_request_payload,
    _estimate_request_input_tokens,
    tool_result_replay_bytes,
)
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

_SMALL_REVIEW_PATCH = (
    "diff --git a/app.py b/app.py\n"
    "--- a/app.py\n"
    "+++ b/app.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


@pytest.mark.parametrize(
    ("base_url", "expected_timeout"),
    [
        ("https://api.deepseek.com", 120.0),
        ("https://relay.example.com/v1", 30.0),
    ],
)
def test_default_provider_factory_extends_only_official_deepseek_timeout(
    monkeypatch,
    base_url: str,
    expected_timeout: float,
) -> None:
    config = OpenAIConfig(
        api_key="fake-provider-key",
        base_url=base_url,
        model="test-model",
        allow_remote_model=True,
    )
    monkeypatch.setattr(
        runtime.OpenAIConfig,
        "from_env",
        lambda *, allow_remote_model: config,
    )
    monkeypatch.setattr(
        runtime,
        "OpenAIProvider",
        lambda provided, **kwargs: (provided, kwargs),
    )

    provider = runtime.default_provider_factory(
        runtime.AGENT_TOOL_DEFINITIONS,
        "system",
        "idempotent",
        allow_remote_model=True,
    )

    assert provider[0].timeout_seconds == expected_timeout


@pytest.mark.parametrize(
    "base_url",
    ("https://api.deepseek.com", "https://API.DEEPSEEK.COM:443/v1/"),
)
def test_official_deepseek_workflow_disables_thinking_for_named_tools(
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
    assert runtime._workflow_reasoning_effort(config, (apply_patch,)) == "none"


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


def test_prepare_rejects_non_python_detected_profile(
    operations_bundle, monkeypatch
) -> None:
    operations, database, _artifacts, record = operations_bundle
    monkeypatch.setattr(runtime, "effective_core_autocrlf", lambda _repo: "false")
    responses = iter(("", "a" * 40, "a" * 40, ""))
    monkeypatch.setattr(runtime, "_git", lambda *_args: next(responses))
    monkeypatch.setattr(
        runtime,
        "detect_check_profile",
        lambda _repo: SimpleNamespace(
            id="maven-test",
            language="java",
            image="repo-agent-maven:0.1",
        ),
    )
    context = _context(record, "prepare", suffix="non-python")

    result = operations.prepare(context)

    assert result.ok is False
    assert result.error == "durable workflows support only Python repositories"
    assert result.detail == {"failure_status": "policy_denied"}
    assert database.side_effect(
        record.run_id, f"workflow:{context.idempotency_key}"
    ) == result.model_dump(mode="json")
    assert operations.prepare(context) == result


def test_finalize_rejects_empty_candidate(operations_bundle) -> None:
    operations, _database, _artifacts, record = operations_bundle

    result = operations.finalize(_context(record, "finalize", suffix="empty"))

    assert result.ok is False
    assert result.error == "candidate patch is empty"


class _DecisionProvider:
    def __init__(self, *decisions, usage=None) -> None:
        self.decisions = list(decisions)
        self.usage = usage
        self.calls = 0
        self.request_count = (
            0 if usage is None else getattr(usage, "response_count", 0)
        )
        self.response_token_budgets: list[int] = []
        self.forced_tool_choices: list[str] = []

    def next_step(self, _task, _results):
        self.calls += 1
        self.request_count += 1
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


class _NoEffectThenCorrectionExecutor(_Executor):
    def execute(self, decision: ToolCall) -> ToolResult:
        self.calls.append(decision)
        if decision.id == "no-effect":
            return ToolResult(
                decision.id,
                decision.name,
                False,
                "",
                error=(
                    "Patch made no effective change to the candidate workspace; "
                    "submit an incremental correction diff"
                ),
            )
        self.workspace_revision += 1
        return ToolResult(decision.id, decision.name, True, "corrected")


class _ParallelViolationProvider(_DecisionProvider):
    def __init__(self, decision: ToolCall, violations: int) -> None:
        super().__init__(decision)
        self.parallel_tool_call_violations = 0
        self._violations = violations

    def next_step(self, task, results):
        self.parallel_tool_call_violations += self._violations
        return super().next_step(task, results)


def test_named_tool_parallel_violations_do_not_consume_execution_budget() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]}),
        violations=2,
    )
    submit = ToolCall("plan", "submit_change_plan", {})
    provider.decisions.append(submit)
    executor = _Executor()

    decision, results = runtime._run_until_named_tool(
        provider,
        "inspect",
        executor,
        target="submit_change_plan",
        max_calls=1,
    )

    assert decision is submit
    assert len(results) == 1
    assert [call.id for call in executor.calls] == ["read"]
    assert provider.parallel_tool_call_violations == 4


def test_change_parallel_violations_do_not_consume_execution_budget() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]}),
        violations=8,
    )
    provider.decisions.append(FinalAnswer("done"))
    executor = _Executor()

    outcome = runtime._run_model_loop(
        provider,
        "implement",
        executor,
        max_calls=1,
    )

    assert len(outcome.tool_results) == 1
    assert [call.id for call in executor.calls] == ["read"]
    assert provider.parallel_tool_call_violations == 16


def test_named_tool_reentry_counts_only_existing_and_new_executions() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read-2", "read_files", {"paths": ["other.py"]}),
        violations=0,
    )
    provider.parallel_tool_call_violations = 1
    submit = ToolCall("plan", "submit_change_plan", {})
    provider.decisions.append(submit)
    executor = _Executor()
    results = [ToolResult("read-1", "read_files", True, "first")]

    decision, returned_results = runtime._run_until_named_tool(
        provider,
        "inspect",
        executor,
        target="submit_change_plan",
        max_calls=2,
        tool_results=results,
    )

    assert decision is submit
    assert len(returned_results) == 2
    assert [call.id for call in executor.calls] == ["read-2"]
    assert provider.parallel_tool_call_violations == 1


def test_change_reentry_counts_only_existing_and_new_executions() -> None:
    provider = _ParallelViolationProvider(
        ToolCall("read-2", "read_files", {"paths": ["other.py"]}),
        violations=0,
    )
    provider.parallel_tool_call_violations = 1
    provider.decisions.append(FinalAnswer("done"))
    executor = _Executor()
    results = [ToolResult("read-1", "read_files", True, "first")]

    outcome = runtime._run_model_loop(
        provider,
        "implement",
        executor,
        max_calls=2,
        tool_results=results,
    )

    assert outcome.ok is True
    assert len(outcome.tool_results) == 2
    assert [call.id for call in executor.calls] == ["read-2"]
    assert provider.parallel_tool_call_violations == 1


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
    assert len(repeated.tool_results) == 2
    assert runtime._tool_outcome(repeated.tool_results[-1]) == "invalid_args"
    assert progressed.ok is True
    assert len(progressed.tool_results) == 1


def test_durable_model_loop_prevents_provider_calls_after_tool_budget() -> None:
    no_budget_provider = _DecisionProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]})
    )
    no_budget = runtime._run_model_loop(
        no_budget_provider,
        "implement",
        _Executor(),
        max_calls=0,
        allow_terminal_response_after_budget=False,
    )

    assert no_budget.tool_budget_exhausted is True
    assert no_budget_provider.calls == 0
    assert no_budget.tool_results == ()

    call = ToolCall("read", "read_files", {"paths": ["app.py"]})
    exhausted_provider = _DecisionProvider(call, FinalAnswer("unneeded"))
    exhausted = runtime._run_model_loop(
        exhausted_provider,
        "implement",
        _Executor(),
        max_calls=1,
        allow_terminal_response_after_budget=False,
    )

    assert exhausted.tool_budget_exhausted is True
    assert exhausted_provider.calls == 1
    assert len(exhausted.tool_results) == 1
    assert len(exhausted_provider.decisions) == 1


def test_model_loop_retries_unknown_tool_without_resetting_provider_history() -> None:
    class UnknownToolThenValidProvider(_DecisionProvider):
        def __init__(self) -> None:
            super().__init__(ToolCall("read", "read_files", {"paths": ["app.py"]}))
            self.requests: list[tuple[str, tuple[ToolResult, ...]]] = []
            self.first = True
            self.recovered = 0

        def next_step(self, task, results):
            self.requests.append((task, results))
            if self.first:
                self.first = False
                raise OpenAIUnknownToolError(
                    "model requested an unknown or inactive tool"
                )
            return super().next_step(task, results)

        def mark_unknown_tool_recovered(self) -> None:
            self.recovered += 1

    provider = UnknownToolThenValidProvider()
    executor = _Executor()
    before_response_calls: list[int] = []

    outcome = runtime._run_model_loop(
        provider,
        "implement the repair",
        executor,
        max_calls=1,
        allow_terminal_response_after_budget=False,
        retry_unknown_tool=True,
        before_response=lambda: before_response_calls.append(1),
    )

    assert outcome.tool_budget_exhausted is True
    assert provider.calls == 1
    assert len(provider.requests) == 2
    assert provider.recovered == 1
    assert before_response_calls == [1]
    assert provider.requests[0] == provider.requests[1] == (
        "implement the repair",
        (),
    )
    assert [call.id for call in executor.calls] == ["read"]


def test_model_loop_does_not_retry_unknown_tool_without_opt_in() -> None:
    class UnknownToolProvider(_DecisionProvider):
        def next_step(self, _task, _results):
            self.calls += 1
            raise OpenAIUnknownToolError(
                "model requested an unknown or inactive tool"
            )

    provider = UnknownToolProvider()
    with pytest.raises(OpenAIUnknownToolError, match="unknown or inactive tool"):
        runtime._run_model_loop(
            provider,
            "implement",
            _Executor(),
            max_calls=1,
            allow_terminal_response_after_budget=False,
            before_response=lambda: pytest.fail(
                "no retry budget check should run without explicit recovery"
            ),
        )

    assert provider.calls == 1


def test_durable_model_loop_fails_closed_after_second_unknown_tool() -> None:
    class UnknownToolProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.recovered = 0

        def next_step(self, _task, _results):
            self.calls += 1
            raise OpenAIUnknownToolError(
                "model requested an unknown or inactive tool"
            )

        def mark_unknown_tool_recovered(self) -> None:
            self.recovered += 1

    provider = UnknownToolProvider()
    executor = _Executor()
    budget_checks: list[int] = []

    with pytest.raises(OpenAIUnknownToolError, match="unknown or inactive tool"):
        runtime._run_model_loop(
            provider,
            "implement",
            executor,
            max_calls=1,
            retry_unknown_tool=True,
            before_response=lambda: budget_checks.append(1),
        )

    assert provider.calls == 2
    assert provider.recovered == 0
    assert budget_checks == [1]
    assert executor.calls == []


def test_durable_model_loop_allows_only_one_unknown_recovery_per_phase() -> None:
    class UnknownValidUnknownProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.recovered = 0

        def next_step(self, _task, _results):
            self.calls += 1
            if self.calls == 1:
                raise OpenAIUnknownToolError(
                    "model requested an unknown or inactive tool"
                )
            if self.calls == 2:
                return ToolCall("read", "read_files", {"paths": ["app.py"]})
            raise OpenAIUnknownToolError(
                "model requested an unknown or inactive tool"
            )

        def mark_unknown_tool_recovered(self) -> None:
            self.recovered += 1

    provider = UnknownValidUnknownProvider()
    executor = _Executor()
    budget_checks: list[int] = []

    with pytest.raises(OpenAIUnknownToolError, match="unknown or inactive tool"):
        runtime._run_model_loop(
            provider,
            "implement",
            executor,
            max_calls=2,
            retry_unknown_tool=True,
            before_response=lambda: budget_checks.append(1),
        )

    assert provider.calls == 3
    assert provider.recovered == 1
    assert budget_checks == [1, 1]
    assert [call.id for call in executor.calls] == ["read"]


def test_durable_model_loop_retries_malformed_tool_arguments_once() -> None:
    class MalformedThenValidProvider(_DecisionProvider):
        def __init__(self) -> None:
            super().__init__(ToolCall("read", "read_files", {"paths": ["app.py"]}))
            self.requests: list[tuple[str, tuple[ToolResult, ...]]] = []
            self.malformed = True
            self.recovered = 0

        def next_step(self, task, results):
            self.requests.append((task, results))
            if self.malformed:
                self.malformed = False
                raise OpenAIToolArgumentsError("tool arguments is invalid JSON")
            return super().next_step(task, results)

        def mark_malformed_tool_arguments_recovered(self) -> None:
            self.recovered += 1

    provider = MalformedThenValidProvider()
    executor = _Executor()
    before_response_calls: list[int] = []

    outcome = runtime._run_model_loop(
        provider,
        "implement",
        executor,
        max_calls=1,
        allow_terminal_response_after_budget=False,
        retry_unknown_tool=False,
        retry_malformed_tool_arguments=True,
        before_response=lambda: before_response_calls.append(1),
    )

    assert outcome.tool_budget_exhausted is True
    assert len(provider.requests) == 2
    assert provider.recovered == 1
    assert before_response_calls == [1]
    assert provider.requests[0] == provider.requests[1] == ("implement", ())
    assert [call.id for call in executor.calls] == ["read"]


def test_durable_model_loop_fails_closed_after_second_malformed_arguments() -> None:
    class MalformedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def next_step(self, _task, _results):
            self.calls += 1
            raise OpenAIToolArgumentsError("tool arguments is invalid JSON")

    provider = MalformedProvider()
    budget_checks: list[int] = []

    with pytest.raises(OpenAIToolArgumentsError, match="invalid JSON"):
        runtime._run_model_loop(
            provider,
            "implement",
            _Executor(),
            max_calls=1,
            retry_malformed_tool_arguments=True,
            before_response=lambda: budget_checks.append(1),
        )

    assert provider.calls == 2
    assert budget_checks == [1]


def test_malformed_arguments_retry_stops_before_http_when_budget_is_exhausted() -> None:
    class MalformedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def next_step(self, _task, _results):
            self.calls += 1
            raise OpenAIToolArgumentsError("tool arguments is invalid JSON")

    provider = MalformedProvider()
    with pytest.raises(runtime._PhaseBudgetExhausted):
        runtime._run_model_loop(
            provider,
            "implement",
            _Executor(),
            max_calls=1,
            retry_malformed_tool_arguments=True,
            before_response=lambda: (_ for _ in ()).throw(
                runtime._PhaseBudgetExhausted("budget exhausted")
            ),
        )

    assert provider.calls == 1


def test_unknown_tool_retry_stops_before_second_request_when_budget_is_exhausted() -> None:
    class UnknownToolProvider(_DecisionProvider):
        def next_step(self, _task, _results):
            self.calls += 1
            raise OpenAIUnknownToolError(
                "model requested an unknown or inactive tool"
            )

    provider = UnknownToolProvider()
    with pytest.raises(runtime._PhaseBudgetExhausted):
        runtime._run_model_loop(
            provider,
            "implement",
            _Executor(),
            max_calls=1,
            retry_unknown_tool=True,
            before_response=lambda: (_ for _ in ()).throw(
                runtime._PhaseBudgetExhausted("budget exhausted")
            ),
        )

    assert provider.calls == 1


def test_model_loop_does_not_execute_a_tool_after_response_time_cancellation() -> None:
    cancelled = False
    executor = _Executor()

    class CancellingProvider(_DecisionProvider):
        def next_step(self, task, results):
            nonlocal cancelled
            decision = super().next_step(task, results)
            cancelled = True
            return decision

    def check_cancelled() -> None:
        if cancelled:
            raise runtime.WorkflowCancelled("run was cancelled")

    with pytest.raises(runtime.WorkflowCancelled, match="run was cancelled"):
        runtime._run_model_loop(
            CancellingProvider(
                ToolCall("read", "read_files", {"paths": ["app.py"]})
            ),
            "task",
            executor,
            max_calls=1,
            check_cancelled=check_cancelled,
        )

    assert executor.calls == []


@pytest.mark.parametrize("has_required_regression_test", (True, False))
def test_model_loop_stops_after_successful_patch_only_for_verifiable_candidate(
    has_required_regression_test: bool,
) -> None:
    provider = _DecisionProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
        FinalAnswer("continued because the candidate is incomplete"),
    )
    checks: list[bool] = []

    outcome = runtime._run_model_loop(
        provider,
        "task",
        _Executor(),
        max_calls=1,
        stop_after_successful_tool="apply_patch",
        stop_after_successful_tool_check=lambda: (
            checks.append(has_required_regression_test)
            or has_required_regression_test
        ),
    )

    assert len(outcome.tool_results) == 1
    assert checks == [has_required_regression_test]
    if has_required_regression_test:
        assert outcome.ok is True
        assert outcome.summary == "candidate submitted for independent verification"
        assert len(provider.decisions) == 1
    else:
        assert outcome.ok is False
        assert outcome.error == (
            "model returned text before submitting a verifiable candidate"
        )
        assert provider.decisions == []


def test_model_loop_skips_result_transform_for_terminal_patch() -> None:
    provider = _DecisionProvider(
        ToolCall("patch", "apply_patch", {"unified_diff": "patch text"}),
    )

    def unexpected_transform(_result):
        raise AssertionError("terminal patch must not consume continuation budget")

    outcome = runtime._run_model_loop(
        provider,
        "task",
        _Executor(),
        max_calls=1,
        allow_terminal_response_after_budget=False,
        stop_after_successful_tool="apply_patch",
        stop_after_successful_tool_check=lambda: True,
        result_transform=unexpected_transform,
    )

    assert outcome.ok is True
    assert len(outcome.tool_results) == 1
    assert provider.calls == 1


def test_repair_loop_feeds_no_effect_error_back_before_accepting_correction() -> None:
    class RecordingProvider(_DecisionProvider):
        def __init__(self, *decisions) -> None:
            super().__init__(*decisions)
            self.results_seen: list[tuple[ToolResult, ...]] = []

        def next_step(self, task, results):
            self.results_seen.append(tuple(results))
            return super().next_step(task, results)

    provider = RecordingProvider(
        ToolCall("no-effect", "apply_patch", {"unified_diff": "same"}),
        ToolCall("corrected", "apply_patch", {"unified_diff": "fixed"}),
    )
    executor = _NoEffectThenCorrectionExecutor()

    outcome = runtime._run_model_loop(
        provider,
        "repair",
        executor,
        max_calls=2,
        allow_terminal_response_after_budget=False,
        stop_after_successful_tool="apply_patch",
        stop_after_successful_tool_check=lambda: True,
    )

    assert outcome.ok is True
    assert [call.id for call in executor.calls] == ["no-effect", "corrected"]
    assert "no effective change" in (provider.results_seen[1][-1].error or "")
    assert executor.workspace_revision == 1
    assert outcome.summary == "candidate submitted for independent verification"


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


def test_run_budget_prefers_executed_tool_calls_over_telemetry_total(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "phase": "implement",
            "executed_tool_calls": 2,
            "tool_calls": 8,
            "parallel_tool_call_violations": 6,
            "model_usage": {"complete": True, "total_tokens": 100},
        },
    )

    assert operations._model_budget(_context(record, "implement")) == (
        runtime.MAX_RUN_TOOL_CALLS - 2,
        100,
    )
    assert operations._phase_model_budget(
        _context(record, "implement"),
        phases=frozenset({"implement", "repair"}),
        tool_limit=runtime.MAX_CHANGE_TOOL_CALLS,
        token_limit=runtime.MAX_CHANGE_TOKENS,
        label="change",
    ) == (runtime.MAX_CHANGE_TOOL_CALLS - 2, 100)


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
    change_tool_names = tuple(
        definition.name for definition in runtime._change_tools()
    )
    assert change_tool_names == ("read_files", "write_files")

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
    input_estimates: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=1_000),
        last_response_usage=SimpleNamespace(complete=True, total_tokens=100),
        pending_internal_tool_result_bytes=37,
        set_response_token_budget=response_budgets.append,
        set_next_response_input_token_estimate=input_estimates.append,
    )
    estimate = runtime._next_response_token_estimate(
        provider, (result,), require_complete=True
    )
    result_bytes = tool_result_replay_bytes(result)
    assert estimate == (
        100
        + result_bytes
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
    assert input_estimates == [estimate]
    with pytest.raises(runtime._PhaseBudgetExhausted, match="predicted next"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=1_000 + estimate - 1,
            soft_phase_boundary=True,
            new_tool_results=(result,),
        )


def test_local_openai_usage_gap_stops_before_an_uncapped_follow_up(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    response_budgets: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(
            response_count=1,
            reported_response_count=0,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            complete=False,
        ),
        last_response_usage=SimpleNamespace(
            response_count=1,
            reported_response_count=0,
            total_tokens=None,
            complete=False,
        ),
        set_response_token_budget=response_budgets.append,
    )

    with pytest.raises(RuntimeError, match="model token usage is incomplete"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=runtime.MAX_CHANGE_TOKENS,
            phase_label="change",
            new_tool_results=(
                ToolResult("read", "read_files", True, "evidence"),
            ),
            reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
        )

    assert response_budgets == []


def test_request_not_started_requires_an_explicit_request_counter() -> None:
    unknown = SimpleNamespace(
        usage=SimpleNamespace(response_count=0, complete=False)
    )
    responded = SimpleNamespace(
        usage=SimpleNamespace(response_count=1, complete=False)
    )

    assert runtime._provider_request_count(unknown) is None
    assert runtime._provider_request_count(responded) == 1
    assert runtime._provider_request_count(
        SimpleNamespace(request_count=0, usage=unknown.usage)
    ) == 0


def test_tool_result_fit_reserves_provider_replay_bytes() -> None:
    result = ToolResult("read", "read_files", True, "x" * 1_000)
    maximum_result_bytes = 300
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
        phase_label="change",
        reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
        require_complete=True,
    )

    assert fitted.truncated is True
    assert tool_result_replay_bytes(fitted) <= (
        (
            phase_limit
            - current_tokens
            - last_tokens
            - pending_bytes
            - runtime.NEXT_RESPONSE_TOKEN_MARGIN
        )
    )


def test_ascii_tool_result_near_limit_fails_before_provider_request(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    result = ToolResult("read", "read_files", True, "x" * 3_000)
    legacy_json_bytes = len(runtime._json(runtime.asdict(result)).encode("utf-8"))
    result_bytes = tool_result_replay_bytes(result)
    current_tokens = 19_000
    last_tokens = 500
    response_budgets: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=current_tokens),
        last_response_usage=SimpleNamespace(
            complete=True,
            total_tokens=last_tokens,
        ),
        set_response_token_budget=response_budgets.append,
    )

    legacy_estimate = (
        last_tokens
        + runtime._bytes_to_estimated_tokens(legacy_json_bytes)
        + runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )
    assert current_tokens + legacy_estimate <= runtime.MAX_CHANGE_TOKENS
    assert (
        current_tokens
        + last_tokens
        + result_bytes
        + runtime.NEXT_RESPONSE_TOKEN_MARGIN
        > runtime.MAX_CHANGE_TOKENS
    )

    with pytest.raises(runtime._PhaseBudgetExhausted, match="predicted next"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=runtime.MAX_CHANGE_TOKENS,
            phase_label="change",
            soft_phase_boundary=True,
            new_tool_results=(result,),
            reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
        )

    assert response_budgets == []


def test_extreme_unicode_replay_is_charged_and_blocked_before_provider_request(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record_value = operations_bundle
    result = ToolResult(
        "unicode-read",
        "read_files",
        True,
        "\U0001f600" * 4_096,
        exit_code=0,
    )
    assert len(result.output.encode("utf-8")) == 16 * 1_024

    unescaped_json_bytes = len(
        runtime._json(runtime.asdict(result)).encode("utf-8")
    )
    replay_bytes = tool_result_replay_bytes(result)
    pending_bytes = 29
    last_tokens = 100
    current_tokens = 1_000
    response_budgets: list[int] = []
    input_estimates: list[int] = []
    provider = SimpleNamespace(
        usage=SimpleNamespace(complete=True, total_tokens=current_tokens),
        last_response_usage=SimpleNamespace(
            complete=True,
            total_tokens=last_tokens,
        ),
        pending_internal_tool_result_bytes=pending_bytes,
        set_response_token_budget=response_budgets.append,
        set_next_response_input_token_estimate=input_estimates.append,
    )

    estimate = runtime._next_response_token_estimate(
        provider,
        (result,),
        require_complete=True,
    )

    assert replay_bytes > unescaped_json_bytes * 3
    assert estimate == (
        last_tokens
        + replay_bytes
        + pending_bytes
        + runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )
    with pytest.raises(runtime._PhaseBudgetExhausted, match="predicted next"):
        operations._enforce_next_model_response(
            0,
            provider,
            phase_prior_tokens=0,
            phase_token_limit=current_tokens + estimate - 1,
            phase_label="change",
            soft_phase_boundary=True,
            new_tool_results=(result,),
        )

    assert response_budgets == []
    assert input_estimates == []


def test_observed_change_usage_and_batch_read_leave_capacity_for_follow_up(
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
    result_bytes = tool_result_replay_bytes(result)
    assert 1_350 <= result_bytes <= 1_400
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
        phase_token_limit=runtime.MAX_CHANGE_TOKENS,
        phase_label="change",
        new_tool_results=(result,),
        reserved_run_tokens=runtime.MAX_REVIEW_TOKENS,
    )

    assert response_budgets == [
        runtime.MAX_CHANGE_TOKENS - first_response_tokens
    ]


def test_planning_is_one_forced_response_without_repository_tool_execution(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    definitions_seen: list[tuple[runtime.ToolDefinition, ...]] = []

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
            self.requests: list[tuple[str, tuple[ToolResult, ...]]] = []

        def set_response_token_budget(self, total_tokens: int) -> None:
            self.response_token_budgets.append(total_tokens)

        def set_forced_tool_choice(self, tool_name: str) -> None:
            self.forced_tool_choices.append(tool_name)

        def next_step(
            self, task: str, results: tuple[ToolResult, ...]
        ) -> ToolCall:
            self.requests.append((task, results))
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

    def provider_factory(definitions, *_args):
        definitions_seen.append(tuple(definitions))
        return provider

    operations._provider_factory = provider_factory
    operations._repository_map = lambda _context: "repository map"
    operations._tool_executor = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("planning must not create a repository tool executor")
    )

    plan = operations.inspect_and_plan(
        _context(record, "inspect_and_plan", suffix="single-response")
    )

    assert plan.goal == "repair"
    assert definitions_seen == [(runtime.PLAN_TOOL,)]
    assert len(provider.requests) == 1
    assert "R:\nrepository map" in provider.requests[0][0]
    assert provider.requests[0][1] == ()
    assert provider.response_token_budgets == [runtime.MAX_PLANNING_TOKENS]
    assert provider.forced_tool_choices == ["submit_change_plan"]
    summary = _only_model_summary(database, record.run_id)
    assert summary["executed_tool_calls"] == 0
    assert summary["tool_calls"] == 0


def test_planning_discards_a_decision_returned_after_cancellation(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    provider = _DecisionProvider(
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
    next_step = provider.next_step

    def cancel_during_response(task, results):
        decision = next_step(task, results)
        operations.cancel_event.set()
        return decision

    provider.next_step = cancel_during_response  # type: ignore[method-assign]
    operations._provider_factory = lambda *_args: provider
    operations._repository_map = lambda _context: "repository map"

    with pytest.raises(runtime.WorkflowCancelled, match="run was cancelled"):
        operations.inspect_and_plan(
            _context(record, "inspect_and_plan", suffix="cancelled-response")
        )

    assert provider.forced_tool_choices == ["submit_change_plan"]


def test_change_truncates_tool_result_before_provider_history(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_change_regression.py "
        "b/tests/test_change_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_change_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )

    class ObservedChangeProvider:
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
        ) -> ToolCall | FinalAnswer:
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
            if len(self.result_histories) == 2:
                return ToolCall(
                    "patch",
                    "write_files",
                    {"files": [{"path": "app.py", "content": "changed\n"}]},
                )
            raise AssertionError("successful write_files must end the model loop")

    provider = ObservedChangeProvider()
    executor = _Executor()
    executor.execute = lambda decision: ToolResult(  # type: ignore[method-assign]
        decision.id,
        decision.name,
        True,
        "x" * 100_000 if decision.name == "read_files" else "patched",
        exit_code=0,
    )
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("tests/test_change_regression.py",),
        patch,
    )
    definitions_seen: list[tuple[str, ...]] = []

    def provider_factory(definitions, *_args):
        definitions_seen.append(tuple(item.name for item in definitions))
        return provider

    operations._provider_factory = provider_factory
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: executor

    outcome = operations.implement(
        _context(record, "implement", suffix="trim-change-result")
    )

    delivered = provider.result_histories[1][0]
    maximum_result_bytes = (
        runtime.MAX_CHANGE_TOKENS
        - 873
        - 873
        - runtime.NEXT_RESPONSE_TOKEN_MARGIN
    )
    assert outcome.ok is True
    assert outcome.summary == "candidate submitted for independent verification"
    assert definitions_seen == [("read_files", "write_files")]
    assert provider.forced_tool_choices == ["read_files", "read_files"]
    assert len(provider.result_histories) == 2
    assert delivered.truncated is True
    assert delivered.output.endswith(runtime._PLANNING_TOOL_RESULT_TRUNCATION_MARKER)
    assert tool_result_replay_bytes(delivered) == maximum_result_bytes


def test_change_allows_phase_tools_after_the_initial_read(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_tool_state_regression.py "
        "b/tests/test_tool_state_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_tool_state_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    provider = _DecisionProvider(
        ToolCall("read-bad", "read_files", {"paths": []}),
        ToolCall("read-good", "read_files", {"paths": ["app.py"]}),
        ToolCall("patch-bad", "write_files", {"files": []}),
        ToolCall(
            "patch-good",
            "write_files",
            {"files": [{"path": "app.py", "content": "changed\n"}]},
        ),
    )

    class ScriptedExecutor:
        workspace_revision = 0
        mutation_persistence_failed = False

        def __init__(self) -> None:
            self.calls: list[ToolCall] = []
            self.outcomes = iter((False, True, False, True))

        def execute(self, decision: ToolCall) -> ToolResult:
            self.calls.append(decision)
            ok = next(self.outcomes)
            return ToolResult(
                decision.id,
                decision.name,
                ok,
                "ok" if ok else "",
                error=None if ok else "rejected",
            )

    executor = ScriptedExecutor()
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("tests/test_tool_state_regression.py",),
        patch,
    )
    operations._provider_factory = lambda *_args: provider
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: executor

    outcome = operations.implement(
        _context(record, "implement", suffix="tool-state-retries")
    )

    assert outcome.ok is True
    assert [call.id for call in executor.calls] == [
        "read-bad",
        "read-good",
        "patch-bad",
        "patch-good",
    ]
    assert provider.forced_tool_choices == [
        "read_files",
        "read_files",
        "write_files",
        "write_files",
    ]
    assert provider.decisions == []


def test_change_recovers_missing_complete_read_before_retrying_write(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_tool_state_regression.py "
        "b/tests/test_tool_state_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_tool_state_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    provider = _DecisionProvider(
        ToolCall("read-main", "read_files", {"paths": ["app.py"]}),
        ToolCall(
            "write-unread",
            "write_files",
            {"files": [{"path": "package/__init__.py", "content": "value = 1\n"}]},
        ),
        ToolCall(
            "read-missing",
            "read_files",
            {"paths": ["package/__init__.py"]},
        ),
        ToolCall(
            "write-good",
            "write_files",
            {"files": [{"path": "app.py", "content": "changed\n"}]},
        ),
    )

    class ScriptedExecutor:
        workspace_revision = 0
        mutation_persistence_failed = False

        def __init__(self) -> None:
            self.calls: list[ToolCall] = []

        def execute(self, decision: ToolCall) -> ToolResult:
            self.calls.append(decision)
            if decision.id == "write-unread":
                return ToolResult(
                    decision.id,
                    decision.name,
                    False,
                    "",
                    error=(
                        "ValueError: write_files target must be read completely "
                        "before editing: package/__init__.py"
                    ),
                )
            return ToolResult(decision.id, decision.name, True, "ok")

    executor = ScriptedExecutor()
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("tests/test_tool_state_regression.py",),
        patch,
    )
    operations._provider_factory = lambda *_args: provider
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: executor

    outcome = operations.implement(
        _context(record, "implement", suffix="missing-complete-read")
    )

    assert outcome.ok is True
    assert [call.id for call in executor.calls] == [
        "read-main",
        "write-unread",
        "read-missing",
        "write-good",
    ]
    assert provider.forced_tool_choices == [
        "read_files",
        "write_files",
        "read_files",
        "write_files",
    ]


def test_change_rejects_candidate_without_required_regression_test(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    candidate = CandidateArtifact("a" * 40, 1, ("app.py",), patch)
    provider = _DecisionProvider(
        ToolCall("read", "read_files", {"paths": ["app.py"]}),
        ToolCall(
            "patch",
            "write_files",
            {"files": [{"path": "app.py", "content": "changed\n"}]},
        ),
        FinalAnswer("done"),
    )
    operations._provider_factory = lambda *_args: provider
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    outcome = operations.implement(
        _context(record, "implement", suffix="missing-regression-test")
    )

    assert outcome.ok is False
    assert outcome.error == "candidate patch does not add the required regression test"
    assert provider.forced_tool_choices == [
        "read_files",
        "write_files",
        "write_files",
    ]
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == ""


def test_candidate_regression_contract_is_python_only(monkeypatch) -> None:
    def unexpected_profile_detection(_repo_path):
        raise AssertionError("candidate validation must not detect a check profile")

    monkeypatch.setattr(runtime, "detect_check_profile", unexpected_profile_detection)
    expected_by_path = {
        "tests/test_value_regression.py": True,
        "tests/unit/test_value_regression.py": False,
        "tests/test_value.py": False,
        "src/test/java/FooRegressionTest.java": False,
    }

    for path, expected in expected_by_path.items():
        patch = (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            "@@ -0,0 +1 @@\n"
            "+def test_regression(): assert True\n"
        )
        candidate = CandidateArtifact("a" * 40, 1, (path,), patch)

        assert runtime._candidate_has_required_regression_test(
            candidate, "unused"
        ) is expected


def test_candidate_scope_allows_approved_paths_and_new_regression_only() -> None:
    plan = ChangePlan(
        goal="Repair the application",
        files=("app.py",),
        steps=("Update the approved implementation file.",),
        checks=("pytest",),
    )

    def candidate_for(path: str) -> CandidateArtifact:
        patch = (
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tests/test_app_regression.py "
            "b/tests/test_app_regression.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_app_regression.py\n"
            "@@ -0,0 +1 @@\n"
            "+def test_regression(): assert True\n"
        )
        return CandidateArtifact(
            "a" * 40,
            1,
            (path, "tests/test_app_regression.py"),
            patch,
        )

    assert runtime._candidate_paths_outside_approved_scope(
        candidate_for("app.py"), plan
    ) == ()
    assert runtime._candidate_paths_outside_approved_scope(
        candidate_for("unplanned.py"), plan
    ) == ("unplanned.py",)
    assert runtime._candidate_paths_outside_approved_scope(
        candidate_for("unplanned.py"), None
    ) == ()


def test_change_rejects_candidate_outside_approved_scope(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    record = record.model_copy(
        update={
            "plan": ChangePlan(
                goal="Repair the application",
                files=("app.py",),
                steps=("Only update the approved implementation file.",),
                checks=("pytest",),
            )
        }
    )
    patch = (
        "diff --git a/unplanned.py b/unplanned.py\n"
        "--- a/unplanned.py\n"
        "+++ b/unplanned.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/tests/test_app_regression.py "
        "b/tests/test_app_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_app_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("unplanned.py", "tests/test_app_regression.py"),
        patch,
    )
    _stub_change_execution(monkeypatch, operations, _CandidateSandbox(candidate))

    outcome = operations.implement(
        _context(record, "implement", suffix="scope-violation")
    )

    assert outcome.ok is False
    assert outcome.error == (
        "candidate changed paths outside the approved plan: unplanned.py"
    )
    assert outcome.detail["failure_status"] == "policy_denied"
    assert outcome.detail["failure_code"] == "candidate_scope_violation"
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == ""


def test_change_terminal_candidate_avoids_unfittable_tool_result(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_envelope_regression.py "
        "b/tests/test_envelope_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_envelope_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_envelope_regression.py",), patch
    )

    class NearCapacityProvider(_DecisionProvider):
        def __init__(self) -> None:
            super().__init__()
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

        def next_step(self, _task, _results):
            self.calls += 1
            self.usage = SimpleNamespace(
                response_count=1,
                reported_response_count=1,
                input_tokens=11_200,
                cached_input_tokens=0,
                output_tokens=100,
                total_tokens=11_300,
                complete=True,
            )
            self.last_response_usage = SimpleNamespace(
                total_tokens=11_300,
                complete=True,
            )
            return ToolCall(
                "patch",
                "write_files",
                {"files": [{"path": "app.py", "content": "changed\n"}]},
            )

    provider = NearCapacityProvider()
    operations._provider_factory = lambda *_args: provider
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    outcome = operations.implement(
        _context(record, "implement", suffix="unfittable-result-envelope")
    )

    assert outcome.ok is True
    assert outcome.summary == "candidate submitted for independent verification"
    assert outcome.detail["soft_budget_boundary"] is False
    assert provider.calls == 1
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch


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
    assert tool_result_replay_bytes(with_reservation) == reserved_result_bytes


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
    assert planning_provider.forced_tool_choices == ["submit_change_plan"]

    review_record = record.model_copy(update={"run_id": "d" * 32})
    review_database = RunDatabase(artifacts.root.parent / "review-runs.sqlite3")
    review_artifacts = ArtifactStore(artifacts.root.parent / "review-artifacts")
    review_directory = review_artifacts.initialize(review_record.run_id)
    review_database.create(review_record, review_directory)
    review_artifacts.write_patch(review_record.run_id, _SMALL_REVIEW_PATCH)
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
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_budget_regression.py "
        "b/tests/test_budget_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_budget_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40,
        1,
        ("tests/test_budget_regression.py",),
        patch,
    )
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()
    provider = _DecisionProvider(FinalAnswer("done"))
    operations._provider_factory = lambda *_args: provider

    outcome = operations.implement(
        _context(record, "implement", suffix="initial-change-budget")
    )

    assert outcome.ok is False
    assert outcome.error == "model returned text before submitting a verifiable candidate"
    assert provider.response_token_budgets == [runtime.MAX_CHANGE_TOKENS]
    assert provider.forced_tool_choices == ["read_files"]


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


@pytest.mark.parametrize(
    "error",
    [
        "Invalid tool call: ToolValidationError: hunk line counts do not match its header",
        "ValueError: patch must be a Git-style unified diff; submit a complete diff",
        "Patch did not apply cleanly to the candidate workspace",
    ],
)
def test_apply_patch_validation_failures_are_patch_rejections(error: str) -> None:
    result = ToolResult("patch", "apply_patch", False, "", error=error)

    assert runtime._tool_outcome(result) == "patch_rejected"


def test_model_summary_keeps_legacy_counts_and_adds_tool_outcomes(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    results = (
        ToolResult("ok", "read_file", True, "ok"),
        ToolResult(
            "bad",
            "write_files",
            False,
            "sensitive successful output must not be logged",
            error="Patch rejected; Authorization: Bearer fake",
        ),
    )

    provider = SimpleNamespace(
        usage=None,
        parallel_tool_call_violations=2,
        rejected_tool_call_violations=1,
    )
    operations._append_model_summary(
        _context(record, "implement"),
        event="model_tool_summary",
        phase="implement",
        provider=provider,
        tool_results=results,
    )

    summary = _only_model_summary(database, record.run_id)
    assert isinstance(summary["timestamp"], str)
    assert summary["timestamp"].endswith("Z")
    assert summary["executed_tool_calls"] == 2
    assert summary["tool_calls"] == 5
    assert summary["tool_errors"] == 4
    assert summary["parallel_tool_call_violations"] == 2
    assert summary["rejected_tool_call_violations"] == 1
    assert summary["tool_outcomes"]["ok"] == 1
    assert summary["tool_outcomes"]["invalid_args"] == 3
    assert summary["tool_outcomes"]["patch_rejected"] == 1
    assert summary["tool_failures"] == [
        {
            "ordinal": 2,
            "tool": "write_files",
            "outcome": "patch_rejected",
            "reason_code": "patch_rejected",
            "error_summary": "Patch rejected; Authorization: [REDACTED]",
        }
    ]
    assert "sensitive successful output" not in str(summary["tool_failures"])


def test_recovered_argument_retry_is_visible_but_not_actionable(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = SimpleNamespace(
        usage=None,
        parallel_tool_call_violations=0,
        rejected_tool_call_violations=1,
        recovered_tool_argument_violations=1,
    )

    operations._append_model_summary(
        _context(record, "implement", suffix="recovered-arguments"),
        event="model_tool_summary",
        phase="implement",
        provider=provider,
        tool_results=(),
    )

    summary = _only_model_summary(database, record.run_id)
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 1
    assert summary["rejected_tool_call_violations"] == 1
    assert summary["recovered_protocol_errors"] == 1
    assert summary["tool_outcomes"]["invalid_args"] == 0


def test_recovered_unknown_tool_retry_is_visible_but_not_actionable(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = SimpleNamespace(
        usage=None,
        parallel_tool_call_violations=0,
        rejected_tool_call_violations=1,
        recovered_unknown_tool_violations=1,
    )

    operations._append_model_summary(
        _context(record, "implement", suffix="recovered-unknown-tool"),
        event="model_tool_summary",
        phase="implement",
        provider=provider,
        tool_results=(),
    )

    summary = _only_model_summary(database, record.run_id)
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 1
    assert summary["rejected_tool_call_violations"] == 1
    assert summary["recovered_protocol_errors"] == 1
    assert summary["tool_outcomes"]["invalid_args"] == 0


def test_change_system_prompt_reserves_pytest_for_the_workflow() -> None:
    prompt = runtime._implementation_system_prompt()

    assert "workflow runs pytest after the change phase" in prompt
    assert "do not request shell, pytest, or run_check tools" in prompt


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


def test_planning_provider_failure_preserves_usage_summary(
    operations_bundle,
) -> None:
    operations, database, _artifacts, record = operations_bundle
    provider = _FailingProvider(total_tokens=19)
    operations._provider_factory = lambda *_args: provider
    operations._repository_map = lambda _context: "repository map"
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.inspect_and_plan(_context(record, "inspect_and_plan", suffix="failure"))

    summary = _only_model_summary(database, record.run_id)
    assert summary["event"] == "planning_tool_summary"
    assert summary["phase"] == "planning"
    assert summary["executed_tool_calls"] == 0
    assert summary["tool_calls"] == 0
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
    operations, database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
    operations._provider_factory = lambda *_args: _FailingProvider(total_tokens=29)

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.review(_context(record, "review", suffix="failure"))

    summary = _only_model_summary(database, record.run_id)
    assert summary["event"] == "model_tool_summary"
    assert summary["phase"] == "review"
    assert summary["tool_calls"] == 0
    assert summary["tool_errors"] == 0
    assert summary["tool_outcomes"]["invalid_args"] == 0
    assert summary["model_usage"]["total_tokens"] == 29


def test_summary_write_failure_does_not_mask_provider_failure(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
    operations._provider_factory = lambda *_args: _FailingProvider()
    operations._append_model_summary = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("summary write failed")
    )

    with pytest.raises(RuntimeError, match="provider failed after reporting usage"):
        operations.review(_context(record, "review", suffix="summary-failure"))


def test_summary_write_failure_propagates_without_provider_failure(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
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
    operations, database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
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
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
    operations._provider_factory = lambda *_args: _DecisionProvider(decision)

    with pytest.raises((RuntimeError, ValueError), match=error):
        operations.review(_context(record, "review", suffix=error))


def test_review_preserves_rejection_as_node_failure(operations_bundle) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
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


def test_planning_tool_keeps_review_semantics_bounded() -> None:
    properties = runtime.PLAN_TOOL.input_schema["properties"]
    steps = properties["steps"]
    risks = properties["risks"]

    assert steps["maxItems"] == 2
    assert steps["items"]["maxLength"] == 80
    assert risks["maxItems"] == 2
    assert risks["items"]["maxLength"] == 40


def test_planning_task_and_full_map_fit_fixed_budget_with_output_room() -> None:
    system_prompt = runtime._planning_system_prompt()
    task = (
        f"T:{runtime._json('x' * 500)}\nR:\n"
        + ("r" * runtime.MAX_PLANNING_REPOSITORY_MAP_BYTES)
    )
    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-provider-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        ),
        tool_definitions=(runtime.PLAN_TOOL,),
        system_prompt=system_prompt,
        reasoning_effort="none",
    )
    provider._prepare_conversation(task, ())
    request = {
        "model": "deepseek-v4-pro",
        "instructions": system_prompt,
        "input": provider._responses_input,
        "tools": provider._responses_tools,
        "parallel_tool_calls": False,
        "store": False,
        "reasoning": {"effort": "none"},
        "tool_choice": {"type": "function", "name": "submit_change_plan"},
    }

    _apply_response_token_budget(
        request,
        field_name="max_output_tokens",
        total_tokens=runtime.MAX_PLANNING_TOKENS,
    )

    output_tokens = request["max_output_tokens"]
    assert type(output_tokens) is int
    assert output_tokens >= 256
    assert len(_encode_request_payload(request)) <= 3_300
    assert (
        _estimate_request_input_tokens(request) + output_tokens
        <= runtime.MAX_PLANNING_TOKENS
    )


def test_typical_review_patch_keeps_complete_evidence_within_budget(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    record = record.model_copy(
        update={
            "task": "Add a stable formatter while preserving legacy output.",
            "plan": ChangePlan(
                goal="Add deterministic machine-readable output.",
                files=("package/formatter.py", "tests/test_formatter_regression.py"),
                steps=("Implement the exporter.",),
                checks=("pytest",),
                risks=("Do not alter the text renderer.",),
            ),
            "checks": (
                CheckSummary(
                    attempt=0,
                    check_id="python-pytest",
                    status="passed",
                    ok=True,
                    duration_ms=123,
                    log_artifact="checks/verify-0.log",
                ),
            ),
        }
    )
    production_patch = (
        "diff --git a/package/formatter.py b/package/formatter.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/package/formatter.py\n"
        "@@ -0,0 +1,48 @@\n"
        + ("+return value\n" * 48)
    )
    regression_patch = (
        "diff --git a/tests/test_formatter_regression.py "
        "b/tests/test_formatter_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_formatter_regression.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_regression_marker():\n"
        "+    assert True\n"
    )
    complete_patch = production_patch + regression_patch
    assert 1_000 <= len(complete_patch.encode("utf-8")) <= 1_500
    artifacts.write_patch(record.run_id, complete_patch)
    decision = ToolCall(
        "review",
        "submit_review",
        {"approved": True, "summary": "ok", "risks": []},
    )
    captured_tasks: list[str] = []
    captured_system_prompts: list[str] = []

    class CapturingProvider(_DecisionProvider):
        def next_step(self, task, results):
            captured_tasks.append(task)
            return super().next_step(task, results)

    provider = CapturingProvider(decision)

    def provider_factory(_definitions, system_prompt, _idempotency_key):
        captured_system_prompts.append(system_prompt)
        return provider

    operations._provider_factory = provider_factory
    result = operations.review(_context(record, "review", suffix="compact-prompt"))

    assert result.ok is True
    assert len(captured_tasks) == 1
    task = captured_tasks[0]
    assert task.startswith('G:"Add deterministic machine-readable output."\nD:\n')
    assert "\nD:\n" in task
    assert "Add a stable formatter" not in task
    assert "\nP:" not in task
    assert "\nC:" not in task
    assert "Implement the exporter." not in task
    assert "Do not alter the text renderer." not in task
    assert '"checks"' not in task
    assert '"goal"' not in task
    assert '"files"' not in task
    assert "return value" in task
    assert "tests/test_formatter_regression.py" in task
    assert "test_regression_marker" in task
    patch = task.split("\nD:\n", maxsplit=1)[1]
    assert 700 <= len(patch.encode("utf-8")) <= 1_000
    assert "diff --git" not in patch
    assert "--- /dev/null" not in patch

    assert captured_system_prompts
    assert "untrusted" in captured_system_prompts[0]
    provider_for_estimate = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-provider-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        ),
        tool_definitions=(runtime.REVIEW_TOOL,),
        system_prompt=captured_system_prompts[0],
        reasoning_effort="none",
    )
    provider_for_estimate._prepare_conversation(task, ())
    request = {
        "model": "deepseek-v4-pro",
        "instructions": captured_system_prompts[0],
        "input": provider_for_estimate._responses_input,
        "tools": provider_for_estimate._responses_tools,
        "parallel_tool_calls": False,
        "store": False,
        "reasoning": {"effort": "none"},
        "tool_choice": {"type": "function", "name": "submit_review"},
    }
    _apply_response_token_budget(
        request,
        field_name="max_output_tokens",
        total_tokens=runtime.MAX_REVIEW_TOKENS,
        minimum_output_tokens=runtime.MIN_REVIEW_OUTPUT_TOKENS,
    )
    output_tokens = request["max_output_tokens"]
    assert type(output_tokens) is int
    assert output_tokens >= runtime.MIN_REVIEW_OUTPUT_TOKENS
    assert len(_encode_request_payload(request)) <= 2_000
    assert (
        _estimate_request_input_tokens(request) + output_tokens
        <= runtime.MAX_REVIEW_TOKENS
    )

    schema = runtime.REVIEW_TOOL.input_schema["properties"]
    summary_limit = schema["summary"]["maxLength"]
    risks_schema = schema["risks"]
    risk_limit = risks_schema["items"]["maxLength"]
    assert risks_schema["maxItems"] == 1
    largest_verdict = runtime._json(
        {
            "type": "function_call",
            "call_id": "r" * 64,
            "name": "submit_review",
            "arguments": runtime._json(
                {
                    "approved": False,
                    "summary": "s" * summary_limit,
                    "risks": ["r" * risk_limit],
                }
            ),
        }
    )
    assert (
        runtime._bytes_to_estimated_tokens(len(largest_verdict.encode("utf-8")))
        <= runtime.MIN_REVIEW_OUTPUT_TOKENS
    )


def test_review_uses_bounded_full_payload_estimate_without_dropping_evidence(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1,180 @@\n"
        "-old = True\n"
        + "".join(f"+value_{index} = {index}\n" for index in range(180))
        + "diff --git a/tests/test_app_regression.py "
        "b/tests/test_app_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_app_regression.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_values_regression():\n"
        "+    assert value_179 == 179\n"
    )
    assert 2_500 < len(patch.encode("utf-8")) < 4_000
    artifacts.write_patch(record.run_id, patch)
    input_estimates: list[int] = []

    class ReviewProvider(_DecisionProvider):
        def prepare_next_request_payload_bytes(self, task, results):
            assert results == ()
            assert "value_179" in task
            assert "test_values_regression" in task
            return 4_500

        def set_next_response_input_token_estimate(self, value):
            input_estimates.append(value)

    provider = ReviewProvider(
        ToolCall(
            "review",
            "submit_review",
            {"approved": True, "summary": "complete", "risks": []},
        )
    )
    operations._provider_factory = lambda *_args: provider

    result = operations.review(
        _context(record, "review", suffix="bounded-full-payload")
    )

    assert result.ok is True
    assert input_estimates == [
        runtime._bytes_to_estimated_tokens(4_500)
        + runtime.REVIEW_INPUT_TOKEN_MARGIN
    ]
    assert input_estimates[0] + runtime.MIN_REVIEW_OUTPUT_TOKENS < (
        runtime.MAX_REVIEW_TOKENS
    )


def test_review_patch_evidence_keeps_all_changed_lines_and_regression_body() -> None:
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,3 +1,3 @@\n"
        " unchanged_before\n"
        "-old_value = 1\n"
        "+new_value = 2\n"
        " unchanged_after\n"
        "diff --git a/tests/test_app_regression.py "
        "b/tests/test_app_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_app_regression.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_new_value_regression():\n"
        "+    assert new_value == 2\n"
    )

    evidence = runtime._review_patch_evidence(patch)

    assert "M:app.py" in evidence
    assert "-old_value = 1" in evidence
    assert "+new_value = 2" in evidence
    assert "unchanged_before" not in evidence
    assert "unchanged_after" not in evidence
    assert "A:tests/test_app_regression.py" in evidence
    assert "+def test_new_value_regression():" in evidence
    assert "+    assert new_value == 2" in evidence
    assert "diff --git" not in evidence


def test_review_goal_preserves_approved_internal_whitespace() -> None:
    goal = "Preserve  two spaces\nand\ttabs"
    plan = ChangePlan(
        goal=goal,
        files=("app.py", "tests/test_app_regression.py"),
        steps=("Implement the approved contract.",),
        checks=("pytest",),
        risks=(),
    )

    assert runtime._review_goal_summary(plan, "fallback task") == goal
    assert runtime._json(goal) == '"Preserve  two spaces\\nand\\ttabs"'


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
    assert summary["request_not_started"] is True
    assert summary["model_usage"]["response_count"] == 0
    assert summary["model_usage"]["total_tokens"] is None
    synchronized = runtime._synchronize_model_metrics(
        record, database.events(record.run_id)
    )
    assert synchronized.metrics.tokens == 0


def test_review_fallback_capacity_failure_is_not_a_zero_request_preflight(
    operations_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
    requests: list[str] = []
    providers: list[OpenAIProvider] = []

    apply_budget = provider_module._apply_response_token_budget

    def fail_chat_preflight(payload, *, field_name, **kwargs):
        if field_name == "max_completion_tokens":
            raise runtime.OpenAIRequestCapacityError(
                "Chat fallback cannot fit the remaining review budget"
            )
        return apply_budget(payload, field_name=field_name, **kwargs)

    monkeypatch.setattr(
        provider_module,
        "_apply_response_token_budget",
        fail_chat_preflight,
    )

    def provider_factory(definitions, system_prompt, idempotency_key):
        provider = OpenAIProvider(
            OpenAIConfig(
                api_key="fake-provider-key",
                base_url="http://127.0.0.1:9",
                model="fake-model",
            ),
            tool_definitions=definitions,
            system_prompt=system_prompt,
            idempotency_key=idempotency_key,
        )

        def post_json(endpoint, _payload, *, idempotency_key=None):
            del idempotency_key
            requests.append(endpoint)
            if endpoint == "/responses":
                raise OpenAIHTTPError(404, endpoint_unsupported=True)
            raise AssertionError("Chat fallback must fail capacity before HTTP")

        monkeypatch.setattr(provider._client, "post_json", post_json)
        providers.append(provider)
        return provider

    operations._provider_factory = provider_factory

    result = operations.review(_context(record, "review", suffix="fallback-capacity"))

    assert result.ok is False
    assert result.detail["failure_code"] == "review_capacity_exceeded"
    assert requests == ["/responses"]
    assert providers[0].request_count == 1
    summary = _only_model_summary(database, record.run_id)
    assert summary["request_not_started"] is False
    assert summary["model_usage"]["response_count"] == 1
    assert summary["model_usage"]["reported_response_count"] == 0
    assert summary["model_usage"]["complete"] is False


def test_review_bounded_estimate_allows_long_approved_goal_within_capacity(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    goal = "Keep the approved behavior " + ("g" * 1900)
    record = record.model_copy(
        update={
            "plan": ChangePlan(
                goal=goal,
                files=("app.py", "tests/test_app_regression.py"),
                steps=("Preserve existing behavior.",),
                checks=("pytest",),
                risks=("Avoid unrelated changes.",),
            )
        }
    )
    artifacts.write_patch(record.run_id, _SMALL_REVIEW_PATCH)
    http_calls: list[tuple[object, ...]] = []

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

        def approve_review(*args, **kwargs):
            http_calls.append((args, kwargs))
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "review-long-goal",
                        "name": "submit_review",
                        "arguments": runtime._json(
                            {"approved": True, "summary": "ok", "risks": []}
                        ),
                    }
                ],
                "usage": {
                    "input_tokens": 1_800,
                    "output_tokens": 20,
                    "total_tokens": 1_820,
                },
            }

        monkeypatch.setattr(provider._client, "post_json", approve_review)
        return provider

    operations._provider_factory = provider_factory

    result = operations.review(_context(record, "review", suffix="long-goal"))

    assert runtime._review_goal_summary(record.plan, "fallback task") == goal
    assert result.ok is True
    assert result.summary == "ok"
    assert result.detail["tokens"] == 1_820
    assert len(http_calls) == 1


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


def test_request_capacity_boundary_preserves_verifiable_candidate(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_capacity_regression.py "
        "b/tests/test_capacity_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_capacity_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_capacity_regression.py",), patch
    )
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.OpenAIRequestCapacityError("request cannot fit response budget")
        ),
    )

    context = _context(record, "implement", suffix="capacity-candidate")
    outcome = operations.implement(context)

    assert outcome.ok is True
    assert outcome.summary == (
        "change budget reached; persisted candidate sent to verification"
    )
    assert outcome.detail["soft_budget_boundary"] is True
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch
    summary = _only_model_summary(database, record.run_id)
    assert summary["executed_tool_calls"] == 0
    assert summary["request_not_started"] is True
    assert summary["model_usage"]["response_count"] == 0


def test_repair_follow_up_capacity_failure_is_not_marked_request_not_started(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_capacity_regression.py "
        "b/tests/test_capacity_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_capacity_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_capacity_regression.py",), patch
    )
    artifacts.write_patch(record.run_id, patch)
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    provider = _DecisionProvider(
        FinalAnswer("done"),
        usage=SimpleNamespace(
            response_count=1,
            reported_response_count=1,
            input_tokens=99,
            cached_input_tokens=0,
            output_tokens=1,
            total_tokens=100,
            complete=True,
        ),
    )
    operations._provider_factory = lambda *_args: provider
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.OpenAIRequestCapacityError("repair follow-up cannot fit")
        ),
    )

    with pytest.raises(
        runtime.OpenAIRequestCapacityError, match="repair follow-up cannot fit"
    ):
        operations.repair(
            _context(record, "repair", suffix="follow-up-capacity")
        )

    summary = _only_model_summary(database, record.run_id)
    assert summary["request_not_started"] is False
    assert summary["model_usage"]["response_count"] == 1


def test_repair_shrinks_optional_log_after_pre_http_capacity_failure(
    operations_bundle, monkeypatch
) -> None:
    operations, _database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    test_path = "tests/test_capacity_regression.py"
    prior_patch = (
        f"diff --git a/{test_path} b/{test_path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{test_path}\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert False\n"
    )
    corrected_patch = prior_patch.replace("assert False", "assert True")
    sandbox = _CandidateSandbox(
        CandidateArtifact("a" * 40, 1, (test_path,), prior_patch)
    )
    artifacts.write_patch(record.run_id, prior_patch)
    operations._sandbox = lambda *_args, **_kwargs: sandbox
    operations._tool_executor = lambda *_args, **_kwargs: _Executor()
    provider = _DecisionProvider(FinalAnswer("unused"))
    operations._provider_factory = lambda *_args: provider
    monkeypatch.setattr(
        operations,
        "_repair_log_tail",
        lambda *_args, **_kwargs: "x" * runtime.MAX_REPAIR_LOG_BYTES,
    )
    prompts: list[str] = []

    def capacity_then_success(_provider, prompt, *_args, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise runtime.OpenAIRequestCapacityError("initial repair request is too large")
        sandbox.candidate = CandidateArtifact(
            "a" * 40, 2, (test_path,), corrected_patch
        )
        return runtime.ModelLoopOutcome(
            True,
            "candidate submitted for independent verification",
            (),
            (),
        )

    monkeypatch.setattr(runtime, "_run_model_loop", capacity_then_success)

    outcome = operations.repair(
        _context(record, "repair", suffix="shrink-pre-http-log")
    )

    assert outcome.ok is True
    assert len(prompts) == 2
    assert len(prompts[0].encode("utf-8")) - len(
        prompts[1].encode("utf-8")
    ) == runtime.MAX_REPAIR_LOG_BYTES // 2
    assert provider.request_count == 0
    assert provider.response_token_budgets == [
        runtime.MAX_CHANGE_TOKENS,
        runtime.MAX_CHANGE_TOKENS,
    ]
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == (
        corrected_patch
    )


def test_request_capacity_boundary_without_verifiable_candidate_fails_explicitly(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(None)
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.OpenAIRequestCapacityError("request cannot fit response budget")
        ),
    )

    context = _context(record, "implement", suffix="capacity-no-candidate")
    with pytest.raises(RuntimeError) as captured:
        operations.implement(context)

    assert str(captured.value) == (
        "request cannot fit response budget; no verifiable candidate was available"
    )
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == ""
    summary = _only_model_summary(database, record.run_id)
    assert summary["executed_tool_calls"] == 0


def test_repair_capacity_before_mutation_preserves_typed_failure(
    operations_bundle, monkeypatch
) -> None:
    operations, database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    patch = (
        "diff --git a/tests/test_capacity_regression.py "
        "b/tests/test_capacity_regression.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_capacity_regression.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_regression(): assert True\n"
    )
    candidate = CandidateArtifact(
        "a" * 40, 1, ("tests/test_capacity_regression.py",), patch
    )
    artifacts.write_patch(record.run_id, patch)
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(candidate)
    operations._provider_factory = lambda *_args: _DecisionProvider(FinalAnswer("done"))
    monkeypatch.setattr(
        runtime,
        "_run_model_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime.OpenAIRequestCapacityError("repair follow-up cannot fit")
        ),
    )

    with pytest.raises(
        runtime.OpenAIRequestCapacityError, match="repair follow-up cannot fit"
    ):
        operations.repair(
            _context(record, "repair", suffix="capacity-before-mutation")
        )

    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == patch
    summary = _only_model_summary(database, record.run_id)
    assert summary["executed_tool_calls"] == 0


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


def test_repair_log_tail_honors_zero_and_tight_byte_limits(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    artifacts.write_check(
        record.run_id,
        "verify-tight.log",
        "prefix\n" + ("x" * 300) + "\napi_key=plain-secret\nTAIL\n",
    )
    failed = CheckSummary(
        attempt=0,
        check_id="python-pytest",
        status="failed",
        ok=False,
        duration_ms=1,
        log_artifact="checks/verify-tight.log",
    )
    context = _context(
        record.model_copy(update={"checks": (failed,)}),
        "repair",
        suffix="tight-log",
    )

    assert operations._repair_log_tail(context, max_bytes=0) == ""
    tail = operations._repair_log_tail(context, max_bytes=32)
    assert len(tail.encode("utf-8")) <= 32
    assert "plain-secret" not in tail
    assert "TAIL" in tail

    with pytest.raises(ValueError, match="byte limit"):
        operations._repair_log_tail(context, max_bytes=-1)


def test_repair_log_budget_shrinks_with_phase_and_prompt_pressure() -> None:
    full = runtime._repair_log_budget_bytes(
        prior_tokens=0,
        phase_prior_tokens=0,
        system_prompt="system",
        prompt_without_repair_log="prompt",
    )
    tight = runtime._repair_log_budget_bytes(
        prior_tokens=22_500,
        phase_prior_tokens=19_500,
        system_prompt="system",
        prompt_without_repair_log="prompt",
    )
    long_prompt = runtime._repair_log_budget_bytes(
        prior_tokens=22_500,
        phase_prior_tokens=19_500,
        system_prompt="system" * 500,
        prompt_without_repair_log="prompt" * 500,
    )

    assert full == runtime.MAX_REPAIR_LOG_BYTES
    assert 0 < tight < full
    assert long_prompt < tight
    assert runtime._repair_log_budget_bytes(
        prior_tokens=None,
        phase_prior_tokens=0,
        system_prompt="system",
        prompt_without_repair_log="prompt",
    ) == runtime.MAX_REPAIR_LOG_BYTES


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


def test_repair_can_change_only_generated_regression_test_and_advance_to_verification(
    operations_bundle,
) -> None:
    operations, _database, artifacts, record = operations_bundle
    repository = Path(record.repo_path)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n",
        encoding="utf-8",
    )
    test_path = "tests/test_widget_regression.py"
    incorrect_test = (
        "from package.widget import normalize\n\n\n"
        "def test_normalize_preserves_public_contract():\n"
        "    assert normalize(['first', 'second']) == ['wrong']\n"
    )
    corrected_test = incorrect_test.replace("['wrong']", "['first', 'second']")

    def added_test_patch(content: str) -> str:
        lines = content.rstrip("\n").splitlines()
        additions = "\n".join(f"+{line}" for line in lines)
        return (
            f"diff --git a/{test_path} b/{test_path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{test_path}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n"
            f"{additions}\n"
        )

    prior_patch = added_test_patch(incorrect_test)
    corrected_patch = added_test_patch(corrected_test)
    prior_candidate = CandidateArtifact(
        "a" * 40,
        1,
        (test_path,),
        prior_patch,
    )
    corrected_candidate = CandidateArtifact(
        "a" * 40,
        2,
        (test_path,),
        corrected_patch,
    )
    artifacts.write_patch(record.run_id, prior_patch)

    class MutableRepairSandbox(_CandidateSandbox):
        def __init__(self) -> None:
            super().__init__(None)
            self.restore_calls: list[ToolCall] = []

        def execute(self, call: ToolCall) -> ToolResult:
            self.restore_calls.append(call)
            assert call.name == "apply_patch"
            assert call.arguments == {"patch": prior_patch}
            self.candidate = prior_candidate
            return ToolResult(call.id, call.name, True, "restored")

    sandbox = MutableRepairSandbox()

    class RepairExecutor:
        workspace_revision = 1
        mutation_persistence_failed = False

        def __init__(self) -> None:
            self.calls: list[ToolCall] = []

        def execute(self, call: ToolCall) -> ToolResult:
            self.calls.append(call)
            if call.name == "read_files":
                assert call.arguments == {"paths": [test_path]}
                return ToolResult(call.id, call.name, True, incorrect_test)
            assert call.name == "write_files"
            assert call.arguments == {
                "files": [{"path": test_path, "content": corrected_test}]
            }
            sandbox.candidate = corrected_candidate
            self.workspace_revision += 1
            return ToolResult(call.id, call.name, True, "updated")

    executor = RepairExecutor()
    provider = _DecisionProvider(
        ToolCall("read", "read_files", {"paths": [test_path]}),
        ToolCall(
            "write",
            "write_files",
            {"files": [{"path": test_path, "content": corrected_test}]},
        ),
    )
    operations._sandbox = lambda *_args, **_kwargs: sandbox
    operations._tool_executor = lambda *_args, **_kwargs: executor
    operations._provider_factory = lambda *_args: provider

    outcome = operations.repair(
        _context(record, "repair", suffix="test-only-correction")
    )

    assert outcome.ok is True
    assert outcome.summary == "candidate submitted for independent verification"
    assert [call.name for call in sandbox.restore_calls] == ["apply_patch"]
    assert [call.name for call in executor.calls] == ["read_files", "write_files"]
    assert provider.forced_tool_choices == ["read_files", "write_files"]
    assert provider.calls == 2
    assert provider.decisions == []
    assert corrected_patch != prior_patch
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == (
        corrected_patch
    )


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


def test_operations_injects_cancellation_check_into_provider(
    operations_bundle,
) -> None:
    operations, _database, _artifacts, _record = operations_bundle
    checks = []
    provider = SimpleNamespace(
        set_cancellation_check=lambda check: checks.append(check)
    )
    operations._provider_factory = lambda *_args: provider

    result = operations._provider((runtime.PLAN_TOOL,), "system", "key")

    assert result is provider
    assert len(checks) == 1
    operations.cancel_event.set()
    with pytest.raises(runtime.WorkflowCancelled, match="run was cancelled"):
        checks[0]()


def test_service_runner_cancel_delegates_only_for_active_run(operations_bundle) -> None:
    _operations, database, artifacts, _record_value = operations_bundle
    runner = runtime.ServiceWorkflowRunner(database, artifacts)
    calls: list[str] = []
    runner._active[RUN_ID] = SimpleNamespace(cancel=lambda: calls.append(RUN_ID))

    runner.cancel("d" * 32)
    runner.cancel(RUN_ID)

    assert calls == [RUN_ID]


def test_service_runner_close_budget_covers_one_provider_request(
    operations_bundle,
) -> None:
    _operations, database, artifacts, _record_value = operations_bundle
    runner = runtime.ServiceWorkflowRunner(database, artifacts)

    assert runner.close_timeout_seconds == (
        runtime.DURABLE_DEEPSEEK_REQUEST_TIMEOUT_SECONDS + 5.0
    )


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
