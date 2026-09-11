from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import repo_agent.workflow_runtime as runtime
from repo_agent.artifacts import ArtifactStore
from repo_agent.models import CandidateArtifact, FinalAnswer, ToolCall, ToolResult
from repo_agent.persistence import RunDatabase
from repo_agent.run_models import ChangePlan, RunRecord
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
    config = object()
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


def test_change_node_restores_cached_patch_artifact(operations_bundle) -> None:
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
    assert artifacts.path(record.run_id, "patch").read_text(encoding="utf-8") == (
        "cached diff\n"
    )


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

    def next_step(self, _task, _results):
        return self.decisions.pop(0)


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
            "tool_calls": 5,
            "model_usage": {"complete": True, "total_tokens": 100},
        },
    )
    assert operations._model_budget(_context(record, "implement")) == (25, 100)

    database.append_event(
        record.run_id,
        {"event": "model_tool_summary", "tool_calls": True, "model_usage": None},
    )
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "tool_calls": 2,
            "model_usage": {"complete": False, "total_tokens": -1},
        },
    )
    assert operations._model_budget(_context(record, "implement")) == (23, None)


def test_run_budget_stops_at_global_tool_and_token_limits(operations_bundle) -> None:
    operations, database, _artifacts, record = operations_bundle
    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
            "tool_calls": runtime.MAX_RUN_TOOL_CALLS,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
    )
    assert operations._model_budget(_context(record, "implement"))[0] == 0

    database.append_event(
        record.run_id,
        {
            "event": "model_tool_summary",
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
            "tool_calls": runtime.MAX_RUN_TOOL_CALLS + 1,
            "model_usage": {"complete": True, "total_tokens": 1},
        },
    )
    with pytest.raises(RuntimeError, match="repository tool budget exceeded"):
        operations._model_budget(_context(record, "implement"))


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


class _CandidateSandbox:
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
        operations.review_repair(
            _context(record, "review_repair", suffix="restore-failed")
        )


def test_repository_map_propagates_sandbox_error(operations_bundle) -> None:
    operations, _database, _artifacts, record = operations_bundle
    failed = ToolResult("map", "repo_map", False, "", error="map rejected")
    operations._sandbox = lambda *_args, **_kwargs: _CandidateSandbox(
        None, restore_result=failed
    )

    with pytest.raises(RuntimeError, match="map rejected"):
        operations._repository_map(_context(record, "inspect_and_plan"))


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
            SimpleNamespace(returncode=2, stdout=b"", stderr=b"git denied"),
            SimpleNamespace(returncode=3, stdout=b"", stderr=b""),
        ]
    )
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(RuntimeError, match="git denied"):
        runtime._git(tmp_path, "status")
    with pytest.raises(RuntimeError, match="Git exited 3"):
        runtime._git(tmp_path, "status")
