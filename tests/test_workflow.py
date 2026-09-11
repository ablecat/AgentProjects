from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from repo_agent.checkpoints import SQLiteCheckpointStore
from repo_agent.run_models import ChangePlan
from repo_agent.workflow import (
    ConcurrentUpdateError,
    DuplicateRunError,
    InvalidTransitionError,
    NodeOutcome,
    OperationContext,
    RunNotFoundError,
    VerificationOutcome,
    WorkflowEngine,
)


RUN_ID = "1" * 32
BASE_COMMIT = "a" * 40
PLAN = ChangePlan(
    goal="Fix the failing greeting behavior",
    files=("src/greeting.py", "tests/test_greeting.py"),
    steps=("Correct greeting punctuation", "Add a regression test"),
    checks=("python -m pytest",),
    risks=("Callers may compare exact output",),
)


class RecordingOperations:
    def __init__(
        self,
        *,
        verify: list[bool] | None = None,
        review: list[bool] | None = None,
        fail_node: str | None = None,
    ) -> None:
        self.verify_results = list(verify or [True])
        self.review_results = list(review or [True])
        self.fail_node = fail_node
        self.calls: list[tuple[str, int, str]] = []
        self.keys_by_node: dict[str, list[str]] = defaultdict(list)

    def _record(self, context: OperationContext) -> None:
        self.calls.append((context.node, context.attempt, context.idempotency_key))
        self.keys_by_node[context.node].append(context.idempotency_key)
        if context.node == self.fail_node:
            raise RuntimeError(f"{context.node} exploded")

    def prepare(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(
            summary="workspace ready", detail={"base_commit": BASE_COMMIT}
        )

    def baseline_check(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(summary="baseline captured")

    def inspect_and_plan(self, context: OperationContext) -> ChangePlan:
        self._record(context)
        return PLAN

    def implement(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(
            summary="patch applied",
            detail={"revision": 1, "tool_calls": 2, "tokens": 100},
        )

    def verify(self, context: OperationContext) -> VerificationOutcome:
        self._record(context)
        ok = self.verify_results.pop(0)
        return VerificationOutcome(
            ok=ok,
            status="passed" if ok else "failed",
            check_id="python-pytest",
            duration_ms=12,
            error=None if ok else f"attempt {context.attempt} failed",
        )

    def repair(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(
            summary=f"repair {context.attempt} applied",
            detail={"tool_calls": 1, "tokens": 20},
        )

    def review(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        ok = self.review_results.pop(0)
        return NodeOutcome(
            ok=ok,
            summary="review accepted" if ok else "review found an issue",
            error=None if ok else "review found an issue",
            detail={"tool_calls": 1, "tokens": 30},
        )

    def review_repair(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(summary="review feedback applied")

    def finalize(self, context: OperationContext) -> NodeOutcome:
        self._record(context)
        return NodeOutcome(summary="artifacts finalized")


def _names(operations: RecordingOperations) -> list[str]:
    return [node for node, _attempt, _key in operations.calls]


def test_happy_path_runs_in_order_and_checkpoints_every_node(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        result = engine.start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            base_ref="refs/heads/main",
            auto_approve=True,
        )

        assert result.status == "succeeded"
        assert result.current_node is None
        assert result.base_ref == "refs/heads/main"
        assert result.base_commit == BASE_COMMIT
        assert result.summary == "artifacts finalized"
        assert result.plan == PLAN
        assert result.metrics.node_count == 8
        assert result.metrics.repair_attempts == 0
        assert result.metrics.review_repairs == 0
        assert result.metrics.tool_calls == 3
        assert result.metrics.tokens == 130
        assert len(result.checks) == 1
        assert _names(operations) == [
            "prepare",
            "baseline_check",
            "inspect_and_plan",
            "implement",
            "verify",
            "review",
            "finalize",
        ]

        graph = engine.graph.get_graph()
        assert set(graph.nodes) == {
            "__start__",
            "dispatch",
            "prepare",
            "baseline_check",
            "inspect_and_plan",
            "approval",
            "implement",
            "verify",
            "repair",
            "review",
            "review_repair",
            "finalize",
            "__end__",
        }
        edges = {(edge.source, edge.target) for edge in graph.edges}
        assert ("__start__", "dispatch") in edges
        assert ("dispatch", "prepare") in edges
        assert ("dispatch", "__end__") in edges
        completed = [
            checkpoint.node
            for checkpoint in store.list_checkpoints(RUN_ID)
            if checkpoint.phase == "completed"
        ]
        assert completed == [
            "prepare",
            "baseline_check",
            "inspect_and_plan",
            "approval",
            "implement",
            "verify",
            "review",
            "finalize",
        ]
        assert all((node, "dispatch") in edges for node in completed)


def test_approval_pauses_without_implementing(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        result = engine.start(
            repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID
        )

        assert result.status == "awaiting_approval"
        assert result.current_node == "approval"
        assert result.plan == PLAN
        assert _names(operations) == [
            "prepare",
            "baseline_check",
            "inspect_and_plan",
        ]
        approval = [
            checkpoint
            for checkpoint in store.list_checkpoints(RUN_ID)
            if checkpoint.node == "approval"
        ]
        assert [checkpoint.phase for checkpoint in approval] == ["started", "paused"]
        assert engine.resume(RUN_ID) == result
        assert "implement" not in _names(operations)


def test_approved_run_resumes_in_new_process_without_repeating_completed_nodes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    first_operations = RecordingOperations()
    with SQLiteCheckpointStore(database) as first_store:
        paused = WorkflowEngine(first_store, first_operations).start(
            repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID
        )
        assert paused.status == "awaiting_approval"

    second_operations = RecordingOperations()
    with SQLiteCheckpointStore(database) as second_store:
        engine = WorkflowEngine(second_store, second_operations)
        approved = engine.approve(
            RUN_ID, True, reason="Plan is scoped", resume=False
        )
        assert approved.status == "running"
        assert approved.current_node == "implement"
        result = engine.resume(RUN_ID)

        assert result.status == "succeeded"
        assert result.approval_reason == "Plan is scoped"
        assert _names(second_operations) == [
            "implement",
            "verify",
            "review",
            "finalize",
        ]
        assert set(_names(first_operations)).isdisjoint(_names(second_operations))


class SimulatedProcessCrash(BaseException):
    pass


class CrashAfterCommittedNodeStore:
    """Raise after SQLite commits one node to model abrupt process death."""

    def __init__(self, delegate: SQLiteCheckpointStore, node: str) -> None:
        self.delegate = delegate
        self.node = node
        self.crashed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def save_checkpoint(self, snapshot, **kwargs):
        saved = self.delegate.save_checkpoint(snapshot, **kwargs)
        if (
            not self.crashed
            and kwargs["node"] == self.node
            and kwargs["phase"] == "completed"
        ):
            self.crashed = True
            raise SimulatedProcessCrash()
        return saved


@pytest.mark.parametrize(
    ("crash_node", "expected_resumed_nodes"),
    [
        ("inspect_and_plan", ["implement", "verify", "review", "finalize"]),
        ("implement", ["verify", "review", "finalize"]),
        ("verify", ["review", "finalize"]),
    ],
)
def test_resume_after_commit_then_process_crash_skips_side_effect(
    tmp_path: Path,
    crash_node: str,
    expected_resumed_nodes: list[str],
) -> None:
    database = tmp_path / "runs.sqlite3"
    first_operations = RecordingOperations()
    with SQLiteCheckpointStore(database) as store:
        crashing_store = CrashAfterCommittedNodeStore(store, crash_node)
        with pytest.raises(SimulatedProcessCrash):
            WorkflowEngine(crashing_store, first_operations).start(
                repo_path=str(tmp_path),
                task="Fix greeting",
                run_id=RUN_ID,
                auto_approve=True,
            )
    assert _names(first_operations).count(crash_node) == 1

    resumed_operations = RecordingOperations()
    with SQLiteCheckpointStore(database) as store:
        result = WorkflowEngine(store, resumed_operations).resume(RUN_ID)

        assert result.status == "succeeded"
        assert _names(resumed_operations) == expected_resumed_nodes
        assert crash_node not in _names(resumed_operations)


def test_verify_can_repair_twice_then_succeed(tmp_path: Path) -> None:
    operations = RecordingOperations(verify=[False, False, True])
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "succeeded"
    assert result.metrics.repair_attempts == 2
    assert result.metrics.tool_calls == 5
    assert result.metrics.tokens == 170
    assert [check.attempt for check in result.checks] == [0, 1, 2]
    assert [check.ok for check in result.checks] == [False, False, True]
    assert _names(operations)[3:] == [
        "implement",
        "verify",
        "repair",
        "verify",
        "repair",
        "verify",
        "review",
        "finalize",
    ]
    assert operations.keys_by_node["repair"] == [
        f"{RUN_ID}:repair:1",
        f"{RUN_ID}:repair:2",
    ]


def test_verify_stops_unverified_after_two_repairs(tmp_path: Path) -> None:
    operations = RecordingOperations(verify=[False, False, False])
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "unverified"
    assert result.metrics.repair_attempts == 2
    assert len(result.checks) == 3
    assert "review" not in _names(operations)
    assert "finalize" not in _names(operations)


def test_review_repair_runs_once_then_reverifies_and_reviews(tmp_path: Path) -> None:
    operations = RecordingOperations(verify=[True, True], review=[False, True])
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "succeeded"
    assert result.metrics.review_repairs == 1
    assert _names(operations)[3:] == [
        "implement",
        "verify",
        "review",
        "review_repair",
        "verify",
        "review",
        "finalize",
    ]
    assert operations.keys_by_node["review_repair"] == [
        f"{RUN_ID}:review_repair:1"
    ]


def test_second_review_rejection_fails_without_second_review_repair(
    tmp_path: Path,
) -> None:
    operations = RecordingOperations(verify=[True, True], review=[False, False])
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "failed"
    assert result.metrics.review_repairs == 1
    assert _names(operations).count("review_repair") == 1
    assert "finalize" not in _names(operations)


def test_operation_exception_is_durable_and_resume_is_idempotent(tmp_path: Path) -> None:
    operations = RecordingOperations(fail_node="implement")
    database = tmp_path / "runs.sqlite3"
    with SQLiteCheckpointStore(database) as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )
        assert result.status == "failed"
        assert "RuntimeError: implement exploded" in (result.error or "")
        assert store.list_checkpoints(RUN_ID)[-1].phase == "failed"

    fresh_operations = RecordingOperations()
    with SQLiteCheckpointStore(database) as store:
        resumed = WorkflowEngine(store, fresh_operations).resume(RUN_ID)
        assert resumed == result
        assert fresh_operations.calls == []


def test_invalid_prepare_commit_fails_before_baseline(tmp_path: Path) -> None:
    class InvalidCommitOperations(RecordingOperations):
        def prepare(self, context: OperationContext) -> NodeOutcome:
            self._record(context)
            return NodeOutcome(detail={"base_commit": "HEAD"})

    operations = InvalidCommitOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "failed"
    assert "prepare base_commit" in (result.error or "")
    assert _names(operations) == ["prepare"]


def test_prepare_policy_failure_preserves_public_policy_status(tmp_path: Path) -> None:
    class DeniedOperations(RecordingOperations):
        def prepare(self, context: OperationContext) -> NodeOutcome:
            self._record(context)
            return NodeOutcome(
                ok=False,
                error="repository worktree must be clean",
                detail={"failure_status": "policy_denied"},
            )

    operations = DeniedOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "policy_denied"
    assert result.error == "repository worktree must be clean"
    assert _names(operations) == ["prepare"]


def test_usage_counters_are_bounded_by_public_metrics(tmp_path: Path) -> None:
    class ExcessiveUsageOperations(RecordingOperations):
        def implement(self, context: OperationContext) -> NodeOutcome:
            self._record(context)
            return NodeOutcome(detail={"tool_calls": 31, "tokens": 1})

    operations = ExcessiveUsageOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        result = WorkflowEngine(store, operations).start(
            repo_path=str(tmp_path),
            task="Fix greeting",
            run_id=RUN_ID,
            auto_approve=True,
        )

    assert result.status == "failed"
    assert "less than or equal to 30" in (result.error or "")
    assert "verify" not in _names(operations)


def test_rejection_is_terminal_and_never_implements(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        engine.start(repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID)
        result = engine.approve(RUN_ID, False, reason="Risk is too broad")

        assert result.status == "rejected"
        assert result.finished_at is not None
        assert result.approval_reason == "Risk is too broad"
        assert "implement" not in _names(operations)


def test_invalid_run_and_approval_transitions_are_rejected(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        with pytest.raises(RunNotFoundError):
            engine.resume("f" * 32)

        engine.start(repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID)
        with pytest.raises(DuplicateRunError):
            engine.start(repo_path=str(tmp_path), task="Again", run_id=RUN_ID)
        engine.approve(RUN_ID, True)
        with pytest.raises(InvalidTransitionError):
            engine.approve(RUN_ID, True)


def test_cancel_paused_run_is_idempotent(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        engine.start(repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID)

        cancelled = engine.cancel(RUN_ID, reason="User stopped the run")
        assert cancelled.status == "cancelled"
        assert cancelled.cancel_requested
        assert cancelled.error == "User stopped the run"
        assert engine.cancel(RUN_ID) == cancelled
        assert engine.resume(RUN_ID) == cancelled
        assert "implement" not in _names(operations)


def test_change_plan_round_trip_and_invalid_payload() -> None:
    assert ChangePlan.model_validate_json(PLAN.model_dump_json()) == PLAN
    with pytest.raises(ValueError, match="inside the repository"):
        ChangePlan(
            goal="Escape",
            files=("../secret.txt",),
            steps=("Read a secret",),
            checks=("pytest",),
        )


def test_stale_snapshot_rolls_back_checkpoint_transaction(tmp_path: Path) -> None:
    operations = RecordingOperations()
    with SQLiteCheckpointStore(tmp_path / "runs.sqlite3") as store:
        engine = WorkflowEngine(store, operations)
        engine.start(repo_path=str(tmp_path), task="Fix greeting", run_id=RUN_ID)
        stale = store.load_run(RUN_ID)
        before = store.list_checkpoints(RUN_ID)
        current = engine.cancel(RUN_ID)

        with pytest.raises(ConcurrentUpdateError):
            store.save_checkpoint(
                stale,
                node="approval",
                attempt=0,
                phase="completed",
                idempotency_key=f"{RUN_ID}:approval:0",
                result={"approved": True},
                record=current,
                next_node="implement",
                created_at=current.updated_at,
            )

        assert store.get_run(RUN_ID) == current
        assert store.list_checkpoints(RUN_ID) == (*before, store.list_checkpoints(RUN_ID)[-1])
