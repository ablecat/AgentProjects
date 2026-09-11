"""Durable, approval-gated workflow orchestration for repository changes.

The workflow deliberately knows nothing about Docker, model providers, or HTTP.
Those integrations implement :class:`WorkflowOperations`; this module supplies
the deterministic state machine and passes a stable idempotency key to every
operation that may have side effects.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import json
import math
import re
import time
from typing import Any, Callable, Literal, Protocol, TypeAlias, TypedDict, cast
import uuid

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .run_models import (
    TERMINAL_STATUSES,
    ChangePlan,
    CheckSummary,
    PublicRunStatus,
    RunMetrics,
    RunRecord,
    WorkflowNode,
    utc_now,
)


MAX_REPAIR_ATTEMPTS = 2
MAX_REVIEW_REPAIRS = 1
_MAX_RESULT_JSON_BYTES = 64 * 1024
_MAX_ENGINE_TRANSITIONS = 32
_GRAPH_END = "__workflow_end__"

_WORKFLOW_NODES: tuple[WorkflowNode, ...] = (
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
)

CheckpointPhase: TypeAlias = Literal[
    "started", "paused", "completed", "failed", "cancelled"
]


class WorkflowError(RuntimeError):
    """Base error for workflow state and persistence failures."""


class RunNotFoundError(WorkflowError):
    """The requested run ID does not exist."""


class DuplicateRunError(WorkflowError):
    """A new run reused an existing run ID."""


class InvalidTransitionError(WorkflowError):
    """A requested transition is not valid for the current run state."""


class ConcurrentUpdateError(WorkflowError):
    """Another worker changed the run after it was loaded."""


class RunBusyError(WorkflowError):
    """A workflow worker is already advancing the run."""


def _validate_json_detail(value: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("detail must contain only finite JSON values") from exc
    if len(payload) > _MAX_RESULT_JSON_BYTES:
        raise ValueError("detail exceeds 64 KiB")
    return value


class NodeOutcome(BaseModel):
    """Bounded result returned by a non-verification workflow operation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    ok: bool = True
    summary: str = Field(default="", max_length=4000)
    error: str | None = Field(default=None, max_length=4000)
    detail: dict[str, Any] = Field(default_factory=dict)

    _detail_is_json = field_validator("detail")(_validate_json_detail)


class VerificationOutcome(BaseModel):
    """Result of one deterministic verification attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    ok: bool
    status: str = Field(min_length=1, max_length=64)
    check_id: str | None = Field(default=None, max_length=128)
    duration_ms: int = Field(default=0, ge=0)
    log_artifact: str | None = Field(default=None, max_length=255)
    error: str | None = Field(default=None, max_length=4000)
    detail: dict[str, Any] = Field(default_factory=dict)

    _detail_is_json = field_validator("detail")(_validate_json_detail)

    def as_check_summary(self, attempt: int) -> CheckSummary:
        return CheckSummary(
            attempt=attempt,
            check_id=self.check_id,
            status=self.status,
            ok=self.ok,
            duration_ms=self.duration_ms,
            log_artifact=self.log_artifact,
            error=self.error,
        )


class WorkflowCheckpoint(BaseModel):
    """One immutable SQLite checkpoint around a node transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    node: WorkflowNode
    attempt: int = Field(ge=0)
    phase: CheckpointPhase
    idempotency_key: str = Field(min_length=1, max_length=128)
    result: dict[str, Any] = Field(default_factory=dict)
    record: RunRecord
    created_at: str

    _result_is_json = field_validator("result")(_validate_json_detail)


class WorkflowSnapshot(BaseModel):
    """Current public run record plus its private durable cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: RunRecord
    next_node: WorkflowNode | None
    version: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class OperationContext:
    """Input shared by adapters implementing workflow node operations.

    ``idempotency_key`` is stable for a run/node/attempt. Implementations must
    return the prior result rather than repeat a side effect when they receive a
    key they have already committed. This covers the narrow crash window between
    an external side effect and its completed SQLite checkpoint.
    """

    run: RunRecord
    node: WorkflowNode
    attempt: int
    idempotency_key: str
    previous_result: dict[str, Any] | None = None


class WorkflowOperations(Protocol):
    """Application-specific work invoked by :class:`WorkflowEngine`."""

    def prepare(self, context: OperationContext) -> NodeOutcome: ...

    def baseline_check(self, context: OperationContext) -> NodeOutcome: ...

    def inspect_and_plan(self, context: OperationContext) -> ChangePlan: ...

    def implement(self, context: OperationContext) -> NodeOutcome: ...

    def verify(self, context: OperationContext) -> VerificationOutcome: ...

    def repair(self, context: OperationContext) -> NodeOutcome: ...

    def review(self, context: OperationContext) -> NodeOutcome: ...

    def review_repair(self, context: OperationContext) -> NodeOutcome: ...

    def finalize(self, context: OperationContext) -> NodeOutcome: ...


class CheckpointStore(Protocol):
    """Persistence boundary used by the workflow engine."""

    def create_run(
        self, record: RunRecord, *, next_node: WorkflowNode
    ) -> WorkflowSnapshot: ...

    def load_run(self, run_id: str) -> WorkflowSnapshot: ...

    def get_run(self, run_id: str) -> RunRecord: ...

    def save_checkpoint(
        self,
        snapshot: WorkflowSnapshot,
        *,
        node: WorkflowNode,
        attempt: int,
        phase: CheckpointPhase,
        idempotency_key: str,
        result: dict[str, Any],
        record: RunRecord,
        next_node: WorkflowNode | None,
        created_at: str,
    ) -> WorkflowSnapshot: ...

    def get_checkpoint(
        self,
        run_id: str,
        *,
        node: WorkflowNode,
        attempt: int,
        phase: CheckpointPhase,
    ) -> WorkflowCheckpoint | None: ...

    def list_checkpoints(self, run_id: str) -> tuple[WorkflowCheckpoint, ...]: ...


Clock: TypeAlias = Callable[[], str]
MonotonicClock: TypeAlias = Callable[[], float]
GraphRoute: TypeAlias = WorkflowNode | Literal["__workflow_end__"]


class _WorkflowGraphState(TypedDict):
    run_id: str
    record: RunRecord
    route: GraphRoute


class WorkflowEngine:
    """Advance one durable repository-change workflow to a pause or terminal state."""

    def __init__(
        self,
        store: CheckpointStore,
        operations: WorkflowOperations,
        *,
        now: Clock = utc_now,
        monotonic: MonotonicClock = time.monotonic,
    ) -> None:
        self._store = store
        self._operations = operations
        self._now = now
        self._monotonic = monotonic
        self._active_runs: set[str] = set()
        self._graph = self._build_graph()

    @property
    def graph(self) -> Any:
        """Return the compiled LangGraph for inspection and integration tests."""

        return self._graph

    def start(
        self,
        *,
        repo_path: str,
        task: str,
        run_id: str | None = None,
        base_ref: str | None = None,
        auto_approve: bool = False,
        allow_remote_model: bool = False,
    ) -> RunRecord:
        """Create and advance a run, stopping at approval unless auto-approved."""

        if type(auto_approve) is not bool:
            raise ValueError("auto_approve must be a boolean")
        if type(allow_remote_model) is not bool:
            raise ValueError("allow_remote_model must be a boolean")
        created_at = self._now()
        record = RunRecord(
            run_id=run_id or uuid.uuid4().hex,
            repo_path=str(repo_path),
            task=task,
            base_ref=base_ref,
            status="queued",
            current_node="prepare",
            auto_approve=auto_approve,
            allow_remote_model=allow_remote_model,
            created_at=created_at,
            updated_at=created_at,
        )
        self._store.create_run(record, next_node="prepare")
        return self.resume(record.run_id)

    def get(self, run_id: str) -> RunRecord:
        return self._store.get_run(run_id)

    def _build_graph(self) -> Any:
        builder = StateGraph(_WorkflowGraphState)
        builder.add_node("dispatch", self._graph_dispatch)
        for node in _WORKFLOW_NODES:
            builder.add_node(node, partial(self._graph_execute_node, node=node))
        builder.add_edge(START, "dispatch")
        route_map: dict[Hashable, str] = {
            **{node: node for node in _WORKFLOW_NODES},
            _GRAPH_END: END,
        }
        builder.add_conditional_edges(
            "dispatch",
            _graph_route,
            route_map,
        )
        for node in _WORKFLOW_NODES:
            builder.add_edge(node, "dispatch")
        return builder.compile()

    def _graph_dispatch(self, state: _WorkflowGraphState) -> dict[str, Any]:
        snapshot = self._store.load_run(state["run_id"])
        record = snapshot.record
        if record.status in TERMINAL_STATUSES or record.status == "awaiting_approval":
            return {"record": record, "route": _GRAPH_END}
        if record.cancel_requested:
            cancelled = self._cancel_snapshot(snapshot)
            return {"record": cancelled.record, "route": _GRAPH_END}
        if snapshot.next_node is None:
            failed = self._fail_without_node(
                snapshot, "non-terminal workflow has no next node"
            )
            return {"record": failed.record, "route": _GRAPH_END}
        return {"record": record, "route": snapshot.next_node}

    def _graph_execute_node(
        self, state: _WorkflowGraphState, *, node: WorkflowNode
    ) -> dict[str, Any]:
        snapshot = self._store.load_run(state["run_id"])
        if snapshot.record.status in TERMINAL_STATUSES:
            return {"record": snapshot.record}
        if snapshot.next_node != node:
            # The dispatcher will select the durable cursor on its next pass.
            return {"record": snapshot.record}
        if node == "approval":
            advanced = self._pause_or_auto_approve(snapshot)
        else:
            advanced = self._execute_node(snapshot, node)
        return {"record": advanced.record}

    def resume(self, run_id: str) -> RunRecord:
        """Advance all ready nodes without repeating any completed node attempt."""

        if run_id in self._active_runs:
            raise RunBusyError(f"run {run_id} is already active in this worker")
        self._active_runs.add(run_id)
        try:
            initial = self._store.load_run(run_id)
            state = self._graph.invoke(
                {
                    "run_id": run_id,
                    "record": initial.record,
                    "route": _GRAPH_END,
                },
                config={"recursion_limit": _MAX_ENGINE_TRANSITIONS * 2 + 4},
            )
            return cast(RunRecord, state["record"])
        finally:
            self._active_runs.discard(run_id)

    def approve(
        self,
        run_id: str,
        approved: bool,
        *,
        reason: str | None = None,
        resume: bool = True,
    ) -> RunRecord:
        """Record the human decision and optionally continue an approved run."""

        if type(approved) is not bool:
            raise ValueError("approved must be a boolean")
        if type(resume) is not bool:
            raise ValueError("resume must be a boolean")
        snapshot = self._store.load_run(run_id)
        record = snapshot.record
        if record.status != "awaiting_approval" or snapshot.next_node != "approval":
            raise InvalidTransitionError(
                f"run {run_id} is not awaiting an approval decision"
            )
        snapshot = self._complete_approval(
            snapshot, approved=approved, reason=reason, automatic=False
        )
        if approved and resume:
            return self.resume(run_id)
        return snapshot.record

    def cancel(self, run_id: str, *, reason: str | None = None) -> RunRecord:
        """Persist a terminal cancellation for a queued or paused run.

        The engine is synchronous, so an adapter that needs mid-operation
        cancellation must also honor its own cancellation primitive. Calls made
        after a terminal state are idempotent and return that state unchanged.
        """

        snapshot = self._store.load_run(run_id)
        if snapshot.record.status in TERMINAL_STATUSES:
            return snapshot.record
        return self._cancel_snapshot(snapshot, reason=reason).record

    def _execute_node(
        self, snapshot: WorkflowSnapshot, node: WorkflowNode
    ) -> WorkflowSnapshot:
        attempt = _node_attempt(snapshot.record, node)
        idempotency_key = _idempotency_key(snapshot.record.run_id, node, attempt)
        completed = self._store.get_checkpoint(
            snapshot.record.run_id,
            node=node,
            attempt=attempt,
            phase="completed",
        )
        if completed is not None:
            # save_checkpoint advances the cursor in the same transaction as the
            # completed checkpoint. Reloading is therefore enough to skip work.
            return self._store.load_run(snapshot.record.run_id)

        started = self._store.get_checkpoint(
            snapshot.record.run_id,
            node=node,
            attempt=attempt,
            phase="started",
        )
        if started is None:
            started_record = self._record_for_started_node(snapshot.record, node)
            snapshot = self._store.save_checkpoint(
                snapshot,
                node=node,
                attempt=attempt,
                phase="started",
                idempotency_key=idempotency_key,
                result={},
                record=started_record,
                next_node=node,
                created_at=self._now(),
            )

        previous_result = _previous_result(
            self._store.list_checkpoints(snapshot.record.run_id), node, attempt
        )
        context = OperationContext(
            run=snapshot.record,
            node=node,
            attempt=attempt,
            idempotency_key=idempotency_key,
            previous_result=previous_result,
        )
        operation_started = self._monotonic()
        try:
            raw_result = self._invoke(node, context)
            duration_ms = _duration_ms(operation_started, self._monotonic())
            return self._complete_node(snapshot, node, attempt, raw_result, duration_ms)
        except Exception as exc:
            duration_ms = _duration_ms(operation_started, self._monotonic())
            return self._fail_node(snapshot, node, attempt, exc, duration_ms)

    def _invoke(
        self, node: WorkflowNode, context: OperationContext
    ) -> NodeOutcome | VerificationOutcome | ChangePlan:
        if node == "prepare":
            return NodeOutcome.model_validate(self._operations.prepare(context))
        if node == "baseline_check":
            return NodeOutcome.model_validate(self._operations.baseline_check(context))
        if node == "inspect_and_plan":
            return ChangePlan.model_validate(self._operations.inspect_and_plan(context))
        if node == "implement":
            return NodeOutcome.model_validate(self._operations.implement(context))
        if node == "verify":
            return VerificationOutcome.model_validate(self._operations.verify(context))
        if node == "repair":
            return NodeOutcome.model_validate(self._operations.repair(context))
        if node == "review":
            return NodeOutcome.model_validate(self._operations.review(context))
        if node == "review_repair":
            return NodeOutcome.model_validate(self._operations.review_repair(context))
        if node == "finalize":
            return NodeOutcome.model_validate(self._operations.finalize(context))
        raise InvalidTransitionError(f"node {node!r} is not executable")

    def _complete_node(
        self,
        snapshot: WorkflowSnapshot,
        node: WorkflowNode,
        attempt: int,
        raw_result: NodeOutcome | VerificationOutcome | ChangePlan,
        duration_ms: int,
    ) -> WorkflowSnapshot:
        latest = self._latest_compatible_snapshot(snapshot)
        record = latest.record
        metrics = record.metrics.model_copy(
            update={"node_count": record.metrics.node_count + 1}
        )
        if isinstance(raw_result, (NodeOutcome, VerificationOutcome)):
            metrics = _accumulate_usage(metrics, raw_result.detail)
        result_payload: dict[str, Any]
        next_node: WorkflowNode | None
        status: PublicRunStatus = "running"
        error: str | None = None
        plan = record.plan
        checks = record.checks
        base_commit = record.base_commit
        summary = record.summary

        if node == "inspect_and_plan":
            plan = cast(ChangePlan, raw_result)
            result_payload = {"plan": plan.model_dump(mode="json")}
            next_node = "approval"
            # The approval node owns both the automatic decision and the manual
            # pause. Keeping the run non-paused here lets auto-approval advance.
            status = "planning"
        elif node == "verify":
            verification = cast(VerificationOutcome, raw_result)
            check = verification.as_check_summary(attempt)
            checks = (*record.checks, check)
            result_payload = verification.model_dump(mode="json")
            if verification.ok:
                next_node = "review"
            elif record.metrics.repair_attempts < MAX_REPAIR_ATTEMPTS:
                next_node = "repair"
            else:
                next_node = None
                status = "unverified"
                error = verification.error or verification.status
        else:
            outcome = cast(NodeOutcome, raw_result)
            result_payload = outcome.model_dump(mode="json")
            if node == "review" and not outcome.ok:
                if record.metrics.review_repairs < MAX_REVIEW_REPAIRS:
                    next_node = "review_repair"
                else:
                    next_node = None
                    status = "failed"
                    error = outcome.error or outcome.summary or "review rejected changes"
            elif not outcome.ok:
                return self._save_failed_outcome(
                    latest,
                    node=node,
                    attempt=attempt,
                    outcome=outcome,
                    duration_ms=duration_ms,
                    metrics=metrics,
                )
            else:
                next_node = _successor(node)
                if node == "prepare":
                    candidate_commit = outcome.detail.get("base_commit")
                    if candidate_commit is not None:
                        if not isinstance(candidate_commit, str) or re.fullmatch(
                            r"[a-f0-9]{40}", candidate_commit
                        ) is None:
                            raise ValueError(
                                "prepare base_commit must be a lowercase 40-character SHA"
                            )
                        base_commit = candidate_commit
                elif node == "repair":
                    metrics = metrics.model_copy(
                        update={"repair_attempts": record.metrics.repair_attempts + 1}
                    )
                elif node == "review_repair":
                    metrics = metrics.model_copy(
                        update={"review_repairs": record.metrics.review_repairs + 1}
                    )
                elif node == "finalize":
                    status = "succeeded"
                    summary = outcome.summary or record.summary

        finished_at = self._now() if status in TERMINAL_STATUSES else None
        updated = self._updated_record(
            record,
            status=status,
            current_node=next_node,
            plan=plan,
            checks=checks,
            base_commit=base_commit,
            metrics=metrics,
            error=error,
            summary=summary,
            finished_at=finished_at,
        )
        result_payload["duration_ms"] = duration_ms
        return self._store.save_checkpoint(
            latest,
            node=node,
            attempt=attempt,
            phase="completed",
            idempotency_key=_idempotency_key(record.run_id, node, attempt),
            result=result_payload,
            record=updated,
            next_node=next_node,
            created_at=self._now(),
        )

    def _save_failed_outcome(
        self,
        snapshot: WorkflowSnapshot,
        *,
        node: WorkflowNode,
        attempt: int,
        outcome: NodeOutcome,
        duration_ms: int,
        metrics: RunMetrics,
    ) -> WorkflowSnapshot:
        if node == "repair":
            metrics = metrics.model_copy(
                update={"repair_attempts": snapshot.record.metrics.repair_attempts + 1}
            )
        elif node == "review_repair":
            metrics = metrics.model_copy(
                update={"review_repairs": snapshot.record.metrics.review_repairs + 1}
            )
        message = outcome.error or outcome.summary or f"{node} failed"
        requested_status = outcome.detail.get("failure_status", "failed")
        failure_status: PublicRunStatus = (
            "policy_denied" if requested_status == "policy_denied" else "failed"
        )
        updated = self._updated_record(
            snapshot.record,
            status=failure_status,
            current_node=None,
            metrics=metrics,
            error=message,
            finished_at=self._now(),
        )
        result = outcome.model_dump(mode="json")
        result["duration_ms"] = duration_ms
        return self._store.save_checkpoint(
            snapshot,
            node=node,
            attempt=attempt,
            phase="failed",
            idempotency_key=_idempotency_key(snapshot.record.run_id, node, attempt),
            result=result,
            record=updated,
            next_node=None,
            created_at=self._now(),
        )

    def _fail_node(
        self,
        snapshot: WorkflowSnapshot,
        node: WorkflowNode,
        attempt: int,
        exc: Exception,
        duration_ms: int,
    ) -> WorkflowSnapshot:
        latest = self._latest_compatible_snapshot(snapshot)
        message = _bounded_error(exc)
        metrics = latest.record.metrics.model_copy(
            update={"node_count": latest.record.metrics.node_count + 1}
        )
        if node == "repair":
            metrics = metrics.model_copy(
                update={"repair_attempts": latest.record.metrics.repair_attempts + 1}
            )
        elif node == "review_repair":
            metrics = metrics.model_copy(
                update={"review_repairs": latest.record.metrics.review_repairs + 1}
            )
        updated = self._updated_record(
            latest.record,
            status="failed",
            current_node=None,
            metrics=metrics,
            error=message,
            finished_at=self._now(),
        )
        return self._store.save_checkpoint(
            latest,
            node=node,
            attempt=attempt,
            phase="failed",
            idempotency_key=_idempotency_key(latest.record.run_id, node, attempt),
            result={"error": message, "duration_ms": duration_ms},
            record=updated,
            next_node=None,
            created_at=self._now(),
        )

    def _pause_or_auto_approve(
        self, snapshot: WorkflowSnapshot
    ) -> WorkflowSnapshot:
        node: WorkflowNode = "approval"
        attempt = 0
        key = _idempotency_key(snapshot.record.run_id, node, attempt)
        completed = self._store.get_checkpoint(
            snapshot.record.run_id,
            node=node,
            attempt=attempt,
            phase="completed",
        )
        if completed is not None:
            return self._store.load_run(snapshot.record.run_id)
        started = self._store.get_checkpoint(
            snapshot.record.run_id,
            node=node,
            attempt=attempt,
            phase="started",
        )
        if started is None:
            planning = self._record_for_started_node(snapshot.record, node)
            snapshot = self._store.save_checkpoint(
                snapshot,
                node=node,
                attempt=attempt,
                phase="started",
                idempotency_key=key,
                result={},
                record=planning,
                next_node=node,
                created_at=self._now(),
            )
        if snapshot.record.auto_approve:
            return self._complete_approval(
                snapshot,
                approved=True,
                reason="auto-approved by run configuration",
                automatic=True,
            )
        paused = self._updated_record(
            snapshot.record,
            status="awaiting_approval",
            current_node="approval",
        )
        return self._store.save_checkpoint(
            snapshot,
            node=node,
            attempt=attempt,
            phase="paused",
            idempotency_key=key,
            result={"reason": "human approval required"},
            record=paused,
            next_node=node,
            created_at=self._now(),
        )

    def _complete_approval(
        self,
        snapshot: WorkflowSnapshot,
        *,
        approved: bool,
        reason: str | None,
        automatic: bool,
    ) -> WorkflowSnapshot:
        normalized_reason = reason.strip() if isinstance(reason, str) else reason
        metrics = snapshot.record.metrics.model_copy(
            update={"node_count": snapshot.record.metrics.node_count + 1}
        )
        if approved:
            status: PublicRunStatus = "running"
            next_node: WorkflowNode | None = "implement"
            finished_at = None
        else:
            status = "rejected"
            next_node = None
            finished_at = self._now()
        updated = self._updated_record(
            snapshot.record,
            status=status,
            current_node=next_node,
            metrics=metrics,
            approval_reason=normalized_reason,
            finished_at=finished_at,
        )
        return self._store.save_checkpoint(
            snapshot,
            node="approval",
            attempt=0,
            phase="completed",
            idempotency_key=_idempotency_key(snapshot.record.run_id, "approval", 0),
            result={
                "approved": approved,
                "automatic": automatic,
                "reason": normalized_reason,
            },
            record=updated,
            next_node=next_node,
            created_at=self._now(),
        )

    def _cancel_snapshot(
        self, snapshot: WorkflowSnapshot, *, reason: str | None = None
    ) -> WorkflowSnapshot:
        node = snapshot.next_node or snapshot.record.current_node or "finalize"
        attempt = _node_attempt(snapshot.record, node)
        normalized_reason = reason.strip() if isinstance(reason, str) else reason
        updated = self._updated_record(
            snapshot.record,
            status="cancelled",
            current_node=None,
            cancel_requested=True,
            error=normalized_reason,
            finished_at=self._now(),
        )
        return self._store.save_checkpoint(
            snapshot,
            node=node,
            attempt=attempt,
            phase="cancelled",
            idempotency_key=_idempotency_key(snapshot.record.run_id, node, attempt),
            result={"reason": normalized_reason},
            record=updated,
            next_node=None,
            created_at=self._now(),
        )

    def _fail_without_node(
        self, snapshot: WorkflowSnapshot, message: str
    ) -> WorkflowSnapshot:
        node = snapshot.record.current_node or "finalize"
        attempt = _node_attempt(snapshot.record, node)
        updated = self._updated_record(
            snapshot.record,
            status="failed",
            current_node=None,
            error=message,
            finished_at=self._now(),
        )
        return self._store.save_checkpoint(
            snapshot,
            node=node,
            attempt=attempt,
            phase="failed",
            idempotency_key=_idempotency_key(snapshot.record.run_id, node, attempt),
            result={"error": message},
            record=updated,
            next_node=None,
            created_at=self._now(),
        )

    def _record_for_started_node(
        self, record: RunRecord, node: WorkflowNode
    ) -> RunRecord:
        status: PublicRunStatus = (
            "planning"
            if node in {"prepare", "baseline_check", "inspect_and_plan", "approval"}
            else "running"
        )
        return self._updated_record(
            record,
            status=status,
            current_node=node,
            started_at=record.started_at or self._now(),
        )

    def _updated_record(self, record: RunRecord, **changes: Any) -> RunRecord:
        now = self._now()
        metrics = cast(RunMetrics, changes.get("metrics", record.metrics))
        started_at = cast(str | None, changes.get("started_at", record.started_at))
        if started_at is not None:
            metrics = metrics.model_copy(
                update={"duration_ms": _wall_duration_ms(started_at, now)}
            )
        changes["metrics"] = metrics
        changes["updated_at"] = now
        payload = record.model_dump(mode="python")
        payload.update(changes)
        return RunRecord.model_validate(payload)

    def _latest_compatible_snapshot(
        self, snapshot: WorkflowSnapshot
    ) -> WorkflowSnapshot:
        latest = self._store.load_run(snapshot.record.run_id)
        if latest.version == snapshot.version:
            return snapshot
        if latest.record.cancel_requested and not snapshot.record.cancel_requested:
            return latest
        raise ConcurrentUpdateError(
            f"run {snapshot.record.run_id} changed while a node was executing"
        )


def _graph_route(state: _WorkflowGraphState) -> GraphRoute:
    return state["route"]


def _successor(node: WorkflowNode) -> WorkflowNode | None:
    successors: dict[WorkflowNode, WorkflowNode | None] = {
        "prepare": "baseline_check",
        "baseline_check": "inspect_and_plan",
        "inspect_and_plan": "approval",
        "approval": "implement",
        "implement": "verify",
        "verify": "review",
        "repair": "verify",
        "review": "finalize",
        "review_repair": "verify",
        "finalize": None,
    }
    return successors[node]


def _node_attempt(record: RunRecord, node: WorkflowNode) -> int:
    if node == "verify":
        return len(record.checks)
    if node == "repair":
        return record.metrics.repair_attempts + 1
    if node in {"review", "review_repair"}:
        return record.metrics.review_repairs + (1 if node == "review_repair" else 0)
    return 0


def _idempotency_key(run_id: str, node: WorkflowNode, attempt: int) -> str:
    return f"{run_id}:{node}:{attempt}"


def _previous_result(
    checkpoints: tuple[WorkflowCheckpoint, ...],
    node: WorkflowNode,
    attempt: int,
) -> dict[str, Any] | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.node == node and checkpoint.attempt == attempt:
            continue
        if checkpoint.phase in {"completed", "failed"}:
            return dict(checkpoint.result)
    return None


def _duration_ms(started: float, finished: float) -> int:
    elapsed = finished - started
    if not math.isfinite(elapsed):
        return 0
    return max(0, round(elapsed * 1000))


def _accumulate_usage(metrics: RunMetrics, detail: dict[str, Any]) -> RunMetrics:
    updates: dict[str, int] = {}
    for field in ("tool_calls", "tokens"):
        increment = detail.get(field, 0)
        if type(increment) is not int or increment < 0:
            raise ValueError(f"{field} detail must be a non-negative integer")
        updates[field] = getattr(metrics, field) + increment
    payload = metrics.model_dump(mode="python")
    payload.update(updates)
    return RunMetrics.model_validate(payload)


def _wall_duration_ms(started_at: str, now: str) -> int:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(now.replace("Z", "+00:00"))
        return max(0, round((end - start).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def _bounded_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {message}"[:4000]


__all__ = [
    "CheckpointPhase",
    "CheckpointStore",
    "ConcurrentUpdateError",
    "DuplicateRunError",
    "InvalidTransitionError",
    "MAX_REPAIR_ATTEMPTS",
    "MAX_REVIEW_REPAIRS",
    "NodeOutcome",
    "OperationContext",
    "RunBusyError",
    "RunNotFoundError",
    "VerificationOutcome",
    "WorkflowCheckpoint",
    "WorkflowEngine",
    "WorkflowError",
    "WorkflowOperations",
    "WorkflowSnapshot",
]
