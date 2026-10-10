from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from repo_agent.checks import CheckProfile
import repo_agent.service as service_module
from repo_agent.run_models import ChangePlan, RunRecord, TERMINAL_STATUSES, utc_now
from repo_agent.service import (
    DataDirectoryInUseError,
    QueueFullError,
    RepositoryNotAllowedError,
    RunExecutionContext,
    RunService,
)


class ApprovalWorkflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def execute(
        self,
        record: RunRecord,
        context: RunExecutionContext,
        *,
        resume: bool,
    ) -> RunRecord:
        self.calls.append((record.run_id, resume))
        if not resume:
            plan = ChangePlan(
                goal="Repair the requested behavior",
                files=("src/app.py", "tests/test_app.py"),
                steps=("Inspect the failing path", "Apply a bounded patch"),
                checks=("python-pytest",),
            )
            waiting = RunRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "status": "awaiting_approval",
                    "current_node": "approval",
                    "plan": plan,
                }
            )
            return context.checkpoint(
                waiting,
                event="node_completed",
                node="approval",
            )

        succeeded = RunRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "status": "succeeded",
                "current_node": "finalize",
            }
        )
        context.artifacts.write_patch(record.run_id, "diff --git a/a b/a\n")
        return context.checkpoint(
            succeeded,
            event="node_completed",
            node="finalize",
        )

    def cancel(self, run_id: str) -> None:
        del run_id


def _observe_terminal_publications(monkeypatch, service: RunService):
    observations: list[tuple[str, str | None, bool, str | None]] = []
    original_put = service.database.put

    def observe(record: RunRecord) -> None:
        if record.status in TERMINAL_STATUSES:
            try:
                result = json.loads(
                    service.artifact_path(record.run_id, "result").read_text(
                        encoding="utf-8"
                    )
                )
                report = service.artifact_path(record.run_id, "report").read_text(
                    encoding="utf-8"
                )
                trace_lines = service.artifact_path(
                    record.run_id, "trace"
                ).read_text(encoding="utf-8").splitlines()
                last_event = (
                    json.loads(trace_lines[-1]).get("event") if trace_lines else None
                )
                observations.append(
                    (
                        record.status,
                        result.get("status"),
                        f"- Status: `{record.status}`" in report,
                        last_event,
                    )
                )
            except (OSError, ValueError, TypeError):
                observations.append((record.status, None, False, None))
        original_put(record)

    monkeypatch.setattr(service.database, "put", observe)
    return observations


def test_run_service_pauses_for_approval_then_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = ApprovalWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        created = service.create_run(repo_path=tmp_path, task=" repair it ")
        waiting = service.wait(created.run_id, timeout=5)

        assert waiting.status == "awaiting_approval"
        assert waiting.plan is not None
        assert workflow.calls == [(created.run_id, False)]

        publications = _observe_terminal_publications(monkeypatch, service)
        queued = service.decide(created.run_id, approve=True, reason="plan reviewed")
        assert queued.status == "queued"
        finished = service.wait(created.run_id, timeout=5)
        assert finished.status == "succeeded"
        assert finished.finished_at is not None
        assert workflow.calls == [(created.run_id, False), (created.run_id, True)]
        assert service.artifact_path(created.run_id, "patch").read_text(
            encoding="utf-8"
        ).startswith("diff --git")
        assert service.artifact_path(created.run_id, "report").read_text(
            encoding="utf-8"
        ).startswith("# Repo Maintainer Agent run")
        assert publications == [
            ("succeeded", "succeeded", True, "workflow_stopped")
        ]


def test_approval_pause_is_not_published_before_worker_finishes_sync(
    tmp_path: Path,
) -> None:
    checkpointed = threading.Event()
    release = threading.Event()

    class SlowPauseWorkflow(ApprovalWorkflow):
        def execute(
            self,
            record: RunRecord,
            context: RunExecutionContext,
            *,
            resume: bool,
        ) -> RunRecord:
            if resume:
                return super().execute(record, context, resume=resume)
            plan = ChangePlan(
                goal="Repair safely",
                files=("src/app.py",),
                steps=("Repair",),
                checks=("python-pytest",),
            )
            waiting = RunRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "status": "awaiting_approval",
                    "current_node": "approval",
                    "plan": plan,
                }
            )
            context.checkpoint(waiting, event="node_completed", node="approval")
            checkpointed.set()
            assert release.wait(timeout=5)
            return waiting

    workflow = SlowPauseWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        created = service.create_run(repo_path=tmp_path, task="repair")
        assert checkpointed.wait(timeout=5)
        assert service.get_run(created.run_id).status == "running"
        release.set()
        waiting = service.wait(created.run_id, timeout=5)
        assert waiting.status == "awaiting_approval"


def test_run_service_rejection_is_terminal_without_resume(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = ApprovalWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        created = service.create_run(repo_path=tmp_path, task="repair it")
        service.wait(created.run_id, timeout=5)
        publications = _observe_terminal_publications(monkeypatch, service)
        rejected = service.decide(
            created.run_id, approve=False, reason="scope is too broad"
        )

        assert rejected.status == "rejected"
        assert rejected.approval_reason == "scope is too broad"
        assert workflow.calls == [(created.run_id, False)]
        assert publications == [("rejected", "rejected", True, "plan_rejected")]
        report = service.artifact_path(created.run_id, "report").read_text(
            encoding="utf-8"
        )
        assert "## Change plan" in report
        assert "## Approved plan" not in report


def test_run_service_cancels_a_queued_run_without_execution(
    tmp_path: Path, monkeypatch
) -> None:
    class CancelSpyWorkflow(ApprovalWorkflow):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls: list[str] = []

        def cancel(self, run_id: str) -> None:
            self.cancel_calls.append(run_id)

    workflow = CancelSpyWorkflow()
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
        start_worker=False,
    )
    try:
        created = service.create_run(repo_path=tmp_path, task="repair it")
        publications = _observe_terminal_publications(monkeypatch, service)
        cancelled = service.cancel(created.run_id)
        assert cancelled.status == "cancelled"
        assert cancelled.cancel_requested is True
        assert workflow.calls == []
        assert workflow.cancel_calls == []
        assert publications == [
            ("cancelled", "cancelled", True, "cancel_requested")
        ]
    finally:
        service.close()


def test_run_service_enforces_waiting_queue_limit(tmp_path: Path) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        max_queue=2,
        start_worker=False,
    )
    try:
        service.create_run(repo_path=tmp_path, task="first")
        service.create_run(repo_path=tmp_path, task="second")
        with pytest.raises(QueueFullError, match="queue is full"):
            service.create_run(repo_path=tmp_path, task="third")
    finally:
        service.close()


def test_scheduler_snapshot_fifo_and_cancelled_tombstone_releases_capacity(
    tmp_path: Path,
) -> None:
    class ControlledWorkflow:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release_first = threading.Event()
            self.calls: list[str] = []
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def execute(self, record, _context, *, resume):
            del resume
            with self.lock:
                self.calls.append(record.run_id)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                call_number = len(self.calls)
            try:
                if call_number == 1:
                    self.started.set()
                    assert self.release_first.wait(timeout=5)
                return RunRecord.model_validate(
                    {
                        **record.model_dump(mode="python"),
                        "status": "succeeded",
                        "current_node": "finalize",
                    }
                )
            finally:
                with self.lock:
                    self.active -= 1

        def cancel(self, _run_id: str) -> None:
            return None

    workflow = ControlledWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
        max_queue=2,
    ) as service:
        first = service.create_run(repo_path=tmp_path, task="first")
        assert workflow.started.wait(timeout=5)
        second = service.create_run(repo_path=tmp_path, task="second")
        third = service.create_run(repo_path=tmp_path, task="third")

        saturated = service.scheduler_snapshot()
        assert saturated.queue_capacity == 2
        assert saturated.max_concurrency == 1
        assert saturated.active_run_ids == (first.run_id,)
        assert saturated.queued_run_ids == (second.run_id, third.run_id)
        assert service.queue_position(second.run_id) == 1
        assert service.queue_position(third.run_id) == 2
        with pytest.raises(QueueFullError, match="queue is full"):
            service.create_run(repo_path=tmp_path, task="rejected")

        assert service.cancel(second.run_id).status == "cancelled"
        after_cancel = service.scheduler_snapshot()
        assert after_cancel.queue_depth == 1
        assert after_cancel.queued_run_ids == (third.run_id,)
        assert service.queue_position(second.run_id) is None
        assert service.queue_position(third.run_id) == 1
        assert tuple(job.run_id for job in service._queue) == (third.run_id,)

        cancelled_ids: list[str] = []
        for index in range(20):
            transient = service.create_run(
                repo_path=tmp_path,
                task=f"cancelled waiting run {index}",
            )
            cancelled_ids.append(transient.run_id)
            assert service.cancel(transient.run_id).status == "cancelled"
            assert tuple(job.run_id for job in service._queue) == (third.run_id,)

        fourth = service.create_run(repo_path=tmp_path, task="fourth")
        assert service.queue_position(fourth.run_id) == 2
        workflow.release_first.set()
        assert service.wait(first.run_id, timeout=5).status == "succeeded"
        assert service.wait(third.run_id, timeout=5).status == "succeeded"
        assert service.wait(fourth.run_id, timeout=5).status == "succeeded"

        fifth = service.create_run(repo_path=tmp_path, task="fifth")
        assert service.wait(fifth.run_id, timeout=5).status == "succeeded"
        assert workflow.calls == [
            first.run_id,
            third.run_id,
            fourth.run_id,
            fifth.run_id,
        ]
        assert not set(cancelled_ids).intersection(workflow.calls)
        assert workflow.max_active == 1
        assert service.scheduler_snapshot().queue_depth == 0
        assert service.ready() is True


def test_concurrent_producers_and_cancellations_preserve_queue_invariants(
    tmp_path: Path,
) -> None:
    class BlockingWorkflow:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls: list[str] = []
            self.active_run_id: str | None = None

        def execute(self, record, _context, *, resume):
            del resume
            self.calls.append(record.run_id)
            self.active_run_id = record.run_id
            self.started.set()
            assert self.release.wait(timeout=10)
            return record.model_copy(
                update={"status": "succeeded", "current_node": "finalize"}
            )

        def cancel(self, run_id: str) -> None:
            if run_id == self.active_run_id:
                self.release.set()

    workflow = BlockingWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
        max_queue=8,
    ) as service:
        active = service.create_run(repo_path=tmp_path, task="hold worker")
        assert workflow.started.wait(timeout=5)
        barrier = threading.Barrier(9)
        failures: list[BaseException] = []
        cancelled_ids: list[str] = []
        result_lock = threading.Lock()

        def produce(worker_index: int) -> None:
            barrier.wait(timeout=5)
            for item_index in range(10):
                try:
                    record = service.create_run(
                        repo_path=tmp_path,
                        task=f"producer {worker_index} item {item_index}",
                    )
                except QueueFullError:
                    continue
                try:
                    service.cancel(record.run_id)
                    with result_lock:
                        cancelled_ids.append(record.run_id)
                except BaseException as exc:
                    with result_lock:
                        failures.append(exc)

        producers = [
            threading.Thread(target=produce, args=(index,)) for index in range(8)
        ]
        for producer in producers:
            producer.start()
        barrier.wait(timeout=5)
        for producer in producers:
            producer.join(timeout=15)

        assert all(not producer.is_alive() for producer in producers)
        assert failures == []
        assert cancelled_ids
        with service._lock:
            queued_ids = tuple(job.run_id for job in service._queue)
            assert queued_ids == tuple(service._scheduled)
            assert len(queued_ids) <= 8
            assert len(set(queued_ids)) == len(queued_ids)
        assert service.scheduler_snapshot().queue_depth == 0

        capacity = [service._pending_slots.acquire(blocking=False) for _ in range(8)]
        try:
            assert capacity == [True] * 8
            assert service._pending_slots.acquire(blocking=False) is False
        finally:
            for held in capacity:
                if held:
                    service._pending_slots.release()

        workflow.release.set()
        assert service.wait(active.run_id, timeout=5).status == "succeeded"
        assert workflow.calls == [active.run_id]


def test_wait_for_events_closes_query_wait_race_and_validates_bounds(
    tmp_path: Path, monkeypatch
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        created = service.create_run(repo_path=tmp_path, task="observe events")
        initial = service.event_page(created.run_id, after_sequence=0)
        assert [event["event"] for event in initial.events] == ["run_created"]

        first_empty_query = threading.Event()
        release_query = threading.Event()
        original_event_page = service.database.event_page
        observed_empty = False

        def pause_after_empty(*args, **kwargs):
            nonlocal observed_empty
            page = original_event_page(*args, **kwargs)
            if not page.events and not observed_empty:
                observed_empty = True
                first_empty_query.set()
                assert release_query.wait(timeout=5)
            return page

        monkeypatch.setattr(service.database, "event_page", pause_after_empty)
        pages = []
        waiter = threading.Thread(
            target=lambda: pages.append(
                service.wait_for_events(
                    created.run_id,
                    after_sequence=initial.latest_sequence,
                    timeout=5,
                )
            )
        )
        waiter.start()
        assert first_empty_query.wait(timeout=5)

        published = threading.Event()

        def publish() -> None:
            service.database.append_event(
                created.run_id,
                {"event": "test_event", "detail": {}, "timestamp": "now"},
            )
            published.set()
            service._notify(created.run_id)

        publisher = threading.Thread(target=publish)
        publisher.start()
        assert published.wait(timeout=5)
        release_query.set()
        publisher.join(timeout=5)
        waiter.join(timeout=5)

        assert publisher.is_alive() is False
        assert waiter.is_alive() is False
        assert [event["event"] for event in pages[0].events] == ["test_event"]

        for cursor in (-1, True, 1 << 63):
            with pytest.raises(ValueError, match="after_sequence"):
                service.event_page(created.run_id, after_sequence=cursor)  # type: ignore[arg-type]
        for limit in (0, True, 257):
            with pytest.raises(ValueError, match="limit"):
                service.event_page(
                    created.run_id,
                    after_sequence=0,
                    limit=limit,  # type: ignore[arg-type]
                )
        for timeout in (-1, True, float("nan")):
            with pytest.raises(ValueError, match="finite non-negative"):
                service.wait_for_events(
                    created.run_id,
                    after_sequence=0,
                    timeout=timeout,  # type: ignore[arg-type]
                )
    finally:
        service.close()


@pytest.mark.parametrize(
    ("status", "current_node", "event_name"),
    (
        ("awaiting_approval", "approval", "plan_approved"),
        ("interrupted", "implement", "run_resumed"),
    ),
)
def test_approve_and_resume_events_wake_unbounded_subscribers_after_schedule(
    tmp_path: Path,
    monkeypatch,
    status: str,
    current_node: str,
    event_name: str,
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    waiter_thread: threading.Thread | None = None
    action_thread: threading.Thread | None = None
    try:
        run_id = "a" * 32
        now = utc_now()
        plan = ChangePlan(
            goal="Repair safely",
            files=("src/app.py",),
            steps=("Apply the reviewed change",),
            checks=("python-pytest",),
        )
        record = RunRecord(
            run_id=run_id,
            repo_path=str(tmp_path),
            task="repair",
            status=status,
            current_node=current_node,
            plan=plan,
            created_at=now,
            updated_at=now,
        )
        artifact_dir = service.artifacts.initialize(run_id)
        service.database.create(record, artifact_dir)
        service.artifacts.write_record(record)
        cursor = service.database.append_event(
            run_id,
            {"event": "workflow_checkpoint", "timestamp": now},
        )

        first_empty_query = threading.Event()
        second_empty_query = threading.Event()
        trace_reached = threading.Event()
        waiter_queries = 0
        original_event_page = service.database.event_page
        original_trace = service._trace

        def observe_empty_page(*args, **kwargs):
            nonlocal waiter_queries
            page = original_event_page(*args, **kwargs)
            if threading.current_thread() is waiter_thread and not page.events:
                waiter_queries += 1
                if waiter_queries == 1:
                    first_empty_query.set()
                elif waiter_queries == 2:
                    second_empty_query.set()
            return page

        def pause_trace(
            traced_record: RunRecord,
            traced_event: str,
            *,
            detail: dict[str, object],
        ) -> None:
            if traced_event == event_name:
                trace_reached.set()
                assert second_empty_query.wait(timeout=5)
            original_trace(traced_record, traced_event, detail=detail)

        monkeypatch.setattr(service.database, "event_page", observe_empty_page)
        monkeypatch.setattr(service, "_trace", pause_trace)
        pages = []
        action_errors: list[BaseException] = []
        waiter_thread = threading.Thread(
            target=lambda: pages.append(
                service.wait_for_events(
                    run_id,
                    after_sequence=cursor,
                    timeout=None,
                )
            )
        )
        waiter_thread.start()
        assert first_empty_query.wait(timeout=5)

        def act() -> None:
            try:
                if status == "awaiting_approval":
                    service.decide(run_id, approve=True)
                else:
                    service.resume(run_id)
            except BaseException as exc:
                action_errors.append(exc)

        action_thread = threading.Thread(target=act)
        action_thread.start()
        assert trace_reached.wait(timeout=5)
        action_thread.join(timeout=5)
        waiter_thread.join(timeout=5)

        assert action_thread.is_alive() is False
        assert waiter_thread.is_alive() is False
        assert action_errors == []
        assert [event["event"] for event in pages[0].events] == [event_name]
    finally:
        service._notify("a" * 32)
        if action_thread is not None:
            action_thread.join(timeout=5)
        if waiter_thread is not None:
            waiter_thread.join(timeout=5)
        service.close()


def test_workflow_auxiliary_event_is_persisted_and_wakes_subscribers(
    tmp_path: Path,
) -> None:
    class EventWorkflow:
        def execute(self, record, context, *, resume):
            del resume
            context.publish_event(
                {
                    "event": "model_tool_summary",
                    "phase": "implement",
                    "tool_calls": 1,
                    "timestamp": "2026-10-09T00:00:00.000Z",
                }
            )
            return record.model_copy(
                update={"status": "succeeded", "current_node": "finalize"}
            )

        def cancel(self, _run_id: str) -> None:
            return None

    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=EventWorkflow(),
    ) as service:
        created = service.create_run(repo_path=tmp_path, task="publish progress")
        finished = service.wait(created.run_id, timeout=5)
        events = service.event_page(created.run_id, after_sequence=0).events

    assert finished.status == "succeeded"
    summary = next(
        event for event in events if event["event"] == "model_tool_summary"
    )
    assert summary["phase"] == "implement"
    assert summary["tool_calls"] == 1


def test_run_service_rejects_extreme_timeout_and_wait_flag_values(
    tmp_path: Path,
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    invalid_timeouts: tuple[object, ...] = (
        float("nan"),
        float("inf"),
        float("-inf"),
        1e300,
        10**1000,
        -0.001,
        True,
        "1",
        None,
    )
    try:
        for value in invalid_timeouts[:-1]:
            with pytest.raises(ValueError, match="finite non-negative"):
                service.wait("a" * 32, timeout=value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="stop_at_approval"):
            service.wait("a" * 32, timeout=0, stop_at_approval=1)  # type: ignore[arg-type]

        for value in invalid_timeouts[:-1]:
            with pytest.raises(ValueError, match="finite non-negative"):
                service.close(timeout=value)  # type: ignore[arg-type]
            assert service._lease is not None and service._lease.held
    finally:
        service.close()


def test_run_service_rejects_nul_in_task_and_approval_reason(tmp_path: Path) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        with pytest.raises(ValueError, match="task"):
            service.create_run(repo_path=tmp_path, task="bad\x00task")
        with pytest.raises(ValueError, match="reason"):
            service.decide("a" * 32, approve=False, reason="bad\x00reason")
    finally:
        service.close()


def test_run_service_rejects_repository_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(allowed,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        with pytest.raises(RepositoryNotAllowedError, match="outside"):
            service.create_run(repo_path=outside, task="repair")
    finally:
        service.close()


def test_service_hides_persisted_runs_outside_current_allowed_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    data_dir = tmp_path / "data"
    first = RunService(
        data_dir,
        allowed_repo_roots=(first_root,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        created = first.create_run(repo_path=first_root, task="repair")
    finally:
        first.close()

    second = RunService(
        data_dir,
        allowed_repo_roots=(second_root,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        assert second.list_runs() == ()
        with pytest.raises(RepositoryNotAllowedError, match="outside"):
            second.get_run(created.run_id)
        with pytest.raises(RepositoryNotAllowedError, match="outside"):
            second.event_page(created.run_id, after_sequence=0)
        with pytest.raises(RepositoryNotAllowedError, match="outside"):
            second.wait_for_events(
                created.run_id,
                after_sequence=0,
                timeout=0,
            )
    finally:
        second.close()


class BlockingWorkflow:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def execute(
        self,
        record: RunRecord,
        context: RunExecutionContext,
        *,
        resume: bool,
    ) -> RunRecord:
        del resume
        self.started.set()
        assert context.cancel_event.wait(timeout=5)
        self.cancelled.set()
        return record

    def cancel(self, run_id: str) -> None:
        del run_id


def test_run_service_cooperatively_cancels_active_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = BlockingWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        created = service.create_run(repo_path=tmp_path, task="repair")
        assert workflow.started.wait(timeout=5)
        publications = _observe_terminal_publications(monkeypatch, service)
        requested = service.cancel(created.run_id)
        assert requested.cancel_requested is True
        finished = service.wait(created.run_id, timeout=5)
        assert workflow.cancelled.wait(timeout=5)
        assert finished.status == "cancelled"
        assert publications == [
            ("cancelled", "cancelled", True, "workflow_cancelled")
        ]


def test_active_cancel_hook_failure_does_not_undo_durable_cancellation(
    tmp_path: Path,
) -> None:
    class RaisingCancelWorkflow(BlockingWorkflow):
        def cancel(self, run_id: str) -> None:
            del run_id
            raise RuntimeError("cancel hook failed")

    workflow = RaisingCancelWorkflow()
    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        created = service.create_run(repo_path=tmp_path, task="repair")
        assert workflow.started.wait(timeout=5)

        requested = service.cancel(created.run_id)
        finished = service.wait(created.run_id, timeout=5)

    assert requested.cancel_requested is True
    assert finished.status == "cancelled"
    assert workflow.cancelled.is_set()


def test_active_cancel_cannot_overwrite_terminal_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = BlockingWorkflow()
    cancel_artifact_started = threading.Event()
    release_cancel_artifact = threading.Event()
    terminal_published = threading.Event()
    cancel_results: list[RunRecord] = []
    cancel_errors: list[BaseException] = []

    with RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    ) as service:
        original_write_record = service.artifacts.write_record
        original_put = service.database.put

        def slow_cancel_record(record: RunRecord):
            if record.cancel_requested and record.status in {"planning", "running"}:
                cancel_artifact_started.set()
                assert release_cancel_artifact.wait(timeout=5)
            return original_write_record(record)

        def observe_terminal_put(record: RunRecord) -> None:
            original_put(record)
            if record.status in TERMINAL_STATUSES:
                terminal_published.set()

        monkeypatch.setattr(service.artifacts, "write_record", slow_cancel_record)
        monkeypatch.setattr(service.database, "put", observe_terminal_put)
        created = service.create_run(repo_path=tmp_path, task="repair")
        assert workflow.started.wait(timeout=5)

        def request_cancel() -> None:
            try:
                cancel_results.append(service.cancel(created.run_id))
            except BaseException as exc:
                cancel_errors.append(exc)

        cancel_thread = threading.Thread(target=request_cancel)
        cancel_thread.start()
        try:
            assert cancel_artifact_started.wait(timeout=5)
            assert workflow.cancelled.wait(timeout=5)
            assert terminal_published.is_set() is False
        finally:
            release_cancel_artifact.set()
            cancel_thread.join(timeout=5)

        assert cancel_thread.is_alive() is False
        assert cancel_errors == []
        assert len(cancel_results) == 1
        finished = service.wait(created.run_id, timeout=5)
        assert finished.status == "cancelled"
        assert terminal_published.is_set() is True
        result = json.loads(
            service.artifact_path(created.run_id, "result").read_text(
                encoding="utf-8"
            )
        )
        assert result["status"] == "cancelled"


def test_run_service_publishes_failure_after_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    class FailingWorkflow(ApprovalWorkflow):
        def execute(
            self,
            record: RunRecord,
            context: RunExecutionContext,
            *,
            resume: bool,
        ) -> RunRecord:
            del record, context, resume
            raise RuntimeError("planned workflow failure")

    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=FailingWorkflow(),
        start_worker=False,
    )
    try:
        created = service.create_run(repo_path=tmp_path, task="repair")
        publications = _observe_terminal_publications(monkeypatch, service)
        service.start()
        finished = service.wait(created.run_id, timeout=5)

        assert finished.status == "failed"
        assert publications == [("failed", "failed", True, "workflow_failed")]
    finally:
        service.close()


def test_default_workflow_receives_service_bootstrap_authorization(
    tmp_path: Path,
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        allow_bootstrap=True,
        start_worker=False,
    )
    try:
        assert getattr(service._runner, "allow_bootstrap") is True
    finally:
        service.close()


@pytest.mark.parametrize(
    ("resume", "status", "current_node"),
    [
        (False, "queued", None),
        (True, "interrupted", "implement"),
    ],
)
def test_default_workflow_rejects_maven_before_execute_or_checkpoint_resume(
    monkeypatch,
    tmp_path: Path,
    resume: bool,
    status: str,
    current_node: str | None,
) -> None:
    delegate = ApprovalWorkflow()
    runner = service_module._PythonOnlyWorkflowRunner(delegate)
    profile = CheckProfile(
        id="maven-test",
        language="java",
        image="repo-agent-maven:0.1",
        manifest="pom.xml",
        bootstrap_argv=(),
        check_argv=("mvn", "--offline", "test"),
        java_release=21,
    )
    monkeypatch.setattr(
        service_module, "detect_check_profile", lambda _repo_path: profile
    )
    record = RunRecord(
        run_id="a" * 32,
        repo_path=str(tmp_path),
        task="repair",
        status=status,
        current_node=current_node,
        created_at="2026-09-29T00:00:00.000Z",
        updated_at="2026-09-29T00:00:00.000Z",
    )

    with pytest.raises(
        RuntimeError, match="durable workflows support only Python repositories"
    ):
        runner.execute(record, None, resume=resume)  # type: ignore[arg-type]

    assert delegate.calls == []


def test_service_rejects_non_boolean_bootstrap_authorization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_bootstrap"):
        RunService(
            tmp_path / "data",
            allowed_repo_roots=(tmp_path,),
            allow_bootstrap=1,  # type: ignore[arg-type]
            start_worker=False,
        )


def test_second_service_in_process_cannot_mutate_owned_data_directory(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    first = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        created = first.create_run(repo_path=tmp_path, task="repair")
        result_path = first.artifact_path(created.run_id, "result")
        trace_path = first.artifact_path(created.run_id, "trace")
        before_result = result_path.read_bytes()
        before_trace = trace_path.read_bytes()

        with pytest.raises(DataDirectoryInUseError, match="already in use"):
            RunService(
                data_dir,
                allowed_repo_roots=(tmp_path,),
                runner=ApprovalWorkflow(),
                start_worker=False,
            )

        assert first.get_run(created.run_id).status == "queued"
        assert result_path.read_bytes() == before_result
        assert trace_path.read_bytes() == before_trace
    finally:
        first.close()

    replacement = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    replacement.close()


def test_second_service_in_another_process_cannot_acquire_data_directory(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    first = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        created = first.create_run(repo_path=tmp_path, task="repair")
        before = first.artifact_path(created.run_id, "result").read_bytes()
        script = "\n".join(
            (
                "import sys",
                "from pathlib import Path",
                "from repo_agent.service import DataDirectoryInUseError, RunService",
                "root, data = map(Path, sys.argv[1:3])",
                "try:",
                "    service = RunService(data, allowed_repo_roots=(root,), start_worker=False)",
                "except DataDirectoryInUseError:",
                "    raise SystemExit(0)",
                "else:",
                "    service.close()",
                "    raise SystemExit(3)",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), str(data_dir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert completed.returncode == 0, completed.stderr
        assert first.get_run(created.run_id).status == "queued"
        assert first.artifact_path(created.run_id, "result").read_bytes() == before
    finally:
        first.close()


def test_constructor_failure_releases_data_directory_lease(
    tmp_path: Path, monkeypatch
) -> None:
    original = RunService._publish_inflight_interrupted
    calls = 0

    def fail_once(service: RunService) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("startup recovery failed")
        return original(service)

    monkeypatch.setattr(RunService, "_publish_inflight_interrupted", fail_once)
    with pytest.raises(RuntimeError, match="startup recovery failed"):
        RunService(
            tmp_path / "data",
            allowed_repo_roots=(tmp_path,),
            runner=ApprovalWorkflow(),
            start_worker=False,
        )

    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    service.close()


def test_close_timeout_keeps_lease_until_worker_exits(tmp_path: Path) -> None:
    class StubbornWorkflow(ApprovalWorkflow):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.cancel_calls: list[str] = []

        def execute(
            self,
            record: RunRecord,
            context: RunExecutionContext,
            *,
            resume: bool,
        ) -> RunRecord:
            del context, resume
            self.started.set()
            assert self.release.wait(timeout=10)
            return record

        def cancel(self, run_id: str) -> None:
            self.cancel_calls.append(run_id)

    workflow = StubbornWorkflow()
    data_dir = tmp_path / "data"
    service = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
    )
    replacement: RunService | None = None
    try:
        created = service.create_run(repo_path=tmp_path, task="repair")
        assert workflow.started.wait(timeout=5)

        with pytest.raises(TimeoutError, match="lease is still held"):
            service.close(timeout=0.01)

        assert workflow.cancel_calls == [created.run_id]
        assert service.get_run(created.run_id).status == "planning"
        with pytest.raises(DataDirectoryInUseError, match="already in use"):
            RunService(
                data_dir,
                allowed_repo_roots=(tmp_path,),
                runner=ApprovalWorkflow(),
                start_worker=False,
            )

        workflow.release.set()
        service.close(timeout=5)
        replacement = RunService(
            data_dir,
            allowed_repo_roots=(tmp_path,),
            runner=ApprovalWorkflow(),
            start_worker=False,
        )
    finally:
        workflow.release.set()
        service.close(timeout=5)
        if replacement is not None:
            replacement.close()


def test_close_cannot_start_a_job_already_dequeued_by_the_worker(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = ApprovalWorkflow()
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=workflow,
        start_worker=False,
    )
    before_execute = threading.Event()
    release_execute = threading.Event()
    observed_cancel_events: list[threading.Event] = []
    close_errors: list[BaseException] = []
    original_execute_job = service._execute_job

    def pause_before_execute(job, cancel_event: threading.Event) -> None:
        observed_cancel_events.append(cancel_event)
        before_execute.set()
        assert release_execute.wait(timeout=5)
        original_execute_job(job, cancel_event)

    monkeypatch.setattr(service, "_execute_job", pause_before_execute)
    service.start()
    created = service.create_run(repo_path=tmp_path, task="queued shutdown")
    assert before_execute.wait(timeout=5)

    def close_service() -> None:
        try:
            service.close(timeout=5)
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=close_service)
    close_thread.start()
    try:
        assert observed_cancel_events[0].wait(timeout=5)
    finally:
        release_execute.set()
        close_thread.join(timeout=5)
        service.close(timeout=5)

    assert close_thread.is_alive() is False
    assert close_errors == []
    assert workflow.calls == []
    assert service.get_run(created.run_id).status == "interrupted"


def test_close_uses_runner_budget_unless_timeout_is_explicit(tmp_path: Path) -> None:
    class BudgetedWorkflow(ApprovalWorkflow):
        close_timeout_seconds = 125.0

    class ObservedWorker:
        def __init__(self) -> None:
            self.join_timeouts: list[float | None] = []

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return False

    observed: list[tuple[float, float]] = []
    for index, explicit_timeout in enumerate((None, 0.01)):
        service = RunService(
            tmp_path / f"data-{index}",
            allowed_repo_roots=(tmp_path,),
            runner=BudgetedWorkflow(),
            start_worker=False,
        )
        worker = ObservedWorker()
        service._worker = worker  # type: ignore[assignment]

        service.close(timeout=explicit_timeout)

        observed.append((service.close_timeout_seconds, worker.join_timeouts[0]))

    assert observed == [(125.0, 125.0), (125.0, 0.01)]


def test_closed_service_cannot_mutate_after_a_new_owner_acquires_the_lease(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    old = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    interrupted = old.create_run(repo_path=tmp_path, task="resume target")
    approval = old.create_run(repo_path=tmp_path, task="approval target")
    waiting = RunRecord.model_validate(
        {
            **approval.model_dump(mode="python"),
            "status": "awaiting_approval",
            "current_node": "approval",
            "plan": ChangePlan(
                goal="Repair safely",
                files=("src/app.py",),
                steps=("Repair",),
                checks=("python-pytest",),
            ),
        }
    )
    old.database.put(waiting)
    old.artifacts.write_record(waiting)
    old.close()

    owner = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        before_records = tuple(
            record.model_dump_json() for record in owner.database.list()
        )
        before_events = {
            record.run_id: owner.database.events(record.run_id)
            for record in owner.database.list()
        }
        artifact_root = data_dir / "runs"
        before_artifacts = {
            path.relative_to(artifact_root).as_posix(): path.read_bytes()
            for path in artifact_root.rglob("*")
            if path.is_file()
        }

        operations = (
            old.start,
            lambda: old.create_run(repo_path=tmp_path, task="must not be created"),
            lambda: old.decide(waiting.run_id, approve=False),
            lambda: old.resume(interrupted.run_id),
            lambda: old.cancel(interrupted.run_id),
            lambda: old._schedule(interrupted.run_id, resume=True),
        )
        for operation in operations:
            with pytest.raises(RuntimeError, match="RunService is closed"):
                operation()

        assert tuple(
            record.model_dump_json() for record in owner.database.list()
        ) == before_records
        assert {
            record.run_id: owner.database.events(record.run_id)
            for record in owner.database.list()
        } == before_events
        assert {
            path.relative_to(artifact_root).as_posix(): path.read_bytes()
            for path in artifact_root.rglob("*")
            if path.is_file()
        } == before_artifacts
    finally:
        owner.close()


def test_close_publishes_interrupted_artifacts_before_database_status(
    tmp_path: Path, monkeypatch
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    created = service.create_run(repo_path=tmp_path, task="queued shutdown")
    publications: list[tuple[str, str | None]] = []
    original_put = service.database.put

    def observe(record: RunRecord) -> None:
        if record.status == "interrupted":
            artifact = json.loads(
                service.artifact_path(record.run_id, "result").read_text(
                    encoding="utf-8"
                )
            )
            trace = service.artifact_path(record.run_id, "trace").read_text(
                encoding="utf-8"
            )
            publications.append(
                (
                    artifact.get("status"),
                    json.loads(trace.splitlines()[-1]).get("event"),
                )
            )
        original_put(record)

    monkeypatch.setattr(service.database, "put", observe)
    service.close()

    assert service.database.get(created.run_id).status == "interrupted"
    assert publications == [("interrupted", "service_interrupted")]


def test_startup_recovery_publishes_interrupted_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    script = "\n".join(
        (
            "import os, sys",
            "from pathlib import Path",
            "from repo_agent.service import RunService",
            "root, data = map(Path, sys.argv[1:3])",
            "service = RunService(",
            "    data, allowed_repo_roots=(root,), runner=object(), start_worker=False",
            ")",
            "record = service.create_run(repo_path=root, task='crash recovery')",
            "print(record.run_id, flush=True)",
            "os._exit(0)",
        )
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(data_dir)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert crashed.returncode == 0, crashed.stderr
    run_id = crashed.stdout.strip()
    result_path = data_dir / "runs" / run_id / "run.json"
    trace_path = data_dir / "runs" / run_id / "trace.jsonl"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "queued"

    owner = RunService(
        data_dir,
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    try:
        assert owner.get_run(run_id).status == "interrupted"
        assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == (
            "interrupted"
        )
        last_trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        assert last_trace["event"] == "service_interrupted"
        assert last_trace["detail"] == {
            "previous_status": "queued",
            "status": "interrupted",
        }
    finally:
        owner.close()


def test_close_notifies_an_unbounded_wait_after_publishing_interrupted_state(
    tmp_path: Path, monkeypatch
) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    created = service.create_run(repo_path=tmp_path, task="queued shutdown")
    recovery_started = threading.Event()
    allow_recovery = threading.Event()
    waiter_checked = threading.Event()
    waiter_rechecked = threading.Event()
    waiter_checks = 0
    wait_results: list[RunRecord] = []
    original_recovery = service._publish_inflight_interrupted
    original_get_run = service.get_run

    def blocking_recovery() -> int:
        recovery_started.set()
        assert allow_recovery.wait(timeout=5)
        return original_recovery()

    def observe_waiter(run_id: str) -> RunRecord:
        nonlocal waiter_checks
        record = original_get_run(run_id)
        if threading.current_thread() is waiter_thread and record.status == "queued":
            waiter_checks += 1
            waiter_checked.set()
            if waiter_checks >= 2:
                waiter_rechecked.set()
        return record

    monkeypatch.setattr(service, "_publish_inflight_interrupted", blocking_recovery)
    monkeypatch.setattr(service, "get_run", observe_waiter)
    waiter_thread = threading.Thread(
        target=lambda: wait_results.append(service.wait(created.run_id))
    )
    close_thread = threading.Thread(target=service.close)
    waiter_thread.start()
    assert waiter_checked.wait(timeout=5)
    close_thread.start()
    try:
        assert recovery_started.wait(timeout=5)
        assert waiter_rechecked.wait(timeout=5)
    finally:
        allow_recovery.set()
        close_thread.join(timeout=5)
        waiter_thread.join(timeout=5)
        service.close()

    assert close_thread.is_alive() is False
    assert waiter_thread.is_alive() is False
    assert [record.status for record in wait_results] == ["interrupted"]


def test_close_serializes_a_concurrent_mutation(tmp_path: Path, monkeypatch) -> None:
    service = RunService(
        tmp_path / "data",
        allowed_repo_roots=(tmp_path,),
        runner=ApprovalWorkflow(),
        start_worker=False,
    )
    created = service.create_run(repo_path=tmp_path, task="existing")
    recovery_started = threading.Event()
    allow_recovery = threading.Event()
    mutation_started = threading.Event()
    close_errors: list[BaseException] = []
    mutation_errors: list[BaseException] = []
    original_recovery = service._publish_inflight_interrupted

    def blocking_recovery() -> int:
        recovery_started.set()
        assert allow_recovery.wait(timeout=5)
        return original_recovery()

    monkeypatch.setattr(service, "_publish_inflight_interrupted", blocking_recovery)

    def close_service() -> None:
        try:
            service.close()
        except BaseException as exc:
            close_errors.append(exc)

    def mutate_service() -> None:
        mutation_started.set()
        try:
            service.create_run(repo_path=tmp_path, task="racing mutation")
        except BaseException as exc:
            mutation_errors.append(exc)

    close_thread = threading.Thread(target=close_service)
    mutation_thread = threading.Thread(target=mutate_service)
    close_thread.start()
    try:
        assert recovery_started.wait(timeout=5)
        mutation_thread.start()
        assert mutation_started.wait(timeout=5)
        mutation_thread.join(timeout=0.05)
        assert mutation_thread.is_alive() is True
    finally:
        allow_recovery.set()
        close_thread.join(timeout=5)
        mutation_thread.join(timeout=5)
        service.close()

    assert close_thread.is_alive() is False
    assert mutation_thread.is_alive() is False
    assert close_errors == []
    assert len(mutation_errors) == 1
    assert isinstance(mutation_errors[0], RuntimeError)
    assert str(mutation_errors[0]) == "RunService is closed"
    assert [record.run_id for record in service.database.list()] == [created.run_id]
