"""Application services shared by command-line and web entry points."""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time
from typing import Any, Literal, Protocol
import uuid

from .artifacts import ArtifactStore
from .loop import AgentLoop
from .models import RunResult
from .persistence import RunDatabase, RunNotFoundError
from .providers import DemoProvider, Provider
from .run_models import RunRecord, TERMINAL_STATUSES, WorkflowNode, utc_now
from .sandbox import DockerSandbox


class SandboxFactory(Protocol):
    def __call__(
        self,
        repo_path: str | os.PathLike[str],
        *,
        image: str,
        timeout_seconds: float,
        max_output_bytes: int,
        allow_mutations: bool,
    ) -> DockerSandbox: ...


def run_agent(
    *,
    task: str,
    repo_path: str | os.PathLike[str],
    image: str = "repo-agent-python:0.1",
    max_steps: int = 8,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 65536,
    provider: Provider | None = None,
    sandbox_factory: SandboxFactory | None = None,
    allow_mutations: bool = False,
) -> RunResult:
    """Run once against a fresh candidate cloned from committed HEAD."""

    if not isinstance(task, str) or not task.strip() or "\x00" in task:
        raise ValueError("task must contain non-whitespace text")
    normalized_task = task.strip()
    if type(allow_mutations) is not bool:
        raise ValueError("allow_mutations must be a boolean")
    selected_provider = provider if provider is not None else DemoProvider()
    selected_sandbox_factory = (
        sandbox_factory if sandbox_factory is not None else DockerSandbox
    )

    with selected_sandbox_factory(
        repo_path,
        image=image,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        allow_mutations=allow_mutations,
    ) as executor:
        result = AgentLoop(
            selected_provider,
            executor,
            max_steps=max_steps,
        ).run(normalized_task)
        capture = getattr(executor, "candidate_artifact", None)
        if callable(capture):
            candidate = capture()
            if candidate is not None:
                result = replace(result, candidate=candidate)
        return result


class QueueFullError(RuntimeError):
    """Raised when the bounded service queue has no available slot."""


class InvalidRunTransitionError(RuntimeError):
    """Raised when a run command is incompatible with its durable state."""


class RepositoryNotAllowedError(ValueError):
    """Raised when a REST/CLI run targets a repository outside configured roots."""


class DataDirectoryInUseError(RuntimeError):
    """Raised when another service owns the durable data directory."""


_LEASE_REGISTRY_LOCK = threading.Lock()
_LEASE_REGISTRY: set[str] = set()


class _DataDirectoryLease:
    """Non-blocking process and OS lease for one durable data directory."""

    def __init__(self, path: Path, key: str, stream: Any) -> None:
        self.path = path
        self._key = key
        self._stream = stream
        self._lock = threading.Lock()

    @classmethod
    def acquire(cls, data_dir: Path) -> "_DataDirectoryLease":
        path = data_dir / ".service.lock"
        key = os.path.normcase(str(path.resolve(strict=False)))
        with _LEASE_REGISTRY_LOCK:
            if key in _LEASE_REGISTRY:
                raise DataDirectoryInUseError(
                    f"run service data directory is already in use: {data_dir}"
                )

            stream = None
            try:
                stream = path.open("a+b")
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                    os.fsync(stream.fileno())
                stream.seek(0)
                _lock_lease_file(stream)
            except OSError as exc:
                if stream is not None:
                    stream.close()
                raise DataDirectoryInUseError(
                    f"run service data directory is already in use: {data_dir}"
                ) from exc
            _LEASE_REGISTRY.add(key)
        return cls(path, key, stream)

    @property
    def held(self) -> bool:
        with self._lock:
            return self._stream is not None

    def release(self) -> None:
        with self._lock:
            stream = self._stream
            if stream is None:
                return
            self._stream = None
            try:
                _unlock_lease_file(stream)
            finally:
                stream.close()
                with _LEASE_REGISTRY_LOCK:
                    _LEASE_REGISTRY.discard(self._key)


def _lock_lease_file(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = importlib.import_module("fcntl")

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_lease_file(stream: Any) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = importlib.import_module("fcntl")

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class _QueuedRun:
    run_id: str
    resume: bool


class RunExecutionContext:
    """Service-owned capabilities exposed to one workflow execution."""

    def __init__(
        self,
        *,
        database: RunDatabase,
        artifacts: ArtifactStore,
        run_id: str,
        cancel_event: threading.Event,
        publication_lock: threading.RLock,
        notify: "NotificationCallback",
    ) -> None:
        self.database = database
        self.artifacts = artifacts
        self.run_id = run_id
        self.cancel_event = cancel_event
        self._publication_lock = publication_lock
        self._notify = notify

    def cancelled(self) -> bool:
        return self.cancel_event.is_set() or self.database.get(
            self.run_id
        ).cancel_requested

    def checkpoint(
        self,
        record: RunRecord,
        *,
        event: str,
        node: WorkflowNode | None = None,
        detail: dict[str, Any] | None = None,
    ) -> RunRecord:
        if record.run_id != self.run_id:
            raise ValueError("workflow attempted to checkpoint another run")
        now = utc_now()
        updated = _validated_copy(record, updated_at=now)
        # A terminal or approval-pause state is published only after RunService
        # has synchronized every artifact and released the current worker job.
        # Otherwise a fast approval can race the first job's final write and be
        # overwritten back to `awaiting_approval`.
        persisted = (
            _validated_copy(updated, status="running", finished_at=None)
            if updated.status in TERMINAL_STATUSES
            or updated.status == "awaiting_approval"
            else updated
        )
        with self._publication_lock:
            self.database.put(persisted)
            sequence = self.database.append_event(
                self.run_id,
                {
                    "event": event,
                    "node": node,
                    "detail": detail or {},
                    "timestamp": now,
                },
            )
            self.artifacts.append_trace(
                self.run_id,
                {
                    "sequence": sequence,
                    "run_id": self.run_id,
                    "event": event,
                    "node": node,
                    "detail": detail or {},
                    "timestamp": now,
                },
            )
            self.artifacts.write_record(persisted)
            self._notify(self.run_id)
        return updated


NotificationCallback = Any


class DurableWorkflowRunner(Protocol):
    """Execution boundary implemented by the Day 4 workflow graph."""

    def execute(
        self,
        record: RunRecord,
        context: RunExecutionContext,
        *,
        resume: bool,
    ) -> RunRecord:
        """Run until approval, cancellation, or a terminal state."""

    def cancel(self, run_id: str) -> None:
        """Best-effort request to stop any active external resource for *run_id*."""


class RunService:
    """Durable single-worker service used by both Typer and FastAPI.

    The queue contains at most ``max_queue`` waiting runs. Only the worker thread
    invokes a workflow, so model calls and Docker verification never overlap.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        allowed_repo_roots: tuple[str | os.PathLike[str], ...],
        runner: DurableWorkflowRunner | None = None,
        max_queue: int = 4,
        start_worker: bool = True,
        allow_bootstrap: bool = False,
        review_enabled: bool = True,
        secrets: tuple[str, ...] = (),
    ) -> None:
        if type(max_queue) is not int or not 1 <= max_queue <= 64:
            raise ValueError("max_queue must be an integer from 1 to 64")
        if not allowed_repo_roots:
            raise ValueError("at least one allowed repository root is required")
        if type(allow_bootstrap) is not bool:
            raise ValueError("allow_bootstrap must be a boolean")
        if type(review_enabled) is not bool:
            raise ValueError("review_enabled must be a boolean")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_repo_roots = tuple(
            Path(root).expanduser().resolve(strict=True)
            for root in allowed_repo_roots
        )
        if any(not root.is_dir() for root in self.allowed_repo_roots):
            raise ValueError("allowed repository roots must be directories")
        lease = _DataDirectoryLease.acquire(self.data_dir)
        try:
            self.database = RunDatabase(self.data_dir / "runs.sqlite3")
            self.artifacts = ArtifactStore(self.data_dir / "runs", secrets=secrets)
            self._runner = (
                runner
                if runner is not None
                else _load_default_runner(
                    self,
                    allow_bootstrap=allow_bootstrap,
                    review_enabled=review_enabled,
                )
            )
            self._queue: queue.Queue[_QueuedRun] = queue.Queue()
            self._pending_slots = threading.BoundedSemaphore(max_queue)
            self._max_queue = max_queue
            self._scheduled: set[str] = set()
            self._cancel_events: dict[str, threading.Event] = {}
            self._lock = threading.RLock()
            self._publication_lock = threading.RLock()
            self._close_lock = threading.RLock()
            self._condition = threading.Condition(self._lock)
            self._stop_event = threading.Event()
            self._worker: threading.Thread | None = None
            self._lease: _DataDirectoryLease | None = lease
            self._publish_inflight_interrupted()
            if start_worker:
                self.start()
        except BaseException:
            lease.release()
            raise

    def __enter__(self) -> "RunService":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        self.close()
        return False

    def start(self) -> None:
        with self._close_lock:
            self._require_open()
            with self._lock:
                if self._worker is not None and self._worker.is_alive():
                    return
                self._worker = threading.Thread(
                    target=self._worker_main,
                    name="repo-agent-worker",
                    daemon=True,
                )
                self._worker.start()

    def close(self, *, timeout: float = 10.0) -> None:
        validated_timeout = _validated_non_negative_timeout(timeout)
        assert validated_timeout is not None
        with self._close_lock:
            lease = self._lease
            if lease is None or not lease.held:
                return

            self._stop_event.set()
            with self._lock:
                active_run_ids = tuple(self._cancel_events)
                for event in self._cancel_events.values():
                    event.set()
                worker = self._worker
                self._condition.notify_all()

            cancel = getattr(self._runner, "cancel", None)
            if callable(cancel):
                for run_id in active_run_ids:
                    try:
                        cancel(run_id)
                    except Exception:
                        # Cancellation is best effort; the cooperative event remains set.
                        pass

            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=validated_timeout)
            if worker is not None and worker.is_alive():
                raise TimeoutError(
                    "run service worker did not stop before the close timeout; "
                    "the data directory lease is still held"
                )

            try:
                self._publish_inflight_interrupted()
            finally:
                lease.release()
                self._lease = None

    def create_run(
        self,
        *,
        repo_path: str | os.PathLike[str],
        task: str,
        base_ref: str | None = None,
        auto_approve: bool = False,
        allow_remote_model: bool = False,
    ) -> RunRecord:
        with self._close_lock:
            self._require_open()
            return self._create_run(
                repo_path=repo_path,
                task=task,
                base_ref=base_ref,
                auto_approve=auto_approve,
                allow_remote_model=allow_remote_model,
            )

    def _create_run(
        self,
        *,
        repo_path: str | os.PathLike[str],
        task: str,
        base_ref: str | None,
        auto_approve: bool,
        allow_remote_model: bool,
    ) -> RunRecord:
        repository = self._validated_repo_path(repo_path)
        normalized_task = _validated_task(task)
        normalized_ref = _validated_base_ref(base_ref)
        if type(auto_approve) is not bool or type(allow_remote_model) is not bool:
            raise ValueError("approval and remote-model flags must be boolean")
        if not self._pending_slots.acquire(blocking=False):
            raise QueueFullError(f"run queue is full (maximum {self._max_queue})")

        run_id = uuid.uuid4().hex
        now = utc_now()
        record = RunRecord(
            run_id=run_id,
            repo_path=str(repository),
            task=normalized_task,
            base_ref=normalized_ref,
            status="queued",
            auto_approve=auto_approve,
            allow_remote_model=allow_remote_model,
            created_at=now,
            updated_at=now,
        )
        try:
            artifact_dir = self.artifacts.initialize(run_id)
            self.database.create(record, artifact_dir)
            self.artifacts.write_record(record)
            self._trace(record, "run_created", detail={"status": "queued"})
            self._schedule(run_id, resume=False, slot_reserved=True)
        except BaseException:
            self._pending_slots.release()
            raise
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        record = self.database.get(_validated_run_id(run_id))
        if not self._record_is_allowed(record):
            raise RepositoryNotAllowedError(
                "run repository is outside the configured allowed roots"
            )
        return record

    def list_runs(self) -> tuple[RunRecord, ...]:
        return tuple(
            record for record in self.database.list() if self._record_is_allowed(record)
        )

    def decide(
        self,
        run_id: str,
        *,
        approve: bool,
        reason: str | None = None,
    ) -> RunRecord:
        with self._close_lock:
            self._require_open()
            return self._decide(run_id, approve=approve, reason=reason)

    def _decide(
        self,
        run_id: str,
        *,
        approve: bool,
        reason: str | None,
    ) -> RunRecord:
        if type(approve) is not bool:
            raise ValueError("approve must be a boolean")
        normalized_reason = _validated_reason(reason)
        with self._publication_lock:
            record = self.get_run(run_id)
            if record.status != "awaiting_approval":
                raise InvalidRunTransitionError(
                    f"run {record.run_id} is {record.status}, not awaiting approval"
                )
            now = utc_now()
            if not approve:
                rejected = _validated_copy(
                    record,
                    status="rejected",
                    approval_reason=normalized_reason,
                    updated_at=now,
                    finished_at=now,
                )
                self._publish_stopped(rejected, "plan_rejected")
                return rejected

            queued = _validated_copy(
                record,
                status="queued",
                approval_reason=normalized_reason,
                updated_at=now,
            )
            self.database.put(queued)
            try:
                self._schedule(queued.run_id, resume=True)
            except BaseException:
                self.database.put(record)
                raise
            self._trace(queued, "plan_approved", detail={"status": "queued"})
            self.artifacts.write_record(queued)
            return queued

    def resume(self, run_id: str) -> RunRecord:
        with self._close_lock:
            self._require_open()
            return self._resume(run_id)

    def _resume(self, run_id: str) -> RunRecord:
        with self._publication_lock:
            record = self.get_run(run_id)
            if record.status != "interrupted":
                raise InvalidRunTransitionError(
                    f"run {record.run_id} is {record.status}, not interrupted"
                )
            queued = _validated_copy(
                record,
                status="queued",
                error=None,
                cancel_requested=False,
                updated_at=utc_now(),
            )
            self.database.put(queued)
            try:
                self._schedule(queued.run_id, resume=True)
            except BaseException:
                self.database.put(record)
                raise
            self._trace(queued, "run_resumed", detail={"status": "queued"})
            self.artifacts.write_record(queued)
            return queued

    def cancel(self, run_id: str) -> RunRecord:
        with self._close_lock:
            self._require_open()
            return self._cancel(run_id)

    def _cancel(self, run_id: str) -> RunRecord:
        with self._publication_lock:
            record = self.get_run(run_id)
            if record.status in TERMINAL_STATUSES:
                return record
            now = utc_now()
            active = record.status in {"planning", "running"}
            cancelled = _validated_copy(
                record,
                status=record.status if active else "cancelled",
                cancel_requested=True,
                updated_at=now,
                finished_at=None if active else now,
            )
            with self._lock:
                event = self._cancel_events.get(record.run_id)
                if event is not None:
                    event.set()
            if active:
                self.database.put(cancelled)
                self._trace(cancelled, "cancel_requested", detail={"active": True})
                self.artifacts.write_record(cancelled)
                self._notify(record.run_id)
            else:
                self._publish_stopped(
                    cancelled,
                    "cancel_requested",
                    detail={"active": False},
                )
        cancel = getattr(self._runner, "cancel", None)
        if callable(cancel):
            cancel(record.run_id)
        return cancelled

    def artifact_path(self, run_id: str, kind: str) -> Path:
        self.get_run(run_id)
        return self.artifacts.path(_validated_run_id(run_id), kind)  # type: ignore[arg-type]

    def wait(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
        stop_at_approval: bool = True,
    ) -> RunRecord:
        validated_timeout = _validated_non_negative_timeout(
            timeout, optional=True
        )
        if type(stop_at_approval) is not bool:
            raise ValueError("stop_at_approval must be a boolean")
        run_id = _validated_run_id(run_id)
        deadline = (
            None
            if validated_timeout is None
            else time.monotonic() + validated_timeout
        )
        with self._condition:
            while True:
                record = self.get_run(run_id)
                if (
                    record.status in TERMINAL_STATUSES
                    or record.status == "interrupted"
                    or (stop_at_approval and record.status == "awaiting_approval")
                ):
                    return record
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return record
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)

    def ready(self) -> bool:
        worker = self._worker
        return (
            not self._stop_event.is_set()
            and worker is not None
            and worker.is_alive()
        )

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _require_open(self) -> None:
        lease = self._lease
        if self._stop_event.is_set() or lease is None or not lease.held:
            raise RuntimeError("RunService is closed")

    def _schedule(
        self, run_id: str, *, resume: bool, slot_reserved: bool = False
    ) -> None:
        with self._close_lock:
            self._require_open()
            self._schedule_open(run_id, resume=resume, slot_reserved=slot_reserved)

    def _schedule_open(
        self, run_id: str, *, resume: bool, slot_reserved: bool
    ) -> None:
        acquired = slot_reserved or self._pending_slots.acquire(blocking=False)
        if not acquired:
            raise QueueFullError(f"run queue is full (maximum {self._max_queue})")
        try:
            with self._lock:
                if run_id in self._scheduled:
                    raise InvalidRunTransitionError(f"run is already scheduled: {run_id}")
                self._scheduled.add(run_id)
                self._queue.put_nowait(_QueuedRun(run_id, resume))
                self._condition.notify_all()
        except BaseException:
            if not slot_reserved:
                self._pending_slots.release()
            raise

    def _worker_main(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._pending_slots.release()
            with self._lock:
                self._scheduled.discard(job.run_id)
                cancel_event = self._cancel_events.setdefault(
                    job.run_id, threading.Event()
                )
            try:
                if self._stop_event.is_set():
                    cancel_event.set()
                else:
                    self._execute_job(job, cancel_event)
            finally:
                with self._lock:
                    self._cancel_events.pop(job.run_id, None)
                    self._condition.notify_all()
                self._queue.task_done()

    def _execute_job(self, job: _QueuedRun, cancel_event: threading.Event) -> None:
        with self._publication_lock:
            try:
                record = self.get_run(job.run_id)
            except RunNotFoundError:
                return
            if record.status in TERMINAL_STATUSES:
                return
            if record.cancel_requested:
                self._finish_cancelled(record)
                return

            started = record.started_at or utc_now()
            running = _validated_copy(
                record,
                status="running" if job.resume else "planning",
                started_at=started,
                updated_at=utc_now(),
            )
            self.database.put(running)
            self._trace(
                running,
                "workflow_resumed" if job.resume else "workflow_started",
                detail={"status": running.status},
            )
            self.artifacts.write_record(running)
            self._notify(job.run_id)
        context = RunExecutionContext(
            database=self.database,
            artifacts=self.artifacts,
            run_id=job.run_id,
            cancel_event=cancel_event,
            publication_lock=self._publication_lock,
            notify=self._notify,
        )
        try:
            outcome = self._runner.execute(running, context, resume=job.resume)
            if not isinstance(outcome, RunRecord) or outcome.run_id != job.run_id:
                raise TypeError("workflow returned an unsupported run record")
            with self._publication_lock:
                latest = self.get_run(job.run_id)
                if cancel_event.is_set() or latest.cancel_requested:
                    self._finish_cancelled(latest)
                    return
                if (
                    outcome.status not in TERMINAL_STATUSES
                    and outcome.status != "awaiting_approval"
                ):
                    raise RuntimeError(
                        f"workflow stopped in invalid public state: {outcome.status}"
                    )
                final = _validated_copy(
                    outcome,
                    updated_at=utc_now(),
                    finished_at=(
                        utc_now() if outcome.status in TERMINAL_STATUSES else None
                    ),
                )
                self._publish_stopped(
                    final,
                    "workflow_stopped",
                    detail={"status": final.status},
                )
        except BaseException as exc:
            with self._publication_lock:
                latest = self.get_run(job.run_id)
                if cancel_event.is_set() or latest.cancel_requested:
                    self._finish_cancelled(latest)
                else:
                    now = utc_now()
                    failed = _validated_copy(
                        latest,
                        status="failed",
                        error=_exception_detail(exc),
                        updated_at=now,
                        finished_at=now,
                    )
                    self._publish_stopped(
                        failed,
                        "workflow_failed",
                        detail={"error": failed.error or "workflow failed"},
                    )
        finally:
            self._notify(job.run_id)

    def _finish_cancelled(self, record: RunRecord) -> RunRecord:
        now = utc_now()
        cancelled = _validated_copy(
            record,
            status="cancelled",
            cancel_requested=True,
            updated_at=now,
            finished_at=now,
        )
        self._publish_stopped(cancelled, "workflow_cancelled")
        return cancelled

    def _publish_inflight_interrupted(self) -> int:
        """Publish recoverable shutdown state without exposing stale artifacts."""

        with self._publication_lock:
            records = self.database.list(statuses=("queued", "planning", "running"))
            for record in records:
                interrupted = _validated_copy(
                    record,
                    status="interrupted",
                    updated_at=utc_now(),
                    error=(
                        "Service stopped before the workflow reached a pause or "
                        "terminal state"
                    ),
                )
                self._publish_stopped(
                    interrupted,
                    "service_interrupted",
                    detail={
                        "previous_status": record.status,
                        "status": "interrupted",
                    },
                )
            return len(records)

    def _publish_stopped(
        self,
        record: RunRecord,
        event: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Publish artifacts before making a stopped state externally visible."""

        with self._publication_lock:
            self.artifacts.write_record(record)
            if record.status in TERMINAL_STATUSES:
                self._write_report(record)
            self._trace(
                record,
                event,
                detail=detail if detail is not None else {"status": record.status},
            )
            self.database.put(record)
            self._notify(record.run_id)

    def _trace(
        self, record: RunRecord, event: str, *, detail: dict[str, Any]
    ) -> None:
        sequence = self.database.append_event(
            record.run_id,
            {"event": event, "detail": detail, "timestamp": utc_now()},
        )
        self.artifacts.append_trace(
            record.run_id,
            {
                "sequence": sequence,
                "run_id": record.run_id,
                "event": event,
                "node": record.current_node,
                "detail": detail,
            },
        )

    def _write_report(self, record: RunRecord) -> None:
        lines = [
            "# Repo Maintainer Agent run",
            "",
            f"- Run: `{record.run_id}`",
            f"- Status: `{record.status}`",
            f"- Task: {record.task}",
            f"- Repository snapshot source: `{record.repo_path}`",
            "",
        ]
        if record.plan is not None:
            lines.extend(("## Change plan", "", record.plan.goal, ""))
            lines.extend(f"- {step}" for step in record.plan.steps)
            lines.append("")
        lines.extend(("## Verification", ""))
        if record.checks:
            lines.extend(
                f"- Attempt {check.attempt}: `{check.status}` ({check.duration_ms} ms)"
                for check in record.checks
            )
        else:
            lines.append("No verification result was recorded.")
        if record.error:
            lines.extend(("", "## Error", "", record.error))
        lines.extend(
            (
                "",
                "The patch is an artifact for human review. It was not applied, committed, or pushed to the source repository.",
                "",
            )
        )
        self.artifacts.write_report(record.run_id, "\n".join(lines))

    def _notify(self, run_id: str) -> None:
        del run_id
        with self._condition:
            self._condition.notify_all()

    def _validated_repo_path(self, value: str | os.PathLike[str]) -> Path:
        repository = Path(value).expanduser().resolve(strict=True)
        if not repository.is_dir():
            raise ValueError("repo_path must be a directory")
        for root in self.allowed_repo_roots:
            try:
                repository.relative_to(root)
            except ValueError:
                continue
            return repository
        raise RepositoryNotAllowedError(
            "repository path is outside the configured allowed roots"
        )

    def _record_is_allowed(self, record: RunRecord) -> bool:
        try:
            repository = Path(record.repo_path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return False
        return any(
            repository == root or root in repository.parents
            for root in self.allowed_repo_roots
        )


class _MissingWorkflowRunner:
    def execute(
        self,
        record: RunRecord,
        context: RunExecutionContext,
        *,
        resume: bool,
    ) -> RunRecord:
        del context, resume
        raise RuntimeError("the durable workflow runner is unavailable")

    def cancel(self, run_id: str) -> None:
        del run_id


def _load_default_runner(
    service: RunService,
    *,
    allow_bootstrap: bool = False,
    review_enabled: bool = True,
) -> DurableWorkflowRunner:
    try:
        from .workflow_runtime import ServiceWorkflowRunner
    except (ImportError, AttributeError):
        return _MissingWorkflowRunner()
    return ServiceWorkflowRunner(
        service.database,
        service.artifacts,
        allow_bootstrap=allow_bootstrap,
        review_enabled=review_enabled,
    )


def _validated_copy(record: RunRecord, **updates: Any) -> RunRecord:
    payload = record.model_dump(mode="python")
    payload.update(updates)
    return RunRecord.model_validate(payload)


def _validated_task(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("task must contain non-whitespace text")
    normalized = value.strip()
    if len(normalized) > 4000:
        raise ValueError("task must be at most 4000 characters")
    return normalized


def _validated_base_ref(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_ref must contain non-whitespace text")
    normalized = value.strip()
    if (
        len(normalized) > 255
        or normalized.startswith("-")
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError("base_ref is invalid")
    return normalized


def _validated_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("reason must be text")
    normalized = value.strip()
    if len(normalized) > 2000:
        raise ValueError("reason must be at most 2000 characters")
    return normalized or None


def _validated_non_negative_timeout(
    value: object, *, optional: bool = False
) -> float | None:
    if value is None:
        if optional:
            return None
        raise ValueError("timeout must be a finite non-negative number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a finite non-negative number")
    try:
        timeout = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("timeout must be a finite non-negative number") from exc
    if (
        not math.isfinite(timeout)
        or timeout < 0
        or timeout > threading.TIMEOUT_MAX
    ):
        raise ValueError("timeout must be a finite non-negative number")
    return timeout


def _validated_run_id(value: object) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError("invalid run id")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("invalid run id") from exc
    if value != value.lower():
        raise ValueError("invalid run id")
    return value


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


__all__ = [
    "DataDirectoryInUseError",
    "DurableWorkflowRunner",
    "InvalidRunTransitionError",
    "QueueFullError",
    "RepositoryNotAllowedError",
    "RunExecutionContext",
    "RunService",
    "run_agent",
]
