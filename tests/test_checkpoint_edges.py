from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from repo_agent.checkpoints import CheckpointStoreClosedError, SQLiteCheckpointStore
from repo_agent.run_models import RunRecord
from repo_agent.workflow import (
    ConcurrentUpdateError,
    RunNotFoundError,
    WorkflowError,
    WorkflowSnapshot,
)


def _record(
    repo: Path,
    *,
    run_id: str = "a" * 32,
    status: str = "queued",
    updated_at: str = "2026-09-11T00:00:00.000Z",
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        repo_path=str(repo),
        task=f"task-{run_id[0]}",
        status=status,
        created_at="2026-09-11T00:00:00.000Z",
        updated_at=updated_at,
    )


@pytest.mark.parametrize("busy_timeout", (True, 0, 60_001, 1.5))
def test_checkpoint_store_validates_busy_timeout(busy_timeout: object) -> None:
    with pytest.raises(ValueError, match="busy_timeout_ms"):
        SQLiteCheckpointStore(":memory:", busy_timeout_ms=busy_timeout)


def test_checkpoint_store_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SQLiteCheckpointStore("")


def test_memory_store_lists_filters_and_pages_runs(tmp_path: Path) -> None:
    first = _record(tmp_path, run_id="a" * 32, updated_at="2026-09-11T00:00:00.000Z")
    second = _record(
        tmp_path,
        run_id="b" * 32,
        status="awaiting_approval",
        updated_at="2026-09-11T00:00:01.000Z",
    )
    with SQLiteCheckpointStore(":memory:") as store:
        assert store.database == ":memory:"
        store.create_run(first, next_node="prepare")
        store.create_run(second, next_node="approval")

        assert store.get_run(first.run_id) == first
        assert store.list_runs(limit=1) == (second,)
        assert store.list_runs(limit=1, offset=1) == (first,)
        assert store.list_runs(status="awaiting_approval") == (second,)
        assert (
            store.get_checkpoint(
                first.run_id,
                node="prepare",
                attempt=0,
                phase="started",
            )
            is None
        )


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(True, 0), (0, 0), (501, 0), (1, True), (1, -1)],
)
def test_checkpoint_store_rejects_invalid_pagination(
    limit: object, offset: object
) -> None:
    with SQLiteCheckpointStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.list_runs(limit=limit, offset=offset)


def test_checkpoint_save_validates_run_identity_and_existence(tmp_path: Path) -> None:
    first = _record(tmp_path)
    other = _record(tmp_path, run_id="b" * 32)
    with SQLiteCheckpointStore(":memory:") as store:
        snapshot = store.create_run(first, next_node="prepare")
        with pytest.raises(ValueError, match="run_id does not match"):
            store.save_checkpoint(
                snapshot,
                node="prepare",
                attempt=0,
                phase="started",
                idempotency_key="mismatch",
                result={},
                record=other,
                next_node="prepare",
                created_at=other.updated_at,
            )

        unknown_snapshot = WorkflowSnapshot(
            record=other,
            next_node="prepare",
            version=0,
        )
        with pytest.raises(RunNotFoundError):
            store.save_checkpoint(
                unknown_snapshot,
                node="prepare",
                attempt=0,
                phase="started",
                idempotency_key="unknown",
                result={},
                record=other,
                next_node="prepare",
                created_at=other.updated_at,
            )


def test_duplicate_checkpoint_identity_rolls_back(tmp_path: Path) -> None:
    record = _record(tmp_path)
    with SQLiteCheckpointStore(":memory:") as store:
        original = store.create_run(record, next_node="prepare")
        current = store.save_checkpoint(
            original,
            node="prepare",
            attempt=0,
            phase="started",
            idempotency_key="first",
            result={"started": True},
            record=record,
            next_node="prepare",
            created_at=record.updated_at,
        )

        with pytest.raises(ConcurrentUpdateError, match="already committed"):
            store.save_checkpoint(
                current,
                node="prepare",
                attempt=0,
                phase="started",
                idempotency_key="duplicate",
                result={"started": True},
                record=record,
                next_node="prepare",
                created_at=record.updated_at,
            )

        assert store.load_run(record.run_id) == current
        assert len(store.list_checkpoints(record.run_id)) == 1


def test_checkpoint_cursor_update_guard_rolls_back_insert(tmp_path: Path) -> None:
    record = _record(tmp_path)
    with SQLiteCheckpointStore(":memory:") as store:
        snapshot = store.create_run(record, next_node="prepare")
        store._connection.execute(
            """
            CREATE TRIGGER ignore_workflow_cursor_update
            BEFORE UPDATE ON workflow_runs
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )

        with pytest.raises(ConcurrentUpdateError, match="changed during"):
            store.save_checkpoint(
                snapshot,
                node="prepare",
                attempt=0,
                phase="completed",
                idempotency_key="guard",
                result={"ok": True},
                record=record,
                next_node="baseline_check",
                created_at=record.updated_at,
            )

        assert store.load_run(record.run_id) == snapshot
        assert store.list_checkpoints(record.run_id) == ()


def test_unknown_schema_closes_failed_store_initialization(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(WorkflowError, match="unsupported checkpoint schema"):
        SQLiteCheckpointStore(database)


def test_closed_store_rejects_context_and_operations() -> None:
    store = SQLiteCheckpointStore(":memory:")
    store.close()
    store.close()

    with pytest.raises(CheckpointStoreClosedError):
        store.__enter__()
    with pytest.raises(CheckpointStoreClosedError):
        store.load_run("a" * 32)
