"""SQLite persistence for durable repository-agent workflows."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Literal

from .run_models import PublicRunStatus, RunRecord, WorkflowNode
from .workflow import (
    CheckpointPhase,
    ConcurrentUpdateError,
    DuplicateRunError,
    RunNotFoundError,
    WorkflowCheckpoint,
    WorkflowError,
    WorkflowSnapshot,
)


_SCHEMA_VERSION = 1
_DEFAULT_BUSY_TIMEOUT_MS = 5_000


class CheckpointStoreClosedError(WorkflowError):
    """The SQLite store was used after it had been closed."""


class SQLiteCheckpointStore:
    """Transactional checkpoint store with compare-and-set run updates.

    A node checkpoint and its new run cursor are committed in one transaction.
    Consequently, a completed checkpoint can never exist without the cursor
    that skips that completed side effect during resume.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
            raise ValueError("busy_timeout_ms must be between 1 and 60000")
        database_text = os.fspath(database)
        if not database_text:
            raise ValueError("database path must not be empty")
        if database_text != ":memory:":
            path = Path(database_text)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._database = database_text
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            database_text,
            timeout=busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            if database_text != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_schema()
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    @property
    def database(self) -> str:
        return self._database

    def __enter__(self) -> SQLiteCheckpointStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        self.close()
        return False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def create_run(
        self, record: RunRecord, *, next_node: WorkflowNode
    ) -> WorkflowSnapshot:
        """Create a version-zero run without executing its first node."""

        snapshot = WorkflowSnapshot(record=record, next_node=next_node, version=0)
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO workflow_runs (
                        run_id, version, record_json, next_node, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        snapshot.version,
                        record.model_dump_json(),
                        next_node,
                        record.created_at,
                        record.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateRunError(f"run {record.run_id} already exists") from exc
        return snapshot

    def load_run(self, run_id: str) -> WorkflowSnapshot:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT version, record_json, next_node
                FROM workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run {run_id} does not exist")
        return _snapshot_from_row(row)

    def get_run(self, run_id: str) -> RunRecord:
        return self.load_run(run_id).record

    def list_runs(
        self,
        *,
        status: PublicRunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[RunRecord, ...]:
        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be non-negative")
        parameters: tuple[object, ...]
        if status is None:
            sql = """
                SELECT record_json
                FROM workflow_runs
                ORDER BY updated_at DESC, run_id ASC
                LIMIT ? OFFSET ?
            """
            parameters = (limit, offset)
        else:
            sql = """
                SELECT record_json
                FROM workflow_runs
                WHERE json_extract(record_json, '$.status') = ?
                ORDER BY updated_at DESC, run_id ASC
                LIMIT ? OFFSET ?
            """
            parameters = (status, limit, offset)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(sql, parameters).fetchall()
        return tuple(RunRecord.model_validate_json(row["record_json"]) for row in rows)

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
    ) -> WorkflowSnapshot:
        """Atomically append a checkpoint and advance the current run snapshot."""

        if record.run_id != snapshot.record.run_id:
            raise ValueError("checkpoint record run_id does not match snapshot")
        # Validate all caller-controlled data before opening a write transaction.
        candidate = WorkflowCheckpoint(
            sequence=1,
            run_id=record.run_id,
            node=node,
            attempt=attempt,
            phase=phase,
            idempotency_key=idempotency_key,
            result=result,
            record=record,
            created_at=created_at,
        )
        new_snapshot = WorkflowSnapshot(
            record=record,
            next_node=next_node,
            version=snapshot.version + 1,
        )
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT version FROM workflow_runs WHERE run_id = ?",
                (record.run_id,),
            ).fetchone()
            if current is None:
                raise RunNotFoundError(f"run {record.run_id} does not exist")
            current_version = int(current["version"])
            if current_version != snapshot.version:
                raise ConcurrentUpdateError(
                    f"run {record.run_id} expected version {snapshot.version}, "
                    f"found {current_version}"
                )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM workflow_checkpoints
                WHERE run_id = ?
                """,
                (record.run_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            checkpoint = candidate.model_copy(update={"sequence": sequence})
            try:
                connection.execute(
                    """
                    INSERT INTO workflow_checkpoints (
                        run_id, sequence, node, attempt, phase, idempotency_key,
                        result_json, record_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.run_id,
                        checkpoint.sequence,
                        checkpoint.node,
                        checkpoint.attempt,
                        checkpoint.phase,
                        checkpoint.idempotency_key,
                        _json_dump(checkpoint.result),
                        checkpoint.record.model_dump_json(),
                        checkpoint.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConcurrentUpdateError(
                    "checkpoint identity was already committed by another worker"
                ) from exc
            changed = connection.execute(
                """
                UPDATE workflow_runs
                SET version = ?, record_json = ?, next_node = ?, updated_at = ?
                WHERE run_id = ? AND version = ?
                """,
                (
                    new_snapshot.version,
                    record.model_dump_json(),
                    next_node,
                    record.updated_at,
                    record.run_id,
                    snapshot.version,
                ),
            ).rowcount
            if changed != 1:
                raise ConcurrentUpdateError(
                    f"run {record.run_id} changed during checkpoint commit"
                )
        return new_snapshot

    def get_checkpoint(
        self,
        run_id: str,
        *,
        node: WorkflowNode,
        attempt: int,
        phase: CheckpointPhase,
    ) -> WorkflowCheckpoint | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT sequence, run_id, node, attempt, phase, idempotency_key,
                       result_json, record_json, created_at
                FROM workflow_checkpoints
                WHERE run_id = ? AND node = ? AND attempt = ? AND phase = ?
                """,
                (run_id, node, attempt, phase),
            ).fetchone()
        if row is None:
            return None
        return _checkpoint_from_row(row)

    def list_checkpoints(self, run_id: str) -> tuple[WorkflowCheckpoint, ...]:
        # Distinguish an unknown run from a known run that has not started yet.
        self.load_run(run_id)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT sequence, run_id, node, attempt, phase, idempotency_key,
                       result_json, record_json, created_at
                FROM workflow_checkpoints
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
                (run_id,),
            ).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def _initialize_schema(self) -> None:
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, _SCHEMA_VERSION}:
                raise WorkflowError(f"unsupported checkpoint schema version {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 0),
                    record_json TEXT NOT NULL,
                    next_node TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 1),
                    node TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK (attempt >= 0),
                    phase TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id),
                    UNIQUE (run_id, sequence),
                    UNIQUE (run_id, node, attempt, phase)
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_run_phase
                ON workflow_checkpoints(run_id, phase, sequence);
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._ensure_open()
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise CheckpointStoreClosedError("checkpoint store is closed")


def _snapshot_from_row(row: sqlite3.Row) -> WorkflowSnapshot:
    return WorkflowSnapshot(
        version=int(row["version"]),
        record=RunRecord.model_validate_json(row["record_json"]),
        next_node=row["next_node"],
    )


def _checkpoint_from_row(row: sqlite3.Row) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        sequence=int(row["sequence"]),
        run_id=row["run_id"],
        node=row["node"],
        attempt=int(row["attempt"]),
        phase=row["phase"],
        idempotency_key=row["idempotency_key"],
        result=json.loads(row["result_json"]),
        record=RunRecord.model_validate_json(row["record_json"]),
        created_at=row["created_at"],
    )


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = ["CheckpointStoreClosedError", "SQLiteCheckpointStore"]
