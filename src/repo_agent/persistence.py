"""SQLite persistence for durable run metadata and service-owned state."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .run_models import RunRecord, utc_now


class RunNotFoundError(LookupError):
    """Raised when a run id is not present in the local database."""


class RunConflictError(RuntimeError):
    """Raised when a requested state transition is no longer valid."""


class RunDatabase:
    """Small connection-per-operation SQLite repository.

    Connections are deliberately short lived so CLI processes and the API worker can
    share one database without holding process-global SQLite objects.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=10,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    engine_state_json TEXT,
                    artifact_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS runs_updated_at_idx
                    ON runs(updated_at);

                CREATE TABLE IF NOT EXISTS service_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS side_effects (
                    run_id TEXT NOT NULL,
                    effect_key TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, effect_key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                """
            )

    def create(self, record: RunRecord, artifact_dir: str | Path) -> None:
        artifact = str(Path(artifact_dir).resolve())
        payload = record.model_dump_json()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, record_json, engine_state_json, artifact_dir,
                        created_at, updated_at
                    ) VALUES (?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        payload,
                        artifact,
                        record.created_at,
                        record.updated_at,
                    ),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise RunConflictError(f"run already exists: {record.run_id}") from exc

    def get(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return RunRecord.model_validate_json(row["record_json"])

    def artifact_dir(self, run_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT artifact_dir FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        return Path(row["artifact_dir"])

    def put(self, record: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE runs
                SET record_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (record.model_dump_json(), record.updated_at, record.run_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RunNotFoundError(f"run not found: {record.run_id}")
            connection.commit()

    def list(self, *, statuses: Iterable[str] | None = None) -> tuple[RunRecord, ...]:
        parameters: tuple[object, ...] = ()
        query = "SELECT record_json FROM runs"
        if statuses is not None:
            status_list = tuple(dict.fromkeys(statuses))
            if not status_list:
                return ()
            placeholders = ",".join("?" for _ in status_list)
            query += f" WHERE json_extract(record_json, '$.status') IN ({placeholders})"
            parameters = status_list
        query += " ORDER BY created_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(RunRecord.model_validate_json(row["record_json"]) for row in rows)

    def save_engine_state(self, run_id: str, state: dict[str, Any]) -> None:
        payload = json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE runs SET engine_state_json = ?, updated_at = ? WHERE run_id = ?",
                (payload, utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RunNotFoundError(f"run not found: {run_id}")
            connection.commit()

    def load_engine_state(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT engine_state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        raw = row["engine_state_json"]
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("stored engine state is not an object")
        return value

    def append_event(self, run_id: str, event: dict[str, Any]) -> int:
        payload = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                connection.rollback()
                raise RunNotFoundError(f"run not found: {run_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence "
                "FROM service_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row["sequence"])
            connection.execute(
                """
                INSERT INTO service_events(run_id, sequence, event_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, sequence, payload, utc_now()),
            )
            connection.commit()
        return sequence

    def events(self, run_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            rows = connection.execute(
                "SELECT event_json FROM service_events "
                "WHERE run_id = ? ORDER BY sequence ASC",
                (run_id,),
            ).fetchall()
        return tuple(json.loads(row["event_json"]) for row in rows)

    def record_side_effect(
        self, run_id: str, effect_key: str, result: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Atomically memoize a node side effect and return the stored result."""

        payload = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT result_json FROM side_effects "
                "WHERE run_id = ? AND effect_key = ?",
                (run_id, effect_key),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return False, json.loads(existing["result_json"])
            try:
                connection.execute(
                    """
                    INSERT INTO side_effects(run_id, effect_key, result_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, effect_key, payload, utc_now()),
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise RunConflictError("could not record side effect") from exc
            connection.commit()
        return True, result

    def side_effect(self, run_id: str, effect_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM side_effects "
                "WHERE run_id = ? AND effect_key = ?",
                (run_id, effect_key),
            ).fetchone()
        return None if row is None else json.loads(row["result_json"])

    def mark_inflight_interrupted(self) -> int:
        records = self.list(statuses=("queued", "planning", "running"))
        count = 0
        for record in records:
            now = utc_now()
            self.put(
                record.model_copy(
                    update={
                        "status": "interrupted",
                        "updated_at": now,
                        "error": "Service stopped before the workflow reached a pause or terminal state",
                    }
                )
            )
            count += 1
        return count


__all__ = [
    "RunConflictError",
    "RunDatabase",
    "RunNotFoundError",
]
