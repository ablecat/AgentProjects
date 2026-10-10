from __future__ import annotations

from pathlib import Path

import pytest

from repo_agent.persistence import (
    MAX_EVENT_PAGE_SIZE,
    MAX_EVENT_SEQUENCE,
    RunConflictError,
    RunDatabase,
    RunNotFoundError,
)
from repo_agent.run_models import RunRecord, utc_now


def _record(run_id: str = "a" * 32, *, status: str = "queued") -> RunRecord:
    now = utc_now()
    return RunRecord(
        run_id=run_id,
        repo_path="D:/fixtures/repo",
        task="repair the boundary case",
        status=status,
        created_at=now,
        updated_at=now,
    )


def test_database_round_trips_record_state_events_and_effects(tmp_path: Path) -> None:
    database = RunDatabase(tmp_path / "state" / "runs.sqlite3")
    artifact_dir = tmp_path / "artifacts" / ("a" * 32)
    record = _record()

    database.create(record, artifact_dir)
    assert database.get(record.run_id) == record
    assert database.artifact_dir(record.run_id) == artifact_dir.resolve()

    database.save_engine_state(record.run_id, {"next_node": "prepare", "attempt": 0})
    assert database.load_engine_state(record.run_id) == {
        "next_node": "prepare",
        "attempt": 0,
    }

    assert database.append_event(record.run_id, {"event": "created"}) == 1
    assert database.append_event(record.run_id, {"event": "queued"}) == 2
    assert database.events(record.run_id) == (
        {"event": "created"},
        {"event": "queued"},
    )

    inserted, result = database.record_side_effect(
        record.run_id, "implement:0", {"patch_sha256": "abc"}
    )
    assert inserted is True
    assert result == {"patch_sha256": "abc"}
    inserted, result = database.record_side_effect(
        record.run_id, "implement:0", {"patch_sha256": "different"}
    )
    assert inserted is False
    assert result == {"patch_sha256": "abc"}


def test_database_event_page_uses_an_exclusive_durable_cursor(tmp_path: Path) -> None:
    database = RunDatabase(tmp_path / "runs.sqlite3")
    record = _record()
    database.create(record, tmp_path / "artifacts")
    database.append_event(record.run_id, {"event": "first", "sequence": 999})
    database.append_event(
        record.run_id,
        {"event": "second", "timestamp": "2026-09-11T00:00:00.000Z"},
    )
    database.append_event(record.run_id, {"event": "third"})

    first = database.event_page(record.run_id, after_sequence=0, limit=2)
    assert first.latest_sequence == 3
    assert [event["sequence"] for event in first.events] == [1, 2]
    assert first.events[0]["event"] == "first"
    assert first.events[0]["timestamp"].endswith("Z")

    second = database.event_page(record.run_id, after_sequence=2, limit=2)
    assert second.latest_sequence == 3
    assert len(second.events) == 1
    assert second.events[0]["event"] == "third"
    assert second.events[0]["sequence"] == 3
    assert second.events[0]["timestamp"].endswith("Z")
    assert database.event_page(record.run_id, after_sequence=3).events == ()


@pytest.mark.parametrize(
    ("after_sequence", "limit"),
    [
        (-1, 1),
        (True, 1),
        (MAX_EVENT_SEQUENCE + 1, 1),
        (0, 0),
        (0, True),
        (0, MAX_EVENT_PAGE_SIZE + 1),
    ],
)
def test_database_event_page_rejects_invalid_bounds(
    tmp_path: Path, after_sequence: object, limit: object
) -> None:
    database = RunDatabase(tmp_path / "runs.sqlite3")
    record = _record()
    database.create(record, tmp_path / "artifacts")

    with pytest.raises(ValueError):
        database.event_page(
            record.run_id,
            after_sequence=after_sequence,  # type: ignore[arg-type]
            limit=limit,  # type: ignore[arg-type]
        )


def test_database_event_page_rejects_unknown_run(tmp_path: Path) -> None:
    database = RunDatabase(tmp_path / "runs.sqlite3")

    with pytest.raises(RunNotFoundError, match="not found"):
        database.event_page("b" * 32, after_sequence=0)


def test_database_rejects_duplicate_and_unknown_runs(tmp_path: Path) -> None:
    database = RunDatabase(tmp_path / "runs.sqlite3")
    record = _record()
    database.create(record, tmp_path / "artifacts")

    with pytest.raises(RunConflictError, match="already exists"):
        database.create(record, tmp_path / "artifacts")
    with pytest.raises(RunNotFoundError, match="not found"):
        database.get("b" * 32)


def test_database_marks_only_inflight_runs_interrupted(tmp_path: Path) -> None:
    database = RunDatabase(tmp_path / "runs.sqlite3")
    for index, status in enumerate(
        ("queued", "planning", "running", "awaiting_approval", "succeeded")
    ):
        run_id = f"{index + 1:032x}"
        database.create(_record(run_id, status=status), tmp_path / run_id)

    assert database.mark_inflight_interrupted() == 3
    statuses = {record.run_id: record.status for record in database.list()}
    assert list(statuses.values()) == [
        "interrupted",
        "interrupted",
        "interrupted",
        "awaiting_approval",
        "succeeded",
    ]
