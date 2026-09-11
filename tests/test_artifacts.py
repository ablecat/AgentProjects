from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

import repo_agent.artifacts as artifacts_module
from repo_agent.artifacts import ArtifactError, ArtifactStore, redact_text, redact_value
from repo_agent.run_models import ARTIFACT_FILENAMES, RunRecord, utc_now


RUN_ID = "d" * 32


def _record() -> RunRecord:
    now = utc_now()
    return RunRecord(
        run_id=RUN_ID,
        repo_path="D:/fixtures/repo",
        task="repair the parser",
        status="queued",
        created_at=now,
        updated_at=now,
    )


def test_artifact_store_creates_fixed_layout_and_writes_json(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run_dir = store.initialize(RUN_ID)

    assert {path.name for path in run_dir.iterdir()} == set(
        ARTIFACT_FILENAMES.values()
    )
    assert store.path(RUN_ID, "checks").is_dir()

    store.write_record(_record())
    payload = json.loads(store.path(RUN_ID, "result").read_text(encoding="utf-8"))
    assert payload["run_id"] == RUN_ID
    assert payload["status"] == "queued"


def test_artifact_store_redacts_trace_report_patch_and_checks(tmp_path: Path) -> None:
    secret = "repo-agent-private-value"
    store = ArtifactStore(tmp_path, secrets=(secret,))
    store.initialize(RUN_ID)

    store.write_report(RUN_ID, f"Bearer abc.def and {secret}")
    store.write_patch(RUN_ID, "+api_key=sk-abcdefghijklmnop")
    store.write_check(RUN_ID, "verify-0.log", "password=hunter2")
    store.append_trace(
        RUN_ID,
        {"event": "provider", "authorization": "Bearer abc.def", "value": secret},
    )

    combined = "\n".join(
        (
            store.path(RUN_ID, "report").read_text(encoding="utf-8"),
            store.path(RUN_ID, "patch").read_text(encoding="utf-8"),
            (store.path(RUN_ID, "checks") / "verify-0.log").read_text(
                encoding="utf-8"
            ),
            store.path(RUN_ID, "trace").read_text(encoding="utf-8"),
        )
    )
    assert secret not in combined
    assert "hunter2" not in combined
    assert "sk-abcdefghijklmnop" not in combined
    assert "abc.def" not in combined
    assert "[REDACTED]" in combined


def test_redact_value_masks_secret_named_fields_recursively() -> None:
    assert redact_value(
        {"nested": [{"apiToken": "visible-looking"}], "normal": "ok"}
    ) == {"nested": [{"apiToken": "[REDACTED]"}], "normal": "ok"}


@pytest.mark.parametrize("name", ["../escape.log", "bad/name.log", "plain.txt"])
def test_artifact_store_rejects_unsafe_check_names(tmp_path: Path, name: str) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    with pytest.raises(ArtifactError, match="invalid check"):
        store.write_check(RUN_ID, name, "output")


def test_artifact_store_rejects_unknown_kind_and_run_id(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="invalid run id"):
        store.initialize("../escape")
    store.initialize(RUN_ID)
    with pytest.raises(ArtifactError, match="unknown artifact"):
        store.path(RUN_ID, "unknown")  # type: ignore[arg-type]


def test_redact_text_preserves_non_secret_diagnostics() -> None:
    assert redact_text("pytest: 12 passed") == "pytest: 12 passed"


def test_trace_append_atomically_replaces_a_complete_jsonl_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    store.append_trace(RUN_ID, {"event": "first"})
    trace_path = store.path(RUN_ID, "trace")
    original_snapshot = trace_path.read_bytes()
    replace_started = threading.Event()
    allow_replace = threading.Event()
    append_errors: list[BaseException] = []
    original_replace = artifacts_module.os.replace

    def blocking_replace(source, target) -> None:
        if Path(target) == trace_path:
            replace_started.set()
            assert allow_replace.wait(timeout=5)
        original_replace(source, target)

    monkeypatch.setattr(artifacts_module.os, "replace", blocking_replace)

    def append() -> None:
        try:
            store.append_trace(RUN_ID, {"event": "second", "detail": "x" * 100_000})
        except BaseException as exc:
            append_errors.append(exc)

    thread = threading.Thread(target=append)
    thread.start()
    try:
        assert replace_started.wait(timeout=5)
        assert trace_path.read_bytes() == original_snapshot
        assert [json.loads(line)["event"] for line in original_snapshot.splitlines()] == [
            "first"
        ]
    finally:
        allow_replace.set()
        thread.join(timeout=5)

    assert thread.is_alive() is False
    assert append_errors == []
    assert [
        json.loads(line)["event"] for line in trace_path.read_bytes().splitlines()
    ] == ["first", "second"]
