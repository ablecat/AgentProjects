from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

import repo_agent.artifacts as artifacts_module
from repo_agent.artifacts import ArtifactError, ArtifactStore, redact_text, redact_value
from repo_agent.run_models import ARTIFACT_FILENAMES, RunRecord, utc_now


RUN_ID = "d" * 32


def _patch_adding(statement: str) -> str:
    return (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -0,0 +1 @@\n"
        f"+{statement}\n"
    )


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


def test_artifact_store_redacts_trace_report_and_checks_and_rejects_secret_patch(
    tmp_path: Path,
) -> None:
    secret = "repo-agent-private-value"
    store = ArtifactStore(tmp_path, secrets=(secret,))
    store.initialize(RUN_ID)

    store.write_report(RUN_ID, f"Bearer abc.def and {secret}")
    with pytest.raises(ArtifactError, match="credential-like"):
        store.write_patch(RUN_ID, "+api_key=sk-abcdefghijklmnop")
    with pytest.raises(ArtifactError, match="credential-like"):
        store.write_patch(RUN_ID, f"+value={secret}")
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


def test_artifact_store_preserves_non_secret_patch_bytes_exactly(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, secrets=("actual-secret-value",))
    store.initialize(RUN_ID)
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-api_key = None\n"
        '+api_key = "placeholder"\n'
    )

    store.write_patch(RUN_ID, patch)

    assert store.path(RUN_ID, "patch").read_bytes() == patch.encode("utf-8")


def test_patch_credential_detection_handles_short_secrets_and_plain_passwords(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path, secrets=("a",))
    store.initialize(RUN_ID)
    benign = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        '-note = "old"\n'
        '+note = "fix a bug"\n'
    )

    store.write_patch(RUN_ID, benign)
    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == benign

    unsafe_values = (
        "hunter2",
        "hunter_2",
        'SecretStr("hunter2")',
        '{"raw":"hunter2"}',
        'os.getenv("PASSWORD","hunter2")',
    )
    for unsafe in unsafe_values:
        with pytest.raises(ArtifactError, match="credential-like"):
            store.write_patch(RUN_ID, f"+password={unsafe}\n")
        assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == benign

    with pytest.raises(ArtifactError, match="credential-like"):
        store.write_patch(RUN_ID, '+OPENAI_API_KEY="hunter2"\n')
    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == benign
    for assignment in (
        '+os.environ["OPENAI_API_KEY"] = "hunter2"\n',
        '+config["CLIENT_SECRET"] = "hunter2"\n',
        '+$env:API_KEY = "hunter2"\n',
    ):
        with pytest.raises(ArtifactError, match="credential-like"):
            store.write_patch(RUN_ID, assignment)
        assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == benign


@pytest.mark.parametrize(
    "safe_value",
    [
        '"placeholder"',
        "REPO_PASSWORD",
        'os.getenv("REPO_PASSWORD")',
        'os.environ["REPO_PASSWORD"]',
        "process.env.REPO_PASSWORD",
        "settings.password",
        "$env:REPO_PASSWORD",
        "${REPO_PASSWORD}",
    ],
)
def test_patch_credential_detection_allows_strict_references_and_placeholders(
    tmp_path: Path,
    safe_value: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    patch = f"+password={safe_value}\n"

    store.write_patch(RUN_ID, patch)

    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == patch


@pytest.mark.parametrize("name", ["password", "api_key"])
@pytest.mark.parametrize("operator", ["==", "!=", "<=", ">=", "=>", ":="])
@pytest.mark.parametrize(
    "expression",
    [
        '{name} {operator} "hunter2"',
        '{name}: str {operator} "hunter2"',
    ],
)
def test_patch_credential_detection_does_not_treat_operators_as_assignment(
    tmp_path: Path,
    name: str,
    operator: str,
    expression: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    patch = _patch_adding(expression.format(name=name, operator=operator))

    store.write_patch(RUN_ID, patch)

    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == patch
    assert redact_text(patch) == patch


@pytest.mark.parametrize(
    "statement",
    [
        'password = "placeholder"',
        '"api_key": "<api-key>"',
        'password: str = "placeholder"',
        'config["api_key"] = os.environ["REPO_API_KEY"]',
        "$env:PASSWORD = $env:REPO_PASSWORD",
    ],
)
def test_patch_credential_detection_allows_placeholder_assignment_in_full_diff(
    tmp_path: Path,
    statement: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    patch = _patch_adding(statement)

    store.write_patch(RUN_ID, patch)

    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == patch


@pytest.mark.parametrize(
    "statement",
    [
        'password = "hunter2"',
        'api_key: "hunter2"',
        'password: str = "hunter2"',
        'config["api_key"] = "hunter2"',
        '$env:PASSWORD = "hunter2"',
    ],
)
def test_patch_credential_detection_rejects_assignment_in_full_diff(
    tmp_path: Path,
    statement: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)

    with pytest.raises(ArtifactError, match="credential-like"):
        store.write_patch(RUN_ID, _patch_adding(statement))

    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "statement",
    [
        '$password = "hunter2"',
        '$apiKey = "hunter2"',
        '${password} = "hunter2"',
        'os.putenv("API_KEY", "hunter2")',
        'os.environ.setdefault("password", "hunter2")',
    ],
)
def test_patch_credential_detection_rejects_common_secret_writes(
    tmp_path: Path,
    statement: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)

    with pytest.raises(ArtifactError, match="credential-like"):
        store.write_patch(RUN_ID, _patch_adding(statement))


@pytest.mark.parametrize(
    "statement",
    [
        "$password = $env:REPO_PASSWORD",
        '$apiKey = "placeholder"',
        "${password} = REPO_PASSWORD",
        'os.putenv("API_KEY", API_KEY)',
        'os.putenv("API_KEY", "placeholder")',
        'os.putenv("CACHE_KEY", "hunter2")',
        'os.environ.setdefault("password", os.getenv("PASSWORD"))',
        'os.environ.setdefault("password", "<password>")',
    ],
)
def test_patch_credential_detection_allows_references_in_common_secret_writes(
    tmp_path: Path,
    statement: str,
) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize(RUN_ID)
    patch = _patch_adding(statement)

    store.write_patch(RUN_ID, patch)

    assert store.path(RUN_ID, "patch").read_text(encoding="utf-8") == patch


def test_redact_text_masks_secret_literal_calls_but_preserves_placeholders() -> None:
    value = (
        'os.putenv("API_KEY", "hunter2")\n'
        'os.environ.setdefault("password", "placeholder")\n'
        'os.putenv("CACHE_KEY", "hunter2")'
    )

    assert redact_text(value) == (
        'os.putenv("API_KEY", [REDACTED])\n'
        'os.environ.setdefault("password", "placeholder")\n'
        'os.putenv("CACHE_KEY", "hunter2")'
    )


def test_redact_value_masks_secret_named_fields_recursively() -> None:
    assert redact_value(
        {
            "nested": [
                {"apiToken": "visible-looking"},
                {"OPENAI_API_KEY": "uppercase-secret"},
                {"CLIENT_SECRET": "client-secret"},
                {"REFRESH_TOKEN": "refresh-token"},
                {"PRIVATE_KEY": "private-key"},
            ],
            "normal": "ok",
        }
    ) == {
        "nested": [
            {"apiToken": "[REDACTED]"},
            {"OPENAI_API_KEY": "[REDACTED]"},
            {"CLIENT_SECRET": "[REDACTED]"},
            {"REFRESH_TOKEN": "[REDACTED]"},
            {"PRIVATE_KEY": "[REDACTED]"},
        ],
        "normal": "ok",
    }


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


def test_redact_text_ignores_ambiguous_short_secret_and_masks_full_assignment() -> None:
    value = (
        'pytest passed\npassword="correct horse battery staple"\n'
        '"api_key": "another secret with spaces"\n'
        "CLIENT_SECRET=hunter2\nREFRESH_TOKEN=hunter3\n"
        'os.environ["OPENAI_API_KEY"] = "hunter4"\n'
        '$env:API_KEY = "hunter5"\nnext=visible'
    )

    redacted = redact_text(value, secrets=("a",))

    assert redacted == (
        "pytest passed\npassword=[REDACTED]\n"
        '"api_key": [REDACTED]\nCLIENT_SECRET=[REDACTED]\n'
        'REFRESH_TOKEN=[REDACTED]\nos.environ["OPENAI_API_KEY"] = [REDACTED]\n'
        "$env:API_KEY = [REDACTED]\nnext=visible"
    )


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
