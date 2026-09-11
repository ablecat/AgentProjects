from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path

from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from repo_agent.api import DEFAULT_HOST, create_app
from repo_agent.artifacts import ArtifactError
from repo_agent.persistence import RunNotFoundError
from repo_agent.run_models import CheckSummary, RunRecord, utc_now
from repo_agent.service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
    RunService,
)


TOKEN = "test-api-token"
RUN_ID = "a" * 32


def _record(
    *,
    run_id: str = RUN_ID,
    repo_path: str,
    task: str = "Inspect repository",
    status: str = "queued",
) -> RunRecord:
    now = utc_now()
    return RunRecord(
        run_id=run_id,
        repo_path=repo_path,
        task=task,
        status=status,
        created_at=now,
        updated_at=now,
    )


class FakeRunService:
    def __init__(self, repo_path: Path, artifact: Path) -> None:
        self.repo_path = repo_path
        self.artifact = artifact
        self.calls: list[tuple[object, ...]] = []
        self.is_ready = True

    def create_run(
        self,
        *,
        repo_path: str,
        task: str,
        base_ref: str | None = None,
        auto_approve: bool = False,
        allow_remote_model: bool = False,
    ) -> RunRecord:
        self.calls.append(
            (
                "create",
                repo_path,
                task,
                base_ref,
                auto_approve,
                allow_remote_model,
            )
        )
        return _record(repo_path=repo_path, task=task)

    def get_run(self, run_id: str) -> RunRecord:
        self.calls.append(("get", run_id))
        return _record(repo_path=str(self.repo_path), status="awaiting_approval")

    def decide(
        self, run_id: str, *, approve: bool, reason: str | None = None
    ) -> RunRecord:
        self.calls.append(("decision", run_id, approve, reason))
        status = "running" if approve else "rejected"
        return _record(repo_path=str(self.repo_path), status=status)

    def resume(self, run_id: str) -> RunRecord:
        self.calls.append(("resume", run_id))
        return _record(repo_path=str(self.repo_path), status="running")

    def cancel(self, run_id: str) -> RunRecord:
        self.calls.append(("cancel", run_id))
        return _record(repo_path=str(self.repo_path), status="cancelled")

    def artifact_path(self, run_id: str, kind: str) -> Path:
        self.calls.append(("artifact", run_id, kind))
        return self.artifact

    def ready(self) -> bool:
        return self.is_ready


@pytest.fixture
def api_fixture(tmp_path: Path) -> tuple[FakeRunService, TestClient, Path]:
    allowed = tmp_path / "allowed"
    repository = allowed / "repository"
    repository.mkdir(parents=True)
    artifact = tmp_path / "patch.diff"
    artifact.write_bytes(b"diff --git a/app.py b/app.py\n")
    service = FakeRunService(repository.resolve(), artifact.resolve())
    app = create_app(
        service,
        bearer_token=TOKEN,
        allowed_roots=(allowed.resolve(),),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield service, client, repository.resolve()


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_is_public_and_readiness_tracks_service(api_fixture) -> None:
    service, client, _repository = api_fixture

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}

    service.is_ready = False
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_every_non_health_endpoint_requires_bearer_authentication(api_fixture) -> None:
    _service, client, repository = api_fixture

    for headers in ({}, _auth("wrong-token"), {"Authorization": "Basic abc"}):
        response = client.post(
            "/v1/runs",
            json={"repo_path": str(repository), "task": "inspect"},
            headers=headers,
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers["www-authenticate"] == "Bearer"
        assert TOKEN not in response.text

    assert client.get("/missing").status_code == 401


def test_create_returns_202_location_and_normalized_record(api_fixture) -> None:
    service, client, repository = api_fixture

    response = client.post(
        "/v1/runs",
        headers=_auth(),
        json={
            "repo_path": str(repository),
            "task": "  inspect repository  ",
            "base_ref": "HEAD~1",
            "auto_approve": True,
            "allow_remote_model": True,
        },
    )

    assert response.status_code == 202
    assert response.headers["location"] == f"/v1/runs/{RUN_ID}"
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["task"] == "inspect repository"
    assert service.calls == [
        ("create", str(repository), "inspect repository", "HEAD~1", True, True)
    ]


def test_get_decision_resume_and_cancel_delegate_to_service(api_fixture) -> None:
    service, client, _repository = api_fixture

    fetched = client.get(f"/v1/runs/{RUN_ID}", headers=_auth())
    decided = client.post(
        f"/v1/runs/{RUN_ID}/decision",
        headers=_auth(),
        json={"approve": False, "reason": "  unsafe change  "},
    )
    resumed = client.post(f"/v1/runs/{RUN_ID}/resume", headers=_auth())
    cancelled = client.post(f"/v1/runs/{RUN_ID}/cancel", headers=_auth())

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "awaiting_approval"
    assert decided.json()["status"] == "rejected"
    assert resumed.json()["status"] == "running"
    assert cancelled.json()["status"] == "cancelled"
    assert service.calls == [
        ("get", RUN_ID),
        ("decision", RUN_ID, False, "unsafe change"),
        ("resume", RUN_ID),
        ("cancel", RUN_ID),
    ]


def test_artifact_is_downloaded_from_service_owned_path(api_fixture) -> None:
    service, client, _repository = api_fixture

    response = client.get(
        f"/v1/runs/{RUN_ID}/artifacts/patch",
        headers=_auth(),
    )

    assert response.status_code == 200
    assert response.text == "diff --git a/app.py b/app.py\n"
    assert response.headers["content-type"].startswith("text/x-diff")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert service.calls == [("artifact", RUN_ID, "patch")]


def test_artifact_download_snapshots_atomic_file_before_sending_headers(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "run.json"
    original = b'{"status":"running"}\n'
    replacement = b'{"status":"succeeded","summary":"larger replacement"}\n'
    artifact.write_bytes(original)
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_bytes(replacement)
    service = FakeRunService(allowed, artifact)
    app = create_app(service, bearer_token=TOKEN, allowed_roots=allowed)
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.start":
            os.replace(replacement_path, artifact)
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/v1/runs/{RUN_ID}/artifacts/result",
        "raw_path": f"/v1/runs/{RUN_ID}/artifacts/result".encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode("ascii"))],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "extensions": {},
    }

    asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]

    start = next(message for message in messages if message["type"] == "http.response.start")
    headers = dict(start["headers"])  # type: ignore[arg-type]
    body = b"".join(
        message.get("body", b"")  # type: ignore[arg-type]
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert body == original
    assert int(headers[b"content-length"]) == len(original)
    assert artifact.read_bytes() == replacement


def test_check_artifacts_have_a_bounded_manifest_and_download_route(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "attempt-1.log").write_bytes(b"pytest passed\n")
    (checks / "README.txt").write_text("not an artifact", encoding="utf-8")
    service = FakeRunService(allowed, checks)
    app = create_app(service, bearer_token=TOKEN, allowed_roots=allowed)

    with TestClient(app) as client:
        manifest = client.get(
            f"/v1/runs/{RUN_ID}/artifacts/checks",
            headers=_auth(),
        )
        downloaded = client.get(
            f"/v1/runs/{RUN_ID}/artifacts/checks/attempt-1.log",
            headers=_auth(),
        )
        rejected = client.get(
            f"/v1/runs/{RUN_ID}/artifacts/checks/README.txt",
            headers=_auth(),
        )

    assert manifest.status_code == 200
    assert manifest.json() == {
        "run_id": RUN_ID,
        "kind": "checks",
        "artifacts": [
            {
                "name": "attempt-1.log",
                "size_bytes": 14,
                "href": f"/v1/runs/{RUN_ID}/artifacts/checks/attempt-1.log",
            }
        ],
        "truncated": False,
    }
    assert downloaded.status_code == 200
    assert downloaded.text == "pytest passed\n"
    assert rejected.status_code == 404
    assert rejected.json()["error"]["code"] == "artifact_not_found"


def test_run_payload_links_canonical_check_path_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "verify-0.log").write_bytes(b"verified\n")

    safe = CheckSummary(
        attempt=0,
        check_id="python-pytest",
        status="passed",
        ok=True,
        duration_ms=8,
        log_artifact="checks/verify-0.log",
    )
    unsafe = safe.model_copy(update={"log_artifact": "checks/../private.log"})

    class CheckService(FakeRunService):
        def get_run(self, run_id: str) -> RunRecord:
            self.calls.append(("get", run_id))
            return _record(repo_path=str(self.repo_path)).model_copy(
                update={"checks": (safe, unsafe,)}
            )

        def artifact_path(self, run_id: str, kind: str) -> Path:
            self.calls.append(("artifact", run_id, kind))
            return checks

    service = CheckService(allowed, checks)
    app = create_app(service, bearer_token=TOKEN, allowed_roots=allowed)
    with TestClient(app) as client:
        run = client.get(f"/v1/runs/{RUN_ID}", headers=_auth())
        href = run.json()["checks"][0]["log_artifact_href"]
        downloaded = client.get(href, headers=_auth())
        traversal = client.get(
            f"/v1/runs/{RUN_ID}/artifacts/checks/..%2Fprivate.log",
            headers=_auth(),
        )

    assert href == f"/v1/runs/{RUN_ID}/artifacts/checks/verify-0.log"
    assert downloaded.status_code == 200
    assert downloaded.content == b"verified\n"
    assert "log_artifact_href" not in run.json()["checks"][1]
    assert traversal.status_code == 404


def test_allowed_roots_reject_relative_missing_and_escaped_paths(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    service = FakeRunService(allowed, artifact)
    app = create_app(service, bearer_token=TOKEN, allowed_roots=(allowed,))

    with TestClient(app) as client:
        relative = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": "relative/repo", "task": "inspect"},
        )
        missing = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(allowed / "missing"), "task": "inspect"},
        )
        escaped = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(allowed / ".." / "outside"), "task": "inspect"},
        )

    assert relative.status_code == 400
    assert missing.status_code == 400
    assert escaped.status_code == 403
    assert escaped.json()["error"]["code"] == "repository_not_allowed"
    assert service.calls == []


def test_path_symlink_cannot_escape_allowed_root_when_supported(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "linked-repository"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    service = FakeRunService(allowed, artifact)
    app = create_app(service, bearer_token=TOKEN, allowed_roots=allowed)

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(link), "task": "inspect"},
        )

    assert response.status_code == 403
    assert service.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"repo_path": "x", "task": "inspect", "token": TOKEN},
        {"repo_path": "x", "task": ""},
        {"repo_path": "x", "task": "inspect", "auto_approve": "yes"},
        {"repo_path": "x", "task": "inspect", "base_ref": "--exec"},
    ],
)
def test_request_validation_is_structured_and_does_not_echo_input(
    api_fixture, payload
) -> None:
    _service, client, _repository = api_fixture

    response = client.post("/v1/runs", headers=_auth(), json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "Request validation failed",
        }
    }
    assert TOKEN not in response.text


def test_service_failures_are_structured_without_exception_or_token_leak(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")

    class BrokenService(FakeRunService):
        def create_run(self, *args, **kwargs):
            raise RuntimeError(f"database failed while using {TOKEN}")

    app = create_app(
        BrokenService(allowed, artifact),
        bearer_token=TOKEN,
        allowed_roots=allowed,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(allowed), "task": "inspect"},
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "service_error",
        "message": "The run service failed",
    }
    assert TOKEN not in response.text
    assert "database failed" not in response.text


def test_missing_runs_and_invalid_transitions_have_stable_errors(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")

    class ErrorService(FakeRunService):
        def get_run(self, run_id: str):
            raise RunNotFoundError(run_id)

        def resume(self, run_id: str):
            raise InvalidRunTransitionError(f"cannot resume {run_id}")

    app = create_app(
        ErrorService(allowed, artifact),
        bearer_token=TOKEN,
        allowed_roots=allowed,
    )
    with TestClient(app) as client:
        missing = client.get(f"/v1/runs/{RUN_ID}", headers=_auth())
        invalid = client.post(f"/v1/runs/{RUN_ID}/resume", headers=_auth())

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"
    assert invalid.status_code == 409
    assert invalid.json()["error"]["code"] == "invalid_run_state"


def test_service_domain_errors_map_without_leaking_details(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")

    class DomainErrorService(FakeRunService):
        def create_run(self, **_kwargs):
            raise RepositoryNotAllowedError("private service root")

        def artifact_path(self, run_id: str, kind: str):
            raise ArtifactError("private artifact location")

    app = create_app(
        DomainErrorService(allowed, artifact),
        bearer_token=TOKEN,
        allowed_roots=allowed,
    )
    with TestClient(app) as client:
        denied = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(allowed), "task": "inspect"},
        )
        missing = client.get(
            f"/v1/runs/{RUN_ID}/artifacts/patch",
            headers=_auth(),
        )

    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "repository_not_allowed"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "artifact_not_found"
    assert "private" not in denied.text + missing.text


def test_mapping_and_attribute_records_are_serialized_without_extra_fields(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    base = _record(repo_path=str(allowed)).model_dump(mode="python")

    @dataclass
    class AttributeRecord:
        schema_version: int
        run_id: str
        repo_path: str
        task: str
        base_ref: str | None
        base_commit: str | None
        status: str
        current_node: None
        plan: None
        checks: tuple[object, ...]
        metrics: object
        error: None
        summary: None
        approval_reason: None
        auto_approve: bool
        allow_remote_model: bool
        cancel_requested: bool
        created_at: str
        updated_at: str
        started_at: None
        finished_at: None
        internal_token: str

    class FlexibleService(FakeRunService):
        def get_run(self, run_id: str):
            return {**base, "internal_token": TOKEN}

        def resume(self, run_id: str):
            return AttributeRecord(**base, internal_token=TOKEN)

    app = create_app(
        FlexibleService(allowed, artifact),
        bearer_token=TOKEN,
        allowed_roots=allowed,
    )
    with TestClient(app) as client:
        mapping = client.get(f"/v1/runs/{RUN_ID}", headers=_auth())
        attributes = client.post(f"/v1/runs/{RUN_ID}/resume", headers=_auth())

    assert mapping.status_code == 200
    assert attributes.status_code == 200
    assert "internal_token" not in mapping.json()
    assert "internal_token" not in attributes.json()
    assert TOKEN not in mapping.text + attributes.text


def test_service_queue_full_error_maps_to_retryable_429(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")

    class SaturatedService(FakeRunService):
        def create_run(self, **_kwargs) -> RunRecord:
            raise QueueFullError("sensitive queue details")

    service = SaturatedService(allowed, artifact)
    app = create_app(
        service,
        bearer_token=TOKEN,
        allowed_roots=allowed,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            headers=_auth(),
            json={"repo_path": str(allowed), "task": "excess"},
        )

    assert response.status_code == 429
    assert response.json()["error"] == {
        "code": "queue_full",
        "message": "The run service queue is full",
    }
    assert response.headers["retry-after"] == "1"
    assert "sensitive queue details" not in response.text


def test_real_run_service_persists_and_enqueues_without_running_workflow(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    class NonRunningWorkflow:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("worker must remain stopped during this test")

        def cancel(self, _run_id: str) -> None:
            return None

    service = RunService(
        tmp_path / "state",
        allowed_repo_roots=(tmp_path,),
        runner=NonRunningWorkflow(),
        start_worker=False,
    )
    app = create_app(
        service,
        bearer_token=TOKEN,
        allowed_roots=tmp_path,
    )
    try:
        with TestClient(app) as client:
            created = client.post(
                "/v1/runs",
                headers=_auth(),
                json={
                    "repo_path": str(repository),
                    "task": "Inspect the repository",
                    "auto_approve": False,
                    "allow_remote_model": True,
                },
            )
            run_id = created.json()["run_id"]
            fetched = client.get(f"/v1/runs/{run_id}", headers=_auth())

        assert created.status_code == 202
        assert created.headers["location"] == f"/v1/runs/{run_id}"
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "queued"
        assert fetched.json()["allow_remote_model"] is True
        assert service.database.get(run_id).status == "queued"
    finally:
        service.close()


def test_static_ui_extension_is_mounted_behind_authentication(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")

    async def homepage(_request):
        return PlainTextResponse("mounted UI")

    static = Starlette(routes=[Route("/", homepage)])
    app = create_app(
        FakeRunService(allowed, artifact),
        bearer_token=TOKEN,
        allowed_roots=allowed,
        static_app=static,
    )

    with TestClient(app) as client:
        assert client.get("/ui/").status_code == 401
        response = client.get("/ui/", headers=_auth())

    assert response.status_code == 200
    assert response.text == "mounted UI"
    assert DEFAULT_HOST == "127.0.0.1"
