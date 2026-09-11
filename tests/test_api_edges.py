from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
import pytest

import repo_agent.api as api
from repo_agent.artifacts import ArtifactError
from repo_agent.persistence import RunNotFoundError
from repo_agent.service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
)


class MinimalService:
    def create_run(self, **_kwargs):
        raise AssertionError("not called")

    def get_run(self, _run_id):
        raise AssertionError("not called")

    def decide(self, _run_id, **_kwargs):
        raise AssertionError("not called")

    def resume(self, _run_id):
        raise AssertionError("not called")

    def cancel(self, _run_id):
        raise AssertionError("not called")

    def artifact_path(self, _run_id, _kind):
        raise AssertionError("not called")

    def ready(self):
        return True


@pytest.mark.parametrize(
    ("model", "payload", "field"),
    [
        (api.CreateRunRequest, {"repo_path": "C:/repo", "task": "bad\x00task"}, "task"),
        (
            api.CreateRunRequest,
            {"repo_path": "C:/repo", "task": "ok", "base_ref": "-unsafe"},
            "base_ref",
        ),
        (
            api.CreateRunRequest,
            {"repo_path": "C:/repo", "task": "ok", "base_ref": "bad\nref"},
            "base_ref",
        ),
        (api.DecisionRequest, {"approve": True, "reason": "bad\x00reason"}, "reason"),
    ],
)
def test_request_models_reject_control_and_option_like_values(
    model, payload: dict[str, object], field: str
) -> None:
    with pytest.raises(ValidationError) as rejected:
        model.model_validate(payload)

    assert field in str(rejected.value)


def test_request_models_accept_optional_values() -> None:
    assert api.CreateRunRequest(repo_path="C:/repo", task="ok").base_ref is None
    assert api.DecisionRequest(approve=False).reason is None


@pytest.mark.parametrize("token", (None, "", "a" * 513, "two words", "token\n", "密钥"))
def test_bearer_token_validation_rejects_non_visible_ascii(token: object) -> None:
    with pytest.raises(ValueError, match="visible ASCII"):
        api._validated_token(token)


def test_bearer_authorization_requires_one_well_formed_value() -> None:
    middleware = api._BearerAuthMiddleware(lambda *_args: None, b"secret")

    assert middleware._authorized(b"Bearer secret") is True
    assert middleware._authorized(b"bearer secret") is True
    assert middleware._authorized(b"Basic secret") is False
    assert middleware._authorized(b"Bearer") is False
    assert middleware._authorized(b"Bearer secret extra") is False
    assert middleware._authorized(b"Bearer secret\t") is False


def test_bearer_middleware_passes_non_http_scopes_through() -> None:
    calls: list[str] = []

    async def downstream(scope, _receive, _send):
        calls.append(scope["type"])

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(_message):
        return None

    middleware = api._BearerAuthMiddleware(downstream, b"secret")
    asyncio.run(middleware({"type": "websocket"}, receive, send))

    assert calls == ["websocket"]


def test_security_headers_preserve_existing_values() -> None:
    messages: list[dict[str, object]] = []

    async def downstream(_scope, _receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"cache-control", b"public")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    middleware = api._SecurityHeadersMiddleware(downstream)
    asyncio.run(middleware({"type": "http"}, receive, send))

    headers = messages[0]["headers"]
    assert headers.count((b"cache-control", b"public")) == 1
    assert (b"x-content-type-options", b"nosniff") in headers
    assert messages[1] == {"type": "http.response.body", "body": b"ok"}


def test_readiness_supports_async_services_and_contains_failures(
    tmp_path: Path,
) -> None:
    class AsyncService(MinimalService):
        async def ready(self):
            return True

    class BrokenService(MinimalService):
        def ready(self):
            raise RuntimeError("private readiness detail")

    async_app = api.create_app(
        AsyncService(), bearer_token="token", allowed_roots=tmp_path
    )
    broken_app = api.create_app(
        BrokenService(), bearer_token="token", allowed_roots=tmp_path
    )
    with TestClient(async_app) as client:
        ready = client.get("/readyz")
    with TestClient(broken_app) as client:
        unavailable = client.get("/readyz")

    assert ready.status_code == 200
    assert unavailable.status_code == 503
    assert "private readiness detail" not in unavailable.text


def test_create_app_validates_service_roots_token_and_mount(tmp_path: Path) -> None:
    missing_method = SimpleNamespace()
    with pytest.raises(ValueError, match="missing required method"):
        api.create_app(missing_method, bearer_token="token", allowed_roots=tmp_path)

    with pytest.raises(ValueError, match="bearer_token"):
        api.create_app(MinimalService(), bearer_token="", allowed_roots=tmp_path)

    file_root = tmp_path / "file"
    file_root.write_text("not a directory", encoding="utf-8")
    for roots in ((), (Path("relative"),), (tmp_path / "missing",), (file_root,)):
        with pytest.raises(ValueError, match="allowed_roots"):
            api.create_app(MinimalService(), bearer_token="token", allowed_roots=roots)

    assert api._validated_allowed_roots((tmp_path, tmp_path)) == (tmp_path.resolve(),)
    for mount in (
        None,
        "",
        "/",
        "relative",
        "/healthz",
        "/readyz/",
        "/v1",
        "/bad//path",
        "/{id}",
    ):
        with pytest.raises(ValueError, match="static_path"):
            api._validated_mount_path(mount)

    assert api._validated_mount_path("/workspace/") == "/workspace"


def test_repository_path_validation_distinguishes_bad_and_forbidden_paths(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    nested = allowed / "nested"
    nested.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    file_path = allowed / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    roots = (allowed.resolve(),)

    assert api._validated_repository_path(str(nested), roots) == nested.resolve()
    for value in ("relative", str(allowed / "missing"), str(file_path), "bad\x00path"):
        with pytest.raises(api.ApiError) as rejected:
            api._validated_repository_path(value, roots)
        assert rejected.value.status_code == 400
        assert rejected.value.code == "invalid_repository"

    with pytest.raises(api.ApiError) as forbidden:
        api._validated_repository_path(str(outside), roots)
    assert forbidden.value.status_code == 403
    assert forbidden.value.code == "repository_not_allowed"


def test_artifact_path_validators_reject_wrong_types_and_shapes(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("artifact", encoding="utf-8")
    directory = tmp_path / "checks"
    directory.mkdir()

    assert api._validated_artifact_path(file_path) == file_path.resolve()
    assert api._validated_checks_directory(directory) == directory.resolve()

    for value in (object(), tmp_path / "missing", directory):
        with pytest.raises(api.ApiError, match="Artifact not found"):
            api._validated_artifact_path(value)
    for value in (object(), tmp_path / "missing", file_path):
        with pytest.raises(api.ApiError, match="Artifact not found"):
            api._validated_checks_directory(value)

    with pytest.raises(api.ApiError) as bad_id:
        api._validated_run_id("A" * 32)
    assert bad_id.value.code == "run_not_found"
    with pytest.raises(api.ApiError) as bad_kind:
        api._validated_artifact_kind("secret")
    assert bad_kind.value.code == "artifact_not_found"


class _ManifestEntry:
    def __init__(
        self,
        name: str,
        *,
        size: int = 1,
        stat_error: bool = False,
        is_file: bool = True,
        is_symlink: bool = False,
    ) -> None:
        self.name = name
        self._size = size
        self._stat_error = stat_error
        self._is_file = is_file
        self._is_symlink = is_symlink

    def is_symlink(self) -> bool:
        return self._is_symlink

    def is_file(self) -> bool:
        return self._is_file

    def stat(self):
        if self._stat_error:
            raise OSError("stat failed")
        return SimpleNamespace(st_size=self._size)


class _ManifestDirectory:
    def __init__(self, entries=None, *, error: bool = False) -> None:
        self.entries = entries or []
        self.error = error

    def iterdir(self):
        if self.error:
            raise OSError("cannot list")
        return iter(self.entries)


def test_checks_manifest_filters_bounds_and_contains_filesystem_errors(
    monkeypatch,
) -> None:
    entries = [
        _ManifestEntry("bad.txt"),
        _ManifestEntry("link.log", is_symlink=True),
        _ManifestEntry("directory.log", is_file=False),
        _ManifestEntry("broken.log", stat_error=True),
        _ManifestEntry("first.log", size=7),
        _ManifestEntry("second.log", size=9),
    ]
    monkeypatch.setattr(api, "MAX_CHECK_ARTIFACTS", 1)

    result = api._checks_manifest("a" * 32, _ManifestDirectory(entries))

    assert result["artifacts"] == [
        {
            "name": "first.log",
            "size_bytes": 7,
            "href": f"/v1/runs/{'a' * 32}/artifacts/checks/first.log",
        }
    ]
    assert result["truncated"] is True

    with pytest.raises(api.ApiError) as rejected:
        api._checks_manifest("a" * 32, _ManifestDirectory(error=True))
    assert rejected.value.code == "artifact_not_found"


def test_check_artifact_validation_rejects_traversal_missing_and_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checks"
    directory.mkdir()
    valid = directory / "verify.log"
    valid.write_text("passed", encoding="utf-8")
    child_directory = directory / "nested.log"
    child_directory.mkdir()

    assert api._validated_check_artifact(directory, "verify.log") == valid.resolve()
    for filename in ("../secret.log", "missing.log", "nested.log", "bad.txt"):
        with pytest.raises(api.ApiError) as rejected:
            api._validated_check_artifact(directory, filename)
        assert rejected.value.code == "artifact_not_found"


@pytest.mark.parametrize(
    ("operation", "error", "status", "code"),
    [
        ("create", QueueFullError("full"), 429, "queue_full"),
        ("create", RepositoryNotAllowedError("denied"), 403, "repository_not_allowed"),
        ("get", RunNotFoundError("missing"), 404, "run_not_found"),
        ("artifact", ArtifactError("missing"), 404, "artifact_not_found"),
        ("get", LookupError("missing"), 404, "run_not_found"),
        ("artifact", FileNotFoundError("missing"), 404, "artifact_not_found"),
        ("get", FileNotFoundError("missing"), 404, "run_not_found"),
        ("resume", InvalidRunTransitionError("state"), 409, "invalid_run_state"),
        ("create", ValueError("bad"), 400, "invalid_run"),
        ("resume", ValueError("bad"), 409, "invalid_run_state"),
        ("get", TimeoutError("slow"), 503, "service_unavailable"),
        ("get", RuntimeError("private"), 500, "service_error"),
    ],
)
def test_service_error_mapping_is_stable(
    operation: str, error: Exception, status: int, code: str
) -> None:
    def fail():
        raise error

    with pytest.raises(api.ApiError) as rejected:
        api._service_call(operation, fail)

    assert rejected.value.status_code == status
    assert rejected.value.code == code
    assert rejected.value.message != str(error)


def test_run_payload_adapters_and_invalid_responses(
    monkeypatch, tmp_path: Path
) -> None:
    class PayloadModel(BaseModel):
        run_id: str

    class ToDict:
        def to_dict(self):
            return {"run_id": "a" * 32}

    class BadToDict:
        def to_dict(self):
            return "not a mapping"

    class Attributes:
        run_id = "a" * 32

    @dataclass
    class DataclassRecord:
        run_id: str

    assert api._object_mapping(PayloadModel(run_id="a" * 32))["run_id"] == "a" * 32
    assert api._object_mapping(ToDict())["run_id"] == "a" * 32
    assert api._object_mapping(Attributes())["run_id"] == "a" * 32
    assert api._object_mapping(DataclassRecord("a" * 32))["run_id"] == "a" * 32
    for value in (object(), BadToDict()):
        with pytest.raises(TypeError, match="unsupported run record"):
            api._object_mapping(value)

    with pytest.raises(api.ApiError) as invalid:
        api._run_payload({"run_id": "a" * 32})
    assert invalid.value.status_code == 502

    intentional = api.ApiError(418, "intentional", "safe")
    monkeypatch.setattr(
        api, "_object_mapping", lambda _value: (_ for _ in ()).throw(intentional)
    )
    with pytest.raises(api.ApiError) as preserved:
        api._run_payload(object())
    assert preserved.value is intentional


def test_check_hrefs_ignore_noncanonical_values() -> None:
    payload: dict[str, object] = {
        "run_id": "bad",
        "checks": [{"log_artifact": "checks/verify.log"}],
    }
    api._add_check_artifact_hrefs(payload, runs_path="/v1/runs")
    assert "log_artifact_href" not in payload["checks"][0]

    payloads = [
        {"run_id": "a" * 32, "checks": "not-list"},
        {"run_id": "a" * 32, "checks": ["not-dict"]},
        {"run_id": "a" * 32, "checks": [{}]},
        {"run_id": "a" * 32, "checks": [{"log_artifact": 1}]},
        {"run_id": "a" * 32, "checks": [{"log_artifact": "../secret.log"}]},
    ]
    for candidate in payloads:
        api._add_check_artifact_hrefs(candidate, runs_path="/v1/runs")

    assert all("log_artifact_href" not in str(candidate) for candidate in payloads)


def test_artifact_response_maps_late_read_failure() -> None:
    class UnreadablePath:
        name = "artifact.txt"

        def read_bytes(self):
            raise OSError("private path failure")

    with pytest.raises(api.ApiError) as rejected:
        api._artifact_response(UnreadablePath(), media_type="text/plain")
    assert rejected.value.code == "artifact_not_found"


def test_serve_validates_bind_and_delegates_to_uvicorn(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(app, *, host, port):
        calls.append({"app": app, "host": host, "port": port})

    monkeypatch.setattr("uvicorn.run", fake_run)
    service = MinimalService()

    for host, port in (("", 8770), (None, 8770), ("127.0.0.1", 0), ("127.0.0.1", True)):
        with pytest.raises(ValueError):
            api.serve(
                service,
                bearer_token="token",
                allowed_roots=tmp_path,
                host=host,
                port=port,
            )

    api.serve(
        service,
        bearer_token="token",
        allowed_roots=tmp_path,
        host=" 127.0.0.1 ",
        port=9000,
    )
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 9000
