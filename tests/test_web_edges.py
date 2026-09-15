from __future__ import annotations

from contextlib import contextmanager
from email.message import Message
from http import HTTPStatus
from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import socket
import subprocess
import threading
from types import SimpleNamespace
from typing import Iterator

import pytest

import repo_agent.web as web
from repo_agent.artifacts import ArtifactError
from repo_agent.persistence import RunNotFoundError
from repo_agent.processes import CapturedProcess
from repo_agent.run_models import RunRecord
from repo_agent.service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=False,
        timeout=30,
    )
    return completed.stdout.strip()


@pytest.fixture
def committed_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Web Edge Test")
    _git(repository, "config", "user.email", "web-edge@example.invalid")
    (repository / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository.resolve()


def _record(repository: Path, *, status: str = "queued") -> RunRecord:
    now = "2026-09-11T00:00:00.000Z"
    return RunRecord(
        run_id="a" * 32,
        repo_path=str(repository),
        task="maintain repository",
        status=status,
        created_at=now,
        updated_at=now,
    )


class EdgeService:
    def __init__(self, repository: Path, artifacts: Path) -> None:
        self.repository = repository
        self.artifacts = artifacts
        self.artifacts.mkdir()
        self.checks = self.artifacts / "checks"
        self.checks.mkdir()
        (self.artifacts / "patch.diff").write_text(
            "diff --git a/a b/a\n", encoding="utf-8"
        )
        (self.checks / "verify.log").write_text("passed\n", encoding="utf-8")
        self.record = _record(repository)
        self.actions: list[str] = []
        self.list_error: Exception | None = None

    @property
    def queue_depth(self) -> int:
        return 0

    def ready(self) -> bool:
        return True

    def list_runs(self):
        if self.list_error is not None:
            raise self.list_error
        return (self.record,)

    def create_run(self, **_kwargs):
        self.actions.append("create")
        return self.record

    def get_run(self, run_id: str):
        if run_id != self.record.run_id:
            raise RunNotFoundError(run_id)
        return self.record

    def decide(self, run_id: str, **_kwargs):
        self.get_run(run_id)
        self.actions.append("decision")
        return self.record

    def resume(self, run_id: str):
        self.get_run(run_id)
        self.actions.append("resume")
        return self.record

    def cancel(self, run_id: str):
        self.get_run(run_id)
        self.actions.append("cancel")
        return self.record

    def artifact_path(self, run_id: str, kind: str) -> Path:
        self.get_run(run_id)
        if kind == "checks":
            return self.checks
        return self.artifacts / {"patch": "patch.diff"}.get(kind, "missing")


@contextmanager
def _running_server(
    repository: Path, *, run_function=lambda **_kwargs: object(), service=None
) -> Iterator[tuple[str, web.RepoAgentHTTPServer]]:
    server = web.RepoAgentHTTPServer(
        (web.HOST, 0),
        repository,
        run_function=run_function,
        durable_service=service,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://{web.HOST}:{server.server_address[1]}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _request(
    server: web.RepoAgentHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, object]:
    connection = HTTPConnection(web.HOST, server.server_address[1], timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw.decode("utf-8"))


def _raw_post(
    server: web.RepoAgentHTTPServer,
    path: str,
    headers: dict[str, str],
) -> tuple[int, object]:
    connection = HTTPConnection(web.HOST, server.server_address[1], timeout=5)
    connection.putrequest("POST", path)
    for name, value in headers.items():
        connection.putheader(name, value)
    connection.endheaders()
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw.decode("utf-8"))


def _short_post(
    server: web.RepoAgentHTTPServer, path: str, body: bytes
) -> tuple[int, object]:
    declared_length = len(body) + 1
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {web.HOST}:{server.server_address[1]}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {declared_length}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    with socket.create_connection(server.server_address, timeout=5) as connection:
        connection.sendall(request)
        connection.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := connection.recv(8192):
            chunks.append(chunk)
    raw_response = b"".join(chunks)
    head, raw_body = raw_response.split(b"\r\n\r\n", 1)
    status = int(head.split(b"\r\n", 1)[0].split(b" ", 2)[1])
    return status, json.loads(raw_body.decode("utf-8"))


def _duplicate_length_post(
    server: web.RepoAgentHTTPServer, path: str, body: bytes
) -> tuple[int, object]:
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {web.HOST}:{server.server_address[1]}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Content-Length: {len(body) + 1000}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    with socket.create_connection(server.server_address, timeout=5) as connection:
        connection.sendall(request)
        chunks: list[bytes] = []
        while chunk := connection.recv(8192):
            chunks.append(chunk)
    raw_response = b"".join(chunks)
    head, raw_body = raw_response.split(b"\r\n\r\n", 1)
    status = int(head.split(b"\r\n", 1)[0].split(b" ", 2)[1])
    return status, json.loads(raw_body.decode("utf-8"))


def test_server_rejects_non_loopback_bind(committed_repository: Path) -> None:
    with pytest.raises(ValueError, match=web.HOST):
        web.RepoAgentHTTPServer(("0.0.0.0", 0), committed_repository)


def test_disabled_service_routes_and_methods_return_structured_errors(
    committed_repository: Path,
) -> None:
    run_id = "a" * 32
    with _running_server(committed_repository) as (_base_url, server):
        requests = [
            ("GET", "/api/agent/runs"),
            ("GET", f"/api/agent/runs/{run_id}"),
            ("GET", f"/api/agent/runs/{run_id}/artifacts/patch"),
            ("GET", f"/api/agent/runs/{run_id}/artifacts/checks/verify.log"),
            ("POST", "/api/agent/runs"),
        ]
        for method, path in requests:
            status, payload = _request(
                server,
                method,
                path,
                body=b"{}" if method == "POST" else None,
                headers={"Content-Type": "application/json"}
                if method == "POST"
                else None,
            )
            assert status == HTTPStatus.SERVICE_UNAVAILABLE
            assert payload["error"]["code"] == "service_unavailable"

        not_found, payload = _request(server, "POST", "/unknown", body=b"{}")
        assert not_found == HTTPStatus.NOT_FOUND
        assert payload["error"]["code"] == "not_found"

        method_not_allowed, payload = _request(server, "OPTIONS", "/api/state")
        assert method_not_allowed == HTTPStatus.METHOD_NOT_ALLOWED
        assert payload["error"]["code"] == "method_not_allowed"


def test_legacy_run_route_rejects_http_framing_and_bad_json(
    committed_repository: Path,
) -> None:
    with _running_server(committed_repository) as (_base_url, server):
        cases = [
            (
                {"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
            ),
            (
                {"Content-Type": "text/plain", "Content-Length": "0"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
            ),
            (
                {"Content-Type": "application/json", "Content-Length": "bad"},
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
            ),
            (
                {"Content-Type": "application/json", "Content-Length": "-1"},
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
            ),
            (
                {
                    "Content-Type": "application/json",
                    "Content-Length": str(web.MAX_REQUEST_BYTES + 1),
                },
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
            ),
        ]
        for headers, expected_status, expected_code in cases:
            status, payload = _raw_post(server, "/api/runs", headers)
            assert status == expected_status
            assert payload["error"]["code"] == expected_code

        status, payload = _request(
            server,
            "POST",
            "/api/runs",
            body=b"{not-json",
            headers={"Content-Type": "application/json"},
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_json"


def test_legacy_run_route_rejects_unsupported_result(
    committed_repository: Path,
) -> None:
    with _running_server(
        committed_repository,
        run_function=lambda **_kwargs: object(),
    ) as (_base_url, server):
        status, payload = _request(
            server,
            "POST",
            "/api/runs",
            body=b'{"task":"inspect"}',
            headers={"Content-Type": "application/json"},
        )

    assert status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert payload["error"]["code"] == "internal_error"
    assert "unsupported result" in payload["error"]["message"]


def test_agent_resume_cancel_and_request_validation(
    committed_repository: Path, tmp_path: Path
) -> None:
    service = EdgeService(committed_repository, tmp_path / "artifacts")
    run_id = service.record.run_id
    with _running_server(committed_repository, service=service) as (_base_url, server):
        for action in ("resume", "cancel"):
            status, payload = _request(
                server,
                "POST",
                f"/api/agent/runs/{run_id}/{action}",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            assert status == HTTPStatus.OK
            assert payload["run_id"] == run_id

        status, payload = _request(
            server,
            "POST",
            f"/api/agent/runs/{run_id}/resume",
            body=b'{"unexpected":true}',
            headers={"Content-Type": "application/json"},
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_request"

        status, payload = _request(
            server,
            "POST",
            "/api/agent/runs",
            body=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert status == HTTPStatus.BAD_REQUEST
        assert payload["error"]["code"] == "invalid_json"

    assert service.actions == ["resume", "cancel"]


def test_agent_service_list_get_and_artifact_errors_are_contained(
    committed_repository: Path, tmp_path: Path
) -> None:
    service = EdgeService(committed_repository, tmp_path / "artifacts")
    service.list_error = RuntimeError("private list failure")
    missing_id = "b" * 32
    with _running_server(committed_repository, service=service) as (_base_url, server):
        status, payload = _request(server, "GET", "/api/agent/runs")
        assert status == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "private" not in json.dumps(payload)

        status, payload = _request(server, "GET", f"/api/agent/runs/{missing_id}")
        assert status == HTTPStatus.NOT_FOUND
        assert payload["error"]["code"] == "run_not_found"

        status, payload = _request(
            server,
            "GET",
            f"/api/agent/runs/{service.record.run_id}/artifacts/report",
        )
        assert status == HTTPStatus.NOT_FOUND
        assert payload["error"]["code"] == "artifact_not_found"


def _bare_handler(
    *,
    body: bytes = b"{}",
    headers: dict[str, str] | None = None,
) -> web.RepoAgentRequestHandler:
    handler = object.__new__(web.RepoAgentRequestHandler)
    message = Message()
    for name, value in (headers or {}).items():
        message[name] = value
    handler.headers = message
    handler.rfile = BytesIO(body)
    return handler


def test_agent_json_reader_maps_body_timeout() -> None:
    class TimedOutReader:
        def read(self, _length: int) -> bytes:
            raise TimeoutError("slow peer")

    handler = _bare_handler(
        headers={"Content-Type": "application/json", "Content-Length": "2"}
    )
    handler.rfile = TimedOutReader()

    with pytest.raises(web._WebRequestError) as rejected:
        handler._read_json_request()
    assert rejected.value.status == HTTPStatus.REQUEST_TIMEOUT
    assert rejected.value.code == "request_timeout"


def test_short_http_body_is_rejected_before_any_run_is_dispatched(
    committed_repository: Path, tmp_path: Path
) -> None:
    service = EdgeService(committed_repository, tmp_path / "artifacts")

    def unexpected_run(**_kwargs):
        raise AssertionError("short request must not be dispatched")

    with _running_server(
        committed_repository, run_function=unexpected_run, service=service
    ) as (_base_url, server):
        for path in ("/api/runs", "/api/agent/runs"):
            status, payload = _short_post(server, path, b"{}")
            assert status == HTTPStatus.BAD_REQUEST
            assert payload["error"]["code"] == "invalid_request"

    assert service.actions == []


def test_duplicate_content_length_is_rejected_before_any_run_is_dispatched(
    committed_repository: Path, tmp_path: Path
) -> None:
    service = EdgeService(committed_repository, tmp_path / "artifacts")

    def unexpected_run(**_kwargs):
        raise AssertionError("ambiguous request must not be dispatched")

    with _running_server(
        committed_repository, run_function=unexpected_run, service=service
    ) as (_base_url, server):
        for path in ("/api/runs", "/api/agent/runs"):
            status, payload = _duplicate_length_post(server, path, b"{}")
            assert status == HTTPStatus.BAD_REQUEST
            assert payload["error"]["code"] == "invalid_request"

    assert service.actions == []


@pytest.mark.parametrize(
    ("error", "artifact", "status", "code"),
    [
        (RunNotFoundError("private"), False, HTTPStatus.NOT_FOUND, "run_not_found"),
        (ArtifactError("private"), True, HTTPStatus.NOT_FOUND, "artifact_not_found"),
        (
            QueueFullError("queue full"),
            False,
            HTTPStatus.TOO_MANY_REQUESTS,
            "queue_full",
        ),
        (
            InvalidRunTransitionError("bad transition"),
            False,
            HTTPStatus.CONFLICT,
            "invalid_transition",
        ),
        (
            RepositoryNotAllowedError("outside root"),
            False,
            HTTPStatus.FORBIDDEN,
            "repository_not_allowed",
        ),
        (ValueError("bad input"), False, HTTPStatus.BAD_REQUEST, "invalid_request"),
        (
            RuntimeError("private"),
            False,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "internal_error",
        ),
    ],
)
def test_agent_service_error_mapping(
    error: Exception,
    artifact: bool,
    status: HTTPStatus,
    code: str,
) -> None:
    handler = object.__new__(web.RepoAgentRequestHandler)
    calls: list[tuple[HTTPStatus, str, str]] = []
    handler._send_api_error = lambda sent_status, sent_code, message: calls.append(
        (sent_status, sent_code, message)
    )

    handler._send_agent_service_error(error, artifact=artifact)

    assert calls[0][:2] == (status, code)
    if status == HTTPStatus.INTERNAL_SERVER_ERROR:
        assert "private" not in calls[0][2]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"task": ""},
        {"task": "ok", "unexpected": True},
        {"task": "x" * (web.MAX_TASK_LENGTH + 1)},
        {"task": "bad\x00task"},
        {"task": "ok", "base_ref": 1},
        {"task": "ok", "base_ref": ""},
        {"task": "ok", "base_ref": "-unsafe"},
        {"task": "ok", "base_ref": "bad\nref"},
        {"task": "ok", "auto_approve": 1},
        {"task": "ok", "allow_remote_model": 1},
    ],
)
def test_agent_run_validation_rejects_unsafe_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        web._validated_agent_run_request(payload)


def test_agent_run_and_decision_validation_normalize_safe_values() -> None:
    request = web._validated_agent_run_request(
        {
            "task": "  inspect  ",
            "base_ref": "  refs/heads/main  ",
            "auto_approve": True,
            "allow_remote_model": False,
        }
    )
    assert request == {
        "task": "inspect",
        "base_ref": "refs/heads/main",
        "auto_approve": True,
        "allow_remote_model": False,
    }

    assert web._validated_agent_decision({"approve": False, "reason": "  "}) == {
        "approve": False,
        "reason": None,
    }


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"approve": True, "unexpected": 1},
        {"approve": 1},
        {"approve": True, "reason": 1},
        {"approve": True, "reason": "bad\x00reason"},
        {"approve": True, "reason": "x" * 2001},
    ],
)
def test_agent_decision_validation_rejects_bad_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        web._validated_agent_decision(payload)


def test_legacy_run_validation_checks_all_numeric_boundaries() -> None:
    valid = web._validated_run_request({"task": "inspect"})
    assert valid["max_steps"] == 8
    assert valid["timeout_seconds"] == 30.0

    bad_payloads = [
        [],
        {"task": "x" * (web.MAX_TASK_LENGTH + 1)},
        {"task": "bad\x00task"},
        {"task": "ok", "max_steps": True},
        {"task": "ok", "timeout_seconds": True},
        {"task": "ok", "timeout_seconds": "30"},
        {"task": "ok", "timeout_seconds": 10**1000},
        {"task": "ok", "timeout_seconds": 121},
        {"task": "ok", "max_output_bytes": 1000},
    ]
    for payload in bad_payloads:
        with pytest.raises(ValueError):
            web._validated_run_request(payload)

    assert web._artifact_content_type("patch").startswith("text/x-diff")
    assert web._artifact_content_type("checks") == "application/octet-stream"


def test_durable_service_state_tracks_model_configuration(monkeypatch) -> None:
    for name in ("REPO_AGENT_API_KEY", "REPO_AGENT_BASE_URL", "REPO_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    disabled = web._durable_service_state(None)
    assert disabled == {
        "enabled": False,
        "ready": False,
        "queue_depth": 0,
        "model_configured": False,
        "model": None,
    }

    for name in ("REPO_AGENT_API_KEY", "REPO_AGENT_BASE_URL", "REPO_AGENT_MODEL"):
        monkeypatch.setenv(name, "configured")
    service = SimpleNamespace(ready=lambda: True, queue_depth=2)
    enabled = web._durable_service_state(service)
    assert enabled["enabled"] is True
    assert enabled["ready"] is True
    assert enabled["queue_depth"] == 2
    assert enabled["model_configured"] is True
    assert enabled["model"] == "configured"


def test_application_state_contains_git_and_docker_failures(
    committed_repository: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        web,
        "_git_capture",
        lambda *_args: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    monkeypatch.setattr(web, "_capture", lambda _argv: "invalid-docker-format")

    state = web._application_state(committed_repository)

    assert "OSError: git unavailable" in state["repository"]["error"]
    assert state["docker"]["available"] is False
    assert all(not image["present"] for image in state["docker"]["images"])


def test_application_state_marks_individually_missing_images(
    committed_repository: Path, monkeypatch
) -> None:
    def capture(argv: tuple[str, ...], *, timeout: float = 5.0) -> str:
        del timeout
        if argv[:2] == ("docker", "version"):
            return "linux|amd64|28.5.1"
        if argv[:3] == ("docker", "context", "show"):
            return "desktop-linux"
        if "repo-agent-python:0.1" in argv:
            raise RuntimeError("missing image")
        return "sha256:" + "b" * 64

    monkeypatch.setattr(web, "_capture", capture)
    state = web._application_state(committed_repository)

    images = {image["tag"]: image for image in state["docker"]["images"]}
    assert images["repo-agent-python:0.1"]["present"] is False
    assert images["repo-agent-maven:0.1"]["present"] is True


def test_command_capture_maps_spawn_timeout_and_exit_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        web,
        "run_isolated_capture",
        lambda *_args, **_kwargs: CapturedProcess(
            -9, b"", b"", False, False, True
        ),
    )
    with pytest.raises(OSError, match="failed to start"):
        web._capture(("tool",))
    with pytest.raises(OSError, match="Git command failed"):
        web._git_capture(tmp_path, "status")

    monkeypatch.setattr(
        web,
        "run_isolated_capture",
        lambda *_args, **_kwargs: CapturedProcess(
            2,
            b"public stdout",
            b"private stderr",
            False,
            False,
            False,
        ),
    )
    with pytest.raises(RuntimeError, match="private stderr"):
        web._capture(("tool",))
    with pytest.raises(RuntimeError, match="private stderr"):
        web._git_capture(tmp_path, "status")


def test_exception_detail_and_default_data_directory(
    committed_repository: Path, monkeypatch, tmp_path: Path
) -> None:
    assert web._exception_detail(RuntimeError()) == "RuntimeError"
    assert web._exception_detail(ValueError("bad")) == "ValueError: bad"

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    configured = web._default_data_dir(committed_repository)
    assert configured.parent == tmp_path / "local" / "RepoMaintainerAgent"

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(web.Path, "home", lambda: tmp_path / "home")
    fallback = web._default_data_dir(committed_repository)
    assert (
        fallback.parent
        == tmp_path / "home" / ".local" / "state" / "RepoMaintainerAgent"
    )
    assert configured.name == fallback.name


def test_log_message_uses_client_and_timestamp(capsys) -> None:
    handler = object.__new__(web.RepoAgentRequestHandler)
    handler.client_address = ("127.0.0.1", 1234)
    handler.log_date_time_string = lambda: "now"

    handler.log_message("status %s", 200)

    assert capsys.readouterr().out == "[now] 127.0.0.1 status 200\n"


def test_serve_validates_configuration_and_runs_injected_services(
    committed_repository: Path, monkeypatch, tmp_path: Path
) -> None:
    events: list[tuple[str, object]] = []

    class FakeRunService:
        def __init__(self, data_dir, **kwargs) -> None:
            events.append(("service_init", (Path(data_dir), kwargs)))

        def __enter__(self):
            events.append(("service_enter", None))
            return self

        def __exit__(self, *_args):
            events.append(("service_exit", None))

    class FakeServer:
        def __init__(self, address, repository, **kwargs) -> None:
            self.server_address = (address[0], 43210)
            self.repo_path = Path(repository)
            events.append(("server_init", kwargs["durable_service"]))

        def __enter__(self):
            events.append(("server_enter", None))
            return self

        def __exit__(self, *_args):
            events.append(("server_exit", None))

        def serve_forever(self, *, poll_interval):
            events.append(("serve", poll_interval))

    opened: list[str] = []
    monkeypatch.setattr(web, "RunService", FakeRunService)
    monkeypatch.setattr(web, "RepoAgentHTTPServer", FakeServer)
    monkeypatch.setattr(web.webbrowser, "open", opened.append)
    monkeypatch.setenv("REPO_AGENT_API_KEY", "  configured-test-secret  ")

    for port, bootstrap in ((0, False), (65536, False), (True, False), (8765, 1)):
        with pytest.raises(ValueError):
            web.serve(committed_repository, port=port, allow_bootstrap=bootstrap)

    result = web.serve(
        committed_repository,
        port=8765,
        data_dir=tmp_path / "state",
        allow_bootstrap=True,
        open_browser=True,
    )

    assert result == 0
    assert opened == ["http://127.0.0.1:43210/"]
    assert ("serve", 0.2) in events
    init = next(value for name, value in events if name == "service_init")
    assert init[0] == (tmp_path / "state").resolve()
    assert init[1]["allow_bootstrap"] is True
    assert init[1]["secrets"] == ("configured-test-secret",)
