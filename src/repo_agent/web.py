"""Loopback-only web workspace for inspecting key-free agent runs."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
import webbrowser

from . import __version__
from .api import _add_check_artifact_hrefs
from .artifacts import ArtifactError
from .checks import MAX_PHASE_TIMEOUT_SECONDS, MAX_RUN_TIMEOUT_SECONDS
from .models import RunResult
from .persistence import RunNotFoundError
from .processes import run_isolated_capture
from .run_models import ARTIFACT_FILENAMES, RunRecord
from .sandbox import (
    DEFAULT_POLICY,
    SandboxError,
    _sanitized_git_environment,
    _validated_repository,
)
from .service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
    RunService,
    run_agent,
)


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 16 * 1024
MAX_TASK_LENGTH = 4000
ALLOWED_IMAGES: Mapping[str, str] = {
    "repo-agent-python:0.1": "Python 3.11",
    "repo-agent-maven:0.1": "Maven 3.9 / Java 21",
}
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_AGENT_RUN_RE = re.compile(r"^/api/agent/runs/([a-f0-9]{32})$")
_AGENT_ACTION_RE = re.compile(
    r"^/api/agent/runs/([a-f0-9]{32})/(decision|resume|cancel)$"
)
_AGENT_ARTIFACT_RE = re.compile(
    r"^/api/agent/runs/([a-f0-9]{32})/artifacts/(patch|report|result|trace|checks)$"
)
_AGENT_CHECK_ARTIFACT_RE = re.compile(
    r"^/api/agent/runs/([a-f0-9]{32})/artifacts/checks/"
    r"([a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log)$"
)

RunFunction = Callable[..., RunResult]


class _WebRequestError(ValueError):
    def __init__(self, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class RepoAgentHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying immutable repository configuration and one run slot."""

    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True
    request_queue_size = 8

    def __init__(
        self,
        server_address: tuple[str, int],
        repo_path: str | os.PathLike[str],
        *,
        run_function: RunFunction = run_agent,
        durable_service: RunService | None = None,
    ) -> None:
        if server_address[0] != HOST:
            raise ValueError(f"web server must bind to {HOST}")
        self.repo_path = _validated_repository(repo_path)
        self.run_function = run_function
        self.durable_service = durable_service
        self.run_slot = threading.BoundedSemaphore(1)
        super().__init__(server_address, RepoAgentRequestHandler)


class RepoAgentRequestHandler(BaseHTTPRequestHandler):
    """Serve fixed assets and a narrowly validated JSON API."""

    server: RepoAgentHTTPServer
    server_version = "RepoAgentWeb/0.1"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def do_GET(self) -> None:
        if not self._trusted_request_source():
            self._send_api_error(HTTPStatus.FORBIDDEN, "forbidden", "Request rejected")
            return

        path = urlsplit(self.path).path
        if path == "/api/state":
            state = _application_state(self.server.repo_path)
            state["service"] = _durable_service_state(self.server.durable_service)
            self._send_json(state)
            return
        if path == "/api/agent/runs":
            self._handle_agent_list()
            return
        run_match = _AGENT_RUN_RE.fullmatch(path)
        if run_match is not None:
            self._handle_agent_get(run_match.group(1))
            return
        artifact_match = _AGENT_ARTIFACT_RE.fullmatch(path)
        if artifact_match is not None:
            self._handle_agent_artifact(
                artifact_match.group(1), artifact_match.group(2)
            )
            return
        check_artifact_match = _AGENT_CHECK_ARTIFACT_RE.fullmatch(path)
        if check_artifact_match is not None:
            self._handle_agent_check_artifact(
                check_artifact_match.group(1), check_artifact_match.group(2)
            )
            return
        static = _STATIC_FILES.get(path)
        if static is None:
            self._send_api_error(HTTPStatus.NOT_FOUND, "not_found", "Resource not found")
            return
        filename, content_type = static
        body = (
            resources.files("repo_agent")
            .joinpath("web_static")
            .joinpath(filename)
            .read_bytes()
        )
        self._send_bytes(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:
        if not self._trusted_request_source():
            self._send_api_error(HTTPStatus.FORBIDDEN, "forbidden", "Request rejected")
            return
        path = urlsplit(self.path).path
        if path == "/api/agent/runs" or _AGENT_ACTION_RE.fullmatch(path):
            self._handle_agent_post(path)
            return
        if path != "/api/runs":
            self._send_api_error(HTTPStatus.NOT_FOUND, "not_found", "Resource not found")
            return
        if self.headers.get("Transfer-Encoding"):
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Transfer-Encoding is not supported",
            )
            return
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self._send_api_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
            return
        try:
            length = self._validated_content_length()
        except _WebRequestError as exc:
            self._send_api_error(exc.status, exc.code, exc.message)
            return

        try:
            raw_body = self._read_exact_body(length)
            payload = json.loads(
                raw_body.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
            request = _validated_run_request(payload)
        except _WebRequestError as exc:
            self._send_api_error(exc.status, exc.code, exc.message)
            return
        except (TimeoutError, OSError):
            self._send_api_error(
                HTTPStatus.REQUEST_TIMEOUT,
                "request_timeout",
                "Request body timed out",
            )
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Request body must be valid UTF-8 JSON",
            )
            return
        except ValueError as exc:
            self._send_api_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(exc),
            )
            return

        if not self.server.run_slot.acquire(blocking=False):
            self._send_api_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "busy",
                "Another inspection is already running",
            )
            return

        started = time.monotonic()
        head = "unknown"
        try:
            head = _git_capture(
                self.server.repo_path,
                "rev-parse",
                "--short=12",
                "HEAD",
            )
            result = self.server.run_function(
                task=request["task"],
                repo_path=self.server.repo_path,
                image=request["image"],
                max_steps=request["max_steps"],
                timeout_seconds=request["timeout_seconds"],
                max_output_bytes=request["max_output_bytes"],
                allow_mutations=False,
            )
            if not isinstance(result, RunResult):
                raise TypeError("run function returned an unsupported result")
        except (OSError, RuntimeError, ValueError, SandboxError) as exc:
            self._send_json(
                {
                    "task": request["task"],
                    "status": "setup_error",
                    "answer": None,
                    "steps": 0,
                    "tool_results": [],
                    "error": _exception_detail(exc),
                    "metadata": _run_metadata(
                        self.server.repo_path,
                        request["image"],
                        head,
                        started,
                    ),
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            self._send_api_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                _exception_detail(exc),
            )
        else:
            response = asdict(result)
            response["metadata"] = _run_metadata(
                self.server.repo_path,
                request["image"],
                head,
                started,
            )
            self._send_json(response)
        finally:
            self.server.run_slot.release()

    def _handle_agent_list(self) -> None:
        service = self.server.durable_service
        if service is None:
            self._send_api_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "The durable maintenance service is not enabled",
            )
            return
        try:
            records = service.list_runs()
        except Exception:
            self._send_api_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "Unable to read durable runs",
            )
            return
        recent = tuple(reversed(records[-50:]))
        self._send_json(
            {
                "runs": [_agent_record_payload(record) for record in recent],
                "queue_depth": service.queue_depth,
                "ready": service.ready(),
            }
        )

    def _handle_agent_get(self, run_id: str) -> None:
        service = self.server.durable_service
        if service is None:
            self._send_api_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "The durable maintenance service is not enabled",
            )
            return
        try:
            record = service.get_run(run_id)
        except Exception as exc:
            self._send_agent_service_error(exc)
            return
        self._send_json(_agent_record_payload(record))

    def _handle_agent_post(self, path: str) -> None:
        service = self.server.durable_service
        if service is None:
            self._send_api_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "The durable maintenance service is not enabled",
            )
            return
        try:
            payload = self._read_json_request()
            if path == "/api/agent/runs":
                request = _validated_agent_run_request(payload)
                record = service.create_run(
                    repo_path=self.server.repo_path,
                    task=request["task"],
                    base_ref=request["base_ref"],
                    auto_approve=request["auto_approve"],
                    allow_remote_model=request["allow_remote_model"],
                )
                self._send_json(
                    _agent_record_payload(record), status=HTTPStatus.ACCEPTED
                )
                return

            match = _AGENT_ACTION_RE.fullmatch(path)
            if match is None:
                raise _WebRequestError(
                    HTTPStatus.NOT_FOUND, "not_found", "Resource not found"
                )
            run_id, action = match.groups()
            if action == "decision":
                decision = _validated_agent_decision(payload)
                record = service.decide(
                    run_id,
                    approve=decision["approve"],
                    reason=decision["reason"],
                )
            else:
                if payload != {}:
                    raise ValueError(f"{action} request must be an empty JSON object")
                record = (
                    service.resume(run_id)
                    if action == "resume"
                    else service.cancel(run_id)
                )
            self._send_json(_agent_record_payload(record))
        except _WebRequestError as exc:
            self._send_api_error(exc.status, exc.code, exc.message)
        except Exception as exc:
            self._send_agent_service_error(exc)

    def _handle_agent_artifact(self, run_id: str, kind: str) -> None:
        service = self.server.durable_service
        if service is None:
            self._send_api_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "The durable maintenance service is not enabled",
            )
            return
        try:
            path = service.artifact_path(run_id, kind)
            if kind == "checks":
                if not path.is_dir() or path.is_symlink():
                    raise ArtifactError("check artifact directory is invalid")
                files = [
                    {
                        "name": item.name,
                        "href": f"/api/agent/runs/{run_id}/artifacts/checks/{item.name}",
                    }
                    for item in sorted(path.iterdir())
                    if item.is_file() and not item.is_symlink()
                ]
                self._send_json({"run_id": run_id, "files": files})
                return
            if not path.is_file() or path.is_symlink():
                raise ArtifactError("artifact file is invalid")
            body = path.read_bytes()
        except Exception as exc:
            self._send_agent_service_error(exc, artifact=True)
            return
        self._send_bytes(
            HTTPStatus.OK,
            body,
            _artifact_content_type(kind),
            extra_headers={
                "Content-Disposition": f'attachment; filename="{path.name}"'
            },
        )

    def _handle_agent_check_artifact(self, run_id: str, filename: str) -> None:
        service = self.server.durable_service
        if service is None:
            self._send_api_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "service_unavailable",
                "The durable maintenance service is not enabled",
            )
            return
        try:
            directory = service.artifact_path(run_id, "checks")
            if not directory.is_dir() or directory.is_symlink():
                raise ArtifactError("check artifact directory is invalid")
            path = directory / filename
            if (
                path.parent.resolve(strict=True) != directory.resolve(strict=True)
                or not path.is_file()
                or path.is_symlink()
            ):
                raise ArtifactError("check artifact is invalid")
            body = path.read_bytes()
        except Exception as exc:
            self._send_agent_service_error(exc, artifact=True)
            return
        self._send_bytes(
            HTTPStatus.OK,
            body,
            "text/plain; charset=utf-8",
            extra_headers={
                "Content-Disposition": f'attachment; filename="{path.name}"'
            },
        )

    def _read_json_request(self) -> object:
        if self.headers.get("Transfer-Encoding"):
            raise _WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Transfer-Encoding is not supported",
            )
        if self.headers.get_content_type() != "application/json":
            raise _WebRequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
        length = self._validated_content_length()
        try:
            raw = self._read_exact_body(length)
        except (TimeoutError, OSError) as exc:
            raise _WebRequestError(
                HTTPStatus.REQUEST_TIMEOUT,
                "request_timeout",
                "Request body timed out",
            ) from exc
        try:
            return json.loads(
                raw.decode("utf-8"), parse_constant=_reject_json_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Request body must be valid UTF-8 JSON",
            ) from exc

    def _validated_content_length(self) -> int:
        values = self.headers.get_all("Content-Length") or []
        if len(values) > 1:
            raise _WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Request must contain exactly one Content-Length header",
            )
        raw_length = values[0] if values else ""
        if (
            not raw_length
            or len(raw_length) > 128
            or any(character not in "0123456789" for character in raw_length)
        ):
            raise _WebRequestError(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "A valid Content-Length is required",
            )
        significant_length = raw_length.lstrip("0") or "0"
        maximum_length = str(MAX_REQUEST_BYTES)
        if (
            len(significant_length) > len(maximum_length)
            or len(significant_length) == len(maximum_length)
            and significant_length > maximum_length
        ):
            raise _WebRequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"Request body must be at most {MAX_REQUEST_BYTES} bytes",
            )
        return int(significant_length)

    def _read_exact_body(self, length: int) -> bytes:
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise _WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "Request body length does not match Content-Length",
            )
        return raw

    def _send_agent_service_error(
        self, exc: BaseException, *, artifact: bool = False
    ) -> None:
        if isinstance(exc, RunNotFoundError):
            self._send_api_error(
                HTTPStatus.NOT_FOUND, "run_not_found", "Run not found"
            )
        elif isinstance(exc, ArtifactError) and artifact:
            self._send_api_error(
                HTTPStatus.NOT_FOUND, "artifact_not_found", "Artifact not found"
            )
        elif isinstance(exc, QueueFullError):
            self._send_api_error(
                HTTPStatus.TOO_MANY_REQUESTS, "queue_full", str(exc)
            )
        elif isinstance(exc, InvalidRunTransitionError):
            self._send_api_error(HTTPStatus.CONFLICT, "invalid_transition", str(exc))
        elif isinstance(exc, RepositoryNotAllowedError):
            self._send_api_error(
                HTTPStatus.FORBIDDEN, "repository_not_allowed", str(exc)
            )
        elif isinstance(exc, (TypeError, ValueError)):
            self._send_api_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        else:
            self._send_api_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The durable maintenance request failed",
            )

    def do_OPTIONS(self) -> None:
        self._send_api_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "Method not allowed",
        )

    def log_message(self, format: str, *args: object) -> None:
        message = format % args
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {message}")

    def _trusted_request_source(self) -> bool:
        port = self.server.server_address[1]
        allowed_hosts = {f"{HOST}:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        if host not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{item}" for item in allowed_hosts}

    def _send_api_error(
        self, status: HTTPStatus, code: str, message: str
    ) -> None:
        self._send_json(
            {"error": {"code": code, "message": message}},
            status=status,
        )

    def _send_json(
        self, payload: object, *, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True


def _durable_service_state(service: RunService | None) -> dict[str, object]:
    configured = all(
        isinstance(os.environ.get(name), str) and bool(os.environ[name].strip())
        for name in (
            "REPO_AGENT_API_KEY",
            "REPO_AGENT_BASE_URL",
            "REPO_AGENT_MODEL",
        )
    )
    if service is None:
        return {
            "enabled": False,
            "ready": False,
            "queue_depth": 0,
            "model_configured": configured,
            "model": os.environ["REPO_AGENT_MODEL"].strip() if configured else None,
        }
    return {
        "enabled": True,
        "ready": service.ready(),
        "queue_depth": service.queue_depth,
        "model_configured": configured,
        "model": os.environ["REPO_AGENT_MODEL"].strip() if configured else None,
    }


def _agent_record_payload(record: RunRecord) -> dict[str, object]:
    payload = record.model_dump(mode="json")
    base = f"/api/agent/runs/{record.run_id}/artifacts"
    _add_check_artifact_hrefs(payload, runs_path="/api/agent/runs")
    payload["artifacts"] = {
        kind: f"{base}/{kind}" for kind in ARTIFACT_FILENAMES
    }
    return payload


def _validated_agent_run_request(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    allowed = {"task", "base_ref", "auto_approve", "allow_remote_model"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected field(s): {', '.join(unexpected)}")
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must contain non-whitespace text")
    task = task.strip()
    if len(task) > MAX_TASK_LENGTH or "\x00" in task:
        raise ValueError(f"task must be at most {MAX_TASK_LENGTH} safe characters")
    base_ref = payload.get("base_ref")
    if base_ref is not None:
        if (
            not isinstance(base_ref, str)
            or not base_ref.strip()
            or len(base_ref.strip()) > 255
            or base_ref.strip().startswith("-")
            or any(ord(character) < 32 for character in base_ref)
        ):
            raise ValueError("base_ref is invalid")
        base_ref = base_ref.strip()
    auto_approve = payload.get("auto_approve", False)
    allow_remote_model = payload.get("allow_remote_model", False)
    if type(auto_approve) is not bool or type(allow_remote_model) is not bool:
        raise ValueError("approval and remote-model flags must be boolean")
    return {
        "task": task,
        "base_ref": base_ref,
        "auto_approve": auto_approve,
        "allow_remote_model": allow_remote_model,
    }


def _validated_agent_decision(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    unexpected = sorted(set(payload) - {"approve", "reason"})
    if unexpected:
        raise ValueError(f"Unexpected field(s): {', '.join(unexpected)}")
    approve = payload.get("approve")
    if type(approve) is not bool:
        raise ValueError("approve must be a boolean")
    reason = payload.get("reason")
    if reason is not None:
        if (
            not isinstance(reason, str)
            or "\x00" in reason
            or len(reason.strip()) > 2000
        ):
            raise ValueError("reason must be at most 2000 characters")
        reason = reason.strip() or None
    return {"approve": approve, "reason": reason}


def _artifact_content_type(kind: str) -> str:
    return {
        "patch": "text/x-diff; charset=utf-8",
        "report": "text/markdown; charset=utf-8",
        "result": "application/json; charset=utf-8",
        "trace": "application/x-ndjson; charset=utf-8",
    }.get(kind, "application/octet-stream")


def _validated_run_request(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    allowed = {
        "task",
        "image",
        "max_steps",
        "timeout_seconds",
        "max_output_bytes",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected field(s): {', '.join(unexpected)}")

    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must contain non-whitespace text")
    task = task.strip()
    if len(task) > MAX_TASK_LENGTH or "\x00" in task:
        raise ValueError(f"task must be at most {MAX_TASK_LENGTH} safe characters")

    image = payload.get("image", "repo-agent-python:0.1")
    if not isinstance(image, str) or image not in ALLOWED_IMAGES:
        raise ValueError("image must be one of the verified sandbox images")

    max_steps = _bounded_integer(payload.get("max_steps", 8), "max_steps", 1, 16)
    timeout_seconds = _bounded_number(
        payload.get("timeout_seconds", 30.0),
        "timeout_seconds",
        1,
        120,
    )
    max_output_bytes = _bounded_integer(
        payload.get("max_output_bytes", 65536),
        "max_output_bytes",
        1024,
        262144,
    )
    return {
        "task": task,
        "image": image,
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "max_output_bytes": max_output_bytes,
    }


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON number must be finite, got {value}")


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _bounded_number(
    value: object, name: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from {minimum:g} to {maximum:g}")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a number from {minimum:g} to {maximum:g}"
        ) from exc
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be a number from {minimum:g} to {maximum:g}")
    return converted


def _application_state(repo_path: Path) -> dict[str, object]:
    repository: dict[str, object]
    try:
        status = _git_capture(
            repo_path,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        branch = _git_capture(repo_path, "branch", "--show-current") or "detached"
        repository = {
            "path": str(repo_path),
            "head": _git_capture(repo_path, "rev-parse", "--short=12", "HEAD"),
            "branch": branch,
            "dirty_count": len(status.splitlines()) if status else 0,
            "head_only": True,
        }
    except (OSError, RuntimeError) as exc:
        repository = {
            "path": str(repo_path),
            "error": _exception_detail(exc),
            "head_only": True,
        }

    docker: dict[str, object]
    try:
        server = _capture(
            (
                "docker",
                "version",
                "--format",
                "{{.Server.Os}}|{{.Server.Arch}}|{{.Server.Version}}",
            )
        )
        operating_system, architecture, version = server.split("|", 2)
        context = _capture(("docker", "context", "show"))
        images = []
        for tag, label in ALLOWED_IMAGES.items():
            try:
                image_id = _capture(
                    ("docker", "image", "inspect", "--format", "{{.Id}}", tag)
                )
            except (OSError, RuntimeError):
                images.append({"tag": tag, "label": label, "present": False})
            else:
                images.append(
                    {
                        "tag": tag,
                        "label": label,
                        "present": True,
                        "id": image_id,
                    }
                )
        docker = {
            "available": True,
            "context": context,
            "server": {
                "os": operating_system,
                "arch": architecture,
                "version": version,
            },
            "images": images,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        docker = {
            "available": False,
            "error": _exception_detail(exc),
            "images": [
                {"tag": tag, "label": label, "present": False}
                for tag, label in ALLOWED_IMAGES.items()
            ],
        }

    return {
        "app": {"name": "Repo Maintainer Agent", "version": __version__},
        "repository": repository,
        "docker": docker,
        "policy": {
            "snapshot": "committed HEAD",
            "mutations": "candidate only",
            "network": DEFAULT_POLICY.network,
            "rootfs": "read-only",
            "user": DEFAULT_POLICY.user,
            "cpus": DEFAULT_POLICY.cpus,
            "memory": DEFAULT_POLICY.memory,
            "pids": DEFAULT_POLICY.pids_limit,
            "capabilities": "drop ALL",
            "security": "no-new-privileges",
            "logs": DEFAULT_POLICY.log_driver,
        },
        "checks": {
            "bootstrap": {
                "authorization": "explicit opt-in",
                "network": "bridge",
            },
            "verify": {
                "network": "none",
                "workspace": "fresh copy",
            },
            "limits": {
                "phase_seconds": int(MAX_PHASE_TIMEOUT_SECONDS),
                "run_seconds": int(MAX_RUN_TIMEOUT_SECONDS),
            },
            "profiles": [
                {
                    "id": "python-pytest",
                    "label": "Python 3.11 / pytest",
                    "image": "repo-agent-python:0.1",
                },
                {
                    "id": "maven-test",
                    "label": "Java 17/21 / Maven + JUnit 5",
                    "image": "repo-agent-maven:0.1",
                },
            ],
        },
        "images": [
            {"tag": tag, "label": label} for tag, label in ALLOWED_IMAGES.items()
        ],
    }


def _capture(argv: tuple[str, ...], *, timeout: float = 5.0) -> str:
    try:
        completed = run_isolated_capture(
            argv,
            timeout_seconds=timeout,
            max_stdout_bytes=65536,
            max_stderr_bytes=65536,
        )
    except OSError as exc:
        raise OSError(f"command failed to start: {_exception_detail(exc)}") from exc
    if completed.timed_out:
        raise OSError("command failed to start: command timed out")
    if completed.stdout_truncated or completed.stderr_truncated:
        raise RuntimeError("command output exceeded the safe limit")
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"command exited {completed.returncode}")
    return stdout


def _git_capture(repo_path: Path, *args: str) -> str:
    try:
        completed = run_isolated_capture(
            ("git", "-C", str(repo_path), *args),
            env=_sanitized_git_environment(),
            timeout_seconds=5,
            max_stdout_bytes=65536,
            max_stderr_bytes=65536,
        )
    except OSError as exc:
        raise OSError(f"Git command failed: {_exception_detail(exc)}") from exc
    if completed.timed_out:
        raise OSError("Git command failed: command timed out")
    if completed.stdout_truncated or completed.stderr_truncated:
        raise RuntimeError("Git command output exceeded the safe limit")
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"Git exited {completed.returncode}")
    return stdout


def _run_metadata(
    repo_path: Path,
    image: object,
    head: str,
    started: float,
) -> dict[str, object]:
    return {
        "repository": str(repo_path),
        "head": head,
        "image": image,
        "provider": "demo",
        "duration_ms": round((time.monotonic() - started) * 1000),
    }


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _default_data_dir(repo_path: Path) -> Path:
    local_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_data).expanduser() if local_data else Path.home() / ".local" / "state"
    identity = hashlib.sha256(str(repo_path).casefold().encode("utf-8")).hexdigest()[:16]
    return base / "RepoMaintainerAgent" / identity


def serve(
    repo_path: str | os.PathLike[str],
    *,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
    data_dir: str | os.PathLike[str] | None = None,
    allow_bootstrap: bool = False,
) -> int:
    """Serve the local demo and durable maintenance workspace until interrupted."""

    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    if type(allow_bootstrap) is not bool:
        raise ValueError("allow_bootstrap must be a boolean")
    repository = _validated_repository(repo_path)
    state_dir = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else _default_data_dir(repository)
    )
    configured_secret = os.environ.get("REPO_AGENT_API_KEY", "").strip()
    with RunService(
        state_dir,
        allowed_repo_roots=(repository,),
        max_queue=4,
        allow_bootstrap=allow_bootstrap,
        secrets=(configured_secret,),
    ) as durable_service:
        with RepoAgentHTTPServer(
            (HOST, port),
            repository,
            durable_service=durable_service,
        ) as server:
            actual_port = server.server_address[1]
            url = f"http://{HOST}:{actual_port}/"
            print(f"Repo Maintainer Agent UI: {url}", flush=True)
            print(f"Repository: {server.repo_path}", flush=True)
            print(f"Run data: {state_dir}", flush=True)
            if open_browser:
                webbrowser.open(url)
            server.serve_forever(poll_interval=0.2)
    return 0


__all__ = [
    "ALLOWED_IMAGES",
    "DEFAULT_PORT",
    "HOST",
    "MAX_REQUEST_BYTES",
    "MAX_TASK_LENGTH",
    "RepoAgentHTTPServer",
    "_default_data_dir",
    "serve",
]
