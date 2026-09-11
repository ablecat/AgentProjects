"""Authenticated FastAPI surface for durable repository-agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
import inspect
import os
from pathlib import Path
import re
import secrets
from typing import Protocol, TypeVar, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .artifacts import ArtifactError
from .persistence import RunNotFoundError
from .run_models import ARTIFACT_FILENAMES, ArtifactKind, RunRecord
from .service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
MAX_TASK_LENGTH = 4000
MAX_REASON_LENGTH = 2000
MAX_BASE_REF_LENGTH = 255
MAX_CHECK_ARTIFACTS = 256
_RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_CHECK_ARTIFACT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log$")
_CHECK_ARTIFACT_PATH_RE = re.compile(
    r"^checks/([a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log)$"
)
_HEALTH_PATHS = frozenset({"/healthz", "/readyz"})
_SERVICE_METHODS = (
    "create_run",
    "get_run",
    "decide",
    "resume",
    "cancel",
    "artifact_path",
    "ready",
)
_RUN_FIELDS = tuple(RunRecord.model_fields)


class RunServiceLike(Protocol):
    """Service contract consumed by the HTTP adapter."""

    def create_run(
        self,
        *,
        repo_path: str | os.PathLike[str],
        task: str,
        base_ref: str | None = None,
        auto_approve: bool = False,
        allow_remote_model: bool = False,
    ) -> object: ...

    def get_run(self, run_id: str) -> object: ...

    def decide(
        self,
        run_id: str,
        *,
        approve: bool,
        reason: str | None = None,
    ) -> object: ...

    def resume(self, run_id: str) -> object: ...

    def cancel(self, run_id: str) -> object: ...

    def artifact_path(
        self, run_id: str, kind: ArtifactKind
    ) -> str | os.PathLike[str]: ...

    def ready(self) -> bool: ...


class CreateRunRequest(BaseModel):
    """Validated request for one durable run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    repo_path: str = Field(min_length=1, max_length=32767)
    task: str = Field(min_length=1, max_length=MAX_TASK_LENGTH)
    base_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_BASE_REF_LENGTH,
    )
    auto_approve: StrictBool = False
    allow_remote_model: StrictBool = False

    @field_validator("task")
    @classmethod
    def reject_nul_task(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("task must not contain NUL characters")
        return value

    @field_validator("base_ref")
    @classmethod
    def validate_base_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.startswith("-") or any(ord(character) < 32 for character in value):
            raise ValueError("base_ref contains unsupported characters")
        return value


class DecisionRequest(BaseModel):
    """Approval decision supplied at a workflow checkpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approve: StrictBool
    reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_REASON_LENGTH,
    )

    @field_validator("reason")
    @classmethod
    def reject_nul_reason(cls, value: str | None) -> str | None:
        if value is not None and "\x00" in value:
            raise ValueError("reason must not contain NUL characters")
        return value


class ApiError(RuntimeError):
    """An intentional, non-sensitive API error."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


_ResultT = TypeVar("_ResultT")


class _BearerAuthMiddleware:
    """Authenticate every HTTP endpoint except liveness and readiness."""

    def __init__(self, app: ASGIApp, token: bytes) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in _HEALTH_PATHS:
            await self.app(scope, receive, send)
            return
        values = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"authorization"
        ]
        if len(values) != 1 or not self._authorized(values[0]):
            response = _error_response(
                401,
                "unauthorized",
                "Bearer authentication is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _authorized(self, value: bytes) -> bool:
        scheme, separator, presented = value.partition(b" ")
        if separator != b" " or scheme.lower() != b"bearer" or not presented:
            return False
        if any(character in b" \t\r\n" for character in presented):
            return False
        return secrets.compare_digest(presented, self._token)


class _SecurityHeadersMiddleware:
    """Apply non-cacheable API defaults, including mounted UI responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", ()))
                existing = {name.lower() for name, _value in headers}
                for name, value in _response_headers().items():
                    encoded_name = name.lower().encode("ascii")
                    if encoded_name not in existing:
                        headers.append((encoded_name, value.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_app(
    service: RunServiceLike,
    *,
    bearer_token: str,
    allowed_roots: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    static_app: ASGIApp | None = None,
    static_path: str = "/ui",
) -> FastAPI:
    """Create an authenticated API around one injected durable run service."""

    _validate_service(service)
    token = _validated_token(bearer_token)
    roots = _validated_allowed_roots(allowed_roots)

    app = FastAPI(
        title="Repo Maintainer Agent API",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.run_service = service
    app.state.allowed_roots = roots
    app.add_middleware(_BearerAuthMiddleware, token=token)
    app.add_middleware(_SecurityHeadersMiddleware)

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            exc.status_code,
            exc.code,
            exc.message,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "invalid_request",
            "Request validation failed",
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = "Resource not found" if exc.status_code == 404 else "Request failed"
        return _error_response(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return _error_response(500, "internal_error", "Internal server error")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def ready() -> JSONResponse:
        try:
            is_ready = service.ready()
            if inspect.isawaitable(is_ready):
                is_ready = asyncio.run(is_ready)
        except Exception:
            is_ready = False
        if is_ready is not True:
            return JSONResponse(
                {"status": "not_ready"},
                status_code=503,
                headers=_response_headers(),
            )
        return JSONResponse(
            {"status": "ready"},
            headers=_response_headers(),
        )

    @app.post("/v1/runs", status_code=202)
    def create_run(request: CreateRunRequest) -> JSONResponse:
        repo_path = _validated_repository_path(request.repo_path, roots)
        result = _service_call(
            "create",
            lambda: service.create_run(
                repo_path=str(repo_path),
                task=request.task,
                base_ref=request.base_ref,
                auto_approve=bool(request.auto_approve),
                allow_remote_model=bool(request.allow_remote_model),
            ),
        )
        payload = _run_payload(result)
        return JSONResponse(
            payload,
            status_code=202,
            headers={
                **_response_headers(),
                "Location": f"/v1/runs/{payload['run_id']}",
            },
        )

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "get",
            lambda: service.get_run(normalized),
        )
        return _run_payload(result)

    @app.post("/v1/runs/{run_id}/decision")
    def decide(run_id: str, request: DecisionRequest) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "decision",
            lambda: service.decide(
                normalized,
                approve=bool(request.approve),
                reason=request.reason,
            ),
        )
        return _run_payload(result)

    @app.post("/v1/runs/{run_id}/resume")
    def resume(run_id: str) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "resume",
            lambda: service.resume(normalized),
        )
        return _run_payload(result)

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "cancel",
            lambda: service.cancel(normalized),
        )
        return _run_payload(result)

    @app.get("/v1/runs/{run_id}/artifacts/{kind}")
    def artifact(run_id: str, kind: str) -> Response:
        normalized = _validated_run_id(run_id)
        artifact_kind = _validated_artifact_kind(kind)
        raw_path = _service_call(
            "artifact",
            lambda: service.artifact_path(normalized, artifact_kind),
        )
        if artifact_kind == "checks":
            directory = _validated_checks_directory(raw_path)
            return JSONResponse(
                _checks_manifest(normalized, directory),
                headers=_response_headers(),
            )
        path = _validated_artifact_path(raw_path)
        return _artifact_response(
            path,
            media_type=_artifact_media_type(artifact_kind),
        )

    @app.get("/v1/runs/{run_id}/artifacts/checks/{filename}")
    def check_artifact(run_id: str, filename: str) -> Response:
        normalized = _validated_run_id(run_id)
        directory = _validated_checks_directory(
            _service_call(
                "artifact",
                lambda: service.artifact_path(normalized, "checks"),
            )
        )
        path = _validated_check_artifact(directory, filename)
        return _artifact_response(
            path,
            media_type="text/plain; charset=utf-8",
        )

    if static_app is not None:
        mount_path = _validated_mount_path(static_path)
        app.mount(mount_path, static_app, name="repo-agent-ui")

    return app


def serve(
    service: RunServiceLike,
    *,
    bearer_token: str,
    allowed_roots: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    static_app: ASGIApp | None = None,
    static_path: str = "/ui",
) -> None:
    """Run the API; loopback is the default and recommended bind address."""

    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("uvicorn is required to serve the REST API") from exc
    uvicorn.run(
        create_app(
            service,
            bearer_token=bearer_token,
            allowed_roots=allowed_roots,
            static_app=static_app,
            static_path=static_path,
        ),
        host=host.strip(),
        port=port,
    )


def _validate_service(service: object) -> None:
    missing = [
        name
        for name in _SERVICE_METHODS
        if not callable(getattr(service, name, None))
    ]
    if missing:
        raise ValueError("service is missing required method(s): " + ", ".join(missing))


def _validated_token(value: object) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("bearer_token must contain 1 to 512 visible ASCII characters")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "bearer_token must contain 1 to 512 visible ASCII characters"
        ) from exc
    if any(character < 33 or character > 126 for character in encoded):
        raise ValueError("bearer_token must contain 1 to 512 visible ASCII characters")
    return encoded


def _validated_allowed_roots(
    values: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> tuple[Path, ...]:
    candidates: tuple[str | os.PathLike[str], ...]
    if isinstance(values, (str, os.PathLike)):
        candidates = (values,)
    else:
        candidates = tuple(values)
    if not candidates:
        raise ValueError("allowed_roots must contain at least one directory")
    roots: list[Path] = []
    for candidate in candidates:
        try:
            path = Path(candidate)
            if not path.is_absolute():
                raise ValueError("allowed_roots must contain absolute paths")
            resolved = path.resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "allowed_roots must contain existing absolute directories"
            ) from exc
        if not resolved.is_dir():
            raise ValueError("allowed_roots must contain existing absolute directories")
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _validated_repository_path(value: str, roots: tuple[Path, ...]) -> Path:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ApiError(
                400,
                "invalid_repository",
                "repo_path must be an existing absolute directory",
            )
        resolved = candidate.resolve(strict=True)
    except ApiError:
        raise
    except (OSError, ValueError) as exc:
        raise ApiError(
            400,
            "invalid_repository",
            "repo_path must be an existing absolute directory",
        ) from exc
    if not resolved.is_dir():
        raise ApiError(
            400,
            "invalid_repository",
            "repo_path must be an existing absolute directory",
        )
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ApiError(
            403,
            "repository_not_allowed",
            "repo_path is outside the configured allowed roots",
        )
    return resolved


def _validated_run_id(value: str) -> str:
    if not _RUN_ID_RE.fullmatch(value):
        raise ApiError(404, "run_not_found", "Run not found")
    return value


def _validated_artifact_kind(value: str) -> ArtifactKind:
    if value not in ARTIFACT_FILENAMES:
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    return value


def _validated_artifact_path(value: object) -> Path:
    try:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError("artifact path must be path-like")
        candidate = Path(value)
        if candidate.is_symlink():
            raise ValueError("artifact symlinks are not allowed")
        path = candidate.resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc
    if not path.is_file():
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    return path


def _validated_checks_directory(value: object) -> Path:
    try:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError("artifact path must be path-like")
        candidate = Path(value)
        if candidate.is_symlink():
            raise ValueError("artifact symlinks are not allowed")
        path = candidate.resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc
    if not path.is_dir():
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    return path


def _checks_manifest(run_id: str, directory: Path) -> dict[str, object]:
    try:
        candidates = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
    except OSError as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc

    eligible = [
        candidate
        for candidate in candidates
        if _CHECK_ARTIFACT_RE.fullmatch(candidate.name)
        and not candidate.is_symlink()
        and candidate.is_file()
    ]
    entries: list[dict[str, object]] = []
    for candidate in eligible:
        if len(entries) >= MAX_CHECK_ARTIFACTS:
            break
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        entries.append(
            {
                "name": candidate.name,
                "size_bytes": size,
                "href": f"/v1/runs/{run_id}/artifacts/checks/{candidate.name}",
            }
        )
    return {
        "run_id": run_id,
        "kind": "checks",
        "artifacts": entries,
        "truncated": len(eligible) > len(entries),
    }


def _validated_check_artifact(directory: Path, filename: str) -> Path:
    if not _CHECK_ARTIFACT_RE.fullmatch(filename):
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    candidate = directory / filename
    if candidate.is_symlink():
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(directory)
    except (OSError, ValueError) as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc
    if not path.is_file():
        raise ApiError(404, "artifact_not_found", "Artifact not found")
    return path


def _validated_mount_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("static_path must be an absolute URL path")
    normalized = value.rstrip("/")
    if (
        not normalized.startswith("/")
        or normalized in {"", "/", "/healthz", "/readyz", "/v1"}
        or "{" in normalized
        or "}" in normalized
        or "//" in normalized
    ):
        raise ValueError("static_path must be a non-reserved absolute URL path")
    return normalized


def _service_call(
    operation: str,
    invoke: Callable[[], _ResultT],
) -> _ResultT:
    try:
        return invoke()
    except QueueFullError as exc:
        raise ApiError(
            429,
            "queue_full",
            "The run service queue is full",
            headers={"Retry-After": "1"},
        ) from exc
    except RepositoryNotAllowedError as exc:
        raise ApiError(
            403,
            "repository_not_allowed",
            "repo_path is outside the configured allowed roots",
        ) from exc
    except RunNotFoundError as exc:
        raise ApiError(404, "run_not_found", "Run not found") from exc
    except ArtifactError as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc
    except LookupError as exc:
        raise ApiError(404, "run_not_found", "Run not found") from exc
    except FileNotFoundError as exc:
        code = "artifact_not_found" if operation == "artifact" else "run_not_found"
        noun = "Artifact" if operation == "artifact" else "Run"
        raise ApiError(404, code, f"{noun} not found") from exc
    except InvalidRunTransitionError as exc:
        raise ApiError(
            409,
            "invalid_run_state",
            "The run state rejects this action",
        ) from exc
    except ValueError as exc:
        if operation == "create":
            raise ApiError(400, "invalid_run", "The run request was rejected") from exc
        raise ApiError(
            409,
            "invalid_run_state",
            "The run state rejects this action",
        ) from exc
    except TimeoutError as exc:
        raise ApiError(
            503,
            "service_unavailable",
            "The run service is unavailable",
        ) from exc
    except Exception as exc:
        raise ApiError(500, "service_error", "The run service failed") from exc


def _run_payload(value: object) -> dict[str, object]:
    try:
        raw = _object_mapping(value)
        selected = {name: raw[name] for name in _RUN_FIELDS if name in raw}
        record = RunRecord.model_validate(selected)
        payload = cast(dict[str, object], record.model_dump(mode="json"))
        _add_check_artifact_hrefs(payload, runs_path="/v1/runs")
        return payload
    except ApiError:
        raise
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid record",
        ) from exc


def _add_check_artifact_hrefs(
    payload: dict[str, object], *, runs_path: str
) -> None:
    """Attach links only for the canonical run-relative checks path contract."""

    run_id = payload.get("run_id")
    checks = payload.get("checks")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        return
    if not isinstance(checks, list):
        return
    for check in checks:
        if not isinstance(check, dict):
            continue
        relative_path = check.get("log_artifact")
        if not isinstance(relative_path, str):
            continue
        match = _CHECK_ARTIFACT_PATH_RE.fullmatch(relative_path)
        if match is None:
            continue
        filename = match.group(1)
        check["log_artifact_href"] = (
            f"{runs_path}/{run_id}/artifacts/checks/{filename}"
        )


def _object_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, RunRecord):
        return value.model_dump(mode="python")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return cast(Mapping[str, object], converted)
    if is_dataclass(value) and not isinstance(value, type):
        return cast(Mapping[str, object], asdict(value))
    attributes: dict[str, object] = {}
    for name in _RUN_FIELDS:
        if hasattr(value, name):
            attributes[name] = getattr(value, name)
    if attributes:
        return attributes
    raise TypeError("unsupported run record")


def _artifact_media_type(kind: ArtifactKind) -> str:
    return {
        "patch": "text/x-diff; charset=utf-8",
        "report": "text/markdown; charset=utf-8",
        "result": "application/json",
        "trace": "application/x-ndjson",
        "checks": "application/octet-stream",
    }[kind]


def _artifact_response(path: Path, *, media_type: str) -> Response:
    """Freeze an atomic artifact version before response headers are emitted."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ApiError(404, "artifact_not_found", "Artifact not found") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={
            **_response_headers(),
            "Content-Disposition": f'attachment; filename="{path.name}"',
        },
    )


def _response_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response_headers = _response_headers()
    response_headers.update(headers or {})
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
        headers=response_headers,
    )


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ApiError",
    "CreateRunRequest",
    "DecisionRequest",
    "RunServiceLike",
    "create_app",
    "serve",
]
