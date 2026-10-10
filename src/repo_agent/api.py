"""Authenticated FastAPI surface for durable repository-agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
import inspect
import json
import os
from pathlib import Path
import re
import secrets
import threading
from typing import Protocol, TypeVar, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .artifacts import ArtifactError
from .event_safety import (
    public_event_name,
    public_event_node,
    public_event_phase,
    public_event_timestamp,
)
from .persistence import MAX_EVENT_SEQUENCE, EventPage, RunNotFoundError
from .run_models import (
    ARTIFACT_FILENAMES,
    TERMINAL_STATUSES,
    ArtifactKind,
    RunRecord,
)
from .service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
    ServiceSnapshot,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
MAX_REQUEST_BYTES = 64 * 1024
REQUEST_BODY_TIMEOUT_SECONDS = 10.0
MAX_TASK_LENGTH = 4000
MAX_REASON_LENGTH = 2000
MAX_BASE_REF_LENGTH = 255
MAX_CHECK_ARTIFACTS = 256
MAX_RUN_LIST_LIMIT = 100
DEFAULT_RUN_LIST_LIMIT = 50
EVENT_PAGE_SIZE = 64
EVENT_HEARTBEAT_SECONDS = 10.0
EVENT_WAIT_SECONDS = 1.0
EVENT_RETRY_MILLISECONDS = 1500
MAX_EVENT_STREAMS = 4
_EVENT_HEARTBEAT = b": heartbeat\n\n"
_RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_RUN_LIST_LIMIT_RE = re.compile(r"^[1-9][0-9]{0,2}$")
_PUBLIC_RUN_STATUSES = frozenset(
    {
        "queued",
        "planning",
        "awaiting_approval",
        "running",
        "interrupted",
        "succeeded",
        "unverified",
        "failed",
        "cancelled",
        "policy_denied",
        "rejected",
    }
)
_CHECK_ARTIFACT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log$")
_CHECK_ARTIFACT_PATH_RE = re.compile(
    r"^checks/([a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\.log)$"
)
_HEALTH_PATHS = frozenset({"/healthz", "/readyz"})
_SERVICE_METHODS = (
    "create_run",
    "get_run",
    "list_runs",
    "decide",
    "resume",
    "cancel",
    "artifact_path",
    "ready",
    "scheduler_snapshot",
    "queue_position",
    "event_page",
    "wait_for_events",
)
_RUN_FIELDS = tuple(RunRecord.model_fields)


class _EventStreamLease:
    """Hold one stream slot until the stream and its thread calls are finished."""

    def __init__(self, slots: threading.BoundedSemaphore) -> None:
        self._slots = slots
        self._lock = threading.Lock()
        self._stream_released = False
        self._slot_released = False
        self._thread_calls = 0

    def retain_thread_call(self) -> None:
        with self._lock:
            if self._slot_released:
                raise RuntimeError("event stream lease is already released")
            self._thread_calls += 1

    def finish_thread_call(self) -> None:
        with self._lock:
            if self._thread_calls <= 0:  # pragma: no cover - invariant guard
                raise RuntimeError("event stream thread lease underflow")
            self._thread_calls -= 1
            self._release_slot_if_idle()

    def release(self) -> None:
        with self._lock:
            if self._stream_released:
                return
            self._stream_released = True
            self._release_slot_if_idle()

    def _release_slot_if_idle(self) -> None:
        if (
            self._stream_released
            and self._thread_calls == 0
            and not self._slot_released
        ):
            self._slot_released = True
            self._slots.release()


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

    def list_runs(self) -> tuple[object, ...]: ...

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

    def scheduler_snapshot(self) -> ServiceSnapshot: ...

    def queue_position(self, run_id: str) -> int | None: ...

    def event_page(
        self,
        run_id: str,
        *,
        after_sequence: int,
        limit: int = EVENT_PAGE_SIZE,
    ) -> EventPage: ...

    def wait_for_events(
        self,
        run_id: str,
        *,
        after_sequence: int,
        timeout: float | None,
        limit: int = EVENT_PAGE_SIZE,
    ) -> EventPage: ...


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


class _RequestBodyLimitMiddleware:
    """Buffer one small request body before application-level JSON parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = MAX_REQUEST_BYTES,
        timeout_seconds: float = REQUEST_BODY_TIMEOUT_SECONDS,
    ) -> None:
        self.app = app
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        length_values = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        if len(length_values) > 1:
            await self._reject(
                scope,
                receive,
                send,
                400,
                "invalid_request",
                "Request must contain at most one Content-Length header",
            )
            return

        declared_length: int | None = None
        if length_values:
            raw_length = length_values[0]
            if (
                not raw_length
                or len(raw_length) > 128
                or not raw_length.isdigit()
            ):
                await self._reject(
                    scope,
                    receive,
                    send,
                    400,
                    "invalid_request",
                    "Content-Length must be a non-negative decimal integer",
                )
                return
            significant_length = raw_length.lstrip(b"0") or b"0"
            maximum_length = str(self._max_bytes).encode("ascii")
            if (
                len(significant_length) > len(maximum_length)
                or len(significant_length) == len(maximum_length)
                and significant_length > maximum_length
            ):
                await self._reject_too_large(scope, receive, send)
                return
            declared_length = int(significant_length)

        messages: list[Message] = []
        total = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                await self._reject_timeout(scope, receive, send)
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError:
                await self._reject_timeout(scope, receive, send)
                return
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                await self._reject(
                    scope,
                    receive,
                    send,
                    400,
                    "invalid_request",
                    "Request body framing is invalid",
                )
                return
            body = message.get("body", b"")
            if not isinstance(body, bytes):
                await self._reject(
                    scope,
                    receive,
                    send,
                    400,
                    "invalid_request",
                    "Request body framing is invalid",
                )
                return
            total += len(body)
            if total > self._max_bytes:
                await self._reject_too_large(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        if declared_length is not None and total != declared_length:
            await self._reject(
                scope,
                receive,
                send,
                400,
                "invalid_request",
                "Request body length does not match Content-Length",
            )
            return

        index = 0

        async def replay() -> Message:
            nonlocal index
            if index >= len(messages):
                return await receive()
            message = messages[index]
            index += 1
            return message

        await self.app(scope, replay, send)

    async def _reject_too_large(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await self._reject(
            scope,
            receive,
            send,
            413,
            "request_too_large",
            f"Request body must be at most {self._max_bytes} bytes",
        )

    async def _reject_timeout(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await self._reject(
            scope,
            receive,
            send,
            408,
            "request_timeout",
            "Request body timed out",
        )

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        response = _error_response(status_code, code, message)
        await response(scope, receive, send)


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
    app.state.event_stream_slots = threading.BoundedSemaphore(MAX_EVENT_STREAMS)
    app.add_middleware(_RequestBodyLimitMiddleware)
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

    @app.get("/v1/service/status")
    def service_status() -> dict[str, object]:
        snapshot = _service_call("status", service.scheduler_snapshot)
        return _service_snapshot_payload(snapshot)

    @app.get("/v1/runs")
    def list_runs(
        status: str | None = None,
        limit: str = str(DEFAULT_RUN_LIST_LIMIT),
    ) -> dict[str, object]:
        normalized_status = _validated_list_status(status)
        normalized_limit = _validated_list_limit(limit)
        records = _service_call("list", service.list_runs)
        snapshot = _validated_service_snapshot(
            _service_call("status", service.scheduler_snapshot)
        )
        scheduler = _service_snapshot_payload(snapshot)
        if not isinstance(records, (tuple, list)):
            raise ApiError(
                502,
                "invalid_service_response",
                "The run service returned an invalid run list",
            )
        positions = {
            run_id: position
            for position, run_id in enumerate(snapshot.queued_run_ids, start=1)
        }
        selected: list[dict[str, object]] = []
        for value in reversed(records):
            payload = _run_payload(value)
            if normalized_status is not None and payload["status"] != normalized_status:
                continue
            payload["queue_position"] = positions.get(cast(str, payload["run_id"]))
            selected.append(payload)
            if len(selected) == normalized_limit:
                break
        return {"runs": selected, "service": scheduler}

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
        payload = _run_payload_with_position(service, result)
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
        return _run_payload_with_position(service, result)

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        normalized = _validated_run_id(run_id)
        requested_cursor = _validated_event_cursor(request)
        slots = cast(threading.BoundedSemaphore, app.state.event_stream_slots)
        if not slots.acquire(blocking=False):
            raise ApiError(
                429,
                "too_many_streams",
                "Too many live event streams",
                headers={"Retry-After": "2"},
            )
        lease = _EventStreamLease(slots)
        try:
            initial_after = requested_cursor or 0
            page = _validated_event_page(
                await _async_service_call(
                    "events",
                    lambda: service.event_page(
                        normalized,
                        after_sequence=initial_after,
                        limit=EVENT_PAGE_SIZE,
                    ),
                    lease=lease,
                ),
                after_sequence=initial_after,
            )
            record = _validated_event_record(
                await _async_service_call(
                    "get",
                    lambda: service.get_run(normalized),
                    lease=lease,
                ),
                run_id=normalized,
            )
        except BaseException:
            lease.release()
            raise
        reset = (
            requested_cursor is not None
            and requested_cursor > page.latest_sequence
        )
        cursor = (
            page.latest_sequence
            if requested_cursor is None or reset
            else requested_cursor
        )

        async def stream() -> AsyncIterator[bytes]:
            current_page = page
            current_record = record
            current_cursor = cursor
            loop = asyncio.get_running_loop()
            heartbeat_at = loop.time() + EVENT_HEARTBEAT_SECONDS
            try:
                yield _sse_event(
                    "snapshot",
                    _event_state(
                        current_record,
                        cursor=page.latest_sequence,
                        reset=reset,
                    ),
                    event_id=(
                        page.latest_sequence
                        if requested_cursor is None or reset
                        else None
                    ),
                    retry=EVENT_RETRY_MILLISECONDS,
                )
                while True:
                    for event in current_page.events:
                        sequence = cast(int, event["sequence"])
                        if sequence <= current_cursor:
                            continue
                        yield _sse_event(
                            "progress",
                            _public_event(normalized, event),
                            event_id=sequence,
                        )
                        current_cursor = sequence

                    if current_cursor < current_page.latest_sequence:
                        current_page = _validated_event_page(
                            await _thread_call(
                                lambda: service.event_page(
                                    normalized,
                                    after_sequence=current_cursor,
                                    limit=EVENT_PAGE_SIZE,
                                ),
                                lease=lease,
                            ),
                            after_sequence=current_cursor,
                        )
                        continue

                    current_record = _validated_event_record(
                        await _thread_call(
                            lambda: service.get_run(normalized),
                            lease=lease,
                        ),
                        run_id=normalized,
                    )
                    if _event_stream_stopped(current_record):
                        final_page = _validated_event_page(
                            await _thread_call(
                                lambda: service.event_page(
                                    normalized,
                                    after_sequence=current_cursor,
                                    limit=EVENT_PAGE_SIZE,
                                ),
                                lease=lease,
                            ),
                            after_sequence=current_cursor,
                        )
                        if final_page.events:
                            current_page = final_page
                            continue
                        yield _sse_event(
                            "done",
                            _event_state(current_record, cursor=current_cursor),
                            event_id=current_cursor,
                        )
                        return

                    current_page = _validated_event_page(
                        await _thread_call(
                            lambda: service.wait_for_events(
                                normalized,
                                after_sequence=current_cursor,
                                timeout=EVENT_WAIT_SECONDS,
                                limit=EVENT_PAGE_SIZE,
                            ),
                            lease=lease,
                        ),
                        after_sequence=current_cursor,
                    )
                    now = loop.time()
                    if not current_page.events and now >= heartbeat_at:
                        yield _EVENT_HEARTBEAT
                        heartbeat_at = now + EVENT_HEARTBEAT_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            finally:
                lease.release()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                **_response_headers(),
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(lease.release),
        )

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
        return _run_payload_with_position(service, result)

    @app.post("/v1/runs/{run_id}/resume")
    def resume(run_id: str) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "resume",
            lambda: service.resume(normalized),
        )
        return _run_payload_with_position(service, result)

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, object]:
        normalized = _validated_run_id(run_id)
        result = _service_call(
            "cancel",
            lambda: service.cancel(normalized),
        )
        return _run_payload_with_position(service, result)

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
    for artifact_kind in ARTIFACT_FILENAMES:
        if value == artifact_kind:
            return artifact_kind
    raise ApiError(404, "artifact_not_found", "Artifact not found")


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


async def _async_service_call(
    operation: str,
    invoke: Callable[[], _ResultT],
    *,
    lease: _EventStreamLease | None = None,
) -> _ResultT:
    return await _thread_call(
        lambda: _service_call(operation, invoke),
        lease=lease,
    )


async def _thread_call(
    invoke: Callable[[], _ResultT],
    *,
    lease: _EventStreamLease | None = None,
) -> _ResultT:
    if lease is not None:
        lease.retain_thread_call()
    try:
        pending = asyncio.create_task(asyncio.to_thread(invoke))
    except BaseException:
        if lease is not None:
            lease.finish_thread_call()
        raise
    if lease is not None:
        pending.add_done_callback(lambda _pending: lease.finish_thread_call())
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        raise


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


def _validated_event_cursor(request: Request) -> int | None:
    parameters = list(request.query_params.multi_items())
    if any(name != "after" for name, _value in parameters):
        raise ApiError(
            400,
            "invalid_cursor",
            "Only the after event cursor is supported",
        )
    query_values = [value for name, value in parameters if name == "after"]
    if len(query_values) > 1:
        raise ApiError(
            400,
            "invalid_cursor",
            "Event cursor must be supplied at most once",
        )

    raw_header_values = [
        value
        for name, value in request.scope.get("headers", ())
        if name.lower() == b"last-event-id"
    ]
    if len(raw_header_values) > 1:
        raise ApiError(
            400,
            "invalid_cursor",
            "Last-Event-ID must be supplied at most once",
        )
    try:
        header_values = [value.decode("ascii") for value in raw_header_values]
    except UnicodeDecodeError as exc:
        raise ApiError(
            400,
            "invalid_cursor",
            "Event cursor must be a decimal integer",
        ) from exc

    raw = (
        header_values[0]
        if header_values
        else query_values[0]
        if query_values
        else None
    )
    if raw is None:
        return None
    if re.fullmatch(r"[0-9]{1,19}", raw) is None:
        raise ApiError(
            400,
            "invalid_cursor",
            "Event cursor must be a decimal integer",
        )
    cursor = int(raw)
    if cursor > MAX_EVENT_SEQUENCE:
        raise ApiError(
            400,
            "invalid_cursor",
            "Event cursor is outside the supported range",
        )
    return cursor


def _validated_event_page(
    value: object,
    *,
    after_sequence: int,
) -> EventPage:
    if isinstance(value, EventPage):
        events_value: object = value.events
        latest_value: object = value.latest_sequence
    elif isinstance(value, Mapping):
        events_value = value.get("events")
        latest_value = value.get("latest_sequence")
    else:
        events_value = getattr(value, "events", None)
        latest_value = getattr(value, "latest_sequence", None)
    if (
        type(latest_value) is not int
        or not 0 <= latest_value <= MAX_EVENT_SEQUENCE
        or not isinstance(events_value, (tuple, list))
        or len(events_value) > EVENT_PAGE_SIZE
    ):
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid event page",
        )

    events: list[dict[str, object]] = []
    previous = after_sequence
    for raw_event in events_value:
        if not isinstance(raw_event, Mapping):
            raise ApiError(
                502,
                "invalid_service_response",
                "The run service returned an invalid event page",
            )
        sequence = raw_event.get("sequence")
        if (
            type(sequence) is not int
            or not 1 <= sequence <= latest_value
            or sequence <= previous
        ):
            raise ApiError(
                502,
                "invalid_service_response",
                "The run service returned an invalid event page",
            )
        event = {str(name): item for name, item in raw_event.items()}
        event["sequence"] = sequence
        events.append(event)
        previous = sequence
    if not events and latest_value > after_sequence:
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid event page",
        )
    return EventPage(tuple(events), latest_value)


def _validated_event_record(value: object, *, run_id: str) -> dict[str, object]:
    payload = _run_payload(value)
    if payload["run_id"] != run_id:
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid record",
        )
    return payload


def _public_event(
    run_id: str,
    event: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "sequence": event["sequence"],
        "event": public_event_name(event.get("event")),
    }
    for name, public_value in (
        ("node", public_event_node(event.get("node"))),
        ("phase", public_event_phase(event.get("phase"))),
        ("timestamp", public_event_timestamp(event.get("timestamp"))),
    ):
        if public_value is not None:
            payload[name] = public_value
    for name in (
        "executed_tool_calls",
        "tool_calls",
        "tool_errors",
        "parallel_tool_call_violations",
        "rejected_tool_call_violations",
        "recovered_protocol_errors",
    ):
        value = event.get(name)
        if type(value) is int and value >= 0:
            payload[name] = value
    request_not_started = event.get("request_not_started")
    if type(request_not_started) is bool:
        payload["request_not_started"] = request_not_started
    model_usage = _public_model_usage(event.get("model_usage"))
    if model_usage:
        payload["model_usage"] = model_usage
    return payload


def _public_model_usage(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    payload: dict[str, object] = {}
    for name in (
        "response_count",
        "reported_response_count",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        item = value.get(name)
        if type(item) is int and item >= 0:
            payload[name] = item
    complete = value.get("complete")
    if type(complete) is bool:
        payload["complete"] = complete
    return payload


def _event_state(
    payload: Mapping[str, object],
    *,
    cursor: int,
    reset: bool = False,
) -> dict[str, object]:
    state: dict[str, object] = {
        "run_id": payload["run_id"],
        "status": payload["status"],
        "cursor": cursor,
        "reset": reset,
    }
    node = public_event_node(payload.get("current_node"))
    if node is not None:
        state["node"] = node
    updated_at = public_event_timestamp(payload.get("updated_at"))
    if updated_at is not None:
        state["updated_at"] = updated_at
    return state


def _event_stream_stopped(payload: Mapping[str, object]) -> bool:
    return payload["status"] in TERMINAL_STATUSES or payload["status"] in {
        "awaiting_approval",
        "interrupted",
    }


def _sse_event(
    event: str,
    payload: Mapping[str, object],
    *,
    event_id: int | None = None,
    retry: int | None = None,
) -> bytes:
    fields: list[str] = []
    if retry is not None:
        fields.append(f"retry: {retry}")
    if event_id is not None:
        fields.append(f"id: {event_id}")
    fields.append(f"event: {event}")
    fields.append(
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return ("\n".join(fields) + "\n\n").encode("utf-8")


def _run_payload_with_position(
    service: RunServiceLike,
    value: object,
) -> dict[str, object]:
    payload = _run_payload(value)
    run_id = cast(str, payload["run_id"])
    raw_position = _service_call(
        "get",
        lambda: service.queue_position(run_id),
    )
    if raw_position is not None and (
        type(raw_position) is not int or raw_position < 1
    ):
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid queue position",
        )
    payload["queue_position"] = raw_position
    return payload


def _validated_list_status(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PUBLIC_RUN_STATUSES:
        raise ApiError(
            422,
            "invalid_request",
            "status must be a supported run status",
        )
    return value


def _validated_list_limit(value: object) -> int:
    if not isinstance(value, str) or _RUN_LIST_LIMIT_RE.fullmatch(value) is None:
        raise ApiError(
            422,
            "invalid_request",
            f"limit must be an integer from 1 to {MAX_RUN_LIST_LIMIT}",
        )
    converted = int(value)
    if converted > MAX_RUN_LIST_LIMIT:
        raise ApiError(
            422,
            "invalid_request",
            f"limit must be an integer from 1 to {MAX_RUN_LIST_LIMIT}",
        )
    return converted


def _validated_service_snapshot(value: object) -> ServiceSnapshot:
    if not isinstance(value, ServiceSnapshot):
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid scheduler snapshot",
        )
    if not isinstance(value.queued_run_ids, tuple) or not isinstance(
        value.active_run_ids, tuple
    ):
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid scheduler snapshot",
        )
    integer_fields = (
        value.queue_depth,
        value.queue_capacity,
        value.active_count,
        value.max_concurrency,
    )
    run_ids = value.queued_run_ids + value.active_run_ids
    if (
        type(value.ready) is not bool
        or any(type(item) is not int for item in integer_fields)
        or not 1 <= value.queue_capacity <= 64
        or not 0 <= value.queue_depth <= value.queue_capacity
        or value.max_concurrency != 1
        or not 0 <= value.active_count <= value.max_concurrency
        or value.queue_depth != len(value.queued_run_ids)
        or value.active_count != len(value.active_run_ids)
        or any(not isinstance(run_id, str) for run_id in run_ids)
        or len(set(run_ids)) != len(run_ids)
        or any(_RUN_ID_RE.fullmatch(run_id) is None for run_id in run_ids)
    ):
        raise ApiError(
            502,
            "invalid_service_response",
            "The run service returned an invalid scheduler snapshot",
        )
    return value


def _service_snapshot_payload(value: object) -> dict[str, object]:
    snapshot = _validated_service_snapshot(value)
    return {
        "ready": snapshot.ready,
        "queue_depth": snapshot.queue_depth,
        "queue_capacity": snapshot.queue_capacity,
        "active_count": snapshot.active_count,
        "max_concurrency": snapshot.max_concurrency,
        "queued_run_ids": list(snapshot.queued_run_ids),
        "active_run_ids": list(snapshot.active_run_ids),
    }


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
    "MAX_REQUEST_BYTES",
    "REQUEST_BODY_TIMEOUT_SECONDS",
    "ApiError",
    "CreateRunRequest",
    "DecisionRequest",
    "RunServiceLike",
    "create_app",
    "serve",
]
