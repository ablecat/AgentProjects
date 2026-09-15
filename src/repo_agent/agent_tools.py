"""The narrow model-facing tool surface for the durable workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from .checks import CheckRunner
from .models import ToolCall, ToolResult
from .policy import DEFAULT_PATH_POLICY
from .sandbox import DockerSandbox
from .tools import ToolDefinition


MAX_AGENT_FILE_RESULTS = 500
MAX_AGENT_SEARCH_RESULTS = 200
MAX_AGENT_BATCH_FILES = 4
MAX_AGENT_BATCH_LINES = 200
MAX_AGENT_BATCH_BYTES = 16 * 1024


AGENT_TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        "list_files",
        "List non-sensitive repository files, optionally under one glob.",
        {
            "type": "object",
            "properties": {
                "glob": {"type": "string", "minLength": 1, "maxLength": 256},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_AGENT_FILE_RESULTS,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "read_file",
        "Read a bounded line range from one non-sensitive repository file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 512},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "read_files",
        (
            "Read the first 200 lines from 1 to 4 non-sensitive repository files "
            "in one bounded response."
        ),
        {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "minItems": 1,
                    "maxItems": MAX_AGENT_BATCH_FILES,
                }
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "search_code",
        "Search repository text and return bounded matching line locations.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 512},
                "path": {"type": "string", "minLength": 1, "maxLength": 512},
                "glob": {"type": "string", "minLength": 1, "maxLength": 256},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_AGENT_SEARCH_RESULTS,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "apply_patch",
        "Apply a bounded Git-style unified diff only to the disposable candidate.",
        {
            "type": "object",
            "properties": {
                "unified_diff": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 65536,
                }
            },
            "required": ["unified_diff"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "get_diff",
        "Read the staged candidate diff against the fixed base commit.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 512}
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "run_check",
        "Run the pre-registered deterministic check in a fresh isolated copy.",
        {
            "type": "object",
            "properties": {
                "check_id": {"type": "string", "minLength": 1, "maxLength": 128}
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        "finish",
        "Finish only after inspecting the final diff and relevant check evidence.",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 4000},
                "risks": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 1000},
                    "maxItems": 10,
                },
            },
            "required": ["summary", "risks"],
            "additionalProperties": False,
        },
    ),
)


class AgentToolExecutor:
    """Translate model-facing calls to already-hardened local capabilities."""

    def __init__(
        self,
        sandbox: DockerSandbox,
        *,
        repo_path: str,
        run_id: str,
        allow_bootstrap: bool,
        phase_timeout_seconds: float = 300.0,
        total_timeout_seconds: float = 1200.0,
        max_output_bytes: int = 65536,
        mutation_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.repo_path = repo_path
        self.run_id = run_id
        self.allow_bootstrap = allow_bootstrap
        self.phase_timeout_seconds = phase_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.mutation_callback = mutation_callback
        self._mutation_persistence_failed = False

    @property
    def workspace_revision(self) -> int:
        return self.sandbox.workspace_revision

    @property
    def mutation_persistence_failed(self) -> bool:
        return self._mutation_persistence_failed

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            if call.name == "list_files":
                return self._list_files(call)
            if call.name == "read_file":
                return self._delegate(call, "read_file", dict(call.arguments))
            if call.name == "read_files":
                return self._read_files(call)
            if call.name == "search_code":
                return self._search_code(call)
            if call.name == "apply_patch":
                return self._apply_patch(call)
            if call.name == "get_diff":
                arguments = dict(call.arguments)
                arguments["staged"] = True
                return self._delegate(call, "git_diff", arguments)
            if call.name == "run_check":
                return self._run_check(call)
            if call.name == "finish":
                return ToolResult(
                    call.id,
                    call.name,
                    False,
                    "",
                    error="finish is handled by the workflow loop",
                )
            return ToolResult(
                call.id,
                call.name,
                False,
                "",
                error=f"Unknown model tool: {call.name}",
            )
        except Exception as exc:
            detail = str(exc).strip()
            return ToolResult(
                call.id,
                call.name,
                False,
                "",
                error=f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__,
            )

    def _list_files(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"glob", "limit"})
        limit = _bounded_int(
            arguments.pop("limit", MAX_AGENT_FILE_RESULTS),
            "limit",
            MAX_AGENT_FILE_RESULTS,
        )
        result = self._delegate(call, "list_files", arguments)
        return _limit_lines(result, limit)

    def _search_code(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"query", "path", "glob", "limit"})
        query = arguments.pop("query", None)
        if not isinstance(query, str) or not query:
            raise ValueError("search_code requires a non-empty query")
        limit = _bounded_int(
            arguments.pop("limit", MAX_AGENT_SEARCH_RESULTS),
            "limit",
            MAX_AGENT_SEARCH_RESULTS,
        )
        path = arguments.pop("path", None)
        glob = arguments.pop("glob", None)
        if path is not None:
            safe_path = DEFAULT_PATH_POLICY.validate(path, access="discover")
            if glob is None:
                glob = safe_path if PurePosixPath(safe_path).suffix else f"{safe_path}/**"
            else:
                if not isinstance(glob, str) or not glob:
                    raise ValueError("glob must be non-empty text")
                glob = f"{safe_path}/{glob}"
        delegated: dict[str, object] = {"pattern": query}
        if glob is not None:
            delegated["glob"] = glob
        result = self._delegate(call, "search", delegated)
        return _limit_lines(result, limit)

    def _read_files(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"paths"})
        raw_paths = arguments.get("paths")
        if not isinstance(raw_paths, list):
            raise ValueError("read_files paths must be an array")
        if not 1 <= len(raw_paths) <= MAX_AGENT_BATCH_FILES:
            raise ValueError(
                f"read_files requires 1 to {MAX_AGENT_BATCH_FILES} paths"
            )
        paths = [DEFAULT_PATH_POLICY.validate(path, access="read") for path in raw_paths]
        if len(set(paths)) != len(paths):
            raise ValueError("read_files paths must be unique")

        sections: list[str] = []
        failures: list[str] = []
        exit_code: int | None = 0
        truncated = False
        for path in paths:
            result = self._delegate(
                call,
                "read_file",
                {"path": path, "start_line": 1, "end_line": MAX_AGENT_BATCH_LINES},
            )
            sections.append(f"===== {path} =====\n{result.output}")
            truncated = truncated or result.truncated
            if not result.ok:
                failures.append(f"{path}: {result.error or 'read failed'}")
                if exit_code == 0:
                    exit_code = result.exit_code

        output, output_truncated = _limit_utf8_bytes(
            "\n".join(sections), MAX_AGENT_BATCH_BYTES
        )
        return ToolResult(
            call.id,
            call.name,
            not failures,
            output,
            error="; ".join(failures) or None,
            exit_code=exit_code,
            truncated=truncated or output_truncated,
        )

    def _apply_patch(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"unified_diff"})
        if "unified_diff" not in arguments:
            raise ValueError("apply_patch requires unified_diff")
        result = self._delegate(
            call,
            "apply_patch",
            {"patch": arguments["unified_diff"]},
        )
        if result.ok and self.mutation_callback is not None:
            try:
                candidate = self.sandbox.candidate_artifact()
                self.mutation_callback("" if candidate is None else candidate.patch)
            except Exception as exc:
                self._mutation_persistence_failed = True
                detail = str(exc).strip()
                raise RuntimeError(
                    "candidate patch persistence failed"
                    + (f": {detail}" if detail else "")
                ) from exc
        return result

    def _run_check(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"check_id"})
        check_id = arguments.get("check_id")
        if check_id is not None and not isinstance(check_id, str):
            raise ValueError("check_id must be text")
        candidate = self.sandbox.candidate_artifact()
        patch = None if candidate is None else candidate.patch
        result = CheckRunner(
            self.repo_path,
            candidate_patch=patch,
            allow_bootstrap=self.allow_bootstrap,
            phase_timeout_seconds=self.phase_timeout_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            container_name_prefix=f"repo-agent-model-{self.run_id[:12]}",
        ).run(check_id)
        output = json.dumps(
            asdict(result),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return ToolResult(
            call.id,
            call.name,
            result.ok,
            output,
            error=result.error,
            exit_code=0 if result.ok else 1,
            truncated=any(phase.truncated for phase in result.phases),
        )

    def _delegate(
        self,
        call: ToolCall,
        name: str,
        arguments: Mapping[str, object],
    ) -> ToolResult:
        result = self.sandbox.execute(ToolCall(call.id, name, arguments))
        return ToolResult(
            call.id,
            call.name,
            result.ok,
            result.output,
            error=result.error,
            exit_code=result.exit_code,
            truncated=result.truncated,
        )


def _arguments(call: ToolCall, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(call.arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    unexpected = sorted(set(call.arguments) - allowed)
    if unexpected:
        raise ValueError(f"unexpected argument(s): {', '.join(unexpected)}")
    return dict(call.arguments)


def _bounded_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _limit_lines(result: ToolResult, limit: int) -> ToolResult:
    lines = result.output.splitlines()
    if len(lines) <= limit:
        return result
    output = "\n".join(lines[:limit])
    if result.output.endswith("\n"):
        output += "\n"
    return ToolResult(
        result.call_id,
        result.name,
        result.ok,
        output,
        error=result.error,
        exit_code=result.exit_code,
        truncated=True,
    )


def _limit_utf8_bytes(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


__all__ = [
    "AGENT_TOOL_DEFINITIONS",
    "AgentToolExecutor",
    "MAX_AGENT_BATCH_BYTES",
    "MAX_AGENT_BATCH_FILES",
    "MAX_AGENT_BATCH_LINES",
    "MAX_AGENT_FILE_RESULTS",
    "MAX_AGENT_SEARCH_RESULTS",
]
