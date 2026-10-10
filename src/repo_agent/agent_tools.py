"""The narrow model-facing tool surface for the durable workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import difflib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping
import unicodedata

from .checks import CheckRunner
from .models import ToolCall, ToolResult
from .patches import PatchValidationError, normalize_model_patch
from .policy import DEFAULT_PATH_POLICY
from .sandbox import DockerSandbox
from .tools import ToolDefinition


MAX_AGENT_FILE_RESULTS = 500
MAX_AGENT_SEARCH_RESULTS = 200
MAX_AGENT_BATCH_FILES = 4
MAX_AGENT_BATCH_LINES = 200
MAX_AGENT_BATCH_BYTES = 16 * 1024
MAX_AGENT_WRITE_FILE_BYTES = MAX_AGENT_BATCH_BYTES
MAX_AGENT_WRITE_TOTAL_BYTES = 48 * 1024
MAX_AGENT_WRITE_FILE_LINES = MAX_AGENT_BATCH_LINES
MAX_AGENT_WRITE_TOTAL_LINES = MAX_AGENT_BATCH_FILES * MAX_AGENT_BATCH_LINES


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
            "Read the complete contents of 1 to 4 non-sensitive repository files "
            "when each file fits within 200 lines and the batch fits within 16 KiB. "
            "Oversized batches fail closed instead of returning an editable prefix. "
            "Call exactly as "
            "{\"paths\":[\"existing/path.py\"]}: paths must be a unique array "
            "of existing repository-relative files, never a bare string, an extra "
            "key, or a newly planned regression-test path."
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
        (
            "Apply a bounded Git-style unified diff only to the disposable candidate. "
            "The sole argument key is unified_diff; do not use patch, diff, or content. "
            "A newly added regression test must use `new file mode 100644`, "
            "`--- /dev/null`, and `+++ b/tests/test_*_regression.py`; do not add "
            "tests/__init__.py or mark a new file as a modification."
        ),
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
        "write_files",
        (
            "Create or replace 1 to 4 UTF-8 text files in the disposable candidate "
            "using their complete contents. Call exactly as "
            '{"files":[{"path":"safe/relative.py","content":"complete file\\n"}]}. '
            "Each path must be unique and repository-relative. Content uses LF line "
            "endings; a final LF is optional and preserved. Deletion is not supported."
        ),
        {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 512,
                            },
                            "content": {
                                "type": "string",
                                "maxLength": MAX_AGENT_WRITE_FILE_BYTES,
                            },
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": MAX_AGENT_BATCH_FILES,
                }
            },
            "required": ["files"],
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
        require_complete_read_before_write: bool = False,
    ) -> None:
        self.sandbox = sandbox
        self.repo_path = repo_path
        self.run_id = run_id
        self.allow_bootstrap = allow_bootstrap
        self.phase_timeout_seconds = phase_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.mutation_callback = mutation_callback
        if type(require_complete_read_before_write) is not bool:
            raise ValueError("require_complete_read_before_write must be a boolean")
        self.require_complete_read_before_write = require_complete_read_before_write
        self._complete_read_paths: set[str] = set()
        self._mutation_persistence_failed = False

    @property
    def workspace_revision(self) -> int:
        return self.sandbox.workspace_revision

    @property
    def mutation_persistence_failed(self) -> bool:
        return self._mutation_persistence_failed

    def confirm_complete_read(self, call: ToolCall, result: ToolResult) -> None:
        """Authorize writes only after the exact read result reached the model intact."""

        if call.name != "read_files" or not result.ok or result.truncated:
            return
        arguments = _arguments(call, {"paths"})
        raw_paths = arguments.get("paths")
        if not isinstance(raw_paths, list):
            return
        paths = [DEFAULT_PATH_POLICY.validate(path, access="read") for path in raw_paths]
        if len(paths) != len(set(paths)):
            return
        self._complete_read_paths.update(paths)

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
            if call.name == "write_files":
                return self._write_files(call)
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

        repository = getattr(self.sandbox, "snapshot_path", None)
        if isinstance(repository, Path):
            targets: list[tuple[str, Path]] = []
            for path in paths:
                target = DEFAULT_PATH_POLICY.validate_disk_path(
                    repository, path, access="read"
                )
                target_stat = os.lstat(target)
                if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_nlink > 1:
                    raise ValueError(
                        f"read_files target is not a safe regular file: {path}"
                    )
                if target_stat.st_size > MAX_AGENT_WRITE_FILE_BYTES:
                    raise ValueError(
                        f"read_files target {path} exceeds the complete-file limit of "
                        f"{MAX_AGENT_WRITE_FILE_BYTES} bytes"
                    )
                targets.append((path, target))

            preview_sections: list[str] = []
            for path, target in targets:
                try:
                    raw_content = target.read_bytes()
                    content = raw_content.decode("utf-8")
                except (OSError, UnicodeError) as exc:
                    raise ValueError(
                        f"read_files target is not readable UTF-8 text: {path}"
                    ) from exc
                if len(raw_content) > MAX_AGENT_WRITE_FILE_BYTES:
                    raise ValueError(
                        f"read_files target {path} exceeds the complete-file limit of "
                        f"{MAX_AGENT_WRITE_FILE_BYTES} bytes"
                    )
                if "\x00" in content or "\r" in content:
                    raise ValueError(
                        f"read_files target must be LF-delimited UTF-8 text: {path}"
                    )
                if _logical_line_count(content) > MAX_AGENT_BATCH_LINES:
                    raise ValueError(
                        f"read_files target {path} exceeds the complete-file limit of "
                        f"{MAX_AGENT_BATCH_LINES} lines"
                    )
                preview_sections.append(f"===== {path} =====\n{content}")
            preview = "\n".join(preview_sections)
            if len(preview.encode("utf-8")) > MAX_AGENT_BATCH_BYTES:
                raise ValueError(
                    "read_files batch exceeds the 16384-byte complete-output limit; "
                    "retry with fewer paths"
                )

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
        raw_patch = arguments["unified_diff"]
        # Production sandboxes expose a disposable snapshot so model patches can
        # be canonicalized and checked against the exact checkout. Keep the
        # executor usable with lightweight test doubles that only implement the
        # tool delegation protocol; those doubles do not have a repository to
        # validate against and retain the historical pass-through behavior.
        repository = getattr(self.sandbox, "snapshot_path", None)
        if repository is None:
            canonical_text = raw_patch
        else:
            try:
                canonical_text = normalize_model_patch(
                    raw_patch, repository=repository
                ).text
            except PatchValidationError as exc:
                raise ValueError(
                    "patch rejected: "
                    f"{exc}; submit a complete diff --git patch with matching hunk counts"
                ) from exc
        result = self._delegate(
            call,
            "apply_patch",
            {"patch": canonical_text},
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

    def _write_files(self, call: ToolCall) -> ToolResult:
        arguments = _arguments(call, {"files"})
        raw_files = arguments.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("write_files files must be an array")
        if not 1 <= len(raw_files) <= MAX_AGENT_BATCH_FILES:
            raise ValueError(
                f"write_files requires 1 to {MAX_AGENT_BATCH_FILES} files"
            )

        repository = getattr(self.sandbox, "snapshot_path", None)
        if not isinstance(repository, Path):
            raise RuntimeError("write_files requires an active candidate snapshot")

        preflight: list[tuple[str, str, Path, bool]] = []
        seen: set[str] = set()
        total_bytes = 0
        total_lines = 0
        for index, raw_file in enumerate(raw_files, 1):
            if not isinstance(raw_file, Mapping):
                raise ValueError(f"write_files entry {index} must be an object")
            if any(not isinstance(key, str) for key in raw_file):
                raise ValueError(f"write_files entry {index} fields must be text")
            unexpected = sorted(set(raw_file) - {"path", "content"})
            if unexpected:
                raise ValueError(
                    "write_files entry "
                    f"{index} has unexpected field(s): {', '.join(unexpected)}"
                )
            if "path" not in raw_file or "content" not in raw_file:
                raise ValueError(
                    f"write_files entry {index} requires path and content"
                )

            path = DEFAULT_PATH_POLICY.validate(raw_file["path"], access="write")
            identity = _portable_path_identity(path)
            if identity in seen:
                raise ValueError("write_files paths must be unique")
            seen.add(identity)

            content = raw_file["content"]
            if not isinstance(content, str):
                raise ValueError(f"write_files content for {path} must be text")
            if "\x00" in content:
                raise ValueError(f"write_files content for {path} must not contain NUL")
            if "\r" in content:
                raise ValueError(
                    f"write_files content for {path} must use LF line endings"
                )
            try:
                content_bytes = content.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"write_files content for {path} must be valid UTF-8 text"
                ) from exc
            if len(content_bytes) > MAX_AGENT_WRITE_FILE_BYTES:
                raise ValueError(
                    f"write_files content for {path} exceeds "
                    f"{MAX_AGENT_WRITE_FILE_BYTES} bytes"
                )
            line_count = _logical_line_count(content)
            if line_count > MAX_AGENT_WRITE_FILE_LINES:
                raise ValueError(
                    f"write_files content for {path} exceeds "
                    f"{MAX_AGENT_WRITE_FILE_LINES} lines"
                )
            total_bytes += len(content_bytes)
            total_lines += line_count
            if total_bytes > MAX_AGENT_WRITE_TOTAL_BYTES:
                raise ValueError(
                    "write_files total content exceeds "
                    f"{MAX_AGENT_WRITE_TOTAL_BYTES} bytes"
                )
            if total_lines > MAX_AGENT_WRITE_TOTAL_LINES:
                raise ValueError(
                    "write_files total content exceeds "
                    f"{MAX_AGENT_WRITE_TOTAL_LINES} lines"
                )

            target = DEFAULT_PATH_POLICY.validate_disk_path(
                repository,
                path,
                access="write",
                allow_missing_leaf=True,
            )
            _reject_case_colliding_path(repository, path)
            exists = target.exists()
            if exists:
                target_stat = os.lstat(target)
                if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_nlink > 1:
                    raise ValueError(f"write_files target is not a safe regular file: {path}")
                if (
                    self.require_complete_read_before_write
                    and path not in self._complete_read_paths
                ):
                    raise ValueError(
                        f"write_files target must be read completely before editing: {path}"
                    )
                if target_stat.st_size > MAX_AGENT_WRITE_FILE_BYTES:
                    raise ValueError(
                        f"write_files target {path} exceeds "
                        f"{MAX_AGENT_WRITE_FILE_BYTES} bytes"
                    )
            elif not content:
                raise ValueError(f"write_files cannot create an empty file: {path}")
            preflight.append((path, content, target, exists))

        # Validate the complete batch, including every disk target, before reading
        # any repository content. A malicious later entry must not turn an otherwise
        # rejected request into a partial repository read.
        writes: list[tuple[str, str, str, bool]] = []
        for path, content, target, exists in preflight:
            if exists:
                try:
                    before_bytes = target.read_bytes()
                    if len(before_bytes) > MAX_AGENT_WRITE_FILE_BYTES:
                        raise ValueError(
                            f"write_files target {path} exceeds "
                            f"{MAX_AGENT_WRITE_FILE_BYTES} bytes"
                        )
                    before = before_bytes.decode("utf-8")
                except (OSError, UnicodeError) as exc:
                    raise ValueError(
                        f"write_files target is not readable UTF-8 text: {path}"
                    ) from exc
                if "\x00" in before or "\r" in before:
                    raise ValueError(
                        f"write_files target must be LF-delimited UTF-8 text: {path}"
                    )
                if _logical_line_count(before) > MAX_AGENT_WRITE_FILE_LINES:
                    raise ValueError(
                        f"write_files target {path} exceeds "
                        f"{MAX_AGENT_WRITE_FILE_LINES} lines"
                    )
                if before == content:
                    continue
            else:
                before = ""
            writes.append((path, before, content, exists))

        if not writes:
            raise ValueError("write_files makes no changes")
        rendered = [
            _render_full_file_diff(path, before, content, exists=exists)
            for path, before, content, exists in sorted(writes, key=lambda item: item[0])
        ]
        unified_diff = "".join(rendered)
        result = self._apply_patch(
            ToolCall(call.id, call.name, {"unified_diff": unified_diff})
        )
        if result.ok:
            self._complete_read_paths.update(path for path, *_rest in writes)
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


def _portable_path_identity(path: str) -> str:
    return "/".join(
        unicodedata.normalize("NFC", part).casefold()
        for part in PurePosixPath(path).parts
    )


def _reject_case_colliding_path(repository: Path, path: str) -> None:
    current = repository
    for part in PurePosixPath(path).parts:
        try:
            entries = tuple(current.iterdir())
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"write_files path cannot be inspected: {path}") from exc
        identity = unicodedata.normalize("NFC", part).casefold()
        matches = [
            entry.name
            for entry in entries
            if unicodedata.normalize("NFC", entry.name).casefold() == identity
        ]
        if len(matches) > 1 or (matches and part not in matches):
            raise ValueError(f"write_files path collides with an existing path: {path}")
        if not matches:
            return
        current /= part


def _logical_line_count(value: str) -> int:
    if not value:
        return 0
    return value.count("\n") + int(not value.endswith("\n"))


def _lf_lines(value: str) -> list[str]:
    if not value:
        return []
    parts = value.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _render_full_file_diff(
    path: str, before: str, after: str, *, exists: bool
) -> str:
    body = difflib.unified_diff(
        _lf_lines(before),
        _lf_lines(after),
        fromfile=f"a/{path}" if exists else "/dev/null",
        tofile=f"b/{path}",
        n=3,
        lineterm="\n",
    )
    rendered: list[str] = [f"diff --git a/{path} b/{path}\n"]
    if not exists:
        rendered.append("new file mode 100644\n")
    for line in body:
        rendered.append(line if line.endswith("\n") else line + "\n")
        if not line.endswith("\n") and line[:1] in {" ", "+", "-"}:
            rendered.append("\\ No newline at end of file\n")
    if len(rendered) <= (2 if not exists else 1):
        raise ValueError(f"write_files makes no change to {path}")
    return "".join(rendered)


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
    "MAX_AGENT_WRITE_FILE_BYTES",
    "MAX_AGENT_WRITE_FILE_LINES",
    "MAX_AGENT_WRITE_TOTAL_BYTES",
    "MAX_AGENT_WRITE_TOTAL_LINES",
]
