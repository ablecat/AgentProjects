"""Validated repository inspection and candidate-edit tools."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

from .models import ToolCall
from .patches import (
    MAX_PATCH_BYTES,
    PatchValidationError,
    ValidatedPatch,
    validate_patch,
)
from .policy import (
    DEFAULT_PATH_POLICY,
    RepositoryPathError,
    SENSITIVE_DISCOVERY_GLOBS,
    is_sensitive_name,
)


WORKSPACE = "/workspace"
MAX_PATH_LENGTH = 512
MAX_GLOB_LENGTH = 256
MAX_PATTERN_LENGTH = 512
MAX_READ_LINES = 400
MAX_LINE_NUMBER = 10_000_000


class ToolValidationError(ValueError):
    """Raised when a model-generated tool call is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """A shell-free command and the exit codes the runner may accept."""

    argv: tuple[str, ...]
    allowed_exit_codes: frozenset[int] = frozenset({0})
    cwd: str = WORKSPACE


@dataclass(frozen=True, slots=True)
class RepoMapSpec:
    """A host-side, non-executing repository map request."""

    path: str | None
    max_files: int
    include_symbols: bool


@dataclass(frozen=True, slots=True)
class ApplyPatchSpec:
    """A validated patch that must be streamed to an isolated apply process."""

    patch: ValidatedPatch


ToolSpec: TypeAlias = CommandSpec | RepoMapSpec | ApplyPatchSpec


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral function tool metadata backed by JSON Schema."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def to_provider_dict(self) -> dict[str, Any]:
        """Return an OpenAI Responses-compatible function definition."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": deepcopy(dict(self.input_schema)),
            # Optional arguments are intentional; OpenAI strict mode would require
            # every property to appear in ``required``.
            "strict": False,
        }


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="git_status",
        description="Show the repository branch and concise working-tree status.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="git_diff",
        description=(
            "Show all candidate changes against HEAD, or only staged changes, "
            "optionally for one safe path."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Read the staged diff instead of the unstaged diff.",
                    "default": False,
                },
                "path": {
                    "type": "string",
                    "description": "Optional repository-relative file or directory path.",
                    "minLength": 1,
                    "maxLength": MAX_PATH_LENGTH,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="list_files",
        description="List repository files, optionally constrained by a safe glob.",
        input_schema={
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": "Optional ripgrep glob such as src/**/*.py.",
                    "minLength": 1,
                    "maxLength": MAX_GLOB_LENGTH,
                }
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="search",
        description="Search non-secret repository files and return matching line locations.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Required regular expression or literal search text.",
                    "minLength": 1,
                    "maxLength": MAX_PATTERN_LENGTH,
                },
                "glob": {
                    "type": "string",
                    "description": "Optional ripgrep glob such as *.py.",
                    "minLength": 1,
                    "maxLength": MAX_GLOB_LENGTH,
                },
                "fixed_strings": {
                    "type": "boolean",
                    "description": "Treat pattern as literal text instead of a regex.",
                    "default": False,
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="read_file",
        description=f"Read at most {MAX_READ_LINES} lines from one safe repository file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Required repository-relative file path.",
                    "minLength": 1,
                    "maxLength": MAX_PATH_LENGTH,
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read, inclusive and one-based.",
                    "minimum": 1,
                    "maximum": MAX_LINE_NUMBER,
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Last line to read, inclusive and one-based. When omitted, "
                        f"read at most {MAX_READ_LINES} lines from start_line."
                    ),
                    "minimum": 1,
                    "maximum": MAX_LINE_NUMBER,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="repo_map",
        description=(
            "Build a bounded repository tree, language/manifest summary, and "
            "Python symbol index without executing repository code."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional safe repository-relative subtree.",
                    "minLength": 1,
                    "maxLength": MAX_PATH_LENGTH,
                },
                "max_files": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 500,
                },
                "include_symbols": {
                    "type": "boolean",
                    "default": True,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDefinition(
        name="apply_patch",
        description=(
            "Apply one bounded Git-style unified diff to the disposable candidate "
            "workspace. The original repository is never changed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Git-style unified diff for safe text files.",
                    "minLength": 1,
                    "maxLength": MAX_PATCH_BYTES,
                }
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    ),
)

_GLOB_META = frozenset("*?[]{}!")
_SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")

# These exclusions also protect calls without a user-supplied path or glob.
_RG_SECRET_EXCLUDES = SENSITIVE_DISCOVERY_GLOBS
_GIT_SECRET_EXCLUDES: tuple[str, ...] = tuple(
    f":(exclude,glob,icase){glob[1:]}" for glob in _RG_SECRET_EXCLUDES
)


def build_tool_command(call: ToolCall) -> ToolSpec:
    """Validate a tool call and translate it into a non-shell command."""

    arguments = _arguments(call)

    if call.name == "git_status":
        _reject_unknown(arguments, allowed=frozenset())
        return CommandSpec(
            argv=(
                "git",
                "-c",
                f"safe.directory={WORKSPACE}",
                "status",
                "--short",
                "--branch",
                "--untracked-files=all",
            )
        )

    if call.name == "git_diff":
        return _build_git_diff(arguments)
    if call.name == "list_files":
        return _build_list_files(arguments)
    if call.name == "search":
        return _build_search(arguments)
    if call.name == "read_file":
        return _build_read_file(arguments)
    if call.name == "repo_map":
        return _build_repo_map(arguments)
    if call.name == "apply_patch":
        return _build_apply_patch(arguments)

    raise ToolValidationError(f"unknown tool: {call.name!r}")


def _build_git_diff(arguments: Mapping[str, object]) -> CommandSpec:
    _reject_unknown(arguments, allowed=frozenset({"staged", "path"}))
    staged = _optional_bool(arguments, "staged", default=False)
    raw_path = arguments.get("path")
    path = None if raw_path is None else _safe_relative_path(raw_path)

    argv = ["git", "-c", f"safe.directory={WORKSPACE}"]
    if path is not None:
        argv.append("--literal-pathspecs")
    argv.extend(("diff", "--no-ext-diff", "--no-textconv", "--no-color"))
    if staged:
        argv.append("--cached")
    else:
        argv.append("HEAD")
    argv.append("--")
    if path is None:
        argv.append(".")
        argv.extend(_GIT_SECRET_EXCLUDES)
    else:
        argv.append(path)
    return CommandSpec(argv=tuple(argv))


def _build_list_files(arguments: Mapping[str, object]) -> CommandSpec:
    _reject_unknown(arguments, allowed=frozenset({"glob"}))
    raw_glob = arguments.get("glob")
    glob = None if raw_glob is None else _safe_glob(raw_glob)

    argv = ["rg", "--files", "--hidden"]
    if glob is not None:
        argv.extend(("--glob", glob))
    _append_rg_excludes(argv)
    return CommandSpec(argv=tuple(argv))


def _build_search(arguments: Mapping[str, object]) -> CommandSpec:
    _reject_unknown(
        arguments, allowed=frozenset({"pattern", "glob", "fixed_strings"})
    )
    if "pattern" not in arguments:
        raise ToolValidationError("search requires 'pattern'")
    pattern = _bounded_string(
        arguments["pattern"], name="pattern", limit=MAX_PATTERN_LENGTH
    )
    if not pattern:
        raise ToolValidationError("pattern must not be empty")
    glob_value = arguments.get("glob")
    glob = None if glob_value is None else _safe_glob(glob_value)
    fixed_strings = _optional_bool(arguments, "fixed_strings", default=False)

    argv = [
        "rg",
        "--line-number",
        "--column",
        "--no-heading",
        "--hidden",
        "--color",
        "never",
    ]
    if fixed_strings:
        argv.append("--fixed-strings")
    if glob is not None:
        argv.extend(("--glob", glob))
    _append_rg_excludes(argv)
    argv.extend(("--", pattern, "."))
    return CommandSpec(argv=tuple(argv), allowed_exit_codes=frozenset({0, 1}))


def _build_read_file(arguments: Mapping[str, object]) -> CommandSpec:
    _reject_unknown(
        arguments, allowed=frozenset({"path", "start_line", "end_line"})
    )
    if "path" not in arguments:
        raise ToolValidationError("read_file requires 'path'")
    path = _safe_relative_path(arguments["path"])
    start_line = _optional_positive_int(arguments, "start_line", default=1)
    end_line = _optional_positive_int(
        arguments,
        "end_line",
        default=min(start_line + MAX_READ_LINES - 1, MAX_LINE_NUMBER),
    )
    if end_line < start_line:
        raise ToolValidationError("end_line must be greater than or equal to start_line")
    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ToolValidationError(f"read_file is limited to {MAX_READ_LINES} lines")

    return CommandSpec(argv=("sed", "-n", f"{start_line},{end_line}p", "--", path))


def _build_repo_map(arguments: Mapping[str, object]) -> RepoMapSpec:
    _reject_unknown(
        arguments, allowed=frozenset({"path", "max_files", "include_symbols"})
    )
    raw_path = arguments.get("path")
    path = None if raw_path is None else _safe_relative_path(raw_path)
    max_files = _optional_bounded_int(arguments, "max_files", default=500, maximum=500)
    include_symbols = _optional_bool(arguments, "include_symbols", default=True)
    return RepoMapSpec(path, max_files, include_symbols)


def _build_apply_patch(arguments: Mapping[str, object]) -> ApplyPatchSpec:
    _reject_unknown(arguments, allowed=frozenset({"patch"}))
    if "patch" not in arguments:
        raise ToolValidationError("apply_patch requires 'patch'")
    try:
        patch = validate_patch(arguments["patch"])
    except PatchValidationError as exc:
        raise ToolValidationError(str(exc)) from exc
    return ApplyPatchSpec(patch)


def _arguments(call: ToolCall) -> Mapping[str, object]:
    arguments = call.arguments
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("tool arguments must be an object")
    if not all(isinstance(key, str) for key in arguments):
        raise ToolValidationError("tool argument names must be strings")
    return arguments


def _reject_unknown(
    arguments: Mapping[str, object], *, allowed: frozenset[str]
) -> None:
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        raise ToolValidationError(f"unexpected argument(s): {', '.join(unexpected)}")


def _optional_bool(
    arguments: Mapping[str, object], name: str, *, default: bool
) -> bool:
    value = arguments.get(name, default)
    if type(value) is not bool:
        raise ToolValidationError(f"{name} must be a boolean")
    return value


def _optional_positive_int(
    arguments: Mapping[str, object], name: str, *, default: int
) -> int:
    value = arguments.get(name, default)
    if type(value) is not int or value < 1 or value > MAX_LINE_NUMBER:
        raise ToolValidationError(
            f"{name} must be an integer from 1 to {MAX_LINE_NUMBER}"
        )
    return value


def _optional_bounded_int(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ToolValidationError(
            f"{name} must be an integer from 1 to {maximum}"
        )
    return value


def _bounded_string(value: object, *, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ToolValidationError(f"{name} must be a string")
    if "\x00" in value:
        raise ToolValidationError(f"{name} must not contain NUL")
    if len(value) > limit:
        raise ToolValidationError(f"{name} must be at most {limit} characters")
    return value


def _safe_relative_path(value: object) -> str:
    try:
        return DEFAULT_PATH_POLICY.validate(value, access="read")
    except RepositoryPathError as exc:
        raise ToolValidationError(str(exc)) from exc


def _safe_glob(value: object) -> str:
    glob = _bounded_string(value, name="glob", limit=MAX_GLOB_LENGTH)
    if not glob:
        raise ToolValidationError("glob must not be empty")
    if "\\" in glob:
        raise ToolValidationError("glob must use forward slashes")
    if glob.startswith("/") or (len(glob) >= 2 and glob[0].isalpha() and glob[1] == ":"):
        raise ToolValidationError("glob must be repository-relative")

    parts = glob.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ToolValidationError("glob must not contain traversal")
    for part in parts:
        folded = part.casefold()
        if folded == ".git" or folded.startswith(".git/"):
            raise ToolValidationError("glob must not target .git internals")
        literal = "".join(character for character in folded if character not in _GLOB_META)
        if is_sensitive_name(literal) or is_sensitive_name(literal.rstrip(".")):
            raise ToolValidationError("glob must not target suspected secret files")
        if folded.startswith(".env") or folded.endswith(_SECRET_SUFFIXES):
            raise ToolValidationError("glob must not target suspected secret files")
    return glob


def _append_rg_excludes(argv: list[str]) -> None:
    for glob in _RG_SECRET_EXCLUDES:
        argv.extend(("--iglob", glob))


READ_ONLY_TOOL_DEFINITIONS = tuple(
    definition for definition in TOOL_DEFINITIONS if definition.name != "apply_patch"
)
CANDIDATE_TOOL_DEFINITIONS = TOOL_DEFINITIONS


__all__ = [
    "ApplyPatchSpec",
    "CANDIDATE_TOOL_DEFINITIONS",
    "CommandSpec",
    "MAX_GLOB_LENGTH",
    "MAX_LINE_NUMBER",
    "MAX_PATH_LENGTH",
    "MAX_PATTERN_LENGTH",
    "MAX_READ_LINES",
    "READ_ONLY_TOOL_DEFINITIONS",
    "RepoMapSpec",
    "TOOL_DEFINITIONS",
    "ToolDefinition",
    "ToolSpec",
    "ToolValidationError",
    "build_tool_command",
]
