"""Strict validation for model-authored Git-style unified diffs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .policy import DEFAULT_PATH_POLICY, RepositoryPathError


MAX_PATCH_BYTES = 64 * 1024
MAX_PATCH_FILES = 12
MAX_CHANGED_LINES = 1200
MAX_PATCH_LINES = 4000

_DIFF_HEADER = re.compile(r"^diff --git a/([^\s]+) b/([^\s]+)$")
_INDEX_LINE = re.compile(
    r"^index [0-9a-fA-F]{4,64}\.\.[0-9a-fA-F]{4,64}(?: 100(?:644|755))?$"
)
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


class PatchValidationError(ValueError):
    """Raised before an unsafe or unsupported patch reaches Git."""


@dataclass(frozen=True, slots=True)
class PatchFile:
    path: str
    operation: str
    added_lines: int
    removed_lines: int


@dataclass(frozen=True, slots=True)
class ValidatedPatch:
    text: str
    files: tuple[PatchFile, ...]

    @property
    def changed_lines(self) -> int:
        return sum(item.added_lines + item.removed_lines for item in self.files)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def added_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files if item.operation == "add")


def validate_patch(value: object) -> ValidatedPatch:
    """Parse a conservative subset of Git unified diff without executing it."""

    if not isinstance(value, str):
        raise PatchValidationError("patch must be a string")
    if not value.strip():
        raise PatchValidationError("patch must not be empty")
    if "\x00" in value:
        raise PatchValidationError("patch must not contain NUL")
    normalized = value.replace("\r\n", "\n")
    if "\r" in normalized:
        raise PatchValidationError("patch contains unsupported carriage returns")
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PatchValidationError("patch must contain valid Unicode text") from exc
    if len(encoded) > MAX_PATCH_BYTES:
        raise PatchValidationError(f"patch is limited to {MAX_PATCH_BYTES} bytes")

    lines = normalized.splitlines()
    if len(lines) > MAX_PATCH_LINES:
        raise PatchValidationError(f"patch is limited to {MAX_PATCH_LINES} lines")
    if not lines or not lines[0].startswith("diff --git "):
        raise PatchValidationError("patch must be a Git-style unified diff")

    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if len(starts) > MAX_PATCH_FILES:
        raise PatchValidationError(f"patch is limited to {MAX_PATCH_FILES} files")
    starts.append(len(lines))

    files: list[PatchFile] = []
    seen_paths: set[str] = set()
    for section_index in range(len(starts) - 1):
        section = lines[starts[section_index] : starts[section_index + 1]]
        parsed = _parse_file_section(section)
        if parsed.path in seen_paths:
            raise PatchValidationError(f"patch repeats path: {parsed.path}")
        seen_paths.add(parsed.path)
        files.append(parsed)

    if not files:
        raise PatchValidationError("patch must change at least one file")
    changed_lines = sum(item.added_lines + item.removed_lines for item in files)
    if changed_lines > MAX_CHANGED_LINES:
        raise PatchValidationError(
            f"patch is limited to {MAX_CHANGED_LINES} changed lines"
        )
    if changed_lines == 0:
        raise PatchValidationError("patch must contain changed hunk lines")

    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return ValidatedPatch(normalized, tuple(files))


def _parse_file_section(lines: list[str]) -> PatchFile:
    header = _DIFF_HEADER.fullmatch(lines[0])
    if header is None:
        raise PatchValidationError(
            "diff paths must be unquoted, whitespace-free repository paths"
        )
    old_header_path, new_header_path = header.groups()
    old_path = _safe_write_path(old_header_path)
    new_path = _safe_write_path(new_header_path)
    if old_path != new_path:
        raise PatchValidationError("rename and copy patches are not supported")

    old_marker_index = next(
        (index for index, line in enumerate(lines[1:], 1) if line.startswith("--- ")),
        None,
    )
    if old_marker_index is None or old_marker_index + 1 >= len(lines):
        raise PatchValidationError(f"patch for {old_path} lacks ---/+++ headers")
    if not lines[old_marker_index + 1].startswith("+++ "):
        raise PatchValidationError(f"patch for {old_path} lacks a +++ header")

    _validate_metadata(lines[1:old_marker_index])
    old_marker = lines[old_marker_index][4:]
    new_marker = lines[old_marker_index + 1][4:]
    if old_marker == "/dev/null" and new_marker == f"b/{new_path}":
        operation = "add"
    elif old_marker == f"a/{old_path}" and new_marker == "/dev/null":
        operation = "delete"
    elif old_marker == f"a/{old_path}" and new_marker == f"b/{new_path}":
        operation = "modify"
    else:
        raise PatchValidationError(f"inconsistent patch paths for {old_path}")

    body = lines[old_marker_index + 2 :]
    if not body or not any(_HUNK_HEADER.fullmatch(line) for line in body):
        raise PatchValidationError(f"patch for {old_path} has no valid hunk")
    added = 0
    removed = 0
    expected_old: int | None = None
    expected_new: int | None = None
    actual_old = 0
    actual_new = 0

    def finish_hunk() -> None:
        if expected_old is None or expected_new is None:
            return
        if actual_old != expected_old or actual_new != expected_new:
            raise PatchValidationError(
                f"hunk line counts do not match its header for {old_path}"
            )

    for line in body:
        hunk = _HUNK_HEADER.fullmatch(line)
        if hunk:
            finish_hunk()
            _old_start, old_count, _new_start, new_count = hunk.groups()
            expected_old = 1 if old_count is None else int(old_count)
            expected_new = 1 if new_count is None else int(new_count)
            actual_old = 0
            actual_new = 0
            continue
        if expected_old is None:
            raise PatchValidationError(f"unsupported patch metadata for {old_path}")
        if line == r"\ No newline at end of file":
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            raise PatchValidationError(f"malformed hunk content for {old_path}")
        if line.startswith("+"):
            added += 1
            actual_new += 1
        elif line.startswith("-"):
            removed += 1
            actual_old += 1
        else:
            actual_old += 1
            actual_new += 1
    finish_hunk()
    return PatchFile(old_path, operation, added, removed)


def _validate_metadata(lines: list[str]) -> None:
    for line in lines:
        if _INDEX_LINE.fullmatch(line):
            continue
        if line in {"new file mode 100644", "deleted file mode 100644"}:
            continue
        forbidden_prefixes = (
            "Binary files ",
            "GIT binary patch",
            "Submodule ",
            "copy from ",
            "copy to ",
            "deleted file mode ",
            "new file mode ",
            "new mode ",
            "old mode ",
            "rename from ",
            "rename to ",
            "similarity index ",
        )
        if line.startswith(forbidden_prefixes):
            raise PatchValidationError("binary, link, mode, rename, and copy patches are not supported")
        raise PatchValidationError(f"unsupported patch metadata: {line!r}")


def _safe_write_path(path: str) -> str:
    try:
        return DEFAULT_PATH_POLICY.validate(path, access="write")
    except RepositoryPathError as exc:
        raise PatchValidationError(str(exc)) from exc


__all__ = [
    "MAX_CHANGED_LINES",
    "MAX_PATCH_BYTES",
    "MAX_PATCH_FILES",
    "MAX_PATCH_LINES",
    "PatchFile",
    "PatchValidationError",
    "ValidatedPatch",
    "validate_patch",
]
