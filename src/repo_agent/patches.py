"""Strict validation for model-authored Git-style unified diffs."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import NamedTuple

from .git_environment import _sanitized_git_environment
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
_HUNK_HEADER_WITH_SUFFIX = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
_EDITOR_UPDATE = "*** Update File: "
_EDITOR_ADD = "*** Add File: "
_EDITOR_DELETE = "*** Delete File: "
_MAX_MODEL_FUZZ_GAP = 64


class _ModelHunk(NamedTuple):
    old_start: int
    old_lines: tuple[tuple[str, str], ...]
    new_lines: tuple[str, ...]


class _ModelChange(NamedTuple):
    path: str
    operation: str
    hunks: tuple[_ModelHunk, ...]


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


def normalize_model_patch(
    value: object, *, repository: Path | None = None
) -> ValidatedPatch:
    """Recount Git hunk headers for a model-authored patch.

    Models occasionally produce a structurally valid unified diff whose hunk
    line counts are stale after editing the body.  The public validator keeps
    rejecting that input; this narrow model boundary recomputes only the
    counts, then sends the canonical result through the same strict parser.
    When a model omits unchanged context or uses the common editor-style
    ``*** Begin Patch`` wrapper, a bounded in-memory reconstruction is used
    only when a disposable repository is supplied.  Paths, metadata,
    line-prefixes, byte/line/file limits, and Git context application remain
    enforced by the normal validation and sandbox layers.
    """

    original: PatchValidationError | None = None
    validated: ValidatedPatch | None = None
    try:
        validated = validate_patch(value)
    except PatchValidationError as exc:
        original = exc

    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n")
        if "\r" in normalized:
            raise original or PatchValidationError(
                "patch contains unsupported carriage returns"
            )
        # Models occasionally wrap an otherwise valid diff in one Markdown
        # fence. Remove only that outer wrapper; all normal validation and
        # repository applicability checks still run below.
        fenced_lines = normalized.splitlines()
        if (
            len(fenced_lines) >= 3
            and fenced_lines[0].strip().casefold() in {"```diff", "```patch", "```"}
            and fenced_lines[-1].strip() == "```"
        ):
            normalized = "\n".join(fenced_lines[1:-1]) + "\n"
        is_editor_patch = normalized.startswith("*** Begin Patch\n")
        is_unified_patch = normalized.startswith("--- ")
        if (
            "\x00" not in normalized
            and len(normalized.encode("utf-8", errors="ignore")) <= MAX_PATCH_BYTES
            and len(normalized.splitlines()) <= MAX_PATCH_LINES
            and (normalized.startswith("diff --git ") or is_editor_patch or is_unified_patch)
        ):
            # Editor-style and bare unified diffs have no strict public form to
            # validate first; canonicalize them directly against the disposable
            # repository. Git-style diffs get one bounded recount pass below.
            if is_editor_patch or is_unified_patch:
                if repository is not None:
                    try:
                        canonical = _canonicalize_model_patch(normalized, repository)
                        result = validate_patch(canonical)
                        if _git_patch_applies(repository, result.text):
                            return result
                    except (OSError, PatchValidationError, UnicodeError):
                        pass
                if original is not None:
                    raise original
                raise PatchValidationError("model patch could not be canonicalized")

            if validated is not None:
                if repository is None or _git_patch_applies(repository, validated.text):
                    return validated
                # A model can preserve valid context while carrying stale Git
                # index metadata from a previous candidate revision. Git's
                # recount mode ignores only the numeric hunk counts; the
                # strict parser above still enforces paths, prefixes, limits,
                # and metadata before this compatibility check.
                if repository is not None and _git_patch_applies_recount(
                    repository, validated.text
                ):
                    return validated
                # A structurally valid diff can still be unusable by Git when a
                # model omitted context or file-mode metadata. Reconstruct it
                # from the actual disposable checkout before rejecting it.
                try:
                    canonical = _canonicalize_model_patch(normalized, repository)
                    result = validate_patch(canonical)
                    if _git_patch_applies(repository, result.text):
                        return result
                except (OSError, PatchValidationError, UnicodeError):
                    pass
                raise original or PatchValidationError(
                    "model patch does not apply to the candidate workspace"
                )

            if original is None:
                original = PatchValidationError("invalid model patch")
            lines = normalized.splitlines()
            if len(lines) > MAX_PATCH_LINES:
                raise original

            rewritten = list(lines)
            header_index: int | None = None
            old_start = new_start = 0
            old_count = new_count = 0
            suffix = ""
            saw_hunk = False

            def flush_hunk() -> None:
                if header_index is None:
                    return
                rewritten[header_index] = (
                    f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"
                )

            for index, line in enumerate(lines):
                if line.startswith("diff --git "):
                    flush_hunk()
                    header_index = None
                    continue
                match = _HUNK_HEADER_WITH_SUFFIX.fullmatch(line)
                if match is not None:
                    flush_hunk()
                    old_start = int(match.group(1))
                    new_start = int(match.group(3))
                    suffix = match.group(5)
                    old_count = 0
                    new_count = 0
                    header_index = index
                    saw_hunk = True
                    continue
                if header_index is None:
                    continue
                if line == r"\ No newline at end of file":
                    continue
                if not line:
                    if repository is None:
                        raise original
                    # Preserve an empty context line when a model omitted its
                    # required leading space. Git applicability remains the
                    # final authority below.
                    rewritten[index] = " "
                    old_count += 1
                    new_count += 1
                    continue
                if line[0] not in {" ", "+", "-"}:
                    if repository is None:
                        raise original
                    rewritten[index] = " " + line
                    old_count += 1
                    new_count += 1
                    continue
                if line.startswith("+"):
                    new_count += 1
                elif line.startswith("-"):
                    old_count += 1
                else:
                    old_count += 1
                    new_count += 1
            flush_hunk()
            if not saw_hunk:
                # A repository-backed model patch may use a bare ``@@``
                # header. Let the canonicalizer recover its cursor from the
                # body instead of discarding the candidate at this stage.
                if repository is None:
                    raise original
            try:
                recounted = (
                    validate_patch("\n".join(rewritten) + "\n")
                    if saw_hunk
                    else None
                )
            except PatchValidationError:
                recounted = None
            if recounted is not None:
                if repository is None or _git_patch_applies(repository, recounted.text):
                    return recounted
                if repository is not None and _git_patch_applies_recount(
                    repository, recounted.text
                ):
                    return recounted
                try:
                    # Reuse the corrected hunk counts when reconstructing a
                    # patch that also omitted context or file metadata.
                    canonical = _canonicalize_model_patch(recounted.text, repository)
                    result = validate_patch(canonical)
                    if _git_patch_applies(repository, result.text) or _git_patch_applies_recount(
                        repository, result.text
                    ):
                        return result
                except (OSError, PatchValidationError, UnicodeError):
                    pass
            if repository is None:
                raise original
            try:
                canonical = _canonicalize_model_patch(normalized, repository)
                result = validate_patch(canonical)
                if _git_patch_applies(repository, result.text) or _git_patch_applies_recount(
                    repository, result.text
                ):
                    return result
            except (OSError, PatchValidationError, UnicodeError):
                pass
            raise original

    if original is not None:
        raise original
    raise PatchValidationError("invalid model patch")


def _git_patch_applies(repository: Path, patch: str) -> bool:
    """Check a candidate patch without mutating the disposable checkout."""

    try:
        completed = subprocess.run(
            [
                "git",
                "apply",
                "--check",
                "--index",
                "--whitespace=error-all",
                "--",
            ],
            cwd=repository,
            input=patch.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            env=_sanitized_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _git_patch_applies_recount(repository: Path, patch: str) -> bool:
    """Check a patch while asking Git to recalculate stale hunk counts.

    This is intentionally narrower than the model canonicalizer: callers must
    already have a :class:`ValidatedPatch`, so only Git's numeric hunk-count
    tolerance is added. Context, path, metadata, and whitespace checks remain
    enforced by the normal parser and Git itself.
    """

    try:
        completed = subprocess.run(
            [
                "git",
                "apply",
                "--check",
                "--recount",
                "--whitespace=error-all",
                "--",
            ],
            cwd=repository,
            input=patch.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            env=_sanitized_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _canonicalize_model_patch(value: str, repository: Path) -> str:
    lines = value.splitlines()
    if lines and lines[0] == "*** Begin Patch":
        changes = _parse_editor_patch(lines)
    elif lines and lines[0].startswith("diff --git "):
        changes = _parse_git_model_patch(lines)
    elif lines and lines[0].startswith("--- "):
        changes = _parse_bare_unified_patch(lines)
    else:
        raise PatchValidationError("unsupported model patch wrapper")
    if not changes or len(changes) > MAX_PATCH_FILES:
        raise PatchValidationError("model patch has too many file sections")

    rendered: list[str] = []
    seen: set[str] = set()
    for change in changes:
        if change.path in seen:
            raise PatchValidationError(f"patch repeats path: {change.path}")
        seen.add(change.path)
        before = _model_file_lines(repository, change.path, change.operation)
        if change.operation == "delete":
            after: list[str] = []
        elif change.operation == "add":
            after = _apply_model_hunks([], change.hunks, change.path)
            # A model-created file occasionally carries an extra empty hunk
            # line after its final logical line.  Keep the canonical file
            # newline, but drop only these terminal blank records so Git's
            # strict whitespace check does not reject an otherwise safe patch.
            while after and after[-1] == "":
                after.pop()
        else:
            after = _apply_model_hunks(before, change.hunks, change.path)
        if before == after and change.operation == "modify":
            raise PatchValidationError(f"model patch does not change {change.path}")
        rendered.extend(_render_model_change(change.path, change.operation, before, after))
    return "\n".join(rendered) + "\n"


def _parse_editor_patch(lines: list[str]) -> tuple[_ModelChange, ...]:
    if not lines or lines[-1] != "*** End Patch":
        raise PatchValidationError("model patch wrapper is incomplete")
    changes: list[_ModelChange] = []
    index = 1
    while index < len(lines) - 1:
        marker = lines[index]
        if marker.startswith(_EDITOR_UPDATE):
            operation = "modify"
            path = _safe_write_path(marker[len(_EDITOR_UPDATE) :])
        elif marker.startswith(_EDITOR_ADD):
            operation = "add"
            path = _safe_write_path(marker[len(_EDITOR_ADD) :])
        elif marker.startswith(_EDITOR_DELETE):
            operation = "delete"
            path = _safe_write_path(marker[len(_EDITOR_DELETE) :])
        else:
            raise PatchValidationError("unsupported editor-style patch section")
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not lines[index].startswith("*** "):
            body.append(lines[index])
            index += 1
        if operation == "delete":
            hunks: tuple[_ModelHunk, ...] = ()
        elif operation == "add":
            if any(not line.startswith("+") for line in body):
                raise PatchValidationError("editor add sections must contain added lines")
            hunks = (_ModelHunk(0, (), tuple(line[1:] for line in body)),)
        else:
            hunks = _parse_model_hunks(body, allow_bare=True)
        changes.append(_ModelChange(path, operation, hunks))
    return tuple(changes)


def _parse_git_model_patch(lines: list[str]) -> tuple[_ModelChange, ...]:
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts or starts[0] != 0:
        raise PatchValidationError("model patch must start with diff --git")
    starts.append(len(lines))
    changes: list[_ModelChange] = []
    for section_index in range(len(starts) - 1):
        section = lines[starts[section_index] : starts[section_index + 1]]
        match = _DIFF_HEADER.fullmatch(section[0])
        if match is None:
            raise PatchValidationError("model diff header is invalid")
        old_header, new_header = match.groups()
        old_path = _safe_write_path(old_header)
        new_path = _safe_write_path(new_header)
        if old_path != new_path:
            raise PatchValidationError("rename and copy patches are not supported")
        marker_index = next(
            (i for i, line in enumerate(section[1:], 1) if line.startswith("--- ")),
            None,
        )
        if marker_index is None or marker_index + 1 >= len(section):
            raise PatchValidationError("model diff lacks ---/+++ headers")
        old_marker = section[marker_index][4:]
        new_marker = section[marker_index + 1][4:]
        if old_marker == "/dev/null" and new_marker == f"b/{new_path}":
            operation = "add"
        elif old_marker == f"a/{old_path}" and new_marker == "/dev/null":
            operation = "delete"
        elif old_marker == f"a/{old_path}" and new_marker == f"b/{new_path}":
            operation = "modify"
        else:
            raise PatchValidationError("model diff paths are inconsistent")
        body = section[marker_index + 2 :]
        # Model-authored diffs occasionally omit the numeric range from a hunk
        # header (``@@``).  The repository-backed canonicalizer can recover the
        # cursor from the hunk body; the public validator remains strict.
        hunks = () if operation == "delete" else _parse_model_hunks(body, allow_bare=True)
        changes.append(_ModelChange(new_path, operation, hunks))
    return tuple(changes)


def _parse_bare_unified_patch(lines: list[str]) -> tuple[_ModelChange, ...]:
    """Parse the common ``---``/``+++`` diff emitted without Git headers."""

    sections: list[list[str]] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- ") or index + 1 >= len(lines):
            raise PatchValidationError("bare unified patch lacks ---/+++ headers")
        if not lines[index + 1].startswith("+++ "):
            raise PatchValidationError("bare unified patch lacks a +++ header")
        end = len(lines)
        # A new file section is recognized only at a header pair. This avoids
        # treating ordinary deleted hunk lines as file boundaries.
        probe = index + 2
        while probe + 1 < len(lines):
            if lines[probe].startswith("--- ") and lines[probe + 1].startswith("+++ "):
                end = probe
                break
            probe += 1
        sections.append(lines[index:end])
        index = end

    changes: list[_ModelChange] = []
    for section in sections:
        old_marker = section[0][4:]
        new_marker = section[1][4:]
        if old_marker == "/dev/null":
            if not new_marker.startswith("b/"):
                raise PatchValidationError("bare add patch has inconsistent paths")
            old_path = new_path = _safe_write_path(new_marker[2:])
            operation = "add"
        elif new_marker == "/dev/null":
            if not old_marker.startswith("a/"):
                raise PatchValidationError("bare delete patch has inconsistent paths")
            old_path = new_path = _safe_write_path(old_marker[2:])
            operation = "delete"
        elif old_marker.startswith("a/") and new_marker.startswith("b/"):
            old_path = _safe_write_path(old_marker[2:])
            new_path = _safe_write_path(new_marker[2:])
            if old_path != new_path:
                raise PatchValidationError("rename and copy patches are not supported")
            operation = "modify"
        else:
            raise PatchValidationError("bare unified patch paths are invalid")
        body = section[2:]
        hunks = () if operation == "delete" else _parse_model_hunks(body, allow_bare=True)
        changes.append(_ModelChange(new_path, operation, hunks))
    return tuple(changes)


def _parse_model_hunks(
    lines: list[str], *, allow_bare: bool
) -> tuple[_ModelHunk, ...]:
    hunks: list[_ModelHunk] = []
    current_start = 1
    current_body: list[tuple[str, str]] | None = None
    cursor = 1

    def flush() -> None:
        nonlocal current_body, cursor
        if current_body is None:
            return
        old_lines = tuple(item for item in current_body if item[0] != "+")
        new_lines = tuple(item[1] for item in current_body if item[0] != "-")
        hunks.append(_ModelHunk(current_start, old_lines, new_lines))
        cursor_value = current_start + len(old_lines)
        cursor = max(cursor, cursor_value)
        current_body = None

    for line in lines:
        if line.startswith("@@"):
            flush()
            match = _HUNK_HEADER_WITH_SUFFIX.fullmatch(line)
            if match is not None:
                current_start = int(match.group(1))
            elif allow_bare and line.strip() == "@@":
                current_start = cursor
            else:
                raise PatchValidationError("model hunk header is invalid")
            current_body = []
            continue
        if current_body is None:
            if line.strip():
                raise PatchValidationError("model patch has unsupported hunk metadata")
            continue
        if line == r"\ No newline at end of file":
            continue
        if not line:
            if allow_bare:
                # JSON/tool serialization can lose the single leading space
                # that represents an empty context line. Treat it as context
                # only for repository-backed model canonicalization; the
                # public validator still rejects the malformed representation.
                current_body.append((" ", ""))
                continue
            raise PatchValidationError("model hunk content is malformed")
        if line[0] not in {" ", "+", "-"}:
            if allow_bare:
                # Likewise recover omitted context prefixes, then require the
                # reconstructed patch to apply exactly to the disposable tree.
                current_body.append((" ", line))
                continue
            raise PatchValidationError("model hunk content is malformed")
        current_body.append((line[0], line[1:]))
    flush()
    if not hunks or any(not hunk.old_lines and not hunk.new_lines for hunk in hunks):
        raise PatchValidationError("model patch must contain non-empty hunks")
    return tuple(hunks)


def _model_file_lines(repository: Path, path: str, operation: str) -> list[str]:
    target = _safe_model_disk_path(repository, path, operation == "add")
    if operation == "add":
        if target.exists():
            raise PatchValidationError(f"add target already exists: {path}")
        return []
    if not target.exists():
        raise PatchValidationError(f"patch target does not exist: {path}")
    if not stat.S_ISREG(os.lstat(target).st_mode):
        raise PatchValidationError(f"patch target is not a regular file: {path}")
    try:
        return target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PatchValidationError(f"patch target is not readable: {path}") from exc


def _safe_model_disk_path(repository: Path, path: str, allow_missing: bool) -> Path:
    try:
        return DEFAULT_PATH_POLICY.validate_disk_path(
            repository,
            path,
            access="write",
            allow_missing_leaf=allow_missing,
        )
    except RepositoryPathError as exc:
        raise PatchValidationError(str(exc)) from exc


def _apply_model_hunks(
    base: list[str], hunks: tuple[_ModelHunk, ...], path: str
) -> list[str]:
    result = list(base)
    offset = 0
    for hunk in hunks:
        old_lines = [text for marker, text in hunk.old_lines]
        new_lines = list(hunk.new_lines)
        preferred = max(0, hunk.old_start - 1 + offset)
        start, end = _locate_model_sequence(result, old_lines, preferred, path)
        result[start:end] = new_lines
        offset += len(new_lines) - (end - start)
    return result


def _locate_model_sequence(
    current: list[str], old_lines: list[str], preferred: int, path: str
) -> tuple[int, int]:
    if not old_lines:
        return min(max(preferred, 0), len(current)), min(max(preferred, 0), len(current))
    exact: list[tuple[int, int]] = []
    width = len(old_lines)
    first_start = max(0, preferred - _MAX_MODEL_FUZZ_GAP)
    last_start = min(
        len(current) - width,
        preferred + _MAX_MODEL_FUZZ_GAP,
    )
    for start in range(first_start, last_start + 1):
        if current[start : start + width] == old_lines:
            exact.append((start, start + width))
    if exact:
        exact.sort(key=lambda item: (abs(item[0] - preferred), item[0]))
        if (
            len(exact) > 1
            and abs(exact[0][0] - preferred) == abs(exact[1][0] - preferred)
        ):
            raise PatchValidationError(f"model patch context is ambiguous for {path}")
        return exact[0]
    # Never bridge an omitted range: replacing the span between two loosely
    # matched lines would silently delete unchanged source. A single changed
    # line is safe to relocate within the bounded fuzz window, which covers the
    # model's usual context-free ``-old/+new`` output.
    if len(old_lines) == 1:
        candidates = [
            (abs(index - preferred), index, index + 1)
            for index, value in enumerate(current)
            if value == old_lines[0]
            and abs(index - preferred) <= _MAX_MODEL_FUZZ_GAP
        ]
        if candidates:
            candidates.sort()
            best = candidates[0]
            if len(candidates) > 1 and candidates[1][0] == best[0]:
                raise PatchValidationError(f"model patch context is ambiguous for {path}")
            return best[1], best[2]
    raise PatchValidationError(f"model patch context does not match {path}")


def _render_model_change(
    path: str, operation: str, before: list[str], after: list[str]
) -> list[str]:
    diff = list(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{path}" if operation != "add" else "/dev/null",
            tofile=f"b/{path}" if operation != "delete" else "/dev/null",
            n=3,
            lineterm="",
        )
    )
    if len(diff) < 3:
        raise PatchValidationError(f"model patch does not change {path}")
    header = [f"diff --git a/{path} b/{path}"]
    if operation == "add":
        header.append("new file mode 100644")
    elif operation == "delete":
        header.append("deleted file mode 100644")
    header.extend(diff)
    return header


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
    "normalize_model_patch",
    "validate_patch",
]
