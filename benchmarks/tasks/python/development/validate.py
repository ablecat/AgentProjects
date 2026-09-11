"""Validate the independent Python prompt-development task suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
IGNORED_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
ALLOWED_SUFFIXES = frozenset({".json", ".md", ".patch", ".py"})
MAX_FILE_BYTES = 256 * 1024
MAX_TASK_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
EXPECTED_TASKS = (
    ("py-development-001", "configurable-path-exclusions"),
    ("py-development-002", "json-report-exporter"),
    ("py-development-003", "tool-timeout-budget"),
    ("py-development-004", "severity-summary"),
)
EXPECTED_ASSETS = {
    "baseline": "baseline",
    "issue": "issue.md",
    "hidden_tests": "hidden_tests",
    "gold_patch": "gold.patch",
}
EXPECTED_CONTRACT = {
    "runner": "unittest",
    "public_pattern": "test_*.py",
    "hidden_pattern": "test_*.py",
    "baseline_public": "pass",
    "baseline_hidden": "fail",
    "gold_public": "pass",
    "gold_hidden": "pass",
}


class ValidationError(RuntimeError):
    """A development fixture failed its structural or lifecycle contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="check structure and content locks without executing task code",
    )
    parser.add_argument(
        "--print-digests",
        action="store_true",
        help="print calculated task digests without enforcing stored locks",
    )
    args = parser.parse_args(argv)

    try:
        manifest = _load_json(MANIFEST_PATH, "development manifest")
        tasks = _validate_manifest(manifest)
        calculated: dict[str, str] = {}
        metadata_by_id: dict[str, dict[str, Any]] = {}
        for expected_id, expected_slug in EXPECTED_TASKS:
            entry = tasks[expected_id]
            task_dir = _task_directory(entry["path"], expected_slug)
            metadata_by_id[expected_id] = _validate_task(
                task_dir, expected_id, expected_slug
            )
            calculated[expected_id] = _task_digest(task_dir)

        if args.print_digests:
            print(json.dumps(calculated, indent=2, sort_keys=True))
            return 0

        for task_id, digest in calculated.items():
            if tasks[task_id]["sha256"] != digest:
                raise ValidationError(
                    f"{task_id}: content lock mismatch; calculated {digest}"
                )

        if not args.structure_only:
            for task_id, _slug in EXPECTED_TASKS:
                task_dir = ROOT / tasks[task_id]["path"]
                _validate_lifecycle(task_dir, metadata_by_id[task_id])
    except (OSError, ValidationError, subprocess.SubprocessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.structure_only:
        print("[ok] development task structure and content locks")
    else:
        print("[ok] development task structure, locks, and lifecycle")
    return 0


def _validate_manifest(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "suite_id",
        "validator",
        "formal_evaluation",
        "tasks",
    }:
        raise ValidationError("development manifest fields are invalid")
    if (
        value["schema_version"] != 1
        or value["suite_id"] != "repo-agent-python-development-v1"
        or value["validator"] != "validate.py"
        or value["formal_evaluation"] is not False
    ):
        raise ValidationError("development manifest identity is invalid")
    entries = value["tasks"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_TASKS):
        raise ValidationError("development manifest must contain exactly four tasks")

    result: dict[str, dict[str, str]] = {}
    for index, (expected_id, expected_slug) in enumerate(EXPECTED_TASKS):
        entry = entries[index]
        if not isinstance(entry, dict) or set(entry) != {"id", "path", "sha256"}:
            raise ValidationError(f"development manifest task {index} is invalid")
        if entry["id"] != expected_id or entry["path"] != expected_slug:
            raise ValidationError("development manifest task order or path changed")
        digest = entry["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValidationError(f"{expected_id}: invalid SHA-256 content lock")
        result[expected_id] = dict(entry)
    return result


def _task_directory(relative: Any, expected_slug: str) -> Path:
    if not isinstance(relative, str) or relative != expected_slug:
        raise ValidationError(f"{expected_slug}: task path is invalid")
    task_dir = (ROOT / relative).resolve(strict=True)
    try:
        task_dir.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValidationError(f"{expected_slug}: task path escapes suite root") from exc
    if task_dir.parent != ROOT.resolve(strict=True) or _is_link(task_dir):
        raise ValidationError(f"{expected_slug}: task directory is not a real child")
    return task_dir


def _validate_task(
    task_dir: Path, expected_id: str, expected_slug: str
) -> dict[str, Any]:
    expected_roots = {
        "baseline",
        "gold.patch",
        "hidden_tests",
        "issue.md",
        "metadata.json",
    }
    actual_roots = {child.name for child in task_dir.iterdir()}
    if actual_roots != expected_roots:
        raise ValidationError(
            f"{expected_id}: expected only {sorted(expected_roots)}, got {sorted(actual_roots)}"
        )
    for path in task_dir.rglob("*"):
        if path.name in IGNORED_NAMES:
            raise ValidationError(
                f"{expected_id}: generated path is committed: {path.name}"
            )
        if _is_link(path):
            raise ValidationError(
                f"{expected_id}: links and reparse points are forbidden"
            )
        if path.is_file():
            if path.suffix not in ALLOWED_SUFFIXES:
                raise ValidationError(
                    f"{expected_id}: unsupported file type: {path.name}"
                )
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValidationError(
                    f"{expected_id}: file exceeds size limit: {path.name}"
                )

    metadata = _load_json(task_dir / "metadata.json", f"{expected_id} metadata")
    required_metadata = {
        "schema_version",
        "id",
        "slug",
        "title",
        "language",
        "kind",
        "category",
        "difficulty",
        "timeout_seconds",
        "assets",
        "test_contract",
        "constraints",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_metadata:
        raise ValidationError(f"{expected_id}: metadata fields are invalid")
    if (
        metadata["schema_version"] != 1
        or metadata["id"] != expected_id
        or metadata["slug"] != expected_slug
        or metadata["language"] != "python"
        or metadata["kind"] != "development"
        or metadata["difficulty"] not in {"easy", "medium"}
        or not isinstance(metadata["title"], str)
        or not metadata["title"].strip()
        or not isinstance(metadata["category"], str)
        or not metadata["category"].strip()
        or type(metadata["timeout_seconds"]) is not int
        or not 1 <= metadata["timeout_seconds"] <= 30
    ):
        raise ValidationError(f"{expected_id}: metadata identity is invalid")
    if metadata["assets"] != EXPECTED_ASSETS:
        raise ValidationError(f"{expected_id}: asset contract is invalid")

    contract = metadata["test_contract"]
    if not isinstance(contract, dict) or set(contract) != {
        *EXPECTED_CONTRACT,
        "failure_signatures",
    }:
        raise ValidationError(f"{expected_id}: test contract fields are invalid")
    if any(contract[key] != expected for key, expected in EXPECTED_CONTRACT.items()):
        raise ValidationError(f"{expected_id}: test lifecycle contract changed")
    signatures = contract["failure_signatures"]
    if (
        not isinstance(signatures, list)
        or not signatures
        or len(signatures) != len(set(signatures))
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"test_[A-Za-z0-9_]+", item) is None
            for item in signatures
        )
    ):
        raise ValidationError(f"{expected_id}: failure signatures are invalid")
    if metadata["constraints"] != {
        "network": False,
        "third_party_dependencies": [],
        "python_min": "3.11",
    }:
        raise ValidationError(f"{expected_id}: execution constraints are invalid")

    if not list((task_dir / "baseline" / "tests").glob("test_*.py")):
        raise ValidationError(f"{expected_id}: baseline has no public tests")
    if not list((task_dir / "hidden_tests").glob("test_*.py")):
        raise ValidationError(f"{expected_id}: task has no hidden tests")
    issue = (task_dir / "issue.md").read_text(encoding="utf-8")
    if not issue.strip() or len(issue) > 16_000:
        raise ValidationError(f"{expected_id}: issue prompt is empty or too large")
    _validate_gold_patch(task_dir / "gold.patch", expected_id)
    return metadata


def _validate_gold_patch(path: Path, task_id: str) -> None:
    text = path.read_text(encoding="utf-8")
    changed = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.MULTILINE)
    if not changed or any(left != right for left, right in changed):
        raise ValidationError(f"{task_id}: gold patch headers are invalid")
    names = [left for left, _right in changed]
    if len(names) != len(set(names)) or any(
        not name.startswith("tasklib/")
        or name.startswith("tasklib/../")
        or "\\" in name
        or "/../" in name
        for name in names
    ):
        raise ValidationError(f"{task_id}: gold patch changes an invalid path")
    forbidden = ("GIT binary patch", "old mode ", "new mode ", "rename from ")
    if any(marker in text for marker in forbidden):
        raise ValidationError(f"{task_id}: gold patch contains forbidden metadata")


def _validate_lifecycle(task_dir: Path, metadata: dict[str, Any]) -> None:
    task_id = metadata["id"]
    timeout = float(metadata["timeout_seconds"])
    with tempfile.TemporaryDirectory(prefix="repo-agent-development-") as temporary:
        worktree = Path(temporary) / "worktree"
        shutil.copytree(task_dir / "baseline", worktree)

        public_before = _run_public(worktree, timeout)
        if public_before.returncode != 0:
            raise ValidationError(
                f"{task_id}: baseline public tests failed: {_detail(public_before)}"
            )
        hidden_before = _run_hidden(worktree, task_dir / "hidden_tests", timeout)
        if hidden_before.returncode == 0:
            raise ValidationError(
                f"{task_id}: baseline hidden tests unexpectedly passed"
            )
        combined = _detail(hidden_before)
        for signature in metadata["test_contract"]["failure_signatures"]:
            if signature not in combined:
                raise ValidationError(
                    f"{task_id}: baseline hidden failure omitted {signature}"
                )

        _apply_patch(worktree, task_dir / "gold.patch", timeout)
        public_after = _run_public(worktree, timeout)
        if public_after.returncode != 0:
            raise ValidationError(
                f"{task_id}: gold public tests failed: {_detail(public_after)}"
            )
        hidden_after = _run_hidden(worktree, task_dir / "hidden_tests", timeout)
        if hidden_after.returncode != 0:
            raise ValidationError(
                f"{task_id}: gold hidden tests failed: {_detail(hidden_after)}"
            )


def _run_public(worktree: Path, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return _run(
        (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ),
        cwd=worktree,
        timeout=timeout,
        python_path=worktree,
    )


def _run_hidden(
    worktree: Path, hidden_root: Path, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    outputs: list[bytes] = []
    returncode = 0
    for test_file in sorted(hidden_root.glob("test_*.py")):
        result = _run(
            (sys.executable, str(test_file)),
            cwd=worktree,
            timeout=timeout,
            python_path=worktree,
        )
        outputs.extend((result.stdout, result.stderr))
        if result.returncode != 0:
            returncode = result.returncode
    combined = b"".join(outputs)[:MAX_OUTPUT_BYTES]
    return subprocess.CompletedProcess(
        args=(sys.executable, "<hidden-tests>"),
        returncode=returncode,
        stdout=combined,
        stderr=b"",
    )


def _apply_patch(worktree: Path, patch: Path, timeout: float) -> None:
    for mode in (("--check",), ()):
        result = _run(
            ("git", "apply", *mode, "--whitespace=error-all", str(patch)),
            cwd=worktree,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise ValidationError(f"gold patch did not apply: {_detail(result)}")


def _run(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout: float,
    python_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if python_path is not None:
        environment["PYTHONPATH"] = str(python_path)
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"command timed out after {timeout:g}s") from exc


def _task_digest(task_dir: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    files = sorted(
        (path for path in task_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(task_dir).as_posix(),
    )
    for path in files:
        relative = path.relative_to(task_dir).as_posix()
        raw = path.read_bytes()
        total += len(raw)
        if total > MAX_TASK_BYTES:
            raise ValidationError(f"{task_dir.name}: task exceeds size limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"{task_dir.name}: non-UTF-8 file: {relative}"
            ) from exc
        canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite number: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValidationError(f"{label} is invalid JSON") from exc


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _detail(result: subprocess.CompletedProcess[bytes]) -> str:
    raw = (result.stderr or result.stdout)[:MAX_OUTPUT_BYTES]
    text = raw.decode("utf-8", errors="replace").strip()
    return text or f"exit code {result.returncode}"


if __name__ == "__main__":
    raise SystemExit(main())
