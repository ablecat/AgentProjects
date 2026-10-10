from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
SUITE_ID = "repo-agent-python-oss-v1"
EXPECTED_TASKS = (
    {
        "id": "py-oss-001",
        "slug": "pysnooper-output-path",
        "repository": "https://github.com/cool-RR/PySnooper",
        "issue_url": "https://github.com/cool-RR/PySnooper/issues/2",
        "base_commit": "6e3d797be3fa0a746fb5b1b7c7fea78eb926c208",
        "fix_commit": "15555ed760000b049aff8fecc79d29339c1224c3",
        "target": "pysnooper/pysnooper.py",
        "base_sha256": "dfcdc4d446cd61a10721dc7691821c24cdf029af4b4ee7aff59a7fc07fd94d77",
        "fix_sha256": "188d29b6ae6c9971e80d1935b65a376ab351daa286c597f1b6e6bccb6e8b0c4b",
        "license": "LICENSE",
        "license_sha256": "0c4250d2a9f0e39d024593ede44eb5044e5fbd1175a8962ec9659d8ace34de9e",
    },
    {
        "id": "py-oss-002",
        "slug": "httpie-case-insensitive-host",
        "repository": "https://github.com/httpie/cli",
        "issue_url": "https://github.com/httpie/cli/issues/235",
        "base_commit": "8c892edd4fe700a7ca5cc733dcb4817831d253e2",
        "fix_commit": "040d981f00c3f6830b2d0db3daf3c64c080e96e3",
        "target": "httpie/models.py",
        "base_sha256": "ea94fc1b01a56fe1630ef24af396d97ad536b9b98c7f301acea23eeb8111ab39",
        "fix_sha256": "2b0b74fc797a8716f5020024b4f6dc88c343e99b70db93daa298201f323cd153",
        "license": "LICENSE",
        "license_sha256": "4396a6f1a58b2d961e28228e005988d7059614a06de03e944f87a8a3087731a4",
    },
    {
        "id": "py-oss-003",
        "slug": "thefuck-ls-prefix",
        "repository": "https://github.com/nvbn/thefuck",
        "issue_url": "https://github.com/nvbn/thefuck/issues/271",
        "base_commit": "cb33c912e5f2f4c2da6b70d708ff0437bfcd3b94",
        "fix_commit": "25cc98a21a3450a046caf418f08713c82a290805",
        "target": "thefuck/rules/ls_lah.py",
        "base_sha256": "d34fcf45876859ad0511438aa687968e533e32cc44414ad3560fc28d318fc147",
        "fix_sha256": "c89281b4660fbd64abd9007dd649fdf88f63849314a796af58e6c3210dca2ca4",
        "license": "LICENSE.md",
        "license_sha256": "84da771f8c614fe890db097f0090404691f6c750776972ee63e68e10c7f43196",
    },
)
GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
GENERATED_SUFFIXES = frozenset({".pyc", ".pyd", ".pyo"})
TEXT_SUFFIXES = frozenset(
    {".ini", ".json", ".md", ".patch", ".py", ".txt", ".toml", ".yml"}
)
MAX_TARGET_LINES = 200
MAX_TARGET_BYTES = 16 * 1024
MAX_REPOSITORY_FILES = 500


class ValidationError(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        information = os.lstat(path)
    except OSError:
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(information, "st_file_attributes", 0)
    return stat.S_ISLNK(information.st_mode) or bool(reparse and attributes & reparse)


def _task_digest(task_dir: Path) -> str:
    if _is_link_or_reparse(task_dir) or not task_dir.is_dir():
        raise ValidationError(f"links are not allowed in frozen tasks: {task_dir}")
    digest = hashlib.sha256()
    files: list[Path] = []

    def walk_error(error: OSError) -> None:
        raise ValidationError(f"could not inspect frozen task: {error}") from error

    for current, directories, filenames in os.walk(
        task_dir,
        topdown=True,
        followlinks=False,
        onerror=walk_error,
    ):
        current_dir = Path(current)
        directories.sort()
        filenames.sort()
        for name in (*directories, *filenames):
            path = current_dir / name
            if _is_link_or_reparse(path):
                raise ValidationError(f"links are not allowed in frozen tasks: {path}")
            relative = path.relative_to(task_dir)
            if (
                any(part in GENERATED_DIRECTORIES for part in relative.parts)
                or path.suffix.lower() in GENERATED_SUFFIXES
            ):
                raise ValidationError(
                    f"generated artifacts are not allowed in frozen tasks: {path}"
                )
        files.extend(current_dir / name for name in filenames)

    files.sort(key=lambda path: path.relative_to(task_dir).as_posix())
    for path in files:
        relative = path.relative_to(task_dir).as_posix().encode("utf-8")
        content = path.read_bytes()
        if path.suffix.lower() in TEXT_SUFFIXES:
            content = content.replace(b"\r\n", b"\n")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_text(path: Path) -> None:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValidationError(f"text file must use LF line endings: {path}")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValidationError(f"text file must be UTF-8: {path}") from exc


def _validate_patch(path: Path, target: str) -> None:
    _validate_text(path)
    content = path.read_text(encoding="utf-8")
    expected = f"diff --git a/{target} b/{target}"
    if content.count("diff --git ") != 1 or expected not in content:
        raise ValidationError(f"{path}: patch must change only {target}")
    if f"--- a/{target}" not in content or f"+++ b/{target}" not in content:
        raise ValidationError(f"{path}: patch headers do not match {target}")


def _validate_metadata(task_dir: Path, expected: dict[str, str]) -> dict[str, Any]:
    metadata = _load_json(task_dir / "metadata.json")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("id") != expected["id"]
        or metadata.get("slug") != expected["slug"]
        or metadata.get("language") != "python"
        or metadata.get("kind") != "bugfix"
    ):
        raise ValidationError(f"{expected['id']}: metadata identity is invalid")
    if metadata.get("assets") != {
        "baseline": "baseline",
        "setup_patch": "setup.patch",
        "issue": "issue.md",
        "hidden_tests": "hidden_tests",
        "gold_patch": "gold.patch",
    }:
        raise ValidationError(f"{expected['id']}: asset contract is invalid")
    contract = metadata.get("test_contract")
    if not isinstance(contract, dict) or contract.get("runner") != "unittest":
        raise ValidationError(f"{expected['id']}: test contract is invalid")
    signatures = contract.get("failure_signatures")
    if (
        not isinstance(signatures, list)
        or not signatures
        or not all(isinstance(item, str) and item.startswith("test_") for item in signatures)
    ):
        raise ValidationError(f"{expected['id']}: failure signatures are invalid")
    if metadata.get("constraints") != {
        "network": False,
        "third_party_dependencies": [],
        "python_min": "3.11",
    }:
        raise ValidationError(f"{expected['id']}: constraints are invalid")
    return metadata


def _validate_provenance(task_dir: Path, expected: dict[str, str]) -> None:
    provenance = _load_json(task_dir / "provenance.json")
    target = provenance.get("target")
    license_info = provenance.get("license")
    commits = provenance.get("commits")
    issue = provenance.get("issue")
    source_files = provenance.get("source_files")
    repository = expected["repository"]
    base_commit = expected["base_commit"]
    fix_commit = expected["fix_commit"]
    raw_root = repository.replace(
        "https://github.com/", "https://raw.githubusercontent.com/", 1
    )
    if (
        provenance.get("schema_version") != 1
        or provenance.get("repository") != repository
        or provenance.get("fix_url") != f"{repository}/commit/{fix_commit}"
    ):
        raise ValidationError(f"{expected['id']}: provenance identity is invalid")
    if not isinstance(issue, dict) or issue.get("url") != expected["issue_url"]:
        raise ValidationError(f"{expected['id']}: issue provenance is invalid")
    if commits != {"base": base_commit, "fix": fix_commit}:
        raise ValidationError(f"{expected['id']}: commit provenance is invalid")
    expected_sources = {
        "base": {
            "url": f"{raw_root}/{base_commit}/{expected['target']}",
            "sha256": expected["base_sha256"],
        },
        "fix": {
            "url": f"{raw_root}/{fix_commit}/{expected['target']}",
            "sha256": expected["fix_sha256"],
        },
        "license": {
            "url": f"{raw_root}/{fix_commit}/{expected['license']}",
            "sha256": expected["license_sha256"],
        },
    }
    if source_files != expected_sources:
        raise ValidationError(f"{expected['id']}: source-file provenance is invalid")
    if target != {
        "path": expected["target"],
        "base_sha256": expected["base_sha256"],
        "fix_sha256": expected["fix_sha256"],
    }:
        raise ValidationError(f"{expected['id']}: target provenance is invalid")
    if license_info != {
        "path": expected["license"],
        "sha256": expected["license_sha256"],
    }:
        raise ValidationError(f"{expected['id']}: license provenance is invalid")
    curation = provenance.get("curation")
    if not isinstance(curation, dict) or curation.get("complete_upstream_repository"):
        raise ValidationError(f"{expected['id']}: curation scope must be explicit")


def validate_structure() -> tuple[tuple[dict[str, str], Path, dict[str, Any]], ...]:
    manifest = _load_json(MANIFEST_PATH)
    if set(manifest) != {
        "schema_version",
        "suite_id",
        "formal_evaluation",
        "languages",
        "tasks",
    }:
        raise ValidationError("manifest keys do not match the OSS suite contract")
    if (
        manifest["schema_version"] != 1
        or manifest["suite_id"] != SUITE_ID
        or manifest["formal_evaluation"] is not False
        or manifest["languages"] != ["python"]
    ):
        raise ValidationError("manifest identity does not match the OSS suite")
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or len(tasks) != len(EXPECTED_TASKS):
        raise ValidationError("manifest must contain exactly three OSS tasks")

    validated = []
    for entry, expected in zip(tasks, EXPECTED_TASKS, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"id", "path", "sha256"}:
            raise ValidationError("manifest task entry is malformed")
        if entry["id"] != expected["id"] or entry["path"] != expected["slug"]:
            raise ValidationError("manifest task order or identity is incorrect")
        relative = PurePosixPath(entry["path"])
        if relative.parts != (entry["path"],):
            raise ValidationError(f"unsafe task path: {entry['path']}")
        task_path = ROOT / entry["path"]
        if _is_link_or_reparse(task_path):
            raise ValidationError(f"unsafe task directory: {entry['path']}")
        task_dir = task_path.resolve(strict=True)
        if task_dir.parent != ROOT or not task_dir.is_dir():
            raise ValidationError(f"unsafe task directory: {entry['path']}")
        if _task_digest(task_dir) != entry["sha256"]:
            raise ValidationError(f"{entry['id']}: content digest does not match lock")
        metadata = _validate_metadata(task_dir, expected)
        _validate_provenance(task_dir, expected)
        baseline = task_dir / "baseline"
        files = tuple(path for path in baseline.rglob("*") if path.is_file())
        if not files or len(files) > MAX_REPOSITORY_FILES:
            raise ValidationError(f"{entry['id']}: repository map exceeds limit")
        for path in (*files, *(p for p in task_dir.iterdir() if p.is_file())):
            if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("LICENSE"):
                _validate_text(path)
        target = baseline / expected["target"]
        raw = target.read_bytes()
        if (
            len(raw) > MAX_TARGET_BYTES
            or len(raw.decode("utf-8").splitlines()) > MAX_TARGET_LINES
            or _sha256(target) != expected["fix_sha256"]
        ):
            raise ValidationError(f"{entry['id']}: target source limit or hash mismatch")
        license_path = baseline / expected["license"]
        if _sha256(license_path) != expected["license_sha256"]:
            raise ValidationError(f"{entry['id']}: license hash mismatch")
        _validate_patch(task_dir / "setup.patch", expected["target"])
        _validate_patch(task_dir / "gold.patch", expected["target"])
        if not list((baseline / "tests").glob("test_*.py")):
            raise ValidationError(f"{entry['id']}: public tests are missing")
        if not list((task_dir / "hidden_tests").glob("test_*.py")):
            raise ValidationError(f"{entry['id']}: hidden tests are missing")
        validated.append((expected, task_dir, metadata))
    return tuple(validated)


def _run_tests(
    worktree: Path, tests_dir: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(worktree)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-v",
            "-s",
            str(tests_dir),
            "-p",
            "test_*.py",
        ),
        cwd=worktree,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def _require_result(
    result: subprocess.CompletedProcess[str],
    *,
    should_pass: bool,
    task_id: str,
    stage: str,
    signatures: Sequence[str] = (),
) -> None:
    if (result.returncode == 0) != should_pass:
        expectation = "pass" if should_pass else "fail"
        raise ValidationError(
            f"{task_id}: {stage} should {expectation}\n{result.stdout[-4000:]}"
        )
    for signature in signatures:
        attributed_failure = re.search(
            rf"(?m)^(?:FAIL|ERROR):\s+{re.escape(signature)}(?:\s|\(|$)",
            result.stdout,
        )
        if attributed_failure is None:
            raise ValidationError(
                f"{task_id}: {stage} did not attribute a failure to {signature}\n"
                f"{result.stdout[-4000:]}"
            )


def _apply_patch(worktree: Path, patch: Path, task_id: str) -> None:
    for check_only in (True, False):
        command = ["git", "-c", "core.autocrlf=false", "apply"]
        if check_only:
            command.append("--check")
        command.append(str(patch))
        result = subprocess.run(
            command,
            cwd=worktree,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
        if result.returncode:
            raise ValidationError(
                f"{task_id}: {patch.name} did not apply\n{result.stdout[-4000:]}"
            )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(part in GENERATED_DIRECTORIES for part in path.relative_to(root).parts)
        and path.suffix.lower() not in GENERATED_SUFFIXES
    }


def validate_lifecycle(
    tasks: tuple[tuple[dict[str, str], Path, dict[str, Any]], ...]
) -> None:
    for expected, task_dir, metadata in tasks:
        task_id = expected["id"]
        timeout = metadata["timeout_seconds"]
        hidden = task_dir / "hidden_tests"
        with tempfile.TemporaryDirectory(prefix=f"repo-agent-{task_id}-") as temporary:
            worktree = Path(temporary) / "repository"
            shutil.copytree(task_dir / "baseline", worktree)
            original = _snapshot(worktree)
            _require_result(
                _run_tests(worktree, worktree / "tests", timeout),
                should_pass=True,
                task_id=task_id,
                stage="healthy public tests",
            )
            _require_result(
                _run_tests(worktree, hidden, timeout),
                should_pass=True,
                task_id=task_id,
                stage="healthy hidden tests",
            )
            _apply_patch(worktree, task_dir / "setup.patch", task_id)
            if _sha256(worktree / expected["target"]) != expected["base_sha256"]:
                raise ValidationError(f"{task_id}: setup did not restore base target")
            _require_result(
                _run_tests(worktree, worktree / "tests", timeout),
                should_pass=True,
                task_id=task_id,
                stage="setup public tests",
            )
            _require_result(
                _run_tests(worktree, hidden, timeout),
                should_pass=False,
                task_id=task_id,
                stage="setup hidden tests",
                signatures=metadata["test_contract"]["failure_signatures"],
            )
            _apply_patch(worktree, task_dir / "gold.patch", task_id)
            if _sha256(worktree / expected["target"]) != expected["fix_sha256"]:
                raise ValidationError(f"{task_id}: gold did not restore fix target")
            _require_result(
                _run_tests(worktree, worktree / "tests", timeout),
                should_pass=True,
                task_id=task_id,
                stage="gold public tests",
            )
            _require_result(
                _run_tests(worktree, hidden, timeout),
                should_pass=True,
                task_id=task_id,
                stage="gold hidden tests",
            )
            if _snapshot(worktree) != original:
                raise ValidationError(f"{task_id}: gold did not restore healthy baseline")
        print(f"[ok] {task_id} {metadata['title']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the public OSS issue suite")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    try:
        tasks = validate_structure()
        if not args.structure_only:
            validate_lifecycle(tasks)
    except (OSError, subprocess.TimeoutExpired, ValidationError) as exc:
        print(f"[error] {exc}")
        return 1
    if args.structure_only:
        print("[ok] public OSS suite structure, provenance, and locks")
    else:
        print("[ok] validated 3 public OSS issue lifecycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
