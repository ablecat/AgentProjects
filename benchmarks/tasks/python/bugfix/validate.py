from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
SUITE_ID = "repo-agent-python-bugfix-v1"
EXPECTED_TASKS = (
    ("py-bugfix-001", "trailing-separator"),
    ("py-bugfix-002", "end-offset"),
    ("py-bugfix-003", "shared-defaults"),
    ("py-bugfix-004", "json-error-boundary"),
    ("py-bugfix-005", "boolean-env"),
    ("py-bugfix-006", "casefold-dedupe"),
)
GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target"}
)
GENERATED_SUFFIXES = frozenset({".pyc", ".pyd", ".pyo"})
TEXT_SUFFIXES = frozenset(
    {".json", ".md", ".patch", ".py", ".toml", ".txt", ".yaml", ".yml"}
)


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
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path.name} must contain a JSON object")
    return value


def _task_digest(task_dir: Path) -> str:
    digest = hashlib.sha256()
    entries = sorted(task_dir.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValidationError(f"links are not allowed in frozen tasks: {path}")
    files = sorted(
        path
        for path in entries
        if path.is_file()
        and not any(
            part in GENERATED_DIRECTORIES for part in path.relative_to(task_dir).parts
        )
        and path.suffix.lower() not in GENERATED_SUFFIXES
    )
    if not files:
        raise ValidationError(f"empty task directory: {task_dir.name}")
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


def _load_root_validator() -> Any:
    path = ROOT.parents[2] / "validate.py"
    spec = importlib.util.spec_from_file_location(
        "repo_agent_python_suite_root_validator", path
    )
    if spec is None or spec.loader is None:
        raise ValidationError("cannot load the root benchmark validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate() -> tuple[tuple[dict[str, Any], Path], ...]:
    manifest = _load_json(MANIFEST_PATH)
    if set(manifest) != {
        "schema_version",
        "suite_id",
        "formal_evaluation",
        "languages",
        "tasks",
    }:
        raise ValidationError("manifest keys do not match the Python-only suite contract")
    if (
        manifest["schema_version"] != 1
        or manifest["suite_id"] != SUITE_ID
        or manifest["formal_evaluation"] is not True
        or manifest["languages"] != ["python"]
    ):
        raise ValidationError("manifest identity does not match the Python-only suite")

    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or len(tasks) != len(EXPECTED_TASKS):
        raise ValidationError("manifest must contain exactly six Python bug-fix tasks")
    validated: list[tuple[dict[str, Any], Path]] = []
    for entry, expected in zip(tasks, EXPECTED_TASKS, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"id", "path", "sha256"}:
            raise ValidationError("manifest task entry is malformed")
        task_id, relative = expected
        if entry["id"] != task_id or entry["path"] != relative:
            raise ValidationError("manifest task order or identity is incorrect")
        path = PurePosixPath(relative)
        if path.is_absolute() or path.parts != (relative,):
            raise ValidationError(f"unsafe task path: {relative}")
        task_path = ROOT / relative
        task_dir = task_path.resolve(strict=True)
        if task_dir.parent != ROOT or not task_dir.is_dir() or task_path.is_symlink():
            raise ValidationError(f"task path is not a real direct child: {relative}")
        metadata = _load_json(task_dir / "metadata.json")
        assets = metadata.get("assets")
        if (
            metadata.get("id") != task_id
            or metadata.get("language") != "python"
            or metadata.get("kind") != "bugfix"
            or not isinstance(assets, dict)
            or assets.get("setup_patch") != "setup.patch"
        ):
            raise ValidationError(f"{task_id}: metadata is not a formal Python bug-fix task")
        if _task_digest(task_dir) != entry["sha256"]:
            raise ValidationError(f"{task_id}: content digest does not match the lock")
        validated.append((entry, task_dir))
    return tuple(validated)


def validate_lifecycle(tasks: tuple[tuple[dict[str, Any], Path], ...]) -> None:
    validator = _load_root_validator()
    schema = validator.load_json(validator.TASK_SCHEMA_PATH)
    try:
        for entry, task_dir in tasks:
            metadata = validator.validate_metadata(task_dir, entry, schema)
            validator.validate_python_lifecycle(task_dir, metadata)
            print(f"[ok] {entry['id']} {metadata['title']}")
    except (validator.ValidationError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Python-only formal suite")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    try:
        tasks = validate()
        if not args.structure_only:
            validate_lifecycle(tasks)
    except (OSError, ValidationError) as exc:
        print(f"[error] {exc}")
        return 1
    if args.structure_only:
        print("[ok] Python-only formal suite structure and locks")
    else:
        print("[ok] validated 6 frozen Python bug-fix tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
