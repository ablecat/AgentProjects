from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import uuid
import xml.etree.ElementTree as ElementTree


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
TASK_SCHEMA_PATH = ROOT / "task.schema.json"
MAVEN_IMAGE = "repo-agent-maven:0.1"
MAVEN_DEPENDENCY_PLUGIN = "org.apache.maven.plugins:maven-dependency-plugin:3.8.1"

GENERATED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "target"}
)
GENERATED_FILE_SUFFIXES = frozenset({".pyc", ".pyd", ".pyo"})
CANONICAL_TEXT_SUFFIXES = frozenset(
    {
        ".java",
        ".json",
        ".md",
        ".patch",
        ".py",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

EXPECTED_CATEGORIES = {
    "local_logic",
    "boundary",
    "cross_file_state",
    "exception_handling",
    "test_gap",
    "collection_invariant",
}
EXPECTED_ASSETS = {
    "baseline": "baseline",
    "setup_patch": "setup.patch",
    "issue": "issue.md",
    "hidden_tests": "hidden_tests",
    "gold_patch": "gold.patch",
}
PYTHON_TEST_CONTRACT = {
    "runner": "unittest",
    "public_pattern": "test_*.py",
    "hidden_pattern": "test_*.py",
    "setup_public": "pass",
    "setup_hidden": "fail",
    "gold_public": "pass",
    "gold_hidden": "pass",
}
JAVA_TEST_CONTRACT = {
    "runner": "maven",
    "public_pattern": "*Test",
    "hidden_pattern": "*HiddenTest",
    "regression_pattern": "*RegressionTest",
    "setup_public": "pass",
    "setup_hidden": "fail",
    "regression_on_setup": "fail",
    "gold_public": "pass",
    "gold_hidden": "pass",
}
REQUIRED_METADATA_KEYS = {
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
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_NAME_RE = re.compile(r"^repo-agent-benchmark-[0-9a-f]{32}$")
_JAVA_CLASS_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_BENCHMARK_LABEL = "io.github.ablecat.repo-agent.benchmark"
_CREATE_RECOVERY_DELAYS = (0.0, 0.1, 0.25, 0.5)
_MAVEN_BOOTSTRAP_ATTEMPTS = 2
_MAVEN_BOOTSTRAP_ARTIFACTS = (
    "org.apache.maven.surefire:surefire-junit-platform:3.5.2",
    "org.junit.platform:junit-platform-launcher:1.9.3",
    "org.junit.platform:junit-platform-launcher:1.11.4",
)
_MAVEN_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
_TRUNCATED_TRANSFER_RE = re.compile(
    r"Could not transfer artifact .+? from/to .+?:\s*"
    r"Premature end of Content-Length delimited message body "
    r"\(expected:\s*([0-9,]+);\s*received:\s*([0-9,]+)\)",
    re.IGNORECASE | re.DOTALL,
)
_MAVEN_TRANSFER_CONTEXT = (
    "could not transfer artifact",
    "could not transfer metadata",
)
_MAVEN_TRANSIENT_TRANSFER_MARKERS = (
    "remote host terminated the handshake",
    "ssl peer shut down incorrectly",
    "connection reset",
    "connection aborted",
    "connection closed unexpectedly",
    "connection refused",
    "broken pipe",
    "read timed out",
    "connect timed out",
    "connection timed out",
    "nohttpresponseexception",
    "failed to respond",
    "premature eof",
    "unexpected end of file",
    "temporary failure in name resolution",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)
_MAVEN_RETRYABLE_HTTP_STATUS_RE = re.compile(
    r"\bstatus code:\s*(?:408|429|500|502|503|504)\b",
    re.IGNORECASE,
)
_MAVEN_CONTAINER_POLICY = (
    "--init",
    "--pull",
    "never",
    "--restart",
    "no",
    "--read-only",
    "--user",
    "10001:10001",
    "--cpus",
    "2",
    "--memory",
    "4g",
    "--memory-swap",
    "4g",
    "--pids-limit",
    "256",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--log-driver",
    "none",
    "--tmpfs",
    "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"expected a JSON object: {path}")
    return value


_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "type",
        "additionalProperties",
        "required",
        "properties",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minimum",
        "maximum",
        "minItems",
        "uniqueItems",
        "items",
        "allOf",
        "if",
        "then",
    }
)


def _validate_schema_definition(schema: object, path: str = "$schema") -> None:
    if not isinstance(schema, dict):
        raise ValidationError(f"{path}: schema node must be an object")
    unsupported = set(schema) - _SUPPORTED_SCHEMA_KEYWORDS
    if unsupported:
        raise ValidationError(
            f"{path}: unsupported JSON Schema keywords: {sorted(unsupported)}"
        )

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict) or not all(
            isinstance(name, str) for name in properties
        ):
            raise ValidationError(f"{path}.properties must be an object")
        for name, child in properties.items():
            _validate_schema_definition(child, f"{path}.properties.{name}")

    items = schema.get("items")
    if items is not None:
        _validate_schema_definition(items, f"{path}.items")

    all_of = schema.get("allOf")
    if all_of is not None:
        if not isinstance(all_of, list) or not all_of:
            raise ValidationError(f"{path}.allOf must be a non-empty array")
        for index, child in enumerate(all_of):
            _validate_schema_definition(child, f"{path}.allOf[{index}]")

    for keyword in ("if", "then"):
        if keyword in schema:
            _validate_schema_definition(schema[keyword], f"{path}.{keyword}")


def _json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _json_equal(a, b) for a, b in zip(left, right)  # type: ignore[arg-type]
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(  # type: ignore[union-attr]
            _json_equal(left[key], right[key]) for key in left  # type: ignore[index]
        )
    return left == right


def _matches_json_type(value: object, expected: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in {int, float},
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }.get(expected, lambda: False)()


def _validate_schema_value(value: object, schema: dict[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str) or not _matches_json_type(
            value, expected_type
        ):
            raise ValidationError(f"{path}: expected JSON type {expected_type!r}")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValidationError(f"{path}: value does not match schema const")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not any(
            _json_equal(value, choice) for choice in choices
        ):
            raise ValidationError(f"{path}: value is not in the schema enum")

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if minimum_length is not None and (
            type(minimum_length) is not int or len(value) < minimum_length
        ):
            raise ValidationError(f"{path}: string is shorter than minLength")
        pattern = schema.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise ValidationError(f"{path}: schema pattern must be a string")
            try:
                matches = re.search(pattern, value)
            except re.error as exc:
                raise ValidationError(f"{path}: invalid schema pattern: {exc}") from exc
            if matches is None:
                raise ValidationError(f"{path}: string does not match schema pattern")

    if type(value) in {int, float}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValidationError(f"{path}: number is below schema minimum")
        if maximum is not None and value > maximum:
            raise ValidationError(f"{path}: number is above schema maximum")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if minimum_items is not None and (
            type(minimum_items) is not int or len(value) < minimum_items
        ):
            raise ValidationError(f"{path}: array is shorter than minItems")
        if schema.get("uniqueItems") is True and any(
            _json_equal(value[left], value[right])
            for left in range(len(value))
            for right in range(left + 1, len(value))
        ):
            raise ValidationError(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            raise ValidationError(f"{path}: schema required must be a string array")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValidationError(f"{path}: missing required properties {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValidationError(f"{path}: schema properties must be an object")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValidationError(f"{path}: unexpected properties {sorted(extra)}")
        for name, child_schema in properties.items():
            if name in value:
                _validate_schema_value(value[name], child_schema, f"{path}.{name}")

    for child_schema in schema.get("allOf", []):
        _validate_schema_value(value, child_schema, path)
    condition = schema.get("if")
    if condition is not None:
        try:
            _validate_schema_value(value, condition, path)
        except ValidationError:
            pass
        else:
            consequent = schema.get("then")
            if consequent is not None:
                _validate_schema_value(value, consequent, path)


def validate_json_schema(value: object, schema: dict[str, Any]) -> None:
    """Validate benchmark metadata with the committed dependency-free schema subset."""

    _validate_schema_definition(schema)
    _validate_schema_value(value, schema, "metadata")


def safe_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValidationError(f"unsafe manifest path: {relative!r}")
    candidate = (ROOT / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValidationError(f"path escapes benchmark root: {relative!r}") from exc
    return candidate


def task_digest(task_dir: Path) -> str:
    digest = hashlib.sha256()
    entries = sorted(task_dir.rglob("*"))
    for path in entries:
        if path.is_symlink():
            raise ValidationError(f"symlinks are not allowed in frozen tasks: {path}")
    files = sorted(
        path
        for path in entries
        if path.is_file()
        and not any(
            part in GENERATED_DIRECTORY_NAMES
            for part in path.relative_to(task_dir).parts
        )
        and path.suffix.lower() not in GENERATED_FILE_SUFFIXES
    )
    if not files:
        raise ValidationError(f"empty task directory: {task_dir}")
    for path in files:
        relative = path.relative_to(task_dir).as_posix().encode("utf-8")
        content = path.read_bytes()
        if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
            content = content.replace(b"\r\n", b"\n")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected_keys = {
        "schema_version",
        "suite_id",
        "languages",
        "task_contract",
        "validator",
        "tasks",
        "reserved_development_tasks",
    }
    if set(manifest) != expected_keys:
        raise ValidationError("manifest keys do not match the v1 contract")
    if manifest["schema_version"] != 1:
        raise ValidationError("manifest schema_version must be 1")
    if manifest["suite_id"] != "repo-agent-python-java-day3-v1":
        raise ValidationError("unexpected suite_id")
    if manifest["languages"] != ["python", "java"]:
        raise ValidationError("manifest languages must be ordered python, java")
    if (
        manifest["task_contract"] != TASK_SCHEMA_PATH.name
        or not TASK_SCHEMA_PATH.is_file()
    ):
        raise ValidationError("task schema is missing")
    if manifest["validator"] != Path(__file__).name:
        raise ValidationError("manifest validator path is inconsistent")

    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 12:
        raise ValidationError("manifest must contain exactly twelve bug-fix tasks")
    expected_ids = [f"py-bugfix-{index:03d}" for index in range(1, 7)] + [
        f"java-bugfix-{index:03d}" for index in range(1, 7)
    ]
    actual_ids = [entry.get("id") for entry in tasks if isinstance(entry, dict)]
    if actual_ids != expected_ids:
        raise ValidationError(
            "bug-fix task IDs must be ordered py-bugfix-001..006 then "
            "java-bugfix-001..006"
        )
    for entry in tasks:
        if not isinstance(entry, dict) or set(entry) != {"id", "path", "sha256"}:
            raise ValidationError(
                "each task manifest entry requires id, path, and sha256"
            )
        digest = entry["sha256"]
        if digest != "PENDING" and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValidationError(f"invalid task digest for {entry['id']}")
        language = "python" if entry["id"].startswith("py-") else "java"
        if not entry["path"].startswith(f"tasks/{language}/bugfix/"):
            raise ValidationError(f"task path is misplaced: {entry['path']}")

    reserved = manifest["reserved_development_tasks"]
    if not isinstance(reserved, list) or len(reserved) != 4:
        raise ValidationError("manifest must reserve exactly four development tasks")
    for index, entry in enumerate(reserved, start=1):
        expected_id = f"py-development-{index:03d}"
        if not isinstance(entry, dict) or set(entry) != {"id", "path", "status"}:
            raise ValidationError("invalid reserved development task entry")
        if entry["id"] != expected_id or entry["status"] != "reserved":
            raise ValidationError(f"invalid reservation for {expected_id}")
        if not entry["path"].startswith("tasks/python/development/"):
            raise ValidationError(f"development path is misplaced: {entry['path']}")
        safe_path(entry["path"])
    return tasks


def _validate_signatures(value: object, task_id: str, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(item for item in value if isinstance(item, str)))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValidationError(f"{task_id}: {field} must be unique and non-empty")
    return value


def validate_metadata(
    task_dir: Path,
    entry: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = load_json(task_dir / "metadata.json")
    validate_json_schema(
        metadata, schema if schema is not None else load_json(TASK_SCHEMA_PATH)
    )
    task_id = entry["id"]
    if set(metadata) != REQUIRED_METADATA_KEYS:
        raise ValidationError(f"{task_id}: metadata keys do not match the v1 schema")
    if metadata["schema_version"] != 1 or metadata["id"] != task_id:
        raise ValidationError(f"{task_id}: identity mismatch")
    language = "python" if task_id.startswith("py-") else "java"
    if metadata["language"] != language or metadata["kind"] != "bugfix":
        raise ValidationError(f"{task_id}: invalid language or kind")
    if metadata["category"] not in EXPECTED_CATEGORIES:
        raise ValidationError(f"{task_id}: invalid category")
    if metadata["difficulty"] not in {"easy", "medium"}:
        raise ValidationError(f"{task_id}: invalid difficulty")
    timeout = metadata["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 30:
        raise ValidationError(f"{task_id}: timeout must be an integer from 1 to 30")
    if metadata["assets"] != EXPECTED_ASSETS:
        raise ValidationError(f"{task_id}: asset paths do not match the fixed layout")

    contract = metadata["test_contract"]
    if not isinstance(contract, dict):
        raise ValidationError(f"{task_id}: test_contract must be an object")
    signatures = _validate_signatures(
        contract.get("failure_signatures"), task_id, "failure_signatures"
    )
    if language == "python":
        fixed_contract = {
            key: value for key, value in contract.items() if key != "failure_signatures"
        }
        if fixed_contract != PYTHON_TEST_CONTRACT:
            raise ValidationError(f"{task_id}: invalid Python test lifecycle contract")
        if not all(item.startswith("test_") for item in signatures):
            raise ValidationError(f"{task_id}: invalid Python failure signature")
        if metadata["constraints"] != {
            "network": False,
            "third_party_dependencies": [],
            "python_min": "3.11",
        }:
            raise ValidationError(
                f"{task_id}: Python tasks must be offline and dependency-free"
            )
    else:
        regression_signatures = _validate_signatures(
            contract.get("regression_failure_signatures"),
            task_id,
            "regression_failure_signatures",
        )
        fixed_contract = {
            key: value
            for key, value in contract.items()
            if key not in {"failure_signatures", "regression_failure_signatures"}
        }
        if fixed_contract != JAVA_TEST_CONTRACT:
            raise ValidationError(f"{task_id}: invalid Java test lifecycle contract")
        if not all(
            _JAVA_CLASS_RE.fullmatch(item)
            for item in (*signatures, *regression_signatures)
        ):
            raise ValidationError(f"{task_id}: invalid Java failure signature")
        if metadata["constraints"] != {
            "network": False,
            "third_party_dependencies": [],
            "test_dependencies": ["org.junit.jupiter:junit-jupiter:5.11.4"],
            "java_release": "17",
            "build": "maven-single-module",
        }:
            raise ValidationError(
                f"{task_id}: Java tasks must use the fixed single-module contract"
            )

    slug = task_dir.name
    if metadata["slug"] != slug or not metadata["title"].strip():
        raise ValidationError(f"{task_id}: slug or title is inconsistent")
    for asset in EXPECTED_ASSETS.values():
        if not (task_dir / asset).exists():
            raise ValidationError(f"{task_id}: missing asset {asset}")
    if not (task_dir / "issue.md").read_text(encoding="utf-8").strip():
        raise ValidationError(f"{task_id}: issue is empty")

    if language == "python":
        if not list((task_dir / "baseline" / "tests").glob("test_*.py")):
            raise ValidationError(f"{task_id}: no public tests")
        if not list((task_dir / "hidden_tests").glob("test_*.py")):
            raise ValidationError(f"{task_id}: no hidden tests")
    else:
        _validate_java_layout(task_dir, task_id)
    return metadata


def _validate_java_layout(task_dir: Path, task_id: str) -> None:
    baseline = task_dir / "baseline"
    pom = baseline / "pom.xml"
    if not pom.is_file():
        raise ValidationError(f"{task_id}: root pom.xml is missing")
    poms = list(baseline.rglob("pom.xml"))
    if poms != [pom]:
        raise ValidationError(f"{task_id}: Maven fixture must contain one root pom.xml")
    if (baseline / ".mvn").exists():
        raise ValidationError(f"{task_id}: project-level .mvn configuration is forbidden")
    _validate_benchmark_pom(pom, task_id, task_dir.name)
    public = list((baseline / "src" / "test" / "java").rglob("*Test.java"))
    hidden = list((task_dir / "hidden_tests").rglob("*HiddenTest.java"))
    production = list((baseline / "src" / "main" / "java").rglob("*.java"))
    if not public or not hidden or not production:
        raise ValidationError(
            f"{task_id}: Java fixture needs production, public, and hidden sources"
        )
    if any(path.name.endswith("RegressionTest.java") for path in public):
        raise ValidationError(f"{task_id}: regression test must be introduced by gold.patch")


def _maven_children(
    element: ElementTree.Element,
    expected: Sequence[str],
    task_id: str,
    context: str,
) -> list[ElementTree.Element]:
    children = list(element)
    expected_tags = [f"{{{_MAVEN_NAMESPACE}}}{name}" for name in expected]
    if [child.tag for child in children] != expected_tags:
        raise ValidationError(f"{task_id}: unexpected Maven {context} structure")
    if any(child.attrib for child in children):
        raise ValidationError(f"{task_id}: Maven {context} elements must not have attributes")
    return children


def _maven_text(element: ElementTree.Element) -> str:
    return (element.text or "").strip()


def _validate_dependency(
    dependency: ElementTree.Element,
    expected: tuple[str, ...],
    task_id: str,
    context: str,
) -> None:
    names = ("groupId", "artifactId", "version")
    if len(expected) == 4:
        names += ("scope",)
    children = _maven_children(dependency, names, task_id, context)
    if tuple(_maven_text(child) for child in children) != expected:
        raise ValidationError(f"{task_id}: unexpected Maven {context} dependency")


def _validate_benchmark_pom(pom: Path, task_id: str, artifact_id: str) -> None:
    raw = pom.read_bytes()
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValidationError(f"{task_id}: Maven fixture must not contain DTDs or entities")
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ValidationError(f"{task_id}: invalid pom.xml: {exc}") from exc
    if root.tag != f"{{{_MAVEN_NAMESPACE}}}project":
        raise ValidationError(f"{task_id}: unexpected Maven project namespace")

    root_children = _maven_children(
        root,
        (
            "modelVersion",
            "groupId",
            "artifactId",
            "version",
            "properties",
            "dependencies",
            "build",
        ),
        task_id,
        "project",
    )
    if tuple(_maven_text(child) for child in root_children[:4]) != (
        "4.0.0",
        "dev.repoagent.benchmarks",
        artifact_id,
        "1.0.0",
    ):
        raise ValidationError(f"{task_id}: unexpected Maven project coordinates")

    properties = _maven_children(
        root_children[4],
        ("maven.compiler.release", "project.build.sourceEncoding", "junit.version"),
        task_id,
        "properties",
    )
    if tuple(_maven_text(child) for child in properties) != ("17", "UTF-8", "5.11.4"):
        raise ValidationError(f"{task_id}: unexpected Maven properties")

    project_dependencies = _maven_children(
        root_children[5], ("dependency",), task_id, "project dependencies"
    )
    _validate_dependency(
        project_dependencies[0],
        ("org.junit.jupiter", "junit-jupiter", "${junit.version}", "test"),
        task_id,
        "project",
    )

    build_children = _maven_children(root_children[6], ("plugins",), task_id, "build")
    plugins = _maven_children(
        build_children[0], ("plugin",), task_id, "build plugins"
    )
    plugin_children = _maven_children(
        plugins[0],
        ("groupId", "artifactId", "version", "configuration"),
        task_id,
        "Surefire plugin",
    )
    if tuple(_maven_text(child) for child in plugin_children[:3]) != (
        "org.apache.maven.plugins",
        "maven-surefire-plugin",
        "3.5.2",
    ):
        raise ValidationError(f"{task_id}: unexpected Maven build plugin")
    configuration = _maven_children(
        plugin_children[3], ("useModulePath",), task_id, "Surefire configuration"
    )
    if _maven_text(configuration[0]) != "false":
        raise ValidationError(f"{task_id}: unexpected Maven Surefire configuration")


def run_python_tests(
    worktree: Path, tests_dir: Path, pattern: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(worktree)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(tests_dir),
            "-p",
            pattern,
        ],
        cwd=worktree,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def require_test_result(
    result: subprocess.CompletedProcess[str],
    *,
    should_pass: bool,
    stage: str,
    task_id: str,
    signatures: Sequence[str] = (),
) -> None:
    passed = result.returncode == 0
    if passed != should_pass:
        expectation = "pass" if should_pass else "fail"
        raise ValidationError(
            f"{task_id}: {stage} should {expectation} (exit {result.returncode})\n"
            f"{_bounded_output(result.stdout)}"
        )
    for signature in signatures:
        if signature not in result.stdout:
            raise ValidationError(
                f"{task_id}: {stage} output is missing failure signature {signature}\n"
                f"{_bounded_output(result.stdout)}"
            )


def apply_patch(
    worktree: Path,
    patch_path: Path,
    task_id: str,
    *,
    include: str | None = None,
) -> None:
    for action in ("--check", None):
        args = ["git", "-c", "core.autocrlf=false", "apply"]
        if action is not None:
            args.append(action)
        if include is not None:
            args.append(f"--include={include}")
        args.append(str(patch_path))
        try:
            result = subprocess.run(
                args,
                cwd=worktree,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValidationError("git is required to validate benchmark patches") from exc
        if result.returncode != 0:
            raise ValidationError(
                f"{task_id}: {' '.join(args[3:-1])} failed for {patch_path.name}\n"
                f"{result.stdout}"
            )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            part in GENERATED_DIRECTORY_NAMES for part in path.relative_to(root).parts
        )
    }


def validate_python_lifecycle(task_dir: Path, metadata: dict[str, Any]) -> None:
    task_id = metadata["id"]
    timeout = metadata["timeout_seconds"]
    baseline = task_dir / "baseline"
    hidden_tests = task_dir / "hidden_tests"
    original = tree_snapshot(baseline)
    with tempfile.TemporaryDirectory(prefix=f"repo-agent-{task_id}-") as temporary:
        worktree = Path(temporary) / "worktree"
        shutil.copytree(baseline, worktree)

        baseline_public = run_python_tests(
            worktree, worktree / "tests", "test_*.py", timeout
        )
        require_test_result(
            baseline_public,
            should_pass=True,
            stage="healthy public tests",
            task_id=task_id,
        )
        baseline_hidden = run_python_tests(
            worktree, hidden_tests, "test_*.py", timeout
        )
        require_test_result(
            baseline_hidden,
            should_pass=True,
            stage="healthy hidden tests",
            task_id=task_id,
        )

        apply_patch(worktree, task_dir / "setup.patch", task_id)
        setup_public = run_python_tests(
            worktree, worktree / "tests", "test_*.py", timeout
        )
        require_test_result(
            setup_public,
            should_pass=True,
            stage="setup public tests",
            task_id=task_id,
        )
        setup_hidden = run_python_tests(
            worktree, hidden_tests, "test_*.py", timeout
        )
        require_test_result(
            setup_hidden,
            should_pass=False,
            stage="setup hidden tests",
            task_id=task_id,
            signatures=metadata["test_contract"]["failure_signatures"],
        )

        apply_patch(worktree, task_dir / "gold.patch", task_id)
        gold_public = run_python_tests(
            worktree, worktree / "tests", "test_*.py", timeout
        )
        require_test_result(
            gold_public,
            should_pass=True,
            stage="gold public tests",
            task_id=task_id,
        )
        gold_hidden = run_python_tests(
            worktree, hidden_tests, "test_*.py", timeout
        )
        require_test_result(
            gold_hidden,
            should_pass=True,
            stage="gold hidden tests",
            task_id=task_id,
        )
        if tree_snapshot(worktree) != original:
            raise ValidationError(
                f"{task_id}: gold patch does not restore the frozen healthy baseline"
            )


def _copy_baseline(baseline: Path, destination: Path) -> None:
    shutil.copytree(
        baseline,
        destination,
        ignore=shutil.ignore_patterns(*GENERATED_DIRECTORY_NAMES),
    )


def prepare_container_writable_tree(root: Path) -> None:
    """Make only a newly created disposable tree writable by sandbox UID 10001."""

    if root.is_symlink() or not root.is_dir():
        raise ValidationError(f"container writable root is not a real directory: {root}")
    entries = [root, *sorted(root.rglob("*"))]
    for path in entries:
        if path.is_symlink():
            raise ValidationError(f"symlink in disposable container tree: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValidationError(f"special file in disposable container tree: {path}")
    for path in entries:
        try:
            path.chmod(0o777 if path.is_dir() else 0o666)
        except OSError as exc:
            raise ValidationError(
                f"cannot prepare disposable tree for sandbox UID 10001: {path}"
            ) from exc


def _java_test_names(root: Path, suffix: str) -> list[str]:
    names = sorted(path.name.removesuffix(".java") for path in root.rglob(f"*{suffix}.java"))
    if not names or any(not _JAVA_CLASS_RE.fullmatch(name) for name in names):
        raise ValidationError(f"invalid or missing Java {suffix} test class")
    if len(names) != len(set(names)):
        raise ValidationError(f"duplicate Java {suffix} test class name")
    return names


def _inject_hidden_tests(worktree: Path, hidden_tests: Path) -> list[str]:
    destination = worktree / "src" / "test" / "java"
    for source in sorted(hidden_tests.rglob("*.java")):
        relative = source.relative_to(hidden_tests)
        target = destination / relative
        if target.exists():
            raise ValidationError(f"hidden test collides with public source: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return _java_test_names(destination, "HiddenTest")


def _docker_capture(
    argv: Sequence[str], timeout: int
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValidationError("docker is required for Java benchmark validation") from exc


def _container_is_absent(output: str) -> bool:
    return "no such container" in output.casefold()


def _remove_container_id(container_id: str) -> None:
    try:
        removed = _docker_capture(
            ("docker", "container", "rm", "--force", container_id), 30
        )
    except (ValidationError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(
            f"failed to remove Maven check container {container_id[:12]}"
        ) from exc
    if removed.returncode == 0 or (
        removed.returncode == 1 and _container_is_absent(removed.stdout)
    ):
        return
    raise ValidationError(
        f"failed to remove Maven check container {container_id[:12]} "
        f"(exit {removed.returncode})\n{_bounded_output(removed.stdout)}"
    )


def _remove_generated_container(reference: str, *, unresolved_name: bool = False) -> None:
    if unresolved_name:
        if not _CONTAINER_NAME_RE.fullmatch(reference):
            return
        for delay in _CREATE_RECOVERY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                inspected = _docker_capture(
                    (
                        "docker",
                        "container",
                        "inspect",
                        "--format",
                        '{{.Id}}|{{index .Config.Labels "'
                        + _BENCHMARK_LABEL
                        + '"}}',
                        reference,
                    ),
                    30,
                )
            except (ValidationError, subprocess.TimeoutExpired):
                continue
            if inspected.returncode != 0:
                if _container_is_absent(inspected.stdout):
                    return
                continue
            identity, separator, owned = inspected.stdout.strip().partition("|")
            if (
                separator
                and _CONTAINER_ID_RE.fullmatch(identity)
                and owned == "true"
            ):
                reference = identity
                break
            return
        else:
            raise ValidationError(
                f"could not resolve generated Maven container {reference!r} for cleanup"
            )
    if not _CONTAINER_ID_RE.fullmatch(reference):
        return
    _remove_container_id(reference)


def _run_maven_container(
    worktree: Path,
    cache: Path,
    maven_args: Sequence[str],
    timeout: int,
    *,
    network: str,
) -> subprocess.CompletedProcess[str]:
    workspace_source = str(worktree.resolve(strict=True))
    cache_source = str(cache.resolve(strict=True))
    if "," in workspace_source or "," in cache_source:
        raise ValidationError("Docker bind source paths containing commas are unsupported")
    container_name = f"repo-agent-benchmark-{uuid.uuid4().hex}"
    create_args = (
        "docker",
        "container",
        "create",
        "--name",
        container_name,
        "--label",
            f"{_BENCHMARK_LABEL}=true",
        *_MAVEN_CONTAINER_POLICY,
        "--env",
        "MAVEN_OPTS=-Djansi.force=false -Djansi.tmpdir=/cache",
        "--network",
        network,
        "--mount",
        f"type=bind,source={workspace_source},target=/workspace",
        "--mount",
        f"type=bind,source={cache_source},target=/cache",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "mvn",
        MAVEN_IMAGE,
        "--batch-mode",
        "--no-transfer-progress",
        "-Dmaven.repo.local=/cache",
        *maven_args,
    )
    try:
        create = _docker_capture(create_args, timeout=30)
    except BaseException:
        _remove_generated_container(container_name, unresolved_name=True)
        raise
    container_id = create.stdout.strip()
    if create.returncode != 0 or not _CONTAINER_ID_RE.fullmatch(container_id):
        _remove_generated_container(container_name, unresolved_name=True)
        raise ValidationError(
            f"Docker create failed for Maven check (exit {create.returncode})\n"
            f"{_bounded_output(create.stdout)}"
        )

    result: subprocess.CompletedProcess[str] | None = None
    pending: BaseException | None = None
    try:
        result = _docker_capture(
            ("docker", "container", "start", "--attach", container_id), timeout
        )
    except BaseException as exc:
        pending = exc
    finally:
        try:
            _remove_container_id(container_id)
        except BaseException as cleanup_exc:
            if pending is not None:
                cleanup_exc.add_note(
                    f"Maven container execution also failed: {type(pending).__name__}: {pending}"
                )
            pending = cleanup_exc
    if pending is not None:
        raise pending
    assert result is not None
    return result


def bootstrap_maven_dependencies(
    worktree: Path, cache: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    """Resolve fixed Maven dependencies without executing repository tests."""

    bootstrap_steps = (
        (
            "-DskipTests",
            f"{MAVEN_DEPENDENCY_PLUGIN}:go-offline",
        ),
        *(
            (
                f"{MAVEN_DEPENDENCY_PLUGIN}:get",
                f"-Dartifact={artifact}",
                "-Dtransitive=true",
            )
            for artifact in _MAVEN_BOOTSTRAP_ARTIFACTS
        ),
    )
    outputs: list[str] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for maven_args in bootstrap_steps:
        for attempt in range(_MAVEN_BOOTSTRAP_ATTEMPTS):
            try:
                result = _run_maven_container(
                    worktree,
                    cache,
                    maven_args,
                    timeout,
                    network="bridge",
                )
            except subprocess.TimeoutExpired:
                if attempt + 1 == _MAVEN_BOOTSTRAP_ATTEMPTS:
                    raise
                continue
            outputs.append(result.stdout)
            last_result = result
            if result.returncode == 0:
                break
            if (
                attempt + 1 == _MAVEN_BOOTSTRAP_ATTEMPTS
                or not _is_retryable_maven_transfer(result.stdout)
            ):
                return subprocess.CompletedProcess(
                    result.args,
                    result.returncode,
                    "\n".join(outputs),
                )

    assert last_result is not None
    return subprocess.CompletedProcess(
        last_result.args,
        0,
        "\n".join(outputs),
    )


def _is_truncated_maven_transfer(output: str) -> bool:
    match = _TRUNCATED_TRANSFER_RE.search(output)
    if match is None:
        return False
    expected = int(match.group(1).replace(",", ""))
    received = int(match.group(2).replace(",", ""))
    return received < expected


def _is_retryable_maven_transfer(output: str) -> bool:
    if _is_truncated_maven_transfer(output):
        return True
    normalized = output.casefold()
    if not any(marker in normalized for marker in _MAVEN_TRANSFER_CONTEXT):
        return False
    return (
        any(marker in normalized for marker in _MAVEN_TRANSIENT_TRANSFER_MARKERS)
        or _MAVEN_RETRYABLE_HTTP_STATUS_RE.search(output) is not None
    )


def run_maven_tests(
    worktree: Path,
    cache: Path,
    test_names: Sequence[str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed Maven test selection with network access disabled."""

    return _run_maven_container(
        worktree,
        cache,
        (
            "--offline",
            f"-Dtest={','.join(test_names)}",
            "-DfailIfNoTests=true",
            "test",
        ),
        timeout,
        network="none",
    )


def _run_java_stage(
    stage_root: Path,
    baseline: Path,
    cache: Path,
    task_dir: Path,
    task_id: str,
    timeout: int,
    *,
    setup: bool = False,
    gold: bool = False,
    test_only_gold: bool = False,
    hidden: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    worktree = stage_root / uuid.uuid4().hex
    _copy_baseline(baseline, worktree)
    if setup:
        apply_patch(worktree, task_dir / "setup.patch", task_id)
    if gold:
        apply_patch(worktree, task_dir / "gold.patch", task_id)
    if test_only_gold:
        apply_patch(
            worktree,
            task_dir / "gold.patch",
            task_id,
            include="src/test/**",
        )

    if hidden:
        test_names = _inject_hidden_tests(worktree, task_dir / "hidden_tests")
    elif gold or test_only_gold:
        test_names = _java_test_names(worktree / "src" / "test" / "java", "Test")
        if test_only_gold:
            test_names = [name for name in test_names if name.endswith("RegressionTest")]
    else:
        test_names = _java_test_names(worktree / "src" / "test" / "java", "Test")
    prepare_container_writable_tree(worktree)
    result = run_maven_tests(worktree, cache, test_names, timeout)
    return result, worktree, test_names


def validate_java_lifecycle(
    task_dir: Path,
    metadata: dict[str, Any],
    cache: Path,
    stage_root: Path,
) -> None:
    task_id = metadata["id"]
    timeout = metadata["timeout_seconds"]
    baseline = task_dir / "baseline"
    contract = metadata["test_contract"]

    bootstrap_worktree = stage_root / uuid.uuid4().hex
    _copy_baseline(baseline, bootstrap_worktree)
    prepare_container_writable_tree(bootstrap_worktree)
    bootstrap = bootstrap_maven_dependencies(
        bootstrap_worktree, cache, 300
    )
    require_test_result(
        bootstrap,
        should_pass=True,
        stage="dependency bootstrap",
        task_id=task_id,
    )

    healthy_public, _, _ = _run_java_stage(
        stage_root, baseline, cache, task_dir, task_id, timeout
    )
    require_test_result(
        healthy_public,
        should_pass=True,
        stage="offline healthy public tests",
        task_id=task_id,
    )
    healthy_hidden, _, _ = _run_java_stage(
        stage_root, baseline, cache, task_dir, task_id, timeout, hidden=True
    )
    require_test_result(
        healthy_hidden,
        should_pass=True,
        stage="offline healthy hidden tests",
        task_id=task_id,
    )

    setup_public, _, _ = _run_java_stage(
        stage_root, baseline, cache, task_dir, task_id, timeout, setup=True
    )
    require_test_result(
        setup_public,
        should_pass=True,
        stage="offline setup public tests",
        task_id=task_id,
    )
    setup_hidden, _, _ = _run_java_stage(
        stage_root,
        baseline,
        cache,
        task_dir,
        task_id,
        timeout,
        setup=True,
        hidden=True,
    )
    require_test_result(
        setup_hidden,
        should_pass=False,
        stage="offline setup hidden tests",
        task_id=task_id,
        signatures=contract["failure_signatures"],
    )

    regression, _, regression_names = _run_java_stage(
        stage_root,
        baseline,
        cache,
        task_dir,
        task_id,
        timeout,
        setup=True,
        test_only_gold=True,
    )
    if not regression_names:
        raise ValidationError(f"{task_id}: gold patch adds no regression test")
    require_test_result(
        regression,
        should_pass=False,
        stage="new regression tests on buggy setup",
        task_id=task_id,
        signatures=contract["regression_failure_signatures"],
    )

    gold_public, gold_worktree, gold_names = _run_java_stage(
        stage_root,
        baseline,
        cache,
        task_dir,
        task_id,
        timeout,
        setup=True,
        gold=True,
    )
    if not any(name.endswith("RegressionTest") for name in gold_names):
        raise ValidationError(f"{task_id}: gold patch adds no focused regression test")
    require_test_result(
        gold_public,
        should_pass=True,
        stage="offline gold public and regression tests",
        task_id=task_id,
    )
    gold_hidden, _, _ = _run_java_stage(
        stage_root,
        baseline,
        cache,
        task_dir,
        task_id,
        timeout,
        setup=True,
        gold=True,
        hidden=True,
    )
    require_test_result(
        gold_hidden,
        should_pass=True,
        stage="offline gold hidden tests",
        task_id=task_id,
    )

    if tree_snapshot(gold_worktree / "src" / "main") != tree_snapshot(
        baseline / "src" / "main"
    ):
        raise ValidationError(f"{task_id}: gold patch does not restore production sources")
    if (gold_worktree / "pom.xml").read_bytes() != (baseline / "pom.xml").read_bytes():
        raise ValidationError(f"{task_id}: gold patch changes the Maven build descriptor")


def _bounded_output(value: str, limit: int = 32 * 1024) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n...[truncated by benchmark validator]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen Day 3 Python and Java bug-fix tasks"
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="validate contracts and content locks without running task tests",
    )
    parser.add_argument(
        "--print-digests",
        action="store_true",
        help="print calculated task digests and skip lock comparison",
    )
    parser.add_argument(
        "--allow-bootstrap-network",
        action="store_true",
        help=(
            "explicitly allow a Maven dependency bootstrap container to use the "
            "network; every lifecycle check after bootstrap remains offline"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        schema = load_json(TASK_SCHEMA_PATH)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValidationError(
                "task.schema.json must use JSON Schema draft 2020-12"
            )
        manifest = load_json(MANIFEST_PATH)
        tasks = validate_manifest(manifest)
        categories: dict[str, set[str]] = {"python": set(), "java": set()}
        metadata_by_id: dict[str, dict[str, Any]] = {}
        task_dirs: dict[str, Path] = {}
        for entry in tasks:
            task_dir = safe_path(entry["path"])
            if not task_dir.is_dir():
                raise ValidationError(f"missing task directory: {entry['path']}")
            metadata = validate_metadata(task_dir, entry, schema)
            metadata_by_id[entry["id"]] = metadata
            task_dirs[entry["id"]] = task_dir
            categories[metadata["language"]].add(metadata["category"])
            calculated = task_digest(task_dir)
            if args.print_digests:
                print(f"{entry['id']} {calculated}")
            elif calculated != entry["sha256"]:
                raise ValidationError(
                    f"{entry['id']}: content lock mismatch; expected "
                    f"{entry['sha256']}, got {calculated}"
                )
        for language, covered in categories.items():
            if covered != EXPECTED_CATEGORIES:
                missing = ", ".join(sorted(EXPECTED_CATEGORIES - covered))
                raise ValidationError(
                    f"{language} suite does not cover every required category: {missing}"
                )

        if not args.structure_only and not args.print_digests:
            for entry in tasks[:6]:
                metadata = metadata_by_id[entry["id"]]
                validate_python_lifecycle(task_dirs[entry["id"]], metadata)
                print(f"[ok] {entry['id']} {metadata['title']}")
            if not args.allow_bootstrap_network:
                raise ValidationError(
                    "Java lifecycle needs an explicitly authorized dependency bootstrap; "
                    "rerun with --allow-bootstrap-network"
                )
            with tempfile.TemporaryDirectory(
                prefix="repo-agent-java-benchmark-"
            ) as temporary:
                temporary_root = Path(temporary)
                cache = temporary_root / "maven-cache"
                stages = temporary_root / "stages"
                cache.mkdir()
                stages.mkdir()
                temporary_root.chmod(0o700)
                prepare_container_writable_tree(cache)
                for entry in tasks[6:]:
                    metadata = metadata_by_id[entry["id"]]
                    validate_java_lifecycle(
                        task_dirs[entry["id"]], metadata, cache, stages
                    )
                    print(f"[ok] {entry['id']} {metadata['title']}")
    except (ValidationError, subprocess.TimeoutExpired) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    if args.structure_only:
        print("[ok] benchmark structure, contracts, and locks")
    elif args.print_digests:
        print("[ok] calculated canonical task digests")
    else:
        print("[ok] validated 12 frozen bug-fix tasks and 4 reservations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
