"""Deterministic Python and Maven checks in staged Docker sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import tomllib
from typing import Callable, Literal, Protocol, Sequence, TypeAlias
import uuid
import xml.etree.ElementTree as ElementTree

from .patches import ValidatedPatch, validate_patch
from .git_config import effective_core_autocrlf
from .policy import DEFAULT_PATH_POLICY, RepositoryPathError
from .sandbox import (
    CommandOutcome,
    DEFAULT_POLICY,
    SubprocessCommandRunner,
    _sanitized_git_environment,
    _validated_repository,
)


MAX_PHASE_TIMEOUT_SECONDS = 300.0
MAX_RUN_TIMEOUT_SECONDS = 1200.0
MAX_CHECK_OUTPUT_BYTES = 256 * 1024
DEFAULT_CHECK_OUTPUT_BYTES = 64 * 1024

_CONTROL_TIMEOUT_SECONDS = 30.0
_CHECK_ROOT_RE = re.compile(r"^repo-agent-check-[0-9a-z_]+$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RUN_LABEL = "io.github.ablecat.repo-agent.check.run"
_SEQUENCE_LABEL = "io.github.ablecat.repo-agent.check.sequence"
_INSPECT_PREFIX = "repo-agent-check-owned"
_RECOVERY_INITIAL_DELAY_SECONDS = 0.05
_RECOVERY_MAX_DELAY_SECONDS = 1.0
_GRADLE_MARKERS = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "gradlew",
        "gradlew.bat",
        "settings.gradle",
        "settings.gradle.kts",
    }
)
_PYTHON_MARKERS = frozenset(
    {"pyproject.toml", "pytest.ini", "requirements.txt", "setup.cfg"}
)
_ALLOWED_JAVA_RELEASES = frozenset({17, 21})
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_REPOSITORY_FILES = 20_000
_MAX_REPOSITORY_BYTES = 512 * 1024 * 1024
_MAVEN_XML_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
_MAVEN_SCHEMA_LOCATION_ATTRIBUTE = (
    "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
)
_MAVEN_ALLOWED_PROPERTIES = frozenset(
    {
        "junit.version",
        "maven.compiler.release",
        "project.build.sourceEncoding",
        "surefire.version",
    }
)
_MAVEN_COMPILER_PLUGIN_VERSION = "3.13.0"
_SUREFIRE_PLATFORM_VERSIONS = {"3.5.2": "1.9.3"}
_JUNIT_PLATFORM_VERSIONS = {"5.11.4": "1.11.4"}
_MAVEN_BOOTSTRAP_ATTEMPTS = 2
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
_MAVEN_TRUNCATED_TRANSFER_RE = re.compile(
    r"Could not transfer artifact .+? from/to .+?:\s*"
    r"Premature end of Content-Length delimited message body "
    r"\(expected:\s*([0-9,]+);\s*received:\s*([0-9,]+)\)",
    re.IGNORECASE | re.DOTALL,
)


CheckStatus: TypeAlias = Literal[
    "passed",
    "failed",
    "timed_out",
    "bootstrap_required",
    "policy_denied",
    "setup_error",
    "cleanup_error",
]
PhaseStatus: TypeAlias = Literal["passed", "failed", "timed_out", "setup_error"]
PhaseKind: TypeAlias = Literal["bootstrap", "verify"]


class CheckError(RuntimeError):
    """Base error raised before a check can produce a structured result."""


class CheckPolicyError(CheckError):
    """The repository or requested execution violates the Day 3 policy."""


class CheckCleanupError(CheckError):
    """One or more run-owned resources could not be removed exactly."""


class CheckTimeoutError(CheckError):
    """The total run deadline expired during host-side preparation."""


@dataclass(frozen=True, slots=True)
class CheckProfile:
    """A detected, fixed command profile for one supported project type."""

    id: str
    language: Literal["python", "java"]
    image: str
    manifest: str
    bootstrap_argv: tuple[tuple[str, ...], ...]
    check_argv: tuple[str, ...]
    cache_target: str = "/dependencies"
    java_release: int | None = None

    @property
    def bootstrap_required(self) -> bool:
        return bool(self.bootstrap_argv)


@dataclass(frozen=True, slots=True)
class CheckPhaseResult:
    """Bounded output and policy facts for one bootstrap or verify container."""

    name: str
    kind: PhaseKind
    status: PhaseStatus
    network: str
    argv: tuple[str, ...]
    exit_code: int | None
    output: str
    error: str | None
    duration_ms: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class CheckRunResult:
    """Complete deterministic result for one check profile invocation."""

    status: CheckStatus
    profile: CheckProfile | None
    base_commit: str | None
    candidate_applied: bool
    phases: tuple[CheckPhaseResult, ...]
    duration_ms: int
    error: str | None = None
    cleanup_ok: bool = True

    @property
    def ok(self) -> bool:
        return self.status == "passed" and self.cleanup_ok


class CheckCommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome: ...


CheckCommandRunnerLike: TypeAlias = CheckCommandRunner | Callable[..., CommandOutcome]


def detect_check_profile(repository: str | os.PathLike[str]) -> CheckProfile:
    """Detect one supported root-level test profile without executing project code."""

    try:
        root = Path(repository).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise CheckPolicyError("check repository must be an existing directory") from exc
    if not root.is_dir():
        raise CheckPolicyError("check repository must be a directory")
    _reject_links_and_sensitive_files(root)

    files = tuple(_visible_files(root))
    gradle = sorted(path.relative_to(root).as_posix() for path in files if path.name in _GRADLE_MARKERS)
    if gradle:
        raise CheckPolicyError(
            "Gradle projects are not supported: " + ", ".join(gradle[:5])
        )

    root_pom = root / "pom.xml"
    nested_poms = sorted(
        path.relative_to(root).as_posix()
        for path in files
        if path.name == "pom.xml" and path != root_pom
    )
    has_maven = root_pom.is_file()
    has_python = any((root / marker).is_file() for marker in _PYTHON_MARKERS) or (
        root / "tests"
    ).is_dir()
    if has_maven and has_python:
        raise CheckPolicyError("mixed Python and Maven repositories are not supported")
    if nested_poms:
        raise CheckPolicyError(
            "only single-module Maven repositories are supported; nested pom.xml: "
            + ", ".join(nested_poms[:5])
        )
    if has_maven:
        return _maven_profile(root_pom)
    if has_python:
        return _python_profile(root)
    raise CheckPolicyError("no supported Python pytest or root Maven profile was detected")


def _python_profile(root: Path) -> CheckProfile:
    bootstrap: list[tuple[str, ...]] = []
    requirements = root / "requirements.txt"
    has_requirements = False
    if requirements.is_file():
        requirements_text = _read_manifest_text(requirements, "requirements.txt")
        has_requirements = any(
            line.strip() and not line.lstrip().startswith("#")
            for line in requirements_text.splitlines()
        )
    has_project_dependencies = False
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            document = tomllib.loads(_read_manifest_text(pyproject, "pyproject.toml"))
        except tomllib.TOMLDecodeError as exc:
            raise CheckPolicyError(f"invalid pyproject.toml: {exc}") from exc
        project = document.get("project", {})
        has_project_dependencies = bool(
            isinstance(project, dict) and project.get("dependencies")
        )

    if has_requirements or has_project_dependencies:
        install = [
            "python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-compile",
            "--target",
            "/dependencies/python",
        ]
        if has_requirements:
            install.extend(("--requirement", "requirements.txt"))
        if has_project_dependencies:
            install.append(".")
        bootstrap.append(tuple(install))
    manifest = next(
        (name for name in ("pyproject.toml", "pytest.ini", "setup.cfg", "requirements.txt") if (root / name).is_file()),
        "tests/",
    )
    return CheckProfile(
        id="python-pytest",
        language="python",
        image="repo-agent-python:0.1",
        manifest=manifest,
        bootstrap_argv=tuple(bootstrap),
        check_argv=("python", "-m", "pytest", "-q", "-p", "no:cacheprovider"),
    )


def _maven_profile(pom: Path) -> CheckProfile:
    project_root = pom.parent
    maven_config = project_root / ".mvn"
    if maven_config.exists():
        raise CheckPolicyError(
            "project-level .mvn configuration, including maven.config, is not "
            "supported; use the fixed Maven profile"
        )
    raw = _read_manifest_bytes(pom, "pom.xml")
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise CheckPolicyError("pom.xml must not contain DTD or entity declarations")
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise CheckPolicyError(f"invalid pom.xml: {exc}") from exc
    namespace = _maven_namespace(root)
    properties = _maven_properties(root, namespace)
    _validate_maven_contract(root, namespace)
    modules = root.find(f"{namespace}modules")
    if modules is not None and list(modules):
        raise CheckPolicyError("multi-module Maven projects are not supported")

    release_value = _maven_release(root, namespace, properties)
    match = re.fullmatch(r"(?:1\.)?([0-9]+)", release_value)
    if match is None or int(match.group(1)) not in _ALLOWED_JAVA_RELEASES:
        raise CheckPolicyError("Maven Java release must be 17 or 21")
    release = int(match.group(1))
    surefire_version = _maven_plugin_version(
        root,
        namespace,
        properties,
        "maven-surefire-plugin",
    )
    platform_version = _SUREFIRE_PLATFORM_VERSIONS.get(surefire_version)
    if platform_version is None:
        raise CheckPolicyError(
            "Maven maven-surefire-plugin version is not supported; expected 3.5.2"
        )
    junit_version = _maven_dependency_version(
        root,
        namespace,
        properties,
        "org.junit.jupiter",
        "junit-jupiter",
    )
    project_platform_version = _JUNIT_PLATFORM_VERSIONS.get(junit_version)
    if project_platform_version is None:
        raise CheckPolicyError(
            "Maven JUnit Jupiter version is not supported; expected 5.11.4"
        )
    common = (
        "mvn",
        "--batch-mode",
        "--no-transfer-progress",
        "-Dstyle.color=never",
        "-Dmaven.repo.local=/dependencies/m2",
    )
    return CheckProfile(
        id="maven-test",
        language="java",
        image="repo-agent-maven:0.1",
        manifest="pom.xml",
        bootstrap_argv=(
            common
            + (
                "-DskipTests",
                "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:go-offline",
            ),
            common
            + (
                "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
                "-Dartifact="
                f"org.apache.maven.surefire:surefire-junit-platform:{surefire_version}",
                "-Dtransitive=true",
            ),
            common
            + (
                "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
                "-Dartifact="
                f"org.junit.platform:junit-platform-launcher:{platform_version}",
                "-Dtransitive=true",
            ),
            common
            + (
                "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:get",
                "-Dartifact="
                f"org.junit.platform:junit-platform-launcher:{project_platform_version}",
                "-Dtransitive=true",
            ),
        ),
        check_argv=common
        + (
            "--offline",
            "-DskipTests=false",
            "-Dmaven.test.skip=false",
            "-Dmaven.test.failure.ignore=false",
            "-DfailIfNoTests=true",
            "process-test-classes",
            "org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test",
        ),
        java_release=release,
    )


def _maven_namespace(root: ElementTree.Element) -> str:
    namespace, local_name = _split_xml_tag(root.tag)
    if local_name != "project" or namespace not in {"", _MAVEN_XML_NAMESPACE}:
        raise CheckPolicyError(
            "pom.xml must use the Maven 4.0.0 project namespace"
        )
    unsupported_root_attributes = set(root.attrib) - {
        _MAVEN_SCHEMA_LOCATION_ATTRIBUTE
    }
    if unsupported_root_attributes:
        raise CheckPolicyError("pom.xml project attributes are not supported")
    for element in root.iter():
        child_namespace, _ = _split_xml_tag(element.tag)
        if child_namespace != namespace:
            raise CheckPolicyError("pom.xml contains an unsupported XML namespace")
        if element is not root and element.attrib:
            raise CheckPolicyError(
                "pom.xml child elements must not contain XML attributes"
            )
    return f"{{{namespace}}}" if namespace else ""


def _split_xml_tag(tag: object) -> tuple[str, str]:
    if not isinstance(tag, str) or not tag:
        raise CheckPolicyError("pom.xml contains an invalid XML element")
    if not tag.startswith("{"):
        return "", tag
    namespace, separator, local_name = tag[1:].partition("}")
    if not separator or not namespace or not local_name:
        raise CheckPolicyError("pom.xml contains an invalid XML element")
    return namespace, local_name


def _maven_direct_children(
    parent: ElementTree.Element, namespace: str, name: str
) -> list[ElementTree.Element]:
    return [child for child in parent if child.tag == f"{namespace}{name}"]


def _maven_scalar(
    parent: ElementTree.Element,
    namespace: str,
    name: str,
    *,
    required: bool = False,
) -> str:
    matches = _maven_direct_children(parent, namespace, name)
    if len(matches) > 1:
        raise CheckPolicyError(f"Maven {name} must not be declared more than once")
    if not matches:
        if required:
            raise CheckPolicyError(f"Maven {name} must be explicitly declared")
        return ""
    node = matches[0]
    if list(node):
        raise CheckPolicyError(f"Maven {name} must be plain text")
    value = (node.text or "").strip()
    if required and not value:
        raise CheckPolicyError(f"Maven {name} must be explicitly declared")
    return value


def _maven_properties(
    root: ElementTree.Element, namespace: str
) -> dict[str, str]:
    containers = _maven_direct_children(root, namespace, "properties")
    if len(containers) > 1:
        raise CheckPolicyError("Maven properties must not be declared more than once")
    if not containers:
        return {}
    properties: dict[str, str] = {}
    for child in containers[0]:
        _, key = _split_xml_tag(child.tag)
        if key not in _MAVEN_ALLOWED_PROPERTIES:
            raise CheckPolicyError(f"Maven property {key!r} is not supported")
        if key in properties:
            raise CheckPolicyError(f"Maven property {key!r} is duplicated")
        if list(child) or child.attrib:
            raise CheckPolicyError(f"Maven property {key!r} must be plain text")
        value = (child.text or "").strip()
        if not value:
            raise CheckPolicyError(f"Maven property {key!r} must not be empty")
        properties[key] = value
    encoding = properties.get("project.build.sourceEncoding")
    if encoding is not None and encoding.casefold().replace("_", "-") != "utf-8":
        raise CheckPolicyError("Maven source encoding must be UTF-8")
    if properties.get("junit.version", "5.11.4") != "5.11.4":
        raise CheckPolicyError("Maven junit.version must be 5.11.4")
    if properties.get("surefire.version", "3.5.2") != "3.5.2":
        raise CheckPolicyError("Maven surefire.version must be 3.5.2")
    return properties


def _validate_maven_contract(
    root: ElementTree.Element, namespace: str
) -> None:
    forbidden = {
        "parent": "Maven parent POMs are not supported",
        "repositories": "Maven repositories are not supported",
        "pluginRepositories": "Maven pluginRepositories are not supported",
        "profiles": "Maven profiles are not supported",
        "extensions": "Maven build extensions are not supported",
        "dependencyManagement": "Maven dependencyManagement is not supported",
        "pluginManagement": "Maven pluginManagement is not supported",
        "modules": "multi-module Maven projects are not supported",
    }
    for element in root.iter():
        _, name = _split_xml_tag(element.tag)
        if name in forbidden:
            raise CheckPolicyError(forbidden[name])

    _validate_maven_children(
        root,
        namespace,
        {
            "modelVersion",
            "groupId",
            "artifactId",
            "version",
            "packaging",
            "properties",
            "dependencies",
            "build",
        },
        "project",
    )
    if _maven_scalar(root, namespace, "modelVersion", required=True) != "4.0.0":
        raise CheckPolicyError("Maven modelVersion must be 4.0.0")
    _maven_scalar(root, namespace, "groupId", required=True)
    _maven_scalar(root, namespace, "artifactId", required=True)
    _maven_scalar(root, namespace, "version", required=True)
    packaging_nodes = _maven_direct_children(root, namespace, "packaging")
    packaging = _maven_scalar(root, namespace, "packaging")
    if packaging_nodes and packaging != "jar":
        raise CheckPolicyError("Maven packaging must be jar or omitted")

    dependency_sections = _maven_direct_children(root, namespace, "dependencies")
    if len(dependency_sections) != 1:
        raise CheckPolicyError(
            "Maven projects must declare one fixed JUnit Jupiter dependency"
        )
    _validate_maven_children(
        dependency_sections[0], namespace, {"dependency"}, "dependencies"
    )
    dependencies = _maven_direct_children(
        dependency_sections[0], namespace, "dependency"
    )
    all_dependencies = [
        element
        for element in root.iter()
        if element.tag == f"{namespace}dependency"
    ]
    if len(dependencies) != 1 or all_dependencies != dependencies:
        raise CheckPolicyError(
            "Maven projects may only depend on JUnit Jupiter 5.11.4"
        )
    dependency = dependencies[0]
    _validate_maven_children(
        dependency,
        namespace,
        {"groupId", "artifactId", "version", "scope"},
        "JUnit Jupiter dependency",
    )
    if (
        _maven_scalar(dependency, namespace, "groupId", required=True)
        != "org.junit.jupiter"
        or _maven_scalar(dependency, namespace, "artifactId", required=True)
        != "junit-jupiter"
    ):
        raise CheckPolicyError(
            "Maven projects may only depend on JUnit Jupiter 5.11.4"
        )
    if _maven_scalar(dependency, namespace, "scope", required=True) != "test":
        raise CheckPolicyError("Maven JUnit Jupiter dependency must use test scope")
    _maven_scalar(dependency, namespace, "version")

    builds = _maven_direct_children(root, namespace, "build")
    if len(builds) != 1:
        raise CheckPolicyError("Maven projects must declare one fixed build section")
    build = builds[0]
    _validate_maven_children(build, namespace, {"plugins"}, "build")
    plugin_sections = _maven_direct_children(build, namespace, "plugins")
    if len(plugin_sections) != 1:
        raise CheckPolicyError("Maven build must declare one plugins section")
    _validate_maven_children(
        plugin_sections[0], namespace, {"plugin"}, "plugins"
    )
    plugins = _maven_direct_children(plugin_sections[0], namespace, "plugin")
    all_plugins = [
        element for element in root.iter() if element.tag == f"{namespace}plugin"
    ]
    if all_plugins != plugins:
        raise CheckPolicyError("Maven plugins must be direct build plugins")

    seen: set[str] = set()
    for plugin in plugins:
        group_id = _maven_scalar(plugin, namespace, "groupId")
        if group_id not in {"", "org.apache.maven.plugins"}:
            raise CheckPolicyError(
                "Maven may only use the fixed Surefire and optional compiler plugins"
            )
        artifact_id = _maven_scalar(
            plugin, namespace, "artifactId", required=True
        )
        if artifact_id in seen:
            raise CheckPolicyError(f"Maven plugin {artifact_id} is duplicated")
        seen.add(artifact_id)
        if artifact_id == "maven-surefire-plugin":
            _validate_maven_surefire_plugin(plugin, namespace)
        elif artifact_id == "maven-compiler-plugin":
            _validate_maven_compiler_plugin(plugin, namespace)
        else:
            raise CheckPolicyError(
                "Maven may only use the fixed Surefire and optional compiler plugins"
            )
    if "maven-surefire-plugin" not in seen:
        raise CheckPolicyError("Maven must declare the fixed Surefire plugin")


def _validate_maven_children(
    parent: ElementTree.Element,
    namespace: str,
    allowed: set[str],
    context: str,
) -> None:
    for child in parent:
        _, name = _split_xml_tag(child.tag)
        if child.tag != f"{namespace}{name}" or name not in allowed:
            raise CheckPolicyError(
                f"Maven {context} element {name!r} is not supported"
            )


def _validate_maven_surefire_plugin(
    plugin: ElementTree.Element, namespace: str
) -> None:
    _validate_maven_children(
        plugin,
        namespace,
        {"groupId", "artifactId", "version", "configuration"},
        "Surefire plugin",
    )
    _maven_scalar(plugin, namespace, "version")
    configurations = _maven_direct_children(plugin, namespace, "configuration")
    if len(configurations) > 1:
        raise CheckPolicyError("Maven Surefire configuration is ambiguous")
    if not configurations:
        return
    configuration = configurations[0]
    _validate_maven_children(
        configuration,
        namespace,
        {"useModulePath"},
        "Surefire configuration",
    )
    value = _maven_scalar(
        configuration, namespace, "useModulePath", required=True
    )
    if value.casefold() != "false":
        raise CheckPolicyError(
            "Maven Surefire useModulePath must be false"
        )


def _validate_maven_compiler_plugin(
    plugin: ElementTree.Element, namespace: str
) -> None:
    _validate_maven_children(
        plugin,
        namespace,
        {"groupId", "artifactId", "version", "configuration"},
        "compiler plugin",
    )
    if (
        _maven_scalar(plugin, namespace, "version", required=True)
        != _MAVEN_COMPILER_PLUGIN_VERSION
    ):
        raise CheckPolicyError(
            "Maven compiler plugin version must be 3.13.0"
        )
    configurations = _maven_direct_children(plugin, namespace, "configuration")
    if len(configurations) != 1:
        raise CheckPolicyError(
            "Maven compiler plugin must configure release"
        )
    configuration = configurations[0]
    _validate_maven_children(
        configuration,
        namespace,
        {"release"},
        "compiler configuration",
    )
    release = _maven_scalar(configuration, namespace, "release")
    if not release:
        raise CheckPolicyError(
            "Maven compiler plugin must configure release"
        )


def _resolve_maven_property(value: str, properties: dict[str, str]) -> str:
    resolved = value.strip()
    visited: set[str] = set()
    for _ in range(8):
        match = re.fullmatch(r"\$\{([^{}]+)\}", resolved)
        if match is None:
            return resolved
        key = match.group(1)
        if key in visited:
            return ""
        visited.add(key)
        resolved = properties.get(key, "").strip()
    return ""


def _maven_release(
    root: ElementTree.Element,
    namespace: str,
    properties: dict[str, str],
) -> str:
    configured: list[str] = []
    for plugin in _maven_project_plugins(root, namespace):
        artifact = plugin.find(f"{namespace}artifactId")
        group = plugin.find(f"{namespace}groupId")
        if artifact is None or (artifact.text or "").strip() != "maven-compiler-plugin":
            continue
        if group is not None and (group.text or "").strip() not in {
            "",
            "org.apache.maven.plugins",
        }:
            continue
        configuration = plugin.find(f"{namespace}configuration")
        if configuration is None:
            continue
        node = configuration.find(f"{namespace}release")
        if node is not None and (node.text or "").strip():
            configured.append(
                _resolve_maven_property((node.text or "").strip(), properties)
            )

    if configured:
        distinct = {value for value in configured if value}
        if len(distinct) != 1 or len(distinct) != len(set(configured)):
            raise CheckPolicyError("Maven compiler release configuration is ambiguous")
        return configured[0]

    declared = properties.get("maven.compiler.release")
    if not declared:
        raise CheckPolicyError("Maven Java release must be explicitly declared as 17 or 21")
    return _resolve_maven_property(declared, properties)


def _maven_plugin_version(
    root: ElementTree.Element,
    namespace: str,
    properties: dict[str, str],
    artifact_id: str,
) -> str:
    versions: set[str] = set()
    for plugin in _maven_project_plugins(root, namespace):
        artifact = plugin.find(f"{namespace}artifactId")
        group = plugin.find(f"{namespace}groupId")
        if artifact is None or (artifact.text or "").strip() != artifact_id:
            continue
        if group is not None and (group.text or "").strip() not in {
            "",
            "org.apache.maven.plugins",
        }:
            continue
        version = plugin.find(f"{namespace}version")
        resolved = _resolve_maven_property(
            (version.text or "").strip() if version is not None else "",
            properties,
        )
        if resolved:
            versions.add(resolved)
    if len(versions) != 1:
        raise CheckPolicyError(
            f"Maven {artifact_id} must declare one explicit version"
        )
    resolved = versions.pop()
    if not re.fullmatch(r"[0-9][A-Za-z0-9_.-]{0,63}", resolved):
        raise CheckPolicyError(f"Maven {artifact_id} version is invalid")
    return resolved


def _maven_dependency_version(
    root: ElementTree.Element,
    namespace: str,
    properties: dict[str, str],
    group_id: str,
    artifact_id: str,
) -> str:
    versions: set[str] = set()
    dependencies = root.find(f"{namespace}dependencies")
    if dependencies is None:
        dependencies_to_check: list[ElementTree.Element] = []
    else:
        dependencies_to_check = _maven_direct_children(
            dependencies, namespace, "dependency"
        )
    for dependency in dependencies_to_check:
        group = dependency.find(f"{namespace}groupId")
        artifact = dependency.find(f"{namespace}artifactId")
        if (
            group is None
            or artifact is None
            or (group.text or "").strip() != group_id
            or (artifact.text or "").strip() != artifact_id
        ):
            continue
        version = dependency.find(f"{namespace}version")
        resolved = _resolve_maven_property(
            (version.text or "").strip() if version is not None else "",
            properties,
        )
        if resolved:
            versions.add(resolved)
    if len(versions) != 1:
        raise CheckPolicyError(
            f"Maven {group_id}:{artifact_id} must declare one explicit version"
        )
    resolved = versions.pop()
    if not re.fullmatch(r"[0-9][A-Za-z0-9_.-]{0,63}", resolved):
        raise CheckPolicyError(f"Maven {group_id}:{artifact_id} version is invalid")
    return resolved


def _maven_project_plugins(
    root: ElementTree.Element, namespace: str
) -> list[ElementTree.Element]:
    build = root.find(f"{namespace}build")
    if build is None:
        return []
    plugins = build.find(f"{namespace}plugins")
    if plugins is None:
        return []
    return _maven_direct_children(plugins, namespace, "plugin")


class CheckRunner:
    """Run bootstrap and verification in separate, exactly cleaned containers."""

    def __init__(
        self,
        repo_path: str | os.PathLike[str],
        *,
        candidate_patch: str | None = None,
        allow_bootstrap: bool = False,
        phase_timeout_seconds: float = MAX_PHASE_TIMEOUT_SECONDS,
        total_timeout_seconds: float = MAX_RUN_TIMEOUT_SECONDS,
        control_timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_CHECK_OUTPUT_BYTES,
        command_runner: CheckCommandRunnerLike | None = None,
        container_name_prefix: str = "repo-agent-check",
        temp_parent: str | os.PathLike[str] | None = None,
    ) -> None:
        self.repo_path = _validated_repository(repo_path)
        if type(allow_bootstrap) is not bool:
            raise ValueError("allow_bootstrap must be a boolean")
        self.allow_bootstrap = allow_bootstrap
        self.phase_timeout_seconds = _bounded_timeout(
            phase_timeout_seconds, "phase_timeout_seconds", MAX_PHASE_TIMEOUT_SECONDS
        )
        self.total_timeout_seconds = _bounded_timeout(
            total_timeout_seconds, "total_timeout_seconds", MAX_RUN_TIMEOUT_SECONDS
        )
        self.control_timeout_seconds = _bounded_timeout(
            control_timeout_seconds,
            "control_timeout_seconds",
            _CONTROL_TIMEOUT_SECONDS,
        )
        if type(max_output_bytes) is not int or not 1024 <= max_output_bytes <= MAX_CHECK_OUTPUT_BYTES:
            raise ValueError(
                f"max_output_bytes must be an integer from 1024 to {MAX_CHECK_OUTPUT_BYTES}"
            )
        self.max_output_bytes = max_output_bytes
        if (
            not isinstance(container_name_prefix, str)
            or len(container_name_prefix) > 48
            or not _CONTAINER_NAME_RE.fullmatch(container_name_prefix)
        ):
            raise ValueError("invalid Docker container_name_prefix")
        self.container_name_prefix = container_name_prefix
        self._candidate: ValidatedPatch | None = None
        if candidate_patch is not None:
            self._candidate = validate_patch(candidate_patch)
        self._command_runner = command_runner or SubprocessCommandRunner()
        if temp_parent is None:
            self._temp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        else:
            self._temp_parent = Path(temp_parent).resolve(strict=True)
        if not self._temp_parent.is_dir():
            raise ValueError("temp_parent must be a directory")
        self._run_id = uuid.uuid4().hex
        self._sequence = 0
        self._container_ids: dict[str, str] = {}
        self._container_names: dict[str, int] = {}
        self._indeterminate_container_names: set[str] = set()
        self._cleanup_deadline: float | None = None
        self._root: Path | None = None
        self._closed = False

    def run(self, check_id: str | None = None) -> CheckRunResult:
        """Run the detected profile once and always clean every run-owned resource."""

        started = time.monotonic()
        result: CheckRunResult
        try:
            result = self._run_inner(check_id, started)
        except CheckPolicyError as exc:
            result = CheckRunResult(
                status="policy_denied",
                profile=None,
                base_commit=None,
                candidate_applied=self._candidate is not None,
                phases=(),
                duration_ms=_elapsed_ms(started),
                error=str(exc),
            )
        except CheckTimeoutError as exc:
            result = CheckRunResult(
                status="timed_out",
                profile=None,
                base_commit=None,
                candidate_applied=self._candidate is not None,
                phases=(),
                duration_ms=_elapsed_ms(started),
                error=str(exc),
            )
        except (OSError, CheckError, subprocess.SubprocessError) as exc:
            result = CheckRunResult(
                status="setup_error",
                profile=None,
                base_commit=None,
                candidate_applied=self._candidate is not None,
                phases=(),
                duration_ms=_elapsed_ms(started),
                error=f"{type(exc).__name__}: {exc}",
            )
        except BaseException as exc:
            try:
                self.close()
            except CheckCleanupError as cleanup_exc:
                exc.add_note(f"check cleanup also failed: {cleanup_exc}")
            raise

        try:
            self.close()
        except CheckCleanupError as exc:
            return replace(
                result,
                status="cleanup_error",
                duration_ms=_elapsed_ms(started),
                error=f"{result.error + '; ' if result.error else ''}{exc}",
                cleanup_ok=False,
            )
        return replace(result, duration_ms=_elapsed_ms(started))

    def _run_inner(self, check_id: str | None, started: float) -> CheckRunResult:
        self._prepare_root()
        self._require_remaining_time(started)
        assert self._root is not None
        base = self._root / "base"
        base_commit = self._create_base_snapshot(base, started)
        self._require_remaining_time(started)
        profile_source = (
            self._fresh_copy(base, "profile", started)
            if self._candidate is not None
            else base
        )
        self._require_remaining_time(started)
        profile = detect_check_profile(profile_source)
        self._require_remaining_time(started)
        if check_id is not None and check_id != profile.id:
            raise CheckPolicyError(
                f"requested check {check_id!r} does not match detected profile {profile.id!r}"
            )
        if profile.bootstrap_required and not self.allow_bootstrap:
            return CheckRunResult(
                status="bootstrap_required",
                profile=profile,
                base_commit=base_commit,
                candidate_applied=self._candidate is not None,
                phases=(),
                duration_ms=_elapsed_ms(started),
                error="dependency bootstrap requires explicit allow_bootstrap=True",
            )

        dependency_cache = self._root / "dependencies"
        dependency_cache.mkdir(mode=0o700)
        _make_container_writable(
            dependency_cache,
            check_deadline=lambda: self._require_remaining_time(started),
        )
        self._require_remaining_time(started)
        phases: list[CheckPhaseResult] = []
        if profile.bootstrap_required:
            bootstrap_workspace = self._fresh_copy(base, "bootstrap", started)
            self._assert_profile(bootstrap_workspace, profile)
            self._require_remaining_time(started)
            for index, argv in enumerate(profile.bootstrap_argv, start=1):
                attempt_phases: list[CheckPhaseResult] = []
                max_attempts = (
                    _MAVEN_BOOTSTRAP_ATTEMPTS if profile.language == "java" else 1
                )
                for attempt in range(max_attempts):
                    timeout = self._remaining_timeout(started)
                    if timeout <= 0:
                        if attempt_phases:
                            phases.append(
                                _merge_phase_attempts(
                                    attempt_phases, self.max_output_bytes
                                )
                            )
                        return self._timeout_result(
                            profile, base_commit, phases, started
                        )
                    attempt_phase = self._run_phase(
                        profile,
                        name=f"bootstrap-{index}",
                        kind="bootstrap",
                        network="bridge",
                        argv=argv,
                        workspace=bootstrap_workspace,
                        dependencies=dependency_cache,
                        timeout_seconds=timeout,
                    )
                    attempt_phases.append(attempt_phase)
                    if attempt_phase.status == "passed":
                        break
                    if (
                        attempt + 1 == max_attempts
                        or not _is_retryable_maven_bootstrap_phase(attempt_phase)
                    ):
                        break
                phase = _merge_phase_attempts(attempt_phases, self.max_output_bytes)
                phases.append(phase)
                if phase.status != "passed":
                    return self._phase_failure_result(
                        profile, base_commit, phases, started, phase
                    )

        verify_workspace = self._fresh_copy(base, "verify", started)
        self._assert_profile(verify_workspace, profile)
        self._require_remaining_time(started)
        timeout = self._remaining_timeout(started)
        if timeout <= 0:
            return self._timeout_result(profile, base_commit, phases, started)
        verify = self._run_phase(
            profile,
            name=profile.id,
            kind="verify",
            network="none",
            argv=profile.check_argv,
            workspace=verify_workspace,
            dependencies=dependency_cache,
            timeout_seconds=timeout,
        )
        phases.append(verify)
        if self._remaining_total_timeout(started) <= 0:
            return self._timeout_result(profile, base_commit, phases, started)
        if verify.status != "passed":
            return self._phase_failure_result(
                profile, base_commit, phases, started, verify
            )
        return CheckRunResult(
            status="passed",
            profile=profile,
            base_commit=base_commit,
            candidate_applied=self._candidate is not None,
            phases=tuple(phases),
            duration_ms=_elapsed_ms(started),
        )

    def _phase_failure_result(
        self,
        profile: CheckProfile,
        base_commit: str,
        phases: list[CheckPhaseResult],
        started: float,
        phase: CheckPhaseResult,
    ) -> CheckRunResult:
        status: CheckStatus = "timed_out" if phase.status == "timed_out" else "failed"
        if phase.status == "setup_error":
            status = "setup_error"
        return CheckRunResult(
            status=status,
            profile=profile,
            base_commit=base_commit,
            candidate_applied=self._candidate is not None,
            phases=tuple(phases),
            duration_ms=_elapsed_ms(started),
            error=phase.error,
        )

    def _timeout_result(
        self,
        profile: CheckProfile,
        base_commit: str,
        phases: list[CheckPhaseResult],
        started: float,
    ) -> CheckRunResult:
        return CheckRunResult(
            status="timed_out",
            profile=profile,
            base_commit=base_commit,
            candidate_applied=self._candidate is not None,
            phases=tuple(phases),
            duration_ms=_elapsed_ms(started),
            error=f"check run exceeded {self.total_timeout_seconds:g} seconds",
        )

    def _remaining_timeout(self, started: float) -> float:
        return min(self.phase_timeout_seconds, self._remaining_total_timeout(started))

    def _remaining_total_timeout(self, started: float) -> float:
        remaining = self.total_timeout_seconds - (time.monotonic() - started)
        return max(0.0, remaining)

    def _host_timeout(self, started: float) -> float:
        remaining = self._remaining_total_timeout(started)
        if remaining <= 0:
            self._require_remaining_time(started)
        return min(self.control_timeout_seconds, remaining)

    def _require_remaining_time(self, started: float) -> None:
        if self._remaining_timeout(started) <= 0:
            raise CheckTimeoutError(
                f"check run exceeded {self.total_timeout_seconds:g} seconds"
            )

    @staticmethod
    def _assert_profile(workspace: Path, expected: CheckProfile) -> None:
        actual = detect_check_profile(workspace)
        if actual != expected:
            raise CheckPolicyError(
                "check profile changed between candidate preparation and execution"
            )

    def _prepare_root(self) -> None:
        if self._root is not None or self._closed:
            raise CheckError("a CheckRunner instance can only run once")
        created = Path(
            tempfile.mkdtemp(prefix="repo-agent-check-", dir=self._temp_parent)
        ).absolute()
        resolved = created.resolve(strict=True)
        if resolved.parent != self._temp_parent or not _CHECK_ROOT_RE.fullmatch(resolved.name):
            raise CheckError(f"refusing unexpected check root: {resolved}")
        if _is_link_or_reparse(resolved):
            raise CheckError("check root must not be a link or reparse point")
        self._root = resolved

    def _create_base_snapshot(self, destination: Path, started: float) -> str:
        try:
            autocrlf = effective_core_autocrlf(
                self.repo_path,
                timeout_seconds=self._host_timeout(started),
            )
        except RuntimeError as exc:
            raise CheckError(str(exc)) from exc
        self._require_remaining_time(started)
        status = _git_bytes(
            self.repo_path,
            "-c",
            f"core.autocrlf={autocrlf}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        if status:
            raise CheckPolicyError("check repository must have a clean worktree")
        self._validate_git_inputs(self.repo_path, started)
        head = _git_text(
            self.repo_path,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        assert self._root is not None
        empty_template = self._root / "empty-git-template"
        empty_hooks = self._root / "empty-git-hooks"
        empty_template.mkdir(mode=0o700)
        empty_hooks.mkdir(mode=0o700)
        _run_git(
            "-c",
            "core.autocrlf=false",
            "-c",
            f"core.hooksPath={empty_hooks}",
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--depth",
            "1",
            "--no-tags",
            "--single-branch",
            f"--template={empty_template}",
            str(self.repo_path),
            str(destination),
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        if _git_text(
            destination,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            timeout_seconds=self._host_timeout(started),
        ) != head:
            raise CheckError("repository HEAD changed while creating check snapshot")
        self._require_remaining_time(started)
        _run_git(
            "-C",
            str(destination),
            "remote",
            "remove",
            "origin",
            timeout_seconds=self._host_timeout(started),
        )
        _run_git(
            "-c",
            f"core.hooksPath={empty_hooks}",
            "-c",
            "core.autocrlf=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--force",
            "--detach",
            head,
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        _reject_links_and_sensitive_files(
            destination,
            check_deadline=lambda: self._require_remaining_time(started),
        )
        self._require_remaining_time(started)
        return head

    def _validate_git_inputs(self, repository: Path, started: float) -> None:
        records = _git_bytes(
            repository,
            "ls-files",
            "--stage",
            "-z",
            timeout_seconds=self._host_timeout(started),
        ).split(b"\0")
        tracked_files = 0
        tracked_bytes = 0
        for record in records:
            self._require_remaining_time(started)
            if not record:
                continue
            tracked_files += 1
            if tracked_files > _MAX_REPOSITORY_FILES:
                raise CheckPolicyError(
                    f"check repository exceeds {_MAX_REPOSITORY_FILES} tracked files"
                )
            header, separator, raw_path = record.partition(b"\t")
            if not separator:
                raise CheckPolicyError("Git index returned malformed path metadata")
            mode = header.split(maxsplit=1)[0]
            try:
                path = raw_path.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise CheckPolicyError(
                    "tracked repository paths must be valid UTF-8"
                ) from exc
            if mode == b"120000":
                raise CheckPolicyError(f"symbolic links are not supported: {path}")
            if mode == b"160000":
                raise CheckPolicyError(f"Git submodules are not supported: {path}")
            try:
                DEFAULT_PATH_POLICY.validate(path, access="discover")
            except RepositoryPathError as exc:
                raise CheckPolicyError(f"unsafe tracked path {path!r}: {exc}") from exc
            try:
                tracked_bytes += (repository / path).stat().st_size
            except OSError as exc:
                raise CheckPolicyError(f"cannot inspect tracked path {path!r}: {exc}") from exc
            if tracked_bytes > _MAX_REPOSITORY_BYTES:
                raise CheckPolicyError("check repository exceeds 512 MiB of tracked files")
        attributes = repository / ".gitattributes"
        if attributes.is_file():
            content = attributes.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?im)(?:^|\s)filter\s*=\s*lfs(?:\s|$)", content):
                raise CheckPolicyError("Git LFS repositories are not supported")
        self._require_remaining_time(started)

    def _fresh_copy(self, base: Path, name: str, started: float) -> Path:
        assert self._root is not None
        destination = self._root / name
        empty_template = self._root / "empty-git-template"
        empty_hooks = self._root / "empty-git-hooks"
        head = _git_text(
            base,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            timeout_seconds=self._host_timeout(started),
        )
        _run_git(
            "-c",
            "core.autocrlf=false",
            "-c",
            f"core.hooksPath={empty_hooks}",
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--depth",
            "1",
            "--no-tags",
            f"--template={empty_template}",
            str(base),
            str(destination),
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        _run_git(
            "-C",
            str(destination),
            "remote",
            "remove",
            "origin",
            timeout_seconds=self._host_timeout(started),
        )
        _run_git(
            "-c",
            f"core.hooksPath={empty_hooks}",
            "-c",
            "core.autocrlf=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--force",
            "--detach",
            head,
            timeout_seconds=self._host_timeout(started),
        )
        self._require_remaining_time(started)
        _reject_links_and_sensitive_files(
            destination,
            check_deadline=lambda: self._require_remaining_time(started),
        )
        if self._candidate is not None:
            _apply_candidate(
                destination,
                self._candidate,
                timeout_seconds=self._host_timeout(started),
            )
        self._require_remaining_time(started)
        _make_container_writable(
            destination,
            check_deadline=lambda: self._require_remaining_time(started),
        )
        self._require_remaining_time(started)
        return destination

    def _run_phase(
        self,
        profile: CheckProfile,
        *,
        name: str,
        kind: PhaseKind,
        network: str,
        argv: tuple[str, ...],
        workspace: Path,
        dependencies: Path,
        timeout_seconds: float,
    ) -> CheckPhaseResult:
        phase_started = time.monotonic()
        phase_deadline = phase_started + timeout_seconds
        self._sequence += 1
        container_name = f"{self.container_name_prefix}-{self._run_id}-{self._sequence}"
        if not _CONTAINER_NAME_RE.fullmatch(container_name):
            raise CheckError("generated Docker container name is invalid")
        self._container_names[container_name] = self._sequence
        create_argv = self._container_create_argv(
            profile,
            container_name,
            network,
            argv,
            workspace,
            dependencies,
        )
        container_id: str | None = None
        try:
            create = self._run_command(
                create_argv,
                timeout_seconds=min(
                    _CONTROL_TIMEOUT_SECONDS,
                    self._deadline_remaining(phase_deadline),
                ),
            )
            if create.timed_out:
                self._indeterminate_container_names.add(container_name)
                self._cleanup_deadline = (
                    time.monotonic() + self.control_timeout_seconds
                )
                cleanup_error = self._recover_container_name(
                    container_name,
                    retry=True,
                    deadline=self._cleanup_deadline,
                )
                return _phase_result(
                    name,
                    kind,
                    "setup_error",
                    network,
                    argv,
                    create,
                    phase_started,
                    _join_error("Docker create timed out", cleanup_error),
                )
            container_id = _container_id(create.output)
            if create.exit_code != 0 or container_id is None:
                cleanup_error = self._recover_container_name(container_name)
                detail: str | None = (
                    f"Docker create exited with code {create.exit_code}"
                    if create.exit_code != 0
                    else "Docker create did not return one container ID"
                )
                return _phase_result(
                    name,
                    kind,
                    "setup_error",
                    network,
                    argv,
                    create,
                    phase_started,
                    _join_error(detail, cleanup_error),
                )
            self._container_ids[container_id] = container_name
            start_timeout = self._deadline_remaining(phase_deadline)
            if start_timeout <= 0:
                cleanup_error = self._remove_container(container_id)
                synthetic = CommandOutcome(None, "", timed_out=True)
                return _phase_result(
                    name,
                    kind,
                    "timed_out",
                    network,
                    argv,
                    synthetic,
                    phase_started,
                    _join_error(
                        f"{kind} phase timed out after {timeout_seconds:g} seconds",
                        cleanup_error,
                    ),
                )
            execution = self._run_command(
                ("docker", "container", "start", "--attach", container_id),
                timeout_seconds=start_timeout,
            )
            cleanup_error = self._remove_container(container_id)
            if execution.timed_out:
                return _phase_result(
                    name,
                    kind,
                    "timed_out",
                    network,
                    argv,
                    execution,
                    phase_started,
                    _join_error(
                        f"{kind} phase timed out after {timeout_seconds:g} seconds",
                        cleanup_error,
                    ),
                )
            if execution.exit_code != 0 or cleanup_error:
                detail = (
                    f"{kind} phase exited with code {execution.exit_code}"
                    if execution.exit_code != 0
                    else None
                )
                return _phase_result(
                    name,
                    kind,
                    "failed",
                    network,
                    argv,
                    execution,
                    phase_started,
                    _join_error(detail, cleanup_error),
                )
            return _phase_result(
                name,
                kind,
                "passed",
                network,
                argv,
                execution,
                phase_started,
                None,
            )
        except (OSError, CheckError) as exc:
            cleanup_error = (
                self._remove_container(container_id)
                if container_id is not None
                else self._recover_container_name(container_name)
            )
            synthetic = CommandOutcome(None, "")
            return _phase_result(
                name,
                kind,
                "setup_error",
                network,
                argv,
                synthetic,
                phase_started,
                _join_error(f"{type(exc).__name__}: {exc}", cleanup_error),
            )

    @staticmethod
    def _deadline_remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _container_create_argv(
        self,
        profile: CheckProfile,
        name: str,
        network: str,
        argv: tuple[str, ...],
        workspace: Path,
        dependencies: Path,
    ) -> tuple[str, ...]:
        workspace_source = str(workspace.resolve(strict=True))
        dependency_source = str(dependencies.resolve(strict=True))
        if "," in workspace_source or "," in dependency_source:
            raise CheckPolicyError("Docker bind source paths containing commas are unsupported")
        policy = DEFAULT_POLICY
        args: list[str] = [
            "docker",
            "container",
            "create",
            "--name",
            name,
            "--label",
            f"{_RUN_LABEL}={self._run_id}",
            "--label",
            f"{_SEQUENCE_LABEL}={self._sequence}",
            "--init",
            "--pull",
            policy.pull,
            "--restart",
            policy.restart,
            "--network",
            network,
            "--read-only",
            "--user",
            policy.user,
            "--cpus",
            policy.cpus,
            "--memory",
            policy.memory,
            "--memory-swap",
            policy.memory_swap,
            "--pids-limit",
            str(policy.pids_limit),
        ]
        for capability in policy.cap_drop:
            args.extend(("--cap-drop", capability))
        for option in policy.security_opt:
            args.extend(("--security-opt", option))
        args.extend(
            (
                "--log-driver",
                policy.log_driver,
                "--tmpfs",
                policy.tmpfs,
                "--mount",
                f"type=bind,source={workspace_source},target=/workspace",
                "--mount",
                f"type=bind,source={dependency_source},target={profile.cache_target}",
                "--workdir",
                "/workspace",
                "--env",
                "HOME=/tmp/home",
                "--env",
                "CI=1",
                "--env",
                "NO_COLOR=1",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
            )
        )
        if profile.language == "python" and profile.bootstrap_required:
            args.extend(("--env", "PYTHONPATH=/dependencies/python"))
        args.extend(("--entrypoint", argv[0], profile.image, *argv[1:]))
        return tuple(args)

    def _run_command(
        self, argv: Sequence[str], *, timeout_seconds: float
    ) -> CommandOutcome:
        outcome = self._command_runner(
            tuple(argv),
            timeout_seconds=max(0.001, timeout_seconds),
            max_output_bytes=self.max_output_bytes,
        )
        if not isinstance(outcome, CommandOutcome):
            raise CheckError("command_runner must return CommandOutcome")
        output, truncated = _bounded_output(outcome.output, self.max_output_bytes)
        return CommandOutcome(
            exit_code=outcome.exit_code,
            output=output,
            truncated=outcome.truncated or truncated,
            timed_out=outcome.timed_out,
        )

    def _recover_container_name(
        self,
        name: str,
        retry: bool = False,
        *,
        deadline: float | None = None,
    ) -> str | None:
        recovery_deadline = deadline or (
            time.monotonic() + self.control_timeout_seconds
        )
        delay = 0.0
        while True:
            remaining = self._deadline_remaining(recovery_deadline)
            if remaining <= 0:
                break
            if delay:
                time.sleep(min(delay, remaining))
                if self._deadline_remaining(recovery_deadline) <= 0:
                    break
            try:
                container_id, absent, error = self._inspect_owned_container(
                    name, recovery_deadline
                )
            except (OSError, CheckError) as exc:
                return f"exact container recovery failed: {type(exc).__name__}: {exc}"
            if error:
                return error
            if absent:
                if retry:
                    delay = (
                        _RECOVERY_INITIAL_DELAY_SECONDS
                        if delay == 0
                        else min(_RECOVERY_MAX_DELAY_SECONDS, delay * 2)
                    )
                    continue
                if name in self._indeterminate_container_names:
                    return (
                        "timed-out Docker create remains indeterminate for "
                        f"container {name!r}"
                    )
                self._container_names.pop(name, None)
                return None
            if container_id is not None:
                self._container_ids[container_id] = name
                return self._remove_container(
                    container_id, deadline=recovery_deadline
                )
        return f"Docker create remains indeterminate for container {name!r}"

    def _inspect_owned_container(
        self, name: str, deadline: float
    ) -> tuple[str | None, bool, str | None]:
        sequence = self._container_names.get(name)
        if sequence is None:
            return None, False, f"container name is not owned by this run: {name!r}"
        format_value = (
            f'{_INSPECT_PREFIX} {{{{.Id}}}} '
            f'{{{{index .Config.Labels "{_RUN_LABEL}"}}}} '
            f'{{{{index .Config.Labels "{_SEQUENCE_LABEL}"}}}}'
        )
        inspected = self._run_command(
            (
                "docker",
                "container",
                "inspect",
                "--format",
                format_value,
                name,
            ),
            timeout_seconds=self._deadline_remaining(deadline),
        )
        if inspected.timed_out:
            return None, False, "exact container recovery timed out"
        if inspected.exit_code != 0:
            if "no such container" in inspected.output.casefold():
                return None, True, None
            return None, False, "exact container recovery failed"
        pattern = re.compile(
            rf"(?m)^[ \t]*{re.escape(_INSPECT_PREFIX)}[ \t]+"
            r"([0-9a-f]{64})[ \t]+([0-9a-f]{32})[ \t]+([0-9]+)[ \t]*$"
        )
        records = pattern.findall(inspected.output)
        if len(records) != 1:
            return None, False, "exact container recovery returned invalid metadata"
        container_id, run_id, found_sequence = records[0]
        if run_id != self._run_id or found_sequence != str(sequence):
            return None, False, "refusing container with mismatched ownership labels"
        return container_id, False, None

    def _remove_container(
        self, container_id: str | None, *, deadline: float | None = None
    ) -> str | None:
        if container_id is None or not _CONTAINER_ID_RE.fullmatch(container_id):
            return f"refusing invalid container ID: {container_id!r}"
        cleanup_deadline = deadline or (
            time.monotonic() + self.control_timeout_seconds
        )
        remaining = self._deadline_remaining(cleanup_deadline)
        if remaining <= 0:
            return f"container {container_id} cleanup deadline expired"
        name = self._container_ids.get(container_id)
        try:
            removed = self._run_command(
                ("docker", "container", "rm", "--force", container_id),
                timeout_seconds=remaining,
            )
        except (OSError, CheckError) as exc:
            return f"container cleanup failed: {type(exc).__name__}: {exc}"
        if removed.timed_out:
            return f"container {container_id} cleanup timed out"
        if removed.exit_code != 0 and "no such container" not in removed.output.casefold():
            return f"container {container_id} cleanup exited with code {removed.exit_code}"
        self._container_ids.pop(container_id, None)
        if name is not None:
            self._container_names.pop(name, None)
            self._indeterminate_container_names.discard(name)
        return None

    def close(self) -> None:
        """Remove only containers and the temporary root owned by this runner."""

        if self._closed:
            return
        errors: list[str] = []
        cleanup_deadline = self._cleanup_deadline or (
            time.monotonic() + self.control_timeout_seconds
        )
        for container_id in tuple(self._container_ids):
            error = self._remove_container(container_id, deadline=cleanup_deadline)
            if error:
                errors.append(error)
        for name in tuple(self._container_names):
            try:
                error = self._recover_container_name(
                    name, retry=True, deadline=cleanup_deadline
                )
            except (OSError, CheckError) as exc:
                error = f"exact container recovery failed: {type(exc).__name__}: {exc}"
            if error:
                errors.append(error)
        if not self._container_ids and not self._container_names and self._root is not None:
            root = self._root
            if (
                root.parent != self._temp_parent
                or not _CHECK_ROOT_RE.fullmatch(root.name)
                or _is_link_or_reparse(root)
            ):
                errors.append(f"refusing to remove unexpected check root: {root}")
            else:
                try:
                    shutil.rmtree(root, onerror=_remove_readonly)
                    self._root = None
                except OSError as exc:
                    errors.append(f"check root cleanup failed: {exc}")
        if self._container_ids or self._container_names:
            errors.append("run-owned check containers remain")
        if errors:
            raise CheckCleanupError("; ".join(errors))
        self._closed = True


def _visible_files(
    root: Path, *, check_deadline: Callable[[], None] | None = None
) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if check_deadline is not None:
            check_deadline()
        relative = path.relative_to(root)
        if ".git" in relative.parts or "target" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_file() or path.is_symlink():
            files.append(path)
    return tuple(files)


def _read_manifest_bytes(path: Path, display_name: str) -> bytes:
    try:
        size = path.stat().st_size
        if size > _MAX_MANIFEST_BYTES:
            raise CheckPolicyError(f"{display_name} exceeds 256 KiB")
        return path.read_bytes()
    except CheckPolicyError:
        raise
    except OSError as exc:
        raise CheckPolicyError(f"cannot read {display_name}: {exc}") from exc


def _read_manifest_text(path: Path, display_name: str) -> str:
    try:
        return _read_manifest_bytes(path, display_name).decode("utf-8")
    except UnicodeError as exc:
        raise CheckPolicyError(f"{display_name} must be UTF-8 text") from exc


def _make_container_writable(
    root: Path, *, check_deadline: Callable[[], None] | None = None
) -> None:
    """Grant the fixed container UID write access inside this disposable tree."""

    for path in (root, *root.rglob("*")):
        if check_deadline is not None:
            check_deadline()
        if _is_link_or_reparse(path):
            raise CheckPolicyError(
                f"symbolic links or reparse points are unsupported: {path}"
            )
        try:
            current_mode = stat.S_IMODE(os.lstat(path).st_mode)
            writable_mode = current_mode | (0o777 if path.is_dir() else 0o666)
            os.chmod(path, writable_mode)
        except OSError as exc:
            raise CheckError(
                f"cannot prepare disposable check path {path}: {exc}"
            ) from exc


def _reject_links_and_sensitive_files(
    root: Path, *, check_deadline: Callable[[], None] | None = None
) -> None:
    for path in root.rglob("*"):
        if check_deadline is not None:
            check_deadline()
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if _is_link_or_reparse(path):
            raise CheckPolicyError(
                f"symbolic links or reparse points are unsupported: {relative.as_posix()}"
            )
        if path.is_file():
            try:
                DEFAULT_PATH_POLICY.validate(relative.as_posix(), access="discover")
            except RepositoryPathError as exc:
                raise CheckPolicyError(
                    f"unsafe repository path {relative.as_posix()!r}: {exc}"
                ) from exc


def _is_link_or_reparse(path: Path) -> bool:
    try:
        path_stat = os.lstat(path)
    except OSError:
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(path_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(path_stat.st_mode) or bool(reparse and attributes & reparse)


def _apply_candidate(
    repository: Path,
    patch: ValidatedPatch,
    *,
    timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
) -> None:
    payload = patch.text.encode("utf-8")
    for arguments in (
        ("apply", "--check", "--index", "--whitespace=error-all", "--"),
        ("apply", "--index", "--whitespace=error-all", "--"),
    ):
        try:
            completed = subprocess.run(
                ("git", "-C", str(repository), *arguments),
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_sanitized_git_environment(),
                shell=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CheckTimeoutError(
                f"candidate Git apply timed out after {timeout_seconds:g} seconds"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise CheckPolicyError(
                f"candidate patch did not apply to fresh verification copy: {detail}"
            )


def _git_text(
    repository: Path,
    *args: str,
    timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
) -> str:
    return _git_bytes(
        repository, *args, timeout_seconds=timeout_seconds
    ).decode("utf-8", errors="strict").strip()


def _git_bytes(
    repository: Path,
    *args: str,
    timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sanitized_git_environment(),
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckTimeoutError(
            f"Git operation timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CheckError(detail or f"Git exited with code {completed.returncode}")
    return completed.stdout


def _run_git(
    *args: str, timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS
) -> None:
    try:
        completed = subprocess.run(
            ("git", *args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_sanitized_git_environment(),
            shell=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckTimeoutError(
            f"Git operation timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CheckError(detail or f"Git exited with code {completed.returncode}")


def _bounded_timeout(value: object, field: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or not 0 < converted <= maximum:
        raise ValueError(f"{field} must be at most {maximum:g} seconds")
    return converted


def _container_id(output: str) -> str | None:
    matches = re.findall(r"(?m)^[ \t]*([0-9a-f]{64})[ \t]*$", output)
    return matches[0] if len(matches) == 1 else None


def _bounded_output(output: str, limit: int) -> tuple[str, bool]:
    if not isinstance(output, str):
        raise CheckError("command_runner output must be text")
    encoded = output.encode("utf-8")
    if len(encoded) <= limit:
        return output, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _is_retryable_maven_bootstrap_phase(phase: CheckPhaseResult) -> bool:
    expected_error = f"bootstrap phase exited with code {phase.exit_code}"
    return (
        phase.status == "failed"
        and phase.exit_code not in {None, 0}
        and phase.error == expected_error
        and _is_retryable_maven_transfer(phase.output)
    )


def _is_retryable_maven_transfer(output: str) -> bool:
    truncated = _MAVEN_TRUNCATED_TRANSFER_RE.search(output)
    if truncated is not None:
        expected = int(truncated.group(1).replace(",", ""))
        received = int(truncated.group(2).replace(",", ""))
        if received < expected:
            return True

    normalized = output.casefold()
    if not any(marker in normalized for marker in _MAVEN_TRANSFER_CONTEXT):
        return False
    return (
        any(marker in normalized for marker in _MAVEN_TRANSIENT_TRANSFER_MARKERS)
        or _MAVEN_RETRYABLE_HTTP_STATUS_RE.search(output) is not None
    )


def _merge_phase_attempts(
    attempts: Sequence[CheckPhaseResult], output_limit: int
) -> CheckPhaseResult:
    if not attempts:
        raise CheckError("bootstrap phase produced no attempt result")
    if len(attempts) == 1:
        return attempts[0]
    final = attempts[-1]
    output, combined_truncated = _bounded_output(
        "\n".join(attempt.output for attempt in attempts), output_limit
    )
    return replace(
        final,
        output=output,
        duration_ms=sum(attempt.duration_ms for attempt in attempts),
        truncated=combined_truncated or any(attempt.truncated for attempt in attempts),
    )


def _phase_result(
    name: str,
    kind: PhaseKind,
    status: PhaseStatus,
    network: str,
    argv: tuple[str, ...],
    outcome: CommandOutcome,
    started: float,
    error: str | None,
) -> CheckPhaseResult:
    return CheckPhaseResult(
        name=name,
        kind=kind,
        status=status,
        network=network,
        argv=argv,
        exit_code=outcome.exit_code,
        output=outcome.output,
        error=error,
        duration_ms=_elapsed_ms(started),
        truncated=outcome.truncated,
    )


def _join_error(primary: str | None, cleanup: str | None) -> str | None:
    if primary and cleanup:
        return f"{primary}; {cleanup}"
    return primary or cleanup


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _remove_readonly(function, path: str, _exc_info) -> None:
    target = Path(path)
    if _is_link_or_reparse(target):
        raise CheckCleanupError(f"refusing to chmod link or reparse point: {target}")
    os.chmod(path, stat.S_IWRITE)
    if not callable(function):
        raise TypeError("rmtree error handler received a non-callable operation")
    function(path)


__all__ = [
    "CheckCleanupError",
    "CheckError",
    "CheckPhaseResult",
    "CheckPolicyError",
    "CheckProfile",
    "CheckRunResult",
    "CheckRunner",
    "DEFAULT_CHECK_OUTPUT_BYTES",
    "MAX_CHECK_OUTPUT_BYTES",
    "MAX_PHASE_TIMEOUT_SECONDS",
    "MAX_RUN_TIMEOUT_SECONDS",
    "detect_check_profile",
]
