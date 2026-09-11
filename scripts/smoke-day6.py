"""Run the Day 6 offline Docker E2E and isolation acceptance probes.

This script is deliberately independent of model configuration.  It never
pulls images, enables container networking, bootstraps dependencies, or passes
host environment variables to a container.  The final stdout value is one JSON
object so the probe can be used by a person or by CI.
"""

from __future__ import annotations

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
from typing import Iterable, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_PARENT = PROJECT_ROOT / ".tmp"
LOCK_PATH = PROJECT_ROOT / "docker" / "image-lock.json"
PYTHON_IMAGE = "repo-agent-python:0.1"
MAVEN_IMAGE = "repo-agent-maven:0.1"

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_RUNTIME_ERROR = 2
EXIT_CLEANUP_FAILED = 3

RUN_LABEL_KEY = "io.github.ablecat.repo-agent.day6.run"
CASE_LABEL_KEY = "io.github.ablecat.repo-agent.day6.case"
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
TEMP_DIRECTORY_RE = re.compile(r"^day6-smoke-[0-9A-Za-z_]+$")
CONTROL_TIMEOUT_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 15.0
TOOL_TIMEOUT_SECONDS = 0.75
MAX_OUTPUT_BYTES = 64 * 1024

sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from repo_agent.sandbox import (  # noqa: E402
    CommandOutcome,
    SubprocessCommandRunner,
)
from repo_agent.artifacts import ArtifactStore  # noqa: E402
from repo_agent.patches import PatchValidationError, validate_patch  # noqa: E402
from repo_agent.policy import (  # noqa: E402
    DEFAULT_PATH_POLICY,
    RepositoryPathError,
)


class AcceptanceError(RuntimeError):
    """A Docker precondition or acceptance assertion failed."""


class CheckFailed(AcceptanceError):
    """A recorded security or E2E assertion failed."""


class DockerCli:
    """Small checked wrapper around the injectable bounded process runner."""

    def __init__(self, command_runner=None) -> None:
        self._runner = command_runner or SubprocessCommandRunner()

    def invoke(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = CONTROL_TIMEOUT_SECONDS,
        allowed_exit_codes: Iterable[int] = (0,),
        allow_timeout: bool = False,
    ) -> CommandOutcome:
        command = ("docker", *tuple(args))
        outcome = self._runner(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_OUTPUT_BYTES,
        )
        if outcome.timed_out and not allow_timeout:
            raise AcceptanceError(
                f"Docker command timed out after {timeout_seconds:g} seconds"
            )
        allowed = frozenset(allowed_exit_codes)
        if not outcome.timed_out and outcome.exit_code not in allowed:
            detail = _bounded_detail(outcome.output)
            suffix = f": {detail}" if detail else ""
            raise AcceptanceError(
                f"Docker command exited with code {outcome.exit_code}{suffix}"
            )
        return outcome


class ContainerRegistry:
    """Track only containers owned by this probe and remove them by exact ID."""

    def __init__(self, docker: DockerCli, run_id: str) -> None:
        self.docker = docker
        self.run_id = run_id
        self._ids: list[str] = []

    @property
    def label(self) -> str:
        return f"{RUN_LABEL_KEY}={self.run_id}"

    def register(self, container_id: str) -> None:
        validated = _parse_container_id(container_id)
        if validated not in self._ids:
            self._ids.append(validated)

    def remove(self, container_id: str) -> None:
        validated = _parse_container_id(container_id)
        outcome = self.docker.invoke(
            ("container", "rm", "--force", validated),
            allowed_exit_codes=(0, 1),
        )
        if outcome.exit_code == 1 and "no such container" not in outcome.output.casefold():
            raise AcceptanceError(
                "exact container cleanup failed: " + _bounded_detail(outcome.output)
            )
        self._ids = [known for known in self._ids if known != validated]

    def cleanup(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        for container_id in reversed(tuple(self._ids)):
            try:
                self.remove(container_id)
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        # Recover a container whose create output was lost, but retain the exact
        # ID boundary before issuing any removal command.
        try:
            for container_id in self._labeled_ids():
                try:
                    self.remove(container_id)
                except BaseException as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            leftovers = self._labeled_ids()
            if leftovers:
                errors.append(
                    f"{len(leftovers)} run-labeled container(s) remain after cleanup"
                )
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        return not errors, errors

    def _labeled_ids(self) -> tuple[str, ...]:
        outcome = self.docker.invoke(
            (
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={self.label}",
            )
        )
        lines = tuple(line.strip() for line in outcome.output.splitlines() if line.strip())
        return tuple(_parse_container_id(line) for line in lines)


def main() -> int:
    run_id = uuid.uuid4().hex
    docker = DockerCli()
    containers = ContainerRegistry(docker, run_id)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "failed",
        "exit_code": EXIT_RUNTIME_ERROR,
        "offline": True,
        "images": {},
        "checks": [],
        "e2e": {},
        "security": {
            "probes": [
                {"id": "path_traversal", "ok": False},
                {"id": "link_escape", "ok": False},
                {"id": "credential_canary", "ok": False},
                {"id": "patch_boundary", "ok": False},
                {"id": "network_isolation", "ok": False},
                {"id": "resource_timeout_cleanup", "ok": False},
            ],
            "passed": "0/6",
            "host_mutations": None,
            "canary_leaks": None,
        },
        "cleanup": {
            "containers_removed": False,
            "temporary_workspace_removed": False,
        },
        "error": None,
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]
    temp_root: Path | None = None
    runtime_failed = False
    check_failed = False
    host_canary: Path | None = None
    canary_digest: str | None = None

    try:
        _verify_engine(docker, checks)
        image_records = _verify_images(docker, checks)
        report["images"] = image_records

        TEMP_PARENT.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix="day6-smoke-", dir=TEMP_PARENT)
        ).resolve(strict=True)
        python_workspace = temp_root / "python-workspace"
        maven_workspace = temp_root / "maven-workspace"
        host_canary = temp_root / "host-boundary-canary.txt"
        canary_value = f"day6-host-only-{uuid.uuid4().hex}"
        host_canary.write_text(canary_value, encoding="utf-8", newline="")
        canary_digest = _sha256_file(host_canary)
        _create_python_fixture(python_workspace)
        _create_maven_fixture(maven_workspace)
        artifact_leaks = _run_local_security_probes(
            temp_root,
            python_workspace,
            checks,
            report,
        )
        security = report["security"]
        assert isinstance(security, dict)
        security["canary_leaks"] = artifact_leaks
        _prepare_container_writable_workspace(temp_root, python_workspace)
        _prepare_container_writable_workspace(temp_root, maven_workspace)

        python_id = _create_container(
            docker,
            containers,
            name=f"repo-agent-day6-python-{run_id}",
            case="python-e2e",
            image=PYTHON_IMAGE,
            workspace=python_workspace,
        )
        maven_id = _create_container(
            docker,
            containers,
            name=f"repo-agent-day6-maven-{run_id}",
            case="maven-e2e",
            image=MAVEN_IMAGE,
            workspace=maven_workspace,
        )
        _start_container(docker, python_id)
        _start_container(docker, maven_id)

        _verify_container_contract(
            docker, checks, "python", python_id, python_workspace
        )
        _verify_container_contract(
            docker, checks, "maven", maven_id, maven_workspace
        )
        _verify_runtime_isolation(
            docker, checks, "python", python_id, python_workspace
        )
        _verify_runtime_isolation(
            docker, checks, "maven", maven_id, maven_workspace
        )
        _verify_network_isolation(docker, checks, python_id, maven_id)
        _pass_security_probe(
            report,
            "network_isolation",
            "DNS and direct outbound TCP were blocked in both offline sandboxes",
        )

        report_e2e = report["e2e"]
        assert isinstance(report_e2e, dict)
        report_e2e["python"] = _run_python_e2e(docker, checks, python_id)
        report_e2e["maven"] = _run_maven_e2e(docker, checks, maven_id)

        timeout_target = _create_container(
            docker,
            containers,
            name=f"repo-agent-day6-timeout-{run_id}",
            case="timeout-target",
            image=PYTHON_IMAGE,
            workspace=python_workspace,
        )
        timeout_control = _create_container(
            docker,
            containers,
            name=f"repo-agent-day6-control-{run_id}",
            case="timeout-control",
            image=PYTHON_IMAGE,
            workspace=python_workspace,
        )
        _start_container(docker, timeout_target)
        _start_container(docker, timeout_control)
        _verify_precise_timeout_cleanup(
            docker,
            containers,
            checks,
            timeout_target,
            timeout_control,
        )
        _pass_security_probe(
            report,
            "resource_timeout_cleanup",
            "resource limits were inspected and the timed-out container was removed exactly",
        )
        host_mutations, host_leaks = _host_canary_metrics(
            host_canary,
            canary_digest,
            canary_value,
            (python_workspace, maven_workspace),
        )
        security["host_mutations"] = host_mutations
        security["canary_leaks"] = int(security["canary_leaks"]) + host_leaks
        _require(
            checks,
            "host_boundary.canary_isolated",
            host_mutations == 0 and host_leaks == 0,
            "the unmounted host canary is unchanged and absent from both workspaces",
        )
    except CheckFailed as exc:
        check_failed = True
        report["error"] = {
            "type": type(exc).__name__,
            "message": _bounded_detail(str(exc)),
        }
    except BaseException as exc:
        runtime_failed = True
        report["error"] = {
            "type": type(exc).__name__,
            "message": _bounded_detail(str(exc)),
        }
    finally:
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        containers_removed, container_errors = containers.cleanup()
        cleanup["containers_removed"] = containers_removed
        if container_errors:
            cleanup["container_errors"] = container_errors
        _record(
            checks,
            "cleanup.no_residual_containers",
            containers_removed,
            "no run-labeled Day 6 containers remain",
        )

        temporary_removed = True
        temporary_error: str | None = None
        if temp_root is not None and temp_root.exists():
            try:
                _remove_temp_root(temp_root)
            except BaseException as exc:
                temporary_removed = False
                temporary_error = f"{type(exc).__name__}: {exc}"
        if temp_root is not None and temp_root.exists():
            temporary_removed = False
        cleanup["temporary_workspace_removed"] = temporary_removed
        if temporary_error is not None:
            cleanup["temporary_workspace_error"] = temporary_error
        _record(
            checks,
            "cleanup.temporary_workspace_removed",
            temporary_removed,
            "run-specific fixture directory was removed",
        )

    all_checks_passed = all(bool(check["ok"]) for check in checks)
    security = report["security"]
    assert isinstance(security, dict)
    probes = security["probes"]
    assert isinstance(probes, list)
    security["passed"] = f"{sum(bool(probe.get('ok')) for probe in probes)}/6"
    cleanup = report["cleanup"]
    assert isinstance(cleanup, dict)
    cleanup_failed = not (
        bool(cleanup["containers_removed"])
        and bool(cleanup["temporary_workspace_removed"])
    )
    if cleanup_failed:
        exit_code = EXIT_CLEANUP_FAILED
    elif runtime_failed:
        exit_code = EXIT_RUNTIME_ERROR
    elif check_failed or not all_checks_passed:
        exit_code = EXIT_CHECK_FAILED
    else:
        exit_code = EXIT_OK
        report["status"] = "passed"
    report["exit_code"] = exit_code
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _run_local_security_probes(
    temp_root: Path,
    repository: Path,
    checks: list[dict[str, object]],
    report: dict[str, object],
) -> int:
    traversal_inputs = (
        "../outside.txt",
        "src/../../outside.txt",
        "/etc/passwd",
        "C:/outside.txt",
    )
    traversal_blocked = all(
        _path_rejected(value, repository) for value in traversal_inputs
    )
    _require(
        checks,
        "security.path_traversal_blocked",
        traversal_blocked,
        "relative traversal and POSIX/Windows absolute paths are rejected",
    )
    _pass_security_probe(
        report,
        "path_traversal",
        "../ and absolute path attempts were rejected before filesystem access",
    )

    escape_target = temp_root / "link-escape-target"
    escape_target.mkdir()
    (escape_target / "outside.txt").write_text("outside", encoding="utf-8")
    escape_link = repository / "escape-link"
    _create_directory_link(escape_link, escape_target)
    try:
        link_blocked = _path_rejected("escape-link/outside.txt", repository)
    finally:
        _remove_directory_link(escape_link)
    _require(
        checks,
        "security.link_escape_blocked",
        link_blocked,
        "symbolic-link or reparse-point traversal is rejected",
    )
    _pass_security_probe(
        report,
        "link_escape",
        "a real link/reparse escape was rejected without following its target",
    )

    artifact_canary = f"day6-artifact-canary-{uuid.uuid4().hex}"
    sensitive_paths = (repository / ".env", repository / "credentials.json")
    sensitive_paths[0].write_text(f"API_KEY={artifact_canary}\n", encoding="utf-8")
    sensitive_paths[1].write_text(artifact_canary, encoding="utf-8")
    try:
        reads_blocked = all(
            _path_rejected(path.name, repository) for path in sensitive_paths
        )
        artifact_root = temp_root / "redaction-artifacts"
        artifacts = ArtifactStore(artifact_root, secrets=(artifact_canary,))
        artifact_run_id = "d" * 32
        artifacts.initialize(artifact_run_id)
        artifacts.write_report(artifact_run_id, f"diagnostic={artifact_canary}")
        artifacts.append_trace(
            artifact_run_id,
            {"event": "probe", "value": artifact_canary},
        )
        marker = artifact_canary.encode("utf-8")
        artifact_leaks = sum(
            marker in path.read_bytes()
            for path in artifact_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    finally:
        for path in sensitive_paths:
            path.unlink(missing_ok=True)
    _require(
        checks,
        "security.credential_canary_blocked",
        reads_blocked and artifact_leaks == 0,
        ".env and credential reads are denied and configured canaries are redacted",
    )
    _pass_security_probe(
        report,
        "credential_canary",
        "credential paths were denied and artifact canary leaks were zero",
    )

    unsafe_patches = (
        _probe_patch("../outside.txt"),
        _probe_patch("C:/outside.txt"),
        _probe_patch(".git/config"),
        _probe_patch(
            "safe.txt",
            metadata="old mode 100644\nnew mode 100755\n",
        ),
    )
    patches_blocked = all(_patch_rejected(value) for value in unsafe_patches)
    _require(
        checks,
        "security.patch_boundary_blocked",
        patches_blocked,
        "traversal, absolute, .git, and mode-changing patches are rejected",
    )
    _pass_security_probe(
        report,
        "patch_boundary",
        "all four unsafe patch forms were rejected by the structured parser",
    )
    return artifact_leaks


def _path_rejected(relative_path: str, repository: Path) -> bool:
    try:
        DEFAULT_PATH_POLICY.validate_disk_path(repository, relative_path, access="read")
    except RepositoryPathError:
        return True
    return False


def _patch_rejected(patch: str) -> bool:
    try:
        validate_patch(patch)
    except PatchValidationError:
        return True
    return False


def _probe_patch(path: str, *, metadata: str = "") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"{metadata}"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt":
            raise AcceptanceError("could not create the link-escape probe") from exc
    outcome = subprocess.run(
        ("cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=CONTROL_TIMEOUT_SECONDS,
        check=False,
    )
    if outcome.returncode != 0 or not _is_link_or_reparse(link):
        raise AcceptanceError("could not create the reparse-point escape probe")


def _remove_directory_link(link: Path) -> None:
    if not os.path.lexists(link):
        return
    if link.is_symlink():
        link.unlink()
    elif _is_link_or_reparse(link):
        os.rmdir(link)
    else:
        raise AcceptanceError("link-escape probe unexpectedly became a real directory")


def _pass_security_probe(
    report: dict[str, object], probe_id: str, detail: str
) -> None:
    security = report["security"]
    assert isinstance(security, dict)
    probes = security["probes"]
    assert isinstance(probes, list)
    for probe in probes:
        if isinstance(probe, dict) and probe.get("id") == probe_id:
            probe.update({"ok": True, "detail": detail})
            return
    raise AcceptanceError(f"unknown security probe: {probe_id}")


def _verify_engine(docker: DockerCli, checks: list[dict[str, object]]) -> None:
    context = docker.invoke(("context", "show")).output.strip()
    _require(
        checks,
        "engine.context",
        _accepted_context(context),
        f"active Docker context is {context or '<empty>'}",
    )
    outcome = docker.invoke(("version", "--format", "{{json .}}"))
    try:
        payload = json.loads(outcome.output)
    except json.JSONDecodeError as exc:
        raise AcceptanceError("docker version did not return valid JSON") from exc
    server = payload.get("Server") if isinstance(payload, dict) else None
    if not isinstance(server, dict):
        raise AcceptanceError("Docker Server metadata is unavailable")
    _require(
        checks,
        "engine.linux_amd64",
        server.get("Os") == "linux" and server.get("Arch") == "amd64",
        f"Docker Server is {server.get('Os')}/{server.get('Arch')}",
    )


def _verify_images(
    docker: DockerCli, checks: list[dict[str, object]]
) -> dict[str, object]:
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError("docker/image-lock.json is unavailable or invalid") from exc
    locked_images = lock.get("images") if isinstance(lock, dict) else None
    if not isinstance(locked_images, dict):
        raise AcceptanceError("docker/image-lock.json has no images object")

    result: dict[str, object] = {}
    for runtime, image in (("python", PYTHON_IMAGE), ("maven", MAVEN_IMAGE)):
        outcome = docker.invoke(
            ("image", "inspect", "--format", "{{json .}}", image)
        )
        try:
            inspect = json.loads(outcome.output)
        except json.JSONDecodeError as exc:
            raise AcceptanceError(f"{image} inspect output is invalid") from exc
        image_id = inspect.get("Id") if isinstance(inspect, dict) else None
        locked = locked_images.get(runtime)
        locked_base = locked.get("base") if isinstance(locked, dict) else None
        locked_id = locked.get("imageId") if isinstance(locked, dict) else None
        _require(
            checks,
            f"image.{runtime}.available",
            isinstance(image_id, str) and _is_sha256_reference(image_id),
            f"{image} is available locally without pulling",
        )
        dockerfile_base = _dockerfile_base(PROJECT_ROOT / "docker" / runtime / "Dockerfile")
        _require(
            checks,
            f"image.{runtime}.base_digest_pinned",
            isinstance(locked_base, str)
            and _is_digest_pinned_image(locked_base)
            and dockerfile_base == locked_base,
            f"{runtime} Dockerfile FROM matches the locked base-image digest",
        )
        result[runtime] = {
            "tag": image,
            "image_id": image_id,
            "base": locked_base,
            "matches_recorded_local_image_id": image_id == locked_id,
        }
    return result


def _create_container(
    docker: DockerCli,
    containers: ContainerRegistry,
    *,
    name: str,
    case: str,
    image: str,
    workspace: Path,
) -> str:
    args = _build_create_args(
        name=name,
        run_label=containers.label,
        case=case,
        image=image,
        workspace=workspace,
    )
    outcome = docker.invoke(args)
    container_id = _parse_container_id(outcome.output)
    containers.register(container_id)
    return container_id


def _build_create_args(
    *,
    name: str,
    run_label: str,
    case: str,
    image: str,
    workspace: Path,
) -> tuple[str, ...]:
    source = str(workspace.resolve(strict=True))
    if "," in source:
        raise AcceptanceError("Docker bind paths containing commas are unsupported")
    return (
        "container",
        "create",
        "--name",
        name,
        "--label",
        run_label,
        "--label",
        f"{CASE_LABEL_KEY}={case}",
        "--init",
        "--pull",
        "never",
        "--restart",
        "no",
        "--network",
        "none",
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
        "--mount",
        f"type=bind,source={source},target=/workspace",
        "--workdir",
        "/workspace",
        image,
        "sleep",
        "300",
    )


def _start_container(docker: DockerCli, container_id: str) -> None:
    docker.invoke(("container", "start", _parse_container_id(container_id)))
    running = docker.invoke(
        (
            "container",
            "inspect",
            "--format",
            "{{.State.Running}}",
            container_id,
        )
    ).output.strip()
    if running != "true":
        raise AcceptanceError("created probe container did not remain running")


def _verify_container_contract(
    docker: DockerCli,
    checks: list[dict[str, object]],
    runtime: str,
    container_id: str,
    workspace: Path,
) -> None:
    outcome = docker.invoke(
        ("container", "inspect", "--format", "{{json .}}", container_id)
    )
    try:
        payload = json.loads(outcome.output)
    except json.JSONDecodeError as exc:
        raise AcceptanceError("docker container inspect returned invalid JSON") from exc
    contract = _inspect_contract(payload, workspace)
    failed = False
    for name, ok, detail in contract:
        _record(checks, f"{runtime}.inspect.{name}", ok, detail)
        failed = failed or not ok
    if failed:
        raise CheckFailed(f"{runtime} Docker inspect contract failed")


def _inspect_contract(
    payload: object, workspace: Path
) -> tuple[tuple[str, bool, str], ...]:
    if not isinstance(payload, dict):
        return (("valid_json_object", False, "inspect payload is not an object"),)
    config = payload.get("Config")
    host = payload.get("HostConfig")
    mounts = payload.get("Mounts")
    if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(mounts, list):
        return (("required_sections", False, "inspect sections are missing"),)

    cap_drop = host.get("CapDrop")
    security_opt = host.get("SecurityOpt")
    restart = host.get("RestartPolicy")
    log_config = host.get("LogConfig")
    tmpfs = host.get("Tmpfs")
    workspace_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == "/workspace"
    ]
    sole_workspace_mount = (
        len(mounts) == 1
        and len(workspace_mounts) == 1
        and workspace_mounts[0].get("Type") == "bind"
        and workspace_mounts[0].get("RW") is True
        and _same_path(workspace_mounts[0].get("Source"), workspace)
    )
    tmpfs_options = tmpfs.get("/tmp") if isinstance(tmpfs, dict) else None
    environment = config.get("Env")
    environment_names = {
        str(value).split("=", 1)[0].upper()
        for value in environment
        if isinstance(environment, list) and isinstance(value, str)
    }
    secret_prefixes = ("OPENAI_", "REPO_AGENT_", "ANTHROPIC_", "AZURE_OPENAI_")
    no_model_environment = not any(
        name.startswith(secret_prefixes) for name in environment_names
    )

    return (
        (
            "non_root_user",
            config.get("User") == "10001:10001",
            "configured user is 10001:10001",
        ),
        (
            "network_none",
            host.get("NetworkMode") == "none",
            "network mode is none",
        ),
        (
            "read_only_rootfs",
            host.get("ReadonlyRootfs") is True,
            "root filesystem is read-only",
        ),
        (
            "cpu_limit",
            host.get("NanoCpus") == 2_000_000_000,
            "CPU quota is 2 cores",
        ),
        (
            "memory_limit",
            host.get("Memory") == 4 * 1024**3
            and host.get("MemorySwap") == 4 * 1024**3,
            "memory and memory+swap are both 4 GiB",
        ),
        (
            "pid_limit",
            host.get("PidsLimit") == 256,
            "PID limit is 256",
        ),
        (
            "capabilities_dropped",
            isinstance(cap_drop, list)
            and {str(value).upper() for value in cap_drop} == {"ALL"},
            "all Linux capabilities are dropped",
        ),
        (
            "no_new_privileges",
            isinstance(security_opt, list)
            and any(
                str(value).casefold().startswith("no-new-privileges")
                for value in security_opt
            ),
            "no-new-privileges is enabled",
        ),
        (
            "lifecycle_hardening",
            host.get("Init") is True
            and isinstance(restart, dict)
            and restart.get("Name") == "no"
            and isinstance(log_config, dict)
            and log_config.get("Type") == "none",
            "init, restart=no, and log-driver=none are active",
        ),
        (
            "workspace_only_bind",
            sole_workspace_mount,
            "the sole bind mount is the writable run-specific workspace",
        ),
        (
            "tmpfs_only_tmp",
            isinstance(tmpfs, dict)
            and set(tmpfs) == {"/tmp"}
            and isinstance(tmpfs_options, str)
            and all(
                option in tmpfs_options
                for option in ("rw", "nosuid", "nodev", "size=256m", "mode=1777")
            ),
            "the sole explicit tmpfs is hardened /tmp",
        ),
        (
            "no_docker_socket",
            all(
                not isinstance(mount, dict)
                or mount.get("Destination") != "/var/run/docker.sock"
                for mount in mounts
            ),
            "Docker socket is absent",
        ),
        (
            "no_model_environment",
            no_model_environment,
            "no model-provider environment variable is present",
        ),
    )


def _verify_runtime_isolation(
    docker: DockerCli,
    checks: list[dict[str, object]],
    runtime: str,
    container_id: str,
    workspace: Path,
) -> None:
    uid = _exec(docker, container_id, ("id", "-u")).output.strip()
    gid = _exec(docker, container_id, ("id", "-g")).output.strip()
    _require(
        checks,
        f"{runtime}.runtime.non_root",
        uid == "10001" and gid == "10001",
        f"runtime UID/GID is {uid}/{gid}",
    )

    process_security = _exec(
        docker,
        container_id,
        (
            "sh",
            "-c",
            "grep -E '^(CapEff|NoNewPrivs):' /proc/self/status",
        ),
    ).output
    fields = {
        key.strip(): value.strip()
        for key, value in (
            line.split(":", 1)
            for line in process_security.splitlines()
            if ":" in line
        )
    }
    _require(
        checks,
        f"{runtime}.runtime.kernel_security",
        fields.get("CapEff") == "0000000000000000"
        and fields.get("NoNewPrivs") == "1",
        "the process has zero effective capabilities and NoNewPrivs=1",
    )

    probe_name = f"day6-{runtime}-write-probe"
    script = (
        "set -eu; "
        f"touch /workspace/{probe_name}; touch /tmp/{probe_name}; "
        "for path in /day6-root-write /bin/day6-write /etc/day6-write "
        "/home/repo-agent/day6-write /opt/day6-write /usr/day6-write /var/day6-write; "
        "do if touch \"$path\" >/dev/null 2>&1; then "
        "echo \"unexpected writable path: $path\"; exit 91; fi; done"
    )
    _exec(docker, container_id, ("sh", "-c", script))
    _require(
        checks,
        f"{runtime}.runtime.write_boundary",
        (workspace / probe_name).is_file(),
        "workspace and /tmp writes succeed while system-root writes fail",
    )


def _verify_network_isolation(
    docker: DockerCli,
    checks: list[dict[str, object]],
    python_id: str,
    maven_id: str,
) -> None:
    interface_results: list[bool] = []
    for container_id in (python_id, maven_id):
        interfaces = _exec(
            docker,
            container_id,
            ("sh", "-c", "ls -1 /sys/class/net | sort"),
        ).output.split()
        interface_results.append(interfaces == ["lo"])
    _require(
        checks,
        "network.loopback_only",
        all(interface_results),
        "both E2E containers expose only the loopback interface",
    )

    dns_code = (
        "import socket,sys; "
        "\ntry: socket.getaddrinfo('example.com', 443)"
        "\nexcept OSError: sys.exit(0)"
        "\nsys.exit(92)"
    )
    outbound_code = (
        "import socket,sys; "
        "\ntry: socket.create_connection(('1.1.1.1', 443), timeout=1)"
        "\nexcept OSError: sys.exit(0)"
        "\nsys.exit(93)"
    )
    dns = _exec(docker, python_id, ("python", "-c", dns_code), allowed=(0,))
    outbound = _exec(
        docker,
        python_id,
        ("python", "-c", outbound_code),
        allowed=(0,),
    )
    _require(
        checks,
        "network.dns_and_outbound_blocked",
        dns.exit_code == 0 and outbound.exit_code == 0,
        "DNS resolution and direct outbound TCP fail under network=none",
    )


def _run_python_e2e(
    docker: DockerCli, checks: list[dict[str, object]], container_id: str
) -> dict[str, object]:
    outcome = _exec(
        docker,
        container_id,
        ("python", "-m", "pytest", "-q", "-p", "no:cacheprovider"),
    )
    _require(
        checks,
        "e2e.python.pytest",
        outcome.exit_code == 0 and "1 passed" in outcome.output,
        "the local Python fixture passes under pytest without network access",
    )
    return {"status": "passed", "output_tail": _bounded_detail(outcome.output, 2048)}


def _run_maven_e2e(
    docker: DockerCli, checks: list[dict[str, object]], container_id: str
) -> dict[str, object]:
    validate = _exec(
        docker,
        container_id,
        (
            "env",
            "MAVEN_CONFIG=/workspace/.m2",
            "MAVEN_OPTS=-Djansi.tmpdir=/workspace/.jansi",
            "mvn",
            "--offline",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dmaven.repo.local=/workspace/.m2/repository",
            "validate",
        ),
    )
    _exec(docker, container_id, ("mkdir", "-p", "target/classes"))
    _exec(
        docker,
        container_id,
        (
            "javac",
            "--release",
            "21",
            "-d",
            "target/classes",
            "src/main/java/dev/repoagent/day6/Smoke.java",
        ),
    )
    execution = _exec(
        docker,
        container_id,
        ("java", "-cp", "target/classes", "dev.repoagent.day6.Smoke"),
    )
    _require(
        checks,
        "e2e.maven.offline_jvm",
        validate.exit_code == 0
        and execution.exit_code == 0
        and execution.output.strip() == "maven-e2e-ok",
        "Maven validates the POM offline and Java 21 compiles/runs the fixture",
    )
    return {
        "status": "passed",
        "maven_output_tail": _bounded_detail(validate.output, 2048),
        "program_output": execution.output.strip(),
    }


def _verify_precise_timeout_cleanup(
    docker: DockerCli,
    containers: ContainerRegistry,
    checks: list[dict[str, object]],
    target_id: str,
    control_id: str,
) -> None:
    timeout = docker.invoke(
        ("container", "exec", target_id, "sleep", "60"),
        timeout_seconds=TOOL_TIMEOUT_SECONDS,
        allow_timeout=True,
    )
    _require(
        checks,
        "timeout.deadline_enforced",
        timeout.timed_out,
        f"long-running command exceeded the {TOOL_TIMEOUT_SECONDS:g}s deadline",
    )
    containers.remove(target_id)
    target = docker.invoke(
        ("container", "inspect", "--format", "{{.State.Running}}", target_id),
        allowed_exit_codes=(0, 1),
    )
    control = docker.invoke(
        ("container", "inspect", "--format", "{{.State.Running}}", control_id)
    )
    _require(
        checks,
        "timeout.exact_container_removed",
        target.exit_code == 1 and control.output.strip() == "true",
        "the timed-out container is gone while the unrelated control stays running",
    )


def _exec(
    docker: DockerCli,
    container_id: str,
    args: Sequence[str],
    *,
    allowed: Iterable[int] = (0,),
) -> CommandOutcome:
    return docker.invoke(
        ("container", "exec", _parse_container_id(container_id), *tuple(args)),
        timeout_seconds=PROBE_TIMEOUT_SECONDS,
        allowed_exit_codes=allowed,
    )


def _create_python_fixture(workspace: Path) -> None:
    (workspace / "tests").mkdir(parents=True)
    (workspace / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n",
        encoding="utf-8",
        newline="",
    )
    (workspace / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\n\n"
        "def test_addition() -> None:\n"
        "    assert add(20, 22) == 42\n",
        encoding="utf-8",
        newline="",
    )


def _create_maven_fixture(workspace: Path) -> None:
    source = workspace / "src" / "main" / "java" / "dev" / "repoagent" / "day6"
    source.mkdir(parents=True)
    (workspace / ".m2" / "repository").mkdir(parents=True)
    (workspace / ".jansi").mkdir()
    (workspace / "pom.xml").write_text(
        """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>dev.repoagent</groupId>
  <artifactId>day6-smoke</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.release>21</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
</project>
""",
        encoding="utf-8",
        newline="",
    )
    (source / "Smoke.java").write_text(
        """\
package dev.repoagent.day6;

public final class Smoke {
    private Smoke() {}

    public static void main(String[] args) {
        if (20 + 22 != 42) {
            throw new IllegalStateException("arithmetic probe failed");
        }
        System.out.println("maven-e2e-ok");
    }
}
""",
        encoding="utf-8",
        newline="",
    )


def _parse_container_id(value: str) -> str:
    matches = [line.strip() for line in str(value).splitlines() if line.strip()]
    ids = [line for line in matches if CONTAINER_ID_RE.fullmatch(line)]
    if len(matches) != 1 or len(ids) != 1:
        raise AcceptanceError("Docker did not return exactly one full container ID")
    return ids[0]


def _same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        actual = os.path.normcase(str(Path(value).resolve(strict=True)))
        wanted = os.path.normcase(str(expected.resolve(strict=True)))
    except (OSError, ValueError):
        return False
    return actual == wanted


def _accepted_context(context: str) -> bool:
    """Docker Desktop is required on Windows; Linux CI context names vary."""

    normalized = str(context).strip()
    return normalized == "desktop-linux" if os.name == "nt" else bool(normalized)


def _dockerfile_base(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.casefold().startswith("from "):
            parts = stripped.split()
            return parts[1] if len(parts) >= 2 else None
    return None


def _is_sha256_reference(value: str) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def _is_digest_pinned_image(value: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_canary_isolated(
    canary_path: Path,
    expected_digest: str,
    canary_value: str,
    workspaces: Sequence[Path],
) -> bool:
    return _host_canary_metrics(
        canary_path, expected_digest, canary_value, workspaces
    ) == (0, 0)


def _host_canary_metrics(
    canary_path: Path,
    expected_digest: str,
    canary_value: str,
    workspaces: Sequence[Path],
) -> tuple[int, int]:
    host_mutations = 0
    canary_leaks = 0
    try:
        if _sha256_file(canary_path) != expected_digest:
            host_mutations = 1
        marker = canary_value.encode("utf-8")
        for workspace in workspaces:
            for path in workspace.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                if marker in path.read_bytes():
                    canary_leaks += 1
    except OSError:
        return (1, max(canary_leaks, 1))
    return host_mutations, canary_leaks


def _prepare_container_writable_workspace(temp_root: Path, workspace: Path) -> None:
    """Grant UID 10001 access only inside a newly created probe workspace."""

    expected_parent = temp_root.resolve(strict=True)
    resolved = workspace.resolve(strict=True)
    if resolved.parent != expected_parent or _is_link_or_reparse(workspace):
        raise AcceptanceError(
            f"container workspace is outside the run-specific temporary root: {workspace}"
        )
    entries = (workspace, *sorted(workspace.rglob("*")))
    for path in entries:
        if _is_link_or_reparse(path):
            raise AcceptanceError(f"link or reparse point in probe workspace: {path}")
        if not path.is_dir() and not path.is_file():
            raise AcceptanceError(f"special file in probe workspace: {path}")
    for path in entries:
        try:
            path.chmod(0o777 if path.is_dir() else 0o666)
        except OSError as exc:
            raise AcceptanceError(
                f"cannot grant sandbox UID access to probe workspace: {path}"
            ) from exc


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise AcceptanceError(f"cannot inspect probe path: {path}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _record(
    checks: list[dict[str, object]], name: str, ok: bool, detail: str
) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _require(
    checks: list[dict[str, object]], name: str, ok: bool, detail: str
) -> None:
    _record(checks, name, ok, detail)
    if not ok:
        raise CheckFailed(f"{name}: {detail}")


def _bounded_detail(value: str, limit: int = 4096) -> str:
    normalized = str(value).replace("\x00", "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "...<truncated>"


def _remove_temp_root(temp_root: Path) -> None:
    resolved = temp_root.resolve(strict=True)
    expected_parent = TEMP_PARENT.resolve(strict=True)
    if resolved.parent != expected_parent or not TEMP_DIRECTORY_RE.fullmatch(resolved.name):
        raise AcceptanceError(f"refusing to remove unexpected temporary path: {resolved}")
    if resolved.is_symlink():
        raise AcceptanceError("refusing to remove a symlinked temporary root")
    shutil.rmtree(resolved, onerror=_remove_readonly)


def _remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    if not callable(function):
        raise TypeError("rmtree error handler received a non-callable operation")
    function(path)


if __name__ == "__main__":
    raise SystemExit(main())
