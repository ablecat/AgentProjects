from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from repo_agent.sandbox import CommandOutcome


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke-day6.py"
SPEC = importlib.util.spec_from_file_location("smoke_day6", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_day6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_day6)


def _option_values(command: tuple[str, ...], option: str) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def _valid_inspect(workspace: Path) -> dict[str, object]:
    return {
        "Config": {
            "User": "10001:10001",
            "Env": ["PATH=/usr/bin", "HOME=/home/repo-agent"],
        },
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "NanoCpus": 2_000_000_000,
            "Memory": 4 * 1024**3,
            "MemorySwap": 4 * 1024**3,
            "PidsLimit": 256,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "Init": True,
            "RestartPolicy": {"Name": "no"},
            "LogConfig": {"Type": "none"},
            "Tmpfs": {
                "/tmp": "rw,nosuid,nodev,size=256m,mode=1777",
            },
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(workspace),
                "Destination": "/workspace",
                "RW": True,
            }
        ],
    }


def test_create_contract_is_offline_hardened_and_has_one_bind(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    command = smoke_day6._build_create_args(
        name="repo-agent-day6-test",
        run_label="io.github.ablecat.repo-agent.day6.run=abc",
        case="python-e2e",
        image=smoke_day6.PYTHON_IMAGE,
        workspace=workspace,
    )

    assert command[:2] == ("container", "create")
    assert "--read-only" in command
    assert "--init" in command
    assert _option_values(command, "--pull") == ["never"]
    assert _option_values(command, "--restart") == ["no"]
    assert _option_values(command, "--network") == ["none"]
    assert _option_values(command, "--user") == ["10001:10001"]
    assert _option_values(command, "--cpus") == ["2"]
    assert _option_values(command, "--memory") == ["4g"]
    assert _option_values(command, "--memory-swap") == ["4g"]
    assert _option_values(command, "--pids-limit") == ["256"]
    assert _option_values(command, "--cap-drop") == ["ALL"]
    assert _option_values(command, "--security-opt") == ["no-new-privileges"]
    assert _option_values(command, "--log-driver") == ["none"]
    assert _option_values(command, "--tmpfs") == [
        "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
    ]
    assert _option_values(command, "--mount") == [
        f"type=bind,source={workspace.resolve()},target=/workspace"
    ]
    assert "--env" not in command
    assert "/var/run/docker.sock" not in " ".join(command)
    assert command[-3:] == (smoke_day6.PYTHON_IMAGE, "sleep", "300")


def test_maven_e2e_process_uses_only_workspace_writable_state() -> None:
    class RecordingDocker:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def invoke(self, args, **_kwargs):
            command = tuple(args)
            self.commands.append(command)
            if command[-4:] == (
                "java",
                "-cp",
                "target/classes",
                "dev.repoagent.day6.Smoke",
            ):
                return CommandOutcome(0, "maven-e2e-ok\n")
            return CommandOutcome(0, "BUILD SUCCESS\n")

    docker = RecordingDocker()
    checks: list[dict[str, object]] = []

    smoke_day6._run_maven_e2e(docker, checks, "a" * 64)

    maven = next(command for command in docker.commands if "mvn" in command)
    assert "--offline" in maven
    assert "MAVEN_CONFIG=/workspace/.m2" in maven
    assert "MAVEN_OPTS=-Djansi.tmpdir=/workspace/.jansi" in maven
    assert "-Dmaven.repo.local=/workspace/.m2/repository" in maven
    assert not any("/home/" in value for value in maven)


def test_inspect_contract_accepts_exact_security_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    results = smoke_day6._inspect_contract(_valid_inspect(workspace), workspace)

    assert results
    assert all(ok for _name, ok, _detail in results)
    assert {name for name, _ok, _detail in results} >= {
        "non_root_user",
        "network_none",
        "read_only_rootfs",
        "cpu_limit",
        "memory_limit",
        "pid_limit",
        "capabilities_dropped",
        "no_new_privileges",
        "workspace_only_bind",
        "tmpfs_only_tmp",
        "no_docker_socket",
        "no_model_environment",
    }


def test_maven_fixture_has_no_remote_dependencies_or_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "maven"

    smoke_day6._create_maven_fixture(workspace)

    pom = (workspace / "pom.xml").read_text(encoding="utf-8")
    assert "<dependencies>" not in pom
    assert "schemaLocation" not in pom
    assert "https://" not in pom
    assert (workspace / ".m2" / "repository").is_dir()
    assert (workspace / ".jansi").is_dir()


def test_base_image_contract_uses_portable_digest_not_local_image_id(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    base = "python:3.11-slim@sha256:" + "a" * 64
    dockerfile.write_text(f"FROM {base}\nRUN true\n", encoding="utf-8")

    assert smoke_day6._dockerfile_base(dockerfile) == base
    assert smoke_day6._is_digest_pinned_image(base)
    assert smoke_day6._is_sha256_reference("sha256:" + "b" * 64)
    assert not smoke_day6._is_digest_pinned_image("python:3.11-slim")


def test_context_contract_allows_linux_ci_but_preserves_desktop_linux_on_windows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smoke_day6.os, "name", "posix")
    assert smoke_day6._accepted_context("default")
    assert not smoke_day6._accepted_context("")

    monkeypatch.setattr(smoke_day6.os, "name", "nt")
    assert smoke_day6._accepted_context("desktop-linux")
    assert not smoke_day6._accepted_context("default")


def test_host_canary_must_be_unchanged_and_absent_from_workspaces(
    tmp_path: Path,
) -> None:
    canary = tmp_path / "canary.txt"
    first = tmp_path / "python"
    second = tmp_path / "maven"
    first.mkdir()
    second.mkdir()
    value = "host-only-canary-value"
    canary.write_text(value, encoding="utf-8")
    digest = smoke_day6._sha256_file(canary)

    assert smoke_day6._host_canary_isolated(
        canary, digest, value, (first, second)
    )

    (first / "leak.txt").write_text(value, encoding="utf-8")
    assert not smoke_day6._host_canary_isolated(
        canary, digest, value, (first, second)
    )


def test_only_run_owned_workspace_is_made_container_writable(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "day6-smoke-fixture"
    workspace = run_root / "python-workspace"
    workspace.mkdir(parents=True)
    fixture = workspace / "fixture.py"
    fixture.write_text("VALUE = 42\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    smoke_day6._prepare_container_writable_workspace(run_root, workspace)

    assert fixture.is_file()
    with pytest.raises(smoke_day6.AcceptanceError, match="outside"):
        smoke_day6._prepare_container_writable_workspace(run_root, outside)


@pytest.mark.parametrize(
    ("section", "key", "value", "expected_failure"),
    [
        ("Config", "User", "0:0", "non_root_user"),
        ("HostConfig", "NetworkMode", "bridge", "network_none"),
        ("HostConfig", "ReadonlyRootfs", False, "read_only_rootfs"),
        ("HostConfig", "NanoCpus", 0, "cpu_limit"),
        ("HostConfig", "Memory", 1024, "memory_limit"),
        ("HostConfig", "PidsLimit", 0, "pid_limit"),
        ("HostConfig", "CapDrop", [], "capabilities_dropped"),
        ("HostConfig", "SecurityOpt", [], "no_new_privileges"),
        ("HostConfig", "Tmpfs", {}, "tmpfs_only_tmp"),
    ],
)
def test_inspect_contract_detects_security_regressions(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    expected_failure: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = copy.deepcopy(_valid_inspect(workspace))
    nested = payload[section]
    assert isinstance(nested, dict)
    nested[key] = value

    failures = {
        name for name, ok, _detail in smoke_day6._inspect_contract(payload, workspace) if not ok
    }

    assert expected_failure in failures


def test_inspect_contract_rejects_extra_mount_and_model_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = _valid_inspect(workspace)
    mounts = payload["Mounts"]
    config = payload["Config"]
    assert isinstance(mounts, list) and isinstance(config, dict)
    mounts.append(
        {
            "Type": "bind",
            "Source": str(tmp_path),
            "Destination": "/var/run/docker.sock",
            "RW": True,
        }
    )
    config["Env"] = ["OPENAI_API_KEY=must-not-cross-boundary"]

    failures = {
        name for name, ok, _detail in smoke_day6._inspect_contract(payload, workspace) if not ok
    }

    assert failures >= {
        "workspace_only_bind",
        "no_docker_socket",
        "no_model_environment",
    }


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 12,
        "g" * 64,
        f"{'a' * 64}\n{'b' * 64}\n",
        f"warning\n{'a' * 64}\n",
    ],
)
def test_container_id_parser_rejects_non_exact_output(value: str) -> None:
    with pytest.raises(smoke_day6.AcceptanceError, match="exactly one"):
        smoke_day6._parse_container_id(value)


def test_container_registry_removes_only_exact_owned_ids() -> None:
    first = "a" * 64
    second = "b" * 64
    runner = _CleanupRunner()
    docker = smoke_day6.DockerCli(runner)
    registry = smoke_day6.ContainerRegistry(docker, "run-123")
    registry.register(first)
    registry.register(second)

    registry.remove(first)
    ok, errors = registry.cleanup()

    assert ok
    assert errors == []
    removes = [
        command
        for command in runner.commands
        if command[:4] == ("docker", "container", "rm", "--force")
    ]
    assert removes == [
        ("docker", "container", "rm", "--force", first),
        ("docker", "container", "rm", "--force", second),
    ]
    assert not any("prune" in command or "kill" in command for command in runner.commands)


def test_precise_timeout_removes_target_and_checks_control() -> None:
    target = "c" * 64
    control = "d" * 64
    runner = _TimeoutRunner(target, control)
    docker = smoke_day6.DockerCli(runner)
    registry = smoke_day6.ContainerRegistry(docker, "timeout-run")
    registry.register(target)
    registry.register(control)
    checks: list[dict[str, object]] = []

    smoke_day6._verify_precise_timeout_cleanup(
        docker, registry, checks, target, control
    )

    assert [check["name"] for check in checks] == [
        "timeout.deadline_enforced",
        "timeout.exact_container_removed",
    ]
    assert all(check["ok"] for check in checks)
    assert runner.commands[0] == (
        "docker",
        "container",
        "exec",
        target,
        "sleep",
        "60",
    )
    assert runner.commands[1] == (
        "docker",
        "container",
        "rm",
        "--force",
        target,
    )
    assert runner.commands[-1][-1] == control


class _CleanupRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv, *, timeout_seconds: float, max_output_bytes: int):
        del timeout_seconds, max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        if command[1:3] == ("container", "rm"):
            return CommandOutcome(0, command[-1] + "\n")
        if command[1:3] == ("container", "ls"):
            return CommandOutcome(0, "")
        raise AssertionError(f"unexpected command: {command}")


class _TimeoutRunner:
    def __init__(self, target: str, control: str) -> None:
        self.target = target
        self.control = control
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv, *, timeout_seconds: float, max_output_bytes: int):
        del timeout_seconds, max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        if command[1:4] == ("container", "exec", self.target):
            return CommandOutcome(-9, "", timed_out=True)
        if command[1:4] == ("container", "rm", "--force"):
            assert command[-1] == self.target
            return CommandOutcome(0, self.target + "\n")
        if command[1:5] == (
            "container",
            "inspect",
            "--format",
            "{{.State.Running}}",
        ):
            if command[-1] == self.target:
                return CommandOutcome(1, "Error: No such container")
            assert command[-1] == self.control
            return CommandOutcome(0, "true\n")
        raise AssertionError(f"unexpected command: {command}")
