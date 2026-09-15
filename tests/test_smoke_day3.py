from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke-day3.py"
SPEC = importlib.util.spec_from_file_location("smoke_day3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_day3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_day3)

RUN_ID = "a" * 32
RUN_PREFIX = "repo-agent-day3-smoke-deadbeef-mvn"
VOLUME_NAME = f"{RUN_PREFIX}-{RUN_ID}-cache"


@pytest.mark.parametrize(
    ("relative_path", "cache_leaf"),
    [
        ("docker/python/Dockerfile", "/cache/python"),
        ("docker/maven/Dockerfile", "/cache/m2"),
    ],
)
def test_sandbox_image_prepares_owned_cache_leaf(
    relative_path: str, cache_leaf: str
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    dockerfile = (project_root / relative_path).read_text(encoding="utf-8")

    assert cache_leaf in dockerfile
    assert "chown --recursive 10001:10001 /workspace /cache" in dockerfile


def _create(
    network: str,
    *arguments: str,
    volume: str = VOLUME_NAME,
) -> tuple[str, ...]:
    return (
        "docker",
        "container",
        "create",
        "--name",
        f"{RUN_PREFIX}-{RUN_ID}-1",
        "--label",
        f"io.github.ablecat.repo-agent.check.run={RUN_ID}",
        "--label",
        "io.github.ablecat.repo-agent.check.sequence=1",
        "--init",
        "--pull",
        "never",
        "--restart",
        "no",
        "--network",
        network,
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
        "--tmpfs",
        "/workspace:rw,nosuid,nodev,size=2g,mode=0700,uid=10001,gid=10001",
        "--mount",
        "type=bind,source=C:/snapshot,target=/source,readonly",
        "--mount",
        f"type=volume,source={volume},target=/cache",
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
        "--entrypoint",
        "/bin/sh",
        "repo-agent-maven:0.1",
        "-c",
        smoke_day3._WORKSPACE_ENTRYPOINT,
        "repo-agent-check",
        "mvn",
        *arguments,
    )


def _valid_commands(bootstrap_attempts: int = 4) -> list[tuple[str, ...]]:
    return [
        *(_create("bridge", "dependency:get") for _ in range(bootstrap_attempts)),
        _create("none", "--offline", "test"),
    ]


@pytest.mark.parametrize("bootstrap_attempts", range(4, 9))
def test_maven_smoke_accepts_bounded_bootstrap_retries(
    bootstrap_attempts: int,
) -> None:
    assert smoke_day3._maven_create_networks_are_separated(
        _valid_commands(bootstrap_attempts)
    )


@pytest.mark.parametrize(
    "commands",
    [
        _valid_commands(3),
        _valid_commands(9),
        [*_valid_commands(4)[:-1], _create("bridge", "--offline", "test")],
        [_create("none", "dependency:get"), *_valid_commands(4)[1:]],
        [*_valid_commands(4)[:-1], _create("none", "test")],
        [
            *_valid_commands(4)[:-1],
            _create("none", "--offline", "test", volume="other-cache"),
        ],
    ],
)
def test_maven_smoke_rejects_invalid_retry_or_network_contract(
    commands: list[tuple[str, ...]],
) -> None:
    assert not smoke_day3._maven_create_networks_are_separated(commands)


def test_hardening_requires_readonly_source_named_cache_and_uid_tmpfs() -> None:
    command = _create("none", "--offline", "test")

    assert smoke_day3._is_hardened(command)
    assert smoke_day3._dependency_volume_name(command).endswith(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-cache"
    )

    source_is_writable = tuple(
        value.replace(",readonly", "")
        if value.startswith("type=bind,source=C:/snapshot")
        else value
        for value in command
    )
    cache_is_a_bind = tuple(
        value.replace("type=volume", "type=bind")
        if "target=/cache" in value
        else value
        for value in command
    )
    workspace_has_wrong_uid = tuple(
        value.replace("uid=10001", "uid=0")
        if value.startswith("/workspace:")
        else value
        for value in command
    )

    assert not smoke_day3._is_hardened(source_is_writable)
    assert not smoke_day3._is_hardened(cache_is_a_bind)
    assert not smoke_day3._is_hardened(workspace_has_wrong_uid)


def _insert_before_image(
    command: tuple[str, ...], *values: str
) -> tuple[str, ...]:
    index = command.index("repo-agent-maven:0.1")
    return (*command[:index], *values, *command[index:])


@pytest.mark.parametrize(
    "extra",
    [
        ("--privileged",),
        ("--cap-add", "SYS_ADMIN"),
        ("--pid", "host"),
        ("--pid=host",),
        ("--device", "/dev/null"),
        ("--env-file", "secrets.env"),
        ("--volume", "C:/host:/host"),
        ("-v", "C:/host:/host"),
        ("--mount", "type=bind,source=C:/host,target=/host,readonly"),
        ("--env", "EXTRA=1"),
    ],
)
def test_hardening_rejects_every_extra_docker_option(extra: tuple[str, ...]) -> None:
    command = _insert_before_image(_create("none", "--offline", "test"), *extra)

    assert not smoke_day3._is_hardened(command)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("/bin/sh", "/bin/true"),
        (smoke_day3._WORKSPACE_ENTRYPOINT, 'exec "$@"'),
        ("repo-agent-maven:0.1", "repo-agent-python:0.1"),
        ("repo-agent-check", "unexpected-sentinel"),
    ],
)
def test_hardening_rejects_entrypoint_image_wrapper_or_tail_change(
    old: str, new: str
) -> None:
    command = list(_create("none", "--offline", "test"))
    command[command.index(old)] = new

    assert not smoke_day3._is_hardened(tuple(command))


def _volume_create(name: str = VOLUME_NAME) -> tuple[str, ...]:
    return (
        "docker",
        "volume",
        "create",
        "--driver",
        "local",
        "--opt",
        "type=tmpfs",
        "--opt",
        "device=tmpfs",
        "--opt",
        "o=rw,nosuid,nodev,size=1g,mode=0700,uid=10001,gid=10001",
        "--label",
        f"io.github.ablecat.repo-agent.check.run={RUN_ID}",
        "--label",
        "io.github.ablecat.repo-agent.check.resource=dependencies",
        name,
    )


def _anchor_create() -> tuple[str, ...]:
    return (
        "docker",
        "container",
        "create",
        "--name",
        f"{RUN_PREFIX}-{RUN_ID}-cache-anchor",
        "--label",
        f"io.github.ablecat.repo-agent.check.run={RUN_ID}",
        "--label",
        "io.github.ablecat.repo-agent.check.sequence=0",
        "--init",
        "--rm",
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
        "--mount",
        f"type=volume,source={VOLUME_NAME},target=/cache",
        "--entrypoint",
        "/bin/sh",
        "repo-agent-maven:0.1",
        "-c",
        smoke_day3._CACHE_ANCHOR_ENTRYPOINT,
    )


def test_dependency_volume_contract_is_bounded_labeled_and_closed() -> None:
    command = _volume_create()
    anchor = _anchor_create()
    anchor_start = ("docker", "container", "start", "f" * 64)
    container = _create("none", "--offline", "test")

    assert smoke_day3._is_hardened_volume_create(command)
    assert smoke_day3._is_hardened_anchor(anchor)
    assert smoke_day3._runner_dependency_volume_contract(
        [command, anchor, anchor_start, container]
    )

    wrong_driver = tuple("other" if value == "local" else value for value in command)
    missing_option = command[: command.index("device=tmpfs") - 1] + command[
        command.index("device=tmpfs") + 1 :
    ]
    wrong_label = tuple(
        value.replace(RUN_ID, "b" * 32) if "check.run=" in value else value
        for value in command
    )
    extra_option = (*command[:-1], "--opt", "copy", command[-1])

    assert not smoke_day3._is_hardened_volume_create(wrong_driver)
    assert not smoke_day3._is_hardened_volume_create(missing_option)
    assert not smoke_day3._is_hardened_volume_create(wrong_label)
    assert not smoke_day3._is_hardened_volume_create(extra_option)
    assert not smoke_day3._runner_dependency_volume_contract([command])
    assert not smoke_day3._runner_dependency_volume_contract([container])
    assert not smoke_day3._runner_dependency_volume_contract(
        [
            command,
            anchor,
            anchor_start,
            _create("none", "--offline", "test", volume="other-cache"),
        ]
    )
    assert not smoke_day3._runner_dependency_volume_contract(
        [command, _volume_create(), anchor, anchor_start, container]
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("none", "bridge"),
        ("/bin/sh", "/bin/true"),
        (smoke_day3._CACHE_ANCHOR_ENTRYPOINT, "exit 0\n"),
        (VOLUME_NAME, "other-cache"),
    ],
)
def test_cache_anchor_hardening_rejects_contract_changes(old: str, new: str) -> None:
    original = _anchor_create()
    command = tuple(value.replace(old, new) for value in original)

    assert command != original
    assert not smoke_day3._is_hardened_anchor(command)


def test_cache_anchor_hardening_rejects_extra_privilege() -> None:
    command = _insert_before_image(_anchor_create(), "--privileged")

    assert not smoke_day3._is_hardened_anchor(command)
