from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke-day3.py"
SPEC = importlib.util.spec_from_file_location("smoke_day3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_day3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_day3)


def _create(network: str, *arguments: str) -> tuple[str, ...]:
    return (
        "docker",
        "container",
        "create",
        "--network",
        network,
        "--mount",
        "type=bind,source=C:/cache,target=/dependencies",
        "--entrypoint",
        "mvn",
        "repo-agent-maven:0.1",
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
    ],
)
def test_maven_smoke_rejects_invalid_retry_or_network_contract(
    commands: list[tuple[str, ...]],
) -> None:
    assert not smoke_day3._maven_create_networks_are_separated(commands)
