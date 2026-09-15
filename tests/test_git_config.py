from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import repo_agent.git_config as git_config_module
from repo_agent.git_config import effective_core_autocrlf
from repo_agent.processes import CapturedProcess


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", "true"),
        ("yes", "true"),
        ("on", "true"),
        ("1", "true"),
        ("false", "false"),
        ("no", "false"),
        ("off", "false"),
        ("0", "false"),
        ("input", "input"),
    ],
)
def test_effective_core_autocrlf_preserves_only_allowed_value(
    tmp_path: Path, value: str, expected: str
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "core.autocrlf", value)
    assert effective_core_autocrlf(tmp_path) == expected


def test_effective_core_autocrlf_defaults_to_git_default(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    # A machine-wide value may still be present; every accepted result is explicit
    # and safe to pass through the sanitized Git boundary.
    assert effective_core_autocrlf(tmp_path) in {"true", "false", "input"}


def test_effective_core_autocrlf_rejects_unexpected_local_value(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "core.autocrlf", "unexpected")
    with pytest.raises(RuntimeError, match="true, false, or input"):
        effective_core_autocrlf(tmp_path)


def test_effective_core_autocrlf_ignores_git_environment_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "core.autocrlf", "false")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "unexpected")

    assert effective_core_autocrlf(tmp_path) == "false"


def test_effective_core_autocrlf_forwards_the_host_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[float] = []
    completed_process = CapturedProcess(
        0, b"input\n", b"", False, False, False
    )

    def record_timeout(*_args, **kwargs):
        observed.append(kwargs["timeout_seconds"])
        return completed_process

    monkeypatch.setattr(git_config_module, "run_isolated_capture", record_timeout)

    assert effective_core_autocrlf(tmp_path, timeout_seconds=0.75) == "input"
    assert observed == [0.75]
