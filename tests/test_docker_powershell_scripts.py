from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build-sandboxes.ps1"
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify-docker.ps1"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _switch_block_spans(script: str) -> tuple[tuple[int, int], ...]:
    marker = "if ($IncludeLegacyMaven)"
    spans: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = script.find(marker, offset)
        if start < 0:
            return tuple(spans)
        opening = script.find("{", start + len(marker))
        assert opening >= 0
        depth = 0
        for index in range(opening, len(script)):
            if script[index] == "{":
                depth += 1
            elif script[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
                    offset = index + 1
                    break
        else:
            raise AssertionError("unterminated IncludeLegacyMaven block")


def _assert_occurrences_are_legacy_gated(script: str, value: str) -> None:
    spans = _switch_block_spans(script)
    positions: list[int] = []
    offset = 0
    while True:
        position = script.find(value, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + len(value)
    assert positions
    assert all(any(start <= position < end for start, end in spans) for position in positions)


@pytest.mark.parametrize("include_legacy_maven", [False, True])
def test_build_script_selects_images_and_lock_entries(
    tmp_path: Path,
    include_legacy_maven: bool,
) -> None:
    powershell = _powershell()
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    project = tmp_path / "project"
    scripts = project / "scripts"
    (project / "docker").mkdir(parents=True)
    scripts.mkdir()
    copied_script = scripts / BUILD_SCRIPT.name
    shutil.copy2(BUILD_SCRIPT, copied_script)

    harness = tmp_path / "harness.ps1"
    harness.write_text(
        r"""
$ErrorActionPreference = "Stop"
$global:testDockerCommands = [System.Collections.Generic.List[object]]::new()
function global:docker {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$DockerArgs)
    $record = @($DockerArgs | ForEach-Object { [string]$_ })
    $global:testDockerCommands.Add([object]$record)
    $global:LASTEXITCODE = 0
    if ($record[0] -eq "image" -and $record[1] -eq "inspect") {
        if ($record[-1] -eq "repo-agent-maven:0.1") {
            return "sha256:$('b' * 64)"
        }
        return "sha256:$('a' * 64)"
    }
    if ($record[0] -eq "version") {
        return "28.5.1"
    }
    if ($record[0] -eq "image" -and $record[1] -eq "ls") {
        return $record[-1]
    }
}

if ($env:TEST_INCLUDE_LEGACY_MAVEN -eq "1") {
    & $env:TEST_BUILD_SCRIPT -IncludeLegacyMaven
}
else {
    & $env:TEST_BUILD_SCRIPT
}
$lock = Get-Content -LiteralPath $env:TEST_LOCK_PATH -Raw | ConvertFrom-Json
$payload = [ordered]@{
    commands = @($global:testDockerCommands)
    lock = $lock
}
Write-Output ("__RESULT__" + ($payload | ConvertTo-Json -Compress -Depth 8))
""".strip()
        + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "TEST_BUILD_SCRIPT": str(copied_script),
            "TEST_LOCK_PATH": str(project / "docker" / "image-lock.json"),
            "TEST_INCLUDE_LEGACY_MAVEN": "1" if include_legacy_maven else "0",
        }
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-File", str(harness)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("__RESULT__")
    )
    payload = json.loads(result_line.removeprefix("__RESULT__"))
    commands = payload["commands"]
    command_text = "\n".join(" ".join(command) for command in commands)
    build_commands = [command for command in commands if command[0] == "build"]

    expected_count = 2 if include_legacy_maven else 1
    assert len(build_commands) == expected_count
    assert "repo-agent-python:0.1" in command_text
    assert set(payload["lock"]["images"]) == (
        {"python", "maven"} if include_legacy_maven else {"python"}
    )
    assert ("repo-agent-maven:0.1" in command_text) is include_legacy_maven


def test_verifier_gates_legacy_maven_and_keeps_python_security_contract() -> None:
    script = VERIFY_SCRIPT.read_text(encoding="utf-8-sig")

    assert "[switch]$IncludeLegacyMaven" in script
    assert '"--pull", "always"' not in script
    _assert_occurrences_are_legacy_gated(script, '"repo-agent-maven:0.1"')
    for required in (
        '"--init"',
        '"--network", "none"',
        '"--read-only"',
        '"--user", "10001:10001"',
        '"--cpus", "2"',
        '"--memory", "4g"',
        '"--memory-swap", "4g"',
        '"--pids-limit", "256"',
        '"--cap-drop", "ALL"',
        '"--security-opt", "no-new-privileges"',
        '"--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777"',
    ):
        assert required in script


def test_docker_ci_uses_only_the_python_release_image() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    docker_job = workflow.split("  docker-e2e:", 1)[1].split(
        "  windows-process-lifecycle:", 1
    )[0]

    assert "repo-agent-python:0.1" in docker_job
    assert "python:3.11.16-slim-bookworm@sha256:" in docker_job
    assert "maven" not in docker_job.casefold()
    for smoke in ("smoke-day1.py", "smoke-day2.py", "smoke-day3.py", "smoke-day6.py"):
        assert smoke in docker_job
