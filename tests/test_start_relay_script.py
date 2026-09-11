from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="relay configuration persistence uses Windows DPAPI",
)


def _powershell_literal(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _powershells() -> tuple[str, ...]:
    resolved: list[str] = []
    for name in ("pwsh", "powershell"):
        executable = shutil.which(name)
        if executable and executable.casefold() not in {
            item.casefold() for item in resolved
        }:
            resolved.append(executable)
    return tuple(resolved)


@pytest.mark.parametrize("powershell", _powershells() or (None,))
def test_relay_config_dpapi_roundtrip_and_fail_closed(
    powershell: str | None,
    tmp_path: Path,
) -> None:
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    config_path = tmp_path / "config" / "relay.json"
    script = f"""
$ErrorActionPreference = "Stop"
. {_powershell_literal(script_path)}
$path = {_powershell_literal(config_path)}
$credentialA = "not-a-real-credential-alpha"
$credentialB = "not-a-real-credential-beta"
$modelA = "fake-model-alpha"
$modelB = "fake-model-beta"

if ((Get-RelayConfigFingerprint -Path $path) -cne "missing") {{
    throw "A missing configuration has the wrong fingerprint."
}}
[IO.Directory]::CreateDirectory((Split-Path -Parent $path)) > $null
[IO.File]::WriteAllBytes($path, [byte[]]@())
$damagedFingerprint = Get-RelayConfigFingerprint -Path $path
Write-RelayConfig `
    -Path $path `
    -ApiKey $credentialA `
    -BaseUrl $relayBaseUrl `
    -ModelId $modelA `
    -ExpectedFingerprint $damagedFingerprint
$raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
if (
    $raw.Contains($credentialA) -or
    $raw.Contains($modelA) -or
    $raw.Contains($relayBaseUrl)
) {{
    throw "Encrypted configuration contains plaintext."
}}
$loadedA = Read-RelayConfig -Path $path
if ($loadedA.ApiKey -cne $credentialA -or $loadedA.Model -cne $modelA) {{
    throw "Initial DPAPI roundtrip failed."
}}

$fingerprintA = Get-RelayConfigFingerprint -Path $path
Write-RelayConfig `
    -Path $path `
    -ApiKey $credentialB `
    -BaseUrl $relayBaseUrl `
    -ModelId $modelB `
    -ExpectedFingerprint $fingerprintA
$staleRejected = $false
try {{
    Write-RelayConfig `
        -Path $path `
        -ApiKey $credentialA `
        -BaseUrl $relayBaseUrl `
        -ModelId $modelA `
        -ExpectedFingerprint $fingerprintA
}}
catch {{
    $staleRejected = $true
}}
if (-not $staleRejected) {{
    throw "A stale configuration writer was accepted."
}}
$loadedB = Read-RelayConfig -Path $path
if ($loadedB.ApiKey -cne $credentialB -or $loadedB.Model -cne $modelB) {{
    throw "Atomic configuration rotation failed."
}}

$configPath = $path
$ConfigureOnly = $true
$Reconfigure = $false
$Verify = $false
$Model = ""
$env:REPO_AGENT_API_KEY = "sentinel-original-key"
$env:REPO_AGENT_BASE_URL = "http://127.0.0.1:9/v1"
$env:REPO_AGENT_MODEL = "sentinel-original-model"
Invoke-RelayLauncher
if (
    $env:REPO_AGENT_API_KEY -cne "sentinel-original-key" -or
    $env:REPO_AGENT_BASE_URL -cne "http://127.0.0.1:9/v1" -or
    $env:REPO_AGENT_MODEL -cne "sentinel-original-model"
) {{
    throw "The original process environment was not restored."
}}

$outer = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
$first = $outer.ciphertext.Substring(0, 1)
if ($first -ceq "0") {{ $replacement = "1" }} else {{ $replacement = "0" }}
$outer.ciphertext = $replacement + $outer.ciphertext.Substring(1)
$outer | ConvertTo-Json | Set-Content -LiteralPath $path -Encoding UTF8
$tamperRejected = $false
try {{ $null = Read-RelayConfig -Path $path }} catch {{ $tamperRejected = $true }}
if (-not $tamperRejected) {{
    throw "A tampered DPAPI configuration was accepted."
}}
"relay-config-tests-ok"
"""
    runner_path = tmp_path / "relay-config-test.ps1"
    runner_path.write_text(script, encoding="utf-8-sig")

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner_path),
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "relay-config-tests-ok" in completed.stdout
    assert "not-a-real-credential" not in config_path.read_text(encoding="utf-8")
