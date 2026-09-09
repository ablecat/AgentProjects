#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonBase = "python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
$mavenBase = "maven:3.9.16-eclipse-temurin-21-noble@sha256:8f6ac126f7810bb5549c4cd122d2bf0e9cda5bdeb0838aa928f09e779fd8bef8"

function Invoke-DockerChecked {
    param([Parameter(Mandatory = $true)][string[]]$CommandArgs)

    & docker @CommandArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "docker $($CommandArgs -join ' ') failed with exit code $exitCode"
    }
}

function Get-DockerValue {
    param([Parameter(Mandatory = $true)][string[]]$CommandArgs)

    $output = @(& docker @CommandArgs 2>&1)
    $exitCode = $LASTEXITCODE
    $text = (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
    if ($exitCode -ne 0) {
        throw "docker $($CommandArgs -join ' ') failed with exit code $exitCode`n$text"
    }
    return $text
}

Invoke-DockerChecked -CommandArgs @(
    "build",
    "--pull=false",
    "--file", (Join-Path $projectRoot "docker/python/Dockerfile"),
    "--tag", "repo-agent-python:0.1",
    $projectRoot
)

Invoke-DockerChecked -CommandArgs @(
    "build",
    "--pull=false",
    "--file", (Join-Path $projectRoot "docker/maven/Dockerfile"),
    "--tag", "repo-agent-maven:0.1",
    $projectRoot
)

$pythonImageId = Get-DockerValue -CommandArgs @("image", "inspect", "--format", "{{.Id}}", "repo-agent-python:0.1")
$mavenImageId = Get-DockerValue -CommandArgs @("image", "inspect", "--format", "{{.Id}}", "repo-agent-maven:0.1")
$dockerServerVersion = Get-DockerValue -CommandArgs @("version", "--format", "{{.Server.Version}}")

$lock = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    dockerServerVersion = $dockerServerVersion
    images = [ordered]@{
        python = [ordered]@{
            tag = "repo-agent-python:0.1"
            imageId = $pythonImageId
            base = $pythonBase
        }
        maven = [ordered]@{
            tag = "repo-agent-maven:0.1"
            imageId = $mavenImageId
            base = $mavenBase
        }
    }
}

$lockPath = Join-Path $projectRoot "docker/image-lock.json"
$lockJson = $lock | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($lockPath, $lockJson + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
Write-Host "Recorded image IDs in $lockPath"

Write-Host "Built sandbox images:"
Invoke-DockerChecked -CommandArgs @(
    "image", "ls", "--no-trunc",
    "--format", "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}",
    "repo-agent-*"
)
