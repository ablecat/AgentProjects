#requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(10, 300)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-DockerCapture {
    param([Parameter(Mandatory = $true)][string[]]$CommandArgs)

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& docker @CommandArgs 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
    }
}

function Assert-DockerSucceeded {
    param(
        [Parameter(Mandatory = $true)]$Result,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if ($Result.ExitCode -ne 0) {
        throw "$Description failed with exit code $($Result.ExitCode).`n$($Result.Output)"
    }
}

$contextResult = Invoke-DockerCapture -CommandArgs @("context", "show")
Assert-DockerSucceeded -Result $contextResult -Description "Reading the Docker context"
$context = $contextResult.Output
if ($context -ne "desktop-linux") {
    $contextSwitch = Invoke-DockerCapture -CommandArgs @("context", "use", "desktop-linux")
    Assert-DockerSucceeded -Result $contextSwitch -Description "Switching to desktop-linux"
}

$infoResult = Invoke-DockerCapture -CommandArgs @("info")
if ($infoResult.ExitCode -ne 0) {
    $startResult = Invoke-DockerCapture -CommandArgs @("desktop", "start", "--detach", "--timeout", "30")
    Assert-DockerSucceeded -Result $startResult -Description "Starting Docker Desktop"
}

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
do {
    $infoResult = Invoke-DockerCapture -CommandArgs @("info")
    if ($infoResult.ExitCode -eq 0) {
        $statusResult = Invoke-DockerCapture -CommandArgs @("desktop", "status", "--format", "json")
        Assert-DockerSucceeded -Result $statusResult -Description "Reading Docker Desktop status"
        Write-Host "Docker Desktop is ready: $($statusResult.Output)"
        exit 0
    }
    Start-Sleep -Seconds 2
} while ([DateTime]::UtcNow -lt $deadline)

throw "Docker Desktop did not become ready within $TimeoutSeconds seconds."
