#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runId = [guid]::NewGuid().ToString("N")
$runLabelKey = "io.github.ablecat.repo-agent.verify.run"
$runLabel = "$runLabelKey=$runId"
$tempBase = Join-Path $projectRoot ".tmp"
$tempRoot = Join-Path $tempBase "docker-verify-$runId"
$pythonWorkspace = Join-Path $tempRoot "python-workspace"
$pythonCache = Join-Path $tempRoot "python-cache"
$mavenWorkspace = Join-Path $tempRoot "maven-workspace"
$mavenCache = Join-Path $tempRoot "maven-cache"
$readOnlyInput = Join-Path $tempRoot "read-only-input"
$createdContainers = New-Object "System.Collections.Generic.List[string]"
$pendingError = $null
$cleanupError = $null

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$CommandArgs,
        [int[]]$AllowedExitCodes = @(0)
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $outputLines = @(& $FilePath @CommandArgs 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    $outputText = (($outputLines | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
    if ($AllowedExitCodes -notcontains $exitCode) {
        throw "$FilePath $($CommandArgs -join ' ') failed with exit code $exitCode`n$outputText"
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $outputText
    }
}

function Invoke-DockerCapture {
    param(
        [Parameter(Mandatory = $true)][string[]]$CommandArgs,
        [int[]]$AllowedExitCodes = @(0)
    )

    return Invoke-NativeCapture -FilePath "docker" -CommandArgs $CommandArgs -AllowedExitCodes $AllowedExitCodes
}

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )

    if (-not $Condition) {
        throw "Assertion failed: $Message"
    }
    Write-Host "[PASS] $Message"
}

function Get-SandboxCreateArgs {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][string]$Image,
        [Parameter(Mandatory = $true)][string]$WorkspacePath,
        [Parameter(Mandatory = $true)][string]$CachePath,
        [Parameter(Mandatory = $true)][string]$CacheTarget,
        [string[]]$AdditionalMounts = @(),
        [string[]]$ContainerCommand = @("sleep", "300")
    )

    foreach ($path in @($WorkspacePath, $CachePath)) {
        if ($path.Contains(",")) {
            throw "Docker bind source paths containing commas are not supported: $path"
        }
    }

    $dockerArgs = @(
        "create",
        "--name", $Name,
        "--label", $runLabel,
        "--label", "io.github.ablecat.repo-agent.verify.case=$CaseName",
        "--init",
        "--pull", "never",
        "--restart", "no",
        "--network", "none",
        "--read-only",
        "--user", "10001:10001",
        "--cpus", "2",
        "--memory", "4g",
        "--memory-swap", "4g",
        "--pids-limit", "256",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--mount", "type=bind,source=$WorkspacePath,target=/workspace",
        "--mount", "type=bind,source=$CachePath,target=$CacheTarget"
    )
    foreach ($mount in $AdditionalMounts) {
        $dockerArgs += @("--mount", $mount)
    }
    $dockerArgs += $Image
    $dockerArgs += $ContainerCommand
    return $dockerArgs
}

function New-SandboxContainer {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][string]$Image,
        [Parameter(Mandatory = $true)][string]$WorkspacePath,
        [Parameter(Mandatory = $true)][string]$CachePath,
        [Parameter(Mandatory = $true)][string]$CacheTarget,
        [string[]]$AdditionalMounts = @(),
        [string[]]$ContainerCommand = @("sleep", "300")
    )

    $createArgs = Get-SandboxCreateArgs @PSBoundParameters
    $result = Invoke-DockerCapture -CommandArgs $createArgs
    $idMatch = [regex]::Match($result.Output, "(?im)^[0-9a-f]{64}$")
    if (-not $idMatch.Success) {
        throw "docker create did not return a container ID: $($result.Output)"
    }
    $containerId = $idMatch.Value
    $createdContainers.Add($containerId) | Out-Null
    return $containerId
}

function Start-SandboxContainer {
    param([Parameter(Mandatory = $true)][string]$ContainerId)

    Invoke-DockerCapture -CommandArgs @("container", "start", $ContainerId) | Out-Null
    $stateResult = Invoke-DockerCapture -CommandArgs @("container", "inspect", "--format", "{{.State.Running}}", $ContainerId)
    if ($stateResult.Output -ne "true") {
        $logs = Invoke-DockerCapture -CommandArgs @("container", "logs", $ContainerId) -AllowedExitCodes @(0, 1)
        throw "Container $ContainerId did not remain running.`n$($logs.Output)"
    }
}

function Invoke-ContainerExec {
    param(
        [Parameter(Mandatory = $true)][string]$ContainerId,
        [Parameter(Mandatory = $true)][string[]]$CommandArgs,
        [int[]]$AllowedExitCodes = @(0)
    )

    $dockerArgs = @("container", "exec", $ContainerId) + $CommandArgs
    return Invoke-DockerCapture -CommandArgs $dockerArgs -AllowedExitCodes $AllowedExitCodes
}

function Remove-ExactContainer {
    param([Parameter(Mandatory = $true)][string]$ContainerId)

    if ($ContainerId -notmatch "^[0-9a-f]{12,64}$") {
        throw "Refusing to remove an invalid container ID: $ContainerId"
    }
    Invoke-DockerCapture -CommandArgs @("container", "rm", "--force", $ContainerId) -AllowedExitCodes @(0, 1) | Out-Null
}

function Get-ContainerRunningState {
    param([Parameter(Mandatory = $true)][string]$ContainerId)

    return Invoke-DockerCapture -CommandArgs @("container", "inspect", "--format", "{{.State.Running}}", $ContainerId) -AllowedExitCodes @(0, 1)
}

function Stop-ContainerAfterTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$ContainerId,
        [ValidateRange(1, 30)][int]$TimeoutSeconds = 1
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $state = Get-ContainerRunningState -ContainerId $ContainerId
        if ($state.ExitCode -ne 0 -or $state.Output -ne "true") {
            return [pscustomobject]@{ TimedOut = $false }
        }
        Start-Sleep -Milliseconds 200
    } while ([DateTime]::UtcNow -lt $deadline)

    Remove-ExactContainer -ContainerId $ContainerId
    return [pscustomobject]@{ TimedOut = $true }
}

function Get-ImageId {
    param([Parameter(Mandatory = $true)][string]$Image)

    return (Invoke-DockerCapture -CommandArgs @("image", "inspect", "--format", "{{.Id}}", $Image)).Output
}

try {
    foreach ($requiredCommand in @("docker", "wsl")) {
        Assert-Condition -Condition ($null -ne (Get-Command $requiredCommand -ErrorAction SilentlyContinue)) -Message "$requiredCommand CLI is installed"
    }

    $context = (Invoke-DockerCapture -CommandArgs @("context", "show")).Output
    Assert-Condition -Condition ($context -eq "desktop-linux") -Message "Docker context is desktop-linux"

    $versionResult = Invoke-DockerCapture -CommandArgs @("version", "--format", "{{json .}}")
    $version = $versionResult.Output | ConvertFrom-Json
    Assert-Condition -Condition ($null -ne $version.Client) -Message "Docker Client is available"
    Assert-Condition -Condition ($null -ne $version.Server) -Message "Docker Server is available"
    Assert-Condition -Condition ($version.Server.Os -eq "linux") -Message "Docker Server OS is Linux"
    Assert-Condition -Condition ($version.Server.Arch -eq "amd64") -Message "Docker Server architecture is amd64"
    Write-Host "Docker Client/Server: $($version.Client.Version) / $($version.Server.Version)"

    $compose = (Invoke-DockerCapture -CommandArgs @("compose", "version", "--short")).Output
    Assert-Condition -Condition ($compose -match "^v?2\.") -Message "Docker Compose v2 is available ($compose)"

    $infoResult = Invoke-DockerCapture -CommandArgs @("info", "--format", "{{json .}}")
    $info = $infoResult.Output | ConvertFrom-Json
    Assert-Condition -Condition ($info.OSType -eq "linux") -Message "Docker info reports the Linux engine"
    Assert-Condition -Condition ([int64]$info.NCPU -gt 0) -Message "Docker reports available CPUs ($($info.NCPU))"
    Assert-Condition -Condition ([int64]$info.MemTotal -ge 4294967296) -Message "Docker has at least 4 GiB available"
    Assert-Condition -Condition (-not [string]::IsNullOrWhiteSpace([string]$info.Driver)) -Message "Docker reports a storage driver ($($info.Driver))"
    Assert-Condition -Condition (@($info.SecurityOptions).Count -gt 0) -Message "Docker reports security options"
    Assert-Condition -Condition ([bool]$info.MemoryLimit) -Message "Docker memory limits are supported"
    Assert-Condition -Condition ([bool]$info.SwapLimit) -Message "Docker swap limits are supported"
    Assert-Condition -Condition ([bool]$info.PidsLimit) -Message "Docker PID limits are supported"

    $mirrorCount = @($info.RegistryConfig.Mirrors).Count
    $httpProxyConfigured = -not [string]::IsNullOrWhiteSpace([string]$info.HttpProxy)
    $httpsProxyConfigured = -not [string]::IsNullOrWhiteSpace([string]$info.HttpsProxy)
    Write-Host "Existing Docker settings preserved: registry mirrors=$mirrorCount, HTTP proxy configured=$httpProxyConfigured, HTTPS proxy configured=$httpsProxyConfigured"

    $wslVersion = Invoke-NativeCapture -FilePath "wsl" -CommandArgs @("--version")
    $wslVersionText = $wslVersion.Output.Replace(([char]0).ToString(), "")
    Assert-Condition -Condition (-not [string]::IsNullOrWhiteSpace($wslVersionText) -and $wslVersionText -match "\d+\.\d+") -Message "WSL version information is available"
    $runningDistros = (Invoke-NativeCapture -FilePath "wsl" -CommandArgs @("--list", "--running", "--quiet")).Output.Replace(([char]0).ToString(), "")
    $runningNames = @($runningDistros -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    Assert-Condition -Condition ($runningNames -contains "docker-desktop") -Message "docker-desktop WSL distribution is running"
    $verboseDistros = (Invoke-NativeCapture -FilePath "wsl" -CommandArgs @("--list", "--verbose")).Output.Replace(([char]0).ToString(), "")
    $dockerDesktopLine = @($verboseDistros -split "`r?`n" | Where-Object { $_ -match "(^|\s)docker-desktop(\s|$)" })
    Assert-Condition -Condition ($dockerDesktopLine.Count -eq 1) -Message "docker-desktop WSL distribution is present"
    Assert-Condition -Condition ($dockerDesktopLine[0] -match "\s2\s*$") -Message "docker-desktop uses WSL version 2"

    $wslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"
    Write-Host "User .wslconfig present (left unchanged): $(Test-Path -LiteralPath $wslConfigPath)"

    $helloName = "repo-agent-hello-$runId"
    $hello = Invoke-DockerCapture -CommandArgs @("run", "--rm", "--pull", "always", "--name", $helloName, "--label", $runLabel, "hello-world")
    Assert-Condition -Condition ($hello.Output -match "Hello from Docker!") -Message "hello-world pull/create/run path succeeds"
    $helloInspect = Invoke-DockerCapture -CommandArgs @("container", "inspect", $helloName) -AllowedExitCodes @(0, 1)
    Assert-Condition -Condition ($helloInspect.ExitCode -ne 0) -Message "hello-world container was automatically removed"

    foreach ($directory in @($pythonWorkspace, $pythonCache, $mavenWorkspace, $mavenCache, $readOnlyInput)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $resolvedTempRoot = [IO.Path]::GetFullPath($tempRoot)
    $resolvedTempBase = [IO.Path]::GetFullPath($tempBase).TrimEnd("\") + "\"
    Assert-Condition -Condition ($resolvedTempRoot.StartsWith($resolvedTempBase, [StringComparison]::OrdinalIgnoreCase)) -Message "temporary verification directory is scoped under the project .tmp directory"
    Assert-Condition -Condition ([IO.Path]::GetPathRoot($resolvedTempRoot) -eq "D:\") -Message "bind-mount verification source is on drive D:"

    $proofText = "host-proof-$runId"
    $proofPath = Join-Path $readOnlyInput "proof.txt"
    [IO.File]::WriteAllText($proofPath, $proofText, (New-Object Text.UTF8Encoding($false)))
    $proofHashBefore = (Get-FileHash -LiteralPath $proofPath -Algorithm SHA256).Hash

    $pythonImage = "repo-agent-python:0.1"
    $mavenImage = "repo-agent-maven:0.1"
    $pythonImageId = Get-ImageId -Image $pythonImage
    $mavenImageId = Get-ImageId -Image $mavenImage
    Assert-Condition -Condition ($pythonImageId -match "^sha256:[0-9a-f]{64}$") -Message "Python sandbox image exists"
    Assert-Condition -Condition ($mavenImageId -match "^sha256:[0-9a-f]{64}$") -Message "Maven sandbox image exists"

    $lockPath = Join-Path $projectRoot "docker/image-lock.json"
    Assert-Condition -Condition (Test-Path -LiteralPath $lockPath -PathType Leaf) -Message "docker/image-lock.json exists"
    $imageLock = [IO.File]::ReadAllText($lockPath) | ConvertFrom-Json
    Assert-Condition -Condition ($imageLock.images.python.imageId -eq $pythonImageId) -Message "Python image ID matches image-lock.json"
    Assert-Condition -Condition ($imageLock.images.maven.imageId -eq $mavenImageId) -Message "Maven image ID matches image-lock.json"

    foreach ($image in @($pythonImage, $mavenImage)) {
        $defaultUser = (Invoke-DockerCapture -CommandArgs @("image", "inspect", "--format", "{{.Config.User}}", $image)).Output
        Assert-Condition -Condition ($defaultUser -eq "10001:10001") -Message "$image defaults to UID/GID 10001"
        $declaredVolumes = (Invoke-DockerCapture -CommandArgs @("image", "inspect", "--format", "{{json .Config.Volumes}}", $image)).Output
        Assert-Condition -Condition ($declaredVolumes -eq "null" -or $declaredVolumes -eq "{}") -Message "$image declares no implicit volumes"
    }

    $readOnlyMount = "type=bind,source=$readOnlyInput,target=/input,readonly"
    $pythonName = "repo-agent-python-$runId"
    $pythonId = New-SandboxContainer -Name $pythonName -CaseName "python" -Image $pythonImage -WorkspacePath $pythonWorkspace -CachePath $pythonCache -CacheTarget "/cache" -AdditionalMounts @($readOnlyMount)
    Start-SandboxContainer -ContainerId $pythonId

    $mavenName = "repo-agent-maven-$runId"
    $mavenId = New-SandboxContainer -Name $mavenName -CaseName "maven" -Image $mavenImage -WorkspacePath $mavenWorkspace -CachePath $mavenCache -CacheTarget "/home/repo-agent/.m2"
    Start-SandboxContainer -ContainerId $mavenId

    $inspectResult = Invoke-DockerCapture -CommandArgs @("container", "inspect", $pythonId)
    $inspect = @($inspectResult.Output | ConvertFrom-Json)[0]
    Assert-Condition -Condition ($inspect.HostConfig.NetworkMode -eq "none") -Message "network mode none is active"
    Assert-Condition -Condition ([bool]$inspect.HostConfig.ReadonlyRootfs) -Message "read-only root filesystem is active"
    Assert-Condition -Condition ($inspect.Config.User -eq "10001:10001") -Message "runtime user override is 10001:10001"
    Assert-Condition -Condition ([int64]$inspect.HostConfig.NanoCpus -eq 2000000000) -Message "2 CPU limit is active"
    Assert-Condition -Condition ([int64]$inspect.HostConfig.Memory -eq 4294967296) -Message "4 GiB memory limit is active"
    Assert-Condition -Condition ([int64]$inspect.HostConfig.MemorySwap -eq 4294967296) -Message "memory-swap is capped at 4 GiB"
    Assert-Condition -Condition ([int64]$inspect.HostConfig.PidsLimit -eq 256) -Message "256 PID limit is active"
    Assert-Condition -Condition (@($inspect.HostConfig.CapDrop) -contains "ALL") -Message "all Linux capabilities are dropped"
    Assert-Condition -Condition ($null -eq $inspect.HostConfig.CapAdd -or @($inspect.HostConfig.CapAdd).Count -eq 0) -Message "no Linux capabilities are added"
    Assert-Condition -Condition (-not [bool]$inspect.HostConfig.Privileged) -Message "privileged mode is disabled"
    Assert-Condition -Condition (@($inspect.HostConfig.SecurityOpt) -contains "no-new-privileges") -Message "no-new-privileges is active"
    Assert-Condition -Condition ([bool]$inspect.HostConfig.Init) -Message "container init is active"
    Assert-Condition -Condition ($inspect.HostConfig.RestartPolicy.Name -eq "no") -Message "automatic restart is disabled"

    $tmpfsProperty = $inspect.HostConfig.Tmpfs.PSObject.Properties["/tmp"]
    Assert-Condition -Condition ($null -ne $tmpfsProperty) -Message "/tmp is backed by tmpfs"
    $tmpfsOptions = @($tmpfsProperty.Value -split ",")
    foreach ($requiredOption in @("rw", "nosuid", "nodev", "size=256m", "mode=1777")) {
        Assert-Condition -Condition ($tmpfsOptions -contains $requiredOption) -Message "/tmp tmpfs includes $requiredOption"
    }

    $workspaceMount = @($inspect.Mounts | Where-Object { $_.Destination -eq "/workspace" })
    $cacheMount = @($inspect.Mounts | Where-Object { $_.Destination -eq "/cache" })
    $inputMount = @($inspect.Mounts | Where-Object { $_.Destination -eq "/input" })
    Assert-Condition -Condition (@($inspect.Mounts).Count -eq 3) -Message "Python container has exactly three explicit bind mounts"
    Assert-Condition -Condition ($workspaceMount.Count -eq 1 -and $workspaceMount[0].Type -eq "bind" -and [bool]$workspaceMount[0].RW) -Message "workspace bind mount is read-write"
    Assert-Condition -Condition ($cacheMount.Count -eq 1 -and $cacheMount[0].Type -eq "bind" -and [bool]$cacheMount[0].RW) -Message "Python dependency cache bind mount is read-write"
    Assert-Condition -Condition ($inputMount.Count -eq 1 -and -not [bool]$inputMount[0].RW) -Message "input repository bind mount is read-only"
    Assert-Condition -Condition (@($inspect.Mounts | Where-Object { $_.Destination -eq "/var/run/docker.sock" }).Count -eq 0) -Message "Docker socket is not mounted"
    foreach ($mount in @($inspect.Mounts | Where-Object { $_.Type -eq "bind" })) {
        Assert-Condition -Condition ([IO.Path]::GetFullPath([string]$mount.Source).StartsWith($resolvedTempRoot, [StringComparison]::OrdinalIgnoreCase)) -Message "bind source stays inside the run-specific temporary directory"
    }

    $mavenInspectResult = Invoke-DockerCapture -CommandArgs @("container", "inspect", $mavenId)
    $mavenInspect = @($mavenInspectResult.Output | ConvertFrom-Json)[0]
    Assert-Condition -Condition ($mavenInspect.HostConfig.NetworkMode -eq "none" -and [bool]$mavenInspect.HostConfig.ReadonlyRootfs) -Message "Maven container uses network-none and a read-only root filesystem"
    Assert-Condition -Condition ([int64]$mavenInspect.HostConfig.NanoCpus -eq 2000000000 -and [int64]$mavenInspect.HostConfig.Memory -eq 4294967296 -and [int64]$mavenInspect.HostConfig.MemorySwap -eq 4294967296 -and [int64]$mavenInspect.HostConfig.PidsLimit -eq 256) -Message "Maven container uses the required CPU, memory, swap, and PID limits"
    Assert-Condition -Condition (@($mavenInspect.HostConfig.CapDrop) -contains "ALL" -and @($mavenInspect.HostConfig.SecurityOpt) -contains "no-new-privileges") -Message "Maven container drops capabilities and prevents privilege escalation"
    $mavenWorkspaceMount = @($mavenInspect.Mounts | Where-Object { $_.Destination -eq "/workspace" })
    $mavenCacheMount = @($mavenInspect.Mounts | Where-Object { $_.Destination -eq "/home/repo-agent/.m2" })
    Assert-Condition -Condition (@($mavenInspect.Mounts).Count -eq 2) -Message "Maven container has exactly two explicit bind mounts"
    Assert-Condition -Condition ($mavenWorkspaceMount.Count -eq 1 -and $mavenWorkspaceMount[0].Type -eq "bind" -and [bool]$mavenWorkspaceMount[0].RW) -Message "Maven workspace bind mount is read-write"
    Assert-Condition -Condition ($mavenCacheMount.Count -eq 1 -and $mavenCacheMount[0].Type -eq "bind" -and [bool]$mavenCacheMount[0].RW) -Message "Maven dependency cache bind mount is read-write"
    foreach ($mount in @($mavenInspect.Mounts)) {
        Assert-Condition -Condition ($mount.Type -eq "bind" -and [IO.Path]::GetFullPath([string]$mount.Source).StartsWith($resolvedTempRoot, [StringComparison]::OrdinalIgnoreCase)) -Message "Maven bind source stays inside the run-specific temporary directory"
    }

    $environmentAllowlists = [ordered]@{
        python = @("PATH", "LANG", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256", "HOME", "PYTHONDONTWRITEBYTECODE", "PIP_DISABLE_PIP_VERSION_CHECK")
        maven = @("PATH", "JAVA_HOME", "LANG", "LANGUAGE", "LC_ALL", "JAVA_VERSION", "MAVEN_HOME", "MAVEN_CONFIG", "HOME")
    }
    $environmentInspects = [ordered]@{
        python = $inspect
        maven = $mavenInspect
    }
    foreach ($runtimeName in @("python", "maven")) {
        $environmentNames = @($environmentInspects[$runtimeName].Config.Env | ForEach-Object { @($_ -split "=", 2)[0] })
        $unexpectedNames = @($environmentNames | Where-Object { $environmentAllowlists[$runtimeName] -notcontains $_ })
        Assert-Condition -Condition ($unexpectedNames.Count -eq 0) -Message "$runtimeName container environment matches its reviewed allowlist"
    }

    $pythonVersion = (Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("python", "--version")).Output
    $pytestVersion = (Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("pytest", "--version")).Output
    $pythonGitVersion = (Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("git", "--version")).Output
    $pythonRgVersion = (Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("rg", "--version")).Output
    Assert-Condition -Condition ($pythonVersion -match "^Python 3\.11\.") -Message "Python 3.11 is available ($pythonVersion)"
    Assert-Condition -Condition ($pytestVersion -match "pytest 9\.1\.1") -Message "pytest 9.1.1 is available"
    Assert-Condition -Condition ($pythonGitVersion -match "^git version ") -Message "Git is available in the Python image"
    Assert-Condition -Condition ($pythonRgVersion -match "^ripgrep ") -Message "ripgrep is available in the Python image"

    $javaVersion = (Invoke-ContainerExec -ContainerId $mavenId -CommandArgs @("java", "-version")).Output
    $mavenVersion = (Invoke-ContainerExec -ContainerId $mavenId -CommandArgs @("mvn", "--version")).Output
    $mavenGitVersion = (Invoke-ContainerExec -ContainerId $mavenId -CommandArgs @("git", "--version")).Output
    $mavenRgVersion = (Invoke-ContainerExec -ContainerId $mavenId -CommandArgs @("rg", "--version")).Output
    Assert-Condition -Condition ($javaVersion -match 'version "21\.') -Message "Java 21 is available"
    Assert-Condition -Condition ($mavenVersion -match "Apache Maven 3\.9\.") -Message "Maven 3.9 is available"
    Assert-Condition -Condition ($mavenGitVersion -match "^git version ") -Message "Git is available in the Maven image"
    Assert-Condition -Condition ($mavenRgVersion -match "^ripgrep ") -Message "ripgrep is available in the Maven image"

    foreach ($containerId in @($pythonId, $mavenId)) {
        $uid = (Invoke-ContainerExec -ContainerId $containerId -CommandArgs @("id", "-u")).Output
        $gid = (Invoke-ContainerExec -ContainerId $containerId -CommandArgs @("id", "-g")).Output
        Assert-Condition -Condition ($uid -eq "10001") -Message "container $($containerId.Substring(0, 12)) runs as UID 10001"
        Assert-Condition -Condition ($gid -eq "10001") -Message "container $($containerId.Substring(0, 12)) runs as GID 10001"
        $dockerCli = Invoke-ContainerExec -ContainerId $containerId -CommandArgs @("sh", "-c", "command -v docker") -AllowedExitCodes @(0, 1, 127)
        Assert-Condition -Condition ($dockerCli.ExitCode -ne 0) -Message "Docker CLI is absent from container $($containerId.Substring(0, 12))"
        $userSsh = Invoke-ContainerExec -ContainerId $containerId -CommandArgs @("sh", "-c", "test -e /home/repo-agent/.ssh") -AllowedExitCodes @(0, 1)
        Assert-Condition -Condition ($userSsh.ExitCode -ne 0) -Message "user SSH configuration is absent from container $($containerId.Substring(0, 12))"
    }

    $mountCode = @'
from pathlib import Path

expected = Path('/input/proof.txt').read_text(encoding='utf-8')
if not expected.startswith('host-proof-'):
    raise SystemExit('read-only input content mismatch')
try:
    Path('/input/proof.txt').write_text('tampered', encoding='utf-8')
except OSError:
    pass
else:
    raise SystemExit('read-only input was writable')
Path('/workspace/container-write.txt').write_text('workspace-write-ok', encoding='utf-8')
'@
    Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("python", "-c", $mountCode) | Out-Null
    $proofHashAfter = (Get-FileHash -LiteralPath $proofPath -Algorithm SHA256).Hash
    Assert-Condition -Condition ($proofHashAfter -eq $proofHashBefore) -Message "read-only D-drive mount cannot modify the host file"
    $workspaceWritePath = Join-Path $pythonWorkspace "container-write.txt"
    Assert-Condition -Condition ([IO.File]::ReadAllText($workspaceWritePath) -eq "workspace-write-ok") -Message "read-write D-drive mount persists container output"

    $filesystemCode = @'
from pathlib import Path

for blocked in (Path('/home/repo-agent/rootfs-write.txt'), Path('/etc/rootfs-write.txt')):
    try:
        blocked.write_text('must-fail', encoding='utf-8')
    except OSError:
        pass
    else:
        raise SystemExit(f'read-only rootfs allowed write: {blocked}')
Path('/tmp/tmpfs-write.txt').write_text('tmp-ok', encoding='utf-8')
Path('/workspace/workspace-write-2.txt').write_text('workspace-ok', encoding='utf-8')
'@
    Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("python", "-c", $filesystemCode) | Out-Null
    Assert-Condition -Condition (Test-Path -LiteralPath (Join-Path $pythonWorkspace "workspace-write-2.txt")) -Message "only tmpfs and explicit workspace writes succeed"

    $interfaces = (Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("sh", "-c", "ls -1 /sys/class/net | sort") ).Output
    Assert-Condition -Condition ($interfaces -eq "lo") -Message "network-none exposes only the loopback interface"
    $dnsAttempt = Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("getent", "hosts", "example.com") -AllowedExitCodes @(0, 1, 2)
    Assert-Condition -Condition ($dnsAttempt.ExitCode -ne 0) -Message "DNS resolution fails with network=none"
    $outboundCode = "import socket; s=socket.socket(); s.settimeout(2); s.connect(('1.1.1.1', 443))"
    $outboundAttempt = Invoke-ContainerExec -ContainerId $pythonId -CommandArgs @("python", "-c", $outboundCode) -AllowedExitCodes @(0, 1)
    Assert-Condition -Condition ($outboundAttempt.ExitCode -ne 0) -Message "outbound access fails with network=none"

    $timeoutTarget = New-SandboxContainer -Name "repo-agent-timeout-$runId" -CaseName "timeout-target" -Image $pythonImage -WorkspacePath $pythonWorkspace -CachePath $pythonCache -CacheTarget "/cache"
    $timeoutControl = New-SandboxContainer -Name "repo-agent-control-$runId" -CaseName "timeout-control" -Image $pythonImage -WorkspacePath $pythonWorkspace -CachePath $pythonCache -CacheTarget "/cache"
    Start-SandboxContainer -ContainerId $timeoutTarget
    Start-SandboxContainer -ContainerId $timeoutControl
    $timeoutResult = Stop-ContainerAfterTimeout -ContainerId $timeoutTarget -TimeoutSeconds 1
    Assert-Condition -Condition ([bool]$timeoutResult.TimedOut) -Message "long-running test reaches its timeout"
    $targetState = Get-ContainerRunningState -ContainerId $timeoutTarget
    Assert-Condition -Condition ($targetState.ExitCode -ne 0) -Message "timed-out container is removed by exact ID"
    $controlState = Get-ContainerRunningState -ContainerId $timeoutControl
    Assert-Condition -Condition ($controlState.ExitCode -eq 0 -and $controlState.Output -eq "true") -Message "timeout cleanup leaves the unrelated control container running"

    $policyFiles = @(
        (Join-Path $projectRoot "README.md"),
        (Join-Path $projectRoot "docker/python/Dockerfile"),
        (Join-Path $projectRoot "docker/maven/Dockerfile")
    ) + @(Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter "*.ps1" | ForEach-Object { $_.FullName })
    $forbiddenText = (($policyFiles | ForEach-Object { [IO.File]::ReadAllText($_) }) -join "`n")
    Assert-Condition -Condition ($forbiddenText -notmatch "docker\s+system\s+prune") -Message "project scripts do not invoke global Docker prune"
}
catch {
    $pendingError = $_
}
finally {
    for ($index = $createdContainers.Count - 1; $index -ge 0; $index--) {
        try {
            Remove-ExactContainer -ContainerId $createdContainers[$index]
        }
        catch {
            if ($null -eq $cleanupError) {
                $cleanupError = $_
            }
        }
    }

    try {
        $labeled = Invoke-DockerCapture -CommandArgs @("container", "ls", "--all", "--quiet", "--filter", "label=$runLabel")
        $leftoverIds = @($labeled.Output -split "`r?`n" | Where-Object { $_ })
        foreach ($leftoverId in $leftoverIds) {
            Remove-ExactContainer -ContainerId $leftoverId
        }
        $afterCleanup = Invoke-DockerCapture -CommandArgs @("container", "ls", "--all", "--quiet", "--filter", "label=$runLabel")
        if (-not [string]::IsNullOrWhiteSpace($afterCleanup.Output) -and $null -eq $cleanupError) {
            $cleanupError = New-Object InvalidOperationException("Run-labeled containers remain after cleanup: $($afterCleanup.Output)")
        }
    }
    catch {
        if ($null -eq $cleanupError) {
            $cleanupError = $_
        }
    }

    if (Test-Path -LiteralPath $tempRoot) {
        try {
            $cleanupRoot = [IO.Path]::GetFullPath($tempRoot)
            $cleanupBase = [IO.Path]::GetFullPath($tempBase).TrimEnd("\") + "\"
            $safeLeaf = Split-Path -Leaf $cleanupRoot
            if (-not $cleanupRoot.StartsWith($cleanupBase, [StringComparison]::OrdinalIgnoreCase) -or $safeLeaf -notmatch "^docker-verify-[0-9a-f]{32}$") {
                throw "Refusing to delete unexpected temporary path: $cleanupRoot"
            }
            Remove-Item -LiteralPath $cleanupRoot -Recurse -Force
        }
        catch {
            if ($null -eq $cleanupError) {
                $cleanupError = $_
            }
        }
    }
}

if ($null -ne $pendingError) {
    throw $pendingError
}
if ($null -ne $cleanupError) {
    throw $cleanupError
}

Assert-Condition -Condition $true -Message "no run-labeled verification containers remain"
Write-Host "Docker prerequisite verification completed successfully."
