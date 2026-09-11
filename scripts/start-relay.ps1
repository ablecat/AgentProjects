[CmdletBinding()]
param(
    [string]$Repository = "",
    [string]$Model = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$OpenBrowser,
    [switch]$Reconfigure,
    [switch]$ConfigureOnly,
    [switch]$Verify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = Split-Path -Parent $scriptDirectory
}

$relayBaseUrl = "https://thz10.airucas.com/v1"
$configFormat = "repo-maintainer-agent-relay"
$configEnvelopeVersion = 1
$configSchemaVersion = 1
$localAppData = [Environment]::GetFolderPath(
    [Environment+SpecialFolder]::LocalApplicationData
)
$configRoot = Join-Path $localAppData "RepoMaintainerAgent"
$configDirectory = Join-Path $configRoot "config"
$configPath = Join-Path $configDirectory "relay.json"
$environmentNames = @(
    "REPO_AGENT_API_KEY",
    "REPO_AGENT_BASE_URL",
    "REPO_AGENT_MODEL"
)
$previousEnvironment = @{}

function Get-PythonInvocation {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        return @($launcher.Source, "-3.11")
    }

    $launcher = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $launcher) {
        throw "Python 3.11 is required, but neither 'py' nor 'python' is available."
    }
    return @($launcher.Source)
}

function Set-ProcessApiKey {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureKey)

    $keyPointer = [IntPtr]::Zero
    $plainKey = $null
    try {
        $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
        $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
        Assert-ApiKey -ApiKey $plainKey
        [Environment]::SetEnvironmentVariable(
            "REPO_AGENT_API_KEY",
            $plainKey,
            [EnvironmentVariableTarget]::Process
        )
    }
    finally {
        $plainKey = $null
        if ($keyPointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
        }
    }
}

function Assert-ApiKey {
    param([Parameter(Mandatory = $true)][string]$ApiKey)

    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        throw "The API key cannot be empty."
    }
    if ($ApiKey.Length -gt 8192 -or $ApiKey -match "[\u0000-\u001f\u007f]") {
        throw "The API key contains invalid characters or is too long."
    }
}

function Assert-ModelId {
    param([Parameter(Mandatory = $true)][string]$ModelId)

    if (
        [string]::IsNullOrWhiteSpace($ModelId) -or
        $ModelId.Length -gt 512 -or
        $ModelId -match "[\u0000-\u001f\u007f]"
    ) {
        throw "The model ID is invalid."
    }
}

function ConvertTo-PlainText {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)

    $valuePointer = [IntPtr]::Zero
    try {
        $valuePointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
            $SecureValue
        )
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($valuePointer)
    }
    finally {
        if ($valuePointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($valuePointer)
        }
    }
}

function Import-CurrentSecurityModule {
    $modulePath = Join-Path $PSHOME (
        "Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
    )
    if (-not (Test-Path -LiteralPath $modulePath -PathType Leaf)) {
        throw "The current PowerShell security module is unavailable."
    }
    $loaded = Get-Module -Name "Microsoft.PowerShell.Security" |
        Where-Object { $_.Path -ceq $modulePath }
    if ($null -eq $loaded) {
        Import-Module -Name $modulePath -ErrorAction Stop
    }
}

function Assert-NotReparsePoint {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw "$Description cannot be a symbolic link, junction, or reparse point."
    }
}

function Assert-ExactProperties {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if ($Value -isnot [pscustomobject]) {
        throw "$Description must be a JSON object."
    }
    $actual = @($Value.PSObject.Properties.Name)
    foreach ($name in $Expected) {
        if ($actual -cnotcontains $name) {
            throw "$Description is missing a required field."
        }
    }
    foreach ($name in $actual) {
        if ($Expected -cnotcontains $name) {
            throw "$Description contains an unknown field."
        }
    }
}

function Move-FileAtomically {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if ($null -eq ("RepoMaintainerAgentRelayNativeMethods" -as [type])) {
        Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;

public static class RepoMaintainerAgentRelayNativeMethods
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool MoveFileEx(
        string existingFileName,
        string newFileName,
        uint flags
    );
}
"@
    }

    $replaceExisting = [uint32]0x1
    $writeThrough = [uint32]0x8
    $moved = [RepoMaintainerAgentRelayNativeMethods]::MoveFileEx(
        $Source,
        $Destination,
        ($replaceExisting -bor $writeThrough)
    )
    if (-not $moved) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw [ComponentModel.Win32Exception]::new(
            $errorCode,
            "Could not atomically replace the relay configuration."
        )
    }
}

function Get-RelayConfigFingerprint {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return "missing"
    }
    Assert-NotReparsePoint -Path $Path -Description "Relay configuration file"
    $stream = [IO.FileStream]::new(
        $Path,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash($stream)
        return [BitConverter]::ToString($digest).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Write-RelayConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ApiKey,
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][string]$ModelId,
        [Parameter(Mandatory = $true)][string]$ExpectedFingerprint
    )

    Import-CurrentSecurityModule
    Assert-ApiKey -ApiKey $ApiKey
    Assert-ModelId -ModelId $ModelId
    if ($BaseUrl -cne $relayBaseUrl) {
        throw "The relay Base URL is invalid."
    }

    $directory = Split-Path -Parent $Path
    $root = Split-Path -Parent $directory
    Assert-NotReparsePoint -Path $root -Description "Relay configuration root"
    [IO.Directory]::CreateDirectory($directory) > $null
    Assert-NotReparsePoint -Path $directory -Description "Relay configuration directory"
    Assert-NotReparsePoint -Path $Path -Description "Relay configuration file"

    $innerJson = [ordered]@{
        schema_version = $configSchemaVersion
        provider = "openai-compatible"
        api_key = $ApiKey
        base_url = $BaseUrl
        model = $ModelId
        verified_at = [DateTimeOffset]::UtcNow.ToString("o")
    } | ConvertTo-Json -Compress
    $innerSecure = Microsoft.PowerShell.Security\ConvertTo-SecureString `
        $innerJson `
        -AsPlainText `
        -Force
    try {
        $ciphertext = Microsoft.PowerShell.Security\ConvertFrom-SecureString `
            $innerSecure
    }
    finally {
        $innerSecure.Dispose()
        $innerJson = $null
    }
    if ([string]::IsNullOrWhiteSpace($ciphertext)) {
        throw "Windows DPAPI did not produce an encrypted configuration."
    }

    $outerJson = [ordered]@{
        format = $configFormat
        envelope_version = $configEnvelopeVersion
        protection = "windows-dpapi-current-user"
        ciphertext = $ciphertext
    } | ConvertTo-Json

    $mutex = [Threading.Mutex]::new(
        $false,
        "Local\RepoMaintainerAgentRelayConfigV1"
    )
    $lockAcquired = $false
    $tempPath = Join-Path $directory ([IO.Path]::GetRandomFileName())
    try {
        try {
            $lockAcquired = $mutex.WaitOne([TimeSpan]::FromSeconds(10))
        }
        catch [Threading.AbandonedMutexException] {
            $lockAcquired = $true
        }
        if (-not $lockAcquired) {
            throw "Timed out waiting for the relay configuration lock."
        }
        $currentFingerprint = Get-RelayConfigFingerprint -Path $Path
        if ($currentFingerprint -cne $ExpectedFingerprint) {
            throw "Relay configuration changed during verification; run -Reconfigure again."
        }

        $encoding = [Text.UTF8Encoding]::new($false)
        $stream = [IO.FileStream]::new(
            $tempPath,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None
        )
        $writer = $null
        try {
            $writer = [IO.StreamWriter]::new($stream, $encoding)
            $writer.Write($outerJson)
            $writer.Flush()
            $stream.Flush($true)
        }
        finally {
            if ($null -ne $writer) {
                $writer.Dispose()
            }
            else {
                $stream.Dispose()
            }
        }

        Assert-NotReparsePoint -Path $Path -Description "Relay configuration file"
        Move-FileAtomically -Source $tempPath -Destination $Path
        $tempPath = $null
    }
    finally {
        if ($null -ne $tempPath -and (Test-Path -LiteralPath $tempPath)) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
        if ($lockAcquired) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
        $outerJson = $null
        $ciphertext = $null
    }
}

function Read-RelayConfig {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        Import-CurrentSecurityModule
        $directory = Split-Path -Parent $Path
        $root = Split-Path -Parent $directory
        Assert-NotReparsePoint -Path $root -Description "Relay configuration root"
        Assert-NotReparsePoint -Path $directory -Description "Relay configuration directory"
        Assert-NotReparsePoint -Path $Path -Description "Relay configuration file"

        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        if ($item.Length -le 0 -or $item.Length -gt 64KB) {
            throw "Relay configuration file size is invalid."
        }
        $outer = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 |
            ConvertFrom-Json
        Assert-ExactProperties -Value $outer -Expected @(
            "format",
            "envelope_version",
            "protection",
            "ciphertext"
        ) -Description "Relay configuration envelope"
        if (
            $outer.format -cne $configFormat -or
            $outer.envelope_version -ne $configEnvelopeVersion -or
            $outer.protection -cne "windows-dpapi-current-user" -or
            $outer.ciphertext -isnot [string] -or
            [string]::IsNullOrWhiteSpace($outer.ciphertext) -or
            $outer.ciphertext.Length -gt 60KB -or
            $outer.ciphertext -match "[\u0000-\u001f\u007f]"
        ) {
            throw "Relay configuration envelope is invalid."
        }

        $innerSecure = Microsoft.PowerShell.Security\ConvertTo-SecureString `
            $outer.ciphertext `
            -ErrorAction Stop
        try {
            $innerJson = ConvertTo-PlainText -SecureValue $innerSecure
        }
        finally {
            $innerSecure.Dispose()
        }
        if ($innerJson.Length -gt 16KB) {
            throw "Decrypted relay configuration is too large."
        }
        try {
            $inner = $innerJson | ConvertFrom-Json
        }
        finally {
            $innerJson = $null
        }
        Assert-ExactProperties -Value $inner -Expected @(
            "schema_version",
            "provider",
            "api_key",
            "base_url",
            "model",
            "verified_at"
        ) -Description "Decrypted relay configuration"
        if (
            $inner.schema_version -ne $configSchemaVersion -or
            $inner.provider -cne "openai-compatible" -or
            $inner.base_url -cne $relayBaseUrl
        ) {
            throw "Decrypted relay configuration has invalid metadata."
        }
        Assert-ApiKey -ApiKey $inner.api_key
        Assert-ModelId -ModelId $inner.model
        $verifiedAt = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse($inner.verified_at, [ref]$verifiedAt)) {
            throw "Decrypted relay configuration has an invalid verification time."
        }
        return [pscustomobject]@{
            ApiKey = $inner.api_key
            BaseUrl = $inner.base_url
            Model = $inner.model
            VerifiedAt = $verifiedAt
        }
    }
    catch {
        throw "Saved relay configuration is invalid or cannot be decrypted. Run start-relay.ps1 -Reconfigure."
    }
}

function Get-RelayModelIds {
    param([Parameter(Mandatory = $true)][string]$BaseUrl)

    Add-Type -AssemblyName System.Net.Http
    $handler = [Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(30)
    $client.MaxResponseContentBufferSize = 2MB
    $request = [Net.Http.HttpRequestMessage]::new(
        [Net.Http.HttpMethod]::Get,
        "$BaseUrl/models"
    )
    $response = $null
    try {
        $request.Headers.Authorization =
            [Net.Http.Headers.AuthenticationHeaderValue]::new(
                "Bearer",
                $env:REPO_AGENT_API_KEY
            )
        $request.Headers.UserAgent.ParseAdd("repo-maintainer-agent/0.1")
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        $statusCode = [int]$response.StatusCode
        if ($statusCode -ge 300 -and $statusCode -lt 400) {
            throw "Model discovery refused an HTTP redirect ($statusCode)."
        }
        if (-not $response.IsSuccessStatusCode) {
            throw "Model discovery failed with HTTP $statusCode."
        }
        $contentLength = $response.Content.Headers.ContentLength
        if ($null -ne $contentLength -and [long]$contentLength -gt 2MB) {
            throw "Model discovery response exceeded 2 MiB."
        }

        $json = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if ($json.Length -gt 2MB) {
            throw "Model discovery response exceeded 2 MiB."
        }
        $payload = $json | ConvertFrom-Json
        $validatedIds = [Collections.Generic.List[string]]::new()
        foreach ($entry in @($payload.data)) {
            $id = $entry.id
            if ($id -isnot [string] -or [string]::IsNullOrWhiteSpace($id)) {
                throw "The relay returned an invalid model ID."
            }
            if ($id.Length -gt 512 -or $id -match "[\u0000-\u001f\u007f]") {
                throw "The relay returned an unsafe model ID."
            }
            $validatedIds.Add($id)
        }
        $modelIds = @($validatedIds | Sort-Object -Unique)
        if ($modelIds.Count -eq 0) {
            throw "The relay returned no model IDs."
        }
        return $modelIds
    }
    finally {
        if ($null -ne $response) {
            $response.Dispose()
        }
        $request.Dispose()
        $client.Dispose()
        $handler.Dispose()
    }
}

function Select-RelayModel {
    param([Parameter(Mandatory = $true)][string[]]$ModelIds)

    Write-Host "Available relay models:"
    for ($index = 0; $index -lt $ModelIds.Count; $index++) {
        Write-Host ("  [{0}] {1}" -f ($index + 1), $ModelIds[$index])
    }

    while ($true) {
        $selection = (Read-Host "Select a model by number or exact ID").Trim()
        $number = 0
        if (
            [int]::TryParse($selection, [ref]$number) -and
            $number -ge 1 -and
            $number -le $ModelIds.Count
        ) {
            return $ModelIds[$number - 1]
        }
        $exact = @($ModelIds | Where-Object { $_ -ceq $selection })
        if ($exact.Count -eq 1) {
            return $exact[0]
        }
        Write-Warning "Choose a displayed number or enter an exact model ID."
    }
}

function Get-CheckedPythonInvocation {
    $pythonInvocation = @(Get-PythonInvocation)
    $pythonExecutable = $pythonInvocation[0]
    $pythonPrefix = @($pythonInvocation | Select-Object -Skip 1)
    $importArguments = @($pythonPrefix + @("-c", "import repo_agent"))
    & $pythonExecutable @importArguments
    if ($LASTEXITCODE -ne 0) {
        throw "repo_agent is not importable by the selected Python interpreter."
    }
    return $pythonInvocation
}

function Invoke-RelayDoctor {
    param([Parameter(Mandatory = $true)][object[]]$PythonInvocation)

    $pythonExecutable = $PythonInvocation[0]
    $pythonPrefix = @($PythonInvocation | Select-Object -Skip 1)
    Write-Host "Verifying the relay with a native tool-call handshake..."
    $doctorArguments = @(
        $pythonPrefix +
        @("-B", "-m", "repo_agent", "doctor", "--allow-remote-model")
    )
    & $pythonExecutable @doctorArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Relay verification failed; the UI was not started."
    }
}

function Set-RelayEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$ApiKey,
        [Parameter(Mandatory = $true)][string]$ModelId
    )

    Assert-ApiKey -ApiKey $ApiKey
    Assert-ModelId -ModelId $ModelId
    [Environment]::SetEnvironmentVariable(
        "REPO_AGENT_API_KEY",
        $ApiKey,
        [EnvironmentVariableTarget]::Process
    )
    [Environment]::SetEnvironmentVariable(
        "REPO_AGENT_BASE_URL",
        $relayBaseUrl,
        [EnvironmentVariableTarget]::Process
    )
    [Environment]::SetEnvironmentVariable(
        "REPO_AGENT_MODEL",
        $ModelId,
        [EnvironmentVariableTarget]::Process
    )
}

function Invoke-RelayLauncher {
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        throw "LOCALAPPDATA is unavailable; encrypted relay configuration cannot be used."
    }
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "Saved relay configuration requires Windows DPAPI."
    }

    foreach ($name in $environmentNames) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable(
            $name,
            [EnvironmentVariableTarget]::Process
        )
    }

    $loadedConfig = $null
    $promptedKey = $null
    $pythonInvocation = $null
    try {
        $repositoryPath = $null
        if (-not $ConfigureOnly) {
            $repositoryPath = (
                Resolve-Path -LiteralPath $Repository -ErrorAction Stop
            ).Path
            if (-not (Test-Path -LiteralPath $repositoryPath -PathType Container)) {
                throw "Repository is not a directory: $repositoryPath"
            }
            $listener = Get-NetTCPConnection `
                -State Listen `
                -LocalPort $Port `
                -ErrorAction SilentlyContinue
            if ($null -ne $listener) {
                $ownerIds = @(
                    $listener | Select-Object -ExpandProperty OwningProcess -Unique
                )
                throw "Port $Port is already in use by PID(s): $($ownerIds -join ', ')."
            }
        }

        $initialFingerprint = Get-RelayConfigFingerprint -Path $configPath
        $configurationExists = $initialFingerprint -cne "missing"
        $shouldConfigure = $Reconfigure -or -not $configurationExists
        if (
            -not $shouldConfigure -and
            -not [string]::IsNullOrWhiteSpace($Model)
        ) {
            throw "-Model can be used only for first-time setup or with -Reconfigure."
        }

        if ($shouldConfigure) {
            $promptedKey = Read-Host (
                "Relay API key (masked as *; DPAPI-encrypted when saved)"
            ) -AsSecureString
            Set-ProcessApiKey -SecureKey $promptedKey
            $promptedKey.Dispose()
            $promptedKey = $null
            [Environment]::SetEnvironmentVariable(
                "REPO_AGENT_BASE_URL",
                $relayBaseUrl,
                [EnvironmentVariableTarget]::Process
            )

            $modelIds = @(Get-RelayModelIds -BaseUrl $relayBaseUrl)
            $selectedModel = $Model.Trim()
            if ([string]::IsNullOrWhiteSpace($selectedModel)) {
                $selectedModel = Select-RelayModel -ModelIds $modelIds
            }
            else {
                Assert-ModelId -ModelId $selectedModel
                if ($modelIds -cnotcontains $selectedModel) {
                    throw "The requested model ID was not returned by the relay."
                }
            }
            [Environment]::SetEnvironmentVariable(
                "REPO_AGENT_MODEL",
                $selectedModel,
                [EnvironmentVariableTarget]::Process
            )

            $pythonInvocation = @(Get-CheckedPythonInvocation)
            Invoke-RelayDoctor -PythonInvocation $pythonInvocation
            Write-RelayConfig `
                -Path $configPath `
                -ApiKey $env:REPO_AGENT_API_KEY `
                -BaseUrl $relayBaseUrl `
                -ModelId $selectedModel `
                -ExpectedFingerprint $initialFingerprint
            Write-Host "Encrypted relay configuration saved: $configPath"
        }
        else {
            $loadedConfig = Read-RelayConfig -Path $configPath
            $loadedModel = $loadedConfig.Model
            Set-RelayEnvironment `
                -ApiKey $loadedConfig.ApiKey `
                -ModelId $loadedModel
            $loadedConfig.ApiKey = $null
            $loadedConfig = $null
            Write-Host (
                "Loaded saved relay configuration for model: {0}" -f
                $loadedModel
            )
            if ($Verify) {
                $pythonInvocation = @(Get-CheckedPythonInvocation)
                Invoke-RelayDoctor -PythonInvocation $pythonInvocation
            }
        }

        if ($ConfigureOnly) {
            Write-Host "Configuration ready. The visual workspace was not started."
            return
        }

        if ($null -eq $pythonInvocation) {
            $pythonInvocation = @(Get-CheckedPythonInvocation)
        }
        $pythonExecutable = $pythonInvocation[0]
        $pythonPrefix = @($pythonInvocation | Select-Object -Skip 1)

        Write-Host "Starting the visual workspace on port $Port..."
        $serveArguments = @(
            $pythonPrefix +
            @(
                "-B",
                "-m",
                "repo_agent",
                "serve",
                "--repo",
                $repositoryPath,
                "--port",
                $Port.ToString()
            )
        )
        if ($OpenBrowser) {
            $serveArguments += "--open"
        }
        & $pythonExecutable @serveArguments
        if ($LASTEXITCODE -ne 0) {
            throw "The visual workspace exited with code $LASTEXITCODE."
        }
    }
    finally {
        if ($null -ne $promptedKey) {
            $promptedKey.Dispose()
        }
        $loadedConfig = $null
        foreach ($name in $environmentNames) {
            [Environment]::SetEnvironmentVariable(
                $name,
                $previousEnvironment[$name],
                [EnvironmentVariableTarget]::Process
            )
        }
        Write-Host "Temporary Repo Agent model environment was restored."
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Invoke-RelayLauncher
}
