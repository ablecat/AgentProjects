[CmdletBinding()]
param(
    [string]$Repository = "",
    [string]$Model = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$OpenBrowser,
    [switch]$Reconfigure,
    [switch]$ConfigureOnly,
    [switch]$Verify,
    [switch]$Evaluate,
    [switch]$FormalEvaluate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$relayInvocationParameterNames = @($PSBoundParameters.Keys)

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = Split-Path -Parent $scriptDirectory
}

$relayBaseUrl = "https://api.deepseek.com"
$defaultModelId = "deepseek-v4-pro"
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
        $plainKey = $plainKey.Trim()
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
    if ($ApiKey.Length -lt 8) {
        throw "The API key is too short."
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

    $ApiKey = $ApiKey.Trim()
    Import-CurrentSecurityModule
    Assert-ApiKey -ApiKey $ApiKey
    Assert-ModelId -ModelId $ModelId
    if ($BaseUrl -cne $relayBaseUrl) {
        throw "The DeepSeek Base URL is invalid."
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
        $canonicalApiKey = $inner.api_key.Trim()
        Assert-ApiKey -ApiKey $canonicalApiKey
        Assert-ModelId -ModelId $inner.model
        $verifiedAt = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse($inner.verified_at, [ref]$verifiedAt)) {
            throw "Decrypted relay configuration has an invalid verification time."
        }
        return [pscustomobject]@{
            ApiKey = $canonicalApiKey
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
                throw "DeepSeek returned an invalid model ID."
            }
            if ($id.Length -gt 512 -or $id -match "[\u0000-\u001f\u007f]") {
                throw "DeepSeek returned an unsafe model ID."
            }
            $validatedIds.Add($id)
        }
        $modelIds = @($validatedIds | Sort-Object -Unique)
        if ($modelIds.Count -eq 0) {
            throw "DeepSeek returned no model IDs."
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

    Write-Host "Available DeepSeek models:"
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
    Write-Host "Verifying DeepSeek with a native tool-call handshake..."
    $doctorArguments = @(
        $pythonPrefix +
        @("-B", "-m", "repo_agent", "doctor", "--allow-remote-model")
    )
    & $pythonExecutable @doctorArguments
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSeek verification failed; the UI was not started."
    }
}

function Invoke-RelayEvaluation {
    param([Parameter(Mandatory = $true)][object[]]$PythonInvocation)

    $pythonExecutable = $PythonInvocation[0]
    $pythonPrefix = @($PythonInvocation | Select-Object -Skip 1)
    $projectRoot = Split-Path -Parent $scriptDirectory
    $benchmarkDirectory = Join-Path (
        $projectRoot
    ) "benchmarks\tasks\python\development"
    if (-not (Test-Path -LiteralPath $benchmarkDirectory -PathType Container)) {
        throw "The fixed development benchmark suite is unavailable."
    }
    $runId = "{0}-{1}" -f (
        [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    ), ([Guid]::NewGuid().ToString("N"))
    $outputDirectory = Join-Path (
        (Join-Path $projectRoot "evaluation-results\development-v1")
    ) $runId
    $evaluationArguments = @(
        $pythonPrefix +
        @(
            "-B",
            "-m",
            "repo_agent",
            "eval",
            "--variant",
            "full",
            "--benchmark-dir",
            $benchmarkDirectory,
            "--output-dir",
            $outputDirectory,
            "--execute",
            "--allow-remote-model",
            "--task-timeout-seconds",
            "1200",
            "--max-output-bytes",
            "65536",
            "--format",
            "json"
        )
    )
    Write-Host "Running the fixed four-task development evaluation..."
    Write-Host "Evaluation output: $outputDirectory"
    & $pythonExecutable @evaluationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Development evaluation failed with code $LASTEXITCODE."
    }
    Assert-DevelopmentEvaluationAcceptance `
        -SummaryPath (Get-EvaluationSummaryPath `
            -OutputDirectory $outputDirectory `
            -SuiteId "repo-agent-python-development-v1") `
        -ExpectedModelId $env:REPO_AGENT_MODEL
}

function Get-EvaluationSummaryPath {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string]$SuiteId
    )

    $variantDirectory = Join-Path (
        (Join-Path $OutputDirectory $SuiteId)
    ) "full"
    return Join-Path (Join-Path $variantDirectory "trial-1") "summary.json"
}

function Assert-DevelopmentEvaluationAcceptance {
    param(
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][string]$ExpectedModelId
    )

    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Development evaluation summary is missing."
    }
    try {
        $report = Get-Content `
            -LiteralPath $SummaryPath `
            -Raw `
            -Encoding UTF8 | ConvertFrom-Json
        $taskCount = [Convert]::ToInt32($report.task_count)
        $solvedTaskCount = [Convert]::ToInt32($report.solved_task_count)
        if ($null -eq $report.actionable_tool_error_rate) {
            throw "missing actionable tool error rate"
        }
        $actionableRate = [Convert]::ToDouble(
            $report.actionable_tool_error_rate,
            [Globalization.CultureInfo]::InvariantCulture
        )
        $tasks = @($report.tasks)
        $taskIds = @($tasks | ForEach-Object { [string]$_.task_id })
        $rowSolvedTaskCount = @(
            $tasks | Where-Object { $_.solved -eq $true }
        ).Count
        $invalidTaskRows = @(
            $tasks | Where-Object {
                $_.status -cne "completed" -or
                $_.model -cne $ExpectedModelId -or
                $_.language -cne "python" -or
                $_.solved -isnot [bool]
            }
        )
    }
    catch {
        throw "Development evaluation summary is invalid."
    }

    if (
        $report.status -cne "completed" -or
        $report.suite_id -cne "repo-agent-python-development-v1" -or
        $report.variant -cne "full" -or
        $report.model -cne $ExpectedModelId -or
        [Convert]::ToInt32($report.trial) -ne 1 -or
        $taskCount -ne 4 -or
        $solvedTaskCount -lt 0 -or
        $solvedTaskCount -gt $taskCount -or
        $tasks.Count -ne 4
    ) {
        throw "Development evaluation did not complete the fixed four-task full suite."
    }
    $expectedTaskIds = @(
        "py-development-001",
        "py-development-002",
        "py-development-003",
        "py-development-004"
    )
    $taskIdDifferences = @(
        Compare-Object `
            -ReferenceObject $expectedTaskIds `
            -DifferenceObject $taskIds `
            -CaseSensitive
    )
    if ($taskIdDifferences.Count -ne 0) {
        throw "Development evaluation task identity does not match the fixed suite."
    }
    if (
        $invalidTaskRows.Count -ne 0 -or
        $rowSolvedTaskCount -ne $solvedTaskCount
    ) {
        throw "Development evaluation task rows do not match its aggregate totals."
    }
    if ($solvedTaskCount -lt 3) {
        throw "Development evaluation acceptance failed: fewer than 3 of 4 tasks solved."
    }
    if (@($tasks | Where-Object { $_.budget_passed -ne $true }).Count -ne 0) {
        throw "Development evaluation acceptance failed: a task exceeded its budget."
    }
    if (
        [Double]::IsNaN($actionableRate) -or
        [Double]::IsInfinity($actionableRate) -or
        $actionableRate -lt 0.0 -or
        $actionableRate -gt 0.05
    ) {
        throw "Development evaluation acceptance failed: actionable tool error rate exceeds 5%."
    }
    Write-Host (
        "Development acceptance passed: {0}/4 solved; actionable tool error rate {1:P2}." -f
        $solvedTaskCount,
        $actionableRate
    )
}

function Invoke-RelayFormalEvaluation {
    param([Parameter(Mandatory = $true)][object[]]$PythonInvocation)

    $pythonExecutable = $PythonInvocation[0]
    $pythonPrefix = @($PythonInvocation | Select-Object -Skip 1)
    $projectRoot = Split-Path -Parent $scriptDirectory
    $benchmarkDirectory = Join-Path $projectRoot "benchmarks"
    if (-not (Test-Path -LiteralPath $benchmarkDirectory -PathType Container)) {
        throw "The fixed formal benchmark suite is unavailable."
    }
    $runId = "{0}-{1}" -f (
        [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    ), ([Guid]::NewGuid().ToString("N"))
    $outputDirectory = Join-Path (
        (Join-Path $projectRoot (
            "evaluation-results\formal-v1-regression-acceptance"
        ))
    ) $runId
    if (Test-Path -LiteralPath $outputDirectory) {
        throw "The fresh formal acceptance output directory already exists."
    }
    $evaluationArguments = @(
        $pythonPrefix +
        @(
            "-B",
            "-m",
            "repo_agent",
            "eval",
            "--variant",
            "full",
            "--trial",
            "1",
            "--benchmark-dir",
            $benchmarkDirectory,
            "--output-dir",
            $outputDirectory,
            "--execute",
            "--allow-remote-model",
            "--allow-bootstrap",
            "--task-timeout-seconds",
            "1200",
            "--max-output-bytes",
            "65536",
            "--format",
            "json"
        )
    )
    Write-Host "Running the fixed 12-task formal full trial-1 acceptance..."
    Write-Host "Formal acceptance output: $outputDirectory"
    & $pythonExecutable @evaluationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal acceptance evaluation failed with code $LASTEXITCODE."
    }
    Assert-FormalEvaluationAcceptance `
        -SummaryPath (Get-EvaluationSummaryPath `
            -OutputDirectory $outputDirectory `
            -SuiteId "repo-agent-python-java-day3-v1") `
        -ExpectedModelId $env:REPO_AGENT_MODEL
}

function Assert-FormalEvaluationAcceptance {
    param(
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][string]$ExpectedModelId
    )

    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Formal acceptance summary is missing."
    }
    Assert-NotReparsePoint `
        -Path $SummaryPath `
        -Description "Formal acceptance summary"
    try {
        $report = Get-Content `
            -LiteralPath $SummaryPath `
            -Raw `
            -Encoding UTF8 | ConvertFrom-Json
        $taskCount = [Convert]::ToInt32($report.task_count)
        $completedTaskCount = [Convert]::ToInt32($report.completed_task_count)
        $solvedTaskCount = [Convert]::ToInt32($report.solved_task_count)
        $trial = [Convert]::ToInt32($report.trial)
        $pythonTaskCount = [Convert]::ToInt32(
            $report.language_success.python.tasks
        )
        $pythonSolved = [Convert]::ToInt32(
            $report.language_success.python.solved
        )
        $javaTaskCount = [Convert]::ToInt32(
            $report.language_success.java.tasks
        )
        $javaSolved = [Convert]::ToInt32(
            $report.language_success.java.solved
        )
        if ($null -eq $report.latency_ms.p95) {
            throw "missing p95 latency"
        }
        $p95Milliseconds = [Convert]::ToDouble(
            $report.latency_ms.p95,
            [Globalization.CultureInfo]::InvariantCulture
        )
        $tasks = @($report.tasks)
        $taskIds = @($tasks | ForEach-Object { [string]$_.task_id })
        $rowSolvedTaskCount = @(
            $tasks | Where-Object { $_.solved -eq $true }
        ).Count
        $pythonRows = @(
            $tasks | Where-Object { $_.language -ceq "python" }
        )
        $javaRows = @(
            $tasks | Where-Object { $_.language -ceq "java" }
        )
        $pythonRowSolved = @(
            $pythonRows | Where-Object { $_.solved -eq $true }
        ).Count
        $javaRowSolved = @(
            $javaRows | Where-Object { $_.solved -eq $true }
        ).Count
        $invalidTaskRows = @(
            $tasks | Where-Object {
                $_.status -cne "completed" -or
                $_.model -cne $ExpectedModelId -or
                $_.solved -isnot [bool] -or
                (
                    $_.task_id -clike "py-*" -and
                    $_.language -cne "python"
                ) -or
                (
                    $_.task_id -clike "java-*" -and
                    $_.language -cne "java"
                )
            }
        )
    }
    catch {
        throw "Formal acceptance summary is invalid."
    }

    if (
        $report.status -cne "completed" -or
        $report.suite_id -cne "repo-agent-python-java-day3-v1" -or
        $report.variant -cne "full" -or
        $trial -ne 1 -or
        $report.model -cne $ExpectedModelId -or
        $taskCount -ne 12 -or
        $completedTaskCount -ne 12 -or
        $tasks.Count -ne 12
    ) {
        throw "Formal acceptance did not complete the fixed full trial-1 suite."
    }

    $expectedTaskIds = @(
        1..6 | ForEach-Object { "py-bugfix-{0:D3}" -f $_ }
    ) + @(
        1..6 | ForEach-Object { "java-bugfix-{0:D3}" -f $_ }
    )
    $taskIdDifferences = @(
        Compare-Object `
            -ReferenceObject $expectedTaskIds `
            -DifferenceObject $taskIds `
            -CaseSensitive
    )
    if ($taskIdDifferences.Count -ne 0) {
        throw "Formal acceptance task identity does not match the frozen suite."
    }
    if (
        $invalidTaskRows.Count -ne 0 -or
        $rowSolvedTaskCount -ne $solvedTaskCount -or
        $pythonRows.Count -ne $pythonTaskCount -or
        $pythonRowSolved -ne $pythonSolved -or
        $javaRows.Count -ne $javaTaskCount -or
        $javaRowSolved -ne $javaSolved
    ) {
        throw "Formal acceptance task rows do not match its aggregate totals."
    }
    if ($solvedTaskCount -lt 8) {
        throw "Formal acceptance failed: fewer than 8 of 12 tasks solved."
    }
    if ($pythonTaskCount -ne 6 -or $pythonSolved -lt 4) {
        throw "Formal acceptance failed: fewer than 4 of 6 Python tasks solved."
    }
    if ($javaTaskCount -ne 6 -or $javaSolved -lt 4) {
        throw "Formal acceptance failed: fewer than 4 of 6 Java tasks solved."
    }

    $budgetFailures = @(
        $tasks | Where-Object { $_.budget_passed -ne $true }
    )
    $timeoutFailures = @(
        $tasks | Where-Object { $_.agent.timed_out -ne $false }
    )
    $budgetCategories = @(
        $tasks | Where-Object {
            $_.failure_category -cin @("budget_exceeded", "agent_timeout")
        }
    )
    $usageFailures = @(
        $tasks | Where-Object {
            $_.agent.usage.complete -ne $true -or
            $null -eq $_.agent.usage.total_tokens -or
            [Convert]::ToInt64($_.agent.usage.total_tokens) -gt 30000
        }
    )
    if (
        $report.tokens.complete -ne $true -or
        $budgetFailures.Count -ne 0 -or
        $timeoutFailures.Count -ne 0 -or
        $budgetCategories.Count -ne 0 -or
        $usageFailures.Count -ne 0
    ) {
        throw "Formal acceptance failed: a task exceeded its budget or timed out."
    }
    if (
        [Double]::IsNaN($p95Milliseconds) -or
        [Double]::IsInfinity($p95Milliseconds) -or
        $p95Milliseconds -lt 0.0 -or
        $p95Milliseconds -gt 480000.0
    ) {
        throw "Formal acceptance failed: p95 latency exceeds 480 seconds."
    }
    Write-Host (
        "Formal acceptance passed: {0}/12 solved; Python {1}/6; Java {2}/6; p95 {3:N0} ms." -f
        $solvedTaskCount,
        $pythonSolved,
        $javaSolved,
        $p95Milliseconds
    )
}

function Set-RelayEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$ApiKey,
        [Parameter(Mandatory = $true)][string]$ModelId
    )

    $ApiKey = $ApiKey.Trim()
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
        $evaluationConflicts = @(
            $relayInvocationParameterNames | Where-Object {
                $_ -cin @(
                    "Repository",
                    "Model",
                    "Port",
                    "OpenBrowser",
                    "Reconfigure",
                    "ConfigureOnly",
                    "Verify",
                    "FormalEvaluate"
                )
            }
        )
        if (
            $Evaluate -and
            (
                $evaluationConflicts.Count -ne 0 -or
                $FormalEvaluate -or
                $OpenBrowser -or
                $Reconfigure -or
                $ConfigureOnly -or
                $Verify -or
                -not [string]::IsNullOrWhiteSpace($Model)
            )
        ) {
            throw "-Evaluate cannot be combined with startup, configuration, verification, or formal evaluation options."
        }
        $formalConflicts = @(
            $relayInvocationParameterNames | Where-Object {
                $_ -cin @(
                    "Repository",
                    "Model",
                    "Port",
                    "OpenBrowser",
                    "Reconfigure",
                    "ConfigureOnly",
                    "Verify",
                    "Evaluate"
                )
            }
        )
        if (
            $FormalEvaluate -and
            (
                $formalConflicts.Count -ne 0 -or
                $Evaluate -or
                $OpenBrowser -or
                $Reconfigure -or
                $ConfigureOnly -or
                $Verify -or
                -not [string]::IsNullOrWhiteSpace($Model)
            )
        ) {
            throw "-FormalEvaluate cannot be combined with startup, configuration, verification, or development evaluation options."
        }
        $repositoryPath = $null
        if (-not $ConfigureOnly -and -not $Evaluate -and -not $FormalEvaluate) {
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
        if (($Evaluate -or $FormalEvaluate) -and $shouldConfigure) {
            $evaluationMode = if ($FormalEvaluate) {
                "-FormalEvaluate"
            }
            else {
                "-Evaluate"
            }
            throw "$evaluationMode requires an existing saved relay configuration."
        }
        if (
            -not $shouldConfigure -and
            -not [string]::IsNullOrWhiteSpace($Model)
        ) {
            throw "-Model can be used only for first-time setup or with -Reconfigure."
        }

        if ($shouldConfigure) {
            $promptedKey = Read-Host (
            "DeepSeek API key (masked as *; DPAPI-encrypted when saved)"
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
                if ($modelIds -cnotcontains $defaultModelId) {
                    throw "The default DeepSeek model is currently unavailable."
                }
                $selectedModel = $defaultModelId
            }
            else {
                Assert-ModelId -ModelId $selectedModel
                if ($modelIds -cnotcontains $selectedModel) {
                    throw "The requested model ID was not returned by DeepSeek."
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
            Write-Host "Encrypted DeepSeek configuration saved: $configPath"
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
                "Loaded saved DeepSeek configuration for model: {0}" -f
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
        if ($FormalEvaluate) {
            Invoke-RelayFormalEvaluation -PythonInvocation $pythonInvocation
            return
        }
        if ($Evaluate) {
            Invoke-RelayEvaluation -PythonInvocation $pythonInvocation
            return
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
