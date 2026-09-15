from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

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


def test_windows_launcher_targets_the_official_deepseek_api() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    script = script_path.read_text(encoding="utf-8-sig")

    assert '$relayBaseUrl = "https://api.deepseek.com"' in script
    assert '$defaultModelId = "deepseek-v4-pro"' in script
    assert "$selectedModel = $defaultModelId" in script
    assert "thz10.airucas.com" not in script


@pytest.fixture
def short_tmp_path() -> Path:
    path = Path(
        tempfile.mkdtemp(prefix=".relay-eval-test-", dir=Path(__file__).parents[1])
    )
    try:
        yield path
    finally:
        shutil.rmtree(path)


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

$shortKeyRejected = $false
try {{ Assert-ApiKey -ApiKey "short" }} catch {{ $shortKeyRejected = $true }}
if (-not $shortKeyRejected) {{
    throw "A key too short for reliable secret scanning was accepted."
}}

if ((Get-RelayConfigFingerprint -Path $path) -cne "missing") {{
    throw "A missing configuration has the wrong fingerprint."
}}
[IO.Directory]::CreateDirectory((Split-Path -Parent $path)) > $null
[IO.File]::WriteAllBytes($path, [byte[]]@())
$damagedFingerprint = Get-RelayConfigFingerprint -Path $path
Write-RelayConfig `
    -Path $path `
    -ApiKey ("  " + $credentialA + "  ") `
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

    $ConfigureOnly = $false
    $Evaluate = $true
    $FormalEvaluate = $false
    $OpenBrowser = $false
$Verify = $false
$script:evaluationInvoked = $false
function Get-CheckedPythonInvocation {{ return @("fake-python") }}
function Invoke-RelayEvaluation {{
    param([Parameter(Mandatory = $true)][object[]]$PythonInvocation)
    if (
        $PythonInvocation[0] -cne "fake-python" -or
        $env:REPO_AGENT_API_KEY -cne $credentialB -or
        $env:REPO_AGENT_BASE_URL -cne $relayBaseUrl -or
        $env:REPO_AGENT_MODEL -cne $modelB
    ) {{
        throw "The evaluation did not receive the saved relay configuration."
    }}
    $script:evaluationInvoked = $true
}}
Invoke-RelayLauncher
if (-not $script:evaluationInvoked) {{
    throw "The restricted evaluation mode was not invoked."
}}
    if (
        $env:REPO_AGENT_API_KEY -cne "sentinel-original-key" -or
    $env:REPO_AGENT_BASE_URL -cne "http://127.0.0.1:9/v1" -or
    $env:REPO_AGENT_MODEL -cne "sentinel-original-model"
) {{
    throw "Evaluation did not restore the original process environment."
}}

$Reconfigure = $true
$evaluationReconfigureRejected = $false
try {{ Invoke-RelayLauncher }} catch {{ $evaluationReconfigureRejected = $true }}
if (-not $evaluationReconfigureRejected) {{
    throw "Development evaluation allowed relay reconfiguration."
}}
$Reconfigure = $false
$savedConfigPath = $configPath
$configPath = Join-Path (Split-Path -Parent $path) "missing-relay.json"
$missingEvaluationConfigRejected = $false
try {{ Invoke-RelayLauncher }} catch {{ $missingEvaluationConfigRejected = $true }}
if (-not $missingEvaluationConfigRejected) {{
    throw "Development evaluation did not require a saved configuration."
}}
$configPath = $savedConfigPath

    $Evaluate = $false
    $FormalEvaluate = $true
    $script:formalEvaluationInvoked = $false
    function Invoke-RelayFormalEvaluation {{
        param([Parameter(Mandatory = $true)][object[]]$PythonInvocation)
        if (
            $PythonInvocation[0] -cne "fake-python" -or
            $env:REPO_AGENT_API_KEY -cne $credentialB -or
            $env:REPO_AGENT_BASE_URL -cne $relayBaseUrl -or
            $env:REPO_AGENT_MODEL -cne $modelB
        ) {{
            throw "The formal evaluation did not receive the saved relay configuration."
        }}
        $script:formalEvaluationInvoked = $true
    }}
    Invoke-RelayLauncher
    if (-not $script:formalEvaluationInvoked) {{
        throw "The restricted formal evaluation mode was not invoked."
    }}
    if (
        $env:REPO_AGENT_API_KEY -cne "sentinel-original-key" -or
        $env:REPO_AGENT_BASE_URL -cne "http://127.0.0.1:9/v1" -or
        $env:REPO_AGENT_MODEL -cne "sentinel-original-model"
    ) {{
        throw "Formal evaluation did not restore the original process environment."
    }}

    $Evaluate = $true
    $conflictRejected = $false
    try {{ Invoke-RelayLauncher }} catch {{ $conflictRejected = $true }}
    if (-not $conflictRejected) {{
        throw "Conflicting development and formal evaluation modes were accepted."
    }}
    $Evaluate = $false
    $FormalEvaluate = $false

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
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "relay-config-tests-ok" in completed.stdout
    assert "not-a-real-credential" not in config_path.read_text(encoding="utf-8")


def test_relay_evaluation_command_is_fixed_and_keeps_key_out_of_arguments() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    script = script_path.read_text(encoding="utf-8-sig")

    start = script.index("function Invoke-RelayEvaluation")
    end = script.index("function Get-EvaluationSummaryPath", start)
    definition = script[start:end]
    assert '"benchmarks\\tasks\\python\\development"' in definition
    assert '"--variant",\n            "full"' in definition
    assert '"--execute"' in definition
    assert '"--allow-remote-model"' in definition
    assert 'Assert-DevelopmentEvaluationAcceptance' in definition
    assert 'Get-EvaluationSummaryPath' in definition
    assert "REPO_AGENT_API_KEY" not in definition
    assert "--api-key" not in definition.casefold()


def test_relay_formal_evaluation_command_is_fixed_and_isolated() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    script = script_path.read_text(encoding="utf-8-sig")

    start = script.index("function Invoke-RelayFormalEvaluation")
    end = script.index("function Assert-FormalEvaluationAcceptance", start)
    definition = script[start:end]
    assert 'Join-Path $projectRoot "benchmarks"' in definition
    assert '"evaluation-results\\formal-v1-regression-acceptance"' in definition
    assert '"--variant",\n            "full"' in definition
    assert '"--trial",\n            "1"' in definition
    assert '"--execute"' in definition
    assert '"--allow-remote-model"' in definition
    assert '"--allow-bootstrap"' in definition
    assert '"--task-timeout-seconds",\n            "1200"' in definition
    assert 'Assert-FormalEvaluationAcceptance' in definition
    assert '"--task-id"' not in definition
    assert '"--matrix"' not in definition
    assert '"--merge"' not in definition
    assert "REPO_AGENT_API_KEY" not in definition
    assert "--api-key" not in definition.casefold()


@pytest.mark.parametrize("powershell", _powershells() or (None,))
def test_relay_evaluation_helpers_use_real_runner_summary_layout(
    powershell: str | None,
    short_tmp_path: Path,
) -> None:
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    project_root = short_tmp_path / "project"
    script_directory = project_root / "scripts"
    development_dir = project_root / "benchmarks" / "tasks" / "python" / "development"
    development_dir.mkdir(parents=True)
    script_directory.mkdir()
    argument_log = short_tmp_path / "evaluation-arguments.jsonl"
    fake_evaluator = short_tmp_path / "fake-evaluator.py"
    fake_evaluator.write_text(
        """\
import json
import os
from pathlib import Path
import sys


arguments = sys.argv[1:]


def option(name):
    index = arguments.index(name)
    return arguments[index + 1]


output_root = Path(option("--output-dir"))
benchmark_root = Path(option("--benchmark-dir"))
model = os.environ["REPO_AGENT_MODEL"]
is_development = benchmark_root.name == "development"
suite_id = (
    "repo-agent-python-development-v1"
    if is_development
    else "repo-agent-python-java-day3-v1"
)
if is_development:
    tasks = [
        {
            "task_id": f"py-development-{index:03d}",
            "status": "completed",
            "solved": index <= 3,
            "model": model,
            "language": "python",
            "budget_passed": True,
        }
        for index in range(1, 5)
    ]
    report = {
        "status": "completed",
        "suite_id": suite_id,
        "variant": "full",
        "trial": 1,
        "model": model,
        "task_count": 4,
        "solved_task_count": 3,
        "actionable_tool_error_rate": 0.0,
        "tasks": tasks,
    }
else:
    tasks = [
        {
            "task_id": f"{language}-bugfix-{index:03d}",
            "status": "completed",
            "solved": index <= 4,
            "model": model,
            "language": "python" if language == "py" else "java",
            "budget_passed": True,
            "failure_category": None,
            "agent": {
                "timed_out": False,
                "usage": {"complete": True, "total_tokens": 1000},
            },
        }
        for language in ("py", "java")
        for index in range(1, 7)
    ]
    report = {
        "status": "completed",
        "suite_id": suite_id,
        "variant": "full",
        "trial": 1,
        "model": model,
        "task_count": 12,
        "completed_task_count": 12,
        "solved_task_count": 8,
        "language_success": {
            "python": {"tasks": 6, "solved": 4},
            "java": {"tasks": 6, "solved": 4},
        },
        "tokens": {"complete": True},
        "latency_ms": {"p95": 480000},
        "tasks": tasks,
    }
summary_path = output_root / suite_id / "full" / "trial-1" / "summary.json"
summary_path.parent.mkdir(parents=True)
summary_path.write_text(json.dumps(report), encoding="utf-8")
with Path(os.environ["FAKE_EVALUATION_ARGUMENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
print(json.dumps(report))
""",
        encoding="utf-8",
    )
    command = f"""
$ErrorActionPreference = "Stop"
. {_powershell_literal(script_path)}
$scriptDirectory = {_powershell_literal(script_directory)}
$env:REPO_AGENT_API_KEY = "not-a-real-evaluation-secret"
$env:REPO_AGENT_MODEL = "fake-model"
$env:FAKE_EVALUATION_ARGUMENT_LOG = {_powershell_literal(argument_log)}
$invocation = @(
    {_powershell_literal(Path(sys.executable))},
    {_powershell_literal(fake_evaluator)}
)
Invoke-RelayEvaluation -PythonInvocation $invocation
Invoke-RelayFormalEvaluation -PythonInvocation $invocation
"evaluation-layout-ok"
"""
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "evaluation-layout-ok" in completed.stdout
    development_runs = list(
        (project_root / "evaluation-results" / "development-v1").iterdir()
    )
    formal_runs = list(
        (
            project_root
            / "evaluation-results"
            / "formal-v1-regression-acceptance"
        ).iterdir()
    )
    assert len(development_runs) == 1
    assert len(formal_runs) == 1
    assert (
        development_runs[0]
        / "repo-agent-python-development-v1"
        / "full"
        / "trial-1"
        / "summary.json"
    ).is_file()
    assert (
        formal_runs[0]
        / "repo-agent-python-java-day3-v1"
        / "full"
        / "trial-1"
        / "summary.json"
    ).is_file()

    invocations = [
        json.loads(line) for line in argument_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(invocations) == 2
    development_arguments, formal_arguments = invocations
    assert "--trial" not in development_arguments
    assert "--allow-bootstrap" not in development_arguments
    assert "--trial" in formal_arguments
    assert formal_arguments[formal_arguments.index("--trial") + 1] == "1"
    assert "--allow-bootstrap" in formal_arguments
    for arguments in invocations:
        assert "--variant" in arguments
        assert arguments[arguments.index("--variant") + 1] == "full"
        assert "--task-id" not in arguments
        assert "--matrix" not in arguments
        assert "--merge" not in arguments
        assert "not-a-real-evaluation-secret" not in arguments


@pytest.mark.parametrize("powershell", _powershells() or (None,))
def test_relay_evaluation_acceptance_gate(
    powershell: str | None,
    tmp_path: Path,
) -> None:
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    summary_path = (
        tmp_path
        / "repo-agent-python-development-v1"
        / "full"
        / "trial-1"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    passing = {
        "status": "completed",
        "suite_id": "repo-agent-python-development-v1",
        "variant": "full",
        "trial": 1,
        "model": "fake-model",
        "task_count": 4,
        "solved_task_count": 3,
        "actionable_tool_error_rate": 0.05,
        "tasks": [
            {
                "task_id": f"py-development-{index:03d}",
                "status": "completed",
                "solved": index <= 3,
                "model": "fake-model",
                "language": "python",
                "budget_passed": True,
            }
            for index in range(1, 5)
        ],
    }
    summary_path.write_text(json.dumps(passing), encoding="utf-8")
    command = (
        f". {_powershell_literal(script_path)}; "
        "Assert-DevelopmentEvaluationAcceptance "
        f"-SummaryPath {_powershell_literal(summary_path)} "
        '-ExpectedModelId "fake-model"'
    )
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Development acceptance passed" in completed.stdout

    failing_tasks = [dict(task) for task in passing["tasks"]]
    failing_tasks[2]["solved"] = False
    failing = {**passing, "solved_task_count": 2, "tasks": failing_tasks}
    summary_path.write_text(json.dumps(failing), encoding="utf-8")
    rejected = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert rejected.returncode != 0
    assert "fewer than 3 of 4 tasks solved" in rejected.stderr

    inconsistent = {**passing, "solved_task_count": 4}
    summary_path.write_text(json.dumps(inconsistent), encoding="utf-8")
    rejected = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0
    assert "task rows do not match its aggregate totals" in rejected.stderr


@pytest.mark.parametrize("powershell", _powershells() or (None,))
def test_relay_formal_evaluation_acceptance_gate(
    powershell: str | None,
    tmp_path: Path,
) -> None:
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    script_path = Path(__file__).parents[1] / "scripts" / "start-relay.ps1"
    summary_path = (
        tmp_path
        / "repo-agent-python-java-day3-v1"
        / "full"
        / "trial-1"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    tasks = [
        {
            "task_id": f"{language}-bugfix-{index:03d}",
            "status": "completed",
            "solved": index <= 4,
            "model": "fake-model",
            "language": "python" if language == "py" else "java",
            "budget_passed": True,
            "failure_category": None,
            "agent": {
                "timed_out": False,
                "usage": {"complete": True, "total_tokens": 30000},
            },
        }
        for language in ("py", "java")
        for index in range(1, 7)
    ]
    passing = {
        "status": "completed",
        "suite_id": "repo-agent-python-java-day3-v1",
        "variant": "full",
        "trial": 1,
        "model": "fake-model",
        "task_count": 12,
        "completed_task_count": 12,
        "solved_task_count": 8,
        "language_success": {
            "python": {"tasks": 6, "solved": 4},
            "java": {"tasks": 6, "solved": 4},
        },
        "tokens": {"complete": True},
        "latency_ms": {"p95": 480000},
        "tasks": tasks,
    }

    def invoke(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        command = (
            f". {_powershell_literal(script_path)}; "
            "Assert-FormalEvaluationAcceptance "
            f"-SummaryPath {_powershell_literal(summary_path)} "
            '-ExpectedModelId "fake-model"'
        )
        return subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
            check=False,
        )

    completed = invoke(passing)
    assert completed.returncode == 0, completed.stderr
    assert "Formal acceptance passed" in completed.stdout

    too_few_tasks = [dict(task) for task in tasks]
    too_few_tasks[9]["solved"] = False
    too_few = {
        **passing,
        "solved_task_count": 7,
        "language_success": {
            "python": {"tasks": 6, "solved": 4},
            "java": {"tasks": 6, "solved": 3},
        },
        "tasks": too_few_tasks,
    }
    rejected = invoke(too_few)
    assert rejected.returncode != 0
    assert "fewer than 8 of 12 tasks solved" in rejected.stderr

    language_miss_tasks = [
        {
            **task,
            "solved": (
                task["task_id"].startswith("py-")
                and int(task["task_id"].rsplit("-", 1)[1]) <= 3
            )
            or (
                task["task_id"].startswith("java-")
                and int(task["task_id"].rsplit("-", 1)[1]) <= 5
            ),
        }
        for task in tasks
    ]
    language_miss = {
        **passing,
        "language_success": {
            "python": {"tasks": 6, "solved": 3},
            "java": {"tasks": 6, "solved": 5},
        },
        "tasks": language_miss_tasks,
    }
    rejected = invoke(language_miss)
    assert rejected.returncode != 0
    assert "fewer than 4 of 6 Python tasks solved" in rejected.stderr

    inconsistent = {**passing, "solved_task_count": 9}
    rejected = invoke(inconsistent)
    assert rejected.returncode != 0
    assert "task rows do not match its aggregate totals" in rejected.stderr

    timed_out_tasks = [dict(task) for task in tasks]
    timed_out_tasks[0] = {
        **timed_out_tasks[0],
        "failure_category": "agent_timeout",
        "agent": {
            "timed_out": True,
            "usage": {"complete": True, "total_tokens": 30000},
        },
    }
    rejected = invoke({**passing, "tasks": timed_out_tasks})
    assert rejected.returncode != 0
    assert "exceeded its budget or timed out" in rejected.stderr

    rejected = invoke({**passing, "latency_ms": {"p95": 480001}})
    assert rejected.returncode != 0
    assert "p95 latency exceeds 480 seconds" in rejected.stderr
