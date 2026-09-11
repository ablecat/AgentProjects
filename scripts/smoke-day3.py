"""Exercise the Day 3 check CLI with real hardened Docker containers."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_PARENT = PROJECT_ROOT / ".tmp"

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_RUNTIME_ERROR = 2
EXIT_CLEANUP_FAILED = 3

sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from repo_agent import cli as cli_module  # noqa: E402
from repo_agent.checks import CheckRunner  # noqa: E402
from repo_agent.sandbox import (  # noqa: E402
    CommandOutcome,
    SubprocessCommandRunner,
)


class RecordingRunner:
    """Record the Docker boundary while delegating to the real process runner."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._delegate = SubprocessCommandRunner()

    def __call__(
        self,
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        command = tuple(argv)
        self.commands.append(command)
        return self._delegate(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )


def main() -> int:
    report: dict[str, object] = {
        "status": "failed",
        "exit_code": EXIT_RUNTIME_ERROR,
        "checks": [],
        "runs": {},
        "cleanup": {
            "temporary_fixtures_removed": False,
            "containers_removed": False,
        },
        "error": None,
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]
    token = uuid.uuid4().hex[:8]
    common_prefix = f"repo-agent-day3-smoke-{token}"
    temp_root: Path | None = None
    runtime_failed = False
    cleanup_failed = False

    try:
        TEMP_PARENT.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix="day3-smoke-", dir=TEMP_PARENT)
        ).resolve(strict=True)
        python_repo = temp_root / "python-repo"
        maven_repo = temp_root / "maven-repo"
        runner_temp = temp_root / "check-runs"
        runner_temp.mkdir()
        _create_python_repo(python_repo)
        _create_maven_repo(maven_repo)

        python_blocked_runner = RecordingRunner()
        python_blocked_run = _invoke_check_cli(
            python_repo,
            allow_bootstrap=False,
            runner=python_blocked_runner,
            prefix=f"{common_prefix}-py-blocked",
            temp_parent=runner_temp,
        )
        report_runs = report["runs"]
        assert isinstance(report_runs, dict)
        report_runs["python_without_bootstrap"] = _run_summary(
            python_blocked_run
        )

        python_runner = RecordingRunner()
        python_run = _invoke_check_cli(
            python_repo,
            allow_bootstrap=True,
            runner=python_runner,
            prefix=f"{common_prefix}-py",
            temp_parent=runner_temp,
        )
        report_runs["python_with_bootstrap"] = _run_summary(python_run)

        blocked_runner = RecordingRunner()
        blocked_run = _invoke_check_cli(
            maven_repo,
            allow_bootstrap=False,
            runner=blocked_runner,
            prefix=f"{common_prefix}-blocked",
            temp_parent=runner_temp,
        )
        report_runs["maven_without_bootstrap"] = _run_summary(blocked_run)

        maven_runner = RecordingRunner()
        maven_run = _invoke_check_cli(
            maven_repo,
            allow_bootstrap=True,
            runner=maven_runner,
            prefix=f"{common_prefix}-mvn",
            temp_parent=runner_temp,
        )
        report_runs["maven_with_bootstrap"] = _run_summary(maven_run)

        python_payload = python_run[1]
        python_blocked_payload = python_blocked_run[1]
        blocked_payload = blocked_run[1]
        maven_payload = maven_run[1]
        python_creates = _create_commands(python_runner.commands)
        python_blocked_creates = _create_commands(python_blocked_runner.commands)
        blocked_creates = _create_commands(blocked_runner.commands)
        maven_creates = _create_commands(maven_runner.commands)

        _check(
            checks,
            "python_requires_explicit_bootstrap",
            python_blocked_run[0] == 1
            and python_blocked_payload.get("status") == "bootstrap_required"
            and _nested(python_blocked_payload, "profile", "id")
            == "python-pytest"
            and python_blocked_payload.get("phases") == []
            and not python_blocked_creates,
            _failure_detail(python_blocked_run),
        )
        _check(
            checks,
            "python_profile_detected",
            python_run[0] == 0
            and python_payload.get("status") == "passed"
            and _nested(python_payload, "profile", "id") == "python-pytest"
            and _nested(python_payload, "profile", "language") == "python",
            _failure_detail(python_run),
        )
        _check(
            checks,
            "python_verify_is_offline",
            _phase_networks(python_payload)
            == [
                ("bootstrap", "bridge"),
                ("verify", "none"),
            ]
            and len(python_creates) == 2
            and [_option_values(command, "--network") for command in python_creates]
            == [["bridge"], ["none"]]
            and _verify_argv(python_payload)[:1] == ["python"]
            and "--target" in _phase_argvs(python_payload, "bootstrap")[0]
            and "/dependencies/python"
            in _phase_argvs(python_payload, "bootstrap")[0]
            and all(
                _option_values(command, "--env").count(
                    "PYTHONPATH=/dependencies/python"
                )
                == 1
                for command in python_creates
            )
            and len({_dependency_mount(command) for command in python_creates}) == 1,
            "Python pip --target bootstrap uses bridge; the shared target is reused offline",
        )
        _check(
            checks,
            "maven_requires_explicit_bootstrap",
            blocked_run[0] == 1
            and blocked_payload.get("status") == "bootstrap_required"
            and _nested(blocked_payload, "profile", "id") == "maven-test"
            and blocked_payload.get("phases") == []
            and not blocked_creates,
            _failure_detail(blocked_run),
        )
        _check(
            checks,
            "maven_profile_detected",
            maven_run[0] == 0
            and maven_payload.get("status") == "passed"
            and _nested(maven_payload, "profile", "id") == "maven-test"
            and _nested(maven_payload, "profile", "language") == "java"
            and _nested(maven_payload, "profile", "java_release") == 17,
            _failure_detail(maven_run),
        )
        _check(
            checks,
            "bootstrap_and_verify_networks_are_separated",
            _phase_networks(maven_payload)
            == [
                ("bootstrap", "bridge"),
                ("bootstrap", "bridge"),
                ("bootstrap", "bridge"),
                ("bootstrap", "bridge"),
                ("verify", "none"),
            ]
            and _maven_create_networks_are_separated(maven_creates),
            "all Maven bootstrap attempts use bridge; the sole verify uses network=none",
        )
        _check(
            checks,
            "maven_verify_is_offline",
            _verify_argv(maven_payload)[:1] == ["mvn"]
            and "--offline" in _verify_argv(maven_payload),
            "Maven verification command includes --offline",
        )
        all_creates = python_creates + maven_creates
        _check(
            checks,
            "docker_hardening_applied",
            bool(all_creates) and all(_is_hardened(command) for command in all_creates),
            f"validated hardening on {len(all_creates)} real Docker create commands",
        )
        _check(
            checks,
            "check_run_directories_removed",
            not any(runner_temp.iterdir()),
            "every CheckRunner-owned temporary directory was removed",
        )
    except BaseException as exc:
        runtime_failed = True
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        containers_removed, container_error = _containers_removed(common_prefix)
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        cleanup["containers_removed"] = containers_removed
        if container_error is not None:
            cleanup["container_check_error"] = container_error
        _check(
            checks,
            "containers_removed",
            containers_removed,
            "no smoke-owned Docker containers remain",
        )

        fixtures_removed = True
        fixture_error: str | None = None
        if temp_root is not None and temp_root.exists():
            try:
                _remove_temp_root(temp_root)
            except BaseException as exc:
                fixtures_removed = False
                fixture_error = f"{type(exc).__name__}: {exc}"
        if temp_root is not None and temp_root.exists():
            fixtures_removed = False
        cleanup["temporary_fixtures_removed"] = fixtures_removed
        if fixture_error is not None:
            cleanup["temporary_fixture_error"] = fixture_error
        _check(
            checks,
            "temporary_fixtures_removed",
            fixtures_removed,
            "temporary Python and Maven repositories were removed",
        )
        cleanup_failed = not containers_removed or not fixtures_removed

    checks_passed = all(bool(item["ok"]) for item in checks)
    if cleanup_failed:
        exit_code = EXIT_CLEANUP_FAILED
    elif runtime_failed:
        exit_code = EXIT_RUNTIME_ERROR
    elif not checks_passed:
        exit_code = EXIT_CHECK_FAILED
    else:
        exit_code = EXIT_OK
        report["status"] = "passed"
    report["exit_code"] = exit_code
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _invoke_check_cli(
    repo: Path,
    *,
    allow_bootstrap: bool,
    runner: RecordingRunner,
    prefix: str,
    temp_parent: Path,
) -> tuple[int, dict[str, object], str, str]:
    original = cli_module.CheckRunner

    def create_runner(repo_path, **kwargs):
        return CheckRunner(
            repo_path,
            **kwargs,
            command_runner=runner,
            container_name_prefix=prefix,
            temp_parent=temp_parent,
        )

    stdout = StringIO()
    stderr = StringIO()
    arguments = ["check", "--repo", str(repo), "--format", "json"]
    if allow_bootstrap:
        arguments.append("--allow-bootstrap")
    cli_module.CheckRunner = create_runner
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli_module.main(arguments)
    finally:
        cli_module.CheckRunner = original
    output = stdout.getvalue()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"check CLI did not return JSON: {output!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("check CLI JSON must be an object")
    return exit_code, payload, output, stderr.getvalue()


def _create_python_repo(repo: Path) -> None:
    files = {
        "requirements.txt": "pytest==9.1.1\n",
        "pyproject.toml": """\
[tool.pytest.ini_options]
testpaths = ["tests"]
""",
        "tests/test_math.py": """\
def test_addition():
    assert 20 + 22 == 42
""",
    }
    _create_repository(repo, files)


def _create_maven_repo(repo: Path) -> None:
    files = {
        "pom.xml": """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>day3-smoke</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.release>17</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <junit.version>5.11.4</junit.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>${junit.version}</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.5.2</version>
      </plugin>
    </plugins>
  </build>
</project>
""",
        "src/main/java/com/example/Calculator.java": """\
package com.example;

public final class Calculator {
    private Calculator() {}

    public static int add(int left, int right) {
        return left + right;
    }
}
""",
        "src/test/java/com/example/CalculatorTest.java": """\
package com.example;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void addsTwoValues() {
        assertEquals(42, Calculator.add(20, 22));
    }
}
""",
    }
    _create_repository(repo, files)


def _create_repository(repo: Path, files: dict[str, str]) -> None:
    repo.mkdir()
    for relative, content in files.items():
        path = repo / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Day 3 Smoke")
    _git(repo, "config", "user.email", "day3-smoke@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "smoke fixture")


def _run_summary(
    run: tuple[int, dict[str, object], str, str]
) -> dict[str, object]:
    exit_code, payload, _stdout, stderr = run
    phases = payload.get("phases")
    summarized_phases: list[dict[str, object]] = []
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, dict):
                summarized_phases.append(
                    {
                        key: phase.get(key)
                        for key in (
                            "name",
                            "kind",
                            "status",
                            "network",
                            "argv",
                            "exit_code",
                            "duration_ms",
                            "truncated",
                            "error",
                        )
                    }
                )
                output = phase.get("output")
                if isinstance(output, str) and output:
                    summarized_phases[-1]["output_tail"] = output[-2048:]
    return {
        "exit_code": exit_code,
        "status": payload.get("status"),
        "profile": payload.get("profile"),
        "base_commit": payload.get("base_commit"),
        "candidate_applied": payload.get("candidate_applied"),
        "phases": summarized_phases,
        "duration_ms": payload.get("duration_ms"),
        "error": payload.get("error"),
        "cleanup_ok": payload.get("cleanup_ok"),
        "stderr": stderr,
    }


def _failure_detail(run: tuple[int, dict[str, object], str, str]) -> str:
    exit_code, payload, stdout, stderr = run
    if payload.get("status") in {"passed", "bootstrap_required"}:
        return f"CLI exit={exit_code}, status={payload.get('status')}"
    return (
        f"CLI exit={exit_code}, status={payload.get('status')}, "
        f"error={payload.get('error')!r}, stderr={stderr!r}, stdout={stdout[-2000:]!r}"
    )


def _nested(payload: dict[str, object], parent: str, child: str) -> object:
    value = payload.get(parent)
    return value.get(child) if isinstance(value, dict) else None


def _phase_networks(payload: dict[str, object]) -> list[tuple[object, object]]:
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return []
    return [
        (phase.get("kind"), phase.get("network"))
        for phase in phases
        if isinstance(phase, dict)
    ]


def _verify_argv(payload: dict[str, object]) -> list[object]:
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return []
    for phase in phases:
        if isinstance(phase, dict) and phase.get("kind") == "verify":
            argv = phase.get("argv")
            return argv if isinstance(argv, list) else []
    return []


def _phase_argvs(payload: dict[str, object], kind: str) -> list[list[object]]:
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return []
    result: list[list[object]] = []
    for phase in phases:
        if isinstance(phase, dict) and phase.get("kind") == kind:
            argv = phase.get("argv")
            if isinstance(argv, list):
                result.append(argv)
    return result


def _create_commands(commands: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [
        command
        for command in commands
        if len(command) >= 3 and command[:3] == ("docker", "container", "create")
    ]


def _option_values(command: tuple[str, ...], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command[:-1]):
        if value == option:
            values.append(command[index + 1])
    return values


def _dependency_mount(command: tuple[str, ...]) -> str:
    return next(
        (value for value in _option_values(command, "--mount") if "target=/dependencies" in value),
        "",
    )


def _maven_create_networks_are_separated(
    commands: list[tuple[str, ...]],
) -> bool:
    # Four logical bootstrap phases may each have one retry, followed by verify.
    if not 5 <= len(commands) <= 9:
        return False
    networks = [_option_values(command, "--network") for command in commands]
    entrypoints = [_option_values(command, "--entrypoint") for command in commands]
    return (
        all(value == ["bridge"] for value in networks[:-1])
        and networks[-1] == ["none"]
        and all(value == ["mvn"] for value in entrypoints)
        and all("--offline" not in command for command in commands[:-1])
        and "--offline" in commands[-1]
        and len({_dependency_mount(command) for command in commands}) == 1
        and bool(_dependency_mount(commands[0]))
    )


def _is_hardened(command: tuple[str, ...]) -> bool:
    required_flags = {"--init", "--read-only"}
    required_options = {
        "--pull": "never",
        "--restart": "no",
        "--user": "10001:10001",
        "--cpus": "2",
        "--memory": "4g",
        "--memory-swap": "4g",
        "--pids-limit": "256",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--log-driver": "none",
    }
    required_environment = {
        "HOME=/tmp/home",
        "CI=1",
        "NO_COLOR=1",
        "PYTHONDONTWRITEBYTECODE=1",
    }
    values_ok = all(
        _option_values(command, option) == [expected]
        for option, expected in required_options.items()
    )
    tmpfs = _option_values(command, "--tmpfs")
    mounts = _option_values(command, "--mount")
    environments = set(_option_values(command, "--env"))
    return (
        required_flags.issubset(command)
        and values_ok
        and len(tmpfs) == 1
        and tmpfs[0].startswith("/tmp:rw,nosuid,nodev,")
        and len(mounts) == 2
        and any("target=/workspace" in mount for mount in mounts)
        and any("target=/dependencies" in mount for mount in mounts)
        and required_environment.issubset(environments)
        and not any("docker.sock" in value.casefold() for value in command)
    )


def _check(
    checks: list[dict[str, object]], name: str, ok: bool, detail: str
) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _containers_removed(prefix: str) -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            (
                "docker",
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"name={prefix}",
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=30,
        )
    except BaseException as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return not bool(completed.stdout.strip()), None


def _remove_temp_root(temp_root: Path) -> None:
    expected_parent = TEMP_PARENT.resolve(strict=True)
    target = temp_root.resolve(strict=True)
    if target.parent != expected_parent or not target.name.startswith("day3-smoke-"):
        raise RuntimeError(f"refusing to remove unexpected path: {target}")
    shutil.rmtree(target, onerror=_remove_readonly)


def _remove_readonly(function, path: str, _error) -> None:
    os.chmod(path, stat.S_IWRITE)
    if not callable(function):
        raise TypeError("rmtree error handler received a non-callable operation")
    function(path)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        timeout=30,
    )


if __name__ == "__main__":
    raise SystemExit(main())
