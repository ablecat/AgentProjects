"""Exercise the Day 3 check CLI with real hardened Docker containers."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import re
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
from repo_agent.checks import (  # noqa: E402
    _CACHE_ANCHOR_ENTRYPOINT,
    _CACHE_RESOURCE,
    _CACHE_VOLUME_OPTIONS,
    _RESOURCE_LABEL,
    _RUN_LABEL,
    _SEQUENCE_LABEL,
    _WORKSPACE_ENTRYPOINT,
    _WORKSPACE_TMPFS,
    CheckRunner,
)
from repo_agent.sandbox import (  # noqa: E402
    CommandOutcome,
    SubprocessCommandRunner,
)


class RecordingRunner:
    """Record the Docker boundary while delegating to the real process runner."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.volume_names: set[str] = set()
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
        if command[:3] == ("docker", "volume", "create") and len(command) >= 4:
            self.volume_names.add(command[-1])
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
            "volumes_removed": False,
        },
        "error": None,
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]
    token = uuid.uuid4().hex[:8]
    common_prefix = f"repo-agent-day3-smoke-{token}"
    temp_root: Path | None = None
    runtime_failed = False
    cleanup_failed = False
    runners: list[RecordingRunner] = []

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
        runners.append(python_blocked_runner)
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
        runners.append(python_runner)
        python_run = _invoke_check_cli(
            python_repo,
            allow_bootstrap=True,
            runner=python_runner,
            prefix=f"{common_prefix}-py",
            temp_parent=runner_temp,
        )
        report_runs["python_with_bootstrap"] = _run_summary(python_run)

        blocked_runner = RecordingRunner()
        runners.append(blocked_runner)
        blocked_run = _invoke_check_cli(
            maven_repo,
            allow_bootstrap=False,
            runner=blocked_runner,
            prefix=f"{common_prefix}-blocked",
            temp_parent=runner_temp,
        )
        report_runs["maven_without_bootstrap"] = _run_summary(blocked_run)

        maven_runner = RecordingRunner()
        runners.append(maven_runner)
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
        python_anchor_creates = _anchor_create_commands(python_runner.commands)
        python_blocked_anchor_creates = _anchor_create_commands(
            python_blocked_runner.commands
        )
        blocked_anchor_creates = _anchor_create_commands(blocked_runner.commands)
        maven_anchor_creates = _anchor_create_commands(maven_runner.commands)
        python_volume_creates = _volume_create_commands(python_runner.commands)
        python_blocked_volume_creates = _volume_create_commands(
            python_blocked_runner.commands
        )
        blocked_volume_creates = _volume_create_commands(blocked_runner.commands)
        maven_volume_creates = _volume_create_commands(maven_runner.commands)

        _check(
            checks,
            "python_requires_explicit_bootstrap",
            python_blocked_run[0] == 1
            and python_blocked_payload.get("status") == "bootstrap_required"
            and _nested(python_blocked_payload, "profile", "id")
            == "python-pytest"
            and python_blocked_payload.get("phases") == []
            and not python_blocked_creates
            and not python_blocked_anchor_creates
            and not python_blocked_volume_creates,
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
            and "/cache/python"
            in _phase_argvs(python_payload, "bootstrap")[0]
            and all(
                _option_values(command, "--env").count(
                    "PYTHONPATH=/cache/python"
                )
                == 1
                for command in python_creates
            )
            and len({_dependency_mount(command) for command in python_creates}) == 1
            and bool(_dependency_mount(python_creates[0]))
            and _create_argvs_match_phases(python_creates, python_payload),
            "Python pip --target bootstrap uses bridge; the shared target is reused offline",
        )
        _check(
            checks,
            "maven_requires_explicit_bootstrap",
            blocked_run[0] == 1
            and blocked_payload.get("status") == "bootstrap_required"
            and _nested(blocked_payload, "profile", "id") == "maven-test"
            and blocked_payload.get("phases") == []
            and not blocked_creates
            and not blocked_anchor_creates
            and not blocked_volume_creates,
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
            and _maven_create_networks_are_separated(maven_creates)
            and _maven_create_argvs_match_phases(maven_creates, maven_payload),
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
        all_anchor_creates = python_anchor_creates + maven_anchor_creates
        python_volume_names = {
            _dependency_volume_name(command) for command in python_creates
        }
        maven_volume_names = {
            _dependency_volume_name(command) for command in maven_creates
        }
        python_created_volumes = {
            _created_volume_name(command) for command in python_volume_creates
        }
        maven_created_volumes = {
            _created_volume_name(command) for command in maven_volume_creates
        }
        _check(
            checks,
            "dependency_volumes_are_run_scoped",
            len(python_volume_names) == 1
            and len(maven_volume_names) == 1
            and "" not in python_volume_names
            and "" not in maven_volume_names
            and python_volume_names == python_created_volumes
            and maven_volume_names == maven_created_volumes
            and python_volume_names.isdisjoint(maven_volume_names)
            and len(python_volume_creates) == 1
            and len(maven_volume_creates) == 1
            and _runner_dependency_volume_contract(python_runner.commands)
            and _runner_dependency_volume_contract(maven_runner.commands)
            and all(
                _is_hardened_volume_create(command)
                for command in python_volume_creates + maven_volume_creates
            ),
            "each run creates, labels, bounds, mounts, and removes one private volume",
        )
        _check(
            checks,
            "docker_hardening_applied",
            bool(all_creates)
            and len(all_anchor_creates) == 2
            and all(_is_hardened(command) for command in all_creates)
            and all(
                _is_hardened_anchor(command) for command in all_anchor_creates
            ),
            (
                f"validated hardening on {len(all_creates)} phase containers and "
                f"{len(all_anchor_creates)} cache anchors"
            ),
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
        volume_names = set().union(*(runner.volume_names for runner in runners))
        volume_names.update(
            _dependency_volume_name(command)
            for runner in runners
            for command in _all_container_create_commands(runner.commands)
            if _dependency_volume_name(command)
        )
        volumes_removed, volume_error = _volumes_removed(volume_names)
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
        cleanup["volumes_removed"] = volumes_removed
        if volume_error is not None:
            cleanup["volume_check_error"] = volume_error
        _check(
            checks,
            "volumes_removed",
            volumes_removed,
            "no smoke-owned Docker dependency volumes remain",
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
        cleanup_failed = (
            not containers_removed or not volumes_removed or not fixtures_removed
        )

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
        "requirements.txt": "pytest==9.1.1\ntomli==2.2.1\n",
        "pyproject.toml": """\
[tool.pytest.ini_options]
testpaths = ["tests"]
""",
        "tests/test_math.py": """\
import tomli


def test_addition():
    assert 20 + 22 == 42


def test_bootstrap_cache_survives_into_offline_verify():
    assert tomli.loads("answer = 42")["answer"] == 42
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
        and not _is_anchor_create_command(command)
    ]


def _anchor_create_commands(
    commands: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    return [
        command
        for command in commands
        if len(command) >= 3
        and command[:3] == ("docker", "container", "create")
        and _is_anchor_create_command(command)
    ]


def _all_container_create_commands(
    commands: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    return [
        command
        for command in commands
        if len(command) >= 3 and command[:3] == ("docker", "container", "create")
    ]


def _is_anchor_create_command(command: tuple[str, ...]) -> bool:
    names = _option_values(command, "--name")
    return len(names) == 1 and names[0].endswith("-cache-anchor")


def _volume_create_commands(
    commands: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    return [
        command
        for command in commands
        if len(command) >= 4 and command[:3] == ("docker", "volume", "create")
    ]


def _option_values(command: tuple[str, ...], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command[:-1]):
        if value == option:
            values.append(command[index + 1])
    return values


def _dependency_mount(command: tuple[str, ...]) -> str:
    parsed = _parse_container_create(command)
    if parsed is None:
        return ""
    _, options, _, _ = parsed
    return next(
        (
            value
            for value in options.get("--mount", [])
            if _mount_fields(value).get("target") == "/cache"
        ),
        "",
    )


def _dependency_volume_name(command: tuple[str, ...]) -> str:
    mount = _dependency_mount(command)
    fields = _mount_fields(mount)
    if fields.get("type") != "volume":
        return ""
    return fields.get("source", "")


def _mount_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in value.split(","):
        key, separator, field_value = item.partition("=")
        if not key or key in fields:
            return {}
        fields[key] = field_value if separator else ""
    return fields


_CREATE_FLAGS = frozenset({"--init", "--read-only", "--rm"})
_CREATE_OPTIONS = frozenset(
    {
        "--name",
        "--label",
        "--pull",
        "--restart",
        "--network",
        "--user",
        "--cpus",
        "--memory",
        "--memory-swap",
        "--pids-limit",
        "--cap-drop",
        "--security-opt",
        "--log-driver",
        "--tmpfs",
        "--mount",
        "--workdir",
        "--env",
        "--entrypoint",
    }
)
_CONTAINER_NAME_PATTERN = re.compile(
    r"^(?P<prefix>repo-agent-day3-smoke-[0-9a-f]{8}-(?:py|mvn))-"
    r"(?P<run>[0-9a-f]{32})-(?P<sequence>[1-9][0-9]*)$"
)
_ANCHOR_NAME_PATTERN = re.compile(
    r"^(?P<prefix>repo-agent-day3-smoke-[0-9a-f]{8}-(?:py|mvn))-"
    r"(?P<run>[0-9a-f]{32})-cache-anchor$"
)
_VOLUME_NAME_PATTERN = re.compile(
    r"^(?P<prefix>repo-agent-day3-smoke-[0-9a-f]{8}-(?:py|mvn))-"
    r"(?P<run>[0-9a-f]{32})-cache$"
)


def _parse_container_create(
    command: tuple[str, ...],
) -> tuple[Counter[str], dict[str, list[str]], str, tuple[str, ...]] | None:
    if command[:3] != ("docker", "container", "create"):
        return None
    flags: Counter[str] = Counter()
    options: dict[str, list[str]] = {}
    index = 3
    while index < len(command) and command[index].startswith("-"):
        token = command[index]
        if token in _CREATE_FLAGS:
            flags[token] += 1
            index += 1
            continue
        if token not in _CREATE_OPTIONS or index + 1 >= len(command):
            return None
        options.setdefault(token, []).append(command[index + 1])
        index += 2
    if index >= len(command):
        return None
    return flags, options, command[index], command[index + 1 :]


def _container_payload_argv(command: tuple[str, ...]) -> tuple[str, ...] | None:
    parsed = _parse_container_create(command)
    if parsed is None:
        return None
    _, _, _, tail = parsed
    expected_prefix = ("-c", _WORKSPACE_ENTRYPOINT, "repo-agent-check")
    if tail[:3] != expected_prefix or len(tail) < 4:
        return None
    return tail[3:]


def _all_phase_argvs(payload: dict[str, object]) -> list[tuple[str, ...]]:
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return []
    result: list[tuple[str, ...]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            return []
        argv = phase.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            return []
        result.append(tuple(argv))
    return result


def _create_argvs_match_phases(
    commands: list[tuple[str, ...]], payload: dict[str, object]
) -> bool:
    actual = [_container_payload_argv(command) for command in commands]
    return None not in actual and actual == _all_phase_argvs(payload)


def _maven_create_argvs_match_phases(
    commands: list[tuple[str, ...]], payload: dict[str, object]
) -> bool:
    actual = [_container_payload_argv(command) for command in commands]
    expected = _all_phase_argvs(payload)
    if None in actual or len(expected) != 5:
        return False
    cursor = 0
    for phase_argv in expected[:-1]:
        attempts = 0
        while cursor < len(actual) and actual[cursor] == phase_argv:
            attempts += 1
            cursor += 1
        if not 1 <= attempts <= 2:
            return False
    return actual[cursor:] == [expected[-1]]


def _created_volume_name(command: tuple[str, ...]) -> str:
    return command[-1] if len(command) >= 4 else ""


def _parse_volume_create(
    command: tuple[str, ...],
) -> tuple[dict[str, list[str]], str] | None:
    if command[:3] != ("docker", "volume", "create"):
        return None
    allowed = {"--driver", "--opt", "--label"}
    options: dict[str, list[str]] = {}
    index = 3
    while index < len(command) - 1:
        token = command[index]
        if token not in allowed or index + 1 >= len(command) - 1:
            return None
        options.setdefault(token, []).append(command[index + 1])
        index += 2
    if index != len(command) - 1 or command[index].startswith("-"):
        return None
    return options, command[index]


def _is_hardened_volume_create(command: tuple[str, ...]) -> bool:
    parsed = _parse_volume_create(command)
    if parsed is None:
        return False
    options, name = parsed
    match = _VOLUME_NAME_PATTERN.fullmatch(name)
    if match is None:
        return False
    expected_labels = Counter(
        {
            f"{_RUN_LABEL}={match.group('run')}": 1,
            f"{_RESOURCE_LABEL}={_CACHE_RESOURCE}": 1,
        }
    )
    return (
        set(options) == {"--driver", "--opt", "--label"}
        and options["--driver"] == ["local"]
        and Counter(options["--opt"]) == Counter(_CACHE_VOLUME_OPTIONS)
        and Counter(options["--label"]) == expected_labels
    )


def _is_hardened_anchor(command: tuple[str, ...]) -> bool:
    parsed = _parse_container_create(command)
    if parsed is None:
        return False
    flags, options, image, tail = parsed
    names = options.get("--name", [])
    match = _ANCHOR_NAME_PATTERN.fullmatch(names[0]) if len(names) == 1 else None
    if match is None:
        return False
    expected_values = {
        "--pull": ["never"],
        "--restart": ["no"],
        "--network": ["none"],
        "--user": ["10001:10001"],
        "--cpus": ["2"],
        "--memory": ["4g"],
        "--memory-swap": ["4g"],
        "--pids-limit": ["256"],
        "--cap-drop": ["ALL"],
        "--security-opt": ["no-new-privileges"],
        "--log-driver": ["none"],
        "--entrypoint": ["/bin/sh"],
    }
    expected_keys = set(expected_values) | {"--name", "--label", "--mount"}
    expected_labels = Counter(
        {
            f"{_RUN_LABEL}={match.group('run')}": 1,
            f"{_SEQUENCE_LABEL}=0": 1,
        }
    )
    mounts = options.get("--mount", [])
    mount = _mount_fields(mounts[0]) if len(mounts) == 1 else {}
    expected_volume = f"{match.group('prefix')}-{match.group('run')}-cache"
    expected_image = (
        "repo-agent-python:0.1"
        if match.group("prefix").endswith("-py")
        else "repo-agent-maven:0.1"
    )
    return (
        flags == Counter({"--init": 1, "--read-only": 1, "--rm": 1})
        and set(options) == expected_keys
        and all(options.get(key) == value for key, value in expected_values.items())
        and Counter(options.get("--label", [])) == expected_labels
        and set(mount) == {"type", "source", "target"}
        and mount.get("type") == "volume"
        and mount.get("source") == expected_volume
        and mount.get("target") == "/cache"
        and image == expected_image
        and tail == ("-c", _CACHE_ANCHOR_ENTRYPOINT)
        and not any("docker.sock" in value.casefold() for value in command)
    )


def _anchor_start_is_recorded(commands: list[tuple[str, ...]]) -> bool:
    starts = [
        command
        for command in commands
        if command[:3] == ("docker", "container", "start")
        and len(command) == 4
    ]
    return len(starts) == 1 and bool(re.fullmatch(r"[0-9a-f]{64}", starts[0][-1]))


def _runner_dependency_volume_contract(commands: list[tuple[str, ...]]) -> bool:
    creates = _all_container_create_commands(commands)
    phase_creates = _create_commands(commands)
    anchor_creates = _anchor_create_commands(commands)
    volume_creates = _volume_create_commands(commands)
    if not phase_creates or len(anchor_creates) != 1 or len(volume_creates) != 1:
        return False
    created = {_created_volume_name(command) for command in volume_creates}
    mounted = {_dependency_volume_name(command) for command in creates}
    return (
        "" not in mounted
        and created == mounted
        and _is_hardened_volume_create(volume_creates[0])
        and _is_hardened_anchor(anchor_creates[0])
        and _anchor_start_is_recorded(commands)
        and all(_is_hardened(command) for command in phase_creates)
    )


def _maven_create_networks_are_separated(
    commands: list[tuple[str, ...]],
) -> bool:
    # Four logical bootstrap phases may each have one retry, followed by verify.
    if not 5 <= len(commands) <= 9:
        return False
    networks = [_option_values(command, "--network") for command in commands]
    return (
        all(value == ["bridge"] for value in networks[:-1])
        and networks[-1] == ["none"]
        and all("--offline" not in command for command in commands[:-1])
        and "--offline" in commands[-1]
        and len({_dependency_mount(command) for command in commands}) == 1
        and bool(_dependency_mount(commands[0]))
    )


def _is_hardened(command: tuple[str, ...]) -> bool:
    parsed = _parse_container_create(command)
    if parsed is None:
        return False
    flags, options, image, tail = parsed
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
        "--workdir": "/workspace",
        "--entrypoint": "/bin/sh",
    }
    required_environment = {
        "HOME=/tmp/home",
        "CI=1",
        "NO_COLOR=1",
        "PYTHONDONTWRITEBYTECODE=1",
    }
    values_ok = all(
        options.get(option) == [expected]
        for option, expected in required_options.items()
    )
    tmpfs = options.get("--tmpfs", [])
    mounts = options.get("--mount", [])
    source_mounts = [
        _mount_fields(value)
        for value in mounts
        if _mount_fields(value).get("target") == "/source"
    ]
    cache_mounts = [
        _mount_fields(value)
        for value in mounts
        if _mount_fields(value).get("target") == "/cache"
    ]
    source_ok = (
        len(source_mounts) == 1
        and set(source_mounts[0]) == {"type", "source", "target", "readonly"}
        and source_mounts[0].get("type") == "bind"
        and bool(source_mounts[0].get("source"))
        and source_mounts[0].get("readonly") == ""
    )
    cache_ok = (
        len(cache_mounts) == 1
        and set(cache_mounts[0]) == {"type", "source", "target"}
        and cache_mounts[0].get("type") == "volume"
        and bool(cache_mounts[0].get("source"))
    )
    name_values = options.get("--name", [])
    name_match = (
        _CONTAINER_NAME_PATTERN.fullmatch(name_values[0])
        if len(name_values) == 1
        else None
    )
    if name_match is None or not cache_ok:
        return False
    expected_volume = (
        f"{name_match.group('prefix')}-{name_match.group('run')}-cache"
    )
    expected_labels = Counter(
        {
            f"{_RUN_LABEL}={name_match.group('run')}": 1,
            f"{_SEQUENCE_LABEL}={name_match.group('sequence')}": 1,
        }
    )
    payload = _container_payload_argv(command)
    if payload is None:
        return False
    expected_image = {
        "python": "repo-agent-python:0.1",
        "mvn": "repo-agent-maven:0.1",
    }.get(payload[0])
    expected_environment = set(required_environment)
    if payload[0] == "python":
        expected_environment.add("PYTHONPATH=/cache/python")
    exact_option_keys = set(required_options) | {
        "--name",
        "--label",
        "--network",
        "--tmpfs",
        "--mount",
        "--env",
    }
    return (
        flags == Counter({"--init": 1, "--read-only": 1})
        and set(options) == exact_option_keys
        and values_ok
        and options.get("--network") in (["bridge"], ["none"])
        and Counter(options.get("--label", [])) == expected_labels
        and Counter(tmpfs)
        == Counter(
            {
                "/tmp:rw,nosuid,nodev,size=256m,mode=1777": 1,
                _WORKSPACE_TMPFS: 1,
            }
        )
        and len(mounts) == 2
        and source_ok
        and cache_ok
        and cache_mounts[0].get("source") == expected_volume
        and Counter(options.get("--env", []))
        == Counter({value: 1 for value in expected_environment})
        and image == expected_image
        and tail[:3] == ("-c", _WORKSPACE_ENTRYPOINT, "repo-agent-check")
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


def _volumes_removed(names: set[str]) -> tuple[bool, str | None]:
    errors: list[str] = []
    remaining: list[str] = []
    for name in sorted(names):
        try:
            completed = subprocess.run(
                ("docker", "volume", "inspect", name),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=15,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except BaseException as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if completed.returncode == 0:
            remaining.append(name)
        elif "no such volume" not in completed.stderr.casefold():
            errors.append(
                f"{name}: docker volume inspect exited {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
    details: list[str] = []
    if remaining:
        details.append("remaining volumes: " + ", ".join(remaining))
    details.extend(errors)
    return not details, "; ".join(details) or None


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
