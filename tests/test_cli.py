from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repo_agent import cli
from repo_agent.artifacts import ArtifactStore
from repo_agent.checks import CheckPhaseResult, CheckProfile, CheckRunResult
from repo_agent.doctor import DoctorResult
import repo_agent.service as service_module
from repo_agent.models import ToolCall, ToolResult
from repo_agent.persistence import RunDatabase
from repo_agent.run_models import ChangePlan, RunRecord, utc_now


class FakeSandbox:
    def __init__(self, repo, **kwargs) -> None:
        self.repo = repo
        self.kwargs = kwargs

    def __enter__(self) -> FakeSandbox:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(self, call: ToolCall) -> ToolResult:
        outputs = {
            "git_status": "## main",
            "repo_map": "FILES 2 shown / 2 visible",
            "search": "src/example.py:1:1:TODO",
        }
        return ToolResult(call.id, call.name, True, outputs[call.name], exit_code=0)


class FailingToolSandbox(FakeSandbox):
    def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call.id,
            call.name,
            False,
            "",
            error="sandbox unavailable",
            exit_code=125,
        )


def _durable_record(
    repository: Path,
    *,
    status: str = "succeeded",
    run_id: str = "a" * 32,
    plan: ChangePlan | None = None,
) -> RunRecord:
    now = utc_now()
    return RunRecord(
        run_id=run_id,
        repo_path=str(repository.resolve()),
        task="repair the bug",
        status=status,
        current_node="approval" if status == "awaiting_approval" else "finalize",
        plan=plan,
        created_at=now,
        updated_at=now,
    )


def test_json_run_is_structured_and_completes(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(service_module, "DockerSandbox", FakeSandbox)

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "inspect the repository",
            "--provider",
            "demo",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["steps"] == 4
    assert [result["name"] for result in payload["tool_results"]] == [
        "git_status",
        "repo_map",
        "search",
    ]
    assert "Demo inspection complete" in payload["answer"]


def test_setup_error_has_distinct_exit_code_and_json(monkeypatch, capsys) -> None:
    class BrokenSandbox:
        def __init__(self, *args, **kwargs) -> None:
            raise ValueError("not a Git repository")

    monkeypatch.setattr(service_module, "DockerSandbox", BrokenSandbox)

    exit_code = cli.main(
        [
            "run",
            "--task",
            "inspect",
            "--provider",
            "demo",
            "--format",
            "json",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "setup_error"
    assert payload["steps"] == 0
    assert payload["error"] == "ValueError: not a Git repository"


def test_json_interrupt_is_structured_and_uses_exit_code_130(
    monkeypatch, capsys
) -> None:
    class InterruptedSandbox:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            raise KeyboardInterrupt

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

    monkeypatch.setattr(service_module, "DockerSandbox", InterruptedSandbox)

    exit_code = cli.main(
        [
            "run",
            "--task",
            "inspect",
            "--provider",
            "demo",
            "--format",
            "json",
        ]
    )

    assert exit_code == 130
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == {
        "task": "inspect",
        "status": "interrupted",
        "answer": None,
        "steps": 0,
        "tool_results": [],
    }
    assert captured.err == ""


def test_tool_failures_produce_degraded_status_and_nonzero_exit(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setattr(service_module, "DockerSandbox", FailingToolSandbox)

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "inspect",
            "--provider",
            "demo",
            "--format",
            "json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed_with_errors"
    assert all(not result["ok"] for result in payload["tool_results"])
    assert payload["answer"].startswith(
        "Demo inspection finished with tool errors."
    )


def test_text_run_summarizes_tool_outcomes(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(service_module, "DockerSandbox", FakeSandbox)

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "inspect",
            "--provider",
            "demo",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: completed" in output
    assert "tool: git_status (ok, exit=0)" in output
    assert "tool: repo_map (ok, exit=0)" in output
    assert "tool: search (ok, exit=0)" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--task", "   "],
        ["run", "--task", "inspect", "--max-steps", "0"],
        ["run", "--task", "inspect", "--timeout-seconds", "nan"],
        ["run", "--task", "inspect", "--max-output-bytes", "1.5"],
        ["serve", "--port", "0"],
    ],
)
def test_invalid_cli_values_are_rejected(arguments) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code == 2


def test_serve_command_uses_loopback_web_entrypoint(monkeypatch, tmp_path) -> None:
    received: dict[str, object] = {}

    def fake_serve(
        repo, *, port, open_browser, data_dir, allow_bootstrap
    ) -> int:
        received.update(
            repo=repo,
            port=port,
            open_browser=open_browser,
            data_dir=data_dir,
            allow_bootstrap=allow_bootstrap,
        )
        return 0

    monkeypatch.setattr(cli, "serve_web", fake_serve)

    exit_code = cli.main(
        ["serve", "--repo", str(tmp_path), "--port", "9876", "--open"]
    )

    assert exit_code == 0
    assert received == {
        "repo": tmp_path,
        "port": 9876,
        "open_browser": True,
        "data_dir": cli._resolve_data_dir(None),
        "allow_bootstrap": False,
    }


def _check_result(status="passed") -> CheckRunResult:
    profile = CheckProfile(
        id="python-pytest",
        language="python",
        image="repo-agent-python:0.1",
        manifest="pyproject.toml",
        bootstrap_argv=(),
        check_argv=("python", "-m", "pytest", "-q"),
    )
    phase = CheckPhaseResult(
        name="python-pytest",
        kind="verify",
        status="passed" if status == "passed" else "failed",
        network="none",
        argv=profile.check_argv,
        exit_code=0 if status == "passed" else 1,
        output="1 passed",
        error=None if status == "passed" else "verification exited with code 1",
        duration_ms=12,
    )
    return CheckRunResult(
        status=status,
        profile=profile,
        base_commit="a" * 40,
        candidate_applied=False,
        phases=(phase,),
        duration_ms=15,
        error=None if status == "passed" else "verification exited with code 1",
    )


def test_check_json_invokes_runner_with_explicit_bootstrap(
    monkeypatch, capsys, tmp_path
) -> None:
    received: dict[str, object] = {}

    class FakeCheckRunner:
        def __init__(self, repo, **kwargs) -> None:
            received.update(repo=repo, **kwargs)

        def run(self) -> CheckRunResult:
            return _check_result()

    monkeypatch.setattr(cli, "CheckRunner", FakeCheckRunner)

    exit_code = cli.main(
        [
            "check",
            "--repo",
            str(tmp_path),
            "--allow-bootstrap",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["profile"]["id"] == "python-pytest"
    assert payload["phases"][0]["network"] == "none"
    assert received == {
        "repo": tmp_path,
        "allow_bootstrap": True,
        "phase_timeout_seconds": 300.0,
        "total_timeout_seconds": 1200.0,
        "max_output_bytes": 65536,
    }


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        ("failed", 1),
        ("timed_out", 1),
        ("bootstrap_required", 1),
        ("policy_denied", 1),
        ("setup_error", 2),
        ("cleanup_error", 2),
    ],
)
def test_check_status_controls_exit_code(
    monkeypatch, capsys, tmp_path, status, expected_exit
) -> None:
    result = _check_result(status)

    class FakeCheckRunner:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def run(self) -> CheckRunResult:
            return result

    monkeypatch.setattr(cli, "CheckRunner", FakeCheckRunner)

    exit_code = cli.main(
        ["check", "--repo", str(tmp_path), "--format", "json"]
    )

    assert exit_code == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_check_setup_exception_is_structured(monkeypatch, capsys, tmp_path) -> None:
    class BrokenCheckRunner:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ValueError("not a clean Git repository")

    monkeypatch.setattr(cli, "CheckRunner", BrokenCheckRunner)

    exit_code = cli.main(
        ["check", "--repo", str(tmp_path), "--format", "json"]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "setup_error"
    assert payload["phases"] == []
    assert payload["error"] == "ValueError: not a clean Git repository"


def test_interrupted_check_closes_runner_before_returning_130(
    monkeypatch, capsys, tmp_path
) -> None:
    closed = False

    class InterruptedCheckRunner:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def run(self) -> CheckRunResult:
            raise KeyboardInterrupt

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(cli, "CheckRunner", InterruptedCheckRunner)

    exit_code = cli.main(
        ["check", "--repo", str(tmp_path), "--format", "json"]
    )

    assert exit_code == 130
    assert closed
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "interrupted"
    assert payload["cleanup_ok"] is True


def test_text_check_lists_each_phase(monkeypatch, capsys, tmp_path) -> None:
    class FakeCheckRunner:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def run(self) -> CheckRunResult:
            return _check_result()

    monkeypatch.setattr(cli, "CheckRunner", FakeCheckRunner)

    exit_code = cli.main(["check", "--repo", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: passed" in output
    assert "profile: python-pytest (python)" in output
    assert "phase: python-pytest (verify, passed, network=none, exit=0" in output
    assert "command: python -m pytest -q" in output


def test_help_exposes_complete_day5_command_surface(capsys) -> None:
    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    for command in (
        "doctor",
        "run",
        "resume",
        "show",
        "cancel",
        "eval",
        "serve",
        "check",
    ):
        assert command in output


def test_doctor_json_reports_verified_native_tool_call(monkeypatch, capsys) -> None:
    sentinel_config = object()
    received: dict[str, object] = {}

    def fake_from_env(*, allow_remote_model):
        received["allow_remote_model"] = allow_remote_model
        return sentinel_config

    def fake_doctor(config):
        assert config is sentinel_config
        return DoctorResult(
            ok=True,
            api_kind="responses",
            model="test-model",
            tool_call_verified=True,
            final_response_verified=True,
            http_requests=2,
            message="Responses API native function calling verified",
        )

    monkeypatch.setattr(cli.OpenAIConfig, "from_env", fake_from_env)
    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    exit_code = cli.main(
        ["doctor", "--allow-remote-model", "--format", "json"]
    )

    assert exit_code == 0
    assert received == {"allow_remote_model": True}
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_kind"] == "responses"
    assert payload["tool_call_verified"] is True
    assert payload["final_response_verified"] is True


def test_real_run_uses_durable_service_and_task_file(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setenv("REPO_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("REPO_AGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("REPO_AGENT_MODEL", "test-model")
    task_file = tmp_path / "issue.txt"
    task_file.write_text("  repair the bug\n", encoding="utf-8")
    data_dir = tmp_path / "state"
    received: dict[str, object] = {}
    finished = _durable_record(tmp_path)

    class FakeDurableService:
        def create_run(self, **kwargs):
            received["create"] = kwargs
            return finished.model_copy(update={"status": "queued"})

        def wait(self, run_id, **kwargs):
            received.setdefault("wait", []).append((run_id, kwargs))
            return finished

        def close(self):
            received["closed"] = True

    service = FakeDurableService()

    def fake_new_service(selected_data_dir, **kwargs):
        received["service"] = (selected_data_dir, kwargs)
        return service

    monkeypatch.setattr(cli, "_new_run_service", fake_new_service)

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task-file",
            str(task_file),
            "--base-ref",
            "main~1",
            "--yes",
            "--allow-remote-model",
            "--allow-bootstrap",
            "--data-dir",
            str(data_dir),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    assert received["create"] == {
        "repo_path": tmp_path,
        "task": "repair the bug",
        "base_ref": "main~1",
        "auto_approve": True,
        "allow_remote_model": True,
    }
    selected_data_dir, service_options = received["service"]
    assert selected_data_dir == data_dir
    assert service_options == {
        "allowed_roots": (tmp_path,),
        "allow_bootstrap": True,
    }
    assert received["closed"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "a" * 32
    assert payload["status"] == "succeeded"


def test_real_run_fails_before_service_when_model_config_is_missing(
    monkeypatch, capsys, tmp_path
) -> None:
    for name in (
        "REPO_AGENT_API_KEY",
        "REPO_AGENT_BASE_URL",
        "REPO_AGENT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    called = False

    def unexpected_service(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("service must not start without model configuration")

    monkeypatch.setattr(cli, "_new_run_service", unexpected_service)

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "repair the bug",
            "--format",
            "json",
        ]
    )

    assert exit_code == 2
    assert called is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "setup_error"
    assert "REPO_AGENT_API_KEY" in payload["error"]


def test_typer_runner_receives_nonzero_command_exit_status(
    monkeypatch, tmp_path
) -> None:
    for name in (
        "REPO_AGENT_API_KEY",
        "REPO_AGENT_BASE_URL",
        "REPO_AGENT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "repair the bug",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "setup_error"


def test_real_run_prompts_then_approves_plan(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("REPO_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("REPO_AGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("REPO_AGENT_MODEL", "test-model")
    plan = ChangePlan(
        goal="Repair the bug",
        files=("src/app.py",),
        steps=("Apply the focused patch",),
        checks=("python-pytest",),
    )
    waiting = _durable_record(tmp_path, status="awaiting_approval", plan=plan)
    finished = _durable_record(tmp_path, plan=plan)
    calls: list[object] = []

    class FakeDurableService:
        def create_run(self, **kwargs):
            return waiting.model_copy(update={"status": "queued"})

        def wait(self, run_id, **kwargs):
            calls.append(("wait", run_id, kwargs))
            return waiting if len(calls) == 1 else finished

        def decide(self, run_id, **kwargs):
            calls.append(("decide", run_id, kwargs))
            return waiting.model_copy(update={"status": "queued"})

        def close(self):
            calls.append("close")

    monkeypatch.setattr(
        cli,
        "_new_run_service",
        lambda *args, **kwargs: FakeDurableService(),
    )
    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: True)

    exit_code = cli.main(
        ["run", "--repo", str(tmp_path), "--task", "repair the bug"]
    )

    assert exit_code == 0
    assert any(call[0] == "decide" for call in calls if isinstance(call, tuple))
    assert "change plan:" in capsys.readouterr().out


def test_real_run_reports_the_persisted_state_after_wait_timeout_and_close(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setenv("REPO_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("REPO_AGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("REPO_AGENT_MODEL", "test-model")
    queued = _durable_record(tmp_path, status="queued")
    running = _durable_record(tmp_path, status="running")
    cancelled = _durable_record(tmp_path, status="cancelled")
    calls: list[str] = []

    class FakeDurableService:
        current = running

        def create_run(self, **kwargs):
            return queued

        def wait(self, run_id, **kwargs):
            calls.append("wait")
            return running

        def close(self):
            calls.append("close")
            self.current = cancelled

        def get_run(self, run_id):
            calls.append("get_run")
            return self.current

    monkeypatch.setattr(
        cli,
        "_new_run_service",
        lambda *args, **kwargs: FakeDurableService(),
    )

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "repair the bug",
            "--wait-seconds",
            "0.01",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out)["status"] == "cancelled"
    assert calls == ["wait", "close", "get_run"]


def test_resume_approves_waiting_run(monkeypatch, capsys, tmp_path) -> None:
    plan = ChangePlan(
        goal="Repair the bug",
        files=("src/app.py",),
        steps=("Apply the focused patch",),
        checks=("python-pytest",),
    )
    waiting = _durable_record(tmp_path, status="awaiting_approval", plan=plan)
    finished = _durable_record(tmp_path, plan=plan)
    received: dict[str, object] = {}

    class FakeDurableService:
        current = waiting

        def get_run(self, run_id):
            assert run_id == waiting.run_id
            return self.current

        def decide(self, run_id, **kwargs):
            received["decision"] = (run_id, kwargs)
            self.current = finished
            return finished

        def close(self):
            received["closed"] = True

    monkeypatch.setattr(cli, "_load_record", lambda *args: waiting)
    monkeypatch.setattr(
        cli,
        "_new_run_service",
        lambda *args, **kwargs: FakeDurableService(),
    )

    exit_code = cli.main(
        [
            "resume",
            waiting.run_id,
            "--approve",
            "--reason",
            "reviewed",
            "--data-dir",
            str(tmp_path / "state"),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    assert received["decision"] == (
        waiting.run_id,
        {"approve": True, "reason": "reviewed"},
    )
    assert received["closed"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_resume_reports_the_persisted_state_after_wait_timeout_and_close(
    monkeypatch, capsys, tmp_path
) -> None:
    interrupted = _durable_record(tmp_path, status="interrupted")
    queued = _durable_record(tmp_path, status="queued")
    running = _durable_record(tmp_path, status="running")
    cancelled = _durable_record(tmp_path, status="cancelled")
    calls: list[str] = []

    class FakeDurableService:
        current = interrupted

        def get_run(self, run_id):
            calls.append("get_run")
            return self.current

        def resume(self, run_id):
            calls.append("resume")
            self.current = queued
            return queued

        def wait(self, run_id, **kwargs):
            calls.append("wait")
            self.current = running
            return running

        def close(self):
            calls.append("close")
            self.current = cancelled

    monkeypatch.setattr(cli, "_load_record", lambda *args: interrupted)
    monkeypatch.setattr(
        cli,
        "_new_run_service",
        lambda *args, **kwargs: FakeDurableService(),
    )

    exit_code = cli.main(
        [
            "resume",
            interrupted.run_id,
            "--data-dir",
            str(tmp_path / "state"),
            "--wait-seconds",
            "0.01",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out)["status"] == "cancelled"
    assert calls == ["get_run", "resume", "wait", "close", "get_run"]


def test_close_error_does_not_override_a_completed_run_result(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setenv("REPO_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("REPO_AGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("REPO_AGENT_MODEL", "test-model")
    queued = _durable_record(tmp_path, status="queued")
    succeeded = _durable_record(tmp_path, status="succeeded")

    class FakeDurableService:
        def create_run(self, **kwargs):
            return queued

        def wait(self, run_id, **kwargs):
            return succeeded

        def close(self):
            raise TimeoutError("worker is still stopping")

        def get_run(self, run_id):
            return succeeded

    monkeypatch.setattr(
        cli,
        "_new_run_service",
        lambda *args, **kwargs: FakeDurableService(),
    )

    exit_code = cli.main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--task",
            "repair the bug",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "succeeded"
    assert "cleanup warning: TimeoutError: worker is still stopping" in captured.err


def test_show_reads_record_without_starting_worker(capsys, tmp_path) -> None:
    data_dir = tmp_path / "state"
    record = _durable_record(tmp_path)
    artifacts = ArtifactStore(data_dir / "runs")
    artifact_dir = artifacts.initialize(record.run_id)
    database = RunDatabase(data_dir / "runs.sqlite3")
    database.create(record, artifact_dir)

    exit_code = cli.main(
        [
            "show",
            record.run_id,
            "--data-dir",
            str(data_dir),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == record.run_id
    assert payload["artifacts"]["patch"].endswith("patch.diff")


def test_cancel_uses_service_without_starting_worker(monkeypatch, capsys, tmp_path) -> None:
    queued = _durable_record(tmp_path, status="queued")
    cancelled = queued.model_copy(update={"status": "cancelled"})
    received: dict[str, object] = {}

    class FakeDurableService:
        def cancel(self, run_id):
            received["run_id"] = run_id
            return cancelled

        def close(self):
            received["closed"] = True

    def fake_new_service(*args, **kwargs):
        received["options"] = kwargs
        return FakeDurableService()

    monkeypatch.setattr(cli, "_load_record", lambda *args: queued)
    monkeypatch.setattr(cli, "_new_run_service", fake_new_service)

    exit_code = cli.main(
        [
            "cancel",
            queued.run_id,
            "--data-dir",
            str(tmp_path / "state"),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    assert received["run_id"] == queued.run_id
    assert received["options"]["start_worker"] is False
    assert received["closed"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "cancelled"


def test_eval_validates_frozen_suite_and_variant(capsys) -> None:
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"

    exit_code = cli.main(
        [
            "eval",
            "--variant",
            "no-review",
            "--benchmark-dir",
            str(benchmark_dir),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["variant"] == "no-review"
    assert payload["task_count"] == 12
    assert payload["execution_started"] is False


def test_eval_matrix_only_persists_exact_44_jobs_without_loading_model(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("matrix-only must not load model credentials")

    monkeypatch.setattr(cli.OpenAIConfig, "from_env", fail_if_loaded)
    exit_code = cli.main(
        [
            "eval",
            "--matrix-only",
            "--benchmark-dir",
            str(benchmark_dir),
            "--output-dir",
            str(tmp_path / "state"),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    jobs = payload["jobs"]
    identities = {
        (job["variant"], job["trial"], job["task_id"]) for job in jobs
    }
    assert payload["job_count"] == 44
    assert len(jobs) == len(identities) == 44
    persisted = json.loads(
        (tmp_path / "state" / "matrix.json").read_text(encoding="utf-8")
    )
    assert persisted == payload


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "--variant is required"),
        (["--matrix"], "--matrix requires --execute"),
        (["--canary"], "--canary requires --execute"),
        (
            ["--matrix-only", "--variant", "full"],
            "--matrix-only does not accept",
        ),
    ],
)
def test_eval_experiment_modes_fail_closed_with_structured_errors(
    arguments, message, capsys
) -> None:
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"

    exit_code = cli.main(
        [
            "eval",
            *arguments,
            "--benchmark-dir",
            str(benchmark_dir),
            "--format",
            "json",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert message in payload["error"]


def test_serve_api_requires_token_and_forwards_security_configuration(
    monkeypatch, tmp_path
) -> None:
    received: dict[str, object] = {}

    class FakeDurableService:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_new_service(*args, **kwargs):
        received["service"] = (args, kwargs)
        return FakeDurableService()

    def fake_serve(service, **kwargs):
        received["api"] = (service, kwargs)

    monkeypatch.setattr(cli, "_new_run_service", fake_new_service)
    monkeypatch.setattr(cli, "serve_api", fake_serve)

    exit_code = cli.main(
        [
            "serve",
            "--api",
            "--repo",
            str(tmp_path),
            "--allowed-root",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "state"),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--token",
            "test-token",
        ]
    )

    assert exit_code == 0
    _, service_options = received["service"]
    assert service_options["allowed_roots"] == (tmp_path,)
    _, api_options = received["api"]
    assert api_options == {
        "bearer_token": "test-token",
        "allowed_roots": (tmp_path,),
        "host": "127.0.0.1",
        "port": 8080,
    }


def test_new_run_service_defers_bootstrap_runner_creation_until_after_lease(
    monkeypatch, tmp_path: Path
) -> None:
    received: dict[str, object] = {}
    sentinel = object()

    def fake_run_service(data_dir, **kwargs):
        received["data_dir"] = data_dir
        received["options"] = kwargs
        return sentinel

    monkeypatch.setenv("REPO_AGENT_API_KEY", "  private-key  ")
    monkeypatch.setattr(cli, "RunService", fake_run_service)

    result = cli._new_run_service(
        tmp_path / "data",
        allowed_roots=(tmp_path,),
        allow_bootstrap=True,
        start_worker=False,
    )

    assert result is sentinel
    assert received == {
        "data_dir": tmp_path / "data",
        "options": {
            "allowed_repo_roots": (tmp_path,),
            "start_worker": False,
            "allow_bootstrap": True,
            "secrets": ("private-key",),
        },
    }
