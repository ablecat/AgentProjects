from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import typer

from repo_agent import cli


@dataclass(frozen=True)
class _ReportStub:
    payload: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.payload


def _set_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPO_AGENT_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("REPO_AGENT_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("REPO_AGENT_MODEL", "unit-test-model")


def _batch_stub(
    *,
    mode: str,
    status: str = "completed",
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> _ReportStub:
    return _ReportStub(
        {
            "schema_version": 1,
            "experiment_id": "repo-agent-day7-v1",
            "mode": mode,
            "suite_id": "suite-v1",
            "manifest_sha256": "a" * 64,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "workers": 1,
            "status": status,
            "job_count": 3,
            "completed_job_count": 3 if status == "completed" else 2,
            "result_dir": "evaluation-results/batch",
            "jobs": [],
            "reports": [],
            "completed_at": "2026-09-11T00:00:00Z",
        }
    )


def _experiment_stub(*, status: str = "completed") -> _ReportStub:
    return _ReportStub(
        {
            "schema_version": 1,
            "experiment_id": "repo-agent-day7-v1",
            "status": status,
            "job_count": 44,
            "completed_job_count": 44 if status == "completed" else 43,
            "solved_job_count": 30,
            "pass_at_1": 0.75,
            "price_source": "unavailable",
            "cost_usd": None,
            "results_path": "publish/results.jsonl",
            "report_path": "publish/report.json",
            "failure_analysis_path": "publish/failure-analysis.json",
        }
    )


def test_matrix_only_writes_manifest_and_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    received: dict[str, Any] = {}

    def fake_write(benchmark_dir, output_dir, *, secrets):
        received.update(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            secrets=secrets,
        )
        return {
            "status": "validated",
            "experiment_id": "repo-agent-day7-v1",
            "suite_id": "suite-v1",
            "execution_started": False,
            "job_count": 44,
            "jobs": [
                {"variant": "baseline", "trial": 1, "task_id": "py-bugfix-001"}
            ],
        }

    monkeypatch.setenv("REPO_AGENT_API_KEY", "matrix-only-secret")
    monkeypatch.setattr(cli, "write_experiment_matrix_manifest", fake_write)
    benchmark_dir = tmp_path / "benchmarks"
    output_dir = tmp_path / "results"

    exit_code = cli.main(
        [
            "eval",
            "--matrix-only",
            "--benchmark-dir",
            str(benchmark_dir),
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["job_count"] == 44
    assert payload["execution_started"] is False
    assert "matrix-only-secret" not in captured.out
    assert received == {
        "benchmark_dir": benchmark_dir,
        "output_dir": output_dir,
        "secrets": ("matrix-only-secret",),
    }


def test_matrix_only_emits_each_valid_job_as_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "status": "validated",
        "experiment_id": "repo-agent-day7-v1",
        "suite_id": "suite-v1",
        "job_count": 44,
        "jobs": [
            "ignored-invalid-job",
            {"variant": "full", "trial": 3, "task_id": "java-bugfix-006"},
        ],
    }
    monkeypatch.setattr(
        cli,
        "write_experiment_matrix_manifest",
        lambda *args, **kwargs: payload,
    )

    exit_code = cli.main(["eval", "--matrix-only"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: validated" in output
    assert "experiment: repo-agent-day7-v1" in output
    assert "suite: suite-v1" in output
    assert "jobs: 44" in output
    assert "job: full trial-3 java-bugfix-006" in output
    assert "ignored-invalid-job" not in output

    cli._emit_experiment_matrix({**payload, "jobs": ()}, "text")
    assert "jobs: 44" in capsys.readouterr().out


def test_matrix_shard_forwards_contract_and_emits_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _set_model_environment(monkeypatch)
    received: dict[str, Any] = {}

    def fake_run(benchmark_dir, output_dir, **kwargs):
        received.update(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            **kwargs,
        )
        return _batch_stub(mode="shard", shard_index=3, shard_count=5)

    monkeypatch.setattr(cli, "run_experiment_shard", fake_run)
    output_dir = tmp_path / "results"

    exit_code = cli.main(
        [
            "eval",
            "--matrix",
            "--execute",
            "--shard-index",
            "3",
            "--shard-count",
            "5",
            "--workers",
            "1",
            "--allow-remote-model",
            "--allow-bootstrap",
            "--task-timeout-seconds",
            "42",
            "--max-output-bytes",
            "2048",
            "--input-cost-per-million",
            "1.25",
            "--output-cost-per-million",
            "2.5",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: completed" in output
    assert "mode: shard" in output
    assert "shard: 3/5" in output
    assert "jobs: 3/3 completed" in output
    assert received["output_dir"] == output_dir
    assert received["shard_index"] == 3
    assert received["shard_count"] == 5
    assert received["workers"] == 1
    assert received["allow_remote_model"] is True
    assert received["allow_bootstrap"] is True
    assert received["task_timeout_seconds"] == 42.0
    assert received["max_output_bytes"] == 2048
    assert received["input_cost_per_million"] == 1.25
    assert received["output_cost_per_million"] == 2.5
    assert received["secrets"] == ("unit-test-api-key",)
    assert received["model_config"].model == "unit-test-model"
    assert received["model_config"].allow_remote_model is True


def test_canary_forwards_custom_task_and_partial_exit_as_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _set_model_environment(monkeypatch)
    received: dict[str, Any] = {}

    def fake_run(benchmark_dir, output_dir, **kwargs):
        received.update(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            **kwargs,
        )
        return _batch_stub(mode="canary", status="partial")

    monkeypatch.setattr(cli, "run_experiment_canary", fake_run)

    exit_code = cli.main(
        [
            "eval",
            "--canary",
            "--execute",
            "--task-id",
            "java-bugfix-004",
            "--workers",
            "1",
            "--output-dir",
            str(tmp_path / "results"),
            "--format",
            "json",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "canary"
    assert payload["status"] == "partial"
    assert payload["completed_job_count"] == 2
    assert received["task_id"] == "java-bugfix-004"
    assert received["workers"] == 1
    assert received["model_config"].model == "unit-test-model"


def test_merge_forwards_publish_directory_and_emits_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    received: dict[str, Any] = {}

    def fake_merge(benchmark_dir, output_dir, *, report_dir, secrets):
        received.update(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            report_dir=report_dir,
            secrets=secrets,
        )
        return _experiment_stub()

    monkeypatch.delenv("REPO_AGENT_API_KEY", raising=False)
    monkeypatch.setattr(cli, "merge_experiment_results", fake_merge)
    output_dir = tmp_path / "results"
    report_dir = tmp_path / "publish"

    exit_code = cli.main(
        [
            "eval",
            "--merge",
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: completed" in output
    assert "jobs: 44/44 completed" in output
    assert "pass_at_1: 0.750" in output
    assert "price_source: unavailable" in output
    assert "cost_usd: -" in output
    assert "results: publish/results.jsonl" in output
    assert received == {
        "benchmark_dir": Path("benchmarks"),
        "output_dir": output_dir,
        "report_dir": report_dir,
        "secrets": (),
    }


def test_merge_partial_emits_one_json_document_and_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "merge_experiment_results",
        lambda *args, **kwargs: _experiment_stub(status="partial"),
    )

    exit_code = cli.main(["eval", "--merge", "--format", "json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "partial"
    assert payload["completed_job_count"] == 43


def test_single_variant_validation_emits_text_and_forwards_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    received: dict[str, Any] = {}

    def fake_validation(benchmark_dir, variant, *, task_ids, secrets):
        received.update(
            benchmark_dir=benchmark_dir,
            variant=variant,
            task_ids=task_ids,
            secrets=secrets,
        )
        return {
            "status": "validated",
            "variant": variant,
            "suite_id": "suite-v1",
            "task_count": 1,
            "benchmark_dir": str(benchmark_dir),
            "execution_started": False,
        }

    monkeypatch.setenv("REPO_AGENT_API_KEY", "validation-secret")
    monkeypatch.setattr(cli, "validation_manifest", fake_validation)
    benchmark_dir = tmp_path / "benchmarks"

    exit_code = cli.main(
        [
            "eval",
            "--variant",
            "no-review",
            "--task-id",
            "py-bugfix-005",
            "--benchmark-dir",
            str(benchmark_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status: validated" in output
    assert "variant: no-review" in output
    assert "suite: suite-v1" in output
    assert "tasks: 1" in output
    assert "execution: not started" in output
    assert "validation-secret" not in output
    assert received == {
        "benchmark_dir": benchmark_dir,
        "variant": "no-review",
        "task_ids": ("py-bugfix-005",),
        "secrets": ("validation-secret",),
    }


def test_single_variant_execute_forwards_options_and_emits_task_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _set_model_environment(monkeypatch)
    received: dict[str, Any] = {}
    task_results = (
        SimpleNamespace(
            solved=True,
            failure_category=None,
            task_id="py-bugfix-001",
            agent=SimpleNamespace(duration_ms=100),
            patch_bytes=20,
        ),
        SimpleNamespace(
            solved=False,
            failure_category="hidden_tests_failed",
            task_id="py-bugfix-002",
            agent=SimpleNamespace(duration_ms=200),
            patch_bytes=30,
        ),
        SimpleNamespace(
            solved=False,
            failure_category=None,
            task_id="py-bugfix-003",
            agent=SimpleNamespace(duration_ms=300),
            patch_bytes=0,
        ),
    )
    report = _ReportStub(
        {
            "status": "completed",
            "variant": "full",
            "trial": 1,
            "suite_id": "suite-v1",
            "completed_task_count": 3,
            "task_count": 3,
            "skipped_task_count": 1,
            "pass_at_1": None,
            "solved_task_count": 1,
            "tool_error_rate": 0.125,
            "actionable_tool_error_rate": 0.025,
            "result_dir": "single-results",
            "tasks": task_results,
        }
    )

    class FakeRunner:
        def __init__(self, benchmark_dir, output_dir, **kwargs) -> None:
            received.update(
                benchmark_dir=benchmark_dir,
                output_dir=output_dir,
                **kwargs,
            )

        def __enter__(self):
            received["entered"] = True
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
            received["exited"] = True
            return False

        def run(self):
            return report

    monkeypatch.setattr(cli, "EvaluationRunner", FakeRunner)
    output_dir = tmp_path / "single-results"

    exit_code = cli.main(
        [
            "eval",
            "--variant",
            "full",
            "--execute",
            "--task-id",
            "py-bugfix-001",
            "--allow-bootstrap",
            "--task-timeout-seconds",
            "90",
            "--max-output-bytes",
            "4096",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "tasks: 3/3 completed, 1 reused" in output
    assert "pass_at_1: -" in output
    assert "tool_error_rate: 0.125" in output
    assert "actionable_tool_error_rate: 0.025" in output
    assert "task: py-bugfix-001 (pass, 100ms, 20 patch bytes)" in output
    assert "task: py-bugfix-002 (hidden_tests_failed, 200ms, 30 patch bytes)" in output
    assert "task: py-bugfix-003 (fail, 300ms, 0 patch bytes)" in output
    assert received["output_dir"] == output_dir
    assert received["variant"] == "full"
    assert received["task_ids"] == ("py-bugfix-001",)
    assert received["allow_bootstrap"] is True
    assert received["task_timeout_seconds"] == 90.0
    assert received["max_output_bytes"] == 4096
    assert received["entered"] is True
    assert received["exited"] is True
    assert received["model_config"].model == "unit-test-model"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "--variant is required unless using an experiment matrix mode"),
        (
            ["--matrix-only", "--canary"],
            "choose only one of --matrix-only, --matrix, --canary, and --merge",
        ),
        (
            ["--matrix-only", "--execute"],
            "--matrix-only does not accept --execute, --variant, or --task-id",
        ),
        (
            ["--merge", "--task-id", "py-bugfix-001"],
            "--merge does not accept --execute, --variant, or --task-id",
        ),
        (["--matrix"], "--matrix requires --execute"),
        (["--matrix", "--execute"], "--matrix requires --shard-index"),
        (
            ["--matrix", "--execute", "--shard-index", "0", "--variant", "full"],
            "--matrix does not accept --variant or --task-id",
        ),
        (["--canary"], "--canary requires --execute"),
        (
            ["--canary", "--execute", "--variant", "full"],
            "--canary accepts at most one --task-id and requires trial 1",
        ),
        (
            ["--canary", "--execute", "--trial", "2"],
            "--canary accepts at most one --task-id and requires trial 1",
        ),
        (
            [
                "--canary",
                "--execute",
                "--task-id",
                "py-bugfix-001",
                "--task-id",
                "py-bugfix-002",
            ],
            "--canary accepts at most one --task-id and requires trial 1",
        ),
        (
            ["--variant", "full", "--report-dir", "publish"],
            "--report-dir is only valid with --merge",
        ),
    ],
)
def test_eval_mode_combinations_return_structured_errors(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["eval", *arguments, "--format", "json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "eval"
    assert payload["status"] == "failed"
    assert payload["error"] == f"ValueError: {message}"


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval", "--matrix", "--execute", "--shard-index", "-1"],
        ["eval", "--canary", "--execute", "--workers", "3"],
        ["eval", "--matrix-only", "--task-timeout-seconds", "nan"],
    ],
)
def test_eval_typer_rejects_out_of_range_values(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code == 2


def test_validate_evaluation_suite_rejects_unknown_variant_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported evaluation variant"):
        cli._validate_evaluation_suite(tmp_path, "unknown")

    expected = {"status": "validated"}
    received: dict[str, Any] = {}

    def fake_validation(benchmark_dir, variant):
        received.update(benchmark_dir=benchmark_dir, variant=variant)
        return expected

    monkeypatch.setattr(cli, "validation_manifest", fake_validation)

    assert cli._validate_evaluation_suite(tmp_path, "baseline") is expected
    assert received == {"benchmark_dir": tmp_path, "variant": "baseline"}


def test_resolve_task_rejects_conflicting_and_unsafe_task_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(typer.BadParameter, match="use exactly one"):
        cli._resolve_task("task", tmp_path / "task.txt")
    with pytest.raises(typer.BadParameter, match="could not read task file"):
        cli._resolve_task(None, tmp_path / "missing.txt")

    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * (cli.MAX_TASK_FILE_BYTES + 1))
    with pytest.raises(typer.BadParameter, match="at most"):
        cli._resolve_task(None, oversized)

    invalid_utf8 = tmp_path / "invalid.txt"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(typer.BadParameter, match="UTF-8"):
        cli._resolve_task(None, invalid_utf8)

    with pytest.raises(typer.BadParameter, match="no NUL"):
        cli._resolve_task("unsafe\x00task", None)


def test_matrix_keyboard_interrupt_is_structured_and_retains_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_model_environment(monkeypatch)

    def interrupt(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_experiment_shard", interrupt)

    exit_code = cli.main(
        [
            "eval",
            "--matrix",
            "--execute",
            "--shard-index",
            "0",
            "--format",
            "json",
        ]
    )

    assert exit_code == 130
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "interrupted"
    assert payload["mode"] == "matrix"
    assert payload["variant"] is None
    assert payload["trial"] == 1
