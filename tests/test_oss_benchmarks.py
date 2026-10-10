from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from repo_agent import cli
from repo_agent import evaluation as evaluation_module
from repo_agent.evaluation import (
    AgentExecution,
    AgentExecutionArtifact,
    EvaluationRunner,
    ModelUsageRecord,
    ScoreOutcome,
    canary_jobs,
    experiment_jobs,
    load_benchmark_suite,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OSS_BENCHMARKS = PROJECT_ROOT / "benchmarks" / "tasks" / "python" / "oss"
OSS_TASK_IDS = tuple(f"py-oss-{index:03d}" for index in range(1, 4))


def _load_oss_validator():
    path = OSS_BENCHMARKS / "validate.py"
    spec = importlib.util.spec_from_file_location("oss_benchmark_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_validator(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(OSS_BENCHMARKS / "validate.py"), *arguments),
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_oss_benchmark_structure_and_provenance_locks() -> None:
    completed = _run_validator("--structure-only")

    assert completed.returncode == 0, completed.stdout
    assert "[ok] public OSS suite structure, provenance, and locks" in completed.stdout


def test_oss_benchmark_rejects_unlocked_generated_artifacts(tmp_path: Path) -> None:
    copied_suite = tmp_path / "oss"
    shutil.copytree(
        OSS_BENCHMARKS,
        copied_suite,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[co]", "*.pyd"),
    )
    injected = (
        copied_suite
        / "pysnooper-output-path"
        / "baseline"
        / "sitecustomize.pyc"
    )
    injected.write_bytes(b"unlocked executable bytes")

    completed = subprocess.run(
        (sys.executable, str(copied_suite / "validate.py"), "--structure-only"),
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 1
    assert "generated artifacts are not allowed" in completed.stdout


def test_oss_benchmark_requires_attributed_hidden_failure() -> None:
    validator = _load_oss_validator()
    completed = subprocess.CompletedProcess(
        args=("python", "-m", "unittest"),
        returncode=1,
        stdout=(
            "test_expected (test_regression.Tests.test_expected) ... ok\n"
            "test_other (test_regression.Tests.test_other) ... FAIL\n\n"
            "FAIL: test_other (test_regression.Tests.test_other)\n"
        ),
    )

    with pytest.raises(
        validator.ValidationError,
        match="did not attribute a failure to test_expected",
    ):
        validator._require_result(
            completed,
            should_pass=False,
            task_id="py-oss-001",
            stage="setup hidden tests",
            signatures=("test_expected",),
        )


def test_oss_benchmark_digest_rejects_windows_reparse_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_oss_validator()
    task = tmp_path / "task"
    junction = task / "junction"
    junction.mkdir(parents=True)
    (junction / "external.py").write_text("value = 1\n", encoding="utf-8")
    real_lstat = validator.os.lstat
    reparse_flag = 0x400
    monkeypatch.setattr(
        validator.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        reparse_flag,
        raising=False,
    )
    marked_path = task

    def fake_lstat(path, *args, **kwargs):
        information = real_lstat(path, *args, **kwargs)
        if Path(path) == marked_path:
            return SimpleNamespace(
                st_mode=information.st_mode,
                st_file_attributes=reparse_flag,
            )
        return information

    monkeypatch.setattr(validator.os, "lstat", fake_lstat)

    with pytest.raises(validator.ValidationError, match="links are not allowed"):
        validator._task_digest(task)

    marked_path = junction
    with pytest.raises(validator.ValidationError, match="links are not allowed"):
        validator._task_digest(task)


def test_oss_benchmark_zero_model_lifecycle() -> None:
    completed = _run_validator()

    assert completed.returncode == 0, completed.stdout
    for task_id in OSS_TASK_IDS:
        assert f"[ok] {task_id} " in completed.stdout
    assert "[ok] validated 3 public OSS issue lifecycles" in completed.stdout


def test_oss_suite_loads_only_for_single_mode() -> None:
    suite = load_benchmark_suite(OSS_BENCHMARKS)

    assert suite.suite_id == "repo-agent-python-oss-v1"
    assert suite.formal_evaluation is False
    assert tuple(task.id for task in suite.tasks) == OSS_TASK_IDS
    assert all(task.setup_patch == task.root / "setup.patch" for task in suite.tasks)
    assert (
        load_benchmark_suite(OSS_BENCHMARKS, task_ids=("py-oss-002",)).tasks[0].id
        == "py-oss-002"
    )
    with pytest.raises(evaluation_module.EvaluationError, match="non-formal"):
        experiment_jobs(OSS_BENCHMARKS)
    with pytest.raises(evaluation_module.EvaluationError, match="non-formal"):
        canary_jobs(OSS_BENCHMARKS, task_id="py-oss-001")


def test_oss_suite_cli_dry_run_does_not_start_model(capsys) -> None:
    exit_code = cli.main(
        [
            "eval",
            "--benchmark-dir",
            str(OSS_BENCHMARKS),
            "--variant",
            "full",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "validated",
        "variant": "full",
        "suite_id": "repo-agent-python-oss-v1",
        "task_count": 3,
        "benchmark_dir": str(OSS_BENCHMARKS.resolve()),
        "manifest_sha256": payload["manifest_sha256"],
        "execution_started": False,
        "formal_evaluation": False,
    }
    assert len(payload["manifest_sha256"]) == 64


def test_oss_suite_zero_model_execution_persists_result_id(tmp_path: Path) -> None:
    task_root = OSS_BENCHMARKS / "pysnooper-output-path"
    candidate_patch = (task_root / "gold.patch").read_text(encoding="utf-8")
    calls: list[str] = []

    def execute(task, _workspace, _strategy, _deadline_epoch):
        calls.append(task.id)
        return AgentExecutionArtifact(
            execution=AgentExecution(
                run_id="a" * 32,
                status="succeeded",
                public_passed=True,
                tool_calls=1,
                duration_ms=1,
                usage=ModelUsageRecord(
                    response_count=1,
                    reported_response_count=1,
                    input_tokens=10,
                    cached_input_tokens=0,
                    output_tokens=5,
                    total_tokens=15,
                    complete=True,
                ),
            ),
            patch=candidate_patch,
        )

    def score(task, patch):
        assert task.id == "py-oss-001"
        assert patch == candidate_patch
        return ScoreOutcome(
            public_passed=True,
            hidden_passed=True,
            original_tests_unchanged=True,
            build_unchanged=True,
            regression_test_added=True,
            regression_failed_on_buggy=True,
            regression_passed_on_candidate=True,
            policy_passed=True,
        )

    options = {
        "variant": "full",
        "task_ids": ("py-oss-001",),
        "model": "offline-test-model",
        "agent_executor": execute,
        "score_executor": score,
    }
    first = EvaluationRunner(
        OSS_BENCHMARKS, tmp_path / "results", **options
    ).run()
    second = EvaluationRunner(
        OSS_BENCHMARKS, tmp_path / "results", **options
    ).run()

    assert calls == ["py-oss-001"]
    assert first.solved_task_count == 1
    assert first.tasks[0].task_id == "py-oss-001"
    assert second.skipped_task_count == 1
    assert second.tasks[0].model_dump() == first.tasks[0].model_dump()
