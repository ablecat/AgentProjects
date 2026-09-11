from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import pytest

from repo_agent import evaluation as evaluation_module
from repo_agent.checks import CheckPhaseResult, CheckRunResult
from repo_agent.evaluation import (
    AgentExecution,
    AgentExecutionArtifact,
    BenchmarkSuite,
    EXPECTED_EXPERIMENT_COUNTS,
    ExperimentJob,
    EvaluationReport,
    EvaluationRunner,
    EvaluationTaskResult,
    FULL_REPEAT_TASK_IDS,
    ModelUsageRecord,
    ScoreOutcome,
    canary_jobs,
    experiment_jobs,
    jobs_for_shard,
    load_benchmark_suite,
    merge_experiment_results,
    run_experiment_canary,
    run_experiment_shard,
    write_experiment_matrix_manifest,
)
from repo_agent.openai_provider import OpenAIConfig
from repo_agent.openai_provider import ProviderUsage
from repo_agent.patches import PatchFile, ValidatedPatch
from repo_agent.models import ToolCall


PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARKS = PROJECT_ROOT / "benchmarks"


def _model_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key="test-evaluation-key",
        base_url="http://127.0.0.1:9999/v1",
        model="test-model",
    )


def _synthetic_result(
    job: ExperimentJob,
    *,
    suite: BenchmarkSuite,
    index: int,
    solved: bool,
    state_root: Path | None = None,
) -> EvaluationTaskResult:
    task = next(task for task in suite.tasks if task.id == job.task_id)
    digest = f"{index + 1:064x}"[-64:]
    return EvaluationTaskResult(
        suite_id=suite.suite_id,
        manifest_sha256=suite.manifest_sha256,
        task_id=task.id,
        task_sha256=task.digest,
        variant=job.variant,
        trial=job.trial,
        model="test-model",
        language=task.language,
        category=task.category,
        difficulty=task.difficulty,
        status="completed",
        solved=solved,
        agent=AgentExecution(
            run_id=f"{index + 1:032x}"[-32:],
            status="succeeded",
            public_passed=solved,
            tool_calls=2,
            tool_errors=0 if solved else 1,
            duration_ms=100 + index,
            usage=ModelUsageRecord(
                response_count=1,
                reported_response_count=1,
                input_tokens=10,
                cached_input_tokens=2,
                output_tokens=5,
                total_tokens=15,
                complete=True,
            ),
        ),
        score=ScoreOutcome(
            public_passed=True,
            hidden_passed=solved,
            original_tests_unchanged=True,
            build_unchanged=True,
            regression_test_added=True,
            regression_failed_on_buggy=True,
            regression_passed_on_candidate=solved,
            policy_passed=True,
            public_check_sha256=digest,
            hidden_check_sha256=digest,
            regression_check_sha256=digest,
        ),
        patch_sha256=digest,
        candidate_artifact_sha256=digest,
        public_check_sha256=digest,
        hidden_check_sha256=digest,
        regression_check_sha256=digest,
        patch_bytes=100,
        changed_files=2,
        added_lines=8,
        removed_lines=2,
        changed_lines=10,
        budget_passed=True,
        cost_usd=None,
        price_source="unavailable",
        failure_category=None if solved else "hidden_tests_failed",
        error=(
            None
            if solved or state_root is None
            else (
                f"failed in {state_root}\\workspaces\\{job.task_id}; "
                f"benchmark {BENCHMARKS}"
            )
        ),
        completed_at="2026-09-11T00:00:00.000Z",
    )


def _synthetic_patch(language: str) -> str:
    if language == "python":
        production = "tasklib/value.py"
        regression = "tests/test_synthetic_regression.py"
        test_line = "+def test_synthetic(): pass"
    else:
        production = "src/main/java/example/Value.java"
        regression = "src/test/java/example/SyntheticRegressionTest.java"
        test_line = "+final class SyntheticRegressionTest {}"
    return (
        f"diff --git a/{production} b/{production}\n"
        f"--- a/{production}\n"
        f"+++ b/{production}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        f"diff --git a/{regression} b/{regression}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{regression}\n"
        "@@ -0,0 +1 @@\n"
        f"{test_line}\n"
    )


def _write_check_artifact(path: Path, result: CheckRunResult) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation_module.hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_formal_results(
    state_root: Path,
) -> tuple[tuple[ExperimentJob, ...], dict[tuple[str, int, str], Path]]:
    jobs = experiment_jobs(BENCHMARKS)
    suite = load_benchmark_suite(BENCHMARKS)
    write_experiment_matrix_manifest(BENCHMARKS, state_root)
    evaluation_module._write_or_validate_formal_manifest(
        state_root,
        suite=suite,
        jobs=jobs,
        model_config=_model_config(),
        allow_remote_model=False,
        allow_bootstrap=False,
        task_timeout_seconds=1200,
        max_output_bytes=65536,
        input_cost_per_million=None,
        output_cost_per_million=None,
        secrets=("test-evaluation-key",),
    )
    grouped: dict[tuple[int, str, int], list[str]] = {}
    for index, job in enumerate(jobs):
        grouped.setdefault((index % 5, job.variant, job.trial), []).append(job.task_id)
    for (shard_index, variant, trial), task_ids in grouped.items():
        group_dir = (
            state_root
            / "shards"
            / f"shard-{shard_index}-of-5"
            / suite.suite_id
            / variant
            / f"trial-{trial}"
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        manifest = evaluation_module._group_manifest_payload(
            suite=suite,
            variant=variant,
            trial=trial,
            model="test-model",
            selected_task_ids=task_ids,
            task_timeout_seconds=1200,
            max_output_bytes=65536,
            input_cost_per_million=None,
            output_cost_per_million=None,
        )
        (group_dir / "experiment.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    paths: dict[tuple[str, int, str], Path] = {}
    for index, job in enumerate(jobs):
        task = next(task for task in suite.tasks if task.id == job.task_id)
        solved = job.trial > 1
        group_dir = (
            state_root
            / "shards"
            / f"shard-{index % 5}-of-5"
            / suite.suite_id
            / job.variant
            / f"trial-{job.trial}"
        )
        patch = _synthetic_patch(task.language)
        candidate_path = group_dir / "candidates" / f"{job.task_id}.patch"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(patch, encoding="utf-8", newline="\n")
        candidate_digest = evaluation_module.hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest()
        validated = evaluation_module.validate_patch(patch)
        phase = CheckPhaseResult(
            "verify", "verify", "passed", "none", ("check",), 0, "", None, 1
        )
        public = CheckRunResult("passed", None, "a" * 40, True, (phase,), 2)
        hidden = CheckRunResult(
            "passed" if solved else "failed",
            None,
            "b" * 40,
            True,
            (phase,),
            2,
        )
        regression_name = (
            "tests/test_synthetic_regression.py FAILED"
            if task.language == "python"
            else "Failures:\nSyntheticRegressionTest"
        )
        regression_phase = CheckPhaseResult(
            "verify",
            "verify",
            "failed",
            "none",
            ("check",),
            1,
            regression_name,
            None,
            1,
        )
        regression = CheckRunResult(
            "failed", None, "c" * 40, True, (regression_phase,), 2
        )
        scoring_dir = group_dir / "scoring" / job.task_id
        public_digest = _write_check_artifact(scoring_dir / "public.json", public)
        hidden_digest = _write_check_artifact(scoring_dir / "hidden.json", hidden)
        regression_digest = _write_check_artifact(
            scoring_dir / "regression.json", regression
        )
        result = _synthetic_result(
            job,
            suite=suite,
            index=index,
            solved=solved,
            state_root=state_root,
        )
        assert result.score is not None
        result = result.model_copy(
            update={
                "score": result.score.model_copy(
                    update={
                        "public_check_sha256": public_digest,
                        "hidden_check_sha256": hidden_digest,
                        "regression_check_sha256": regression_digest,
                    }
                ),
                "patch_sha256": evaluation_module.hashlib.sha256(
                    validated.text.encode("utf-8")
                ).hexdigest(),
                "candidate_artifact_sha256": candidate_digest,
                "public_check_sha256": public_digest,
                "hidden_check_sha256": hidden_digest,
                "regression_check_sha256": regression_digest,
                "patch_bytes": len(validated.text.encode("utf-8")),
                "changed_files": len(validated.files),
                "added_lines": sum(item.added_lines for item in validated.files),
                "removed_lines": sum(item.removed_lines for item in validated.files),
                "changed_lines": validated.changed_lines,
            }
        )
        path = (
            group_dir
            / "tasks"
            / f"{job.task_id}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        paths[job.identity] = path
    return jobs, paths


def test_committed_experiment_matrix_is_locked_to_44_runs() -> None:
    manifest = json.loads(
        (BENCHMARKS / "experiment-v1.json").read_text(encoding="utf-8")
    )
    matrix = manifest["matrix"]

    assert {
        "baseline": matrix["baseline"]["run_count"],
        "no-review": matrix["no-review"]["run_count"],
        "full": matrix["full"]["run_count"],
        "total": matrix["total_run_count"],
    } == EXPECTED_EXPERIMENT_COUNTS
    assert tuple(matrix["full"]["extra_trial_task_ids"]) == FULL_REPEAT_TASK_IDS


def test_suite_loader_returns_no_hidden_contract_or_gold_content() -> None:
    suite = load_benchmark_suite(BENCHMARKS, task_ids=("py-bugfix-001",))

    task = suite.tasks[0]
    assert task.id == "py-bugfix-001"
    assert "failure_signatures" not in repr(task)
    assert "gold.patch" not in repr(task)
    assert "test_trailing_separators" not in repr(task)


def test_runner_materializes_public_setup_and_skips_completed_result(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []
    secret = "sk-evaluation-secret-that-must-not-leak"
    gold_patch = (
        BENCHMARKS
        / "tasks"
        / "python"
        / "bugfix"
        / "trailing-separator"
        / "gold.patch"
    ).read_text(encoding="utf-8")

    def execute(task, workspace, strategy, deadline):
        del task, deadline
        calls.append(workspace)
        assert strategy.name == "full"
        assert not (workspace / "hidden_tests").exists()
        assert not (workspace / "gold.patch").exists()
        assert "lstrip" in (workspace / "tasklib" / "text.py").read_text(
            encoding="utf-8"
        )
        return AgentExecutionArtifact(
            AgentExecution(
                run_id="a" * 32,
                status="succeeded",
                public_passed=True,
                tool_calls=3,
                tool_errors=1,
                duration_ms=25,
                usage=ModelUsageRecord(
                    response_count=2,
                    reported_response_count=2,
                    input_tokens=100,
                    cached_input_tokens=30,
                    output_tokens=20,
                    total_tokens=120,
                    complete=True,
                ),
                error=f"diagnostic accidentally contained {secret}",
            ),
            gold_patch,
        )

    def score(task, patch):
        assert task.id == "py-bugfix-001"
        assert patch == gold_patch
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
        "task_ids": ("py-bugfix-001",),
        "model": "test-model",
        "secrets": (secret,),
        "agent_executor": execute,
        "score_executor": score,
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 2.0,
    }
    first = EvaluationRunner(BENCHMARKS, tmp_path / "results", **options).run()
    second = EvaluationRunner(BENCHMARKS, tmp_path / "results", **options).run()

    assert len(calls) == 1
    assert first.status == "completed"
    assert first.solved_task_count == 1
    assert first.pass_at_1 == 1.0
    assert first.tool_error_rate == pytest.approx(1 / 3)
    assert first.tasks[0].changed_files == 1
    assert first.tasks[0].added_lines == 1
    assert first.tasks[0].removed_lines == 1
    assert first.tasks[0].changed_lines == 2
    assert first.tokens == {
        "complete": True,
        "input": 100,
        "cached_input": 30,
        "output": 20,
        "total": 120,
    }
    assert first.cost_usd == pytest.approx(0.00014)
    assert first.price_source == "configured"
    assert secret not in first.model_dump_json()
    assert second.skipped_task_count == 1
    assert second.tasks[0].model_dump() == first.tasks[0].model_dump()
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (tmp_path / "results").rglob("*")
        if path.is_file() and path.name != "runs.sqlite3"
    )
    assert secret not in persisted


def test_repeated_trials_are_limited_to_locked_full_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the full variant"):
        EvaluationRunner(
            BENCHMARKS,
            tmp_path,
            variant="baseline",
            trial=2,
            model="test-model",
            agent_executor=lambda *_: None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="locked"):
        EvaluationRunner(
            BENCHMARKS,
            tmp_path,
            variant="full",
            trial=2,
            task_ids=("py-bugfix-001",),
            model="test-model",
            agent_executor=lambda *_: None,  # type: ignore[arg-type]
        )


def test_output_is_bounded_even_when_executor_raises_with_a_secret(
    tmp_path: Path,
) -> None:
    secret = "sk-secret-value-that-must-be-redacted"

    def fail(*_):
        raise RuntimeError(("large failure " * 1000) + secret)

    report = EvaluationRunner(
        BENCHMARKS,
        tmp_path,
        variant="full",
        task_ids=("py-bugfix-001",),
        model="test-model",
        secrets=(secret,),
        agent_executor=fail,
    ).run()

    result = report.tasks[0]
    assert report.status == "partial"
    assert report.pass_at_1 == 0.0
    assert report.test_restoration_rate == 0.0
    assert result.status == "error"
    assert result.error is not None
    assert len(result.error.encode("utf-8")) <= 4000
    assert secret not in report.model_dump_json()


def test_missing_usage_is_a_failed_budget_with_explicit_category(
    tmp_path: Path,
) -> None:
    gold_patch = (
        BENCHMARKS
        / "tasks"
        / "python"
        / "bugfix"
        / "trailing-separator"
        / "gold.patch"
    ).read_text(encoding="utf-8")

    def execute(*_args):
        return AgentExecutionArtifact(
            AgentExecution(
                run_id="c" * 32,
                status="completed",
                public_passed=True,
            ),
            gold_patch,
        )

    report = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "results",
        variant="full",
        task_ids=("py-bugfix-001",),
        model="test-model",
        agent_executor=execute,
        score_executor=lambda *_args: ScoreOutcome(
            public_passed=True,
            hidden_passed=True,
            original_tests_unchanged=True,
            build_unchanged=True,
            regression_test_added=True,
            regression_failed_on_buggy=True,
            regression_passed_on_candidate=True,
            policy_passed=True,
        ),
    ).run()

    assert report.tasks[0].budget_passed is False
    assert report.tasks[0].failure_category == "usage_unavailable"
    assert report.tasks[0].solved is False


def test_matrix_jobs_are_unique_complete_and_stably_sharded() -> None:
    jobs = experiment_jobs(BENCHMARKS)

    assert len(jobs) == 44
    assert len({job.identity for job in jobs}) == 44
    assert sum(job.variant == "baseline" for job in jobs) == 12
    assert sum(job.variant == "no-review" for job in jobs) == 12
    assert sum(job.variant == "full" for job in jobs) == 20
    shards = [jobs_for_shard(BENCHMARKS, index) for index in range(5)]
    assert [len(shard) for shard in shards] == [9, 9, 9, 9, 8]
    assert {
        job.identity for shard in shards for job in shard
    } == {job.identity for job in jobs}
    with pytest.raises(ValueError, match="exactly 5"):
        jobs_for_shard(BENCHMARKS, 0, shard_count=4)
    with pytest.raises(ValueError, match="between 0 and 4"):
        jobs_for_shard(BENCHMARKS, 5)


def test_canary_has_one_trial_one_job_for_every_variant() -> None:
    jobs = canary_jobs(BENCHMARKS)

    assert [job.identity for job in jobs] == [
        ("baseline", 1, "py-bugfix-001"),
        ("no-review", 1, "py-bugfix-001"),
        ("full", 1, "py-bugfix-001"),
    ]


def test_canary_and_shard_batches_group_jobs_without_exceeding_two_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite = load_benchmark_suite(BENCHMARKS)
    calls: list[tuple[str, int, tuple[str, ...], Path]] = []

    class FakeRunner:
        def __init__(self, _benchmark_dir, output_dir, **options):
            self.output_dir = Path(output_dir)
            self.variant = options["variant"]
            self.trial = options["trial"]
            self.task_ids = tuple(options["task_ids"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def run(self):
            calls.append(
                (self.variant, self.trial, self.task_ids, self.output_dir)
            )
            results = tuple(
                _synthetic_result(
                    ExperimentJob(
                        variant=self.variant,
                        trial=self.trial,
                        task_id=task_id,
                    ),
                    suite=suite,
                    index=index,
                    solved=True,
                )
                for index, task_id in enumerate(self.task_ids)
            )
            return EvaluationReport(
                experiment_id="repo-agent-day7-v1",
                suite_id=suite.suite_id,
                manifest_sha256=suite.manifest_sha256,
                variant=self.variant,
                trial=self.trial,
                model="test-model",
                status="completed",
                task_count=len(results),
                completed_task_count=len(results),
                skipped_task_count=0,
                solved_task_count=len(results),
                pass_at_1=1.0,
                language_success={},
                test_restoration_rate=1.0,
                tool_error_rate=0.0,
                patch_bytes={"total": 100 * len(results)},
                tokens={"complete": True, "total": 15 * len(results)},
                cost_usd=None,
                price_source="unavailable",
                latency_ms={"p50": 100, "p95": 100},
                failure_categories={},
                request_policy={},
                budget_policy={},
                result_dir=str(self.output_dir),
                tasks=results,
                completed_at="2026-09-11T00:00:00.000Z",
            )

    monkeypatch.setattr(evaluation_module, "EvaluationRunner", FakeRunner)
    shard = run_experiment_shard(
        BENCHMARKS,
        tmp_path / "state",
        shard_index=0,
        workers=2,
        model_config=_model_config(),
        secrets=("test-evaluation-key",),
    )
    assert shard.status == "completed"
    assert shard.job_count == 9
    assert shard.completed_job_count == 9
    assert shard.workers == 2
    assert (tmp_path / "state" / "matrix.json").is_file()
    formal_text = (tmp_path / "state" / "experiment.json").read_text(
        encoding="utf-8"
    )
    assert "test-evaluation-key" not in formal_text
    assert {task_id for _variant, _trial, ids, _root in calls for task_id in ids} == {
        job.task_id for job in jobs_for_shard(BENCHMARKS, 0)
    }
    incompatible = OpenAIConfig(
        api_key="test-evaluation-key",
        base_url="http://127.0.0.1:9999/v1",
        model="different-model",
    )
    with pytest.raises(evaluation_module.EvaluationError, match="different experiment"):
        run_experiment_shard(
            BENCHMARKS,
            tmp_path / "state",
            shard_index=1,
            model_config=incompatible,
        )

    calls.clear()
    canary = run_experiment_canary(
        BENCHMARKS,
        tmp_path / "state",
        model_config=_model_config(),
        workers=2,
    )
    assert canary.status == "completed"
    assert canary.job_count == 3
    assert {(variant, trial) for variant, trial, _ids, _root in calls} == {
        ("baseline", 1),
        ("no-review", 1),
        ("full", 1),
    }
    with pytest.raises(ValueError, match="between 1 and 2"):
        run_experiment_canary(
            BENCHMARKS,
            tmp_path / "bad-workers",
            model_config=_model_config(),
            workers=3,
        )


def test_merge_writes_only_sanitized_44_row_publish_artifacts(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "private-state"
    publish = tmp_path / "publish"
    _materialize_formal_results(state_root)

    report = merge_experiment_results(
        BENCHMARKS,
        state_root,
        report_dir=publish,
    )

    results_path = publish / "results.jsonl"
    report_path = publish / "report.json"
    failure_path = publish / "failure-analysis.json"
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 44
    assert report.status == "completed"
    assert report.job_count == 44
    assert report.completed_job_count == 44
    assert report.trial_one_job_count == 36
    assert report.trial_one_solved_job_count == 0
    assert report.solved_job_count == 8
    assert report.pass_at_1 == 0.0
    assert report.variant_success["full"] == {
        "jobs": 12,
        "solved": 0,
        "rate": 0.0,
    }
    assert report.all_run_success["full"] == {
        "jobs": 20,
        "solved": 8,
        "rate": 0.4,
    }
    assert report.language_success["python"]["jobs"] == 18
    assert report.language_success["java"]["jobs"] == 18
    assert report.repeat_stability["py-bugfix-003"] == {
        "trials": 3,
        "solved": 2,
        "rate": pytest.approx(2 / 3),
        "all_solved": False,
    }
    assert report.tokens["cached_input"] == 88
    assert report.cost_usd is None
    assert report.price_source == "unavailable"
    assert report.results_path == "results.jsonl"
    assert report.report_path == "report.json"
    assert report.failure_analysis_path == "failure-analysis.json"
    assert report_path.is_file()
    assert failure_path.is_file()
    published = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (results_path, report_path, failure_path)
    )
    assert str(state_root) not in published
    assert state_root.as_posix() not in published
    assert str(PROJECT_ROOT) not in published
    assert PROJECT_ROOT.as_posix() not in published
    assert "test-evaluation-key" not in "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in publish.iterdir()
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "results_sha256"
    ] == report.results_sha256


def test_merge_rejects_missing_duplicate_and_extra_trials(tmp_path: Path) -> None:
    state_root = tmp_path / "private-state"
    jobs, paths = _materialize_formal_results(state_root)
    victim = jobs[0]
    victim_path = paths[victim.identity]
    victim_payload = victim_path.read_text(encoding="utf-8")

    victim_path.unlink()
    with pytest.raises(evaluation_module.EvaluationError, match="missing=1"):
        merge_experiment_results(BENCHMARKS, state_root)
    victim_path.write_text(victim_payload, encoding="utf-8")

    duplicate_path = Path(
        str(victim_path).replace("shard-0-of-5", "shard-1-of-5")
    )
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(victim_path, duplicate_path)
    with pytest.raises(evaluation_module.EvaluationError, match="duplicate=1"):
        merge_experiment_results(BENCHMARKS, state_root)
    duplicate_path.unlink()

    suite = load_benchmark_suite(BENCHMARKS)
    extra = ExperimentJob(variant="full", trial=2, task_id="py-bugfix-001")
    extra_result = _synthetic_result(
        extra,
        suite=suite,
        index=99,
        solved=False,
    )
    extra_path = (
        state_root
        / "shards"
        / "shard-0-of-5"
        / suite.suite_id
        / "full"
        / "trial-2"
        / "tasks"
        / "py-bugfix-001.json"
    )
    extra_path.parent.mkdir(parents=True, exist_ok=True)
    extra_path.write_text(
        extra_result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation_module.EvaluationError, match="unexpected=1"):
        merge_experiment_results(BENCHMARKS, state_root)


def test_merge_rejects_tampered_candidate_scoring_and_group_manifest(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "private-state"
    jobs, paths = _materialize_formal_results(state_root)
    result_path = paths[jobs[0].identity]
    group_dir = result_path.parent.parent

    candidate = group_dir / "candidates" / f"{jobs[0].task_id}.patch"
    candidate_bytes = candidate.read_bytes()
    candidate.write_bytes(candidate_bytes + b"\n")
    with pytest.raises(evaluation_module.EvaluationError, match="candidate artifact"):
        merge_experiment_results(BENCHMARKS, state_root)
    candidate.write_bytes(candidate_bytes)

    scoring = group_dir / "scoring" / jobs[0].task_id / "public.json"
    scoring_bytes = scoring.read_bytes()
    scoring_value = json.loads(scoring_bytes)
    scoring_value["duration_ms"] += 1
    scoring.write_text(
        json.dumps(scoring_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation_module.EvaluationError, match="scoring digest"):
        merge_experiment_results(BENCHMARKS, state_root)
    scoring.write_bytes(scoring_bytes)

    manifest = group_dir / "experiment.json"
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["model"] = "tampered-model"
    manifest.write_text(
        json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation_module.EvaluationError, match="group manifest"):
        merge_experiment_results(BENCHMARKS, state_root)


def test_merge_rejects_unlocked_formal_budget_policy(tmp_path: Path) -> None:
    state_root = tmp_path / "private-state"
    _materialize_formal_results(state_root)
    manifest = state_root / "experiment.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["budget_policy"]["max_tool_calls"] = 29
    manifest.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evaluation_module.EvaluationError, match="budget policy"):
        merge_experiment_results(BENCHMARKS, state_root)


def test_scoring_error_retry_preserves_agent_execution(tmp_path: Path) -> None:
    agent_calls = 0
    score_calls = 0
    gold_patch = (
        BENCHMARKS
        / "tasks"
        / "python"
        / "bugfix"
        / "trailing-separator"
        / "gold.patch"
    ).read_text(encoding="utf-8")

    def execute(_task, workspace, _strategy, _deadline):
        nonlocal agent_calls
        agent_calls += 1
        (workspace / "agent-side-effect.txt").write_text("once", encoding="utf-8")
        return AgentExecutionArtifact(
            AgentExecution(
                run_id="d" * 32,
                status="succeeded",
                public_passed=True,
                tool_calls=1,
                duration_ms=10,
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
            gold_patch,
        )

    def score(*_args):
        nonlocal score_calls
        score_calls += 1
        if score_calls == 1:
            raise RuntimeError("transient scoring failure")
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
        "task_ids": ("py-bugfix-001",),
        "model": "test-model",
        "agent_executor": execute,
        "score_executor": score,
    }
    first = EvaluationRunner(BENCHMARKS, tmp_path / "results", **options).run()
    second = EvaluationRunner(BENCHMARKS, tmp_path / "results", **options).run()

    assert first.tasks[0].failure_category == "scoring_error"
    assert second.tasks[0].status == "completed"
    assert second.tasks[0].solved
    assert agent_calls == 1
    assert score_calls == 2
    assert len(list((tmp_path / "results").rglob("*.scoring-error.*.json"))) == 1


def test_interrupted_baseline_is_terminal_without_model_retry(tmp_path: Path) -> None:
    calls = 0

    def execute(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("interrupted baseline must not be called again")

    runner = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "results",
        variant="baseline",
        task_ids=("py-bugfix-001",),
        model="test-model",
        agent_executor=execute,
    )
    task = runner.suite.tasks[0]
    runner._write_state(
        task,
        {
            **runner._state_identity(task),
            "phase": "agent_running",
            "agent_started_epoch": time.time() - 1,
            "run_id": "e" * 32,
        },
    )

    first = runner.run()
    second = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "results",
        variant="baseline",
        task_ids=("py-bugfix-001",),
        model="test-model",
        agent_executor=execute,
    ).run()

    assert calls == 0
    assert first.tasks[0].agent.status == "interrupted"
    assert first.tasks[0].failure_category == "agent_interrupted"
    assert first.tasks[0].agent.run_id == "e" * 32
    assert second.skipped_task_count == 1


def test_evaluation_root_lock_is_exclusive_and_releasable(tmp_path: Path) -> None:
    root = evaluation_module._evaluation_output_root(tmp_path / "state")
    with evaluation_module._formal_shard_lock(root):
        with pytest.raises(evaluation_module.EvaluationError, match="already active"):
            with evaluation_module._formal_shard_lock(root):
                pass
    with evaluation_module._formal_shard_lock(root):
        pass


def test_regression_test_contract_covers_python_and_java() -> None:
    python_task = load_benchmark_suite(
        BENCHMARKS, task_ids=("py-bugfix-001",)
    ).tasks[0]
    java_task = load_benchmark_suite(
        BENCHMARKS, task_ids=("java-bugfix-001",)
    ).tasks[0]
    python_patch = ValidatedPatch(
        "patch",
        (
            PatchFile("tests/test_slug_regression.py", "add", 3, 0),
            PatchFile("tests/helper.py", "add", 1, 0),
        ),
    )
    java_patch = ValidatedPatch(
        "patch",
        (
            PatchFile(
                "src/test/java/dev/SlugRegressionTest.java", "add", 3, 0
            ),
            PatchFile("src/test/java/dev/HelperTest.java", "add", 1, 0),
        ),
    )

    assert evaluation_module._regression_test_paths(
        python_task, python_patch
    ) == ("tests/test_slug_regression.py",)
    assert evaluation_module._regression_test_paths(java_task, java_patch) == (
        "src/test/java/dev/SlugRegressionTest.java",
    )
    failed = CheckRunResult("failed", None, None, False, (), 2)
    passed = CheckRunResult("passed", None, None, False, (), 2)
    assert evaluation_module._attributable_regression_failure(
        python_task,
        ("tests/test_slug_regression.py",),
        failed,
        "tests/test_slug_regression.py::test_suffix FAILED",
    )
    assert not evaluation_module._attributable_regression_failure(
        python_task,
        ("tests/test_slug_regression.py",),
        failed,
        "ERROR collecting tests/test_slug_regression.py",
    )
    assert evaluation_module._attributable_regression_failure(
        java_task,
        ("src/test/java/dev/SlugRegressionTest.java",),
        failed,
        "Failures:\nSlugRegressionTest",
    )
    assert not evaluation_module._attributable_regression_failure(
        java_task,
        ("src/test/java/dev/SlugRegressionTest.java",),
        passed,
        "Failures:\nSlugRegressionTest",
    )


def test_scoring_summaries_are_redacted_and_hashed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "sk-scoring-secret-value"
    runner = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "results",
        variant="full",
        task_ids=("py-bugfix-001",),
        model="test-model",
        secrets=(secret,),
    )
    task = runner.suite.tasks[0]
    passed = CheckRunResult(
        "passed",
        None,
        "a" * 40,
        True,
        (),
        3,
        error=f"diagnostic {secret}",
    )
    failed = CheckRunResult("failed", None, "b" * 40, False, (), 2)
    patch = ValidatedPatch(
        "patch",
        (
            PatchFile("tasklib/text.py", "modify", 1, 1),
            PatchFile("tests/test_slug_regression.py", "add", 2, 0),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_score_check",
        lambda _task, _patch, *, hidden: passed,
    )
    monkeypatch.setattr(
        runner,
        "_regression_fails_on_buggy",
        lambda _task, _patch, _paths: (True, failed),
    )

    score = runner._score_candidate(task, patch)

    assert score.regression_test_added
    assert score.regression_failed_on_buggy
    assert score.regression_passed_on_candidate
    for phase, digest in (
        ("public", score.public_check_sha256),
        ("hidden", score.hidden_check_sha256),
        ("regression", score.regression_check_sha256),
    ):
        path = runner.result_dir / "scoring" / task.id / f"{phase}.json"
        assert digest == evaluation_module.hashlib.sha256(path.read_bytes()).hexdigest()
        assert secret not in path.read_text(encoding="utf-8")
    runner.close()


def test_workspace_preparation_cleans_temporary_directory_on_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "results",
        variant="full",
        task_ids=("py-bugfix-001",),
        model="test-model",
    )
    task = runner.suite.tasks[0]

    def fail_after_creation(_task, destination):
        destination.mkdir(parents=True)
        (destination / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected preparation failure")

    monkeypatch.setattr(
        evaluation_module,
        "_copy_baseline_and_setup",
        fail_after_creation,
    )
    with pytest.raises(RuntimeError, match="injected"):
        runner._prepare_workspace(task, {})

    assert list(runner._workspaces.iterdir()) == []
    runner.close()


@pytest.mark.parametrize(
    ("usage", "expected_status"),
    [
        (
            ProviderUsage(1, 1, 25000, 1000, 6000, 31000),
            "budget_exceeded",
        ),
        (
            ProviderUsage(1, 0, None, None, None, None),
            "usage_unavailable",
        ),
    ],
)
def test_baseline_stops_before_tool_execution_when_usage_budget_is_unprovable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    usage: ProviderUsage,
    expected_status: str,
) -> None:
    runner = EvaluationRunner(
        BENCHMARKS,
        tmp_path / expected_status,
        variant="baseline",
        task_ids=("py-bugfix-001",),
        model_config=_model_config(),
    )
    task = runner.suite.tasks[0]
    workspace = task.root / "baseline"
    tool_executed = False

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.usage = usage

        def set_request_timeout(self, _timeout_seconds):
            pass

        def next_step(self, _task, _results):
            return ToolCall("call-1", "git_status", {})

    class FakeSandbox:
        workspace_revision = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def candidate_artifact(self):
            return None

    class FakeExecutor:
        total_timeout_seconds = 0.0

        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, _decision):
            nonlocal tool_executed
            tool_executed = True
            raise AssertionError("tool must not execute after budget failure")

    monkeypatch.setattr(evaluation_module, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(evaluation_module, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(evaluation_module, "AgentToolExecutor", FakeExecutor)

    artifact = runner._execute_baseline(task, workspace, time.time() + 30)

    assert artifact.execution.status == expected_status
    assert artifact.execution.tool_calls == 0
    assert not tool_executed
    runner.close()


def test_baseline_does_not_execute_tool_after_model_crosses_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = EvaluationRunner(
        BENCHMARKS,
        tmp_path / "deadline",
        variant="baseline",
        task_ids=("py-bugfix-001",),
        model_config=_model_config(),
    )
    task = runner.suite.tasks[0]
    tool_executed = False
    observed_timeouts: list[float] = []

    class FakeProvider:
        usage = ProviderUsage(1, 1, 10, 0, 5, 15)

        def __init__(self, *_args, **_kwargs):
            pass

        def set_request_timeout(self, timeout_seconds):
            observed_timeouts.append(timeout_seconds)

        def next_step(self, _task, _results):
            time.sleep(0.04)
            return ToolCall("call-1", "git_status", {})

    class FakeSandbox:
        workspace_revision = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def candidate_artifact(self):
            return None

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, _decision):
            nonlocal tool_executed
            tool_executed = True
            raise AssertionError("tool must not execute after the deadline")

    monkeypatch.setattr(evaluation_module, "OpenAIProvider", FakeProvider)
    monkeypatch.setattr(evaluation_module, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(evaluation_module, "AgentToolExecutor", FakeExecutor)

    artifact = runner._execute_baseline(
        task,
        task.root / "baseline",
        time.time() + 0.02,
    )

    assert artifact.execution.status == "timed_out"
    assert artifact.execution.timed_out
    assert artifact.execution.tool_calls == 0
    assert observed_timeouts and observed_timeouts[0] <= 0.02
    assert not tool_executed
    runner.close()
