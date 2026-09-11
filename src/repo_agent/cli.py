"""Typer command-line entry point for the repository maintenance agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from enum import Enum
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional, Sequence

import typer

try:  # Typer 0.21 vendors Click; older supported releases import it directly.
    typer_click: Any = importlib.import_module("typer._click")
except ImportError:  # pragma: no cover - exercised by Typer releases before 0.21
    typer_click = importlib.import_module("click")

from .api import serve as serve_api
from .checks import CheckError, CheckRunResult, CheckRunner
from .doctor import run_doctor
from .evaluation import (
    EvaluationBatchReport,
    EvaluationError,
    ExperimentReport,
    EvaluationReport,
    EvaluationRunner,
    merge_experiment_results,
    run_experiment_canary,
    run_experiment_shard,
    validation_manifest,
    write_experiment_matrix_manifest,
)
from .models import RunResult
from .openai_provider import OpenAIConfig, OpenAIProviderError
from .persistence import RunDatabase, RunNotFoundError
from .run_models import ARTIFACT_FILENAMES, RunRecord, TERMINAL_STATUSES
from .sandbox import SandboxError
from .service import (
    InvalidRunTransitionError,
    QueueFullError,
    RepositoryNotAllowedError,
    RunService,
    run_agent,
)
from .web import serve as serve_web


MAX_TASK_FILE_BYTES = 16 * 1024
DEFAULT_WAIT_SECONDS = 20 * 60.0
_API_KEY_ENV = "REPO_AGENT_API_KEY"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class ProviderChoice(str, Enum):
    openai = "openai"
    demo = "demo"


class EvalVariant(str, Enum):
    baseline = "baseline"
    full = "full"
    no_review = "no-review"


def _positive_float_option(value: float) -> float:
    if not 0 < value < float("inf"):
        raise typer.BadParameter("must be a positive finite number")
    return value


def _command_result(exit_code: int) -> int:
    """Preserve exit status for both the console script and Typer CliRunner."""

    if exit_code:
        raise typer.Exit(exit_code)
    return 0


app = typer.Typer(
    name="repo-agent",
    help="Maintain a committed Git snapshot with bounded model tools and Docker.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.command("doctor")
def doctor_command(
    allow_remote_model: bool = typer.Option(
        False,
        "--allow-remote-model",
        help="Allow source and prompts to be sent to a non-loopback HTTPS endpoint.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Verify model connectivity and a two-turn native tool call."""

    try:
        config = OpenAIConfig.from_env(allow_remote_model=allow_remote_model)
        result = run_doctor(config)
    except KeyboardInterrupt:
        return _command_result(_emit_simple_error(
            "doctor",
            "interrupted",
            "repo-agent doctor interrupted",
            output_format.value,
            exit_code=130,
        ))
    except (OSError, ValueError, OpenAIProviderError) as exc:
        return _command_result(_emit_simple_error(
            "doctor",
            "failed",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
        ))

    payload = asdict(result)
    if output_format is OutputFormat.json:
        _print_json(payload)
    else:
        typer.echo("status: ok")
        typer.echo(f"api: {result.api_kind}")
        typer.echo(f"model: {result.model}")
        typer.echo("native_tool_call: verified")
        typer.echo(f"http_requests: {result.http_requests}")
        typer.echo(f"message: {result.message}")
    return 0


@app.command("run")
def run_command(
    repo: Path = typer.Option(Path("."), "--repo", help="Local Git repository."),
    task: Optional[str] = typer.Option(None, "--task", help="Maintenance task."),
    task_file: Optional[Path] = typer.Option(
        None,
        "--task-file",
        help="UTF-8 file containing the maintenance task.",
    ),
    base_ref: Optional[str] = typer.Option(
        None,
        "--base-ref",
        help="Optional Git revision to use instead of HEAD.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Approve the generated plan without an interactive prompt.",
    ),
    allow_remote_model: bool = typer.Option(
        False,
        "--allow-remote-model",
        help="Allow source and prompts to be sent to a non-loopback HTTPS endpoint.",
    ),
    allow_bootstrap: bool = typer.Option(
        False,
        "--allow-bootstrap",
        help="Allow the registered dependency bootstrap phase to use the network.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Durable SQLite and artifact directory.",
    ),
    provider: ProviderChoice = typer.Option(
        ProviderChoice.openai,
        "--provider",
        help="Use the real OpenAI-compatible workflow or the explicit read-only demo.",
    ),
    image: str = typer.Option(
        "repo-agent-python:0.1",
        "--image",
        help="Sandbox image used by the legacy demo provider.",
    ),
    max_steps: int = typer.Option(
        8,
        "--max-steps",
        min=1,
        max=30,
        help="Maximum legacy demo tool steps.",
    ),
    timeout_seconds: float = typer.Option(
        30.0,
        "--timeout-seconds",
        callback=_positive_float_option,
        help="Legacy demo tool timeout in seconds.",
    ),
    max_output_bytes: int = typer.Option(
        65536,
        "--max-output-bytes",
        min=1,
        max=16 * 1024 * 1024,
        help="Maximum legacy demo output bytes.",
    ),
    wait_seconds: float = typer.Option(
        DEFAULT_WAIT_SECONDS,
        "--wait-seconds",
        callback=_positive_float_option,
        help="Maximum time to wait for the workflow.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Create a run, approve its plan, and wait for a durable result."""

    normalized_task = _resolve_task(task, task_file)
    if provider is ProviderChoice.demo:
        return _command_result(_run_demo(
            repo=repo,
            task=normalized_task,
            image=image,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            output_format=output_format.value,
        ))

    selected_data_dir = _resolve_data_dir(data_dir)
    service: RunService | None = None
    run_id: str | None = None
    try:
        repository = repo.expanduser().resolve(strict=True)
        # Fail before cloning or Docker work when model configuration is absent
        # or a remote endpoint was not explicitly authorized.
        OpenAIConfig.from_env(allow_remote_model=allow_remote_model)
        service = _new_run_service(
            selected_data_dir,
            allowed_roots=(repository,),
            allow_bootstrap=allow_bootstrap,
        )
        run = service.create_run(
            repo_path=repository,
            task=normalized_task,
            base_ref=base_ref,
            auto_approve=yes,
            allow_remote_model=allow_remote_model,
        )
        run_id = run.run_id
        result = service.wait(run.run_id, timeout=wait_seconds)
        if result.status == "awaiting_approval":
            _emit_plan(result, output_format.value)
            approved = typer.confirm(
                "Approve this change plan?",
                default=False,
                err=output_format is OutputFormat.json,
            )
            result = service.decide(
                result.run_id,
                approve=approved,
                reason="Approved from CLI" if approved else "Rejected from CLI",
            )
            if approved:
                result = service.wait(
                    result.run_id,
                    timeout=wait_seconds,
                    stop_at_approval=True,
                )
        result = _close_and_refresh_record(
            service,
            result,
            data_dir=selected_data_dir,
            command="run",
        )
        service = None
        _emit_durable_record(result, selected_data_dir, output_format.value)
        return _command_result(_durable_exit_code(result))
    except KeyboardInterrupt:
        if service is not None and run_id is not None:
            try:
                service.cancel(run_id)
            except (OSError, RuntimeError, ValueError):
                pass
        return _command_result(_emit_simple_error(
            "run",
            "interrupted",
            "repo-agent run interrupted",
            output_format.value,
            exit_code=130,
            extra={"run_id": run_id},
        ))
    except typer.Exit:
        raise
    except (
        OSError,
        ValueError,
        RuntimeError,
        SandboxError,
        QueueFullError,
        RepositoryNotAllowedError,
    ) as exc:
        return _command_result(_emit_simple_error(
            "run",
            "setup_error",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
            extra={"run_id": run_id},
        ))
    finally:
        if service is not None:
            _close_without_overriding_result(service, command="run")


@app.command("resume")
def resume_command(
    run_id: str = typer.Argument(..., help="32-character run identifier."),
    approve: bool = typer.Option(
        False,
        "--approve",
        help="Approve a run paused at its plan checkpoint.",
    ),
    reject: bool = typer.Option(
        False,
        "--reject",
        help="Reject a run paused at its plan checkpoint.",
    ),
    reason: Optional[str] = typer.Option(None, "--reason", help="Decision reason."),
    allow_bootstrap: bool = typer.Option(
        False,
        "--allow-bootstrap",
        help="Allow the registered dependency bootstrap phase to use the network.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Durable SQLite and artifact directory.",
    ),
    wait_seconds: float = typer.Option(
        DEFAULT_WAIT_SECONDS,
        "--wait-seconds",
        callback=_positive_float_option,
        help="Maximum time to wait after resuming.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Approve/reject a plan, or resume an interrupted run."""

    if approve and reject:
        raise typer.BadParameter(
            "--approve and --reject are mutually exclusive",
            param_hint="--approve/--reject",
        )
    selected_data_dir = _resolve_data_dir(data_dir)
    service: RunService | None = None
    try:
        stored = _load_record(selected_data_dir, run_id)
        repository = Path(stored.repo_path).expanduser().resolve(strict=True)
        service = _new_run_service(
            selected_data_dir,
            allowed_roots=(repository,),
            allow_bootstrap=allow_bootstrap,
        )
        current = service.get_run(run_id)
        if current.status == "awaiting_approval":
            if not approve and not reject:
                raise InvalidRunTransitionError(
                    "an approval decision is required: use --approve or --reject"
                )
            result = service.decide(
                current.run_id,
                approve=approve,
                reason=reason,
            )
        else:
            if approve or reject:
                raise InvalidRunTransitionError(
                    f"run {current.run_id} is {current.status}, not awaiting approval"
                )
            result = service.resume(current.run_id)

        if result.status not in TERMINAL_STATUSES:
            result = service.wait(
                result.run_id,
                timeout=wait_seconds,
                stop_at_approval=True,
            )
        result = _close_and_refresh_record(
            service,
            result,
            data_dir=selected_data_dir,
            command="resume",
        )
        service = None
        _emit_durable_record(result, selected_data_dir, output_format.value)
        return _command_result(_durable_exit_code(result))
    except KeyboardInterrupt:
        return _command_result(_emit_simple_error(
            "resume",
            "interrupted",
            "repo-agent resume interrupted",
            output_format.value,
            exit_code=130,
            extra={"run_id": run_id},
        ))
    except typer.Exit:
        raise
    except (OSError, ValueError, RuntimeError, RunNotFoundError) as exc:
        return _command_result(_emit_simple_error(
            "resume",
            "failed",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
            extra={"run_id": run_id},
        ))
    finally:
        if service is not None:
            _close_without_overriding_result(service, command="resume")


@app.command("show")
def show_command(
    run_id: str = typer.Argument(..., help="32-character run identifier."),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Durable SQLite and artifact directory.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Show durable status, plan, checks, metrics, and artifact paths."""

    selected_data_dir = _resolve_data_dir(data_dir)
    try:
        record = _load_record(selected_data_dir, run_id)
    except (OSError, ValueError, RunNotFoundError) as exc:
        return _command_result(_emit_simple_error(
            "show",
            "not_found" if isinstance(exc, RunNotFoundError) else "failed",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
            extra={"run_id": run_id},
        ))
    _emit_durable_record(record, selected_data_dir, output_format.value)
    return 0


@app.command("cancel")
def cancel_command(
    run_id: str = typer.Argument(..., help="32-character run identifier."),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Durable SQLite and artifact directory.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Request cancellation and terminate the run's active sandbox."""

    selected_data_dir = _resolve_data_dir(data_dir)
    service: RunService | None = None
    try:
        stored = _load_record(selected_data_dir, run_id)
        repository = Path(stored.repo_path).expanduser().resolve(strict=True)
        service = _new_run_service(
            selected_data_dir,
            allowed_roots=(repository,),
            allow_bootstrap=False,
            start_worker=False,
        )
        result = service.cancel(run_id)
        _emit_durable_record(result, selected_data_dir, output_format.value)
        return 0
    except (OSError, ValueError, RuntimeError, RunNotFoundError) as exc:
        return _command_result(_emit_simple_error(
            "cancel",
            "failed",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
            extra={"run_id": run_id},
        ))
    finally:
        if service is not None:
            service.close()


@app.command("eval")
def eval_command(
    variant: Optional[EvalVariant] = typer.Option(
        None,
        "--variant",
        help="Experiment variant: baseline, full, or no-review.",
    ),
    benchmark_dir: Path = typer.Option(
        Path("benchmarks"),
        "--benchmark-dir",
        help="Frozen benchmark suite directory.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run model-backed tasks and independent hidden-test scoring.",
    ),
    matrix_only: bool = typer.Option(
        False,
        "--matrix-only",
        help="Print the locked 44-job matrix without executing it.",
    ),
    matrix_run: bool = typer.Option(
        False,
        "--matrix",
        help="Execute one stable shard of the locked formal matrix.",
    ),
    canary: bool = typer.Option(
        False,
        "--canary",
        help="Execute one task with baseline, no-review, and full.",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Strictly merge five completed shards into results.jsonl.",
    ),
    task_id: Optional[list[str]] = typer.Option(
        None,
        "--task-id",
        help="Frozen task ID; repeat to run a subset.",
    ),
    trial: int = typer.Option(
        1,
        "--trial",
        min=1,
        max=3,
        help="Trial number; 2 and 3 are reserved for four locked full tasks.",
    ),
    shard_index: Optional[int] = typer.Option(
        None,
        "--shard-index",
        min=0,
        help="Zero-based formal shard index; required with --matrix.",
    ),
    shard_count: int = typer.Option(
        5,
        "--shard-count",
        min=1,
        help="Formal shard count; the locked experiment requires five.",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        min=1,
        max=2,
        help="Concurrent variant/trial groups per shard (maximum two).",
    ),
    output_dir: Path = typer.Option(
        Path("evaluation-results"),
        "--output-dir",
        help="Durable evaluation state and result directory.",
    ),
    report_dir: Optional[Path] = typer.Option(
        None,
        "--report-dir",
        help="Sanitized publish directory used only with --merge.",
    ),
    allow_remote_model: bool = typer.Option(
        False,
        "--allow-remote-model",
        help="Allow benchmark source and prompts to reach a remote HTTPS model.",
    ),
    allow_bootstrap: bool = typer.Option(
        False,
        "--allow-bootstrap",
        help="Authorize registered dependency bootstrap before offline checks.",
    ),
    task_timeout_seconds: float = typer.Option(
        1200.0,
        "--task-timeout-seconds",
        callback=_positive_float_option,
        help="Hard Agent deadline per task (maximum 1200 seconds).",
    ),
    max_output_bytes: int = typer.Option(
        65536,
        "--max-output-bytes",
        min=1024,
        max=256 * 1024,
        help="Maximum captured output per sandbox phase.",
    ),
    input_cost_per_million: Optional[float] = typer.Option(
        None,
        "--input-cost-per-million",
        min=0.0,
        help="Optional input-token price used for cost reporting.",
    ),
    output_cost_per_million: Optional[float] = typer.Option(
        None,
        "--output-cost-per-million",
        min=0.0,
        help="Optional output-token price used for cost reporting.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Validate or execute the frozen benchmark with durable independent scoring."""

    variant_value = None if variant is None else variant.value
    mode = (
        "matrix-only"
        if matrix_only
        else "matrix"
        if matrix_run
        else "canary"
        if canary
        else "merge"
        if merge
        else "single"
    )
    try:
        special_modes = sum((matrix_only, matrix_run, canary, merge))
        if special_modes > 1:
            raise ValueError(
                "choose only one of --matrix-only, --matrix, --canary, and --merge"
            )
        if report_dir is not None and not merge:
            raise ValueError("--report-dir is only valid with --merge")
        selected_task_ids = tuple(task_id or ())
        configured_secrets = tuple(
            value for value in (os.environ.get(_API_KEY_ENV),) if value
        )
        if matrix_only:
            if execute or variant is not None or selected_task_ids:
                raise ValueError(
                    "--matrix-only does not accept --execute, --variant, or --task-id"
                )
            payload = write_experiment_matrix_manifest(
                benchmark_dir,
                output_dir,
                secrets=configured_secrets,
            )
            _emit_experiment_matrix(payload, output_format.value)
            return 0
        if merge:
            if execute or variant is not None or selected_task_ids:
                raise ValueError(
                    "--merge does not accept --execute, --variant, or --task-id"
                )
            merged = merge_experiment_results(
                benchmark_dir,
                output_dir,
                report_dir=report_dir,
                secrets=configured_secrets,
            )
            _emit_experiment_report(merged, output_format.value)
            return _command_result(0 if merged.status == "completed" else 2)
        if matrix_run:
            if not execute:
                raise ValueError("--matrix requires --execute")
            if variant is not None or selected_task_ids:
                raise ValueError("--matrix does not accept --variant or --task-id")
            if shard_index is None:
                raise ValueError("--matrix requires --shard-index")
            config = OpenAIConfig.from_env(allow_remote_model=allow_remote_model)
            batch = run_experiment_shard(
                benchmark_dir,
                output_dir,
                shard_index=shard_index,
                shard_count=shard_count,
                workers=workers,
                model_config=config,
                allow_remote_model=allow_remote_model,
                allow_bootstrap=allow_bootstrap,
                task_timeout_seconds=task_timeout_seconds,
                max_output_bytes=max_output_bytes,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
                secrets=(config.api_key,),
            )
            _emit_evaluation_batch(batch, output_format.value)
            return _command_result(0 if batch.status == "completed" else 2)
        if canary:
            if not execute:
                raise ValueError("--canary requires --execute")
            if variant is not None or len(selected_task_ids) > 1 or trial != 1:
                raise ValueError(
                    "--canary accepts at most one --task-id and requires trial 1"
                )
            config = OpenAIConfig.from_env(allow_remote_model=allow_remote_model)
            canary_task_id = (
                selected_task_ids[0] if selected_task_ids else "py-bugfix-001"
            )
            batch = run_experiment_canary(
                benchmark_dir,
                output_dir,
                model_config=config,
                task_id=canary_task_id,
                workers=workers,
                allow_remote_model=allow_remote_model,
                allow_bootstrap=allow_bootstrap,
                task_timeout_seconds=task_timeout_seconds,
                max_output_bytes=max_output_bytes,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
                secrets=(config.api_key,),
            )
            _emit_evaluation_batch(batch, output_format.value)
            return _command_result(0 if batch.status == "completed" else 2)
        if variant is None:
            raise ValueError(
                "--variant is required unless using an experiment matrix mode"
            )
        if not execute:
            payload = validation_manifest(
                benchmark_dir,
                variant.value,
                task_ids=selected_task_ids,
                secrets=configured_secrets,
            )
            if output_format is OutputFormat.json:
                _print_json(payload)
            else:
                typer.echo(f"status: {payload['status']}")
                typer.echo(f"variant: {payload['variant']}")
                typer.echo(f"suite: {payload['suite_id']}")
                typer.echo(f"tasks: {payload['task_count']}")
                typer.echo(f"benchmark_dir: {payload['benchmark_dir']}")
                typer.echo("execution: not started (pass --execute to run the model)")
            return 0

        config = OpenAIConfig.from_env(allow_remote_model=allow_remote_model)
        with EvaluationRunner(
            benchmark_dir,
            output_dir,
            variant=variant.value,
            task_ids=selected_task_ids,
            trial=trial,
            model_config=config,
            allow_remote_model=allow_remote_model,
            allow_bootstrap=allow_bootstrap,
            task_timeout_seconds=task_timeout_seconds,
            max_output_bytes=max_output_bytes,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            secrets=(config.api_key,),
        ) as runner:
            report = runner.run()
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        return _command_result(_emit_simple_error(
            "eval",
            "interrupted",
            "repo-agent eval interrupted; durable state was retained",
            output_format.value,
            exit_code=130,
            extra={"mode": mode, "variant": variant_value, "trial": trial},
        ))
    except (
        OSError,
        ValueError,
        RuntimeError,
        EvaluationError,
        OpenAIProviderError,
        subprocess.SubprocessError,
    ) as exc:
        return _command_result(_emit_simple_error(
            "eval",
            "failed",
            _exception_detail(exc),
            output_format.value,
            exit_code=2,
            extra={"mode": mode, "variant": variant_value, "trial": trial},
        ))
    _emit_evaluation_report(report, output_format.value)
    return _command_result(0 if report.status == "completed" else 2)


@app.command("serve")
def serve_command(
    repo: Path = typer.Option(Path("."), "--repo", help="Default repository/root."),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address for --api mode; loopback is recommended.",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Listening port (default: UI 8765, API 8080).",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the visual workspace in the default browser.",
    ),
    api_mode: bool = typer.Option(
        False,
        "--api",
        help="Serve the authenticated Day 5 FastAPI surface instead of the UI server.",
    ),
    bearer_token: Optional[str] = typer.Option(
        None,
        "--token",
        envvar="REPO_AGENT_BEARER_TOKEN",
        help="Bearer token for --api mode (or REPO_AGENT_BEARER_TOKEN).",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Durable SQLite and artifact directory.",
    ),
    allowed_root: Optional[list[Path]] = typer.Option(
        None,
        "--allowed-root",
        help="Allowed repository root for --api; repeat for multiple roots.",
    ),
    allow_bootstrap: bool = typer.Option(
        False,
        "--allow-bootstrap",
        help="Authorize registered dependency bootstrap at service startup.",
    ),
) -> int:
    """Serve the visual workspace, or the authenticated durable REST API."""

    selected_data_dir = _resolve_data_dir(data_dir)
    try:
        repository = repo.expanduser().resolve(strict=True)
        selected_port = port if port is not None else (8080 if api_mode else 8765)
        if not api_mode:
            if host.strip() != "127.0.0.1":
                raise ValueError("the visual workspace only binds to 127.0.0.1")
            return _command_result(serve_web(
                repository,
                port=selected_port,
                open_browser=open_browser,
                data_dir=selected_data_dir,
                allow_bootstrap=allow_bootstrap,
            ))

        if open_browser:
            raise ValueError("--open is available only for the visual workspace")
        if not isinstance(bearer_token, str) or not bearer_token.strip():
            raise ValueError("--api requires --token or REPO_AGENT_BEARER_TOKEN")
        roots = tuple(allowed_root or (repository,))
        with _new_run_service(
            selected_data_dir,
            allowed_roots=roots,
            allow_bootstrap=allow_bootstrap,
        ) as service:
            serve_api(
                service,
                bearer_token=bearer_token,
                allowed_roots=roots,
                host=host,
                port=selected_port,
            )
        return 0
    except KeyboardInterrupt:
        typer.echo("repo-agent server interrupted", err=True)
        return _command_result(130)
    except (OSError, ValueError, RuntimeError, SandboxError) as exc:
        typer.echo(f"repo-agent server failed: {_exception_detail(exc)}", err=True)
        return _command_result(2)


@app.command("check")
def check_command(
    repo: Path = typer.Option(Path("."), "--repo", help="Local Git repository."),
    allow_bootstrap: bool = typer.Option(
        False,
        "--allow-bootstrap",
        help="Explicitly allow dependency bootstrap with network access.",
    ),
    phase_timeout_seconds: float = typer.Option(
        300.0,
        "--phase-timeout-seconds",
        callback=_positive_float_option,
    ),
    total_timeout_seconds: float = typer.Option(
        1200.0,
        "--total-timeout-seconds",
        callback=_positive_float_option,
    ),
    max_output_bytes: int = typer.Option(
        65536,
        "--max-output-bytes",
        min=1,
        max=16 * 1024 * 1024,
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--format",
        help="Output format.",
    ),
) -> int:
    """Run the detected repository test profile in hardened Docker."""

    return _command_result(_run_check(
        repo=repo,
        allow_bootstrap=allow_bootstrap,
        phase_timeout_seconds=phase_timeout_seconds,
        total_timeout_seconds=total_timeout_seconds,
        max_output_bytes=max_output_bytes,
        output_format=output_format.value,
    ))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Typer app while retaining the historical integer-returning API."""

    try:
        result = app(
            args=None if argv is None else list(argv),
            prog_name="repo-agent",
            standalone_mode=False,
        )
    except typer_click.ClickException as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except typer_click.Abort:
        typer.echo("Aborted!", err=True)
        return 130
    return int(result or 0)


def _run_demo(
    *,
    repo: Path,
    task: str,
    image: str,
    max_steps: int,
    timeout_seconds: float,
    max_output_bytes: int,
    output_format: str,
) -> int:
    try:
        result = run_agent(
            task=task,
            repo_path=repo,
            image=image,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            allow_mutations=False,
        )
    except KeyboardInterrupt:
        return _emit_interrupted(task, output_format)
    except (OSError, ValueError, SandboxError) as exc:
        return _emit_setup_error(task, exc, output_format)

    _emit_result(result, output_format)
    return 0 if result.status == "completed" else 1


def _run_check(
    *,
    repo: Path,
    allow_bootstrap: bool,
    phase_timeout_seconds: float,
    total_timeout_seconds: float,
    max_output_bytes: int,
    output_format: str,
) -> int:
    runner: CheckRunner | None = None
    try:
        runner = CheckRunner(
            repo,
            allow_bootstrap=allow_bootstrap,
            phase_timeout_seconds=phase_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        result = runner.run()
    except KeyboardInterrupt:
        cleanup_error: str | None = None
        if runner is not None:
            try:
                runner.close()
            except (OSError, CheckError) as exc:
                cleanup_error = _exception_detail(exc)
        if output_format == "json":
            _print_json(
                {
                    "status": "interrupted",
                    "profile": None,
                    "base_commit": None,
                    "candidate_applied": False,
                    "phases": [],
                    "duration_ms": 0,
                    "error": cleanup_error,
                    "cleanup_ok": cleanup_error is None,
                }
            )
        else:
            typer.echo("repo-agent check interrupted", err=True)
        return 130
    except (OSError, ValueError, CheckError) as exc:
        result = CheckRunResult(
            status="setup_error",
            profile=None,
            base_commit=None,
            candidate_applied=False,
            phases=(),
            duration_ms=0,
            error=_exception_detail(exc),
        )

    _emit_check_result(result, output_format)
    if result.status == "passed":
        return 0
    if result.status in {"setup_error", "cleanup_error"}:
        return 2
    return 1


def _new_run_service(
    data_dir: Path,
    *,
    allowed_roots: tuple[Path, ...],
    allow_bootstrap: bool,
    start_worker: bool = True,
) -> RunService:
    api_key = os.environ.get(_API_KEY_ENV)
    secrets = (api_key,) if api_key else ()
    return RunService(
        data_dir,
        allowed_repo_roots=allowed_roots,
        start_worker=start_worker,
        allow_bootstrap=allow_bootstrap,
        secrets=secrets,
    )


def _resolve_task(task: str | None, task_file: Path | None) -> str:
    if task is not None and task_file is not None:
        raise typer.BadParameter(
            "use exactly one of --task or --task-file",
            param_hint="--task/--task-file",
        )
    if task_file is not None:
        try:
            path = task_file.expanduser().resolve(strict=True)
            raw = path.read_bytes()
        except OSError as exc:
            raise typer.BadParameter(
                f"could not read task file: {exc}",
                param_hint="--task-file",
            ) from exc
        if not path.is_file():
            raise typer.BadParameter(
                "task file must be a regular file",
                param_hint="--task-file",
            )
        if len(raw) > MAX_TASK_FILE_BYTES:
            raise typer.BadParameter(
                f"task file must be at most {MAX_TASK_FILE_BYTES} bytes",
                param_hint="--task-file",
            )
        try:
            task = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise typer.BadParameter(
                "task file must contain UTF-8 text",
                param_hint="--task-file",
            ) from exc
    if not isinstance(task, str) or not task.strip():
        raise typer.BadParameter(
            "provide non-whitespace text with --task or --task-file",
            param_hint="--task/--task-file",
        )
    normalized = task.strip()
    if "\x00" in normalized or len(normalized) > 4000:
        raise typer.BadParameter(
            "task must contain at most 4000 characters and no NUL",
            param_hint="--task/--task-file",
        )
    return normalized


def _resolve_data_dir(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.environ.get("REPO_AGENT_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if base:
            return (Path(base) / "repo-agent").resolve()
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    if state_home:
        return (Path(state_home) / "repo-agent").expanduser().resolve()
    return (Path.home() / ".local" / "state" / "repo-agent").resolve()


def _load_record(data_dir: Path, run_id: str) -> RunRecord:
    database_path = data_dir / "runs.sqlite3"
    if not database_path.is_file():
        raise RunNotFoundError(f"run not found: {run_id}")
    return RunDatabase(database_path).get(run_id)


def _close_and_refresh_record(
    service: RunService,
    record: RunRecord,
    *,
    data_dir: Path,
    command: str,
) -> RunRecord:
    """Close the owner, then report the state persisted by shutdown."""

    _close_without_overriding_result(service, command=command)
    if record.status not in {"queued", "planning", "running"}:
        return record
    getter = getattr(service, "get_run", None)
    if callable(getter):
        try:
            refreshed = getter(record.run_id)
        except (OSError, ValueError, RuntimeError, RunNotFoundError):
            pass
        else:
            if isinstance(refreshed, RunRecord):
                return refreshed
    try:
        return _load_record(data_dir, record.run_id)
    except (OSError, ValueError, RuntimeError, RunNotFoundError):
        return record


def _close_without_overriding_result(service: RunService, *, command: str) -> None:
    try:
        service.close()
    except Exception as exc:
        typer.echo(
            f"repo-agent {command} cleanup warning: {_exception_detail(exc)}",
            err=True,
        )


def _artifact_paths(data_dir: Path, record: RunRecord) -> dict[str, str]:
    run_dir = data_dir / "runs" / record.run_id
    return {
        kind: str((run_dir / filename).resolve())
        for kind, filename in ARTIFACT_FILENAMES.items()
        if (run_dir / filename).exists()
    }


def _emit_durable_record(
    record: RunRecord,
    data_dir: Path,
    output_format: str,
) -> None:
    artifacts = _artifact_paths(data_dir, record)
    if output_format == "json":
        payload = record.model_dump(mode="json")
        payload["artifacts"] = artifacts
        _print_json(payload)
        return

    typer.echo(f"run_id: {record.run_id}")
    typer.echo(f"status: {record.status}")
    typer.echo(f"node: {record.current_node or '-'}")
    typer.echo(f"repository: {record.repo_path}")
    typer.echo(f"task: {record.task}")
    if record.base_commit:
        typer.echo(f"base_commit: {record.base_commit}")
    if record.plan is not None:
        typer.echo(f"plan_goal: {record.plan.goal}")
        for index, step in enumerate(record.plan.steps, start=1):
            typer.echo(f"plan_step_{index}: {step}")
    for check in record.checks:
        typer.echo(
            f"check: attempt={check.attempt} status={check.status} "
            f"ok={str(check.ok).lower()} duration_ms={check.duration_ms}"
        )
    typer.echo(f"tool_calls: {record.metrics.tool_calls}")
    typer.echo(f"tokens: {record.metrics.tokens}")
    if record.summary:
        typer.echo(f"summary: {record.summary}")
    if record.error:
        typer.echo(f"error: {record.error}")
    for kind, path in artifacts.items():
        typer.echo(f"artifact_{kind}: {path}")


def _emit_plan(record: RunRecord, output_format: str) -> None:
    if record.plan is None:
        return
    if output_format == "json":
        typer.echo(
            json.dumps(
                {"approval_plan": record.plan.model_dump(mode="json")},
                ensure_ascii=False,
                allow_nan=False,
            ),
            err=True,
        )
        return
    typer.echo("change plan:")
    typer.echo(f"  goal: {record.plan.goal}")
    if record.plan.files:
        typer.echo(f"  files: {', '.join(record.plan.files)}")
    for index, step in enumerate(record.plan.steps, start=1):
        typer.echo(f"  {index}. {step}")
    if record.plan.risks:
        typer.echo(f"  risks: {'; '.join(record.plan.risks)}")


def _durable_exit_code(record: RunRecord) -> int:
    if record.status in {"succeeded", "awaiting_approval"}:
        return 0
    if record.status in TERMINAL_STATUSES:
        return 1
    return 2


def _validate_evaluation_suite(
    benchmark_dir: Path,
    variant: str,
) -> dict[str, object]:
    if variant not in {item.value for item in EvalVariant}:
        raise ValueError(f"unsupported evaluation variant: {variant}")
    return validation_manifest(benchmark_dir, variant)  # type: ignore[arg-type]


def _emit_evaluation_report(report: EvaluationReport, output_format: str) -> None:
    if output_format == "json":
        _print_json(report.model_dump(mode="json"))
        return
    typer.echo(f"status: {report.status}")
    typer.echo(f"variant: {report.variant}")
    typer.echo(f"trial: {report.trial}")
    typer.echo(f"suite: {report.suite_id}")
    typer.echo(
        f"tasks: {report.completed_task_count}/{report.task_count} completed, "
        f"{report.skipped_task_count} reused"
    )
    pass_rate = "-" if report.pass_at_1 is None else f"{report.pass_at_1:.3f}"
    typer.echo(f"pass_at_1: {pass_rate}")
    typer.echo(f"solved: {report.solved_task_count}")
    if report.tool_error_rate is not None:
        typer.echo(f"tool_error_rate: {report.tool_error_rate:.3f}")
    typer.echo(f"result_dir: {report.result_dir}")
    for result in report.tasks:
        outcome = "pass" if result.solved else result.failure_category or "fail"
        typer.echo(
            f"task: {result.task_id} ({outcome}, {result.agent.duration_ms}ms, "
            f"{result.patch_bytes} patch bytes)"
        )


def _emit_experiment_matrix(payload: dict[str, object], output_format: str) -> None:
    if output_format == "json":
        _print_json(payload)
        return
    typer.echo(f"status: {payload['status']}")
    typer.echo(f"experiment: {payload['experiment_id']}")
    typer.echo(f"suite: {payload['suite_id']}")
    typer.echo(f"jobs: {payload['job_count']}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        typer.echo(
            f"job: {job['variant']} trial-{job['trial']} {job['task_id']}"
        )


def _emit_evaluation_batch(
    report: EvaluationBatchReport, output_format: str
) -> None:
    if output_format == "json":
        _print_json(report.model_dump(mode="json"))
        return
    typer.echo(f"status: {report.status}")
    typer.echo(f"mode: {report.mode}")
    if report.shard_index is not None and report.shard_count is not None:
        typer.echo(f"shard: {report.shard_index}/{report.shard_count}")
    typer.echo(f"jobs: {report.completed_job_count}/{report.job_count} completed")
    typer.echo(f"workers: {report.workers}")
    typer.echo(f"result_dir: {report.result_dir}")


def _emit_experiment_report(report: ExperimentReport, output_format: str) -> None:
    if output_format == "json":
        _print_json(report.model_dump(mode="json"))
        return
    typer.echo(f"status: {report.status}")
    typer.echo(f"experiment: {report.experiment_id}")
    typer.echo(f"jobs: {report.completed_job_count}/{report.job_count} completed")
    typer.echo(f"solved: {report.solved_job_count}")
    pass_rate = "-" if report.pass_at_1 is None else f"{report.pass_at_1:.3f}"
    typer.echo(f"pass_at_1: {pass_rate}")
    typer.echo(f"price_source: {report.price_source}")
    typer.echo(f"cost_usd: {report.cost_usd if report.cost_usd is not None else '-'}")
    typer.echo(f"results: {report.results_path}")
    typer.echo(f"report: {report.report_path}")
    typer.echo(f"failure_analysis: {report.failure_analysis_path}")


def _emit_check_result(result: CheckRunResult, output_format: str) -> None:
    if output_format == "json":
        _print_json(asdict(result))
        return

    typer.echo(f"status: {result.status}")
    if result.profile is not None:
        typer.echo(f"profile: {result.profile.id} ({result.profile.language})")
        typer.echo(f"image: {result.profile.image}")
        typer.echo(f"manifest: {result.profile.manifest}")
    if result.base_commit is not None:
        typer.echo(f"base_commit: {result.base_commit}")
    typer.echo(f"candidate_applied: {str(result.candidate_applied).lower()}")
    typer.echo(f"cleanup_ok: {str(result.cleanup_ok).lower()}")
    typer.echo(f"duration_ms: {result.duration_ms}")
    for phase in result.phases:
        exit_detail = "" if phase.exit_code is None else f", exit={phase.exit_code}"
        truncation = ", truncated" if phase.truncated else ""
        typer.echo(
            f"phase: {phase.name} ({phase.kind}, {phase.status}, "
            f"network={phase.network}{exit_detail}, {phase.duration_ms}ms{truncation})"
        )
        typer.echo(f"command: {' '.join(phase.argv)}")
        if phase.error:
            typer.echo(f"phase_error: {phase.error}")
    if result.error:
        typer.echo(f"error: {result.error}")


def _emit_result(result: RunResult, output_format: str) -> None:
    if output_format == "json":
        _print_json(asdict(result))
        return

    typer.echo(f"status: {result.status}")
    typer.echo(f"steps: {result.steps}")
    if result.answer:
        typer.echo(f"answer: {result.answer}")
    for tool_result in result.tool_results:
        outcome = "ok" if tool_result.ok else "failed"
        exit_detail = (
            "" if tool_result.exit_code is None else f", exit={tool_result.exit_code}"
        )
        truncation = ", truncated" if tool_result.truncated else ""
        typer.echo(f"tool: {tool_result.name} ({outcome}{exit_detail}{truncation})")


def _emit_setup_error(task: str, exc: BaseException, output_format: str) -> int:
    if output_format == "json":
        _print_json(
            {
                "task": task,
                "status": "setup_error",
                "answer": None,
                "steps": 0,
                "tool_results": [],
                "error": _exception_detail(exc),
            }
        )
    else:
        typer.echo(f"repo-agent setup failed: {_exception_detail(exc)}", err=True)
    return 2


def _emit_interrupted(task: str, output_format: str) -> int:
    if output_format == "json":
        _print_json(
            {
                "task": task,
                "status": "interrupted",
                "answer": None,
                "steps": 0,
                "tool_results": [],
            }
        )
    else:
        typer.echo("repo-agent interrupted", err=True)
    return 130


def _emit_simple_error(
    command: str,
    status: str,
    error: str,
    output_format: str,
    *,
    exit_code: int,
    extra: dict[str, object] | None = None,
) -> int:
    if output_format == "json":
        payload: dict[str, object] = {
            "command": command,
            "status": status,
            "error": error,
        }
        if extra:
            payload.update(extra)
        _print_json(payload)
    else:
        typer.echo(f"repo-agent {command} failed: {error}", err=True)
    return exit_code


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
