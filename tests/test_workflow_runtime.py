from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

import pytest

from repo_agent.checks import CheckPhaseResult, CheckProfile, CheckRunResult
from repo_agent.models import CandidateArtifact, FinalAnswer, ToolCall, ToolResult
from repo_agent.providers import Provider
from repo_agent.service import RunService
from repo_agent.tools import ToolDefinition
from repo_agent.workflow_runtime import ServiceWorkflowRunner
import repo_agent.workflow_runtime as runtime_module


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def clean_python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Workflow Test")
    _git(repo, "config", "user.email", "workflow@example.invalid")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_app.py").write_text(
        "def test_value():\n    from app import VALUE\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


class FakeSandbox:
    def __init__(self, repo_path, **kwargs) -> None:
        self.repo_path = repo_path
        self.kwargs = kwargs
        self.patch = ""
        self.workspace_revision = 0
        self.base_commit = _git(Path(repo_path), "rev-parse", "HEAD")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def close(self) -> None:
        return None

    def execute(self, call: ToolCall) -> ToolResult:
        if call.name == "repo_map":
            return ToolResult(call.id, call.name, True, "FILES\napp.py\ntest_app.py")
        if call.name == "apply_patch":
            self.patch = str(call.arguments["patch"])
            self.workspace_revision += 1
            return ToolResult(call.id, call.name, True, "patch applied", exit_code=0)
        if call.name == "git_diff":
            return ToolResult(call.id, call.name, True, self.patch, exit_code=0)
        return ToolResult(call.id, call.name, True, "ok", exit_code=0)

    def candidate_artifact(self) -> CandidateArtifact:
        return CandidateArtifact(
            self.base_commit,
            self.workspace_revision,
            ("app.py", "test_app.py") if self.patch else (),
            self.patch,
        )


class FakeCheckRunner:
    def __init__(self, repo_path, **kwargs) -> None:
        self.repo_path = repo_path
        self.kwargs = kwargs

    def run(self) -> CheckRunResult:
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
            status="passed",
            network="none",
            argv=profile.check_argv,
            exit_code=0,
            output="2 passed",
            error=None,
            duration_ms=4,
        )
        return CheckRunResult(
            status="passed",
            profile=profile,
            base_commit=_git(Path(self.repo_path), "rev-parse", "HEAD"),
            candidate_applied=self.kwargs.get("candidate_patch") is not None,
            phases=(phase,),
            duration_ms=5,
        )

    def close(self) -> None:
        return None


class ScriptedProvider:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls = 0

    def next_step(
        self, task: str, results: Sequence[ToolResult]
    ) -> ToolCall | FinalAnswer:
        del task
        self.calls += 1
        if self.kind == "plan":
            return ToolCall(
                "plan",
                "submit_change_plan",
                {
                    "goal": "Repair value handling",
                    "files": ["app.py", "test_app.py"],
                    "steps": ["Correct the value", "Add a regression test"],
                    "checks": ["python-pytest"],
                    "risks": [],
                },
            )
        if self.kind == "review":
            return ToolCall(
                "review",
                "submit_review",
                {"approved": True, "summary": "Patch is focused", "risks": []},
            )
        if not results:
            patch = (
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 1\n"
                "+VALUE = 2\n"
            )
            return ToolCall("patch", "apply_patch", {"unified_diff": patch})
        return ToolCall(
            "finish",
            "finish",
            {"summary": "Fixed value handling", "risks": []},
        )


def provider_factory(
    definitions: Sequence[ToolDefinition],
    system_prompt: str,
    idempotency_key: str,
) -> Provider:
    del system_prompt, idempotency_key
    names = {definition.name for definition in definitions}
    if "submit_change_plan" in names:
        return ScriptedProvider("plan")
    if "submit_review" in names:
        return ScriptedProvider("review")
    return ScriptedProvider("change")


def test_real_service_workflow_runs_fixed_graph_with_fake_model(
    monkeypatch, tmp_path: Path, clean_python_repo: Path
) -> None:
    monkeypatch.setattr(runtime_module, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(runtime_module, "CheckRunner", FakeCheckRunner)
    data = tmp_path / "data"
    bootstrap_service = RunService(
        data,
        allowed_repo_roots=(tmp_path,),
        runner=None,
        start_worker=False,
    )
    runner = ServiceWorkflowRunner(
        bootstrap_service.database,
        bootstrap_service.artifacts,
        provider_factory=provider_factory,
    )
    bootstrap_service._runner = runner
    bootstrap_service.start()
    try:
        created = bootstrap_service.create_run(
            repo_path=clean_python_repo,
            task="Change value handling and cover it",
            auto_approve=True,
        )
        finished = bootstrap_service.wait(created.run_id, timeout=10)

        assert finished.status == "succeeded"
        assert finished.base_commit == _git(clean_python_repo, "rev-parse", "HEAD")
        assert finished.plan is not None
        assert finished.summary == "Candidate patch and verification evidence are ready for human review"
        assert [check.ok for check in finished.checks] == [True]
        patch = bootstrap_service.artifact_path(created.run_id, "patch").read_text(
            encoding="utf-8"
        )
        assert "+VALUE = 2" in patch
        trace = bootstrap_service.artifact_path(created.run_id, "trace").read_text(
            encoding="utf-8"
        )
        assert '"event":"workflow_checkpoint"' in trace
        assert '"node":"finalize"' in trace
    finally:
        bootstrap_service.close()


def test_no_review_variant_short_circuits_review_provider(
    monkeypatch, tmp_path: Path, clean_python_repo: Path
) -> None:
    monkeypatch.setattr(runtime_module, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(runtime_module, "CheckRunner", FakeCheckRunner)
    review_requests = 0

    def no_review_provider_factory(definitions, system_prompt, idempotency_key):
        nonlocal review_requests
        names = {definition.name for definition in definitions}
        if "submit_review" in names:
            review_requests += 1
            raise AssertionError("no-review must not create a review provider")
        return provider_factory(definitions, system_prompt, idempotency_key)

    service = RunService(
        tmp_path / "no-review-data",
        allowed_repo_roots=(tmp_path,),
        start_worker=False,
    )
    service._runner = ServiceWorkflowRunner(
        service.database,
        service.artifacts,
        provider_factory=no_review_provider_factory,
        review_enabled=False,
    )
    service.start()
    try:
        created = service.create_run(
            repo_path=clean_python_repo,
            task="Change value handling and cover it",
            auto_approve=True,
        )
        finished = service.wait(created.run_id, timeout=10)

        assert finished.status == "succeeded"
        assert review_requests == 0
        cached = service.database.side_effect(
            created.run_id,
            f"workflow:{created.run_id}:review:0",
        )
        assert cached is not None
        assert cached["detail"]["review_enabled"] is False
    finally:
        service.close()


def test_model_loops_do_not_execute_past_repository_tool_budget(
    clean_python_repo: Path,
) -> None:
    sandbox = FakeSandbox(clean_python_repo)

    outcome = runtime_module._run_model_loop(
        ScriptedProvider("change"),
        "repair the value",
        sandbox,
        max_calls=0,
    )

    assert outcome.ok is False
    assert outcome.tool_results == ()
    assert outcome.error == "model exceeded 0 repository tool calls"
    assert sandbox.patch == ""

    with pytest.raises(RuntimeError, match="exceeded 0 repository tool calls"):
        runtime_module._run_until_named_tool(
            ScriptedProvider("change"),
            "plan the repair",
            sandbox,
            target="submit_change_plan",
            max_calls=0,
        )
    assert sandbox.patch == ""


def test_model_token_budget_accepts_boundary_and_rejects_overage() -> None:
    usage = {"complete": True, "total_tokens": 1}

    runtime_module._enforce_token_budget(runtime_module.MAX_RUN_TOKENS - 1, usage)
    with pytest.raises(RuntimeError, match="token budget exceeded"):
        runtime_module._enforce_token_budget(runtime_module.MAX_RUN_TOKENS, usage)
