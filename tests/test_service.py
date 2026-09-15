from __future__ import annotations

from collections.abc import Sequence

import pytest

from repo_agent.models import CandidateArtifact, FinalAnswer, ToolCall, ToolResult
from repo_agent.service import run_agent


class RecordingSandbox:
    instances: list[RecordingSandbox] = []

    def __init__(self, repo_path, **kwargs) -> None:
        self.repo_path = repo_path
        self.kwargs = kwargs
        self.calls: list[ToolCall] = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self) -> RecordingSandbox:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.closed = True
        return False

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call.id, call.name, True, f"ran {call.name}", exit_code=0)


class OneToolProvider:
    def next_step(
        self, task: str, results: Sequence[ToolResult]
    ) -> ToolCall | FinalAnswer:
        if not results:
            return ToolCall("one", "git_status", {})
        return FinalAnswer(f"finished {task}")


def test_run_agent_normalizes_task_and_owns_one_sandbox(tmp_path) -> None:
    RecordingSandbox.instances.clear()

    result = run_agent(
        task="  inspect repository  ",
        repo_path=tmp_path,
        image="repo-agent-maven:0.1",
        max_steps=4,
        timeout_seconds=12.5,
        max_output_bytes=4096,
        provider=OneToolProvider(),
        sandbox_factory=RecordingSandbox,
    )

    assert result.status == "completed"
    assert result.task == "inspect repository"
    assert result.answer == "finished inspect repository"
    assert len(RecordingSandbox.instances) == 1
    sandbox = RecordingSandbox.instances[0]
    assert sandbox.repo_path == tmp_path
    assert sandbox.kwargs == {
        "image": "repo-agent-maven:0.1",
        "timeout_seconds": 12.5,
        "max_output_bytes": 4096,
        "allow_mutations": False,
    }
    assert sandbox.calls == [ToolCall("one", "git_status", {})]
    assert sandbox.closed is True


def test_run_agent_captures_candidate_before_sandbox_closes(tmp_path) -> None:
    class CandidateSandbox(RecordingSandbox):
        def candidate_artifact(self) -> CandidateArtifact:
            assert self.closed is False
            return CandidateArtifact(
                base_commit="a" * 40,
                revision=1,
                changed_paths=("src/app.py",),
                patch="diff --git a/src/app.py b/src/app.py\n",
            )

    CandidateSandbox.instances.clear()
    result = run_agent(
        task="prepare candidate",
        repo_path=tmp_path,
        provider=OneToolProvider(),
        sandbox_factory=CandidateSandbox,
    )

    assert result.candidate == CandidateArtifact(
        base_commit="a" * 40,
        revision=1,
        changed_paths=("src/app.py",),
        patch="diff --git a/src/app.py b/src/app.py\n",
    )
    assert CandidateSandbox.instances[-1].closed is True


def test_run_agent_passes_explicit_mutation_capability(tmp_path) -> None:
    RecordingSandbox.instances.clear()

    run_agent(
        task="prepare candidate",
        repo_path=tmp_path,
        provider=OneToolProvider(),
        sandbox_factory=RecordingSandbox,
        allow_mutations=True,
    )

    assert RecordingSandbox.instances[-1].kwargs["allow_mutations"] is True


def test_run_agent_rejects_non_boolean_mutation_capability(tmp_path) -> None:
    RecordingSandbox.instances.clear()

    with pytest.raises(ValueError, match="allow_mutations must be a boolean"):
        run_agent(
            task="prepare candidate",
            repo_path=tmp_path,
            sandbox_factory=RecordingSandbox,
            allow_mutations=1,  # type: ignore[arg-type]
        )

    assert RecordingSandbox.instances == []


@pytest.mark.parametrize("task", ["", "   ", "bad\x00task", None, 42])
def test_run_agent_rejects_empty_or_non_text_tasks_before_sandbox(
    task, tmp_path
) -> None:
    RecordingSandbox.instances.clear()

    with pytest.raises(ValueError, match="non-whitespace"):
        run_agent(
            task=task,
            repo_path=tmp_path,
            sandbox_factory=RecordingSandbox,
        )

    assert RecordingSandbox.instances == []
