from __future__ import annotations

from repo_agent.agent_tools import AGENT_TOOL_DEFINITIONS, AgentToolExecutor
from repo_agent.models import CandidateArtifact, ToolCall, ToolResult


class FakeSandbox:
    workspace_revision = 3

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if call.name == "list_files":
            output = "\n".join(f"src/file{index}.py" for index in range(5))
        else:
            output = f"ran {call.name}"
        return ToolResult(call.id, call.name, True, output, exit_code=0)

    def candidate_artifact(self) -> CandidateArtifact:
        return CandidateArtifact("a" * 40, 1, ("src/app.py",), "")


def _executor() -> tuple[AgentToolExecutor, FakeSandbox]:
    sandbox = FakeSandbox()
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path="D:/fixture",
        run_id="a" * 32,
        allow_bootstrap=False,
    )
    return executor, sandbox


def test_model_tool_surface_matches_the_workflow_contract() -> None:
    assert [definition.name for definition in AGENT_TOOL_DEFINITIONS] == [
        "list_files",
        "read_file",
        "search_code",
        "apply_patch",
        "get_diff",
        "run_check",
        "finish",
    ]
    assert all(
        definition.input_schema.get("additionalProperties") is False
        for definition in AGENT_TOOL_DEFINITIONS
    )


def test_executor_translates_canonical_names_without_shell() -> None:
    executor, sandbox = _executor()

    search = executor.execute(
        ToolCall(
            "s1",
            "search_code",
            {"query": "needle", "path": "src", "glob": "*.py", "limit": 12},
        )
    )
    patch = executor.execute(
        ToolCall("p1", "apply_patch", {"unified_diff": "patch text"})
    )
    diff = executor.execute(ToolCall("d1", "get_diff", {}))

    assert search.ok and patch.ok and diff.ok
    assert sandbox.calls == [
        ToolCall("s1", "search", {"pattern": "needle", "glob": "src/*.py"}),
        ToolCall("p1", "apply_patch", {"patch": "patch text"}),
        ToolCall("d1", "git_diff", {"staged": True}),
    ]


def test_executor_enforces_file_result_limit_after_sandbox() -> None:
    executor, _ = _executor()
    result = executor.execute(ToolCall("l1", "list_files", {"limit": 2}))

    assert result.ok
    assert result.output.splitlines() == ["src/file0.py", "src/file1.py"]
    assert result.truncated is True


def test_executor_rejects_unknown_and_unsafe_arguments() -> None:
    executor, sandbox = _executor()

    unknown = executor.execute(ToolCall("x", "shell", {"command": "whoami"}))
    escaped = executor.execute(
        ToolCall("s", "search_code", {"query": "x", "path": "../outside"})
    )

    assert unknown.ok is False
    assert "Unknown model tool" in (unknown.error or "")
    assert escaped.ok is False
    assert "RepositoryPathError" in (escaped.error or "")
    assert sandbox.calls == []


def test_finish_is_not_forwarded_to_the_sandbox() -> None:
    executor, sandbox = _executor()
    result = executor.execute(
        ToolCall("f", "finish", {"summary": "done", "risks": []})
    )
    assert result.ok is False
    assert sandbox.calls == []
