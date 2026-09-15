from __future__ import annotations

import pytest

from repo_agent.agent_tools import AGENT_TOOL_DEFINITIONS, AgentToolExecutor
from repo_agent.models import CandidateArtifact, ToolCall, ToolResult
from repo_agent.openai_provider import strict_response_tools


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
        "read_files",
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


def test_read_files_strict_schema_keeps_supported_array_bounds() -> None:
    tools = strict_response_tools(AGENT_TOOL_DEFINITIONS)
    read_files = next(tool for tool in tools if tool["name"] == "read_files")
    parameters = read_files["parameters"]
    assert isinstance(parameters, dict)
    paths = parameters["properties"]["paths"]

    assert paths["minItems"] == 1
    assert paths["maxItems"] == 4
    assert "uniqueItems" not in paths


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


def test_read_files_batches_safe_paths_with_hard_line_and_byte_limits() -> None:
    executor, sandbox = _executor()

    result = executor.execute(
        ToolCall("r1", "read_files", {"paths": ["src/a.py", "tests/test_a.py"]})
    )

    assert result.ok
    assert "===== src/a.py =====" in result.output
    assert "===== tests/test_a.py =====" in result.output
    assert sandbox.calls == [
        ToolCall(
            "r1",
            "read_file",
            {"path": "src/a.py", "start_line": 1, "end_line": 200},
        ),
        ToolCall(
            "r1",
            "read_file",
            {"path": "tests/test_a.py", "start_line": 1, "end_line": 200},
        ),
    ]

    sandbox.execute = lambda call: ToolResult(
        call.id, call.name, True, "界" * 20_000, exit_code=0
    )
    bounded = executor.execute(ToolCall("r2", "read_files", {"paths": ["src/a.py"]}))
    assert len(bounded.output.encode("utf-8")) <= 16 * 1024
    assert bounded.truncated is True


@pytest.mark.parametrize(
    "paths",
    [[], ["a", "b", "c", "d", "e"], ["a", "a"], ["../outside"]],
)
def test_read_files_rejects_invalid_path_sets(paths: list[str]) -> None:
    executor, sandbox = _executor()

    result = executor.execute(ToolCall("r", "read_files", {"paths": paths}))

    assert result.ok is False
    assert sandbox.calls == []


def test_successful_patch_invokes_mutation_callback_and_failure_is_closed() -> None:
    sandbox = FakeSandbox()
    sandbox.candidate_artifact = lambda: CandidateArtifact(
        "a" * 40, 4, ("src/app.py",), "persisted diff"
    )
    persisted: list[str] = []
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path="D:/fixture",
        run_id="a" * 32,
        allow_bootstrap=False,
        mutation_callback=persisted.append,
    )

    result = executor.execute(
        ToolCall("p", "apply_patch", {"unified_diff": "patch text"})
    )
    assert result.ok
    assert persisted == ["persisted diff"]

    def fail_write(_patch: str) -> None:
        raise OSError("disk full")

    failing = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path="D:/fixture",
        run_id="a" * 32,
        allow_bootstrap=False,
        mutation_callback=fail_write,
    )
    failed = failing.execute(
        ToolCall("p2", "apply_patch", {"unified_diff": "patch text"})
    )
    assert failed.ok is False
    assert "candidate patch persistence failed" in (failed.error or "")
    assert "disk full" in (failed.error or "")
    assert failing.mutation_persistence_failed is True


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
