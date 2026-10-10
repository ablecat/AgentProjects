from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from repo_agent.agent_tools import (
    AGENT_TOOL_DEFINITIONS,
    MAX_AGENT_BATCH_BYTES,
    MAX_AGENT_BATCH_LINES,
    AgentToolExecutor,
)
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


class ApplyingSandbox:
    workspace_revision = 0

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if call.name == "read_file":
            path = self.snapshot_path.joinpath(*str(call.arguments["path"]).split("/"))
            content = path.read_text(encoding="utf-8")
            start = int(call.arguments.get("start_line", 1))
            end = int(call.arguments.get("end_line", 200))
            lines = content.splitlines(keepends=True)
            return ToolResult(
                call.id,
                call.name,
                True,
                "".join(lines[start - 1 : end]),
                exit_code=0,
            )
        assert call.name == "apply_patch"
        patch = call.arguments["patch"]
        completed = subprocess.run(
            ["git", "apply", "--index", "--whitespace=error-all", "--"],
            cwd=self.snapshot_path,
            input=str(patch).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            return ToolResult(call.id, call.name, False, "", error="apply failed")
        self.workspace_revision += 1
        return ToolResult(call.id, call.name, True, "applied", exit_code=0)

    def candidate_artifact(self) -> CandidateArtifact:
        patch = subprocess.run(
            ["git", "diff", "--cached", "--no-color", "HEAD", "--"],
            cwd=self.snapshot_path,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.decode("utf-8")
        paths = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "HEAD", "--"],
            cwd=self.snapshot_path,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.decode("utf-8").splitlines()
        return CandidateArtifact(
            "a" * 40, self.workspace_revision, tuple(paths), patch
        )


def _git_repository(path: Path, content: bytes = b"old\n") -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Tool Test"], cwd=path, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "tool@example.invalid"],
        cwd=path,
        check=True,
    )
    (path / "app.py").write_bytes(content)
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=path, check=True)
    return path


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
        "write_files",
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


def test_write_files_schema_is_bounded_and_strict() -> None:
    tools = strict_response_tools(AGENT_TOOL_DEFINITIONS)
    write_files = next(tool for tool in tools if tool["name"] == "write_files")
    parameters = write_files["parameters"]
    files = parameters["properties"]["files"]

    assert files["minItems"] == 1
    assert files["maxItems"] == 4
    assert files["items"]["required"] == ["path", "content"]
    assert files["items"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("before", "after", "marker_count"),
    [
        (b"old\n", "new\n", 0),
        (b"old", "new\n", 1),
        (b"old\n", "new", 1),
        (b"old", "new", 2),
    ],
)
def test_write_files_preserves_all_eof_newline_combinations(
    tmp_path: Path, before: bytes, after: str, marker_count: int
) -> None:
    repository = _git_repository(tmp_path / "repo", before)
    sandbox = ApplyingSandbox(repository)
    persisted: list[str] = []
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="a" * 32,
        allow_bootstrap=False,
        mutation_callback=persisted.append,
    )

    result = executor.execute(
        ToolCall(
            "write",
            "write_files",
            {"files": [{"path": "app.py", "content": after}]},
        )
    )

    assert result.ok
    assert result.name == "write_files"
    assert (repository / "app.py").read_bytes() == after.encode("utf-8")
    delegated = str(sandbox.calls[0].arguments["patch"])
    assert delegated.count("\\ No newline at end of file") == marker_count
    assert persisted == [sandbox.candidate_artifact().patch]


def test_write_files_skips_mixed_noops_and_rejects_an_all_noop_batch(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    (repository / "other.py").write_bytes(b"before\n")
    subprocess.run(["git", "add", "other.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "other"], cwd=repository, check=True
    )
    sandbox = ApplyingSandbox(repository)
    persisted: list[str] = []
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="b" * 32,
        allow_bootstrap=False,
        mutation_callback=persisted.append,
    )

    changed = executor.execute(
        ToolCall(
            "mixed",
            "write_files",
            {
                "files": [
                    {"path": "app.py", "content": "old\n"},
                    {"path": "other.py", "content": "after\n"},
                    {"path": "created.py", "content": "created"},
                ]
            },
        )
    )
    no_op = executor.execute(
        ToolCall(
            "noop",
            "write_files",
            {"files": [{"path": "app.py", "content": "old\n"}]},
        )
    )

    assert changed.ok
    assert len(sandbox.calls) == 1
    assert "a/app.py" not in str(sandbox.calls[0].arguments["patch"])
    assert (repository / "other.py").read_text(encoding="utf-8") == "after\n"
    assert (repository / "created.py").read_bytes() == b"created"
    assert not no_op.ok
    assert "makes no changes" in (no_op.error or "")
    assert len(persisted) == 1


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "../outside", "content": "x\n"}],
        [
            {"path": "same.py", "content": "x\n"},
            {"path": "SAME.py", "content": "y\n"},
        ],
        [
            {"path": "caf\u00e9.py", "content": "x\n"},
            {"path": "cafe\u0301.py", "content": "y\n"},
        ],
        [{"path": "APP.py", "content": "x\n"}],
    ],
)
def test_write_files_rejects_unsafe_duplicate_and_colliding_paths(
    tmp_path: Path, files: list[dict[str, str]]
) -> None:
    repository = _git_repository(tmp_path / "repo")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="c" * 32,
        allow_bootstrap=False,
    )

    result = executor.execute(ToolCall("write", "write_files", {"files": files}))

    assert not result.ok
    assert sandbox.calls == []


def test_write_files_rejects_hardlinked_targets(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path / "repo")
    hardlink = repository / "alias.py"
    try:
        hardlink.hardlink_to(repository / "app.py")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="2" * 32,
        allow_bootstrap=False,
    )

    result = executor.execute(
        ToolCall(
            "hardlink",
            "write_files",
            {"files": [{"path": "alias.py", "content": "changed\n"}]},
        )
    )

    assert not result.ok
    assert "not a safe regular file" in (result.error or "")
    assert sandbox.calls == []


def test_write_files_creates_a_new_file_below_a_missing_parent(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path / "repo")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="3" * 32,
        allow_bootstrap=False,
    )

    result = executor.execute(
        ToolCall(
            "nested",
            "write_files",
            {"files": [{"path": "new/package.py", "content": "VALUE = 1\n"}]},
        )
    )

    assert result.ok
    assert (repository / "new" / "package.py").read_bytes() == b"VALUE = 1\n"


def test_write_files_rejects_a_batch_over_the_total_byte_limit(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="4" * 32,
        allow_bootstrap=False,
    )
    content = "x" * (MAX_AGENT_BATCH_BYTES - 1) + "\n"

    result = executor.execute(
        ToolCall(
            "too-large",
            "write_files",
            {
                "files": [
                    {"path": f"new-{index}.py", "content": content}
                    for index in range(4)
                ]
            },
        )
    )

    assert not result.ok
    assert "total content exceeds" in (result.error or "")
    assert sandbox.calls == []


def test_write_files_preflights_every_disk_target_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _git_repository(tmp_path / "repo")
    (repository / "directory.py").mkdir()
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="d" * 32,
        allow_bootstrap=False,
    )
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)

    result = executor.execute(
        ToolCall(
            "preflight",
            "write_files",
            {
                "files": [
                    {"path": "app.py", "content": "changed\n"},
                    {"path": "directory.py", "content": "blocked\n"},
                ]
            },
        )
    )

    assert not result.ok
    assert "not a safe regular file" in (result.error or "")
    assert reads == []
    assert sandbox.calls == []


def test_durable_write_files_requires_a_complete_read_of_existing_targets(
    tmp_path: Path,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="1" * 32,
        allow_bootstrap=False,
        require_complete_read_before_write=True,
    )
    write = ToolCall(
        "write-known",
        "write_files",
        {"files": [{"path": "app.py", "content": "changed\n"}]},
    )

    read_call = ToolCall("read-known", "read_files", {"paths": ["app.py"]})
    rejected = executor.execute(write)
    read = executor.execute(read_call)
    executor.confirm_complete_read(
        read_call,
        ToolResult(
            read.call_id,
            read.name,
            read.ok,
            read.output,
            error=read.error,
            exit_code=read.exit_code,
            truncated=True,
        ),
    )
    rejected_after_truncation = executor.execute(write)
    executor.confirm_complete_read(read_call, read)
    sandbox.calls.clear()
    accepted = executor.execute(write)

    assert not rejected.ok
    assert "must be read completely" in (rejected.error or "")
    assert not rejected_after_truncation.ok
    assert read.ok and not read.truncated
    assert accepted.ok
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0].name == "apply_patch"


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
    ("content", "expected"),
    [
        (("line\n" * (MAX_AGENT_BATCH_LINES + 1)).encode(), "200 lines"),
        (b"x" * (MAX_AGENT_BATCH_BYTES + 1), "16384 bytes"),
    ],
)
def test_read_files_rejects_files_that_cannot_be_returned_completely(
    tmp_path: Path, content: bytes, expected: str
) -> None:
    repository = _git_repository(tmp_path / "repo", content)
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="e" * 32,
        allow_bootstrap=False,
    )

    result = executor.execute(
        ToolCall("read-oversized", "read_files", {"paths": ["app.py"]})
    )

    assert not result.ok
    assert expected in (result.error or "")
    assert sandbox.calls == []


@pytest.mark.parametrize(
    "content",
    [
        "line\n" * (MAX_AGENT_BATCH_LINES + 1),
        "x" * (MAX_AGENT_BATCH_BYTES + 1),
        "contains\x00nul\n",
        "uses\r\ncrlf\r\n",
        "lone-surrogate-\ud800",
    ],
)
def test_write_files_rejects_unreadable_or_oversized_content(
    tmp_path: Path, content: str
) -> None:
    repository = _git_repository(tmp_path / "repo")
    sandbox = ApplyingSandbox(repository)
    executor = AgentToolExecutor(
        sandbox,  # type: ignore[arg-type]
        repo_path=str(repository),
        run_id="f" * 32,
        allow_bootstrap=False,
    )

    result = executor.execute(
        ToolCall(
            "write-invalid",
            "write_files",
            {"files": [{"path": "new.py", "content": content}]},
        )
    )

    assert not result.ok
    assert sandbox.calls == []


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
