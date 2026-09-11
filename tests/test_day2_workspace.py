from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

import pytest

import repo_agent.sandbox as sandbox_module
from repo_agent.loop import AgentLoop
from repo_agent.models import FinalAnswer, ToolCall, ToolResult
from repo_agent.sandbox import CommandOutcome, DockerSandbox, SandboxError


CONTAINER_ID = "d" * 64


class FakeDockerRunner:
    """Return deterministic container outcomes while recording the Docker boundary."""

    def __init__(self, start_output: str = "read result\n") -> None:
        self.commands: list[tuple[str, ...]] = []
        self.start_output = start_output

    def __call__(
        self,
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        del timeout_seconds, max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        operation = command[2]
        if operation == "create":
            return CommandOutcome(0, f"{CONTAINER_ID}\n")
        if operation == "start":
            return CommandOutcome(0, self.start_output)
        if operation == "rm":
            return CommandOutcome(0, f"{CONTAINER_ID}\n")
        if operation == "inspect":
            return CommandOutcome(1, "No such container\n")
        raise AssertionError(f"unexpected Docker command: {command}")


@pytest.fixture
def day2_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Day 2 Test")
    _git(repo, "config", "user.email", "day2@example.invalid")
    (repo / "app.py").write_text(
        'def greet(name):\n    return f"hello {name}"\n', encoding="utf-8"
    )
    (repo / "obsolete.py").write_text("OBSOLETE = True\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "day2-fixture"\n', encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo.resolve()


SUCCESSFUL_PATCH = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello {name}"
+    return f"hello, {name}!"
diff --git a/created.py b/created.py
new file mode 100644
--- /dev/null
+++ b/created.py
@@ -0,0 +1,2 @@
+def answer():
+    return 42
diff --git a/obsolete.py b/obsolete.py
deleted file mode 100644
--- a/obsolete.py
+++ /dev/null
@@ -1 +0,0 @@
-OBSOLETE = True
"""


FAILING_MULTIFILE_PATCH = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello {name}"
+    return f"changed {name}"
diff --git a/obsolete.py b/obsolete.py
--- a/obsolete.py
+++ b/obsolete.py
@@ -1 +1 @@
-THIS_CONTEXT_DOES_NOT_EXIST = True
+OBSOLETE = False
"""


CORRECTION_PATCH = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello, {name}!"
+    return f"hi, {name}!"
"""


def test_repo_map_describes_committed_python_snapshot_without_docker(
    day2_repo: Path,
) -> None:
    runner = FakeDockerRunner()
    with DockerSandbox(day2_repo, command_runner=runner) as sandbox:
        result = sandbox.execute(
            ToolCall(
                "map-1",
                "repo_map",
                {"max_files": 20, "include_symbols": True},
            )
        )

        assert result.ok
        assert result.exit_code == 0
        assert not result.truncated
        assert f"HEAD {_git(day2_repo, 'rev-parse', 'HEAD')}" in result.output
        assert "MANIFESTS pyproject.toml" in result.output
        assert "- app.py" in result.output
        assert "- app.py:1 def greet" in result.output
        assert runner.commands == []


def test_apply_patch_is_disabled_without_explicit_mutation_capability(
    day2_repo: Path,
) -> None:
    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner()
    ) as sandbox:
        before = (sandbox.snapshot_path / "app.py").read_bytes()
        result = sandbox.execute(
            ToolCall("blocked", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )

        assert not result.ok
        assert result.error == "Candidate mutations are disabled for this sandbox run"
        assert sandbox.workspace_revision == 0
        assert (sandbox.snapshot_path / "app.py").read_bytes() == before
        assert sandbox.candidate_artifact() is None


def test_apply_patch_changes_only_candidate_and_captures_complete_artifact(
    day2_repo: Path,
) -> None:
    source_head = _git(day2_repo, "rev-parse", "HEAD")
    source_status = _git(day2_repo, "status", "--short")
    source_app = (day2_repo / "app.py").read_bytes()
    runner = FakeDockerRunner()

    with DockerSandbox(
        day2_repo, command_runner=runner, allow_mutations=True
    ) as sandbox:
        assert sandbox.workspace_revision == 0

        result = sandbox.execute(
            ToolCall("patch-1", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )

        assert result.ok
        assert result.exit_code == 0
        assert sandbox.workspace_revision == 1
        assert (sandbox.snapshot_path / "app.py").read_text(encoding="utf-8") == (
            'def greet(name):\n    return f"hello, {name}!"\n'
        )
        assert (sandbox.snapshot_path / "created.py").read_text(
            encoding="utf-8"
        ) == "def answer():\n    return 42\n"
        assert not (sandbox.snapshot_path / "obsolete.py").exists()

        artifact = sandbox.candidate_artifact()
        assert artifact is not None
        assert artifact.base_commit == source_head
        assert artifact.revision == 1
        assert artifact.changed_paths == ("app.py", "created.py", "obsolete.py")
        assert not artifact.truncated
        assert "diff --git a/app.py b/app.py" in artifact.patch
        assert "diff --git a/created.py b/created.py" in artifact.patch
        assert "diff --git a/obsolete.py b/obsolete.py" in artifact.patch
        assert runner.commands == []

    assert _git(day2_repo, "rev-parse", "HEAD") == source_head
    assert _git(day2_repo, "status", "--short") == source_status
    assert (day2_repo / "app.py").read_bytes() == source_app
    assert not (day2_repo / "created.py").exists()
    assert (day2_repo / "obsolete.py").read_text(encoding="utf-8") == (
        "OBSOLETE = True\n"
    )


def test_failed_multifile_patch_is_atomic_and_does_not_advance_revision(
    day2_repo: Path,
) -> None:
    runner = FakeDockerRunner()
    with DockerSandbox(
        day2_repo, command_runner=runner, allow_mutations=True
    ) as sandbox:
        before_app = (sandbox.snapshot_path / "app.py").read_bytes()
        before_obsolete = (sandbox.snapshot_path / "obsolete.py").read_bytes()

        result = sandbox.execute(
            ToolCall(
                "patch-fails",
                "apply_patch",
                {"patch": FAILING_MULTIFILE_PATCH},
            )
        )

        assert not result.ok
        assert result.exit_code != 0
        assert "did not apply cleanly" in (result.error or "")
        assert sandbox.workspace_revision == 0
        assert (sandbox.snapshot_path / "app.py").read_bytes() == before_app
        assert (sandbox.snapshot_path / "obsolete.py").read_bytes() == before_obsolete
        assert sandbox.candidate_artifact() is None
        assert _git(sandbox.snapshot_path, "status", "--short") == ""
        assert runner.commands == []


def test_apply_timeout_after_mutation_restores_candidate(
    day2_repo: Path, monkeypatch
) -> None:
    real_run_git_input = sandbox_module._run_git_input

    def apply_then_lose_result(repo, input_bytes, *args):
        outcome = real_run_git_input(repo, input_bytes, *args)
        if args and args[0] == "apply" and "--check" not in args:
            assert outcome.returncode == 0
            raise SandboxError("simulated lost apply result")
        return outcome

    monkeypatch.setattr(sandbox_module, "_run_git_input", apply_then_lose_result)

    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner(), allow_mutations=True
    ) as sandbox:
        before = (sandbox.snapshot_path / "app.py").read_bytes()
        result = sandbox.execute(
            ToolCall("uncertain-apply", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )

        assert not result.ok
        assert "simulated lost apply result" in (result.error or "")
        assert sandbox.workspace_revision == 0
        assert (sandbox.snapshot_path / "app.py").read_bytes() == before
        assert not (sandbox.snapshot_path / "created.py").exists()
        assert (sandbox.snapshot_path / "obsolete.py").exists()
        assert sandbox.candidate_artifact() is None


def test_later_patch_can_correct_a_path_changed_earlier(day2_repo: Path) -> None:
    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner(), allow_mutations=True
    ) as sandbox:
        first = sandbox.execute(
            ToolCall("initial", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )
        second = sandbox.execute(
            ToolCall("correction", "apply_patch", {"patch": CORRECTION_PATCH})
        )

        assert first.ok and second.ok
        assert sandbox.workspace_revision == 2
        assert (sandbox.snapshot_path / "app.py").read_text(
            encoding="utf-8"
        ) == 'def greet(name):\n    return f"hi, {name}!"\n'
        artifact = sandbox.candidate_artifact()
        assert artifact is not None
        assert artifact.revision == 2
        assert artifact.changed_paths == ("app.py", "created.py", "obsolete.py")
        assert 'return f"hi, {name}!"' in artifact.patch
        assert 'return f"hello, {name}!"' not in artifact.patch


def test_failed_correction_restores_all_prior_candidate_changes(
    day2_repo: Path, monkeypatch
) -> None:
    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner(), allow_mutations=True
    ) as sandbox:
        first = sandbox.execute(
            ToolCall("initial", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )
        assert first.ok
        prior_artifact = sandbox.candidate_artifact()

        def reject_postcondition(*_args, **_kwargs) -> None:
            raise SandboxError("simulated postcondition failure")

        monkeypatch.setattr(sandbox, "_verify_patch_postcondition", reject_postcondition)
        correction = sandbox.execute(
            ToolCall("correction", "apply_patch", {"patch": CORRECTION_PATCH})
        )

        assert not correction.ok
        assert "simulated postcondition failure" in (correction.error or "")
        assert sandbox.workspace_revision == 1
        assert sandbox.candidate_artifact() == prior_artifact
        assert (sandbox.snapshot_path / "app.py").read_text(
            encoding="utf-8"
        ) == 'def greet(name):\n    return f"hello, {name}!"\n'


def test_candidate_size_limit_rolls_back_the_attempt(
    day2_repo: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sandbox_module, "_MAX_ARTIFACT_BYTES", 32)

    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner(), allow_mutations=True
    ) as sandbox:
        result = sandbox.execute(
            ToolCall("too-large", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )

        assert not result.ok
        assert "candidate patch exceeds 32 bytes" in (result.error or "")
        assert sandbox.workspace_revision == 0
        assert sandbox.candidate_artifact() is None
        assert (sandbox.snapshot_path / "app.py").read_text(
            encoding="utf-8"
        ) == 'def greet(name):\n    return f"hello {name}"\n'


def test_candidate_file_limit_rejects_before_mutation(
    day2_repo: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sandbox_module, "_MAX_CANDIDATE_FILES", 2)

    with DockerSandbox(
        day2_repo, command_runner=FakeDockerRunner(), allow_mutations=True
    ) as sandbox:
        result = sandbox.execute(
            ToolCall("too-many", "apply_patch", {"patch": SUCCESSFUL_PATCH})
        )

        assert not result.ok
        assert "candidate is limited to 2 changed files" in (result.error or "")
        assert sandbox.workspace_revision == 0
        assert sandbox.candidate_artifact() is None
        assert (sandbox.snapshot_path / "app.py").read_text(
            encoding="utf-8"
        ) == 'def greet(name):\n    return f"hello {name}"\n'


def test_agent_loop_allows_same_read_after_successful_mutation(
    day2_repo: Path,
) -> None:
    class ReadPatchReadProvider:
        def next_step(
            self, task: str, results: Sequence[ToolResult]
        ) -> ToolCall | FinalAnswer:
            del task
            if len(results) == 0:
                return ToolCall("read-before", "read_file", {"path": "app.py"})
            if len(results) == 1:
                return ToolCall(
                    "mutate",
                    "apply_patch",
                    {
                        "patch": """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello {name}"
+    return f"hello, {name}!"
"""
                    },
                )
            if len(results) == 2:
                return ToolCall("read-after", "read_file", {"path": "app.py"})
            return FinalAnswer("candidate inspected after mutation")

    runner = FakeDockerRunner()
    with DockerSandbox(
        day2_repo, command_runner=runner, allow_mutations=True
    ) as sandbox:
        result = AgentLoop(
            ReadPatchReadProvider(), sandbox, max_steps=4
        ).run("update greeting")

        assert result.status == "completed"
        assert result.answer == "candidate inspected after mutation"
        assert [item.name for item in result.tool_results] == [
            "read_file",
            "apply_patch",
            "read_file",
        ]
        assert all(item.ok for item in result.tool_results)
        assert sandbox.workspace_revision == 1

    docker_operations = [command[2] for command in runner.commands]
    assert docker_operations == ["create", "start", "rm"] * 2


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()
