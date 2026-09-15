from __future__ import annotations

import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import pytest

import repo_agent.processes as process_module
import repo_agent.sandbox as sandbox_module
from repo_agent.models import ToolCall
from repo_agent.processes import IsolatedProcess
from repo_agent.sandbox import (
    CommandOutcome,
    DockerSandbox,
    SandboxCleanupError,
    SandboxError,
    SandboxPolicy,
    SubprocessCommandRunner,
)


CONTAINER_ID = "a" * 64


class FakeDockerRunner:
    def __init__(self, start: CommandOutcome | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []
        self.start = start or CommandOutcome(0, "tool output\n")
        self.create = CommandOutcome(0, f"{CONTAINER_ID}\n")
        self.inspect = CommandOutcome(1, "No such container\n")
        self.inspect_container_id: str | None = None
        self.remove = CommandOutcome(0, f"{CONTAINER_ID}\n")

    def __call__(
        self,
        argv,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        del max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        self.timeouts.append(timeout_seconds)
        operation = command[2]
        if operation == "create":
            return self.create
        if operation == "start":
            return self.start
        if operation == "inspect":
            if self.inspect_container_id is not None:
                return self._owned_inspect(self.inspect_container_id)
            return self.inspect
        if operation == "rm":
            return self.remove
        raise AssertionError(f"unexpected Docker command: {command}")

    def _owned_inspect(self, container_id: str) -> CommandOutcome:
        create = next(command for command in self.commands if command[2] == "create")
        labels = [
            create[index + 1]
            for index, value in enumerate(create)
            if value == "--label"
        ]
        run_id = labels[0].split("=", 1)[1]
        sequence = labels[1].split("=", 1)[1]
        return CommandOutcome(
            0,
            "Docker warning before metadata\n"
            f"repo-agent-owned {container_id} {run_id} {sequence}\n",
        )


@pytest.fixture
def committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Sandbox Test")
    _git(repo, "config", "user.email", "sandbox@example.invalid")
    (repo / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "fixture")

    (repo / "tracked.txt").write_text("dirty host value\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=must-not-leak\n", encoding="utf-8")
    return repo.resolve()


def test_default_policy_matches_verified_docker_contract() -> None:
    policy = SandboxPolicy()

    assert policy.docker_args() == (
        "--init",
        "--pull",
        "never",
        "--restart",
        "no",
        "--network",
        "none",
        "--read-only",
        "--user",
        "10001:10001",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--pids-limit",
        "256",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--log-driver",
        "none",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
    )


def test_snapshot_contains_only_committed_head_and_is_mounted_read_only(
    committed_repo: Path,
) -> None:
    runner = FakeDockerRunner()
    sandbox = DockerSandbox(committed_repo, command_runner=runner)

    with sandbox:
        snapshot = sandbox.snapshot_path
        run_root = snapshot.parent
        temp_base = run_root.parent
        assert snapshot != committed_repo
        assert not snapshot.is_relative_to(committed_repo)
        assert temp_base.parent == Path(tempfile.gettempdir()).resolve()
        assert (snapshot / "tracked.txt").read_text(encoding="utf-8") == "committed\n"
        assert not (snapshot / ".env").exists()
        assert _git(snapshot, "remote") == ""
        assert _git(snapshot, "rev-list", "--all", "--count") == "1"

        result = sandbox.execute(ToolCall("call-1", "git_status", {}))
        assert result.ok
        assert result.output == "tool output\n"
        assert result.exit_code == 0

        create = runner.commands[0]
        assert create[:3] == ("docker", "container", "create")
        name = create[create.index("--name") + 1]
        assert re.fullmatch(
            r"repo-agent-tool-[0-9a-f]{32}-1", name
        )
        labels = [
            create[index + 1]
            for index, value in enumerate(create)
            if value == "--label"
        ]
        assert len(labels) == 2
        assert labels[0].startswith(
            "io.github.ablecat.repo-agent.sandbox.run="
        )
        assert labels[1].endswith("=1")
        mount = create[create.index("--mount") + 1]
        assert mount == (
            f"type=bind,source={snapshot},target=/workspace,readonly"
        )
        assert create.count("--mount") == 1
        assert "/var/run/docker.sock" not in " ".join(create)
        entrypoint_index = create.index("--entrypoint")
        assert create[entrypoint_index : entrypoint_index + 3] == (
            "--entrypoint",
            "git",
            "repo-agent-python:0.1",
        )
        assert create[entrypoint_index + 3 :] == (
            "-c",
            "safe.directory=/workspace",
            "status",
            "--short",
            "--branch",
            "--untracked-files=all",
        )

        assert runner.commands[1] == (
            "docker",
            "container",
            "start",
            "--attach",
            CONTAINER_ID,
        )
        assert runner.commands[2] == (
            "docker",
            "container",
            "rm",
            "--force",
            CONTAINER_ID,
        )

    assert not run_root.exists()
    assert not temp_base.exists()
    assert (committed_repo / "tracked.txt").read_text(encoding="utf-8") == (
        "dirty host value\n"
    )
    assert (committed_repo / ".env").read_text(encoding="utf-8") == (
        "TOKEN=must-not-leak\n"
    )


def test_host_git_uses_sanitized_environment_and_safe_checkout_order(
    committed_repo: Path, monkeypatch, tmp_path: Path
) -> None:
    hostile_global = tmp_path / "hostile.gitconfig"
    hostile_global.write_text(
        "[filter \"evil\"]\n\tsmudge = definitely-must-not-run\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "hostile-hooks"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_global))
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    blocked_credentials = {
        "REPO_AGENT_API_KEY",
        "REPO_AGENT_BEARER_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
    }
    for index, name in enumerate(blocked_credentials):
        mixed_case_name = name.lower() if index % 2 else name
        monkeypatch.setenv(mixed_case_name, f"must-not-reach-git-{index}")
    monkeypatch.setenv("REPO_AGENT_TEST_MARKER", "preserved")

    real_capture = sandbox_module.run_isolated_capture
    git_calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def recording_capture(argv, **kwargs):
        if tuple(argv)[0] == "git":
            git_calls.append((tuple(argv), dict(kwargs["env"])))
        return real_capture(argv, **kwargs)

    monkeypatch.setattr(sandbox_module, "run_isolated_capture", recording_capture)
    with DockerSandbox(committed_repo, command_runner=FakeDockerRunner()):
        pass

    assert git_calls
    for _, environment in git_calls:
        normalized_names = {name.upper() for name in environment}
        assert "GIT_DIR" not in environment
        assert "GIT_CONFIG_COUNT" not in environment
        assert normalized_names.isdisjoint(blocked_credentials)
        assert environment["REPO_AGENT_TEST_MARKER"] == "preserved"
        assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
        assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_ATTR_NOSYSTEM"] == "1"
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert environment["GIT_NO_LAZY_FETCH"] == "1"
        assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"

    commands = [argv for argv, _ in git_calls]
    clone_index = next(index for index, argv in enumerate(commands) if "clone" in argv)
    remove_origin_index = next(
        index
        for index, argv in enumerate(commands)
        if argv[-3:] == ("remote", "remove", "origin")
    )
    checkout_index = next(
        index for index, argv in enumerate(commands) if "checkout" in argv
    )
    clone = commands[clone_index]
    checkout = commands[checkout_index]
    assert "--no-checkout" in clone
    assert any(argument.startswith("--template=") for argument in clone)
    assert clone_index < remove_origin_index < checkout_index
    assert any(argument.startswith("core.hooksPath=") for argument in checkout)
    assert "filter.lfs.smudge=" in checkout
    assert "filter.lfs.process=" in checkout
    assert "filter.lfs.required=false" in checkout


def test_lfs_attributes_cannot_start_a_smudge_process(committed_repo: Path) -> None:
    _git(committed_repo, "restore", "tracked.txt")
    (committed_repo / ".gitattributes").write_text(
        "tracked.txt filter=lfs\n", encoding="utf-8"
    )
    _git(committed_repo, "add", ".gitattributes")
    _git(committed_repo, "commit", "--quiet", "-m", "add lfs attributes")

    with DockerSandbox(
        committed_repo, command_runner=FakeDockerRunner()
    ) as sandbox:
        assert (sandbox.snapshot_path / "tracked.txt").read_text(
            encoding="utf-8"
        ) == "committed\n"


def test_timeout_returns_failure_and_removes_only_exact_container(
    committed_repo: Path,
) -> None:
    runner = FakeDockerRunner(
        CommandOutcome(None, "partial", truncated=True, timed_out=True)
    )

    with DockerSandbox(
        committed_repo, timeout_seconds=0.1, command_runner=runner
    ) as sandbox:
        result = sandbox.execute(ToolCall("slow", "list_files", {}))

    assert not result.ok
    assert result.exit_code is None
    assert result.output == "partial"
    assert result.truncated
    assert result.error == "Tool timed out after 0.1 seconds"
    assert [command[2] for command in runner.commands] == [
        "create",
        "start",
        "rm",
    ]
    assert runner.commands[-1][-1] == CONTAINER_ID
    assert runner.timeouts == [30.0, 0.1, 30.0]
    assert all(command[2] not in {"ls", "prune"} for command in runner.commands)


def test_non_allowed_exit_code_and_output_truncation_are_reported(
    committed_repo: Path,
) -> None:
    runner = FakeDockerRunner(CommandOutcome(7, "prefix", truncated=True))

    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        result = sandbox.execute(ToolCall("bad-exit", "read_file", {"path": "tracked.txt"}))

    assert not result.ok
    assert result.exit_code == 7
    assert result.output == "prefix"
    assert result.truncated
    assert result.error == "Tool exited with code 7"


def test_invalid_tool_call_never_reaches_docker(committed_repo: Path) -> None:
    runner = FakeDockerRunner()

    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        result = sandbox.execute(ToolCall("unknown", "shell", {"command": "whoami"}))

    assert not result.ok
    assert result.error is not None and "unknown tool" in result.error
    assert runner.commands == []


def test_read_file_rejects_internal_and_external_symlinks_before_docker(
    committed_repo: Path, tmp_path: Path
) -> None:
    runner = FakeDockerRunner()
    external = tmp_path / "outside.txt"
    external.write_text("outside\n", encoding="utf-8")

    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        secret = sandbox.snapshot_path / ".env"
        secret.write_text("TOKEN=private\n", encoding="utf-8")
        secret_link = sandbox.snapshot_path / "public.txt"
        external_link = sandbox.snapshot_path / "external.txt"
        try:
            secret_link.symlink_to(".env")
            external_link.symlink_to(external)
        except OSError:
            pytest.skip("file symlinks are unavailable on this platform")

        for call_id, path in (
            ("secret-link", "public.txt"),
            ("external-link", "external.txt"),
        ):
            result = sandbox.execute(ToolCall(call_id, "read_file", {"path": path}))

            assert result.ok is False
            assert "symbolic links or reparse points" in (result.error or "")

    assert runner.commands == []


def test_read_file_rejects_symlinked_parent_before_docker(
    committed_repo: Path, tmp_path: Path
) -> None:
    runner = FakeDockerRunner()
    external = tmp_path / "outside-directory"
    external.mkdir()
    (external / "value.txt").write_text("outside\n", encoding="utf-8")

    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        linked_parent = sandbox.snapshot_path / "linked-directory"
        try:
            linked_parent.symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable on this platform")

        result = sandbox.execute(
            ToolCall(
                "linked-parent",
                "read_file",
                {"path": "linked-directory/value.txt"},
            )
        )

        assert result.ok is False
        assert "symbolic links or reparse points" in (result.error or "")

    assert runner.commands == []


def test_malformed_create_output_recovers_by_exact_name(
    committed_repo: Path,
) -> None:
    runner = FakeDockerRunner()
    runner.create = CommandOutcome(0, "not-a-container-id\n")
    runner.inspect_container_id = CONTAINER_ID

    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        result = sandbox.execute(ToolCall("create-failed", "list_files", {}))

    assert not result.ok
    assert result.error == "Docker create did not return a valid container ID"
    assert [command[2] for command in runner.commands] == [
        "create",
        "inspect",
        "rm",
    ]
    inspected_name = runner.commands[1][-1]
    assert inspected_name == runner.commands[0][runner.commands[0].index("--name") + 1]
    assert runner.commands[2][-1] == CONTAINER_ID


def test_create_accepts_one_id_amid_warnings_but_rejects_multiple_ids(
    committed_repo: Path,
) -> None:
    warning_runner = FakeDockerRunner()
    warning_runner.create = CommandOutcome(
        0, f"Docker warning\n{CONTAINER_ID}\nmore context\n"
    )
    with DockerSandbox(committed_repo, command_runner=warning_runner) as sandbox:
        accepted = sandbox.execute(ToolCall("warning", "list_files", {}))
    assert accepted.ok

    ambiguous_runner = FakeDockerRunner()
    ambiguous_runner.create = CommandOutcome(
        0, f"{CONTAINER_ID}\n{'b' * 64}\n"
    )
    ambiguous_runner.inspect_container_id = CONTAINER_ID
    with DockerSandbox(committed_repo, command_runner=ambiguous_runner) as sandbox:
        rejected = sandbox.execute(ToolCall("ambiguous", "list_files", {}))
    assert not rejected.ok
    assert rejected.error == "Docker create did not return a valid container ID"
    assert [command[2] for command in ambiguous_runner.commands] == [
        "create",
        "inspect",
        "rm",
    ]


def test_recovery_refuses_container_with_mismatched_labels(
    committed_repo: Path,
) -> None:
    class WrongLabelRunner(FakeDockerRunner):
        def __call__(self, argv, **kwargs) -> CommandOutcome:
            command = tuple(argv)
            if command[2] == "inspect" and self.inspect_container_id is None:
                create = next(
                    item for item in self.commands if item[2] == "create"
                )
                labels = [
                    create[index + 1]
                    for index, value in enumerate(create)
                    if value == "--label"
                ]
                sequence = labels[1].split("=", 1)[1]
                self.inspect = CommandOutcome(
                    0,
                    "warning\n"
                    f"repo-agent-owned {CONTAINER_ID} {'b' * 32} {sequence}\n",
                )
            return super().__call__(argv, **kwargs)

    runner = WrongLabelRunner()
    runner.create = CommandOutcome(0, "malformed create output\n")
    sandbox = DockerSandbox(committed_repo, command_runner=runner)
    sandbox.__enter__()
    result = sandbox.execute(ToolCall("wrong-owner", "list_files", {}))

    assert not result.ok
    assert "mismatched ownership labels" in (result.error or "")
    assert all(command[2] != "rm" for command in runner.commands)

    runner.inspect_container_id = CONTAINER_ID
    sandbox.close()
    assert runner.commands[-1][2] == "rm"


def test_create_timeout_is_checked_again_during_context_cleanup(
    committed_repo: Path,
) -> None:
    class DelayedCreateRunner(FakeDockerRunner):
        def __init__(self) -> None:
            super().__init__()
            self.create = CommandOutcome(None, "", timed_out=True)
            self.inspect_count = 0

        def __call__(self, argv, **kwargs) -> CommandOutcome:
            command = tuple(argv)
            if command[2] == "inspect":
                self.inspect_count += 1
                self.inspect = (
                    CommandOutcome(1, "Error: No such container\n")
                    if self.inspect_count == 1
                    else self._owned_inspect(CONTAINER_ID)
                )
            return super().__call__(argv, **kwargs)

    runner = DelayedCreateRunner()
    with DockerSandbox(committed_repo, command_runner=runner) as sandbox:
        result = sandbox.execute(ToolCall("late-create", "list_files", {}))

    assert not result.ok
    assert result.error == "Docker create timed out after 30 seconds"
    assert [command[2] for command in runner.commands] == [
        "create",
        "inspect",
        "inspect",
        "rm",
    ]
    assert runner.commands[-1][-1] == CONTAINER_ID


def test_create_timeout_requires_multiple_delayed_absence_checks(
    committed_repo: Path, monkeypatch
) -> None:
    runner = FakeDockerRunner()
    runner.create = CommandOutcome(None, "", timed_out=True)
    delays: list[float] = []
    monkeypatch.setattr(sandbox_module.time, "sleep", delays.append)

    sandbox = DockerSandbox(committed_repo, command_runner=runner)
    sandbox.__enter__()
    run_root = sandbox.snapshot_path.parent
    result = sandbox.execute(ToolCall("absent", "list_files", {}))

    assert not result.ok
    assert "remains indeterminate" in (result.error or "")
    assert [command[2] for command in runner.commands] == [
        "create",
        "inspect",
        "inspect",
        "inspect",
        "inspect",
    ]
    assert delays == [0.1, 0.25, 0.5]
    with pytest.raises(SandboxCleanupError, match="remains indeterminate"):
        sandbox.close()
    assert run_root.exists()

    runner.inspect_container_id = CONTAINER_ID
    sandbox.close()
    assert not run_root.exists()
    assert runner.commands[-1][2] == "rm"


def test_failed_timeout_recovery_retains_state_for_a_later_retry(
    committed_repo: Path, monkeypatch
) -> None:
    runner = FakeDockerRunner()
    runner.create = CommandOutcome(None, "", timed_out=True)
    runner.inspect = CommandOutcome(2, "daemon unavailable\n")
    monkeypatch.setattr(sandbox_module.time, "sleep", lambda _: None)
    sandbox = DockerSandbox(committed_repo, command_runner=runner)
    sandbox.__enter__()
    run_root = sandbox.snapshot_path.parent

    result = sandbox.execute(ToolCall("unconfirmed", "list_files", {}))
    assert not result.ok
    assert "recovery exited with code 2" in (result.error or "")
    with pytest.raises(SandboxCleanupError, match="container names remain"):
        sandbox.close()
    assert run_root.exists()

    runner.inspect_container_id = CONTAINER_ID
    sandbox.close()
    assert not run_root.exists()


def test_cleanup_failure_can_be_retried_without_deleting_snapshot(
    committed_repo: Path,
) -> None:
    class RetryCleanupRunner(FakeDockerRunner):
        def __init__(self) -> None:
            super().__init__()
            self.fail_cleanup = True

        def __call__(self, argv, **kwargs) -> CommandOutcome:
            command = tuple(argv)
            if command[2] == "inspect":
                self.inspect = self._owned_inspect(CONTAINER_ID)
            if command[2] == "rm":
                self.remove = (
                    CommandOutcome(1, "daemon unavailable\n")
                    if self.fail_cleanup
                    else CommandOutcome(0, f"{CONTAINER_ID}\n")
                )
            return super().__call__(argv, **kwargs)

    runner = RetryCleanupRunner()
    sandbox = DockerSandbox(committed_repo, command_runner=runner)
    sandbox.__enter__()
    run_root = sandbox.snapshot_path.parent
    result = sandbox.execute(ToolCall("cleanup-retry", "list_files", {}))
    assert not result.ok
    assert "cleanup exited with code 1" in (result.error or "")

    with pytest.raises(Exception, match="run-owned containers remain"):
        sandbox.close()
    assert run_root.exists()

    runner.fail_cleanup = False
    sandbox.close()
    assert not run_root.exists()


def test_cleanup_refuses_replaced_temporary_parent_without_following_it(
    committed_repo: Path, tmp_path: Path
) -> None:
    sandbox = DockerSandbox(committed_repo, command_runner=FakeDockerRunner())
    sandbox.__enter__()
    run_root = sandbox.snapshot_path.parent
    temp_base = run_root.parent
    held_base = temp_base.with_name(f"{temp_base.name}-held")
    external = tmp_path / "must-survive"
    external.mkdir()
    proof = external / "proof.txt"
    proof.write_text("untouched\n", encoding="utf-8")

    temp_base.rename(held_base)
    try:
        try:
            temp_base.symlink_to(external, target_is_directory=True)
        except OSError:
            held_base.rename(temp_base)
            sandbox.close()
            pytest.skip("directory symlinks are unavailable on this platform")

        with pytest.raises(SandboxCleanupError, match="untrusted.*temporary base"):
            sandbox.close()
        assert proof.read_text(encoding="utf-8") == "untouched\n"
        assert held_base.exists()
    finally:
        if temp_base.is_symlink():
            temp_base.unlink()
        if held_base.exists():
            held_base.rename(temp_base)

    sandbox.close()
    assert proof.read_text(encoding="utf-8") == "untouched\n"


def test_remove_readonly_omits_unsupported_follow_symlinks(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "readonly.txt"
    target.write_text("delete me\n", encoding="utf-8")
    real_chmod = os.chmod
    chmod_calls: list[dict[str, object]] = []

    def chmod_without_follow_symlinks(path, mode, **kwargs) -> None:
        chmod_calls.append(kwargs)
        real_chmod(path, mode)

    monkeypatch.setattr(sandbox_module.os, "chmod", chmod_without_follow_symlinks)
    monkeypatch.setattr(sandbox_module.os, "supports_follow_symlinks", set())

    sandbox_module._remove_readonly(os.unlink, str(target), None)

    assert chmod_calls == [{}]
    assert not target.exists()


def test_remove_readonly_rejects_reparse_point_before_chmod(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "must-survive.txt"
    target.write_text("keep me\n", encoding="utf-8")

    monkeypatch.setattr(sandbox_module, "_is_reparse_point", lambda *_: True)
    monkeypatch.setattr(
        sandbox_module.os,
        "chmod",
        lambda *_args, **_kwargs: pytest.fail("chmod must not be called"),
    )

    with pytest.raises(SandboxError, match="refusing to chmod reparse point"):
        sandbox_module._remove_readonly(os.unlink, str(target), None)

    assert target.read_text(encoding="utf-8") == "keep me\n"


def test_sixty_four_character_name_prefix_still_generates_valid_name(
    committed_repo: Path,
) -> None:
    runner = FakeDockerRunner()
    with DockerSandbox(
        committed_repo,
        container_name_prefix="x" * 64,
        command_runner=runner,
    ) as sandbox:
        result = sandbox.execute(ToolCall("long-name", "list_files", {}))

    assert result.ok
    generated_name = runner.commands[0][runner.commands[0].index("--name") + 1]
    assert len(generated_name) == 99
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", generated_name)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("image", "--privileged"),
        ("image", "repo image:latest"),
        ("container_name_prefix", "bad,name"),
        ("container_name_prefix", "-looks-like-an-option"),
        ("timeout_seconds", 0),
        ("max_output_bytes", 0),
        ("allow_mutations", 1),
    ],
)
def test_invalid_configuration_is_rejected(
    committed_repo: Path, keyword: str, value: object
) -> None:
    with pytest.raises(ValueError):
        DockerSandbox(committed_repo, **{keyword: value})


def test_non_git_repository_and_empty_repository_are_rejected(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="Git worktree"):
        DockerSandbox(plain)

    empty = tmp_path / "empty"
    empty.mkdir()
    _git(empty, "init", "--quiet")
    with pytest.raises(ValueError, match="non-empty Git worktree"):
        DockerSandbox(empty)


def test_execute_requires_an_active_context(committed_repo: Path) -> None:
    sandbox = DockerSandbox(committed_repo, command_runner=FakeDockerRunner())
    with pytest.raises(SandboxError, match="active context manager"):
        sandbox.execute(ToolCall("call", "list_files", {}))


def test_subprocess_runner_enforces_byte_limit_and_timeout() -> None:
    runner = SubprocessCommandRunner()
    started = time.monotonic()
    limited = runner(
        (
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('x'*32); "
            "sys.stdout.flush(); time.sleep(30)",
        ),
        timeout_seconds=10,
        max_output_bytes=31,
    )
    assert limited.exit_code != 0
    assert limited.output == "x" * 31
    assert limited.truncated
    assert limited.timed_out is False
    assert time.monotonic() - started < 5

    timed_out = runner(
        (sys.executable, "-c", "import time; time.sleep(2)"),
        timeout_seconds=0.05,
        max_output_bytes=31,
    )
    assert timed_out.timed_out


@pytest.mark.parametrize(
    ("timeout_seconds", "max_output_bytes"),
    (
        (float("nan"), 1),
        (float("inf"), 1),
        (10**1000, 1),
        (-1, 1),
        (True, 1),
        (1, 0),
        (1, True),
    ),
)
def test_subprocess_runner_rejects_extreme_numeric_limits(
    timeout_seconds: object, max_output_bytes: object
) -> None:
    with pytest.raises(ValueError):
        SubprocessCommandRunner()(
            (sys.executable, "-c", "pass"),
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
            max_output_bytes=max_output_bytes,  # type: ignore[arg-type]
        )


def _process_state(process_id: int) -> str:
    if os.name == "nt":
        completed = subprocess.run(
            (
                "tasklist",
                "/FI",
                f"PID eq {process_id}",
                "/FO",
                "CSV",
                "/NH",
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        marker = f'"{process_id}"'.encode("ascii")
        return "running" if marker in completed.stdout else "gone"
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        stat_fields = stat_path.read_text(encoding="ascii").rsplit(") ", 1)[1]
    except (IndexError, OSError, UnicodeError):
        stat_fields = ""
    if stat_fields.startswith("Z "):
        return "zombie"
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return "gone"
    return "running"


def _kill_test_process(process_id: int) -> None:
    if _process_state(process_id) != "running":
        return
    if os.name == "nt":
        subprocess.run(
            ("taskkill", "/PID", str(process_id), "/T", "/F"),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    else:
        os.kill(process_id, 9)


@pytest.mark.parametrize(
    ("parent_delay", "expected_timeout"),
    ((0, False), (30, True)),
)
def test_subprocess_runner_terminates_descendants_without_unbounded_pipe_wait(
    tmp_path: Path, parent_delay: int, expected_timeout: bool
) -> None:
    child_pid_file = tmp_path / f"child-{parent_delay}.pid"
    parent_script = "\n".join(
        (
            "import pathlib, subprocess, sys, time",
            "child = subprocess.Popen([",
            "    sys.executable, '-c', 'import time; time.sleep(30)'",
            "])",
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))",
            f"time.sleep({parent_delay})",
        )
    )
    started = time.monotonic()
    outcome = SubprocessCommandRunner()(
        (sys.executable, "-c", parent_script),
        timeout_seconds=1,
        max_output_bytes=64,
    )
    elapsed = time.monotonic() - started
    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))

    try:
        deadline = time.monotonic() + 5
        while _process_state(child_pid) == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        state = _process_state(child_pid)
        if os.name == "nt":
            assert state == "gone"
        else:
            # A process group can terminate descendants but only their parent or
            # a subreaper/init process can wait for and remove their zombie entry.
            assert state in {"gone", "zombie"}
    finally:
        _kill_test_process(child_pid)

    assert outcome.timed_out is expected_timeout
    assert elapsed < 5


def test_injected_runner_cannot_bypass_output_limit(committed_repo: Path) -> None:
    runner = FakeDockerRunner(CommandOutcome(0, "\u00e9" * 10))
    with DockerSandbox(
        committed_repo, max_output_bytes=5, command_runner=runner
    ) as sandbox:
        result = sandbox.execute(ToolCall("bounded", "list_files", {}))

    assert result.ok
    assert result.output == "\u00e9\u00e9"
    assert len(result.output.encode("utf-8")) <= 5
    assert result.truncated


def test_subprocess_runner_kills_child_when_wait_is_interrupted(monkeypatch) -> None:
    class InterruptedProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.returncode = None
            self.killed = False
            self.wait_count = 0

        def wait(self, timeout=None):
            del timeout
            self.wait_count += 1
            if self.wait_count == 1:
                raise KeyboardInterrupt
            self.returncode = -9
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = InterruptedProcess()
    monkeypatch.setattr(
        process_module,
        "start_isolated_process",
        lambda *args, **kwargs: IsolatedProcess(
            process, process_group=None, windows_job=None  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        process_module.run_isolated_capture(
            ("unused",),
            timeout_seconds=1,
            max_stdout_bytes=8,
            max_stderr_bytes=0,
            merge_stderr=True,
        )

    assert process.killed
    assert process.wait_count == 2
    assert process.stdout.closed


def test_subprocess_runner_cleans_up_when_output_reader_cannot_start(
    monkeypatch,
) -> None:
    class PendingProcess:
        def __init__(self) -> None:
            self.stdout = io.BytesIO()
            self.returncode = None
            self.killed = False

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                raise AssertionError("process must be terminated before waiting")
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = PendingProcess()
    monkeypatch.setattr(
        process_module,
        "start_isolated_process",
        lambda *args, **kwargs: IsolatedProcess(
            process, process_group=None, windows_job=None  # type: ignore[arg-type]
        ),
    )

    def fail_to_start(_thread) -> None:
        raise RuntimeError("thread capacity exhausted")

    monkeypatch.setattr(process_module.threading.Thread, "start", fail_to_start)

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        process_module.run_isolated_capture(
            ("unused",),
            timeout_seconds=1,
            max_stdout_bytes=8,
            max_stderr_bytes=0,
            merge_stderr=True,
        )

    assert process.killed
    assert process.stdout.closed


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()
