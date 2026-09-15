"""Manage a disposable candidate and hardened repository tool containers."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path
import re
import hashlib
import shutil
import stat
import subprocess
import tempfile
from time import sleep
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence, TypeAlias, cast
import uuid

from .models import CandidateArtifact, ToolCall, ToolResult
from .patches import ValidatedPatch
from .policy import DEFAULT_PATH_POLICY, RepositoryPathError
from .processes import run_isolated_capture
from .repository_map import build_repository_map
from .tools import ApplyPatchSpec, CommandSpec, RepoMapSpec, build_tool_command


_RUN_DIRECTORY_RE = re.compile(r"^repo-agent-run-[0-9a-f]{32}$")
_TEMP_BASE_RE = re.compile(r"^repo-agent-sandbox-[0-9a-z_]+$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_REFERENCE_RE = re.compile(
    r"^(?=.{1,255}$)"
    r"(?:(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]+)?/)*"
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*)"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
    r"(?:@sha256:[0-9a-f]{64})?$"
)
_RUN_LABEL = "io.github.ablecat.repo-agent.sandbox.run"
_SEQUENCE_LABEL = "io.github.ablecat.repo-agent.sandbox.sequence"
_CONTROL_OUTPUT_BYTES = 64 * 1024
_CONTROL_TIMEOUT_SECONDS = 30.0
_RECOVERY_DELAYS = (0.0, 0.1, 0.25, 0.5)
_INSPECT_PREFIX = "repo-agent-owned"
_MAX_ARTIFACT_BYTES = 256 * 1024
_MAX_CANDIDATE_FILES = 12


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """The verified isolation contract plus tool-runner-only hardening."""

    network: str = "none"
    read_only: bool = True
    user: str = "10001:10001"
    cpus: str = "2"
    memory: str = "4g"
    memory_swap: str = "4g"
    pids_limit: int = 256
    cap_drop: tuple[str, ...] = ("ALL",)
    security_opt: tuple[str, ...] = ("no-new-privileges",)
    log_driver: str = "none"
    init: bool = True
    pull: str = "never"
    restart: str = "no"
    tmpfs: str = "/tmp:rw,nosuid,nodev,size=256m,mode=1777"
    workspace_target: str = "/workspace"

    def docker_args(self) -> tuple[str, ...]:
        """Translate this policy to Docker ``create`` arguments."""

        args: list[str] = ["--init", "--pull", self.pull, "--restart", self.restart]
        args.extend(("--network", self.network))
        if self.read_only:
            args.append("--read-only")
        args.extend(
            (
                "--user",
                self.user,
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                "--memory-swap",
                self.memory_swap,
                "--pids-limit",
                str(self.pids_limit),
            )
        )
        for capability in self.cap_drop:
            args.extend(("--cap-drop", capability))
        for option in self.security_opt:
            args.extend(("--security-opt", option))
        args.extend(("--log-driver", self.log_driver))
        args.extend(("--tmpfs", self.tmpfs))
        return tuple(args)


DEFAULT_POLICY = SandboxPolicy()


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Bounded output from one host-side process."""

    exit_code: int | None
    output: str
    truncated: bool = False
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int


class CommandRunner(Protocol):
    """Injectable Docker command runner used by ``DockerSandbox``."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome: ...


CommandRunnerLike: TypeAlias = CommandRunner | Callable[..., CommandOutcome]


class SandboxError(RuntimeError):
    """Base class for sandbox lifecycle failures."""


class SandboxCleanupError(SandboxError):
    """Raised when an exact, run-owned resource cannot be removed."""


class SubprocessCommandRunner:
    """Drain combined process output while retaining only a bounded prefix."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> CommandOutcome:
        timeout = _positive_finite_number(timeout_seconds, "timeout_seconds")
        output_limit = _positive_integer(max_output_bytes, "max_output_bytes")
        completed = run_isolated_capture(
            argv,
            timeout_seconds=timeout,
            max_stdout_bytes=output_limit,
            max_stderr_bytes=0,
            merge_stderr=True,
        )
        return CommandOutcome(
            exit_code=completed.returncode,
            output=completed.stdout.decode("utf-8", errors="replace"),
            truncated=completed.stdout_truncated,
            timed_out=completed.timed_out,
        )


class DockerSandbox:
    """Create a clean candidate clone and execute bounded repository tools."""

    def __init__(
        self,
        repo_path: str | os.PathLike[str],
        image: str = "repo-agent-python:0.1",
        timeout_seconds: float = 30,
        max_output_bytes: int = 65536,
        *,
        policy: SandboxPolicy = DEFAULT_POLICY,
        container_name_prefix: str = "repo-agent-tool",
        command_runner: CommandRunnerLike | None = None,
        allow_mutations: bool = False,
    ) -> None:
        self.repo_path = _validated_repository(repo_path)
        self.image = _validated_image(image)
        self.timeout_seconds = _positive_finite_number(
            timeout_seconds, "timeout_seconds"
        )
        self.max_output_bytes = _positive_integer(
            max_output_bytes, "max_output_bytes"
        )
        if policy != DEFAULT_POLICY:
            raise ValueError("policy must match the verified Docker isolation contract")
        self.policy = policy
        self.container_name_prefix = _validated_container_name(
            container_name_prefix, "container_name_prefix"
        )
        if type(allow_mutations) is not bool:
            raise ValueError("allow_mutations must be a boolean")
        self.allow_mutations = allow_mutations
        self._command_runner = command_runner or SubprocessCommandRunner()
        self._run_id = uuid.uuid4().hex
        self._temp_parent: Path | None = None
        self._temp_parent_identity: _DirectoryIdentity | None = None
        self._temp_base: Path | None = None
        self._temp_base_identity: _DirectoryIdentity | None = None
        self._run_root: Path | None = None
        self._run_root_identity: _DirectoryIdentity | None = None
        self._snapshot_path: Path | None = None
        self._container_ids: set[str] = set()
        self._container_names: dict[str, int] = {}
        self._indeterminate_container_names: set[str] = set()
        self._names_by_id: dict[str, str] = {}
        self._sequence = 0
        self._entered = False
        self._closed = False
        self._workspace_revision = 0
        self._candidate_valid = True

    @property
    def snapshot_path(self) -> Path:
        """Return the isolated repository path while the context is active."""

        if self._snapshot_path is None:
            raise SandboxError("sandbox context has not been entered")
        return self._snapshot_path

    @property
    def workspace_revision(self) -> int:
        """Monotonically increase after every successful candidate mutation."""

        return self._workspace_revision

    def __enter__(self) -> DockerSandbox:
        if self._entered or self._closed:
            raise SandboxError("a DockerSandbox instance can only be entered once")
        self._entered = True
        try:
            self._create_snapshot()
        except BaseException as exc:
            try:
                self._cleanup_run_directory()
            except (OSError, SandboxError) as cleanup_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "Snapshot cleanup also failed: "
                        f"{_exception_detail(cleanup_exc)}"
                    )
            self._closed = True
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        cleanup_error: BaseException | None = None
        try:
            self.close()
        except BaseException as cleanup_exc:
            cleanup_error = cleanup_exc

        if cleanup_error is not None:
            if exc_value is None:
                raise cleanup_error
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(f"Sandbox cleanup also failed: {cleanup_error}")
        return False

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a validated tool against the disposable candidate workspace."""

        if not self._entered or self._closed or self._snapshot_path is None:
            raise SandboxError("sandbox must be used as an active context manager")
        if not self._candidate_valid:
            return _failed_result(
                call, "Candidate workspace is invalid after a failed rollback"
            )

        try:
            command = build_tool_command(call)
            if isinstance(command, RepoMapSpec):
                return self._execute_repo_map(call, command)
            if isinstance(command, ApplyPatchSpec):
                if not self.allow_mutations:
                    return _failed_result(
                        call,
                        "Candidate mutations are disabled for this sandbox run",
                    )
                return self._execute_apply_patch(call, command.patch)
            if not isinstance(command, CommandSpec):
                raise ValueError("unsupported tool specification")
            argv = _validated_tool_argv(command.argv)
            cwd = getattr(command, "cwd", self.policy.workspace_target)
            if cwd != self.policy.workspace_target:
                raise ValueError(
                    f"tool working directory must be {self.policy.workspace_target!r}"
                )
            if call.name == "read_file":
                _validate_snapshot_read_target(self.snapshot_path, argv[-1])
            allowed_exit_codes = frozenset(command.allowed_exit_codes)
            if not allowed_exit_codes or any(
                not isinstance(code, int) for code in allowed_exit_codes
            ):
                raise ValueError("tool allowed_exit_codes must contain integers")
        except (AttributeError, TypeError, ValueError) as exc:
            return _failed_result(call, f"Invalid tool call: {_exception_detail(exc)}")

        self._sequence += 1
        container_name = _validated_container_name(
            f"{self.container_name_prefix}-{self._run_id}-{self._sequence}",
            "generated container name",
            max_length=128,
        )
        create_args = self._create_args(container_name, argv, cwd)
        self._container_names[container_name] = self._sequence

        container_id: str | None = None
        try:
            create = self._run_docker(
                create_args,
                _CONTROL_OUTPUT_BYTES,
                timeout_seconds=_CONTROL_TIMEOUT_SECONDS,
            )
            if create.timed_out:
                self._indeterminate_container_names.add(container_name)
                cleanup_error = self._confirm_and_remove_container(container_name)
                timeout_error = (
                    f"Docker create timed out after "
                    f"{_CONTROL_TIMEOUT_SECONDS:g} seconds"
                )
                if cleanup_error:
                    timeout_error = f"{timeout_error}; {cleanup_error}"
                return _failed_result(
                    call,
                    timeout_error,
                    output=create.output,
                    truncated=create.truncated,
                )
            container_id = _container_id_from_output(create.output)
            if create.exit_code != 0 or container_id is None:
                cleanup_error = self._recover_and_remove_container(container_name)
                detail = (
                    f"Docker create exited with code {create.exit_code}"
                    if create.exit_code != 0
                    else "Docker create did not return a valid container ID"
                )
                if cleanup_error:
                    detail = f"{detail}; {cleanup_error}"
                return _failed_result(
                    call,
                    detail,
                    output=create.output,
                    exit_code=create.exit_code,
                    truncated=create.truncated,
                )

            self._container_ids.add(container_id)
            self._names_by_id[container_id] = container_name
            execution = self._run_docker(
                ("docker", "container", "start", "--attach", container_id),
                self.max_output_bytes,
                timeout_seconds=self.timeout_seconds,
            )
            cleanup_error = self._remove_exact_container(
                container_id, container_name=container_name
            )

            if execution.timed_out:
                execution_error = (
                    f"Tool timed out after {self.timeout_seconds:g} seconds"
                )
                if cleanup_error:
                    execution_error = f"{execution_error}; {cleanup_error}"
                return _failed_result(
                    call,
                    execution_error,
                    output=execution.output,
                    truncated=execution.truncated,
                )

            ok = execution.exit_code in allowed_exit_codes
            error: str | None = None
            if not ok:
                error = f"Tool exited with code {execution.exit_code}"
            if cleanup_error:
                ok = False
                error = f"{error + '; ' if error else ''}{cleanup_error}"
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=ok,
                output=execution.output,
                error=error,
                exit_code=execution.exit_code,
                truncated=execution.truncated,
            )
        except (OSError, SandboxError) as exc:
            if container_id is not None:
                cleanup_error = self._remove_exact_container(
                    container_id, container_name=container_name
                )
            else:
                cleanup_error = self._recover_and_remove_container(container_name)
            error = f"Sandbox execution failed: {_exception_detail(exc)}"
            if cleanup_error:
                error = f"{error}; {cleanup_error}"
            return _failed_result(call, error)

    def _execute_repo_map(self, call: ToolCall, spec: RepoMapSpec) -> ToolResult:
        try:
            mapped = build_repository_map(
                self.snapshot_path,
                path=spec.path,
                max_files=spec.max_files,
                max_output_bytes=min(self.max_output_bytes, 64 * 1024),
                include_symbols=spec.include_symbols,
            )
        except (OSError, ValueError) as exc:
            return _failed_result(
                call, f"Repository map failed: {_exception_detail(exc)}"
            )
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            output=mapped.output,
            exit_code=0,
            truncated=mapped.truncated,
        )

    def _execute_apply_patch(
        self, call: ToolCall, patch: ValidatedPatch
    ) -> ToolResult:
        """Atomically apply a validated patch to the disposable candidate clone."""

        mutation_attempted = False
        before_patch = b""
        before_paths: set[str] = set()
        try:
            self._preflight_patch_targets(patch)
            patch_bytes = patch.text.encode("utf-8")
            before_patch = _git_bytes(
                self.snapshot_path,
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            )
            before_paths = _candidate_changed_paths(self.snapshot_path)
            if len(before_paths.union(patch.paths)) > _MAX_CANDIDATE_FILES:
                raise SandboxError(
                    f"candidate is limited to {_MAX_CANDIDATE_FILES} changed files"
                )
            checked = _run_git_input(
                self.snapshot_path,
                patch_bytes,
                "apply",
                "--check",
                "--index",
                "--whitespace=error-all",
                "--",
            )
            if checked.returncode != 0:
                return _failed_result(
                    call,
                    "Patch did not apply cleanly to the candidate workspace",
                    exit_code=checked.returncode,
                )

            mutation_attempted = True
            applied = _run_git_input(
                self.snapshot_path,
                patch_bytes,
                "apply",
                "--index",
                "--whitespace=error-all",
                "--",
            )
            if applied.returncode != 0:
                try:
                    _restore_candidate_state(self.snapshot_path, before_patch)
                except (OSError, SandboxError) as restore_exc:
                    self._candidate_valid = False
                    return _failed_result(
                        call,
                        "Candidate rollback failed: "
                        f"{_exception_detail(restore_exc)}",
                    )
                return _failed_result(
                    call,
                    "Patch application failed after preflight; candidate was restored",
                    exit_code=applied.returncode,
                )
            self._verify_patch_postcondition(patch, before_paths)
            self._workspace_revision += 1
        except (OSError, RepositoryPathError, SandboxError) as exc:
            if mutation_attempted:
                try:
                    _restore_candidate_state(self.snapshot_path, before_patch)
                except (OSError, SandboxError) as restore_exc:
                    self._candidate_valid = False
                    return _failed_result(
                        call,
                        "Candidate rollback failed: "
                        f"{_exception_detail(restore_exc)}",
                    )
            return _failed_result(call, f"Patch rejected: {_exception_detail(exc)}")

        digest = hashlib.sha256(patch_bytes).hexdigest()
        operations = ", ".join(
            f"{item.operation}:{item.path}" for item in patch.files
        )
        output, truncated = _bounded_text(
            (
                f"Applied {len(patch.files)} file(s), {patch.changed_lines} changed "
                f"line(s) to disposable candidate at revision "
                f"{self._workspace_revision}. Paths: {operations}. sha256:{digest}"
            ),
            self.max_output_bytes,
        )
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            output=output,
            exit_code=0,
            truncated=truncated,
        )

    def _preflight_patch_targets(self, patch: ValidatedPatch) -> None:
        for item in patch.files:
            target = DEFAULT_PATH_POLICY.validate_disk_path(
                self.snapshot_path,
                item.path,
                access="write",
                allow_missing_leaf=item.operation == "add",
            )
            if item.operation == "add":
                if target.exists():
                    raise SandboxError(f"add target already exists: {item.path}")
                continue
            target_stat = os.lstat(target)
            if not stat.S_ISREG(target_stat.st_mode):
                raise SandboxError(f"patch target is not a regular file: {item.path}")
            if target_stat.st_nlink > 1:
                raise SandboxError(f"patch target has multiple hard links: {item.path}")

    def _verify_patch_postcondition(
        self, patch: ValidatedPatch, before_paths: set[str]
    ) -> None:
        changed_paths = _candidate_changed_paths(self.snapshot_path)
        patch_paths = set(patch.paths)
        untouched_prior_paths = before_paths - patch_paths
        if not untouched_prior_paths.issubset(changed_paths):
            raise SandboxError(
                "patch removed unrelated prior candidate changes"
            )
        if not changed_paths.issubset(before_paths.union(patch_paths)):
            raise SandboxError("patch changed unexpected candidate paths")
        if len(changed_paths) > _MAX_CANDIDATE_FILES:
            raise SandboxError(
                f"candidate exceeds {_MAX_CANDIDATE_FILES} changed files"
            )
        if _candidate_unstaged_paths(self.snapshot_path):
            raise SandboxError("patch left unstaged candidate changes")
        candidate_patch = _git_bytes(
            self.snapshot_path,
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-renames",
            "HEAD",
            "--",
            stdout_overflow_error=(
                f"candidate patch exceeds {_MAX_ARTIFACT_BYTES} bytes"
            ),
        )
        if len(candidate_patch) > _MAX_ARTIFACT_BYTES:
            raise SandboxError(
                f"candidate patch exceeds {_MAX_ARTIFACT_BYTES} bytes"
            )
        for item in patch.files:
            if item.operation == "delete":
                if (self.snapshot_path / Path(item.path)).exists():
                    raise SandboxError(f"deleted patch target still exists: {item.path}")
                continue
            target = DEFAULT_PATH_POLICY.validate_disk_path(
                self.snapshot_path, item.path, access="write"
            )
            target_stat = os.lstat(target)
            if not stat.S_ISREG(target_stat.st_mode) or target_stat.st_nlink > 1:
                raise SandboxError(f"unsafe post-patch target: {item.path}")
            stage = _git_output(
                self.snapshot_path, "ls-files", "--stage", "--", item.path
            )
            mode = stage.split(maxsplit=1)[0] if stage else ""
            if mode not in {"100644", "100755"}:
                raise SandboxError(f"unsupported Git mode after patch: {item.path}")

    def candidate_artifact(self) -> CandidateArtifact | None:
        """Capture the candidate diff while the disposable clone still exists."""

        if not self._entered or self._closed:
            raise SandboxError("sandbox must be active to capture its candidate")
        if not self._candidate_valid:
            raise SandboxError("cannot capture an invalid candidate workspace")
        changed_paths = tuple(sorted(_candidate_changed_paths(self.snapshot_path)))
        if not changed_paths:
            return None
        patch_bytes = _git_bytes(
            self.snapshot_path,
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-renames",
            "HEAD",
            "--",
        )
        truncated = len(patch_bytes) > _MAX_ARTIFACT_BYTES
        bounded = patch_bytes[:_MAX_ARTIFACT_BYTES]
        patch_text = bounded.decode("utf-8", errors="replace")
        return CandidateArtifact(
            base_commit=_git_output(
                self.snapshot_path, "rev-parse", "--verify", "HEAD^{commit}"
            ),
            revision=self._workspace_revision,
            changed_paths=changed_paths,
            patch=patch_text,
            truncated=truncated,
        )

    def close(self) -> None:
        """Remove only this run's validated container IDs and directory."""

        if self._closed:
            return
        errors: list[str] = []
        for container_id in tuple(self._container_ids):
            cleanup_error = self._remove_exact_container(
                container_id,
                container_name=self._names_by_id.get(container_id),
            )
            if cleanup_error:
                errors.append(cleanup_error)
        for container_name in tuple(self._container_names):
            cleanup_error = self._confirm_and_remove_container(container_name)
            if cleanup_error:
                errors.append(cleanup_error)
        if self._container_ids:
            errors.append(
                "run-owned containers remain: "
                + ", ".join(sorted(self._container_ids))
            )
        if self._container_names:
            errors.append(
                "run-owned container names remain: "
                + ", ".join(sorted(self._container_names))
            )

        if not self._container_ids and not self._container_names:
            try:
                self._cleanup_run_directory()
            except (OSError, SandboxError) as exc:
                errors.append(f"temporary snapshot cleanup failed: {_exception_detail(exc)}")
        if errors:
            raise SandboxCleanupError("; ".join(errors))
        self._closed = True

    def _create_snapshot(self) -> None:
        temp_parent = Path(tempfile.gettempdir()).resolve(strict=True)
        self._temp_parent = temp_parent
        self._temp_parent_identity = _trusted_directory_identity(
            temp_parent, "system temporary directory"
        )
        created_base = Path(
            tempfile.mkdtemp(prefix="repo-agent-sandbox-", dir=temp_parent)
        ).absolute()
        self._temp_base = created_base
        self._temp_base_identity = _trusted_directory_identity(
            created_base, "sandbox temporary base"
        )
        resolved_base = created_base.resolve(strict=True)
        if (
            resolved_base.parent != temp_parent
            or not _TEMP_BASE_RE.fullmatch(resolved_base.name)
        ):
            raise SandboxError(
                f"refusing to use unexpected temporary base: {resolved_base}"
            )
        self._temp_base = resolved_base

        run_root = resolved_base / f"repo-agent-run-{self._run_id}"
        _validate_run_directory(run_root, resolved_base)
        run_root.mkdir(mode=0o700)
        self._run_root = run_root
        self._run_root_identity = _trusted_directory_identity(
            run_root, "sandbox run directory"
        )
        empty_template = resolved_base / "empty-git-template"
        empty_hooks = resolved_base / "empty-git-hooks"
        empty_template.mkdir(mode=0o700)
        empty_hooks.mkdir(mode=0o700)
        snapshot = run_root / "repository"

        source_head = _git_output(
            self.repo_path, "rev-parse", "--verify", "HEAD^{commit}"
        )
        _run_git(
            "-c",
            "core.autocrlf=false",
            "-c",
            f"core.hooksPath={empty_hooks}",
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--depth",
            "1",
            "--no-tags",
            "--single-branch",
            f"--template={empty_template}",
            str(self.repo_path),
            str(snapshot),
        )
        cloned_head = _git_output(snapshot, "rev-parse", "--verify", "HEAD^{commit}")
        if cloned_head != source_head:
            raise SandboxError(
                "repository HEAD changed while its sandbox snapshot was created"
            )
        _run_git("-C", str(snapshot), "remote", "remove", "origin")
        if _git_output(snapshot, "remote"):
            raise SandboxError("temporary repository still has a configured remote")
        _run_git(
            "-c",
            f"core.hooksPath={empty_hooks}",
            "-c",
            "core.autocrlf=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "-C",
            str(snapshot),
            "checkout",
            "--quiet",
            "--force",
            "--detach",
            source_head,
        )
        _run_git("-C", str(snapshot), "clean", "--force", "-d", "-x")
        if _git_output(snapshot, "rev-list", "--all", "--count") != "1":
            raise SandboxError("temporary repository is not a single-commit snapshot")
        resolved_snapshot = snapshot.resolve(strict=True)
        if resolved_snapshot.parent != run_root:
            raise SandboxError(
                f"temporary repository escaped its run directory: {resolved_snapshot}"
            )
        self._snapshot_path = resolved_snapshot

    def _create_args(
        self, container_name: str, argv: tuple[str, ...], cwd: str
    ) -> tuple[str, ...]:
        snapshot = self.snapshot_path
        source = str(snapshot)
        if "," in source:
            raise SandboxError(
                "Docker bind source paths containing commas are not supported"
            )
        mount = (
            f"type=bind,source={source},"
            f"target={self.policy.workspace_target},readonly"
        )
        return (
            "docker",
            "container",
            "create",
            "--name",
            container_name,
            "--label",
            f"{_RUN_LABEL}={self._run_id}",
            "--label",
            f"{_SEQUENCE_LABEL}={self._sequence}",
            *self.policy.docker_args(),
            "--mount",
            mount,
            "--workdir",
            cwd,
            "--entrypoint",
            argv[0],
            self.image,
            *argv[1:],
        )

    def _run_docker(
        self,
        argv: Sequence[str],
        max_output_bytes: int,
        *,
        timeout_seconds: float = _CONTROL_TIMEOUT_SECONDS,
    ) -> CommandOutcome:
        outcome = self._command_runner(
            tuple(argv),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if not isinstance(outcome, CommandOutcome):
            raise SandboxError(
                "command_runner must return a CommandOutcome instance"
            )
        bounded_output, was_truncated = _bounded_text(
            outcome.output, max_output_bytes
        )
        return CommandOutcome(
            exit_code=outcome.exit_code,
            output=bounded_output,
            truncated=outcome.truncated or was_truncated,
            timed_out=outcome.timed_out,
        )

    def _recover_and_remove_container(self, name: str) -> str | None:
        """Resolve one exact generated name and verify ownership before removal."""

        container_id, absent, error = self._inspect_owned_container(name)
        if error is not None:
            return error
        if absent:
            return None
        assert container_id is not None
        self._container_ids.add(container_id)
        self._names_by_id[container_id] = name
        return self._remove_exact_container(container_id, container_name=name)

    def _confirm_and_remove_container(self, name: str) -> str | None:
        """Allow a timed-out create request a bounded grace period to settle."""

        for delay in _RECOVERY_DELAYS:
            if delay:
                sleep(delay)
            container_id, absent, error = self._inspect_owned_container(name)
            if error is not None:
                return error
            if not absent:
                assert container_id is not None
                self._container_ids.add(container_id)
                self._names_by_id[container_id] = name
                return self._remove_exact_container(
                    container_id, container_name=name
                )

        if name in self._indeterminate_container_names:
            return (
                "timed-out Docker create remains indeterminate for exact "
                f"container name {name!r}"
            )
        self._container_names.pop(name, None)
        return None

    def _inspect_owned_container(
        self, name: str
    ) -> tuple[str | None, bool, str | None]:
        sequence = self._container_names.get(name)
        if sequence is None:
            return None, False, f"container name is not owned by this run: {name!r}"
        inspect_format = (
            f'{_INSPECT_PREFIX} {{{{.Id}}}} '
            f'{{{{index .Config.Labels "{_RUN_LABEL}"}}}} '
            f'{{{{index .Config.Labels "{_SEQUENCE_LABEL}"}}}}'
        )
        try:
            inspected = self._run_docker(
                (
                    "docker",
                    "container",
                    "inspect",
                    "--format",
                    inspect_format,
                    name,
                ),
                _CONTROL_OUTPUT_BYTES,
            )
        except (OSError, SandboxError) as exc:
            return (
                None,
                False,
                f"exact container recovery failed: {_exception_detail(exc)}",
            )
        if inspected.timed_out:
            return None, False, "exact container recovery timed out"
        if inspected.exit_code != 0:
            if "no such container" in inspected.output.casefold():
                return None, True, None
            return (
                None,
                False,
                f"exact container recovery exited with code {inspected.exit_code}",
            )

        record = _owned_container_record(inspected.output)
        if record is None:
            return None, False, "exact container recovery returned invalid metadata"
        container_id, run_label, sequence_label = record
        if run_label != self._run_id or sequence_label != str(sequence):
            return (
                None,
                False,
                "refusing to remove container with mismatched ownership labels",
            )
        return container_id, False, None

    def _remove_exact_container(
        self, container_id: str, *, container_name: str | None = None
    ) -> str | None:
        if not _CONTAINER_ID_RE.fullmatch(container_id):
            return f"refusing to remove invalid container ID: {container_id!r}"
        try:
            removed = self._run_docker(
                ("docker", "container", "rm", "--force", container_id),
                _CONTROL_OUTPUT_BYTES,
            )
        except (OSError, SandboxError) as exc:
            return f"container {container_id} cleanup failed: {_exception_detail(exc)}"
        if removed.timed_out:
            return f"container {container_id} cleanup timed out"
        if removed.exit_code != 0:
            if "no such container" in removed.output.casefold():
                self._forget_container(container_id, container_name)
                return None
            return (
                f"container {container_id} cleanup exited with code "
                f"{removed.exit_code}"
            )
        self._forget_container(container_id, container_name)
        return None

    def _forget_container(
        self, container_id: str, container_name: str | None
    ) -> None:
        self._container_ids.discard(container_id)
        known_name = self._names_by_id.pop(container_id, None)
        if container_name is not None:
            self._container_names.pop(container_name, None)
            self._indeterminate_container_names.discard(container_name)
        if known_name is not None:
            self._container_names.pop(known_name, None)
            self._indeterminate_container_names.discard(known_name)

    def _cleanup_run_directory(self) -> None:
        if self._temp_base is None:
            self._clear_temporary_state()
            return

        if self._temp_parent is None or self._temp_parent_identity is None:
            raise SandboxError("temporary parent identity was not recorded")
        if self._temp_base_identity is None:
            raise SandboxError("temporary base identity was not recorded")
        _assert_trusted_directory(
            self._temp_parent,
            self._temp_parent_identity,
            "system temporary directory",
        )
        if (
            self._temp_base.parent != self._temp_parent
            or not _TEMP_BASE_RE.fullmatch(self._temp_base.name)
        ):
            raise SandboxError(
                f"refusing to remove unexpected temporary base: {self._temp_base}"
            )
        _assert_trusted_directory(
            self._temp_base,
            self._temp_base_identity,
            "sandbox temporary base",
        )

        if self._run_root is not None:
            _validate_run_directory(self._run_root, self._temp_base)
            if self._run_root_identity is None:
                raise SandboxError("sandbox run directory identity was not recorded")
            _assert_trusted_directory(
                self._run_root,
                self._run_root_identity,
                "sandbox run directory",
            )

        shutil.rmtree(self._temp_base, onerror=_remove_readonly)
        self._clear_temporary_state()

    def _clear_temporary_state(self) -> None:
        self._temp_parent = None
        self._temp_parent_identity = None
        self._temp_base = None
        self._temp_base_identity = None
        self._run_root = None
        self._run_root_identity = None
        self._snapshot_path = None


def _validated_repository(repo_path: str | os.PathLike[str]) -> Path:
    try:
        candidate = Path(repo_path).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"repo_path is not an existing directory: {repo_path!r}") from exc
    if not candidate.is_dir():
        raise ValueError(f"repo_path is not a directory: {candidate}")
    try:
        inside = _git_output(candidate, "rev-parse", "--is-inside-work-tree")
        top_level = Path(
            _git_output(candidate, "rev-parse", "--show-toplevel")
        ).resolve(strict=True)
        _git_output(top_level, "rev-parse", "--verify", "HEAD^{commit}")
    except (OSError, SandboxError) as exc:
        raise ValueError(f"repo_path is not a non-empty Git worktree: {candidate}") from exc
    if inside != "true" or not top_level.is_dir():
        raise ValueError(f"repo_path is not a Git worktree: {candidate}")
    return top_level


def _validated_image(image: str) -> str:
    if not isinstance(image, str) or not _IMAGE_REFERENCE_RE.fullmatch(image):
        raise ValueError(f"invalid Docker image reference: {image!r}")
    return image


def _validated_container_name(
    name: str, field: str, *, max_length: int = 64
) -> str:
    if (
        not isinstance(name, str)
        or len(name) > max_length
        or not _CONTAINER_NAME_RE.fullmatch(name)
    ):
        raise ValueError(f"invalid Docker {field}: {name!r}")
    return name


def _positive_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return converted


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validated_tool_argv(argv: object) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise ValueError("tool argv must be a sequence of argument strings")
    try:
        converted: tuple[object, ...] = tuple(cast(Iterable[object], argv))
    except TypeError as exc:
        raise ValueError("tool argv must be iterable") from exc
    if not converted or any(
        not isinstance(argument, str) or "\x00" in argument
        for argument in converted
    ):
        raise ValueError("tool argv must contain non-NUL argument strings")
    return cast(tuple[str, ...], converted)


def _validate_snapshot_read_target(snapshot: Path, relative_path: str) -> None:
    """Reject links and non-files without following them outside the snapshot."""

    current = snapshot
    parts = relative_path.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            path_stat = os.lstat(current)
        except OSError as exc:
            raise ValueError(
                f"read_file path cannot be inspected: {relative_path!r}"
            ) from exc
        if _is_reparse_point(current, path_stat):
            raise ValueError(
                "read_file path must not contain symbolic links or reparse points"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError("read_file parent path must be a directory")

    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError("read_file path must name a regular file")


def _container_id_from_output(output: str) -> str | None:
    matches = re.findall(r"(?m)^[ \t]*([0-9a-f]{64})[ \t]*$", output)
    if len(matches) != 1:
        return None
    return matches[0]


def _owned_container_record(output: str) -> tuple[str, str, str] | None:
    pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(_INSPECT_PREFIX)}[ \t]+"
        r"([0-9a-f]{64})[ \t]+([0-9a-f]{32})[ \t]+([0-9]+)[ \t]*$"
    )
    matches = pattern.findall(output)
    if len(matches) != 1:
        return None
    return matches[0]


def _bounded_text(output: str, max_output_bytes: int) -> tuple[str, bool]:
    if not isinstance(output, str):
        raise SandboxError("command_runner output must be text")
    encoded = output.encode("utf-8")
    if len(encoded) <= max_output_bytes:
        return output, False
    return encoded[:max_output_bytes].decode("utf-8", errors="ignore"), True


def _failed_result(
    call: ToolCall,
    error: str,
    *,
    output: str = "",
    exit_code: int | None = None,
    truncated: bool = False,
) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        name=call.name,
        ok=False,
        output=output,
        error=error,
        exit_code=exit_code,
        truncated=truncated,
    )


def _run_git(*args: str, cwd: Path | None = None) -> str:
    try:
        completed = run_isolated_capture(
            ("git", *args),
            cwd=cwd,
            env=_sanitized_git_environment(),
            timeout_seconds=30,
            max_stdout_bytes=_CONTROL_OUTPUT_BYTES,
            max_stderr_bytes=_CONTROL_OUTPUT_BYTES,
        )
    except OSError as exc:
        raise SandboxError(f"Git command failed to start: {_exception_detail(exc)}") from exc
    if completed.timed_out:
        raise SandboxError("Git command timed out after 30 seconds")
    if completed.stdout_truncated or completed.stderr_truncated:
        raise SandboxError("Git command output exceeded its safe limit")
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no diagnostic output"
        raise SandboxError(
            f"git {' '.join(args)} exited with code {completed.returncode}: {detail}"
        )
    return stdout


def _run_git_input(
    repo: Path, input_bytes: bytes, *args: str
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = run_isolated_capture(
            ("git", "-C", str(repo), *args),
            input_bytes=input_bytes,
            env=_sanitized_git_environment(),
            timeout_seconds=30,
            max_stdout_bytes=_CONTROL_OUTPUT_BYTES,
            max_stderr_bytes=_CONTROL_OUTPUT_BYTES,
        )
    except OSError as exc:
        raise SandboxError(
            f"Git patch command failed to complete: {_exception_detail(exc)}"
        ) from exc
    if completed.timed_out:
        raise SandboxError("Git patch command timed out after 30 seconds")
    return subprocess.CompletedProcess(
        ("git", "-C", str(repo), *args),
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def _git_bytes(
    repo: Path,
    *args: str,
    stdout_overflow_error: str | None = None,
) -> bytes:
    try:
        completed = run_isolated_capture(
            ("git", "-C", str(repo), *args),
            env=_sanitized_git_environment(),
            timeout_seconds=30,
            max_stdout_bytes=_MAX_ARTIFACT_BYTES + 1,
            max_stderr_bytes=_CONTROL_OUTPUT_BYTES,
        )
    except OSError as exc:
        raise SandboxError(
            f"Git command failed to complete: {_exception_detail(exc)}"
        ) from exc
    if completed.timed_out:
        raise SandboxError("Git command timed out after 30 seconds")
    if completed.stdout_truncated:
        raise SandboxError(
            stdout_overflow_error or "Git command stdout exceeded its safe limit"
        )
    if completed.stderr_truncated:
        raise SandboxError("Git command stderr exceeded its safe limit")
    if completed.returncode != 0:
        raise SandboxError(f"Git command exited with code {completed.returncode}")
    return completed.stdout


def _candidate_changed_paths(repo: Path) -> set[str]:
    raw = _git_bytes(
        repo,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--diff-filter=ACDMRTUXB",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
    )
    return {
        record.decode("utf-8", errors="strict")
        for record in raw.split(b"\x00")
        if record
    }


def _candidate_unstaged_paths(repo: Path) -> set[str]:
    raw = _git_bytes(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--diff-filter=ACDMRTUXB",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "--",
    )
    return {
        record.decode("utf-8", errors="strict")
        for record in raw.split(b"\x00")
        if record
    }


def _restore_candidate_state(repo: Path, prior_patch: bytes) -> None:
    _run_git("-C", str(repo), "reset", "--quiet", "--hard", "HEAD")
    _run_git("-C", str(repo), "clean", "--quiet", "--force", "-d", "-x")
    if prior_patch:
        restored = _run_git_input(
            repo,
            prior_patch,
            "apply",
            "--index",
            "--binary",
            "--whitespace=nowarn",
            "--",
        )
        if restored.returncode != 0:
            raise SandboxError("could not restore the prior candidate state")


def _git_output(repo: Path, *args: str) -> str:
    return _run_git("-C", str(repo), *args)


def _sanitized_git_environment() -> dict[str, str]:
    blocked_credentials = {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "REPO_AGENT_API_KEY",
        "REPO_AGENT_BEARER_TOKEN",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
        and key.upper() not in blocked_credentials
    }
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _validate_run_directory(run_root: Path, expected_base: Path) -> None:
    if run_root.parent != expected_base or not _RUN_DIRECTORY_RE.fullmatch(
        run_root.name
    ):
        raise SandboxError(f"refusing to operate on unexpected path: {run_root}")


def _trusted_directory_identity(path: Path, description: str) -> _DirectoryIdentity:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise SandboxError(
            f"cannot inspect {description} {path}: {_exception_detail(exc)}"
        ) from exc
    if not stat.S_ISDIR(path_stat.st_mode) or _is_reparse_point(path, path_stat):
        raise SandboxError(f"refusing untrusted {description}: {path}")
    return _DirectoryIdentity(path_stat.st_dev, path_stat.st_ino)


def _assert_trusted_directory(
    path: Path, expected: _DirectoryIdentity, description: str
) -> None:
    actual = _trusted_directory_identity(path, description)
    if actual != expected:
        raise SandboxError(f"{description} identity changed: {path}")


def _is_reparse_point(path: Path, path_stat: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(path_stat, "st_file_attributes", 0) & reparse_flag):
        return True
    if stat.S_ISLNK(path_stat.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _remove_readonly(function, path: str, exc_info) -> None:
    del exc_info
    path_stat = os.lstat(path)
    if _is_reparse_point(Path(path), path_stat):
        raise SandboxError(f"refusing to chmod reparse point during cleanup: {path}")
    mode = stat.S_IWRITE | stat.S_IREAD
    if os.chmod in os.supports_follow_symlinks:
        os.chmod(path, mode, follow_symlinks=False)
    else:
        # Windows only toggles the read-only attribute and rejects the
        # follow_symlinks keyword, including when it is explicitly false.
        os.chmod(path, mode)
    function(path)


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


__all__ = [
    "CommandOutcome",
    "DEFAULT_POLICY",
    "DockerSandbox",
    "SandboxCleanupError",
    "SandboxError",
    "SandboxPolicy",
    "SubprocessCommandRunner",
]
