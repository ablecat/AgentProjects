"""Exercise the Day 2 candidate-workspace flow against the real Docker sandbox."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_PARENT = PROJECT_ROOT / ".tmp"
IMAGE = "repo-agent-python:0.1"

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_RUNTIME_ERROR = 2
EXIT_CLEANUP_FAILED = 3

COMMITTED_APP = 'def greet(name):\n    return f"hello {name}"\n'
HOST_DIRTY_APP = 'def greet(name):\n    return f"HOST DIRTY {name}"\n'
HOST_ONLY_FILE = 'HOST_ONLY = "must never enter the candidate"\n'
PATCHED_APP = 'def greet(name):\n    return f"hello, {name}!"\n'
ADDED_MODULE = "def answer():\n    return 42\n"

PATCH = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return f"hello {name}"
+    return f"hello, {name}!"
diff --git a/candidate.py b/candidate.py
new file mode 100644
--- /dev/null
+++ b/candidate.py
@@ -0,0 +1,2 @@
+def answer():
+    return 42
"""

sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from repo_agent.models import CandidateArtifact, ToolCall, ToolResult  # noqa: E402
from repo_agent.sandbox import DockerSandbox  # noqa: E402


def main() -> int:
    report: dict[str, object] = {
        "status": "failed",
        "exit_code": EXIT_RUNTIME_ERROR,
        "image": IMAGE,
        "tools": [],
        "checks": [],
        "candidate": None,
        "source": {},
        "cleanup": {
            "sandbox_snapshot_removed": False,
            "containers_removed": False,
            "temporary_repository_removed": False,
        },
        "error": None,
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]
    temp_root: Path | None = None
    snapshot_path: Path | None = None
    source_repo: Path | None = None
    container_prefix = f"repo-agent-day2-smoke-{uuid.uuid4().hex[:8]}"
    runtime_failed = False
    cleanup_failed = False

    try:
        TEMP_PARENT.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix="day2-smoke-", dir=TEMP_PARENT)
        ).resolve(strict=True)
        source_repo = temp_root / "source"
        _create_dirty_source_repository(source_repo)

        source_head_before = _git_text(source_repo, "rev-parse", "HEAD")
        source_status_before = _git_bytes(
            source_repo, "status", "--short", "--untracked-files=all", "-z"
        )
        source_manifest_before = _worktree_manifest(source_repo)
        report["source"] = {
            "head": source_head_before,
            "dirty_status_sha256": _sha256(source_status_before),
            "worktree_sha256": _manifest_digest(source_manifest_before),
        }

        _check(
            checks,
            "source_fixture_is_dirty",
            b"app.py" in source_status_before
            and b"host-only.py" in source_status_before,
            "source has a modified tracked file and an untracked host-only file",
        )

        calls = (
            ToolCall(
                "day2-map",
                "repo_map",
                {"max_files": 20, "include_symbols": True},
            ),
            ToolCall(
                "day2-read-before",
                "read_file",
                {"path": "app.py", "start_line": 1, "end_line": 20},
            ),
            ToolCall("day2-apply", "apply_patch", {"patch": PATCH}),
            ToolCall(
                "day2-read-after",
                "read_file",
                {"path": "app.py", "start_line": 1, "end_line": 20},
            ),
            ToolCall("day2-diff", "git_diff", {}),
        )

        with DockerSandbox(
            source_repo,
            image=IMAGE,
            container_name_prefix=container_prefix,
            allow_mutations=True,
        ) as sandbox:
            snapshot_path = sandbox.snapshot_path
            results = [sandbox.execute(call) for call in calls]
            artifact = sandbox.candidate_artifact()
            revision = sandbox.workspace_revision
            candidate_app = (sandbox.snapshot_path / "app.py").read_text(
                encoding="utf-8"
            )
            candidate_added = (sandbox.snapshot_path / "candidate.py").read_text(
                encoding="utf-8"
            )

        report["tools"] = [_tool_record(result) for result in results]
        report["candidate"] = _artifact_record(artifact)

        map_result, initial_read, apply_result, final_read, diff_result = results
        _check(
            checks,
            "tool_sequence",
            [result.name for result in results]
            == ["repo_map", "read_file", "apply_patch", "read_file", "git_diff"],
            "repo_map -> read_file -> apply_patch -> read_file -> git_diff",
        )
        _check(
            checks,
            "all_tools_succeeded",
            all(result.ok and result.exit_code == 0 for result in results),
            "every real tool call returned ok with exit code 0",
        )
        _check(
            checks,
            "repo_map_uses_committed_head",
            source_head_before in map_result.output
            and "- app.py" in map_result.output
            and "- app.py:1 def greet" in map_result.output,
            "map contains the committed HEAD, app.py, and its Python symbol",
        )
        _check(
            checks,
            "host_dirty_files_excluded",
            "host-only.py" not in map_result.output
            and "HOST DIRTY" not in initial_read.output
            and initial_read.output == COMMITTED_APP,
            "candidate starts from committed HEAD and excludes host-only dirty content",
        )
        _check(
            checks,
            "patch_modified_and_added",
            apply_result.ok
            and "modify:app.py" in apply_result.output
            and "add:candidate.py" in apply_result.output,
            "one patch modifies app.py and adds candidate.py",
        )
        _check(
            checks,
            "candidate_reads_updated_content",
            final_read.output == PATCHED_APP
            and candidate_app == PATCHED_APP
            and candidate_added == ADDED_MODULE,
            "subsequent container read and candidate files reflect the patch",
        )
        _check(
            checks,
            "git_diff_reports_both_paths",
            "diff --git a/app.py b/app.py" in diff_result.output
            and "diff --git a/candidate.py b/candidate.py" in diff_result.output
            and 'return f"hello, {name}!"' in diff_result.output
            and "return 42" in diff_result.output,
            "candidate diff contains the modified and added files",
        )
        _check(
            checks,
            "workspace_revision_advanced_once",
            revision == 1,
            f"candidate revision is {revision}",
        )
        _check_candidate_artifact(checks, artifact, source_head_before)

        source_head_after = _git_text(source_repo, "rev-parse", "HEAD")
        source_status_after = _git_bytes(
            source_repo, "status", "--short", "--untracked-files=all", "-z"
        )
        source_manifest_after = _worktree_manifest(source_repo)
        head_unchanged = source_head_after == source_head_before
        status_unchanged = source_status_after == source_status_before
        content_unchanged = source_manifest_after == source_manifest_before
        source_report = report["source"]
        assert isinstance(source_report, dict)
        source_report.update(
            {
                "head_unchanged": head_unchanged,
                "status_unchanged": status_unchanged,
                "content_unchanged": content_unchanged,
            }
        )
        _check(
            checks,
            "source_repository_unchanged",
            head_unchanged and status_unchanged and content_unchanged,
            "source HEAD, porcelain status, file set, and file bytes are unchanged",
        )
    except BaseException as exc:
        runtime_failed = True
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if snapshot_path is not None:
            snapshot_removed = not snapshot_path.exists()
            cleanup = report["cleanup"]
            assert isinstance(cleanup, dict)
            cleanup["sandbox_snapshot_removed"] = snapshot_removed
            _check(
                checks,
                "sandbox_snapshot_removed",
                snapshot_removed,
                "DockerSandbox removed its disposable clone",
            )

        containers_removed, container_error = _containers_removed(container_prefix)
        cleanup = report["cleanup"]
        assert isinstance(cleanup, dict)
        cleanup["containers_removed"] = containers_removed
        if container_error is not None:
            cleanup["container_check_error"] = container_error
        _check(
            checks,
            "containers_removed",
            containers_removed,
            "no run-owned smoke containers remain",
        )

        temporary_repository_removed = True
        cleanup_error: str | None = None
        if temp_root is not None and temp_root.exists():
            try:
                _remove_temp_root(temp_root)
            except BaseException as exc:
                temporary_repository_removed = False
                cleanup_error = f"{type(exc).__name__}: {exc}"
        if temp_root is not None and temp_root.exists():
            temporary_repository_removed = False
        cleanup["temporary_repository_removed"] = temporary_repository_removed
        if cleanup_error is not None:
            cleanup["temporary_repository_error"] = cleanup_error
        _check(
            checks,
            "temporary_repository_removed",
            temporary_repository_removed,
            "temporary source fixture was removed",
        )
        cleanup_failed = not (
            bool(cleanup["containers_removed"])
            and bool(cleanup["temporary_repository_removed"])
            and (snapshot_path is None or bool(cleanup["sandbox_snapshot_removed"]))
        )

    checks_passed = all(bool(check["ok"]) for check in checks)
    if cleanup_failed:
        exit_code = EXIT_CLEANUP_FAILED
    elif runtime_failed:
        exit_code = EXIT_RUNTIME_ERROR
    elif not checks_passed:
        exit_code = EXIT_CHECK_FAILED
    else:
        exit_code = EXIT_OK
        report["status"] = "passed"
    report["exit_code"] = exit_code
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def _create_dirty_source_repository(repo: Path) -> None:
    repo.mkdir()
    _git_text(repo, "init", "--quiet")
    _git_text(repo, "config", "user.name", "Day 2 Smoke")
    _git_text(repo, "config", "user.email", "day2-smoke@example.invalid")
    (repo / "app.py").write_text(COMMITTED_APP, encoding="utf-8", newline="")
    _git_text(repo, "add", "app.py")
    _git_text(repo, "commit", "--quiet", "-m", "committed fixture")

    (repo / "app.py").write_text(HOST_DIRTY_APP, encoding="utf-8", newline="")
    (repo / "host-only.py").write_text(
        HOST_ONLY_FILE, encoding="utf-8", newline=""
    )


def _check(
    checks: list[dict[str, object]], name: str, ok: bool, detail: str
) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _check_candidate_artifact(
    checks: list[dict[str, object]],
    artifact: CandidateArtifact | None,
    expected_head: str,
) -> None:
    ok = (
        artifact is not None
        and artifact.base_commit == expected_head
        and artifact.revision == 1
        and artifact.changed_paths == ("app.py", "candidate.py")
        and not artifact.truncated
        and "diff --git a/app.py b/app.py" in artifact.patch
        and "diff --git a/candidate.py b/candidate.py" in artifact.patch
    )
    _check(
        checks,
        "candidate_artifact_complete",
        ok,
        "artifact records base commit, revision, paths, and a complete patch",
    )


def _tool_record(result: ToolResult) -> dict[str, object]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "ok": result.ok,
        "exit_code": result.exit_code,
        "truncated": result.truncated,
        "output": result.output,
        "error": result.error,
    }


def _artifact_record(artifact: CandidateArtifact | None) -> dict[str, object] | None:
    if artifact is None:
        return None
    return {
        "base_commit": artifact.base_commit,
        "revision": artifact.revision,
        "changed_paths": list(artifact.changed_paths),
        "patch": artifact.patch,
        "truncated": artifact.truncated,
    }


def _worktree_manifest(repo: Path) -> dict[str, tuple[int, str]]:
    manifest: dict[str, tuple[int, str]] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file():
            payload = path.read_bytes()
            manifest[relative.as_posix()] = (len(payload), _sha256(payload))
    return manifest


def _manifest_digest(manifest: dict[str, tuple[int, str]]) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(payload)


def _git_text(repo: Path, *args: str) -> str:
    return _run(("git", "-C", str(repo), *args)).stdout.decode(
        "utf-8", errors="strict"
    ).strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    return _run(("git", "-C", str(repo), *args)).stdout


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        timeout=30,
    )


def _containers_removed(prefix: str) -> tuple[bool, str | None]:
    try:
        completed = _run(
            (
                "docker",
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"name={prefix}",
            )
        )
    except BaseException as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return not bool(completed.stdout.strip()), None


def _remove_temp_root(temp_root: Path) -> None:
    resolved_parent = TEMP_PARENT.resolve(strict=True)
    resolved_target = temp_root.resolve(strict=True)
    if resolved_target.parent != resolved_parent or not resolved_target.name.startswith(
        "day2-smoke-"
    ):
        raise RuntimeError(f"refusing to remove unexpected path: {resolved_target}")
    shutil.rmtree(resolved_target, onerror=_remove_readonly)


def _remove_readonly(
    function: object, path: str, _error: tuple[type[BaseException], BaseException, object]
) -> None:
    os.chmod(path, stat.S_IWRITE)
    if not callable(function):
        raise TypeError("rmtree error handler received a non-callable operation")
    function(path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
