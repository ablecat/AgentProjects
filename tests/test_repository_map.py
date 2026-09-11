from __future__ import annotations

from pathlib import Path
import subprocess

from repo_agent.repository_map import build_repository_map


def test_tiny_output_limit_terminates_and_remains_bounded(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository", "main.py")

    result = build_repository_map(repository, max_output_bytes=50)

    assert result.truncated
    assert len(result.output.encode("utf-8")) <= 50


def test_ambient_git_directory_cannot_redirect_repository_map(
    tmp_path: Path, monkeypatch
) -> None:
    repository = _repository(tmp_path / "repository", "main.py")
    unrelated = _repository(tmp_path / "unrelated", "other.py")
    expected_head = _git(repository, "rev-parse", "HEAD")
    unrelated_head = _git(unrelated, "rev-parse", "HEAD")
    assert expected_head != unrelated_head
    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))

    result = build_repository_map(repository)

    assert f"HEAD {expected_head}" in result.output
    assert f"HEAD {unrelated_head}" not in result.output
    assert "- main.py" in result.output
    assert "other.py" not in result.output


def _repository(path: Path, filename: str) -> Path:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Map Test")
    _git(path, "config", "user.email", "map@example.invalid")
    (path / filename).write_text("def entry():\n    return 1\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(path, "commit", "--quiet", "-m", filename)
    return path.resolve()


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        shell=False,
        timeout=30,
    )
    return completed.stdout.strip()
