from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import repo_agent.repository_map as repository_map
from repo_agent.policy import RepositoryPathError
from repo_agent.processes import CapturedProcess


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
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


@pytest.fixture
def mapped_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Map Edge Test")
    _git(repository, "config", "user.email", "map-edge@example.invalid")
    (repository / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n", encoding="utf-8"
    )
    (repository / "src" / "app.py").write_text(
        "async def load():\n"
        "    return 1\n\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        return 2\n"
        "    async def stop(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    (repository / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (repository / ".env").write_text("SECRET=hidden\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    (repository / "notes.txt").write_text("visible untracked\n", encoding="utf-8")
    return repository.resolve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_files": True},
        {"max_files": 0},
        {"max_files": repository_map.MAX_MAP_FILES + 1},
        {"max_output_bytes": True},
        {"max_output_bytes": 0},
        {"max_output_bytes": repository_map.MAX_MAP_BYTES + 1},
        {"include_symbols": 1},
    ],
)
def test_repository_map_rejects_invalid_bounds(
    mapped_repository: Path, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        repository_map.build_repository_map(mapped_repository, **kwargs)


def test_repository_map_describes_visible_files_symbols_and_prefix(
    mapped_repository: Path,
) -> None:
    result = repository_map.build_repository_map(mapped_repository)

    assert result.truncated is False
    assert "LANGUAGES Python=2" in result.output
    assert "MANIFESTS pyproject.toml" in result.output
    assert "- notes.txt" in result.output
    assert ".env" not in result.output
    assert "src/app.py:1 async def load" in result.output
    assert "src/app.py:4 class Worker" in result.output
    assert "src/app.py:5 method Worker.run" in result.output
    assert "src/app.py:7 method Worker.stop" in result.output
    assert "broken.py:" not in result.output

    prefixed = repository_map.build_repository_map(
        mapped_repository,
        path="src",
        include_symbols=False,
    )
    assert "src/app.py" in prefixed.output
    assert "pyproject.toml" not in prefixed.output
    assert "PYTHON SYMBOLS" not in prefixed.output

    with pytest.raises(RepositoryPathError):
        repository_map.build_repository_map(mapped_repository, path="../outside")


def test_repository_map_marks_file_and_symbol_limits(
    mapped_repository: Path, monkeypatch
) -> None:
    files = repository_map.build_repository_map(mapped_repository, max_files=1)
    assert files.files_included == 1
    assert files.truncated is True

    monkeypatch.setattr(repository_map, "MAX_SYMBOLS", 1)
    symbols = repository_map.build_repository_map(mapped_repository)
    assert symbols.truncated is True
    assert symbols.output.count(" method ") + symbols.output.count(" def ") == 1


def test_repository_map_handles_oversized_and_disappearing_symbol_files(
    mapped_repository: Path, monkeypatch
) -> None:
    monkeypatch.setattr(repository_map, "MAX_SYMBOL_FILE_BYTES", 1)
    oversized = repository_map.build_repository_map(mapped_repository)
    assert oversized.truncated is True

    monkeypatch.setattr(repository_map, "MAX_SYMBOL_FILE_BYTES", 1024)
    monkeypatch.setattr(repository_map, "MAX_SYMBOL_FILES", 0)
    file_limited = repository_map.build_repository_map(mapped_repository)
    assert file_limited.truncated is True


def test_repository_map_filters_unsafe_missing_and_non_regular_entries(
    mapped_repository: Path, monkeypatch
) -> None:
    (mapped_repository / "folder").mkdir()
    monkeypatch.setattr(
        repository_map,
        "_listed_paths",
        lambda _repo: (
            ["../escape", ".env", "missing.py", "folder", "src/app.py"],
            True,
        ),
    )
    monkeypatch.setattr(repository_map, "_git_text", lambda *_args: "deadbeef")

    result = repository_map.build_repository_map(mapped_repository)

    assert result.truncated is True
    assert result.files_included == 1
    assert "src/app.py" in result.output
    assert "escape" not in result.output
    assert ".env" not in result.output
    assert "missing.py" not in result.output
    assert "folder" not in result.output


def test_python_symbol_parser_contains_invalid_and_large_sources(
    tmp_path: Path, monkeypatch
) -> None:
    invalid = tmp_path / "invalid.py"
    invalid.write_text("def broken(:\n", encoding="utf-8")
    binary = tmp_path / "binary.py"
    binary.write_bytes(b"\xff\xfe")
    large = tmp_path / "large.py"
    large.write_text("def value():\n    return 1\n", encoding="utf-8")

    assert repository_map._python_symbols(invalid) == ()
    assert repository_map._python_symbols(binary) == ()
    monkeypatch.setattr(repository_map, "MAX_SYMBOL_FILE_BYTES", 1)
    assert repository_map._python_symbols(large) == ()


def test_git_path_listing_is_bounded_and_decodes_invalid_names(
    tmp_path: Path, monkeypatch
) -> None:
    process = CapturedProcess(
        0, b"abc\xff\x00second.py\x00", b"", False, False, False
    )
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: process,
    )

    paths, truncated = repository_map._listed_paths(tmp_path)

    assert paths == ["abc\ufffd", "second.py"]
    assert truncated is False

    bounded = CapturedProcess(
        1, b"complete.py\x00partial.py", b"", True, False, False
    )
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: bounded,
    )
    paths, truncated = repository_map._listed_paths(tmp_path)
    assert paths == ["complete.py"]
    assert truncated is True

    complete_boundary = CapturedProcess(
        1, b"complete.py\x00", b"", True, False, False
    )
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: complete_boundary,
    )
    paths, truncated = repository_map._listed_paths(tmp_path)
    assert paths == ["complete.py"]
    assert truncated is True

    no_complete_record = CapturedProcess(1, b"partial.py", b"", True, False, False)
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: no_complete_record,
    )
    paths, truncated = repository_map._listed_paths(tmp_path)
    assert paths == []
    assert truncated is True


def test_truncated_git_path_cannot_expose_an_ignored_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitignore").write_text("notes.py\n", encoding="utf-8")
    (repository / "notes.py").write_text(
        "def private_notes():\n    return 'ignored'\n", encoding="utf-8"
    )
    (repository / "notes.py-long").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "notes.py-long")

    monkeypatch.setattr(
        repository_map, "MAX_SCAN_BYTES", len(b".gitignore\x00notes.py")
    )

    paths, truncated = repository_map._listed_paths(repository)
    result = repository_map.build_repository_map(repository)

    assert paths == [".gitignore"]
    assert truncated is True
    assert result.truncated is True
    assert "- notes.py\n" not in result.output
    assert "private_notes" not in result.output


def test_git_path_listing_maps_timeout_exit_and_pipe_failures(
    tmp_path: Path, monkeypatch
) -> None:
    timeout = CapturedProcess(-9, b"", b"", False, False, True)
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: timeout,
    )
    with pytest.raises(OSError, match="timed out"):
        repository_map._listed_paths(tmp_path)

    failed = CapturedProcess(2, b"", b"", False, False, False)
    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: failed,
    )
    with pytest.raises(OSError, match="exited with code 2"):
        repository_map._listed_paths(tmp_path)

    def unreadable(*_args, **_kwargs):
        raise OSError("failed to process stdout")

    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        unreadable,
    )
    with pytest.raises(OSError, match="failed to process stdout"):
        repository_map._listed_paths(tmp_path)


def test_git_text_and_environment_are_failure_tolerant(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("GIT_DIR", "private")
    monkeypatch.setenv("VISIBLE_SETTING", "kept")
    environment = repository_map._git_environment()
    assert "GIT_DIR" not in environment
    assert environment["VISIBLE_SETTING"] == "kept"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"

    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: CapturedProcess(
            1, b"secret", b"", False, False, False
        ),
    )
    assert repository_map._git_text(tmp_path, "rev-parse", "HEAD") == ""

    monkeypatch.setattr(
        repository_map,
        "run_isolated_capture",
        lambda *_args, **_kwargs: CapturedProcess(
            0, b"abc\xff\n", b"", False, False, False
        ),
    )
    assert repository_map._git_text(tmp_path, "rev-parse", "HEAD") == "abc\ufffd"


def test_bounded_lines_handles_marker_fit_and_tiny_limits() -> None:
    complete, truncated = repository_map._bounded_lines(["a", "b"], 4)
    assert complete == "a\nb\n"
    assert truncated is False

    marked, truncated = repository_map._bounded_lines(["short", "x" * 30], 25)
    assert marked.endswith("[repo map truncated]\n")
    assert len(marked.encode("utf-8")) <= 25
    assert truncated is True

    empty, truncated = repository_map._bounded_lines(["too long"], 1)
    assert empty == ""
    assert truncated is True
