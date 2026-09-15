"""Bounded, deterministic repository map generation."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import stat

from .policy import DEFAULT_PATH_POLICY, RepositoryPathError
from .processes import run_isolated_capture


MAX_MAP_FILES = 500
MAX_MAP_BYTES = 64 * 1024
MAX_SCAN_BYTES = 2 * 1024 * 1024
MAX_SYMBOL_FILE_BYTES = 256 * 1024
MAX_SYMBOLS = 500
MAX_SYMBOL_FILES = 100
MAX_SYMBOL_TOTAL_BYTES = 4 * 1024 * 1024

_MANIFEST_NAMES = frozenset(
    {
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "package.json",
        "pom.xml",
        "pyproject.toml",
        "requirements.txt",
        "settings.gradle",
        "settings.gradle.kts",
    }
)
_LANGUAGES = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".h": "C/C++ header",
    ".hpp": "C++ header",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript JSX",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript JSX",
}


@dataclass(frozen=True, slots=True)
class RepoMapResult:
    output: str
    truncated: bool
    files_included: int


def build_repository_map(
    repository: Path,
    *,
    path: str | None = None,
    max_files: int = MAX_MAP_FILES,
    max_output_bytes: int = MAX_MAP_BYTES,
    include_symbols: bool = True,
) -> RepoMapResult:
    """Describe tracked and visible untracked files without executing repository code."""

    if type(max_files) is not int or not 1 <= max_files <= MAX_MAP_FILES:
        raise ValueError(f"max_files must be an integer from 1 to {MAX_MAP_FILES}")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= MAX_MAP_BYTES:
        raise ValueError(f"max_output_bytes must be an integer from 1 to {MAX_MAP_BYTES}")
    if type(include_symbols) is not bool:
        raise ValueError("include_symbols must be a boolean")
    prefix = None
    if path is not None:
        prefix = DEFAULT_PATH_POLICY.validate(path, access="discover")

    all_paths, scan_truncated = _listed_paths(repository)
    visible: list[str] = []
    for candidate in sorted(set(all_paths), key=lambda item: (item.casefold(), item)):
        try:
            normalized = DEFAULT_PATH_POLICY.validate(candidate, access="discover")
        except RepositoryPathError:
            continue
        if prefix is not None and normalized != prefix and not normalized.startswith(f"{prefix}/"):
            continue
        try:
            disk_path = DEFAULT_PATH_POLICY.validate_disk_path(
                repository, normalized, access="discover"
            )
            file_stat = os.lstat(disk_path)
        except (OSError, RepositoryPathError):
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        visible.append(normalized)

    files_truncated = len(visible) > max_files
    files = visible[:max_files]
    head = _git_text(repository, "rev-parse", "--verify", "HEAD^{commit}")
    languages = Counter(
        language
        for item in files
        if (language := _LANGUAGES.get(Path(item).suffix.casefold())) is not None
    )
    manifests = [item for item in files if Path(item).name.casefold() in _MANIFEST_NAMES]

    lines = [
        f"HEAD {head or 'unknown'}",
        f"FILES {len(files)} shown / {len(visible)} visible",
        "LANGUAGES "
        + (", ".join(f"{name}={count}" for name, count in sorted(languages.items())) or "none detected"),
        "MANIFESTS " + (", ".join(manifests) or "none detected"),
        "TREE",
    ]
    lines.extend(f"- {item}" for item in files)

    symbol_lines: list[str] = []
    symbols_truncated = False
    if include_symbols:
        symbol_files = 0
        symbol_bytes = 0
        for item in files:
            if not item.casefold().endswith(".py"):
                continue
            symbol_path = repository / Path(item)
            try:
                file_bytes = symbol_path.stat().st_size
            except OSError:
                continue
            if file_bytes > MAX_SYMBOL_FILE_BYTES:
                symbols_truncated = True
                continue
            if (
                symbol_files >= MAX_SYMBOL_FILES
                or symbol_bytes + file_bytes > MAX_SYMBOL_TOTAL_BYTES
            ):
                symbols_truncated = True
                break
            symbol_files += 1
            symbol_bytes += file_bytes
            for symbol in _python_symbols(symbol_path):
                symbol_lines.append(f"- {item}:{symbol}")
                if len(symbol_lines) >= MAX_SYMBOLS:
                    symbols_truncated = True
                    break
            if len(symbol_lines) >= MAX_SYMBOLS:
                break
        lines.append("PYTHON SYMBOLS")
        lines.extend(symbol_lines or ["- none detected"])

    truncated = scan_truncated or files_truncated or symbols_truncated
    output, byte_truncated = _bounded_lines(lines, max_output_bytes)
    return RepoMapResult(output, truncated or byte_truncated, len(files))


def _listed_paths(repository: Path) -> tuple[list[str], bool]:
    argv = (
        "git",
        "-C",
        str(repository),
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    completed = run_isolated_capture(
        argv,
        timeout_seconds=10,
        max_stdout_bytes=MAX_SCAN_BYTES,
        max_stderr_bytes=0,
        env=_git_environment(),
        terminate_on_stdout_limit=True,
    )
    if completed.timed_out:
        raise OSError("git ls-files timed out")
    if completed.returncode != 0 and not completed.stdout_truncated:
        raise OSError(f"git ls-files exited with code {completed.returncode}")
    captured = completed.stdout
    if completed.stdout_truncated and not captured.endswith(b"\x00"):
        captured = captured.rsplit(b"\x00", 1)[0] if b"\x00" in captured else b""
    records = captured.split(b"\x00")
    paths = [record.decode("utf-8", errors="replace") for record in records if record]
    return paths, completed.stdout_truncated


def _python_symbols(path: Path) -> tuple[str, ...]:
    try:
        if path.stat().st_size > MAX_SYMBOL_FILE_BYTES:
            return ()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError, ValueError, RecursionError, MemoryError):
        return ()
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            symbols.append(f"{node.lineno} {prefix} {node.name}")
        elif isinstance(node, ast.ClassDef):
            symbols.append(f"{node.lineno} class {node.name}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"{child.lineno} method {node.name}.{child.name}")
    return tuple(symbols)


def _git_text(repository: Path, *args: str) -> str:
    try:
        completed = run_isolated_capture(
            ("git", "-C", str(repository), *args),
            timeout_seconds=10,
            max_stdout_bytes=128,
            max_stderr_bytes=0,
            env=_git_environment(),
        )
    except OSError:
        return ""
    if (
        completed.timed_out
        or completed.returncode != 0
        or completed.stdout_truncated
    ):
        return ""
    return completed.stdout.decode("ascii", errors="replace").strip()


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _bounded_lines(lines: list[str], limit: int) -> tuple[str, bool]:
    marker = "[repo map truncated]\n"
    accepted: list[str] = []
    accepted_bytes = 0
    for line in lines:
        candidate = f"{line}\n"
        candidate_bytes = len(candidate.encode("utf-8"))
        if accepted_bytes + candidate_bytes > limit:
            marker_bytes = len(marker.encode("utf-8"))
            while accepted and accepted_bytes + marker_bytes > limit:
                accepted_bytes -= len(accepted.pop().encode("utf-8"))
            if accepted_bytes + marker_bytes <= limit:
                accepted.append(marker)
            return "".join(accepted), True
        accepted.append(candidate)
        accepted_bytes += candidate_bytes
    return "".join(accepted), False


__all__ = [
    "MAX_MAP_BYTES",
    "MAX_MAP_FILES",
    "RepoMapResult",
    "build_repository_map",
]
