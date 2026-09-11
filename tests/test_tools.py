from __future__ import annotations

from collections.abc import Callable

import pytest

from repo_agent.models import ToolCall
from repo_agent.tools import (
    MAX_GLOB_LENGTH,
    MAX_LINE_NUMBER,
    MAX_PATTERN_LENGTH,
    MAX_READ_LINES,
    CommandSpec,
    TOOL_DEFINITIONS,
    ToolValidationError,
    build_tool_command,
)


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call-1", name=name, arguments=arguments)


def test_tool_definitions_are_provider_schemas() -> None:
    assert {definition.name for definition in TOOL_DEFINITIONS} == {
        "git_status",
        "git_diff",
        "list_files",
        "search",
        "read_file",
        "repo_map",
        "apply_patch",
    }
    for definition in TOOL_DEFINITIONS:
        provider_tool = definition.to_provider_dict()
        assert provider_tool["type"] == "function"
        assert provider_tool["name"] == definition.name
        assert provider_tool["strict"] is False
        assert provider_tool["parameters"]["type"] == "object"
        assert provider_tool["parameters"]["additionalProperties"] is False


def test_provider_schema_results_do_not_mutate_global_definitions() -> None:
    definition = next(item for item in TOOL_DEFINITIONS if item.name == "search")
    provider_tool = definition.to_provider_dict()

    provider_tool["parameters"]["properties"].clear()

    fresh = definition.to_provider_dict()
    assert "pattern" in fresh["parameters"]["properties"]


def test_read_file_schema_does_not_claim_a_static_end_line_default() -> None:
    definition = next(item for item in TOOL_DEFINITIONS if item.name == "read_file")
    end_line = definition.to_provider_dict()["parameters"]["properties"]["end_line"]

    assert "default" not in end_line
    assert f"at most {MAX_READ_LINES} lines from start_line" in end_line["description"]


def test_git_status_is_safe_for_docker_bind_mount() -> None:
    spec = build_tool_command(call("git_status"))

    assert spec == CommandSpec(
        argv=(
            "git",
            "-c",
            "safe.directory=/workspace",
            "status",
            "--short",
            "--branch",
            "--untracked-files=all",
        )
    )
    assert spec.cwd == "/workspace"


def test_git_diff_supports_staged_literal_path() -> None:
    spec = build_tool_command(
        call("git_diff", staged=True, path="src/repo_agent/tools.py")
    )

    assert spec.argv[:5] == (
        "git",
        "-c",
        "safe.directory=/workspace",
        "--literal-pathspecs",
        "diff",
    )
    assert "--cached" in spec.argv
    assert spec.argv[-2:] == ("--", "src/repo_agent/tools.py")


def test_unscoped_git_diff_excludes_secret_paths() -> None:
    spec = build_tool_command(call("git_diff"))

    assert spec.argv[:4] == (
        "git",
        "-c",
        "safe.directory=/workspace",
        "diff",
    )
    assert "." in spec.argv
    assert any(".env" in argument and "exclude" in argument for argument in spec.argv)
    assert any("*.pem" in argument and "exclude" in argument for argument in spec.argv)


def test_list_files_supports_glob_and_global_secret_excludes() -> None:
    spec = build_tool_command(call("list_files", glob="src/**/*.py"))

    assert spec.argv[:5] == (
        "rg",
        "--files",
        "--hidden",
        "--glob",
        "src/**/*.py",
    )
    assert ("--iglob", "!**/.env") in tuple(zip(spec.argv, spec.argv[1:]))
    assert spec.allowed_exit_codes == frozenset({0})


def test_search_uses_end_of_options_and_accepts_no_matches() -> None:
    spec = build_tool_command(
        call("search", pattern="-TODO", glob="*.py", fixed_strings=True)
    )

    assert spec.argv[:2] == ("rg", "--line-number")
    assert "--hidden" in spec.argv
    assert "--fixed-strings" in spec.argv
    assert ("--glob", "*.py") in tuple(zip(spec.argv, spec.argv[1:]))
    assert spec.argv[-3:] == ("--", "-TODO", ".")
    assert spec.allowed_exit_codes == frozenset({0, 1})


@pytest.mark.parametrize(
    "secret_glob",
    [
        "!**/auth.json",
        "!**/settings.xml",
        "!**/.dockerconfigjson",
        "!**/credential",
        "!**/credential.*",
    ],
)
def test_search_globally_excludes_every_central_secret_name(
    secret_glob: str,
) -> None:
    spec = build_tool_command(call("search", pattern="TOPSECRET"))

    assert ("--iglob", secret_glob) in tuple(zip(spec.argv, spec.argv[1:]))


def test_read_file_defaults_to_bounded_range() -> None:
    spec = build_tool_command(call("read_file", path="README.md"))

    assert spec.argv == ("sed", "-n", f"1,{MAX_READ_LINES}p", "--", "README.md")


def test_read_file_accepts_custom_bounded_range() -> None:
    spec = build_tool_command(
        call("read_file", path="src/repo_agent/tools.py", start_line=25, end_line=40)
    )

    assert spec.argv == (
        "sed",
        "-n",
        "25,40p",
        "--",
        "src/repo_agent/tools.py",
    )


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("missing", {}),
        ("git_status", {"path": "README.md"}),
        ("git_diff", {"staged": 1}),
        ("list_files", {"glob": 123}),
        ("search", {}),
        ("search", {"pattern": ""}),
        ("search", {"pattern": "x", "fixed_strings": "yes"}),
        ("read_file", {}),
        ("read_file", {"path": "README.md", "start_line": True}),
        ("read_file", {"path": "README.md", "start_line": 0}),
        ("read_file", {"path": "README.md", "start_line": MAX_LINE_NUMBER + 1}),
        ("read_file", {"path": "README.md", "start_line": 5, "end_line": 4}),
        (
            "read_file",
            {"path": "README.md", "start_line": 1, "end_line": MAX_READ_LINES + 1},
        ),
    ],
)
def test_invalid_calls_are_rejected(name: str, arguments: dict[str, object]) -> None:
    with pytest.raises(ToolValidationError):
        build_tool_command(ToolCall(id="bad", name=name, arguments=arguments))


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:/Users/Administrator/.ssh/id_rsa",
        "../outside.txt",
        "src/../../outside.txt",
        "src\\..\\outside.txt",
        "src//main.py",
        "src/./main.py",
        ".git/config",
        "nested/.GIT/index",
        ".env",
        "config/.env.production",
        "certs/client.pem",
        "private/server.KEY",
        "id_rsa",
        "config/credentials.json",
        "secrets.yaml",
        "bad\x00name.py",
    ],
)
@pytest.mark.parametrize("tool", ["git_diff", "read_file"])
def test_unsafe_paths_are_rejected(path: str, tool: str) -> None:
    arguments: dict[str, object] = {"path": path}
    with pytest.raises(ToolValidationError):
        build_tool_command(ToolCall(id="unsafe", name=tool, arguments=arguments))


@pytest.mark.parametrize(
    "glob",
    [
        "../*.py",
        "/src/*.py",
        "C:/*.py",
        ".git/**",
        "**/.env*",
        "**/*.pem",
        "**/credentials.*",
        "bad\x00*.py",
        "a" * (MAX_GLOB_LENGTH + 1),
    ],
)
@pytest.mark.parametrize("builder", ["list_files", "search"])
def test_unsafe_globs_are_rejected(glob: str, builder: str) -> None:
    arguments: dict[str, object] = {"glob": glob}
    if builder == "search":
        arguments["pattern"] = "needle"
    with pytest.raises(ToolValidationError):
        build_tool_command(ToolCall(id="unsafe", name=builder, arguments=arguments))


def test_pattern_length_and_nul_are_rejected() -> None:
    for pattern in ("x" * (MAX_PATTERN_LENGTH + 1), "bad\x00pattern"):
        with pytest.raises(ToolValidationError):
            build_tool_command(call("search", pattern=pattern))


def test_command_specs_are_argv_only() -> None:
    builders: tuple[Callable[[], ToolCall], ...] = (
        lambda: call("git_status"),
        lambda: call("git_diff"),
        lambda: call("list_files"),
        lambda: call("search", pattern="TODO"),
        lambda: call("read_file", path="README.md"),
    )

    for make_call in builders:
        spec = build_tool_command(make_call())
        assert isinstance(spec.argv, tuple)
        assert spec.argv
        assert all(isinstance(argument, str) for argument in spec.argv)
        assert spec.cwd == "/workspace"
