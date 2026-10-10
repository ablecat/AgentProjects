from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from repo_agent import patches as patches_module
from repo_agent.patches import (
    MAX_CHANGED_LINES,
    MAX_PATCH_BYTES,
    MAX_PATCH_FILES,
    MAX_PATCH_LINES,
    PatchValidationError,
    normalize_model_patch,
    validate_patch,
)


def modify_patch(
    path: str = "src/app.py",
    *,
    old: str = "old value",
    new: str = "new value",
    mode: str = "100644",
) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"index 1111111..2222222 {mode}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{old}\n"
        f"+{new}\n"
    )


def add_patch(path: str = "src/new_file.py", *, content: str = "new value") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..2222222\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        f"+{content}\n"
    )


def delete_patch(path: str = "src/old_file.py", *, content: str = "old value") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        f"--- a/{path}\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        f"-{content}\n"
    )


def test_valid_multi_file_patch_returns_structured_operations() -> None:
    patch = modify_patch() + add_patch() + delete_patch()

    validated = validate_patch(patch)

    assert validated.text == patch
    assert validated.paths == (
        "src/app.py",
        "src/new_file.py",
        "src/old_file.py",
    )
    assert validated.added_paths == ("src/new_file.py",)
    assert [item.operation for item in validated.files] == [
        "modify",
        "add",
        "delete",
    ]
    assert validated.changed_lines == 4


def test_crlf_is_normalized_and_missing_final_newline_is_added() -> None:
    patch = modify_patch().rstrip("\n").replace("\n", "\r\n")

    validated = validate_patch(patch)

    assert "\r" not in validated.text
    assert validated.text.endswith("\n")


def test_valid_multiple_hunks_with_context_are_counted_per_file() -> None:
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,3 +1,3 @@\n"
        " first\n"
        "-old one\n"
        "+new one\n"
        " third\n"
        "@@ -10,2 +10,3 @@ def later():\n"
        " context\n"
        "-old two\n"
        "+new two\n"
        "+extra\n"
    )

    validated = validate_patch(patch)

    assert validated.changed_lines == 5
    assert validated.files[0].added_lines == 3
    assert validated.files[0].removed_lines == 2


def test_no_newline_marker_does_not_change_hunk_counts() -> None:
    patch = modify_patch().replace(
        "-old value\n+new value\n",
        "-old value\n\\ No newline at end of file\n"
        "+new value\n\\ No newline at end of file\n",
    )

    validated = validate_patch(patch)

    assert validated.changed_lines == 2


@pytest.mark.parametrize(
    "patch",
    [None, b"diff", 7, "", "   \n", "not a diff\n", "\ud800"],
)
def test_non_text_empty_and_non_diff_inputs_are_rejected(patch: object) -> None:
    with pytest.raises(PatchValidationError):
        validate_patch(patch)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda patch: patch.replace("diff --git", 'diff --git "a/src/app.py"', 1),
        lambda patch: patch.replace("--- a/src/app.py\n", "", 1),
        lambda patch: patch.replace("+++ b/src/app.py\n", "", 1),
        lambda patch: patch.replace("+++ b/src/app.py", "+++ b/src/other.py", 1),
        lambda patch: patch.replace("@@ -1 +1 @@", "@@ malformed @@", 1),
        lambda patch: patch.replace("-old value", "?old value", 1),
        lambda patch: patch + "unsupported metadata\n",
        lambda patch: patch.replace("\n", "\r", 1),
        lambda patch: patch.replace("old value", "old\x00value", 1),
    ],
)
def test_malformed_patch_structures_are_rejected(mutation) -> None:
    with pytest.raises(PatchValidationError):
        validate_patch(mutation(modify_patch()))


def test_duplicate_file_sections_are_rejected() -> None:
    with pytest.raises(PatchValidationError, match="repeats path"):
        validate_patch(modify_patch() + modify_patch(new="another value"))


def test_hunk_header_counts_must_match_the_hunk_body() -> None:
    malformed = modify_patch().replace("@@ -1 +1 @@", "@@ -1,9 +1,7 @@")

    with pytest.raises(PatchValidationError):
        validate_patch(malformed)


def test_model_patch_recounts_stale_hunk_headers_but_public_validator_stays_strict() -> None:
    malformed = modify_patch().replace("@@ -1 +1 @@", "@@ -1,9 +1,7 @@")

    with pytest.raises(PatchValidationError):
        validate_patch(malformed)

    normalized = normalize_model_patch(malformed)

    assert normalized.text.startswith("diff --git a/src/app.py b/src/app.py\n")
    assert "@@ -1,1 +1,1 @@" in normalized.text
    assert normalized.changed_lines == 2


def test_model_patch_recount_then_canonicalize_uses_corrected_hunks(
    tmp_path: Path,
) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,99 +1,99 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
        " four\n"
        "diff --git a/src/new.py b/src/new.py\n"
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +99 @@\n"
        "+created\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    _assert_git_check(repository, normalized.text)
    assert "@@ -1,4 +1,4 @@" in normalized.text
    assert "new file mode 100644\n" in normalized.text


def test_model_patch_recovers_bare_hunk_and_empty_context_lines(
    tmp_path: Path,
) -> None:
    repository = _git_fixture(tmp_path, content="one\ntwo\n\nfour\n")
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        "\n"
        " four\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    _assert_git_check(repository, normalized.text)
    assert "-two\n" in normalized.text
    assert "+TWO\n" in normalized.text


def test_model_patch_strips_one_markdown_diff_fence() -> None:
    patch = "```diff\n" + modify_patch() + "```\n"

    normalized = normalize_model_patch(patch)

    assert normalized.paths == ("src/app.py",)
    assert normalized.text.startswith("diff --git a/src/app.py b/src/app.py\n")


def test_model_patch_recount_does_not_relax_path_or_hunk_content_policy() -> None:
    unsafe = modify_patch("../outside.py").replace("@@ -1 +1 @@", "@@ -1,9 +1,7 @@")
    malformed_body = modify_patch().replace("-old value", "?old value")
    bare_carriage_return = modify_patch().replace("\n", "\r", 1)

    with pytest.raises(PatchValidationError):
        normalize_model_patch(unsafe)
    with pytest.raises(PatchValidationError):
        normalize_model_patch(malformed_body)
    with pytest.raises(PatchValidationError, match="carriage returns"):
        normalize_model_patch(bare_carriage_return)


def _git_fixture(tmp_path: Path, *, content: str = "one\ntwo\nthree\nfour\n") -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is required for model patch integration checks")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "src" / "app.py").write_bytes(content.encode("utf-8"))
    for args in (
        ("init", "-q"),
        ("config", "core.autocrlf", "false"),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "user.name", "patch-tests"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        completed = subprocess.run(
            ["git", *args], cwd=repository, capture_output=True, check=False
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return repository


def _assert_git_check(repository: Path, patch: str) -> None:
    completed = subprocess.run(
        ["git", "apply", "--check", "--index", "--whitespace=error-all", "--"],
        cwd=repository,
        input=patch.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


@pytest.mark.parametrize(
    "check_patch",
    [patches_module._git_patch_applies, patches_module._git_patch_applies_recount],
)
def test_git_patch_checks_remove_hostile_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check_patch,
) -> None:
    repository = _git_fixture(tmp_path)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    hostile_config = hostile / "gitconfig"
    hostile_config.write_text("[apply]\n\twhitespace = nowarn\n", encoding="utf-8")
    hostile_values = {
        "GIT_DIR": str(hostile / ".git"),
        "GIT_WORK_TREE": str(hostile),
        "GIT_INDEX_FILE": str(hostile / "index"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.attributesFile",
        "GIT_CONFIG_VALUE_0": str(hostile / "attributes"),
        "GIT_CONFIG_GLOBAL": str(hostile_config),
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_CONFIG_SYSTEM": str(hostile_config),
    }
    for key, value in hostile_values.items():
        monkeypatch.setenv(key, value)

    real_run = subprocess.run
    observed_environments: list[dict[str, str]] = []

    def recording_run(*args, **kwargs):
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environments.append(environment)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(patches_module.subprocess, "run", recording_run)
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,4 +1,4 @@\n"
        "-one\n"
        "+ONE\n"
        " two\n"
        " three\n"
        " four\n"
    )

    assert check_patch(repository, patch)
    assert len(observed_environments) == 1
    environment = observed_environments[0]
    for key in hostile_values:
        if key in {"GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_SYSTEM"}:
            continue
        assert key not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert not (hostile / "index").exists()


def test_model_patch_canonicalizes_bare_unified_diff_without_deleting_omitted_lines(
    tmp_path: Path,
) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-two\n"
        "+TWO\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    _assert_git_check(repository, normalized.text)
    assert " three\n" in normalized.text
    assert normalized.paths == ("src/app.py",)


def test_model_patch_canonicalizes_editor_style_update(tmp_path: Path) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "*** Begin Patch\n"
        "*** Update File: src/app.py\n"
        "@@\n"
        "-two\n"
        "+TWO\n"
        "*** End Patch\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    _assert_git_check(repository, normalized.text)
    assert "-two\n" in normalized.text
    assert "+TWO\n" in normalized.text


def test_model_add_without_mode_gets_git_safe_metadata(tmp_path: Path) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "diff --git a/src/new.py b/src/new.py\n"
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+created\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    assert "new file mode 100644\n" in normalized.text


def test_model_add_trims_terminal_blank_hunk_line(tmp_path: Path) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "diff --git a/src/new.py b/src/new.py\n"
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+one\n"
        "+two\n"
        "+\n"
    )

    normalized = normalize_model_patch(patch, repository=repository)

    _assert_git_check(repository, normalized.text)
    assert "+two\n" in normalized.text
    assert normalized.text.endswith("+two\n")
    _assert_git_check(repository, normalized.text)


def test_model_patch_rejects_noncontiguous_fuzzy_context_instead_of_deleting_lines(
    tmp_path: Path,
) -> None:
    repository = _git_fixture(tmp_path, content="one\ntwo\nthree\nfour\n")
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,4 +1,4 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " four\n"
    )

    # The body is deliberately inconsistent with the checkout: ``three`` is
    # omitted between two context lines. It must not be treated as disposable.
    with pytest.raises(PatchValidationError):
        normalize_model_patch(patch, repository=repository)


def test_model_patch_rejects_context_outside_bounded_fuzz_window(
    tmp_path: Path,
) -> None:
    content = "".join(f"line-{index}\n" for index in range(100))
    repository = _git_fixture(tmp_path, content=content)
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-line-99\n"
        "+changed\n"
    )

    with pytest.raises(PatchValidationError):
        normalize_model_patch(patch, repository=repository)


def test_model_patch_rejects_equidistant_ambiguous_context(tmp_path: Path) -> None:
    lines = [f"line-{index}" for index in range(30)]
    lines[9] = "TARGET"
    lines[19] = "TARGET"
    repository = _git_fixture(tmp_path, content="\n".join(lines) + "\n")
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -15 +15 @@\n"
        "-TARGET\n"
        "+changed\n"
    )

    with pytest.raises(PatchValidationError):
        normalize_model_patch(patch, repository=repository)


def test_model_editor_path_traversal_is_rejected(tmp_path: Path) -> None:
    repository = _git_fixture(tmp_path)
    patch = (
        "*** Begin Patch\n"
        "*** Update File: ../outside.py\n"
        "@@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )

    with pytest.raises(PatchValidationError):
        normalize_model_patch(patch, repository=repository)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside.py",
        "src/../../outside.py",
        ".git/config",
        "nested/.GIT/index",
        ".env",
        "config/.env.production",
        "config/credentials.json",
        "private/server.pem",
        ".ssh/id_rsa",
        ".gitmodules",
        ".gitattributes",
        "stream.py:payload",
        "src\\outside.py",
        "CON.py",
        "nested/AUX.txt",
        "trailing-dot./file.py",
        "bad\tname.py",
    ],
)
def test_unsafe_patch_paths_are_rejected(path: str) -> None:
    with pytest.raises(PatchValidationError):
        validate_patch(modify_patch(path))


def test_paths_with_spaces_are_rejected_by_the_day_two_patch_subset() -> None:
    patch = modify_patch("docs/user guide.md")

    with pytest.raises(PatchValidationError, match="unquoted, whitespace-free"):
        validate_patch(patch)


@pytest.mark.parametrize(
    "metadata",
    [
        "Binary files a/src/blob.bin and b/src/blob.bin differ",
        "GIT binary patch",
        "old mode 100644\nnew mode 100755",
        "new file mode 100755",
        "new file mode 120000",
        "deleted file mode 120000",
        "index 1111111..2222222 120000",
        "index 1111111..2222222 160000",
        "Submodule vendor/library contains modified content",
    ],
)
def test_binary_mode_symlink_and_gitlink_metadata_are_rejected(metadata: str) -> None:
    patch = modify_patch().replace(
        "index 1111111..2222222 100644",
        metadata,
        1,
    )

    with pytest.raises(PatchValidationError):
        validate_patch(patch)


def test_symlink_creation_patch_is_rejected() -> None:
    patch = (
        "diff --git a/public.txt b/public.txt\n"
        "new file mode 120000\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/public.txt\n"
        "@@ -0,0 +1 @@\n"
        "+../outside.txt\n"
    )

    with pytest.raises(PatchValidationError, match="binary, link, mode"):
        validate_patch(patch)


@pytest.mark.parametrize(
    "metadata",
    [
        "similarity index 100%\nrename from src/old.py\nrename to src/new.py",
        "similarity index 100%\ncopy from src/old.py\ncopy to src/new.py",
    ],
)
def test_rename_and_copy_patches_are_rejected(metadata: str) -> None:
    patch = (
        "diff --git a/src/old.py b/src/new.py\n"
        f"{metadata}\n"
        "--- a/src/old.py\n"
        "+++ b/src/new.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    with pytest.raises(PatchValidationError, match="rename|copy"):
        validate_patch(patch)


def test_unchanged_executable_mode_is_allowed() -> None:
    validated = validate_patch(modify_patch("scripts/run.sh", mode="100755"))

    assert validated.paths == ("scripts/run.sh",)


def test_forbidden_words_inside_added_source_are_not_treated_as_metadata() -> None:
    patch = modify_patch(new="GIT binary patch")

    validated = validate_patch(patch)

    assert validated.changed_lines == 2


def test_patch_byte_limit_is_enforced_for_multibyte_text() -> None:
    patch = add_patch(content="\u754c" * (MAX_PATCH_BYTES // 3 + 1))

    with pytest.raises(PatchValidationError, match=str(MAX_PATCH_BYTES)):
        validate_patch(patch)


def test_patch_line_limit_is_enforced() -> None:
    patch = modify_patch().rstrip("\n") + ("\n context" * MAX_PATCH_LINES) + "\n"

    with pytest.raises(PatchValidationError, match=str(MAX_PATCH_LINES)):
        validate_patch(patch)


def test_patch_file_limit_is_enforced() -> None:
    patch = "".join(
        modify_patch(f"src/file_{index}.py")
        for index in range(MAX_PATCH_FILES + 1)
    )

    with pytest.raises(PatchValidationError, match=str(MAX_PATCH_FILES)):
        validate_patch(patch)


def test_changed_line_limit_is_enforced() -> None:
    additions = MAX_CHANGED_LINES + 1
    patch = (
        "diff --git a/src/generated.py b/src/generated.py\n"
        "new file mode 100644\n"
        "index 0000000..2222222\n"
        "--- /dev/null\n"
        "+++ b/src/generated.py\n"
        f"@@ -0,0 +1,{additions} @@\n"
        + "+line\n" * additions
    )

    with pytest.raises(PatchValidationError, match=str(MAX_CHANGED_LINES)):
        validate_patch(patch)
