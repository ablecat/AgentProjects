from __future__ import annotations

from pathlib import Path

import pytest

from repo_agent.policy import (
    DEFAULT_PATH_POLICY,
    RepositoryPathError,
    RepositoryPathPolicy,
    is_sensitive_name,
)


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/repo_agent/policy.py",
        "docs/user guide.md",
        ".github/workflows/ci.yml",
        "assets/cafe.txt",
    ],
)
@pytest.mark.parametrize("access", ["read", "discover", "write"])
def test_portable_repository_paths_are_accepted(path: str, access: str) -> None:
    assert DEFAULT_PATH_POLICY.validate(path, access=access) == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "//server/share/file.txt",
        "C:/Windows/win.ini",
        "c:relative.txt",
        "../outside.txt",
        "src/../../outside.txt",
        "src/./main.py",
        "src//main.py",
        "src\\main.py",
        ".git/config",
        "nested/.GIT/index",
        "nested/.git./config",
        "nested/.git /config",
        "bad\x00name.py",
        "bad\nname.py",
        "bad\rname.py",
        "bad\tname.py",
        "bad\x7fname.py",
        "stream.txt:secret",
        'bad"name.py',
        "bad*name.py",
        "bad?name.py",
        "bad<name.py",
        "bad>name.py",
        "bad|name.py",
        "trailing-dot./file.py",
        "trailing-space /file.py",
        "CON",
        "con.txt",
        "nested/PRN.log",
        "nested/aux",
        "nested/NUL.json",
        "nested/CLOCK$.txt",
        "nested/COM1.py",
        "nested/com9.cfg",
        "nested/LPT1",
        "nested/lpt9.txt",
    ],
)
def test_noncanonical_or_windows_unsafe_paths_are_rejected(path: str) -> None:
    with pytest.raises(RepositoryPathError):
        DEFAULT_PATH_POLICY.validate(path)


@pytest.mark.parametrize("value", [None, 1, b"README.md", Path("README.md")])
def test_non_string_paths_are_rejected(value: object) -> None:
    with pytest.raises(RepositoryPathError, match="must be a string"):
        DEFAULT_PATH_POLICY.validate(value)


def test_path_length_and_access_mode_are_bounded() -> None:
    policy = RepositoryPathPolicy(max_path_length=8)

    assert policy.validate("a/b.txt") == "a/b.txt"
    with pytest.raises(RepositoryPathError, match="at most 8"):
        policy.validate("long-name.txt")
    with pytest.raises(ValueError, match="unsupported path access"):
        policy.validate("a.py", access="execute")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".ENV.production",
        ".envrc",
        "config/.env.local",
        ".netrc",
        ".npmrc",
        ".npmrc.bak",
        ".pypirc",
        ".git-credentials",
        "auth.json",
        "auth.json.backup",
        "credentials",
        "config/CREDENTIALS.json",
        "config/.credentials.json",
        "secret.toml",
        ".secret.toml",
        "secrets.yaml",
        "id_rsa",
        "id_ed25519.pub",
        "certs/client.pem",
        "certs/client.pem.bak",
        "certs/client.KEY",
        "certs/client.p12",
        "certs/client.PFX",
        "state/terraform.tfstate",
        "state/terraform.tfstate.backup",
        "settings.xml",
        ".aws/credentials",
        ".docker/config.json",
        ".kube/config",
        ".ssh/config",
    ],
)
@pytest.mark.parametrize("access", ["read", "discover", "write"])
def test_sensitive_paths_are_rejected_for_every_access_mode(
    path: str, access: str
) -> None:
    with pytest.raises(RepositoryPathError, match="credential|secret"):
        DEFAULT_PATH_POLICY.validate(path, access=access)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.test",
        ".NETRC",
        "credentials.json",
        "ID_RSA.backup",
        "private.pem",
        "terraform.tfstate",
    ],
)
def test_sensitive_name_matching_is_case_insensitive(name: str) -> None:
    assert is_sensitive_name(name)


@pytest.mark.parametrize("name", ["environment.py", "secretary.md", "keynote.txt"])
def test_non_secret_lookalikes_are_not_overblocked(name: str) -> None:
    assert not is_sensitive_name(name)


@pytest.mark.parametrize("path", [".gitattributes", ".gitmodules"])
def test_checkout_control_files_are_readable_but_not_writable(path: str) -> None:
    assert DEFAULT_PATH_POLICY.validate(path, access="read") == path
    assert DEFAULT_PATH_POLICY.validate(path, access="discover") == path
    with pytest.raises(RepositoryPathError, match="checkout-control"):
        DEFAULT_PATH_POLICY.validate(path, access="write")


def test_disk_path_validation_accepts_regular_paths_and_controlled_missing_leaf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    existing = root / "src" / "main.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("print('ok')\n", encoding="utf-8")

    assert DEFAULT_PATH_POLICY.validate_disk_path(root, "src/main.py") == existing
    missing = DEFAULT_PATH_POLICY.validate_disk_path(
        root,
        "new/package/module.py",
        access="write",
        allow_missing_leaf=True,
    )
    assert missing == root / "new" / "package" / "module.py"
    with pytest.raises(RepositoryPathError, match="does not exist"):
        DEFAULT_PATH_POLICY.validate_disk_path(root, "missing.py")


def test_disk_path_validation_rejects_a_regular_file_as_parent(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "not-a-directory").write_text("content\n", encoding="utf-8")

    with pytest.raises(RepositoryPathError, match="parent is not a directory"):
        DEFAULT_PATH_POLICY.validate_disk_path(
            root,
            "not-a-directory/child.py",
            access="write",
            allow_missing_leaf=True,
        )


def test_disk_path_validation_rejects_symlink_leaf_and_parent(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "value.txt").write_text("outside\n", encoding="utf-8")
    file_link = root / "public.txt"
    directory_link = root / "linked"
    try:
        file_link.symlink_to(outside / "value.txt")
        directory_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    for path in ("public.txt", "linked/value.txt"):
        with pytest.raises(RepositoryPathError, match="symbolic links|reparse"):
            DEFAULT_PATH_POLICY.validate_disk_path(root, path)


def test_disk_path_validation_rejects_symlink_repository_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-repository"
    real_root.mkdir()
    linked_root = tmp_path / "linked-repository"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(RepositoryPathError, match="root is not a trusted directory"):
        DEFAULT_PATH_POLICY.validate_disk_path(
            linked_root,
            "new.py",
            access="write",
            allow_missing_leaf=True,
        )
