"""Shared repository path and sensitive-file policy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Literal


PathAccess = Literal["read", "discover", "write"]

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "clock$",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_SECRET_EXACT_NAMES = frozenset(
    {
        ".dockerconfigjson",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "auth.json",
        "credential",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secret",
        "secrets",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "settings.xml",
    }
)
_SECRET_DIRECTORIES = frozenset({".aws", ".docker", ".kube", ".ssh"})
_SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".tfstate")
_SECRET_STEMS = frozenset({"credential", "credentials", "secret", "secrets"})
_SECRET_PREFIX_NAMES = (
    ".dockerconfigjson",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "auth.json",
    "settings.xml",
)
_WRITE_CONTROL_FILES = frozenset({".gitattributes", ".gitmodules"})

# Ripgrep/Git pathspec equivalents of the literal policy. Keeping these beside
# the path policy prevents discovery tools from exposing names that direct
# reads and writes reject.
SENSITIVE_DISCOVERY_GLOBS: tuple[str, ...] = (
    "!.git/**",
    "!**/.env",
    "!**/.env.*",
    "!**/.envrc",
    "!**/.envrc.*",
    "!**/*.pem",
    "!**/*.pem.*",
    "!**/*.key",
    "!**/*.key.*",
    "!**/*.p12",
    "!**/*.p12.*",
    "!**/*.pfx",
    "!**/*.pfx.*",
    "!**/id_rsa",
    "!**/id_rsa.*",
    "!**/id_dsa",
    "!**/id_dsa.*",
    "!**/id_ecdsa",
    "!**/id_ecdsa.*",
    "!**/id_ed25519",
    "!**/id_ed25519.*",
    "!**/credential",
    "!**/credential.*",
    "!**/credentials",
    "!**/credentials.*",
    "!**/.credential",
    "!**/.credential.*",
    "!**/.credentials",
    "!**/.credentials.*",
    "!**/secret",
    "!**/secret.*",
    "!**/secrets",
    "!**/secrets.*",
    "!**/.secret",
    "!**/.secret.*",
    "!**/.secrets",
    "!**/.secrets.*",
    "!**/.netrc",
    "!**/.netrc.*",
    "!**/.npmrc",
    "!**/.npmrc.*",
    "!**/.pypirc",
    "!**/.pypirc.*",
    "!**/.git-credentials",
    "!**/.git-credentials.*",
    "!**/.dockerconfigjson",
    "!**/.dockerconfigjson.*",
    "!**/auth.json",
    "!**/auth.json.*",
    "!**/settings.xml",
    "!**/settings.xml.*",
    "!**/.aws/**",
    "!**/.docker/**",
    "!**/.kube/**",
    "!**/.ssh/**",
    "!**/*.tfstate",
    "!**/*.tfstate.*",
)


class RepositoryPathError(ValueError):
    """Raised when a repository-relative path crosses the safety policy."""


@dataclass(frozen=True, slots=True)
class RepositoryPathPolicy:
    """Validate portable repository paths for read, discovery, and writes."""

    max_path_length: int = 512

    def validate(self, value: object, *, access: PathAccess = "read") -> str:
        if access not in {"read", "discover", "write"}:
            raise ValueError(f"unsupported path access: {access!r}")
        if not isinstance(value, str):
            raise RepositoryPathError("path must be a string")
        if not value:
            raise RepositoryPathError("path must not be empty")
        if len(value) > self.max_path_length:
            raise RepositoryPathError(
                f"path must be at most {self.max_path_length} characters"
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise RepositoryPathError("path must not contain control characters")
        if "\\" in value:
            raise RepositoryPathError("path must use forward slashes")
        if value.startswith("/") or _WINDOWS_ABSOLUTE.match(value):
            raise RepositoryPathError("path must be relative to the repository")
        if any(character in value for character in '<>"|*?') or ":" in value:
            raise RepositoryPathError("path contains characters unsafe on Windows")

        raw_parts = value.split("/")
        if any(part in {"", ".", ".."} for part in raw_parts):
            raise RepositoryPathError(
                "path must be canonical and must not contain traversal"
            )
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts:
            raise RepositoryPathError("path must be relative to the repository")

        for part in path.parts:
            folded = part.casefold()
            if part.endswith((" ", ".")):
                raise RepositoryPathError(
                    "path components must not end in a dot or space"
                )
            if folded == ".git":
                raise RepositoryPathError("access to .git internals is not allowed")
            device_stem = folded.split(".", 1)[0]
            if device_stem in _WINDOWS_RESERVED:
                raise RepositoryPathError("path contains a reserved Windows name")
            if is_sensitive_name(part):
                raise RepositoryPathError(
                    "access to suspected secret files is not allowed"
                )
            if folded in _SECRET_DIRECTORIES:
                raise RepositoryPathError(
                    "access to suspected credential directories is not allowed"
                )

        if access == "write" and path.name.casefold() in _WRITE_CONTROL_FILES:
            raise RepositoryPathError(
                "changing Git checkout-control files is not allowed"
            )
        return path.as_posix()

    def validate_disk_path(
        self,
        root: Path,
        relative_path: str,
        *,
        access: PathAccess = "read",
        allow_missing_leaf: bool = False,
    ) -> Path:
        """Reject links/reparse points without resolving through them."""

        normalized = self.validate(relative_path, access=access)
        root_stat = os.lstat(root)
        if not stat.S_ISDIR(root_stat.st_mode) or _is_reparse_point(root_stat):
            raise RepositoryPathError(f"repository root is not a trusted directory: {root}")

        current = root
        parts = PurePosixPath(normalized).parts
        missing = False
        for index, part in enumerate(parts):
            current = current / part
            if missing:
                continue
            try:
                current_stat = os.lstat(current)
            except FileNotFoundError:
                if allow_missing_leaf:
                    missing = True
                    continue
                raise RepositoryPathError(f"repository path does not exist: {normalized}")
            if _is_reparse_point(current_stat):
                raise RepositoryPathError(
                    "repository path must not contain symbolic links or reparse points"
                )
            if index < len(parts) - 1 and not stat.S_ISDIR(current_stat.st_mode):
                raise RepositoryPathError(
                    f"repository path parent is not a directory: {normalized}"
                )
        return current


DEFAULT_PATH_POLICY = RepositoryPathPolicy()


def is_sensitive_name(name: str) -> bool:
    """Return whether one literal path component resembles a credential file."""

    folded = name.casefold()
    if (
        folded == ".env"
        or folded.startswith(".env.")
        or folded == ".envrc"
        or folded.startswith(".envrc.")
    ):
        return True
    if folded in _SECRET_EXACT_NAMES:
        return True
    if folded.endswith(_SECRET_SUFFIXES) or any(
        f"{suffix}." in folded for suffix in _SECRET_SUFFIXES
    ):
        return True
    if any(
        folded == prefix or folded.startswith(f"{prefix}.")
        for prefix in _SECRET_PREFIX_NAMES
    ):
        return True
    if folded.lstrip(".").split(".", 1)[0] in _SECRET_STEMS:
        return True
    if folded in _SECRET_DIRECTORIES:
        return True
    return any(
        folded.startswith(f"{key_name}.")
        for key_name in ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")
    )


def _is_reparse_point(path_stat: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & reparse_flag
    )


__all__ = [
    "DEFAULT_PATH_POLICY",
    "PathAccess",
    "RepositoryPathError",
    "RepositoryPathPolicy",
    "SENSITIVE_DISCOVERY_GLOBS",
    "is_sensitive_name",
]
