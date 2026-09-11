"""Deterministic repository path collection."""

from __future__ import annotations

from collections.abc import Iterable


def collect_paths(paths: Iterable[str]) -> list[str]:
    """Return normalized repository paths in deterministic order."""

    normalized = {path.removeprefix("./") for path in paths if path}
    return sorted(normalized, key=lambda value: (value.casefold(), value))
