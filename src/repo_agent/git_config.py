"""Read the one host Git setting required to judge a Windows checkout."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


_AUTOCRLF_ALIASES = {
    "true": "true",
    "yes": "true",
    "on": "true",
    "1": "true",
    "false": "false",
    "no": "false",
    "off": "false",
    "0": "false",
    "input": "input",
}


def effective_core_autocrlf(
    repository: str | os.PathLike[str],
    *,
    timeout_seconds: float = 10.0,
) -> str:
    """Return a validated effective ``core.autocrlf`` value.

    Repository snapshots run with all ambient Git configuration disabled. A clean
    worktree check is different: it must use the line-ending conversion under which
    the user's checkout was created, otherwise CRLF files can be reported as dirty.
    Only this three-value setting crosses the boundary; aliases, hooks, filters, and
    every other ambient option remain excluded from subsequent Git commands.
    """

    root = Path(repository).resolve(strict=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "config",
                "--get",
                "core.autocrlf",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            shell=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not read the Git line-ending configuration") from exc
    if completed.returncode == 1:
        return "false"
    if completed.returncode != 0:
        raise RuntimeError("could not read the Git line-ending configuration")
    if len(completed.stdout) > 16:
        raise RuntimeError("Git core.autocrlf is not a supported value")
    try:
        value = completed.stdout.decode("ascii", errors="strict").strip().casefold()
    except UnicodeError as exc:
        raise RuntimeError("Git core.autocrlf is not a supported value") from exc
    if value not in _AUTOCRLF_ALIASES:
        raise RuntimeError("Git core.autocrlf is not true, false, or input")
    return _AUTOCRLF_ALIASES[value]


__all__ = ["effective_core_autocrlf"]
