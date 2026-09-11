"""Render repository findings for people."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    severity: str
    message: str


def render_text(findings: list[Finding]) -> str:
    """Render one stable, human-readable line per finding."""

    return "".join(
        f"{finding.path}:{finding.line}: "
        f"{finding.severity.upper()}: {finding.message}\n"
        for finding in findings
    )
