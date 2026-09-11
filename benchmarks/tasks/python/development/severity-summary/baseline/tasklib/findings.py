"""Finding records and text formatting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    message: str


def format_findings(findings: list[Finding]) -> str:
    """Render findings in their source order."""

    return "".join(
        f"{finding.source}: {finding.severity}: {finding.message}\n"
        for finding in findings
    )
