"""Execute one registered tool action."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    name: str
    status: str
    output: str
    error: str | None = None


def run_tool(name: str, action: Callable[[float], str]) -> ToolResult:
    """Execute an action using the historical default timeout hint."""

    return ToolResult(name=name, status="completed", output=action(30.0))
