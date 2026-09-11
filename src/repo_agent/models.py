"""Data exchanged by the repository agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias


RunStatus: TypeAlias = Literal[
    "completed",
    "completed_with_errors",
    "max_steps",
    "repeated_call",
    "provider_error",
]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider request to invoke one named tool."""

    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized outcome of a tool invocation."""

    call_id: str
    name: str
    ok: bool
    output: str
    error: str | None = None
    exit_code: int | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """A provider response indicating that no more tools are needed."""

    content: str


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    """The reviewable result captured before a candidate workspace is removed."""

    base_commit: str
    revision: int
    changed_paths: tuple[str, ...]
    patch: str
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class RunResult:
    """A complete, serializable record of an agent-loop run."""

    task: str
    status: RunStatus
    answer: str | None
    steps: int
    tool_results: tuple[ToolResult, ...]
    candidate: CandidateArtifact | None = None
