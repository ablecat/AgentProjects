"""Typed records shared by the durable workflow, CLI, and REST API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


PublicRunStatus: TypeAlias = Literal[
    "queued",
    "planning",
    "awaiting_approval",
    "running",
    "interrupted",
    "succeeded",
    "unverified",
    "failed",
    "cancelled",
    "policy_denied",
    "rejected",
]

WorkflowNode: TypeAlias = Literal[
    "prepare",
    "baseline_check",
    "inspect_and_plan",
    "approval",
    "implement",
    "verify",
    "repair",
    "review",
    "review_repair",
    "finalize",
]

ArtifactKind: TypeAlias = Literal["patch", "report", "result", "trace", "checks"]


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for JSON and SQLite."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class ChangePlan(BaseModel):
    """A model-produced plan that must be approved before mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    goal: str = Field(min_length=1, max_length=2000)
    files: tuple[str, ...] = Field(default_factory=tuple, max_length=12)
    steps: tuple[str, ...] = Field(min_length=1, max_length=20)
    checks: tuple[str, ...] = Field(min_length=1, max_length=10)
    risks: tuple[str, ...] = Field(default_factory=tuple, max_length=10)

    @field_validator("files")
    @classmethod
    def validate_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for value in values:
            if not value or "\\" in value or "\x00" in value:
                raise ValueError("plan files must be non-empty POSIX paths")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or value.startswith(".git/"):
                raise ValueError("plan files must stay inside the repository")
            folded = value.casefold()
            if folded in seen:
                raise ValueError("plan files must not contain duplicates")
            seen.add(folded)
        return values

    @field_validator("steps", "checks", "risks")
    @classmethod
    def validate_text_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in values):
            raise ValueError("plan list items must not be empty")
        return values


class CheckSummary(BaseModel):
    """Bounded verification information safe to expose to clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=0, le=4)
    check_id: str | None = Field(default=None, max_length=128)
    status: str = Field(min_length=1, max_length=64)
    ok: bool
    duration_ms: int = Field(ge=0)
    log_artifact: str | None = Field(default=None, max_length=255)
    error: str | None = Field(default=None, max_length=4000)


class RunMetrics(BaseModel):
    """Deterministic counters accumulated over a workflow run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Runtime policy caps execution; the record keeps real overage values for diagnosis.
    node_count: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0, le=2)
    review_repairs: int = Field(default=0, ge=0, le=1)
    duration_ms: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)


class RunRecord(BaseModel):
    """Durable public run state used by both local interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    repo_path: str = Field(min_length=1, max_length=32767)
    task: str = Field(min_length=1, max_length=4000)
    base_ref: str | None = Field(default=None, max_length=255)
    base_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    status: PublicRunStatus
    current_node: WorkflowNode | None = None
    plan: ChangePlan | None = None
    checks: tuple[CheckSummary, ...] = Field(default_factory=tuple)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    error: str | None = Field(default=None, max_length=4000)
    summary: str | None = Field(default=None, max_length=4000)
    approval_reason: str | None = Field(default=None, max_length=2000)
    auto_approve: bool = False
    allow_remote_model: bool = False
    cancel_requested: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None

    @field_validator("run_id")
    @classmethod
    def normalize_run_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("invalid run id")
        return value


class TraceEvent(BaseModel):
    """One append-only, redacted workflow event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    timestamp: str
    node: WorkflowNode | None = None
    event: str = Field(min_length=1, max_length=128)
    detail: dict[str, Any] = Field(default_factory=dict)


TERMINAL_STATUSES: frozenset[PublicRunStatus] = frozenset(
    {
        "succeeded",
        "unverified",
        "failed",
        "cancelled",
        "policy_denied",
        "rejected",
    }
)

ARTIFACT_FILENAMES: dict[ArtifactKind, str] = {
    "patch": "patch.diff",
    "report": "report.md",
    "result": "run.json",
    "trace": "trace.jsonl",
    "checks": "checks",
}


__all__ = [
    "ARTIFACT_FILENAMES",
    "TERMINAL_STATUSES",
    "ArtifactKind",
    "ChangePlan",
    "CheckSummary",
    "PublicRunStatus",
    "RunMetrics",
    "RunRecord",
    "TraceEvent",
    "WorkflowNode",
    "utc_now",
]
