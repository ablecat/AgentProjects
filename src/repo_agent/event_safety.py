"""Allowlisted string projection for public workflow event streams."""

from __future__ import annotations

import re


PUBLIC_EVENT_NAMES = frozenset(
    {
        "cancel_requested",
        "model_tool_summary",
        "node_completed",
        "plan_approved",
        "plan_rejected",
        "planning_tool_summary",
        "run_created",
        "run_resumed",
        "service_interrupted",
        "workflow_cancelled",
        "workflow_checkpoint",
        "workflow_event",
        "workflow_failed",
        "workflow_resumed",
        "workflow_started",
        "workflow_state_synchronized",
        "workflow_stopped",
    }
)
PUBLIC_EVENT_NODES = frozenset(
    {
        "approval",
        "baseline_check",
        "finalize",
        "implement",
        "inspect_and_plan",
        "prepare",
        "repair",
        "review",
        "review_repair",
        "verify",
    }
)
PUBLIC_EVENT_PHASES = frozenset({"implement", "planning", "repair", "review"})
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)


def public_event_name(value: object) -> str:
    if isinstance(value, str) and value in PUBLIC_EVENT_NAMES:
        return value
    return "workflow_event"


def public_event_node(value: object) -> str | None:
    if isinstance(value, str) and value in PUBLIC_EVENT_NODES:
        return value
    return None


def public_event_phase(value: object) -> str | None:
    if isinstance(value, str) and value in PUBLIC_EVENT_PHASES:
        return value
    return None


def public_event_timestamp(value: object) -> str | None:
    if isinstance(value, str) and _UTC_TIMESTAMP_RE.fullmatch(value) is not None:
        return value
    return None


__all__ = [
    "PUBLIC_EVENT_NAMES",
    "PUBLIC_EVENT_NODES",
    "PUBLIC_EVENT_PHASES",
    "public_event_name",
    "public_event_node",
    "public_event_phase",
    "public_event_timestamp",
]
