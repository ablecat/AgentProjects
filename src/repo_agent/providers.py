"""Provider interfaces and a deterministic provider for local demos."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import FinalAnswer, ToolCall, ToolResult


@runtime_checkable
class Provider(Protocol):
    """Choose the next tool call or finish from the accumulated evidence."""

    def next_step(
        self, task: str, results: Sequence[ToolResult]
    ) -> ToolCall | FinalAnswer:
        """Return the next action for *task*."""


class DemoProvider:
    """Exercise the loop without an API key or model dependency."""

    def next_step(
        self, task: str, results: Sequence[ToolResult]
    ) -> ToolCall | FinalAnswer:
        if not results:
            return ToolCall(id="demo-git-status", name="git_status", arguments={})

        if len(results) == 1:
            return ToolCall(
                id="demo-repo-map",
                name="repo_map",
                arguments={"max_files": 200, "include_symbols": True},
            )

        if len(results) == 2:
            return ToolCall(
                id="demo-search-todos",
                name="search",
                arguments={"pattern": "TODO|FIXME"},
            )

        git_summary = _summarize_result(results[0])
        map_summary = _summarize_result(results[1])
        search_summary = _summarize_result(results[2])
        outcome = (
            "Demo inspection complete."
            if all(result.ok for result in results[:3])
            else "Demo inspection finished with tool errors."
        )
        return FinalAnswer(
            f"{outcome} "
            f"Git status: {git_summary}. Repository map: {map_summary}. "
            f"TODO/FIXME search: {search_summary}."
        )


def _summarize_result(result: ToolResult, limit: int = 160) -> str:
    if result.ok:
        detail = result.output.strip() or "no output"
    else:
        detail = result.error or result.output.strip() or "tool failed"

    compact = " ".join(detail.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."
