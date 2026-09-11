"""Bounded orchestration for provider decisions and tool execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import FinalAnswer, RunResult, ToolCall, ToolResult
from .providers import Provider


@runtime_checkable
class ToolExecutor(Protocol):
    """Execute a normalized tool call."""

    def execute(self, call: ToolCall) -> ToolResult:
        """Return a result even when a known or unknown tool fails."""


class AgentLoop:
    """Run a provider with bounded, diagnosable tool execution."""

    def __init__(
        self,
        provider: Provider,
        executor: ToolExecutor,
        max_steps: int = 8,
    ) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._provider = provider
        self._executor = executor
        self._max_steps = max_steps

    def run(self, task: str) -> RunResult:
        results: list[ToolResult] = []
        previous_calls: list[tuple[int, ToolCall]] = []

        for step in range(1, self._max_steps + 1):
            try:
                decision = self._provider.next_step(task, tuple(results))
            except Exception as exc:
                return RunResult(
                    task=task,
                    status="provider_error",
                    answer=f"Provider failed: {_exception_detail(exc)}",
                    steps=step,
                    tool_results=tuple(results),
                )

            if isinstance(decision, FinalAnswer):
                if not isinstance(decision.content, str) or not decision.content.strip():
                    return RunResult(
                        task=task,
                        status="provider_error",
                        answer="Provider returned an empty or non-text final answer",
                        steps=step,
                        tool_results=tuple(results),
                    )
                return RunResult(
                    task=task,
                    status=(
                        "completed"
                        if all(result.ok for result in results)
                        else "completed_with_errors"
                    ),
                    answer=decision.content,
                    steps=step,
                    tool_results=tuple(results),
                )

            if not isinstance(decision, ToolCall):
                return RunResult(
                    task=task,
                    status="provider_error",
                    answer=(
                        "Provider returned an unsupported decision: "
                        f"{type(decision).__name__}"
                    ),
                    steps=step,
                    tool_results=tuple(results),
                )

            workspace_revision = _workspace_revision(self._executor)
            if any(
                revision == workspace_revision
                and _same_invocation(decision, prior)
                for revision, prior in previous_calls
            ):
                duplicate = ToolResult(
                    call_id=decision.id,
                    name=decision.name,
                    ok=False,
                    output="",
                    error="Repeated tool call blocked: identical name and arguments",
                )
                results.append(duplicate)
                return RunResult(
                    task=task,
                    status="repeated_call",
                    answer=duplicate.error,
                    steps=step,
                    tool_results=tuple(results),
                )

            previous_calls.append((workspace_revision, decision))
            results.append(self._execute(decision))

        return RunResult(
            task=task,
            status="max_steps",
            answer=f"Maximum of {self._max_steps} steps reached before a final answer",
            steps=self._max_steps,
            tool_results=tuple(results),
        )

    def _execute(self, call: ToolCall) -> ToolResult:
        try:
            result = self._executor.execute(call)
        except Exception as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                output="",
                error=f"Tool executor failed: {_exception_detail(exc)}",
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                output="",
                error=(
                    "Tool executor returned an unsupported result: "
                    f"{type(result).__name__}"
                ),
            )

        if result.call_id != call.id or result.name != call.name:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                output="",
                error=(
                    "Tool executor returned a result for a different call "
                    f"(call_id={result.call_id!r}, name={result.name!r})"
                ),
            )

        return result


def _same_invocation(left: ToolCall, right: ToolCall) -> bool:
    """Compare semantic invocations while allowing providers to rotate call IDs."""

    try:
        return left.name == right.name and left.arguments == right.arguments
    except Exception:
        return left.name == right.name and repr(left.arguments) == repr(right.arguments)


def _workspace_revision(executor: ToolExecutor) -> int:
    """Treat executors without mutable-workspace support as revision zero."""

    revision = getattr(executor, "workspace_revision", 0)
    if type(revision) is not int or revision < 0:
        return 0
    return revision


def _exception_detail(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__
