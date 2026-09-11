from __future__ import annotations

from collections.abc import Sequence

import pytest

from repo_agent.loop import AgentLoop
from repo_agent.models import FinalAnswer, ToolCall, ToolResult


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            output=f"ran {call.name}",
        )


class TwoStepProvider:
    def next_step(
        self, task: str, results: Sequence[ToolResult]
    ) -> ToolCall | FinalAnswer:
        if not results:
            return ToolCall("call-1", "git_status", {})
        return FinalAnswer(f"finished {task} from {results[0].output}")


def test_loop_completes_with_structured_history() -> None:
    executor = RecordingExecutor()

    result = AgentLoop(TwoStepProvider(), executor).run("inspect repository")

    assert result.status == "completed"
    assert result.answer == "finished inspect repository from ran git_status"
    assert result.steps == 2
    assert result.tool_results == (
        ToolResult("call-1", "git_status", True, "ran git_status"),
    )
    assert executor.calls == [ToolCall("call-1", "git_status", {})]


def test_failed_tool_result_is_returned_to_provider() -> None:
    class FailingExecutor:
        def execute(self, call: ToolCall) -> ToolResult:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                output="",
                error="unknown tool",
            )

    class RecoveryProvider:
        def next_step(
            self, task: str, results: Sequence[ToolResult]
        ) -> ToolCall | FinalAnswer:
            if not results:
                return ToolCall("missing-1", "missing", {})
            assert results[0].ok is False
            return FinalAnswer(f"could not inspect: {results[0].error}")

    result = AgentLoop(RecoveryProvider(), FailingExecutor()).run("inspect")

    assert result.status == "completed_with_errors"
    assert result.answer == "could not inspect: unknown tool"
    assert result.steps == 2
    assert result.tool_results[0].error == "unknown tool"


def test_loop_stops_at_max_steps() -> None:
    class AlwaysCallingProvider:
        def next_step(
            self, task: str, results: Sequence[ToolResult]
        ) -> ToolCall:
            index = len(results)
            return ToolCall(f"call-{index}", "search", {"pattern": str(index)})

    executor = RecordingExecutor()
    result = AgentLoop(AlwaysCallingProvider(), executor, max_steps=2).run("inspect")

    assert result.status == "max_steps"
    assert result.answer == "Maximum of 2 steps reached before a final answer"
    assert result.steps == 2
    assert len(result.tool_results) == 2
    assert len(executor.calls) == 2


def test_loop_blocks_same_invocation_even_with_new_call_id() -> None:
    class RepeatingProvider:
        def next_step(
            self, task: str, results: Sequence[ToolResult]
        ) -> ToolCall:
            return ToolCall(
                f"call-{len(results)}",
                "search",
                {"pattern": "TODO", "paths": ["src", "tests"]},
            )

    executor = RecordingExecutor()
    result = AgentLoop(RepeatingProvider(), executor).run("inspect")

    assert result.status == "repeated_call"
    assert result.steps == 2
    assert len(executor.calls) == 1
    assert len(result.tool_results) == 2
    assert result.tool_results[-1].ok is False
    assert "Repeated tool call blocked" in (result.tool_results[-1].error or "")


def test_provider_exception_is_a_structured_error() -> None:
    class ExplodingProvider:
        def next_step(
            self, task: str, results: Sequence[ToolResult]
        ) -> ToolCall | FinalAnswer:
            raise RuntimeError("model unavailable")

    result = AgentLoop(ExplodingProvider(), RecordingExecutor()).run("inspect")

    assert result.status == "provider_error"
    assert result.answer == "Provider failed: RuntimeError: model unavailable"
    assert result.steps == 1
    assert result.tool_results == ()


def test_executor_exception_becomes_a_failed_result() -> None:
    class ExplodingExecutor:
        def execute(self, call: ToolCall) -> ToolResult:
            raise OSError("process launch failed")

    result = AgentLoop(TwoStepProvider(), ExplodingExecutor()).run("inspect")

    assert result.status == "completed_with_errors"
    assert result.tool_results[0].ok is False
    assert result.tool_results[0].error == (
        "Tool executor failed: OSError: process launch failed"
    )


def test_invalid_provider_decision_is_a_structured_error() -> None:
    class InvalidProvider:
        def next_step(self, task: str, results: Sequence[ToolResult]) -> object:
            return {"tool": "search"}

    result = AgentLoop(InvalidProvider(), RecordingExecutor()).run("inspect")  # type: ignore[arg-type]

    assert result.status == "provider_error"
    assert result.answer == "Provider returned an unsupported decision: dict"
    assert result.steps == 1


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5, "2"])
def test_max_steps_must_be_a_positive_integer(max_steps: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AgentLoop(TwoStepProvider(), RecordingExecutor(), max_steps=max_steps)  # type: ignore[arg-type]


@pytest.mark.parametrize("content", ["", "   ", None, 42])
def test_final_answer_must_be_nonempty_text(content: object) -> None:
    class InvalidFinalProvider:
        def next_step(self, task: str, results: Sequence[ToolResult]) -> FinalAnswer:
            return FinalAnswer(content)  # type: ignore[arg-type]

    result = AgentLoop(InvalidFinalProvider(), RecordingExecutor()).run("inspect")

    assert result.status == "provider_error"
    assert result.answer == "Provider returned an empty or non-text final answer"
