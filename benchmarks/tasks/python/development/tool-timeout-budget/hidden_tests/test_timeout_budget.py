import unittest

from tasklib.tools import ToolResult, run_tool


class ScriptedClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class TimeoutBudgetTests(unittest.TestCase):
    def test_elapsed_budget_returns_explicit_timeout(self) -> None:
        result = run_tool(
            "search",
            lambda timeout: f"budget={timeout}",
            timeout_seconds=0.5,
            clock=ScriptedClock(10.0, 10.75),
        )
        self.assertEqual(
            result,
            ToolResult("search", "timed_out", "", "tool timed out"),
        )

    def test_timeout_exception_is_converted_to_a_result(self) -> None:
        def expire(_timeout: float) -> str:
            raise TimeoutError("implementation detail")

        result = run_tool(
            "read", expire, timeout_seconds=2.0, clock=ScriptedClock(3.0, 3.1)
        )
        self.assertEqual(result.status, "timed_out")
        self.assertEqual(result.error, "tool timed out")

    def test_invalid_budgets_are_rejected(self) -> None:
        for value in (True, 0, -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    run_tool("status", lambda timeout: "ok", timeout_seconds=value)


if __name__ == "__main__":
    unittest.main()
