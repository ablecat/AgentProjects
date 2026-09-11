import unittest

from tasklib.tools import ToolResult, run_tool


class ToolExecutionTests(unittest.TestCase):
    def test_action_receives_default_timeout(self) -> None:
        seen: list[float] = []
        result = run_tool("status", lambda timeout: seen.append(timeout) or "clean")
        self.assertEqual(seen, [30.0])
        self.assertEqual(result, ToolResult("status", "completed", "clean"))


if __name__ == "__main__":
    unittest.main()
