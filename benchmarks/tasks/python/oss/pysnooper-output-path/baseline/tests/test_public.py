from __future__ import annotations

import io
import sys
import unittest
from unittest import mock

from tests.support import load_pysnooper_module


class WriteFunctionSmokeTests(unittest.TestCase):
    def test_writable_stream_receives_text(self) -> None:
        module = load_pysnooper_module()
        output = io.StringIO()
        module.get_write_function(output)("trace line")
        self.assertEqual(output.getvalue(), "trace line")

    def test_default_output_uses_current_stderr(self) -> None:
        module = load_pysnooper_module()
        output = io.StringIO()
        with mock.patch.object(sys, "stderr", output):
            module.get_write_function(None)("diagnostic")
        self.assertEqual(output.getvalue(), "diagnostic")


if __name__ == "__main__":
    unittest.main()
