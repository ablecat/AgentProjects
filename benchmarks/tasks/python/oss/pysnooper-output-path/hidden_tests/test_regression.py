from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.support import load_pysnooper_module


class PathOutputRegressionTests(unittest.TestCase):
    def test_path_output_appends_text(self) -> None:
        module = load_pysnooper_module()
        with tempfile.TemporaryDirectory() as temporary:
            for kind in ("string", "pathlike"):
                with self.subTest(kind=kind):
                    output = Path(temporary) / f"trace-{kind}.log"
                    destination = str(output) if kind == "string" else output
                    write = module.get_write_function(destination)
                    write("first\n")
                    write("second\n")
                    self.assertEqual(
                        output.read_text(encoding="utf-8"),
                        "first\nsecond\n",
                    )


if __name__ == "__main__":
    unittest.main()
