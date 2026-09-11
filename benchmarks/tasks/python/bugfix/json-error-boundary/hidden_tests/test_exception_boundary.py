import unittest
from unittest.mock import patch

from tasklib.records import decode_record


class ExceptionBoundaryTests(unittest.TestCase):
    def test_decoder_runtime_error_is_not_masked(self) -> None:
        with patch("tasklib.records.json.loads", side_effect=RuntimeError("decoder unavailable")):
            with self.assertRaisesRegex(RuntimeError, "decoder unavailable"):
                decode_record("{}")

    def test_invalid_input_type_is_not_masked(self) -> None:
        with self.assertRaises(TypeError):
            decode_record(None)


if __name__ == "__main__":
    unittest.main()
