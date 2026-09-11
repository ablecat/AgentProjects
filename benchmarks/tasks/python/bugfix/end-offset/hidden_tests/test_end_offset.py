import unittest

from tasklib.window import take_window


class EndOffsetTests(unittest.TestCase):
    def test_offset_at_end(self) -> None:
        self.assertEqual(take_window(["a", "b"], 2, 3), [])

    def test_zero_offset_for_empty_input(self) -> None:
        self.assertEqual(take_window([], 0, 1), [])


if __name__ == "__main__":
    unittest.main()
