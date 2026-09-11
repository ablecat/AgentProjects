import unittest

from tasklib.window import take_window


class TakeWindowTests(unittest.TestCase):
    def test_middle_window(self) -> None:
        self.assertEqual(take_window(["a", "b", "c"], 1, 2), ["b", "c"])

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            take_window(["a"], -1, 1)
        with self.assertRaises(ValueError):
            take_window(["a"], 0, 0)
        with self.assertRaises(IndexError):
            take_window(["a"], 2, 1)


if __name__ == "__main__":
    unittest.main()
