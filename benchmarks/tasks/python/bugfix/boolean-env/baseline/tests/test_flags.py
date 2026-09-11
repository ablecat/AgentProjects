import unittest

from tasklib.flags import parse_flag


class ParseFlagTests(unittest.TestCase):
    def test_one_is_true(self) -> None:
        self.assertTrue(parse_flag("1"))

    def test_empty_is_false(self) -> None:
        self.assertFalse(parse_flag(""))


if __name__ == "__main__":
    unittest.main()
