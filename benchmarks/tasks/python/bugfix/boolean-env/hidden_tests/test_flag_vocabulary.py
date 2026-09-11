import unittest

from tasklib.flags import parse_flag


class FlagVocabularyTests(unittest.TestCase):
    def test_false_words_are_false(self) -> None:
        for value in ("false", "FALSE", "no", "off", "0"):
            with self.subTest(value=value):
                self.assertFalse(parse_flag(value))

    def test_unknown_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_flag("sometimes")

    def test_whitespace_is_empty(self) -> None:
        self.assertFalse(parse_flag("   "))


if __name__ == "__main__":
    unittest.main()
