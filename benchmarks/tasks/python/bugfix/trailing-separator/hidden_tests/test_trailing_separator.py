import unittest

from tasklib.text import slugify


class TrailingSeparatorTests(unittest.TestCase):
    def test_trailing_separators(self) -> None:
        self.assertEqual(slugify("  Release candidate!!!  "), "release-candidate")


if __name__ == "__main__":
    unittest.main()
