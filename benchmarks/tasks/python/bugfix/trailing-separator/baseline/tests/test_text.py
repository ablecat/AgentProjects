import unittest

from tasklib.text import slugify


class SlugifyTests(unittest.TestCase):
    def test_words_are_normalized(self) -> None:
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_clean_slug_is_unchanged(self) -> None:
        self.assertEqual(slugify("already-clean"), "already-clean")


if __name__ == "__main__":
    unittest.main()
