import unittest

from tasklib.labels import merge_labels


class CasefoldDedupeTests(unittest.TestCase):
    def test_case_variants_are_duplicates(self) -> None:
        self.assertEqual(
            merge_labels(["Bug", "docs"], ["bug", "DOCS", "feature"]),
            ["Bug", "docs", "feature"],
        )


if __name__ == "__main__":
    unittest.main()
