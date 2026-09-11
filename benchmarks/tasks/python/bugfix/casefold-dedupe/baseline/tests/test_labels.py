import unittest

from tasklib.labels import merge_labels


class MergeLabelsTests(unittest.TestCase):
    def test_exact_duplicates_are_removed(self) -> None:
        self.assertEqual(merge_labels(["bug", "docs"], ["bug"]), ["bug", "docs"])

    def test_source_order_is_preserved(self) -> None:
        self.assertEqual(merge_labels(["zeta"], ["alpha"]), ["zeta", "alpha"])


if __name__ == "__main__":
    unittest.main()
