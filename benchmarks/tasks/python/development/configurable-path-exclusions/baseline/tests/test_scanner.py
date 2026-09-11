import unittest

from tasklib.scanner import collect_paths


class CollectPathsTests(unittest.TestCase):
    def test_paths_are_normalized_deduplicated_and_sorted(self) -> None:
        self.assertEqual(
            collect_paths(["./src/z.py", "src/a.py", "src/a.py", "README.md"]),
            ["README.md", "src/a.py", "src/z.py"],
        )

    def test_empty_paths_are_ignored(self) -> None:
        self.assertEqual(collect_paths(["", "app.py"]), ["app.py"])


if __name__ == "__main__":
    unittest.main()
