import unittest

from tasklib.scanner import collect_paths


class ExclusionTests(unittest.TestCase):
    def test_globs_are_deterministic_and_segment_aware(self) -> None:
        paths = [
            "src/cache/keep.py",
            "build/output.js",
            "src/app.py",
            "src/app.tmp",
            "root.tmp",
            "docs/generated/index.md",
        ]
        self.assertEqual(
            collect_paths(
                reversed(paths),
                exclusions=("build/**", "**/generated/**", "*.tmp"),
            ),
            ["src/app.py", "src/app.tmp", "src/cache/keep.py"],
        )

    def test_invalid_patterns_are_rejected(self) -> None:
        invalid = ("", "/tmp/**", "src\\*.py", "src/../*.py", "src/**.py")
        for pattern in invalid:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    collect_paths(["src/app.py"], exclusions=(pattern,))


if __name__ == "__main__":
    unittest.main()
