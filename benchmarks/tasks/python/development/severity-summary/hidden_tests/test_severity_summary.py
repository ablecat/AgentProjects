import unittest

from tasklib import findings
from tasklib.findings import Finding


class SeveritySummaryTests(unittest.TestCase):
    def test_groups_and_findings_preserve_source_order(self) -> None:
        first = Finding("b.py:8", " Warning ", "first")
        second = Finding("a.py:2", "ERROR", "second")
        third = Finding("c.py:4", "warning", "third")
        self.assertEqual(
            findings.summarize_by_severity([first, second, third]),
            (
                findings.SeveritySummary("warning", 2, (first, third)),
                findings.SeveritySummary("error", 1, (second,)),
            ),
        )

    def test_empty_input_and_blank_severity(self) -> None:
        self.assertEqual(findings.summarize_by_severity([]), ())
        with self.assertRaises(ValueError):
            findings.summarize_by_severity([Finding("app.py:1", "  ", "invalid")])


if __name__ == "__main__":
    unittest.main()
