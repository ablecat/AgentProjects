import unittest

from tasklib.findings import Finding, format_findings


class FindingFormatTests(unittest.TestCase):
    def test_findings_remain_in_source_order(self) -> None:
        findings = [
            Finding("z.py:9", "warning", "late file"),
            Finding("a.py:1", "error", "early file"),
        ]
        self.assertEqual(
            format_findings(findings),
            "z.py:9: warning: late file\na.py:1: error: early file\n",
        )


if __name__ == "__main__":
    unittest.main()
