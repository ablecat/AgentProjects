import unittest

from tasklib.report import Finding, render_text


class TextReportTests(unittest.TestCase):
    def test_text_report_remains_human_readable(self) -> None:
        findings = [Finding("src/app.py", 7, "warning", "Handle empty input")]
        self.assertEqual(
            render_text(findings),
            "src/app.py:7: WARNING: Handle empty input\n",
        )

    def test_empty_text_report_is_empty(self) -> None:
        self.assertEqual(render_text([]), "")


if __name__ == "__main__":
    unittest.main()
