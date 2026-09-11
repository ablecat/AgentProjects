import json
import unittest

from tasklib import report
from tasklib.report import Finding, render_text


class JsonReportTests(unittest.TestCase):
    def test_json_is_stable_and_preserves_finding_order(self) -> None:
        findings = [
            Finding("src/z.py", 9, "warning", "Use cafe fallback"),
            Finding("src/a.py", 2, "error", "Reject an empty value"),
        ]
        expected = (
            '{"findings":['
            '{"line":9,"message":"Use cafe fallback","path":"src/z.py","severity":"warning"},'
            '{"line":2,"message":"Reject an empty value","path":"src/a.py","severity":"error"}'
            '],"schema_version":1}\n'
        )
        self.assertEqual(report.render_json(findings), expected)
        self.assertEqual(json.loads(report.render_json(findings))["schema_version"], 1)

    def test_text_report_is_unchanged_after_json_export(self) -> None:
        finding = Finding("app.py", 1, "info", "Ready")
        before = render_text([finding])
        report.render_json([finding])
        self.assertEqual(render_text([finding]), before)

    def test_non_finite_values_are_not_emitted(self) -> None:
        invalid = Finding("app.py", float("nan"), "error", "invalid line")
        with self.assertRaises(ValueError):
            report.render_json([invalid])


if __name__ == "__main__":
    unittest.main()
