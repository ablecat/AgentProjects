# Add severity summaries

Finding output is currently rendered one item at a time. Add
`tasklib.findings.summarize_by_severity(findings)` so callers can display a
compact summary without reordering the underlying evidence.

Return one immutable `SeveritySummary` per distinct severity. Summary groups
must appear in the order each severity is first encountered, and the findings
inside each group must retain their original source order. Each summary exposes
the normalized lowercase severity, its count, and a tuple of the original
`Finding` objects. Severity matching is case-insensitive and surrounding
whitespace is ignored. Reject a blank severity with `ValueError`; an empty input
returns an empty tuple. Preserve the existing finding formatter.
