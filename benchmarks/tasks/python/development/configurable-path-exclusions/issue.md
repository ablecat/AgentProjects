# Add configurable path exclusions

Repository scans currently normalize, deduplicate, and sort their input paths,
but callers cannot exclude generated or irrelevant files.

Extend `tasklib.scanner.collect_paths` with an optional keyword-only
`exclusions` argument. Each exclusion is a repository-relative POSIX glob.
`*`, `?`, character classes, and a complete `**` path segment must be
supported. Matching is case-sensitive, and the result must remain sorted and
deduplicated regardless of input order.

Reject an exclusion that is empty, absolute, contains a backslash, contains an
empty, `.` or `..` segment, or embeds `**` inside another segment. Raise
`ValueError` before returning a partial result. Preserve all existing behavior
when no exclusions are supplied, and do not add third-party dependencies.
