# Add a stable JSON report exporter

The finding reporter currently supports a human-readable text format only.
Machine consumers need a deterministic JSON representation, while existing
text output must remain byte-for-byte unchanged.

Add `tasklib.report.render_json(findings)`. It must return one compact JSON
object followed by a newline, with `schema_version` set to `1` and a `findings`
array in the same order as the input. Each finding object contains exactly
`message`, `path`, `severity`, and `line`; object keys must be sorted, non-ASCII
text must remain readable, and non-finite or otherwise unsupported values must
not be emitted. Do not add a third-party serialization dependency.
