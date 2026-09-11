# Malformed Retry-After values are silently accepted

`RetryAfterParser.retryAfterSeconds()` returns the fallback when a present
`Retry-After` header is not an integer. Missing headers should use the fallback,
but malformed values must be rejected with `IllegalArgumentException` so callers
do not mistake invalid upstream data for an ordinary default.

Fix the exception boundary and add focused regression coverage. Keep valid and
missing-header behavior unchanged, and do not add a runtime dependency.
