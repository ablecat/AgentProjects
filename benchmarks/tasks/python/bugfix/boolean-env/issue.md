# False environment values enable the feature

`tasklib.flags.parse_flag()` treats every non-empty string as true.  As a
result, common values such as `false`, `no`, and `0` unexpectedly enable the
feature.  The parser should be case-insensitive, ignore surrounding whitespace,
accept the documented true and false spellings, and reject any unknown value
with `ValueError`.

Implement the documented parsing behavior and add tests for the cases missing
from the current suite.  Keep the function dependency-free.
