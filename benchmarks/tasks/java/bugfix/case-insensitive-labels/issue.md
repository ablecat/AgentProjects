# Label uniqueness changes with letter case

`Labels.uniqueNormalized()` currently treats `Bug` and `bug` as distinct even
though downstream label identity is case-insensitive. The method must keep the
first spelling, preserve the order of first occurrence, trim surrounding
whitespace, and omit blank labels.

Restore the case-insensitive uniqueness invariant and add focused regression
coverage. Do not sort the result or add a collection dependency.
