# Slugs retain a trailing separator

`tasklib.text.slugify()` returns a slug ending in `-` when the input ends with
spaces or punctuation.  Callers use these values as stable report identifiers,
so leading and trailing separators must both be removed.  Interior runs of
non-alphanumeric characters should still collapse to one separator, and the
existing behavior for already-clean values must remain unchanged.

Fix the implementation and add focused regression coverage.  Do not add a
third-party slugification dependency.
