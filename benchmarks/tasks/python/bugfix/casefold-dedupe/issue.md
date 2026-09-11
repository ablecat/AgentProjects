# Label merging keeps case-only duplicates

`tasklib.labels.merge_labels()` preserves the first occurrence and source
order, but it currently considers `Bug`, `bug`, and `BUG` to be different
labels.  Repository providers treat label identity case-insensitively, so these
variants must collapse to the first spelling encountered.

Fix duplicate detection while retaining the original spelling and order of
each first occurrence.  Do not sort the result or change the input sequences.
