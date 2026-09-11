# Page zero produces a negative offset

`Pages.offset()` uses one-based page numbers but currently accepts page zero,
returning a negative database offset. Page number and page size must both be
strictly positive. Valid page calculations and overflow behavior must remain
unchanged.

Reject the zero boundary and add a regression test that captures it.
