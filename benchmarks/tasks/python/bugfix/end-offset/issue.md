# End offsets are rejected

`tasklib.window.take_window()` raises `IndexError` when `offset` is exactly the
number of available values.  That offset is valid and should return an empty
list, matching normal Python slicing and allowing callers to request the page
after the final item.  Offsets greater than the collection length must continue
to raise, as must negative offsets and non-positive limits.

Correct the boundary check without changing the return type or validation for
the other invalid inputs.
