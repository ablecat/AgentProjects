# Retry delays lag one attempt behind

`Backoff.delayMillis()` returns half the expected delay for every retry after
the initial attempt. Attempt zero must use the base delay and attempt `n` must
use `base * 2^n`, while the existing validation rules remain unchanged.

Fix the calculation and add a focused regression test for a later attempt.
Do not introduce a scheduling or arithmetic dependency.
