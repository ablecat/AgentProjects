# Evaluation errata

## v1 pytest subtest attribution

The published `benchmarks/results/v1` files are immutable historical artifacts
and have not been rewritten. Their source-of-truth SHA-256 remains
`1f083cdf6ce1630ec50da47cadf3b16a7e628c09c23f10e697bc407549579761`.

The original scorer recognized pytest `FAILED` lines but not `SUBFAILED` lines.
For the `full`, trial-1 `py-bugfix-005` candidate, the public and hidden checks
passed, the required regression test was present and passed on the candidate,
and policy checks passed. Its buggy-state regression run reported pytest
subtest failures, but v1 recorded `regression_failed_on_buggy=false` and
classified the otherwise passing candidate as `regression_not_reproduced`.

With only that attribution corrected, the historical trial-1 `full` result is
equivalent to 2/12 overall and 2/6 Python, rather than the published 1/12 and
1/6. It still misses the predeclared 8/12, Python 4/6, and Java 4/6 release
thresholds. This correction does not turn v1 into an independent holdout for
later prompt or workflow changes.

Runs produced after this fix must use a new output directory and be described
as **v1 regression acceptance**, not as a rewrite of v1 or evidence that one
architecture is generally superior.
