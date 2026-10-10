# NameError when trace output is a file path

Upstream issue: https://github.com/cool-RR/PySnooper/issues/2

Passing a string or `os.PathLike` output destination to PySnooper fails when
the returned writer is called. Instead of appending trace text to that path,
the implementation references an undefined name and raises `NameError`.

Fix path-based output without changing the existing behavior for stderr or
writable stream objects. Add a focused regression test for the failure.
