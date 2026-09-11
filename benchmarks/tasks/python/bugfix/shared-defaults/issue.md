# Configuration instances share label state

Two calls to `tasklib.config.new_config()` should return independent mutable
configuration dictionaries.  Currently, appending a label to one result also
changes later results and the defaults exported by `tasklib.defaults`.  This
causes one repository scan to affect the next scan in the same process.

Restore per-call isolation while keeping `config["labels"]` as a list and
preserving the declared default values.  Avoid a broad deep copy when a focused
construction is sufficient.
