# Custom Host header is duplicated when its casing differs

Upstream issue: https://github.com/httpie/cli/issues/235

Rendering a request with a custom lowercase `host` header adds a second `Host`
header derived from the URL. The original request header mapping performs
case-insensitive membership checks, but converting it to a plain dictionary
before checking loses that behavior.

Keep exactly one caller-provided Host header regardless of its casing, while
still adding the URL host when no Host header was supplied. Add a focused
regression test and do not add a network dependency.
