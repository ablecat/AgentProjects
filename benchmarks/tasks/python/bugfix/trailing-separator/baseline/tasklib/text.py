import re


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a lowercase, URL-safe identifier."""
    normalized = _NON_ALNUM.sub("-", value.casefold())
    return normalized.strip("-")
