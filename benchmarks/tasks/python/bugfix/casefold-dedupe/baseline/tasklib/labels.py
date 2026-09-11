from collections.abc import Iterable


def merge_labels(primary: Iterable[str], extra: Iterable[str]) -> list[str]:
    """Merge labels in source order, preserving the first display spelling."""
    result: list[str] = []
    seen: set[str] = set()
    for label in [*primary, *extra]:
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(label)
    return result
