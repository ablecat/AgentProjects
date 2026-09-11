from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def take_window(values: Sequence[T], offset: int, limit: int) -> list[T]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit < 1:
        raise ValueError("limit must be positive")
    if offset > len(values):
        raise IndexError("offset is past the end")
    return list(values[offset : offset + limit])
