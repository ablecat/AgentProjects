from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
import unittest

from tests.support import load_models_module


class CaseInsensitiveHeaders(Mapping[str, str]):
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> str:
        for original, value in self._items:
            if original.casefold() == key.casefold():
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


class CustomHostRegressionTests(unittest.TestCase):
    def test_lowercase_custom_host_is_not_duplicated(self) -> None:
        module = load_models_module()
        for header_name in ("host", "hOsT", "Host"):
            with self.subTest(header_name=header_name):
                request = SimpleNamespace(
                    method="GET",
                    url="http://127.0.0.1/resource",
                    headers=CaseInsensitiveHeaders(
                        [(header_name, "service.example")]
                    ),
                    body=None,
                )
                rendered = module.HTTPRequest(request).headers.split("\r\n")
                host_lines = [
                    line
                    for line in rendered
                    if line.split(":", 1)[0].casefold() == "host"
                ]
                self.assertEqual(host_lines, [f"{header_name}: service.example"])


if __name__ == "__main__":
    unittest.main()
