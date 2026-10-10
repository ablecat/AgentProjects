from __future__ import annotations

from types import SimpleNamespace
import unittest

from tests.support import load_models_module


class RequestModelSmokeTests(unittest.TestCase):
    def test_url_host_is_added_when_header_is_absent(self) -> None:
        module = load_models_module()
        request = SimpleNamespace(
            method="GET",
            url="https://example.test:8443/path?q=1",
            headers={"Accept": "application/json"},
            body=None,
        )
        rendered = module.HTTPRequest(request).headers.split("\r\n")
        self.assertEqual(rendered[0], "GET /path?q=1 HTTP/1.1")
        self.assertIn("Host: example.test:8443", rendered)

    def test_text_body_is_encoded(self) -> None:
        module = load_models_module()
        request = SimpleNamespace(headers={}, body="hello")
        self.assertEqual(module.HTTPRequest(request).body, b"hello")


if __name__ == "__main__":
    unittest.main()
