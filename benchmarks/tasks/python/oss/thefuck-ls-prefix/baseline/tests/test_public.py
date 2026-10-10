from __future__ import annotations

from types import SimpleNamespace
import unittest

from thefuck.rules.ls_lah import get_new_command, match


class LsRuleSmokeTests(unittest.TestCase):
    def test_plain_ls_and_path_match(self) -> None:
        self.assertTrue(match(SimpleNamespace(script="ls"), None))
        self.assertTrue(match(SimpleNamespace(script="ls /opt"), None))

    def test_existing_options_do_not_match(self) -> None:
        self.assertFalse(match(SimpleNamespace(script="ls -lah /opt"), None))

    def test_rewrite_preserves_arguments(self) -> None:
        command = SimpleNamespace(script="ls example.txt")
        self.assertEqual(get_new_command(command, None), "ls -lah example.txt")


if __name__ == "__main__":
    unittest.main()
