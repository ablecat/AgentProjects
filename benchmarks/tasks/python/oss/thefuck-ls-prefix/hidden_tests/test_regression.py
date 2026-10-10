from __future__ import annotations

from types import SimpleNamespace
import unittest

from thefuck.rules.ls_lah import match


class LsRuleBoundaryTests(unittest.TestCase):
    def test_commands_containing_ls_do_not_match(self) -> None:
        for script in ("pacman -S binutils", "lsof", "printf false"):
            with self.subTest(script=script):
                self.assertFalse(match(SimpleNamespace(script=script), None))


if __name__ == "__main__":
    unittest.main()
