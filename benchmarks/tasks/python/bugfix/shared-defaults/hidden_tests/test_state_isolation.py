import unittest

from tasklib.config import new_config
from tasklib.defaults import DEFAULT_LABELS


class StateIsolationTests(unittest.TestCase):
    def test_label_mutation_does_not_leak(self) -> None:
        first = new_config()
        first["labels"].append("urgent")

        self.assertEqual(new_config()["labels"], ["bug", "maintenance"])
        self.assertEqual(tuple(DEFAULT_LABELS), ("bug", "maintenance"))


if __name__ == "__main__":
    unittest.main()
