import unittest

from tasklib.config import new_config


class NewConfigTests(unittest.TestCase):
    def test_default_values(self) -> None:
        self.assertEqual(new_config(), {"labels": ["bug", "maintenance"], "retries": 2})

    def test_labels_are_mutable(self) -> None:
        config = new_config()
        config["labels"].append("urgent")
        self.assertEqual(config["labels"][-1], "urgent")


if __name__ == "__main__":
    unittest.main()
