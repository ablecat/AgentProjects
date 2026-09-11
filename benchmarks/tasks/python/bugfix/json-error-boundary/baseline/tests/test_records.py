import unittest

from tasklib.records import decode_record


class DecodeRecordTests(unittest.TestCase):
    def test_object_is_returned(self) -> None:
        self.assertEqual(decode_record('{"status": "ok"}'), {"status": "ok"})

    def test_malformed_json_has_stable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "record is not valid JSON"):
            decode_record("{")

    def test_non_object_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "record must be an object"):
            decode_record("[]")


if __name__ == "__main__":
    unittest.main()
