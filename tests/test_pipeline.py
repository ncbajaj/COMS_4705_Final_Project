import unittest

from rarecrash.pipeline import condition_key, extract, split_for_id


EXAMPLE = """<s>Human: This crash occurred in Cook County. The road surface was Wet with Daylight lighting conditions and Rain weather at the time of the crash. The crash involved 2 vehicles. The primary cause of the crash was Following Too Closely, with secondary contributing cause (None). Vehicle 0 was moving East. The driver was 30 years old.</s><s>Assistant: <FRONT TO REAR><\\s>"""


class PipelineTests(unittest.TestCase):
    def test_extract_keeps_label_as_metadata_only(self):
        record, error = extract({"ID": "123", "text": EXAMPLE}, "train")
        self.assertIsNone(error)
        self.assertEqual(condition_key(record), ("wet", "daylight", "2", "Following Too Closely"))
        self.assertEqual(record["collision_type_label"], "FRONT TO REAR")
        self.assertNotIn("Assistant", record["target"])
        self.assertNotIn("30 years old", record["target"])

    def test_unknown_surface_is_excluded(self):
        _, error = extract({"ID": "123", "text": EXAMPLE.replace("Wet", "Unknown")}, "train")
        self.assertEqual(error, "outside_schema")

    def test_split_is_stable(self):
        self.assertEqual(split_for_id("40951"), split_for_id("40951"))
        self.assertIn(split_for_id("40951"), {"train", "dev", "test"})


if __name__ == "__main__":
    unittest.main()
