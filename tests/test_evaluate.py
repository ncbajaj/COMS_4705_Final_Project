import unittest

from rarecrash.evaluate import distinct_n, retrieval_outputs, rouge_l_f1, score_row, summarize
from rarecrash.pipeline import parse_conditions


TEXT = ("The road surface was Wet with Daylight lighting conditions and Rain weather at the "
        "time of the crash. The crash involved 4 vehicles. The primary cause of the crash was "
        "Following Too Closely, with secondary contributing cause (None).")
RECORD = {"id": "1", "surface": "wet", "lighting": "daylight", "vehicles": "3+",
          "primary_cause": "Following Too Closely", "target": TEXT}


class ParseConditionsTests(unittest.TestCase):
    def test_parses_and_buckets_vehicle_count(self):
        self.assertEqual(parse_conditions(TEXT), {"surface": "wet", "lighting": "daylight",
                                                  "vehicles": "3+",
                                                  "primary_cause": "Following Too Closely"})

    def test_missing_and_out_of_schema_fields_are_none(self):
        parsed = parse_conditions("The road surface was Gravel with Daylight lighting conditions.")
        self.assertIsNone(parsed["surface"])
        self.assertEqual(parsed["lighting"], "daylight")
        self.assertIsNone(parsed["vehicles"])
        self.assertIsNone(parsed["primary_cause"])


class MetricTests(unittest.TestCase):
    def test_rouge_l(self):
        self.assertEqual(rouge_l_f1("a b c", "a b c"), 1.0)
        self.assertEqual(rouge_l_f1("x y", "a b"), 0.0)
        self.assertAlmostEqual(rouge_l_f1("a b", "a c b d"), 2 * 1.0 * 0.5 / 1.5)

    def test_distinct_n(self):
        self.assertEqual(distinct_n(["a a a"], 1), 1 / 3)
        self.assertEqual(distinct_n(["a b", "a b"], 2), 0.5)

    def test_score_row_flags_wrong_field_and_copy(self):
        wrong = TEXT.replace("Wet", "Dry")
        score = score_row(RECORD, wrong, {wrong})
        self.assertFalse(score["fields_correct"]["surface"])
        self.assertTrue(score["fields_correct"]["lighting"])
        self.assertFalse(score["all_fields_correct"])
        self.assertTrue(score["copied_from_train"])

    def test_cause_match_ignores_spacing_and_case(self):
        record = dict(RECORD, primary_cause="DisregardingOther Traffic Signs")
        text = TEXT.replace("Following Too Closely", "Disregarding Other Traffic Signs")
        self.assertTrue(score_row(record, text, set())["fields_correct"]["primary_cause"])

    def test_missing_output_counts_as_failure(self):
        score = score_row(RECORD, None, set())
        self.assertFalse(score["has_output"])
        self.assertFalse(score["all_fields_correct"])

    def test_within_tuple_distinct_rate(self):
        other = dict(RECORD, id="2")
        same = [{"record": RECORD, "output": TEXT}, {"record": other, "output": TEXT}]
        varied = [{"record": RECORD, "output": TEXT}, {"record": other, "output": TEXT + " x"}]
        for rows in (same, varied):
            for r in rows:
                r["score"] = score_row(r["record"], r["output"], set())
        self.assertEqual(summarize(same)["within_tuple_distinct_rate"], 0.5)
        self.assertEqual(summarize(varied)["within_tuple_distinct_rate"], 1.0)

    def test_retrieval_uses_first_training_narrative_per_tuple(self):
        train = [dict(RECORD, id="t1", target="first"), dict(RECORD, id="t2", target="second")]
        dev = [RECORD, dict(RECORD, id="2", surface="ice")]
        self.assertEqual(retrieval_outputs(train, dev), {"1": "first", "2": None})


if __name__ == "__main__":
    unittest.main()
