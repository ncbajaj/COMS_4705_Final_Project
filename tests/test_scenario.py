import unittest

from rarecrash.diversity import score_sample, summarize
from rarecrash.events import (SCENARIO_SIGNATURE_FIELDS, SIGNATURE_FIELDS, parse_events,
                              signature)
from rarecrash.scenario import (FAMILIES, PHRASES, ScenarioPrior, audit, contingency,
                                parse_scenario, scenario_sentence, target_text, with_scenario)
from test_events import narrative


SAME = (("North", "traveling straight ahead"), ("North", "traveling straight ahead"))
OPPOSITE = (("North", "traveling straight ahead"), ("South", "traveling straight ahead"))
CROSSING = (("North", "traveling straight ahead"), ("East", "traveling straight ahead"))


def record(label, vehicles=SAME, bin_="2", cause="Speeding"):
    return {"surface": "wet", "lighting": "daylight", "vehicles": bin_, "primary_cause": cause,
            "collision_type_label": label, "target": narrative(vehicles)}


class SentenceTests(unittest.TestCase):
    def test_every_label_round_trips(self):
        for label in PHRASES:
            text = with_scenario(narrative(), label)
            self.assertEqual(parse_scenario(text), (label, PHRASES[label]))
        self.assertEqual(set(PHRASES), set(FAMILIES))

    def test_inserted_before_vehicle_sentences(self):
        text = with_scenario(narrative(), "FRONT TO REAR")
        self.assertIn("(Not Applicable). The manner of collision was front to rear.Vehicle 0, a",
                      text)

    def test_appended_without_vehicle_sentences(self):
        text = with_scenario(narrative((), stated=6), "TURNING")
        self.assertTrue(text.endswith(" The manner of collision was turning."))

    def test_other_event_fields_unchanged(self):
        base, scen = parse_events(narrative()), parse_events(with_scenario(narrative(), "TURNING"))
        for f in SIGNATURE_FIELDS + ("vehicles", "vehicle_count_consistent"):
            self.assertEqual(base[f], scen[f])
        self.assertIsNone(base["scenario"])
        self.assertEqual(scen["scenario"], "TURNING")

    def test_missing_and_unknown_phrases(self):
        self.assertEqual(parse_scenario("No scenario here."), (None, None))
        self.assertEqual(parse_scenario("The manner of collision was a drag race. Vehicle"),
                         (None, "a drag race"))
        self.assertEqual(parse_scenario("The manner of collision was Front To Rear."),
                         ("FRONT TO REAR", "Front To Rear"))

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            scenario_sentence("NOT A LABEL")

    def test_target_text_default_is_unchanged(self):
        r = record("TURNING")
        self.assertEqual(target_text(r), r["target"])
        self.assertIn("manner of collision was turning", target_text(r, scenario=True))

    def test_default_signature_ignores_scenario(self):
        e = parse_events(with_scenario(narrative(), "TURNING"))
        self.assertEqual(signature(e), signature(parse_events(narrative())))
        self.assertEqual(signature(e, SCENARIO_SIGNATURE_FIELDS)[-1], "TURNING")
        self.assertIsNone(signature(parse_events(narrative()), SCENARIO_SIGNATURE_FIELDS))


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.train = ([record("FRONT TO REAR", SAME)] * 49 + [record("FRONT TO REAR", CROSSING)]
                      + [record("ANGLE IMPACTS", CROSSING)] * 10
                      + [record("ANIMAL COLLISIONS", (("North", "traveling straight ahead"),),
                                "1")] * 5)
        self.prior = ScenarioPrior.fit(self.train, min_share=0.05)

    def test_contingency(self):
        table = contingency(self.train)
        self.assertEqual(table["FRONT TO REAR"], {"same": 49, "crossing": 1})
        self.assertEqual(table["ANIMAL COLLISIONS"], {"single": 5})

    def test_consistency(self):
        p = self.prior
        self.assertTrue(p.consistent("FRONT TO REAR", "same", "2"))
        self.assertFalse(p.consistent("FRONT TO REAR", "crossing", "2"))  # 2% < 5%
        self.assertFalse(p.consistent("FRONT TO REAR", "opposite"))
        self.assertTrue(p.consistent("ANGLE IMPACTS", "crossing"))
        self.assertFalse(p.consistent("ANIMAL COLLISIONS", "single", "2"))  # vehicle bin
        self.assertTrue(p.consistent("ANIMAL COLLISIONS", "single", "1"))

    def test_unjudged(self):
        self.assertTrue(self.prior.consistent("FRONT TO REAR", "unknown", "2"))
        self.assertTrue(self.prior.consistent("FRONT TO REAR", "none"))
        self.assertIsNone(self.prior.consistent(None, "same"))
        self.assertIsNone(self.prior.consistent("TURNING", "same"))  # not in training

    def test_audit_runs(self):
        report = audit(self.train, self.train[:5])
        self.assertEqual(report["sentence_round_trip_rate"], {"train": 1.0, "dev": 1.0})
        self.assertEqual(report["labels_without_phrase"], [])
        self.assertEqual(report["collision_type_by_heading_relation_train"]["ANGLE IMPACTS"]["n"],
                         10)
        self.assertEqual(report["dev_consistency_true_label"], 1.0)


class DiversityScenarioTests(unittest.TestCase):
    def test_score_and_summarize_with_scenario(self):
        train = [record("FRONT TO REAR", SAME)] * 10 + [record("FRONT TO FRONT", OPPOSITE)] * 10
        prior = ScenarioPrior.fit(train)
        conditions = {"surface": "wet", "lighting": "daylight", "vehicles": "2",
                      "primary_cause": "Speeding"}
        texts = [with_scenario(narrative(SAME), "FRONT TO REAR"),
                 with_scenario(narrative(OPPOSITE), "FRONT TO REAR"),
                 narrative(SAME)]
        scored = [{**score_sample(conditions, t, set(), _empty(), set(),
                                  SCENARIO_SIGNATURE_FIELDS, prior), "text": t}
                  for t in texts]
        self.assertEqual([s["scenario_consistent"] for s in scored], [True, False, None])
        self.assertIsNone(scored[2]["signature"])
        summary = summarize([scored], SCENARIO_SIGNATURE_FIELDS)
        self.assertAlmostEqual(summary["scenario_parse_rate"], 2 / 3)
        self.assertEqual(summary["scenario_consistency_rate"], 0.5)
        self.assertEqual(summary["mean_distinct_scenarios_per_tuple"], 1)
        self.assertIn("scenario", summary["field_variation"])

    def test_default_scoring_has_no_scenario_keys(self):
        conditions = {"surface": "wet", "lighting": "daylight", "vehicles": "2",
                      "primary_cause": "Speeding"}
        s = {**score_sample(conditions, narrative(), set(), _empty(), set()), "text": narrative()}
        self.assertNotIn("scenario_consistent", s)
        self.assertEqual(set(s["events"]), set(SIGNATURE_FIELDS))
        summary = summarize([[s]])
        self.assertNotIn("scenario_parse_rate", summary)
        self.assertEqual(set(summary["field_variation"]), set(SIGNATURE_FIELDS))


def _empty():
    from collections import defaultdict
    return defaultdict(set)


if __name__ == "__main__":
    unittest.main()
