import unittest

from rarecrash.review import (cohen_kappa, parse_details, parse_preference, parse_yes_no,
                              sample_items, sheet_fields, unblind)


def prediction(crash_id, system, surface="wet", rare=True, output="text"):
    return {"id": crash_id, "system": system, "rare": rare, "output": output,
            "conditions": {"surface": surface, "lighting": "daylight", "vehicles": "2",
                           "primary_cause": "Following Too Closely"}}


class SampleItemsTests(unittest.TestCase):
    def test_one_item_per_condition_tuple(self):
        preds = [prediction(i, s, surface) for i, surface in
                 (("1", "wet"), ("2", "wet"), ("3", "ice")) for s in ("model", "retrieval")]
        items = sample_items(preds, 10, seed=1)
        self.assertEqual(len(items), 2)
        self.assertEqual({it["conditions"]["surface"] for it in items}, {"wet", "ice"})

    def test_skips_common_and_missing_outputs(self):
        preds = [prediction("1", "model", rare=False), prediction("1", "retrieval", rare=False),
                 prediction("2", "model", "ice"), prediction("2", "retrieval", "ice", output=None)]
        self.assertEqual(sample_items(preds, 10, seed=1), [])

    def test_sides_are_a_permutation_and_seeded(self):
        preds = [prediction(str(i), s, surface=f"s{i}") for i in range(20)
                 for s in ("model", "retrieval")]
        items = sample_items(preds, 20, seed=7)
        self.assertEqual(items, sample_items(preds, 20, seed=7))
        self.assertTrue(all({it["A"], it["B"]} == {"model", "retrieval"} for it in items))
        self.assertEqual(len({it["A"] for it in items}), 2)  # both orders occur


class ScoringTests(unittest.TestCase):
    def test_parsers(self):
        for v in ("yes", "Y", "1", "TRUE", " true "):
            self.assertIs(parse_yes_no(v), True)
        for v in ("no", "N", "0", "False"):
            self.assertIs(parse_yes_no(v), False)
        self.assertIsNone(parse_yes_no(""))
        self.assertIsNone(parse_yes_no(None))
        with self.assertRaises(ValueError):
            parse_yes_no("maybe")
        for v, want in (("coherent", "coherent"), ("C", "coherent"), ("IC", "contradictory"),
                        ("Contradictory", "contradictory"), ("imp", "implausible"),
                        ("Implausible", "implausible")):
            self.assertEqual(parse_details(v), want)
        self.assertIsNone(parse_details(" "))
        for bad in ("i", "x", "fine"):
            with self.assertRaises(ValueError):
                parse_details(bad)
        self.assertEqual(parse_preference("b"), "B")
        self.assertEqual(parse_preference("Tie"), "tie")
        with self.assertRaises(ValueError):
            parse_preference("model")

    def test_sheet_columns(self):
        self.assertEqual(sheet_fields(), [
            "item", "surface", "lighting", "vehicles", "primary_cause", "output_A", "output_B",
            "A_condition_contradiction", "A_details", "B_condition_contradiction", "B_details",
            "preference", "notes"])

    def test_unblind_maps_sides_to_systems(self):
        key = {"1": {"A": "retrieval", "B": "model"}}
        row = {"item": "1", "A_condition_contradiction": "yes", "A_details": "imp",
               "B_condition_contradiction": "n", "B_details": "", "preference": "B"}
        judged = unblind([row], key)["1"]
        self.assertEqual(judged["preference"], "model")
        self.assertIs(judged["condition_contradiction"]["retrieval"], True)
        self.assertIs(judged["condition_contradiction"]["model"], False)
        self.assertEqual(judged["details"]["retrieval"], "implausible")
        self.assertIsNone(judged["details"]["model"])

    def test_unblind_rejects_bad_label(self):
        key = {"1": {"A": "model", "B": "retrieval"}}
        with self.assertRaises(ValueError):
            unblind([{"item": "1", "A_details": "great"}], key)

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)
        self.assertIsNone(cohen_kappa([None], ["a"]))


if __name__ == "__main__":
    unittest.main()
