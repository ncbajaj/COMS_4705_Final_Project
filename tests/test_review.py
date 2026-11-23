import unittest

from rarecrash.review import cohen_kappa, parse_preference, parse_rating, sample_items, unblind


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
        self.assertEqual(parse_rating(" 4 "), 4)
        self.assertIsNone(parse_rating(""))
        with self.assertRaises(ValueError):
            parse_rating("6")
        self.assertEqual(parse_preference("b"), "B")
        self.assertEqual(parse_preference("Tie"), "tie")
        with self.assertRaises(ValueError):
            parse_preference("model")

    def test_unblind_maps_sides_to_systems(self):
        key = {"1": {"A": "retrieval", "B": "model"}}
        row = {"item": "1", "A_fluency": "2", "B_fluency": "5", "preference": "B"}
        judged = unblind([row], key)["1"]
        self.assertEqual(judged["preference"], "model")
        self.assertEqual(judged["ratings"]["model"]["fluency"], 5)
        self.assertEqual(judged["ratings"]["retrieval"]["fluency"], 2)
        self.assertIsNone(judged["ratings"]["model"]["plausibility"])

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)
        self.assertIsNone(cohen_kappa([None], ["a"]))


if __name__ == "__main__":
    unittest.main()
