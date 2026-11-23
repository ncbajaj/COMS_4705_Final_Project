import unittest
from collections import Counter

from rarecrash.pipeline import condition_key
from rarecrash.train import CheckpointSelector, dev_subsets, eval_steps


def record(crash_id, cause):
    return {"id": crash_id, "surface": "dry", "lighting": "daylight",
            "vehicles": "2", "primary_cause": cause}


class EvalStepsTests(unittest.TestCase):
    def test_zero_means_last_step_only(self):
        self.assertEqual(eval_steps(10, 0), {10})

    def test_includes_last_step_when_not_a_multiple(self):
        self.assertEqual(eval_steps(10, 4), {4, 8, 10})


class CheckpointSelectorTests(unittest.TestCase):
    def test_keeps_lowest_loss(self):
        s = CheckpointSelector()
        self.assertTrue(s.update(1, 2.0))
        self.assertTrue(s.update(2, 1.5))
        self.assertFalse(s.update(3, 1.7))
        self.assertEqual((s.best_step, s.best_loss), (2, 1.5))

    def test_patience_stops_after_stalled_evals(self):
        s = CheckpointSelector(patience=2)
        s.update(1, 1.0)
        s.update(2, 1.1)
        self.assertFalse(s.should_stop)
        s.update(3, 1.2)
        self.assertTrue(s.should_stop)

    def test_patience_zero_never_stops(self):
        s = CheckpointSelector(patience=0)
        for step in range(1, 6):
            s.update(step, float(step))
        self.assertFalse(s.should_stop)

    def test_min_delta_ignores_tiny_improvements(self):
        s = CheckpointSelector(min_delta=0.1)
        s.update(1, 1.0)
        self.assertFalse(s.update(2, 0.95))
        self.assertTrue(s.update(3, 0.85))


class DevSubsetsTests(unittest.TestCase):
    def setUp(self):
        self.train_counts = Counter({condition_key(record("x", "Rare")): 7,
                                     condition_key(record("x", "Common")): 500})
        self.dev = [record("3", "Common"), record("1", "Rare"), record("2", "Common"),
                    record("4", "Rare")]

    def test_rare_subset_uses_training_counts(self):
        subsets = dev_subsets(self.dev, self.train_counts, 0)
        self.assertEqual([r["id"] for r in subsets["dev"]], ["1", "2", "3", "4"])
        self.assertEqual([r["id"] for r in subsets["rare-dev"]], ["1", "4"])

    def test_limit_caps_each_subset(self):
        subsets = dev_subsets(self.dev, self.train_counts, 1)
        self.assertEqual([r["id"] for r in subsets["dev"]], ["1"])
        self.assertEqual([r["id"] for r in subsets["rare-dev"]], ["1"])


if __name__ == "__main__":
    unittest.main()
