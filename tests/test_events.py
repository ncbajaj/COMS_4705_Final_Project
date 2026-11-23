import unittest

from rarecrash.diversity import summarize
from rarecrash.events import heading_relation, parse_events, signature


def narrative(vehicles=(("North", "traveling straight ahead"), ("South", "traveling straight ahead")),
              stated=None, weather="Rain", work_zone="did not occur"):
    stated = len(vehicles) if stated is None else stated
    text = ("The road surface was Wet with Daylight lighting conditions and "
            f"{weather} weather at the time of the crash. The crash occurred on a Four way with "
            f"Stop Sign in place, and it was confirmed that the crash {work_zone} in a work zone. "
            f"The crash involved {stated} vehicles. The primary cause of the crash was Speeding, "
            "with secondary contributing cause (Not Applicable).")
    for i, (heading, maneuver) in enumerate(vehicles):
        text += f"Vehicle {i}, a 2015 model, was moving {heading} and was {maneuver} before the crash."
    return text


class ParseEventsTests(unittest.TestCase):
    def test_fields(self):
        e = parse_events(narrative())
        self.assertTrue(e["parsed"])
        self.assertEqual((e["weather"], e["road_layout"], e["traffic_control"], e["work_zone"],
                          e["secondary_cause"], e["heading_relation"], e["maneuvers"]),
                         ("Rain", "Four way", "Stop Sign", False, "Not Applicable", "opposite",
                          ("traveling straight ahead",)))

    def test_work_zone_happened(self):
        self.assertTrue(parse_events(narrative(work_zone="happened"))["work_zone"])

    def test_missing_weather_is_unparsed(self):
        text = narrative().replace("Rain weather", "Darkness lighting conditions")
        e = parse_events(text)
        self.assertFalse(e["parsed"])
        self.assertIsNone(signature(e))

    def test_heading_relation(self):
        v = lambda *hs: [{"heading": h} for h in hs]
        self.assertEqual(heading_relation(v("North", "North")), "same")
        self.assertEqual(heading_relation(v("East", "West")), "opposite")
        self.assertEqual(heading_relation(v("North", "West")), "crossing")
        self.assertEqual(heading_relation(v("North", "Northeast")), "angled")
        self.assertEqual(heading_relation(v("North", "Unknown")), "unknown")
        self.assertEqual(heading_relation(v("North")), "single")
        self.assertEqual(heading_relation([]), "none")

    def test_vehicle_count_consistency(self):
        two = (("North", "traveling straight ahead"),) * 2
        self.assertTrue(parse_events(narrative(two))["vehicle_count_consistent"])
        self.assertFalse(parse_events(narrative(two, stated=1))["vehicle_count_consistent"])
        self.assertTrue(parse_events(narrative((), stated=6))["vehicle_count_consistent"])
        self.assertFalse(parse_events(narrative((), stated=2))["vehicle_count_consistent"])

    def test_maneuvers_ignore_vehicle_count(self):
        three = (("North", "traveling straight ahead"),) * 3
        four = (("North", "traveling straight ahead"),) * 4
        self.assertEqual(parse_events(narrative(three))["maneuvers"],
                         parse_events(narrative(four))["maneuvers"])


class DiversitySummaryTests(unittest.TestCase):
    def sample(self, text):
        e = parse_events(text)
        return {"text": text, "events": e, "signature": signature(e),
                "all_fields_correct": True, "vehicle_count_consistent": True,
                "copied_from_train": False, "novel_given_conditions": False,
                "novel_anywhere": False}

    def test_counts_distinct_signatures_not_strings(self):
        same_event = [self.sample(narrative()), self.sample(narrative() + " ")]
        different = [self.sample(narrative()), self.sample(narrative(weather="Snow"))]
        summary = summarize([same_event, different])
        self.assertEqual(summary["mean_distinct_signatures_per_tuple"], 1.5)
        self.assertEqual(summary["tuples_with_multiple_signatures"], 0.5)
        self.assertEqual(summary["field_variation"]["weather"], 0.5)
        self.assertEqual(summary["field_variation"]["road_layout"], 0.0)

    def test_valid_diversity_excludes_inconsistent_samples(self):
        bad = self.sample(narrative(weather="Snow"))
        bad["vehicle_count_consistent"] = False
        summary = summarize([[self.sample(narrative()), bad]])
        self.assertEqual(summary["mean_distinct_signatures_per_tuple"], 2)
        self.assertEqual(summary["mean_distinct_valid_signatures_per_tuple"], 1)
        self.assertEqual(summary["valid_sample_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
