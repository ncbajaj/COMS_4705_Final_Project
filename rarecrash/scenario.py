"""Opt-in crash scenario signal from the record's collision type.

The Illinois narratives carry no maneuver information: every vehicle sentence
says "traveling straight ahead", and the text after the target cut is only
driver and passenger descriptions (the sev/inj task files repeat the same text
with other labels). The crash record's manner-of-collision label
(`collision_type_label`, e.g. FRONT TO REAR, TURNING, ANGLE IMPACTS) is the one
field that says what kind of crash happened, and it varies within rare
condition tuples (3.2 distinct labels per rare training tuple on average).

This module turns that label into one fixed-template sentence,

    "The manner of collision was front to rear."

which can be added to a training target (before the vehicle sentences) so the model learns to generate a
scenario (opt-in: nothing in the default pipeline uses it). It also parses the
sentence back out of generated text, and checks it against the vehicle
headings with a co-occurrence prior estimated on training data only: e.g.
front-to-rear crashes have the first two vehicles heading the same way in 97%
of training narratives, so a generated front-to-rear crash with opposite
headings is flagged as implausible.

    python -m rarecrash.scenario --data data/processed --out runs/scenario_audit.json
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re


# Illinois collision type label -> phrase used in the sentence. Every label in
# the processed train and dev data is covered.
PHRASES = {
    "FRONT TO REAR": "front to rear",
    "TURNING": "turning",
    "COLLISION WITH OBJECTS": "collision with an object",
    "ANGLE IMPACTS": "angle",
    "SIDESWIPE SAME DIRECTION": "sideswipe, same direction",
    "PARKED MOTO VEHICLE": "collision with a parked motor vehicle",
    "PEDESTRIAN COLLISIONS": "collision with a pedestrian",
    "ANIMAL COLLISIONS": "collision with an animal",
    "PEDALCYCLIST COLLISIONS": "collision with a pedalcyclist",
    "OVERTURNED": "overturned",
    "FRONT TO FRONT": "front to front",
    "SIDESWIPE OPPOSITE DIRECTION": "sideswipe, opposite direction",
    "REAR TO FRONT": "rear to front",
    "REAR TO SIDE": "rear to side",
    "REAR TO REAR": "rear to rear",
    "OTHERS": "other",
}
LABELS = {phrase: label for label, phrase in PHRASES.items()}
# Coarser grouping for reporting.
FAMILIES = {
    "FRONT TO REAR": "rear_end",
    "TURNING": "turning",
    "ANGLE IMPACTS": "angle",
    "FRONT TO FRONT": "head_on",
    "SIDESWIPE SAME DIRECTION": "sideswipe",
    "SIDESWIPE OPPOSITE DIRECTION": "sideswipe",
    "PARKED MOTO VEHICLE": "parked_vehicle",
    "REAR TO FRONT": "backing",
    "REAR TO SIDE": "backing",
    "REAR TO REAR": "backing",
    "COLLISION WITH OBJECTS": "single_vehicle",
    "PEDESTRIAN COLLISIONS": "single_vehicle",
    "ANIMAL COLLISIONS": "single_vehicle",
    "PEDALCYCLIST COLLISIONS": "single_vehicle",
    "OVERTURNED": "single_vehicle",
    "OTHERS": "other",
}
TEMPLATE = "The manner of collision was {}."
VEHICLE_START_RE = re.compile(r"Vehicle 0, an? ")
SCENARIO_RE = re.compile(r"The manner of collision was (.*?)\.(?=\s|$|[A-Z])", re.S)


def scenario_sentence(label):
    if label not in PHRASES:
        raise ValueError(f"No scenario phrase for collision type {label!r}")
    return TEMPLATE.format(PHRASES[label])


def with_scenario(target, label):
    """Target with the scenario sentence inserted before the vehicle sentences.

    Placing it before "Vehicle 0" (rather than at the end) keeps it inside the
    192-token training window: about 9% of training targets are longer than
    that and lose their last vehicle sentences to truncation. Targets without
    vehicle sentences (5+ vehicles) get it at the end. The source style of
    sentences joined without a space ("...(Unknown).Vehicle 0") is kept.
    """
    target = target.rstrip()
    first_vehicle = VEHICLE_START_RE.search(target)
    if not first_vehicle:
        return f"{target} {scenario_sentence(label)}"
    i = first_vehicle.start()
    head = target[:i].rstrip()
    return f"{head} {scenario_sentence(label)}{target[i:]}"


def target_text(record, scenario=False):
    """A record's training target, with the scenario sentence when requested."""
    if not scenario:
        return record["target"]
    return with_scenario(record["target"], record["collision_type_label"])


def parse_scenario(text):
    """(label, raw phrase) of the first scenario sentence in text.

    (None, None) when there is no sentence; (None, phrase) when the phrase is
    not one of the known ones.
    """
    match = SCENARIO_RE.search(text)
    if not match:
        return None, None
    phrase = re.sub(r"\s+", " ", match.group(1)).strip()
    return LABELS.get(phrase.lower()), phrase


class ScenarioPrior:
    """Training co-occurrence of scenario with heading relation and vehicle bin.

    A (scenario, heading relation) pair is plausible when the relation occurs
    with that scenario in at least `min_share` of its training narratives;
    likewise for the vehicle bin. Relations that carry no heading information
    ("none": no vehicle sentences, "unknown": a heading was Unknown) are not
    judged.
    """

    UNJUDGED_RELATIONS = {"none", "unknown"}

    def __init__(self, min_share=0.02):
        self.min_share = min_share
        self.relation = defaultdict(Counter)
        self.vehicles = defaultdict(Counter)

    @classmethod
    def fit(cls, records, min_share=0.02):
        from .events import parse_events
        prior = cls(min_share)
        for record in records:
            label = record["collision_type_label"]
            prior.relation[label][parse_events(record["target"])["heading_relation"]] += 1
            prior.vehicles[label][record["vehicles"]] += 1
        return prior

    @staticmethod
    def _share(counter, value):
        total = sum(counter.values())
        return counter[value] / total if total else 0.0

    def relation_share(self, label, relation):
        return self._share(self.relation[label], relation)

    def vehicles_share(self, label, vehicles):
        return self._share(self.vehicles[label], vehicles)

    def consistent(self, label, relation, vehicles=None):
        """True/False, or None when the scenario is missing or unknown to training."""
        if label is None or label not in self.relation:
            return None
        if vehicles is not None and self.vehicles_share(label, vehicles) < self.min_share:
            return False
        if relation in self.UNJUDGED_RELATIONS:
            return True
        return self.relation_share(label, relation) >= self.min_share


def contingency(records):
    """collision type -> heading relation -> count."""
    from .events import parse_events
    table = defaultdict(Counter)
    for record in records:
        table[record["collision_type_label"]][parse_events(record["target"])["heading_relation"]] += 1
    return table


def audit(train, dev, min_share=0.02, seed=4705):
    """Scenario statistics from train (and dev for the calibration check only)."""
    from .events import parse_events
    from .pipeline import condition_key, is_rare_count

    table = contingency(train)
    relations = ["single", "same", "opposite", "crossing", "angled", "none", "unknown"]
    rows = {label: {"n": sum(c.values()), **{r: c[r] for r in relations}}
            for label, c in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))}

    groups = defaultdict(list)
    for record in train:
        groups[condition_key(record)].append(record)
    rare = [rows_ for rows_ in groups.values() if is_rare_count(len(rows_))]
    distinct = lambda key: [len({key(r) for r in g}) for g in rare]
    labels_per = distinct(lambda r: r["collision_type_label"])
    families_per = distinct(lambda r: FAMILIES[r["collision_type_label"]])
    headings_per = distinct(lambda r: parse_events(r["target"])["heading_relation"])

    def round_trip(records):
        ok = 0
        for record in records:
            text = with_scenario(record["target"], record["collision_type_label"])
            ok += parse_scenario(text)[0] == record["collision_type_label"]
        return ok / len(records) if records else None

    prior = ScenarioPrior.fit(train, min_share)
    rng = random.Random(seed)
    labels = [r["collision_type_label"] for r in train]

    def consistency(records, shuffled):
        judged = []
        for record in records:
            label = rng.choice(labels) if shuffled else record["collision_type_label"]
            relation = parse_events(record["target"])["heading_relation"]
            judged.append(prior.consistent(label, relation, record["vehicles"]))
        known = [j for j in judged if j is not None]
        return sum(known) / len(known) if known else None

    return {
        "status": "train statistics; dev used only for the round trip and calibration; test not read",
        "labels_in_train": len(table),
        "labels_without_phrase": sorted({r["collision_type_label"] for r in train + dev} - set(PHRASES)),
        "collision_type_by_heading_relation_train": rows,
        "rare_tuples": len(rare),
        "rare_tuple_mean_distinct_collision_types": sum(labels_per) / len(rare),
        "rare_tuple_share_with_multiple_collision_types": sum(n > 1 for n in labels_per) / len(rare),
        "rare_tuple_mean_distinct_families": sum(families_per) / len(rare),
        "rare_tuple_mean_distinct_heading_relations": sum(headings_per) / len(rare),
        "sentence_round_trip_rate": {"train": round_trip(train), "dev": round_trip(dev)},
        "prior_min_share": min_share,
        "dev_consistency_true_label": consistency(dev, shuffled=False),
        "dev_consistency_random_train_label": consistency(dev, shuffled=True),
        "notes": [
            "Heading relation is for the first two vehicles; 'none' means no vehicle sentences "
            "(5+ vehicles or missing) and 'unknown' an Unknown heading; neither is judged.",
            "Consistency with a random training label shows how much the check can discriminate.",
        ],
    }


def main():
    from .pipeline import load_jsonl, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-share", type=float, default=0.02)
    args = parser.parse_args()
    base = Path(args.data)
    train, dev = (load_jsonl(base / f"{s}.jsonl") for s in ("train", "dev"))
    report = audit(train, dev, args.min_share)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
