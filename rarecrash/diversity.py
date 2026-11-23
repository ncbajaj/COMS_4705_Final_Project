"""Event-level diversity within rare condition tuples.

For every rare training tuple (5-20 training examples), the model generates K
narratives from the same four conditions. Each narrative is parsed into an
event signature (see events.py), and three systems are compared per tuple:

- model:     K sampled generations;
- reference: K real training narratives for the tuple, drawn without
             replacement; this is the variation the data actually has;
- retrieval: the baseline's one narrative, repeated K times.

Reported per system: distinct event signatures per tuple (all samples, and
only valid samples: parsed, adherent, vehicle-count consistent), how often each
signature field varies within a tuple, condition adherence, internal
vehicle-count consistency, verbatim copying, and how often a generated
signature never occurs in training with those conditions (or at all). Novel
signatures are what the blinded review should inspect for coherence.

Only training conditions are used as inputs; no development or test narrative
is read.

With --scenario (for a model trained on targets with the scenario sentence,
see scenario.py), reference and retrieval narratives carry their collision
type sentence, the signature adds the scenario, and every sample is checked
against the training scenario/heading co-occurrence prior.

    python -m rarecrash.diversity --data data/processed --model runs/gpu/train-uniform/model --out runs/gpu/diversity/uniform
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .evaluate import field_matches, generate, normalize
from .events import SCENARIO_SIGNATURE_FIELDS, SIGNATURE_FIELDS, parse_events, signature
from .pipeline import (CONDITION_FIELDS, condition_key, is_rare_count, load_jsonl,
                       parse_conditions, write_json)
from .scenario import ScenarioPrior, target_text


def rare_tuples(train):
    groups = defaultdict(list)
    for record in train:
        groups[condition_key(record)].append(record)
    return {key: rows for key, rows in sorted(groups.items()) if is_rare_count(len(rows))}


def score_sample(conditions, text, train_targets, signatures_by_tuple, all_signatures,
                 fields=SIGNATURE_FIELDS, prior=None):
    events = parse_events(text)
    sig = signature(events, fields)
    parsed = parse_conditions(text)
    key = tuple(conditions[f] for f in CONDITION_FIELDS)
    scored = {
        "events": {f: events[f] for f in fields},
        "signature": sig,
        "event_parsed": events["parsed"],
        "all_fields_correct": all(field_matches(f, parsed[f], conditions[f])
                                  for f in CONDITION_FIELDS),
        "vehicle_count_consistent": bool(events["vehicle_count_consistent"]),
        "copied_from_train": normalize(text) in train_targets,
        "novel_given_conditions": sig is not None and sig not in signatures_by_tuple[key],
        "novel_anywhere": sig is not None and sig not in all_signatures,
    }
    if prior is not None:
        scored["scenario_consistent"] = prior.consistent(
            events["scenario"], events["heading_relation"], conditions["vehicles"])
    return scored


def summarize(per_tuple, fields=SIGNATURE_FIELDS):
    """per_tuple: list of lists of scored samples, one list per condition tuple."""
    samples = [s for group in per_tuple for s in group]
    parsed = [s for s in samples if s["signature"] is not None]
    n, k = len(samples), max(len(g) for g in per_tuple)
    rate = lambda key, pool=samples: sum(s[key] for s in pool) / len(pool) if pool else None
    distinct = [len({s["signature"] for s in g if s["signature"] is not None}) for g in per_tuple]
    # Diversity that does not come from errors: only samples that parse, follow
    # the conditions, and describe as many vehicles as they state.
    valid = lambda s: (s["signature"] is not None and s["all_fields_correct"]
                       and s["vehicle_count_consistent"])
    distinct_valid = [len({s["signature"] for s in g if valid(s)}) for g in per_tuple]
    summary = {
        "tuples": len(per_tuple), "samples_per_tuple": k, "samples": n,
        "event_parse_rate": len(parsed) / n,
        "mean_distinct_signatures_per_tuple": sum(distinct) / len(distinct),
        "tuples_with_multiple_signatures": sum(d > 1 for d in distinct) / len(distinct),
        "valid_sample_rate": sum(valid(s) for s in samples) / n,
        "mean_distinct_valid_signatures_per_tuple": sum(distinct_valid) / len(distinct_valid),
        "mean_distinct_texts_per_tuple": sum(len({normalize(s["text"]) for s in g})
                                             for g in per_tuple) / len(per_tuple),
        "field_variation": {
            f: sum(len({s["events"][f] for s in g if s["events"][f] is not None}) > 1
                   for g in per_tuple) / len(per_tuple)
            for f in fields},
        "all_fields_adherence": rate("all_fields_correct"),
        "vehicle_count_consistency": rate("vehicle_count_consistent"),
        "copied_from_train_rate": rate("copied_from_train"),
        "novel_signature_given_conditions_rate": rate("novel_given_conditions", parsed),
        "novel_signature_anywhere_rate": rate("novel_anywhere", parsed),
    }
    if "scenario" in fields:
        with_scenario = [s for s in samples if s["events"]["scenario"] is not None]
        judged = [s for s in with_scenario if s.get("scenario_consistent") is not None]
        summary["scenario_parse_rate"] = len(with_scenario) / n
        summary["scenario_consistency_rate"] = rate("scenario_consistent", judged)
        summary["mean_distinct_scenarios_per_tuple"] = sum(
            len({s["events"]["scenario"] for s in g if s["events"]["scenario"] is not None})
            for g in per_tuple) / len(per_tuple)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True,
                        help="directory written by train --save-model, or a Hugging Face model ID")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5,
                        help="samples per tuple; at most 5 so every rare tuple has K references")
    parser.add_argument("--limit-tuples", type=int, default=0, help="first N rare tuples; 0 = all")
    parser.add_argument("--decoding", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4705)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--scenario", action="store_true",
                        help="model was trained with the scenario sentence; add the scenario to "
                             "references, signatures, and a heading-consistency check")
    args = parser.parse_args()
    if not 2 <= args.k <= 5:
        parser.error("k must be between 2 and 5")
    if args.limit_tuples < 0:
        parser.error("limit-tuples must be nonnegative")

    train = load_jsonl(Path(args.data) / "train.jsonl")
    tuples = rare_tuples(train)
    if args.limit_tuples:
        tuples = dict(list(tuples.items())[:args.limit_tuples])
    fields = SCENARIO_SIGNATURE_FIELDS if args.scenario else SIGNATURE_FIELDS
    prior = ScenarioPrior.fit(train) if args.scenario else None
    text_of = lambda record: target_text(record, args.scenario)
    train_targets = {normalize(text_of(r)) for r in train}
    signatures_by_tuple = defaultdict(set)
    for record in train:
        signatures_by_tuple[condition_key(record)].add(
            signature(parse_events(text_of(record)), fields))
    all_signatures = set().union(*signatures_by_tuple.values())

    rng = random.Random(args.seed)
    prompts, texts = [], {"model": [], "reference": [], "retrieval": []}
    for t, (key, rows) in enumerate(tuples.items()):
        conditions = dict(zip(CONDITION_FIELDS, key))
        prompts += [{"id": f"{t}:{j}", **conditions} for j in range(args.k)]
        texts["reference"].append([text_of(r) for r in rng.sample(rows, args.k)])
        texts["retrieval"].append([text_of(rows[0])] * args.k)
    generated, runtime = generate(args, prompts)
    texts["model"] = [[generated[f"{t}:{j}"] for j in range(args.k)] for t in range(len(tuples))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored = {name: [] for name in texts}
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for t, key in enumerate(tuples):
            conditions = dict(zip(CONDITION_FIELDS, key))
            for name in texts:
                group = []
                for j, text in enumerate(texts[name][t]):
                    s = score_sample(conditions, text, train_targets, signatures_by_tuple,
                                     all_signatures, fields, prior)
                    s["text"] = text
                    group.append(s)
                    f.write(json.dumps({"tuple": t, "sample": j, "system": name,
                                        "conditions": conditions,
                                        "train_examples": len(tuples[key]), **s}) + "\n")
                scored[name].append(group)

    summary = {
        "status": "rare-tuple event diversity; development and test narratives not read",
        "model": str(args.model), "k": args.k, "seed": args.seed, "tuples": len(tuples),
        **runtime,
        "results": {name: summarize(groups, fields) for name, groups in scored.items()},
        "notes": [
            "Event signature: weather, road layout, traffic control, work zone, secondary cause, "
            "first-two-vehicle heading relation, and maneuvers (see rarecrash/events.py).",
            "Every source vehicle is 'traveling straight ahead', so maneuvers cannot vary in real data.",
            "Reference narratives come from training, so their copy rate is 1 and novelty 0 by construction.",
            "A novel signature is not an error; samples.jsonl flags them for the blinded review.",
        ],
    }
    if args.scenario:
        summary["scenario"] = True
        summary["notes"].append(
            "Scenario: collision type sentence (rarecrash/scenario.py) is part of the signature; "
            "scenario_consistency_rate checks it against the training heading-relation prior.")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
