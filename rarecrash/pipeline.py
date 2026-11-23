"""Standard-library data preparation and preliminary baseline.

This is deliberately independent of a training framework so either teammate can
audit the same examples on a CPU machine.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile


DATA_URL = "https://zenodo.org/records/16896765/files/Puw242/SafeTraffic-v0.1.0.zip?download=1"
SHA256 = "9641f5d782048c86cde46a788ff0fae6e6ff0227039652c251a65789200ebbc3"
SURFACES = {"Dry": "dry", "Wet": "wet", "Snow or slush": "snow/slush", "Ice": "ice"}
LIGHTING = {
    "Daylight": "daylight",
    "Darkness, Lighted Road": "dark_lighted",
    "Darkness": "dark",
    "Dusk": "dusk",
    "Dawn": "dawn",
}
ROAD_RE = re.compile(r"The road surface was (.*?) with (.*?) lighting conditions", re.I | re.S)
VEHICLE_RE = re.compile(r"The crash involved (\d+) vehicles", re.I)
CAUSE_RE = re.compile(r"The primary cause of the crash was (.*?), with secondary contributing cause", re.I | re.S)
HUMAN_RE = re.compile(r"<s>Human:\s*(.*?)</s>", re.I | re.S)
LABEL_RE = re.compile(r"<s>Assistant:\s*<(.*?)>", re.I | re.S)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_archive(path):
    digest = sha256(path)
    if digest != SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: {digest}; expected {SHA256}")


def read_illinois_rows(path):
    with zipfile.ZipFile(path) as archive:
        for original_split in ("train", "val", "test"):
            suffix = f"/data/IL/{original_split}/type.csv"
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {suffix}, found {len(matches)}")
            stream = io.TextIOWrapper(archive.open(matches[0]), encoding="utf-8")
            for row in csv.DictReader(stream):
                yield original_split, row


def extract(row, original_split):
    text = row["text"]
    human = HUMAN_RE.search(text)
    label = LABEL_RE.search(text)
    if not human or not label:
        return None, "bad_format"
    narrative = human.group(1).strip()
    road = ROAD_RE.search(narrative)
    vehicle = VEHICLE_RE.search(narrative)
    cause = CAUSE_RE.search(narrative)
    if not road or not vehicle or not cause:
        return None, "missing_field"
    raw_surface, raw_lighting = (s.strip() for s in road.groups())
    if raw_surface not in SURFACES or raw_lighting not in LIGHTING:
        return None, "outside_schema"
    primary_cause = cause.group(1).strip()
    if primary_cause.lower() in {"unable to determine", "not applicable", "unknown"}:
        return None, "unknown_cause"
    count = int(vehicle.group(1))
    if count < 1:
        return None, "bad_vehicle_count"
    target = narrative[road.start():]
    driver_markers = [m.start() for m in re.finditer(r"(?:The driver was|The driver’s|The driver's)", target)]
    if driver_markers:
        target = target[:driver_markers[0]]
    target = re.sub(r"\s+", " ", target).strip()
    if len(target) < 100:
        return None, "short_target"
    record = {
        "id": str(row["ID"]),
        "original_split": original_split,
        "surface": SURFACES[raw_surface],
        "lighting": LIGHTING[raw_lighting],
        "vehicles": "3+" if count >= 3 else str(count),
        "vehicle_count_original": count,
        "primary_cause": primary_cause,
        "collision_type_label": label.group(1).strip(),
        "target": target,
    }
    return record, None


def split_for_id(crash_id):
    value = int(hashlib.sha256(f"rarecrash-v1:{crash_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.8 else "dev" if value < 0.9 else "test"


def condition_key(record):
    return tuple(record[k] for k in ("surface", "lighting", "vehicles", "primary_cause"))


# Provisional rare-condition range: training examples per condition tuple.
RARE_MIN, RARE_MAX = 5, 20


def is_rare_count(n):
    return RARE_MIN <= n <= RARE_MAX


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def download(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        urllib.request.urlretrieve(DATA_URL, out)
    check_archive(out)
    print(json.dumps({"archive": str(out), "sha256": SHA256}))


def prepare(args):
    check_archive(args.zip)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records, excluded = [], Counter()
    ids = set()
    targets = set()
    for original_split, row in read_illinois_rows(args.zip):
        if row["ID"] in ids:
            raise ValueError(f"Duplicate crash ID {row['ID']}")
        ids.add(row["ID"])
        record, reason = extract(row, original_split)
        if reason:
            excluded[reason] += 1
            continue
        target_key = record["target"].casefold()
        if target_key in targets:
            excluded["duplicate_target"] += 1
            continue
        targets.add(target_key)
        record["split"] = split_for_id(record["id"])
        records.append(record)
    for split in ("train", "dev", "test"):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                if record["split"] == split:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "source": DATA_URL, "archive_sha256": SHA256,
        "original_illinois_ids": len(ids), "retained": len(records),
        "excluded": dict(excluded),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "split_rule": "SHA256('rarecrash-v1:' + Illinois crash ID), 80/10/10",
        "target_rule": "from road-surface sentence up to first driver-description marker",
        "processed_sha256": {split: sha256(output / f"{split}.jsonl") for split in ("train", "dev", "test")},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def audit(args):
    base = Path(args.data)
    train, dev = (load_jsonl(base / f"{s}.jsonl") for s in ("train", "dev"))
    test_count = sum(1 for _ in (base / "test.jsonl").open(encoding="utf-8"))
    counts = Counter(condition_key(r) for r in train)
    rare = {key for key, n in counts.items() if is_rare_count(n)}
    dev_rare = [r for r in dev if condition_key(r) in rare]
    exact_target_duplicates = sum(n - 1 for n in Counter(r["target"] for r in train).values() if n > 1)
    report = {
        "train": len(train), "dev": len(dev), "test_count_unopened": test_count,
        "schema_values": {
            "surface": sorted(set(r["surface"] for r in train)),
            "lighting": sorted(set(r["lighting"] for r in train)),
            "vehicles": sorted(set(r["vehicles"] for r in train)),
            "primary_cause_categories": len(set(r["primary_cause"] for r in train)),
        },
        "train_condition_combinations": len(counts),
        "rare_train_combinations_5_to_20": len(rare),
        "rare_train_examples_5_to_20": sum(counts[key] for key in rare),
        "dev_examples_matching_rare_train_combinations": len(dev_rare),
        "train_exact_target_duplicate_rows": exact_target_duplicates,
        "train_original_release_splits": dict(Counter(r["original_split"] for r in train)),
        "warning": "Collision type is metadata; most source narratives do not explicitly state its label.",
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


def retrieve(args):
    base = Path(args.data)
    train, dev = (load_jsonl(base / f"{s}.jsonl") for s in ("train", "dev"))
    pool = defaultdict(list)
    for record in train:
        pool[condition_key(record)].append(record)
    train_counts = {key: len(rows) for key, rows in pool.items()}
    groups = {"all": dev, "rare_5_to_20": [r for r in dev if is_rare_count(train_counts.get(condition_key(r), 0))]}
    report = {}
    for name, examples in groups.items():
        exact = [r for r in examples if condition_key(r) in pool]
        retrieved = []
        for r in exact:
            # One fixed retrieved example per condition tuple. This makes the
            # baseline's lack of within-condition diversity explicit.
            retrieved.append(pool[condition_key(r)][0]["target"])
        report[name] = {
            "dev_examples": len(examples),
            "exact_condition_retrieval_coverage": len(exact) / len(examples) if examples else None,
            "exact_matches": len(exact),
            "distinct_retrieved_texts": len(set(retrieved)),
            "distinct_condition_tuples_with_exact_match": len({condition_key(r) for r in exact}),
            "condition_adherence_of_exact_matches": 1.0 if exact else None,
            "note": "Adherence here follows from the source metadata, not an independent text parser.",
        }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


def review_queue(args):
    import random
    rng = random.Random(args.seed)
    base = Path(args.data)
    records = load_jsonl(base / "train.jsonl") + load_jsonl(base / "dev.jsonl")
    selected = rng.sample(records, min(args.count, len(records)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "split", "surface", "lighting", "vehicles", "primary_cause",
              "target", "field_extraction_correct", "target_usable", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            writer.writerow({k: record.get(k, "") for k in fields})
    print(json.dumps({"review_queue": str(path), "rows": len(selected), "seed": args.seed}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download"); p.add_argument("--out", required=True); p.set_defaults(func=download)
    p = sub.add_parser("prepare"); p.add_argument("--zip", required=True); p.add_argument("--out", required=True); p.set_defaults(func=prepare)
    p = sub.add_parser("audit"); p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.set_defaults(func=audit)
    p = sub.add_parser("retrieve"); p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.set_defaults(func=retrieve)
    p = sub.add_parser("review-queue"); p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--count", type=int, default=60); p.add_argument("--seed", type=int, default=4705); p.set_defaults(func=review_queue)
    args = parser.parse_args()
    args.func(args)
