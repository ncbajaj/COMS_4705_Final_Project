"""Summarize several `rarecrash.evaluate` runs into one table.

Runs are grouped by label, so sampled runs that differ only in seed are
averaged and reported as mean ± standard deviation. The retrieval baseline is
read from the first run, since it is identical in every run.

    python -m rarecrash.compare --runs uniform-sample=runs/gpu/eval/uniform-s1 \\
        uniform-sample=runs/gpu/eval/uniform-s2 frequency-greedy=runs/gpu/eval/freq-greedy \\
        --out runs/gpu/comparison.md
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


METRICS = [
    ("all_fields_adherence", "adherence"),
    ("mean_rouge_l_f1", "ROUGE-L"),
    ("within_tuple_distinct_rate", "within-tuple distinct"),
    ("distinct_2", "distinct-2"),
    ("copied_from_train_rate", "copied"),
]


def parse_run(spec):
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label, Path(path)


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def table(groups, subset):
    header = ["system", "runs", *(name for _, name in METRICS)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, results in groups.items():
        rows = [r[subset] for r in results]
        cells = [label, str(len(rows)), *(fmt([r.get(key) for r in rows]) for key, _ in METRICS)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=parse_run, required=True,
                        help="LABEL=PATH pairs; PATH holds an evaluate summary.json")
    parser.add_argument("--out", help="write the Markdown tables here as well")
    args = parser.parse_args()

    groups = defaultdict(list)
    scope = None
    for label, path in args.runs:
        summary = json.loads((path / "summary.json").read_text())
        run_scope = (summary["subset"], summary["limit"])
        if scope is None:
            scope = run_scope
        elif run_scope != scope:
            parser.error(f"{path} scored subset/limit {run_scope}, but the first run "
                         f"scored {scope}; compare runs on the same dev records only")
        if not groups:
            groups["retrieval"].append(summary["results"]["retrieval"])
        groups[label].append(summary["results"]["model"])

    text = "\n\n".join(f"### {title}\n\n{table(groups, subset)}" for subset, title in
                       (("all", "All development records"),
                        ("rare_5_to_20", "Rare-tuple development records (5-20 training examples)")))
    text += ("\n\nMean ± standard deviation across runs with the same label (sampling seeds). "
             "Development data only.\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
