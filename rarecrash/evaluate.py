"""Development-set evaluation of a trained model against the retrieval baseline.

Both systems are scored with the same metrics:

- condition adherence: the four fields are parsed back out of the output text
  with the extraction patterns (per field and all four together);
- ROUGE-L F1 against the held-out dev narrative;
- diversity: distinct outputs, distinct-1/2, and distinct outputs within
  condition tuples that occur more than once in the subset;
- copying: share of outputs identical to a training narrative.

Only the development split is read. The test split stays unopened.

    python -m rarecrash.evaluate --data data/processed --model runs/pilot-select/model --out runs/dev-eval

--model may also be a Hugging Face model ID with --revision, which evaluates
the untuned model as a baseline.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

from .pipeline import (CONDITION_FIELDS, condition_key, is_rare_count, load_jsonl,
                       parse_conditions, write_json)
from .scenario import target_text
from .train import choose_device, prompt


TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def rouge_l_f1(candidate, reference):
    a, b = tokens(candidate), tokens(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def distinct_n(texts, n):
    grams = [tuple(t[i:i + n]) for t in map(tokens, texts) for i in range(len(t) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def retrieval_outputs(train, dev, scenario=False):
    """Same rule as `rarecrash retrieve`: the first training narrative per tuple."""
    pool = {}
    for record in train:
        pool.setdefault(condition_key(record), target_text(record, scenario))
    return {r["id"]: pool.get(condition_key(r)) for r in dev}


def field_matches(field, parsed, expected):
    """Exact match, except causes ignore spacing and case.

    Some source cause names contain typos such as "DisregardingOther Traffic
    Signs"; a correctly spaced cause in the output should not count as wrong.
    """
    if parsed is None:
        return False
    if field == "primary_cause":
        squash = lambda s: re.sub(r"\s+", "", s).lower()
        return squash(parsed) == squash(expected)
    return parsed == expected


def score_row(record, output, train_targets, scenario=False):
    if output is None:
        return {"has_output": False, "fields_correct": dict.fromkeys(CONDITION_FIELDS, False),
                "all_fields_correct": False, "rouge_l": 0.0, "copied_from_train": False}
    parsed = parse_conditions(output)
    correct = {f: field_matches(f, parsed[f], record[f]) for f in CONDITION_FIELDS}
    return {"has_output": True, "parsed": parsed, "fields_correct": correct,
            "all_fields_correct": all(correct.values()),
            "rouge_l": rouge_l_f1(output, target_text(record, scenario)),
            "copied_from_train": normalize(output) in train_targets}


def summarize(rows):
    """rows: dicts with record, output, and score for one system and subset."""
    n = len(rows)
    if not n:
        return {"dev_examples": 0}
    outputs = [r["output"] for r in rows if r["output"] is not None]
    by_tuple = defaultdict(list)
    for r in rows:
        if r["output"] is not None:
            by_tuple[condition_key(r["record"])].append(normalize(r["output"]))
    repeated = [outs for outs in by_tuple.values() if len(outs) > 1]
    return {
        "dev_examples": n,
        "output_coverage": len(outputs) / n,
        "all_fields_adherence": sum(r["score"]["all_fields_correct"] for r in rows) / n,
        "field_adherence": {f: sum(r["score"]["fields_correct"][f] for r in rows) / n
                            for f in CONDITION_FIELDS},
        "mean_rouge_l_f1": sum(r["score"]["rouge_l"] for r in rows) / n,
        "distinct_outputs": len({normalize(o) for o in outputs}),
        "distinct_1": distinct_n(outputs, 1),
        "distinct_2": distinct_n(outputs, 2),
        "repeated_condition_tuples": len(repeated),
        "within_tuple_distinct_rate": (sum(len(set(o)) / len(o) for o in repeated) / len(repeated)
                                       if repeated else None),
        "copied_from_train_rate": sum(r["score"]["copied_from_train"] for r in rows) / n,
        "mean_output_tokens": (sum(len(tokens(o)) for o in outputs) / len(outputs)
                               if outputs else None),
    }


def generate(args, records):
    os.environ.setdefault("USE_TF", "0")
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision).to(device).eval()
    settings = {"max_new_tokens": args.max_new_tokens}
    if args.decoding == "sample":
        settings.update(do_sample=True, top_p=args.top_p, temperature=args.temperature)
    else:
        settings.update(do_sample=False, num_beams=args.num_beams)
    outputs = {}
    with torch.no_grad():
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            inputs = tokenizer([prompt(r) for r in batch], padding=True, truncation=True,
                               max_length=args.max_input_tokens, return_tensors="pt").to(device)
            generated = model.generate(**inputs, **settings)
            for r, text in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                outputs[r["id"]] = text
            print(f"generated {min(i + args.batch_size, len(records))}/{len(records)}", flush=True)
    runtime = {"requested_revision": args.revision,
               "resolved_revision": getattr(model.config, "_commit_hash", None),
               "device": device, "torch": torch.__version__,
               "transformers": transformers.__version__, "generation": settings}
    return outputs, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True,
                        help="directory written by train --save-model, or a Hugging Face model ID")
    parser.add_argument("--revision", default=None,
                        help="Hugging Face revision; pin a commit hash for an untuned baseline")
    parser.add_argument("--out", required=True)
    parser.add_argument("--subset", choices=["all", "rare-dev"], default="all",
                        help="generate for every dev record, or only rare-tuple records")
    parser.add_argument("--limit", type=int, default=0, help="first N records by ID; 0 means all")
    parser.add_argument("--decoding", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4705)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--scenario-sentence", action="store_true",
                        help="score against targets that include the scenario sentence")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("limit must be nonnegative; batch-size and max-new-tokens positive")

    base = Path(args.data)
    train = load_jsonl(base / "train.jsonl")
    dev = sorted(load_jsonl(base / "dev.jsonl"), key=lambda r: r["id"])
    train_counts = Counter(condition_key(r) for r in train)
    rare_ids = {r["id"] for r in dev if is_rare_count(train_counts[condition_key(r)])}
    if args.subset == "rare-dev":
        dev = [r for r in dev if r["id"] in rare_ids]
    if args.limit:
        dev = dev[:args.limit]
    if not dev:
        raise ValueError("No development records selected")

    train_targets = {normalize(target_text(r, args.scenario_sentence)) for r in train}
    model_outputs, runtime = generate(args, dev)
    systems = {"model": model_outputs,
               "retrieval": retrieval_outputs(train, dev, args.scenario_sentence)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {name: [] for name in systems}
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in dev:
            for name, outputs in systems.items():
                row = {"record": record, "output": outputs[record["id"]]}
                row["score"] = score_row(record, row["output"], train_targets,
                                         args.scenario_sentence)
                rows[name].append(row)
                f.write(json.dumps({
                    "id": record["id"], "system": name, "rare": record["id"] in rare_ids,
                    "conditions": {k: record[k] for k in CONDITION_FIELDS},
                    "output": row["output"], "reference": target_text(record, args.scenario_sentence), **row["score"],
                }) + "\n")

    run_info = Path(args.model).parent / "run.json" if Path(args.model).is_dir() else None
    summary = {
        "status": "development evaluation; test split not read",
        "model": str(args.model),
        "training_run": json.loads(run_info.read_text()) if run_info and run_info.exists() else None,
        "subset": args.subset, "limit": args.limit, "seed": args.seed,
        "scenario_sentence": args.scenario_sentence,
        **runtime,
        "results": {
            name: {"all": summarize(sys_rows),
                   "rare_5_to_20": summarize([r for r in sys_rows if r["record"]["id"] in rare_ids])}
            for name, sys_rows in rows.items()
        },
        "notes": [
            "Adherence parses the output text with the extraction patterns; a missing output counts as non-adherent.",
            "Primary cause matching ignores spacing and case, because some source cause names contain spacing typos.",
            "Retrieval reuses one training narrative per condition tuple, so its within-tuple distinct rate is low by construction.",
            "ROUGE-L rewards closeness to one reference narrative, not plausibility.",
        ],
    }
    if summary["training_run"]:
        summary["training_run"].pop("generation_examples", None)
        summary["training_run"].pop("dev_history", None)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
