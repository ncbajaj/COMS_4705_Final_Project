"""Portable pilot/full fine-tuning loop for a condition-to-narrative model.

Use a small model for a CPU/MPS smoke test, then flan-t5-base on a shared GPU.
Every run writes its settings and results to JSON. The model is saved only when
--save-model is set.

Checkpoint selection: with --eval-every N the model is scored on development
data every N steps (and after the last step). The checkpoint with the lowest
mean teacher-forced dev loss on the --select-on subset is kept in memory,
restored before the final evaluation, and is the one written by --save-model.
--patience stops training after that many evaluations without improvement.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

from .pipeline import condition_key, is_rare_count, load_jsonl, write_json


def prompt(record):
    surface = {"dry": "Dry", "wet": "Wet", "snow/slush": "Snow or slush", "ice": "Ice"}[record["surface"]]
    lighting = {"daylight": "Daylight", "dark_lighted": "Darkness, Lighted Road",
                "dark": "Darkness", "dusk": "Dusk", "dawn": "Dawn"}[record["lighting"]]
    return (
        "Write a short, coherent crash narrative with these conditions: "
        f"road surface = {surface}; lighting = {lighting}; "
        f"vehicles = {record['vehicles']}; primary cause = {record['primary_cause']}."
    )


def dev_subsets(dev, full_train_counts, limit):
    """Deterministic dev subsets: all records and those in rare training tuples.

    limit=0 keeps every record. Each subset is capped independently, so a small
    pilot still has rare-tuple records to score.
    """
    ordered = sorted(dev, key=lambda r: r["id"])
    rare = [r for r in ordered if is_rare_count(full_train_counts.get(condition_key(r), 0))]
    cap = (lambda rows: rows[:limit]) if limit else (lambda rows: rows)
    return {"dev": cap(ordered), "rare-dev": cap(rare)}


class CheckpointSelector:
    """Tracks the best (lowest) selection loss and early-stopping patience."""

    def __init__(self, patience=0, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.best_step = None
        self.evals_since_best = 0

    def update(self, step, loss):
        """Record an evaluation; return True if it is a new best."""
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_step, self.evals_since_best = loss, step, 0
            return True
        self.evals_since_best += 1
        return False

    @property
    def should_stop(self):
        return self.patience > 0 and self.evals_since_best >= self.patience


def eval_steps(total_steps, every):
    """1-based steps after which to evaluate; the last step is always included."""
    steps = set(range(every, total_steps + 1, every)) if every else set()
    steps.add(total_steps)
    return steps


def choose_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--mode", choices=["uniform", "frequency-aware"], default="uniform")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=64, help="0 means all training records")
    parser.add_argument("--dev-limit", type=int, default=16,
                        help="records per dev subset (all and rare); 0 means all")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="evaluate every N steps; 0 means only after the last step")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--select-on", choices=["dev", "rare-dev"], default="dev",
                        help="dev subset whose loss picks the checkpoint")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many evals without improvement; 0 disables")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="loss decrease required to count as an improvement")
    parser.add_argument("--seed", type=int, default=4705)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input-tokens", type=int, default=96)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("steps, batch-size and eval-batch-size must be positive")
    if args.dev_limit < 0 or args.eval_every < 0 or args.patience < 0 or args.min_delta < 0:
        parser.error("dev-limit, eval-every, patience and min-delta must be nonnegative")

    # Transformers may otherwise try to import a separately installed but
    # incompatible TensorFlow stack even though this experiment uses PyTorch.
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    base = Path(args.data)
    train = load_jsonl(base / "train.jsonl")
    dev = load_jsonl(base / "dev.jsonl")
    full_train_counts = Counter(condition_key(r) for r in train)
    if args.limit and args.limit < len(train):
        train = rng.sample(train, args.limit)
    if not train or not dev:
        raise ValueError("Training and development data must be nonempty")
    subsets = dev_subsets(dev, full_train_counts, args.dev_limit)
    if not subsets[args.select_on]:
        raise ValueError(f"No development records in selection subset {args.select_on!r}")
    weights = [1 / max(1, full_train_counts[condition_key(r)]) ** 0.5 for r in train]
    device = choose_device(torch, args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch_tensors(rows):
        inputs = tokenizer(
            [prompt(r) for r in rows], padding=True, truncation=True,
            max_length=args.max_input_tokens, return_tensors="pt",
        )
        targets = tokenizer(
            text_target=[r["target"] for r in rows], padding=True,
            truncation=True, max_length=args.max_target_tokens, return_tensors="pt",
        )
        targets["input_ids"][targets["attention_mask"] == 0] = -100
        return {k: v.to(device) for k, v in inputs.items()}, targets["input_ids"].to(device)

    def per_example_losses(rows):
        """Mean token cross-entropy per example; equals the HF loss at batch size 1."""
        losses = []
        for i in range(0, len(rows), args.eval_batch_size):
            inputs, labels = batch_tensors(rows[i:i + args.eval_batch_size])
            logits = model(**inputs, labels=labels).logits
            token_loss = F.cross_entropy(logits.transpose(1, 2), labels,
                                         ignore_index=-100, reduction="none")
            mask = labels != -100
            losses += ((token_loss * mask).sum(1) / mask.sum(1)).tolist()
        return losses

    def evaluate():
        model.eval()
        with torch.no_grad():
            result = {}
            for name, rows in subsets.items():
                losses = per_example_losses(rows)
                result[name] = sum(losses) / len(losses) if losses else None
        model.train()
        return result

    selector = CheckpointSelector(args.patience, args.min_delta)
    checkpoints = eval_steps(args.steps, args.eval_every)
    history = []
    best_state = None
    stopped_early = False
    steps_run = 0

    first_loss = None
    last_loss = None
    model.train()
    for step in range(args.steps):
        if args.mode == "frequency-aware":
            rows = rng.choices(train, weights=weights, k=args.batch_size)
        else:
            rows = rng.choices(train, k=args.batch_size)
        inputs, labels = batch_tensors(rows)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = last_loss
        steps_run = step + 1
        if step == 0 or steps_run % max(1, args.steps // 5) == 0:
            print(f"step={steps_run}/{args.steps} train_loss={last_loss:.4f}", flush=True)
        if steps_run in checkpoints:
            scores = evaluate()
            improved = selector.update(steps_run, scores[args.select_on])
            if improved:
                # Kept on CPU so selection does not double accelerator memory.
                best_state = {k: v.detach().to("cpu", copy=True)
                              for k, v in model.state_dict().items()}
            history.append({"step": steps_run, "train_loss": last_loss,
                            **{f"{k}_loss": v for k, v in scores.items()}, "best": improved})
            fmt = " ".join(f"{k}_loss={v:.4f}" for k, v in scores.items() if v is not None)
            print(f"eval step={steps_run} {fmt}{' *best*' if improved else ''}", flush=True)
            if selector.should_stop:
                stopped_early = True
                print(f"early stop: {args.patience} evals without improvement", flush=True)
                break

    model.load_state_dict(best_state)
    best_scores = next(h for h in history if h["step"] == selector.best_step)
    model.eval()
    with torch.no_grad():
        examples = []
        for row in subsets["dev"][:3]:
            inputs = tokenizer(prompt(row), return_tensors="pt", truncation=True,
                               max_length=args.max_input_tokens).to(device)
            tokens = model.generate(**inputs, max_new_tokens=min(96, args.max_target_tokens),
                                    do_sample=False)
            examples.append({"id": row["id"], "prompt": prompt(row),
                             "generated": tokenizer.decode(tokens[0], skip_special_tokens=True)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "pilot, not final evaluation" if args.limit else "training run",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "mode": args.mode, "seed": args.seed,
        "steps": args.steps, "steps_run": steps_run, "stopped_early": stopped_early,
        "batch_size": args.batch_size, "lr": args.lr,
        "training_records_used": len(train),
        "dev_records_used": {k: len(v) for k, v in subsets.items()},
        "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
        "first_train_loss": first_loss, "last_train_loss": last_loss,
        "selection": {"select_on": args.select_on, "eval_every": args.eval_every,
                      "patience": args.patience, "min_delta": args.min_delta,
                      "best_step": selector.best_step},
        "mean_teacher_forced_dev_loss": best_scores["dev_loss"],
        "mean_teacher_forced_rare_dev_loss": best_scores["rare-dev_loss"],
        "dev_history": history,
        "generation_examples": examples,
        "warning": "Pilot loss is a plumbing check, not a comparable model-quality result.",
    }
    write_json(out / "run.json", result)
    if args.save_model:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("generation_examples", "dev_history")}, indent=2))


if __name__ == "__main__":
    main()
