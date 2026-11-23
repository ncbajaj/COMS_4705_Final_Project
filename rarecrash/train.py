"""Portable pilot/full fine-tuning loop for a condition-to-narrative model.

Use a small model for a CPU/MPS smoke test, then flan-t5-base on a shared GPU.
Every run writes its settings and results to JSON. The model is saved only when
--save-model is set.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

from .pipeline import condition_key, load_jsonl, write_json


def prompt(record):
    surface = {"dry": "Dry", "wet": "Wet", "snow/slush": "Snow or slush", "ice": "Ice"}[record["surface"]]
    lighting = {"daylight": "Daylight", "dark_lighted": "Darkness, Lighted Road",
                "dark": "Darkness", "dusk": "Dusk", "dawn": "Dawn"}[record["lighting"]]
    return (
        "Write a short, coherent crash narrative with these conditions: "
        f"road surface = {surface}; lighting = {lighting}; "
        f"vehicles = {record['vehicles']}; primary cause = {record['primary_cause']}."
    )


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
    parser.add_argument("--dev-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4705)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input-tokens", type=int, default=96)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.dev_limit < 1:
        parser.error("steps, batch-size and dev-limit must be positive")

    # Transformers may otherwise try to import a separately installed but
    # incompatible TensorFlow stack even though this experiment uses PyTorch.
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
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
    dev = sorted(dev, key=lambda r: r["id"])[:args.dev_limit]
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
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step={step + 1}/{args.steps} train_loss={last_loss:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        losses = []
        for row in dev:
            inputs, labels = batch_tensors([row])
            losses.append(float(model(**inputs, labels=labels).loss.detach().cpu()))
        examples = []
        for row in dev[:3]:
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
        "steps": args.steps, "batch_size": args.batch_size,
        "training_records_used": len(train), "dev_records_used": len(dev),
        "device": device, "torch": torch.__version__, "transformers": transformers.__version__,
        "first_train_loss": first_loss, "last_train_loss": last_loss,
        "mean_teacher_forced_dev_loss": sum(losses) / len(losses),
        "generation_examples": examples,
        "warning": "Pilot loss is a plumbing check, not a comparable model-quality result.",
    }
    write_json(out / "run.json", result)
    if args.save_model:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    print(json.dumps({k: v for k, v in result.items() if k != "generation_examples"}, indent=2))


if __name__ == "__main__":
    main()
