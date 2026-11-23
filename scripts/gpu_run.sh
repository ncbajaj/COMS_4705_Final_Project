#!/usr/bin/env bash
# Full FLAN-T5-base experiment on one GPU machine, from a fresh clone:
#   1. environment and data (the split is checked against the milestone numbers)
#   2. uniform and frequency-aware training, identical except for the sampler
#   3. dev evaluation: greedy plus several sampling seeds per model, and the
#      untuned FLAN-T5-base baseline
#   4. one comparison table
#
# Usage, from the repository root:
#   bash scripts/gpu_run.sh
# Settings can be overridden with environment variables, e.g.
#   STEPS=4000 BF16=0 bash scripts/gpu_run.sh
#
# Each finished stage is skipped on a rerun (its run.json or summary.json
# exists), so after a crash the script can simply be started again. A training
# stage that crashed restarts from step 0; delete its directory first.
# The test split is never read.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
OUT=${OUT:-runs/gpu}
MODEL=${MODEL:-google/flan-t5-base}
REVISION=${REVISION:-7bcac572ce56db69c1ea7c8af255c5d7c9672fc2}  # flan-t5-base, resolved 2026-11-23
STEPS=${STEPS:-3000}          # about 5 epochs of the 9,625 training records at batch 16
BATCH=${BATCH:-16}
LR=${LR:-3e-4}
EVAL_EVERY=${EVAL_EVERY:-250}
SEED=${SEED:-4705}
SAMPLE_SEEDS=${SAMPLE_SEEDS:-"1 2 3"}
BF16=${BF16:-auto}            # auto: on if the GPU supports bfloat16
# Smoke-test overrides only; leave at their defaults for the real run.
REQUIRE_CUDA=${REQUIRE_CUDA:-1}
DEV_LIMIT=${DEV_LIMIT:-0}     # dev records scored during training; 0 = all
EVAL_LIMIT=${EVAL_LIMIT:-0}   # dev records generated in evaluation; 0 = all

log() { printf '\n== %s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "environment"
if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q -r requirements-train.txt
fi
PY=.venv/bin/python
"$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok or sys.argv[1] == '0' else 'No CUDA GPU visible; this script is for the GPU run.')" "$REQUIRE_CUDA"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; fi
if [ "$BF16" = auto ]; then
    BF16=$("$PY" -c "import torch; print(int(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))")
fi
BF16_FLAG=(); if [ "$BF16" = 1 ]; then BF16_FLAG=(--bf16); fi
echo "bf16: $BF16"

log "data"
mkdir -p data "$OUT"
[ -f data/safetraffic-v0.1.0.zip ] || "$PY" -m rarecrash download --out data/safetraffic-v0.1.0.zip
[ -f data/processed/train.jsonl ] || "$PY" -m rarecrash prepare --zip data/safetraffic-v0.1.0.zip --out data/processed
"$PY" -m rarecrash audit --data data/processed --out "$OUT/audit.json" >/dev/null
"$PY" - "$OUT/audit.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
got = (a["train"], a["dev"], a["test_count_unopened"], a["rare_train_combinations_5_to_20"])
want = (9625, 1219, 1199, 156)
if got != want:
    sys.exit(f"Split differs from the milestone: {got} != {want}. Stop and compare manifests.")
print("split matches milestone:", got)
EOF
[ -f "$OUT/retrieval_baseline.json" ] || "$PY" -m rarecrash retrieve --data data/processed --out "$OUT/retrieval_baseline.json" >/dev/null

for MODE in uniform frequency-aware; do
    DIR="$OUT/train-$MODE"
    if [ -f "$DIR/run.json" ]; then log "train $MODE: done, skipping"; continue; fi
    log "train $MODE"
    "$PY" -m rarecrash.train --data data/processed --out "$DIR" \
        --model "$MODEL" --revision "$REVISION" --mode "$MODE" \
        --limit 0 --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED" \
        --eval-every "$EVAL_EVERY" --dev-limit "$DEV_LIMIT" --eval-batch-size 32 --select-on dev \
        --save-model ${BF16_FLAG[@]+"${BF16_FLAG[@]}"} 2>&1 | tee "$DIR.log"
done

evaluate() {  # label, model, extra args...
    local label=$1 model=$2; shift 2
    local dir="$OUT/eval/$label"
    if [ -f "$dir/summary.json" ]; then log "eval $label: done, skipping"; return; fi
    log "eval $label"
    "$PY" -m rarecrash.evaluate --data data/processed --model "$model" --out "$dir" \
        --batch-size 32 --limit "$EVAL_LIMIT" "$@" > "$dir.log" 2>&1 || { tail -20 "$dir.log"; exit 1; }
}

mkdir -p "$OUT/eval"
RUNS=()
for MODE in uniform frequency-aware; do
    evaluate "$MODE-greedy" "$OUT/train-$MODE/model" --decoding greedy
    RUNS+=("$MODE-greedy=$OUT/eval/$MODE-greedy")
    for S in $SAMPLE_SEEDS; do
        evaluate "$MODE-sample-s$S" "$OUT/train-$MODE/model" --decoding sample --seed "$S"
        RUNS+=("$MODE-sample=$OUT/eval/$MODE-sample-s$S")
    done
done
evaluate untuned-greedy "$MODEL" --revision "$REVISION" --decoding greedy
RUNS+=("untuned-greedy=$OUT/eval/untuned-greedy")

log "comparison"
"$PY" -m rarecrash.compare --runs "${RUNS[@]}" --out "$OUT/comparison.md"
log "done: results in $OUT (comparison.md, train-*/run.json, eval/*/summary.json)"
