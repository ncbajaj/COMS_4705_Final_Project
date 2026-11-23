#!/usr/bin/env bash
# Full FLAN-T5-base experiment on one GPU machine, from a fresh clone:
#   1. environment and data (the split is checked against the milestone numbers)
#   2. uniform and frequency-aware training for each of TRAIN_SEEDS, identical
#      except for the sampler (ALPHA sets the frequency-aware exponent)
#   3. dev evaluation: greedy plus several sampling seeds per model, and the
#      untuned FLAN-T5-base control (label control-untuned)
#   4. event-level diversity: K samples per rare condition tuple per model,
#      compared with K real training narratives and with retrieval
#   5. one comparison table (mean ± sd over training and sampling seeds,
#      plus bootstrap CIs on the rare slice)
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
SCENARIO=${SCENARIO:-0}       # 1: targets include the collision-type scenario sentence
if [ "$SCENARIO" = 1 ]; then OUT=${OUT:-runs/gpu-scenario}; else OUT=${OUT:-runs/gpu}; fi
MODEL=${MODEL:-google/flan-t5-base}
REVISION=${REVISION:-7bcac572ce56db69c1ea7c8af255c5d7c9672fc2}  # flan-t5-base, resolved 2026-11-23
STEPS=${STEPS:-3000}          # about 5 epochs of the 9,625 training records at batch 16
BATCH=${BATCH:-16}
LR=${LR:-3e-4}
EVAL_EVERY=${EVAL_EVERY:-250}
TRAIN_SEEDS=${TRAIN_SEEDS:-"4705 4706 4707"}  # one training run per mode per seed
ALPHA=${ALPHA:-0.5}           # frequency-aware weight = count ** -ALPHA
SAMPLE_SEEDS=${SAMPLE_SEEDS:-"1 2 3"}
BF16=${BF16:-auto}            # auto: on if the GPU supports bfloat16
# Smoke-test overrides only; leave at their defaults for the real run.
REQUIRE_CUDA=${REQUIRE_CUDA:-1}
DEV_LIMIT=${DEV_LIMIT:-0}     # dev records scored during training; 0 = all
EVAL_LIMIT=${EVAL_LIMIT:-0}   # dev records generated in evaluation; 0 = all
DIVERSITY_LIMIT=${DIVERSITY_LIMIT:-0}  # rare tuples in the diversity stage; 0 = all 156

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
SC_FLAG=(); SC_DIV_FLAG=()
if [ "$SCENARIO" = 1 ]; then SC_FLAG=(--scenario-sentence); SC_DIV_FLAG=(--scenario); fi
echo "scenario sentence: $SCENARIO (results in $OUT)"

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

ALPHA_FLAG=(--sampling-alpha "$ALPHA")

for SEED in $TRAIN_SEEDS; do
    for MODE in uniform frequency-aware; do
        DIR="$OUT/train-$MODE-s$SEED"
        if [ -f "$DIR/run.json" ]; then log "train $MODE seed $SEED: done, skipping"; continue; fi
        log "train $MODE seed $SEED"
        EXTRA=(); if [ "$MODE" = frequency-aware ]; then EXTRA=("${ALPHA_FLAG[@]}"); fi
        mkdir -p "$OUT"
        "$PY" -m rarecrash.train --data data/processed --out "$DIR" \
            --model "$MODEL" --revision "$REVISION" --mode "$MODE" \
            ${EXTRA[@]+"${EXTRA[@]}"} \
            --limit 0 --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED" \
            --eval-every "$EVAL_EVERY" --dev-limit "$DEV_LIMIT" --eval-batch-size 32 --select-on dev \
            --save-model ${BF16_FLAG[@]+"${BF16_FLAG[@]}"} ${SC_FLAG[@]+"${SC_FLAG[@]}"} \
            2>&1 | tee "$DIR.log"
    done
done

evaluate() {  # label, model, extra args...
    local label=$1 model=$2; shift 2
    local dir="$OUT/eval/$label"
    if [ -f "$dir/summary.json" ]; then log "eval $label: done, skipping"; return; fi
    log "eval $label"
    "$PY" -m rarecrash.evaluate --data data/processed --model "$model" --out "$dir" \
        --batch-size 32 --limit "$EVAL_LIMIT" ${SC_FLAG[@]+"${SC_FLAG[@]}"} "$@" > "$dir.log" 2>&1 || { tail -20 "$dir.log"; exit 1; }
}

mkdir -p "$OUT/eval" "$OUT/diversity"
RUNS=()
DRUNS=()
for SEED in $TRAIN_SEEDS; do
    for MODE in uniform frequency-aware; do
        MODEL_DIR="$OUT/train-$MODE-s$SEED/model"
        evaluate "$MODE-s$SEED-greedy" "$MODEL_DIR" --decoding greedy
        RUNS+=("$MODE-greedy=$OUT/eval/$MODE-s$SEED-greedy")
        for S in $SAMPLE_SEEDS; do
            evaluate "$MODE-s$SEED-sample-s$S" "$MODEL_DIR" --decoding sample --seed "$S"
            RUNS+=("$MODE-sample=$OUT/eval/$MODE-s$SEED-sample-s$S")
        done
    done
done
evaluate control-untuned "$MODEL" --revision "$REVISION" --decoding greedy
RUNS+=("control-untuned=$OUT/eval/control-untuned")

for SEED in $TRAIN_SEEDS; do
    for MODE in uniform frequency-aware; do
        DIR="$OUT/diversity/$MODE-s$SEED"
        DRUNS+=("$MODE=$DIR")
        if [ -f "$DIR/summary.json" ]; then log "diversity $MODE seed $SEED: done, skipping"; continue; fi
        log "diversity $MODE seed $SEED"
        "$PY" -m rarecrash.diversity --data data/processed --model "$OUT/train-$MODE-s$SEED/model" \
            --out "$DIR" --k 5 --seed "$SEED" --limit-tuples "$DIVERSITY_LIMIT" \
            ${SC_DIV_FLAG[@]+"${SC_DIV_FLAG[@]}"} \
            > "$DIR.log" 2>&1 || { tail -20 "$DIR.log"; exit 1; }
    done
done

log "comparison"
"$PY" -m rarecrash.compare --bootstrap 1000 --runs "${RUNS[@]}" --out "$OUT/comparison.md"
"$PY" -m rarecrash.compare --diversity --runs "${DRUNS[@]}" | tee -a "$OUT/comparison.md"
log "done: results in $OUT (comparison.md, train-*/run.json, eval/*/summary.json)"
