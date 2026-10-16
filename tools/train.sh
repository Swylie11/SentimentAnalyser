#!/usr/bin/env bash
# Trains Wren 1.5 unattended: builds the data and databases if they're missing, then
# runs a multi-seed training. Safe to re-run; finished steps are skipped.
#
#   bash tools/train.sh                     # 5 seeds x 15,000 batches, about 6 hours
#   SEEDS="0 1 2" bash tools/train.sh       # about 3.5 hours
#
# Override with environment variables: SEEDS, BATCHES, BATCH_SIZE, WORKDIR.
# Everything is printed and also logged to $WORKDIR/train_<timestamp>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1 2 3 4"}
BATCHES=${BATCHES:-15000}
BATCH_SIZE=${BATCH_SIZE:-32}
WORKDIR=${WORKDIR:-wren-work}

mkdir -p "$WORKDIR"
LOG="$WORKDIR/train_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

# Keep Windows awake while this script runs. The request ends when the helper is
# killed, so normal sleep settings come back on their own. Closing the lid still sleeps.
python -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(10**9)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null || true' EXIT

# Data/TestData.jsonl should be Wren 1.0's training set, exported from
# sentiment/test_data.db (test_dataset4), because Data/eval_test.jsonl excludes exactly
# those reviews. This fallback samples a DIFFERENT 100,000 that overlap 141 of the test
# reviews, so only rely on it if you rebuild the test set too.
if [[ ! -f Data/TestData.jsonl || ! -f Data/WordEmbeddings.jsonl ]]; then
    echo "== Building Data/TestData.jsonl and Data/WordEmbeddings.jsonl =="
    # PYTHONPATH=src: the pip-installed wren leaves out wren.train, so use the repo's copy.
    PYTHONPATH=src python -u -m wren.train.real_data_setup --reviews-gz Data/Software.jsonl.gz \
        --vec Data/wiki-news-300d-1M.vec --per-class 20000 --seed 0 --out-dir Data
fi

if [[ ! -f "$WORKDIR/word_embeddings.db" ]]; then
    echo "== Building the databases in $WORKDIR =="
    python -u tools/wren15.py setup --workdir "$WORKDIR" \
        --reviews Data/TestData.jsonl --embeddings Data/WordEmbeddings.jsonl
fi

echo "== Training: seeds $SEEDS, $BATCHES batches of $BATCH_SIZE =="
# --yes: the overwrite prompt would otherwise wait all night. A backup is still made.
# shellcheck disable=SC2086  # SEEDS is meant to split into separate arguments
python -u tools/wren15.py train --workdir "$WORKDIR" --batches "$BATCHES" \
    --batch-size "$BATCH_SIZE" --seeds $SEEDS --yes

echo "== Done. Loss video: python tools/visualise.py loss --file $WORKDIR/seed_runs.json =="
