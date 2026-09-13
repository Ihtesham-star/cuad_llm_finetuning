#!/bin/bash
# Overnight training queue (runs inside WSL). Each run logs separately, failures do not
# stop the queue, and the summary greps for the ALL_STEPS_DONE sentinel per run.
# Runs: seeds 42/43/44 on full data (42 also merges for deployment) + data curve 50/150/300.
BASE=/mnt/c/Users/01/Desktop/cuad_extractor
PY=/root/venv/bin/python
cp $BASE/pipeline/4_train.py /root/cuad_train.py
QLOG=$BASE/queue.log
echo "=== queue started $(date) ===" > "$QLOG"

run() {
    local tag="$1"; shift
    local log="$BASE/train_${tag}.log"
    echo "--- $tag: started $(date)" >> "$QLOG"
    $PY /root/cuad_train.py "$@" > "$log" 2>&1
    local code=$?
    if grep -q "ALL_STEPS_DONE" "$log"; then
        echo "--- $tag: OK (exit $code) $(date)" >> "$QLOG"
    else
        echo "--- $tag: FAILED (exit $code) $(date) - see train_${tag}.log" >> "$QLOG"
    fi
}

run s42full  --seed 42 --merge
run s43full  --seed 43
run s44full  --seed 44
run s42n50   --seed 42 --n-contracts 50
run s42n150  --seed 42 --n-contracts 150
run s42n300  --seed 42 --n-contracts 300

echo "=== queue finished $(date) ===" >> "$QLOG"
cat "$QLOG"
