#!/bin/bash
# Deploy + predict + evaluate queue for the remaining 5 trained variants (Windows Git Bash).
# Per tag: WSL merge adapter -> GGUF f16 -> Windows ollama create (q4_K_M, shared chat
# template) -> predictions (primary + fallback 1.05 + escalation 1.15) -> evaluation.
# Failures do not stop the queue; summary greps EVAL_DONE per tag.
set -u
# Git Bash rewrites /root/... args into C:/Program Files/Git/... when calling wsl.exe -
# disable that entirely; WSL-side logic lives in wsl_merge_convert.sh (copied in, CRLF
# stripped) so no Linux paths cross the Windows/WSL boundary as arguments.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'
BASE="C:/Users/01/Desktop/cuad_extractor"
QLOG="$BASE/eval_queue.log"
echo "=== eval queue started $(date) ===" > "$QLOG"

# tag -> adapter dir suffix (ollama model name = cuad-<tag>)
declare -A ADAPTERS=(
  [s43]="cuad-4b-s43-lora"
  [s44]="cuad-4b-s44-lora"
  [n50]="cuad-4b-s42-n50-lora"
  [n150]="cuad-4b-s42-n150-lora"
  [n300]="cuad-4b-s42-n300-lora"
)

for tag in s43 s44 n50 n150 n300; do
  name="cuad-${tag}"
  adapter="${ADAPTERS[$tag]}"
  log="$BASE/deploy_${tag}.log"
  echo "--- $tag: started $(date)" >> "$QLOG"
  {
    echo "=== merge + convert $adapter ==="
    wsl -d Ubuntu-24.04 -- bash -c "tr -d '\r' < /mnt/c/Users/01/Desktop/cuad_extractor/pipeline/wsl_merge_convert.sh > /root/wmc.sh && bash /root/wmc.sh $adapter" \
      || { echo "MERGE/CONVERT FAILED"; exit 1; }
    echo "=== ollama create $name ==="
    sed "1s|.*|FROM $BASE/models/tmp-f16.gguf|" "$BASE/models/Modelfile" > "$BASE/models/Modelfile.tmp"
    ollama create "$name" --quantize q4_K_M -f "$BASE/models/Modelfile.tmp" \
      || { echo "OLLAMA CREATE FAILED"; exit 1; }
    rm -f "$BASE/models/tmp-f16.gguf" "$BASE/models/Modelfile.tmp"
    echo "=== predict $name ==="
    cd "$BASE"
    python pipeline/2_predict.py --model "$name" --prompt short
    python pipeline/2b_fallback.py --model "$name" --prompt short --repeat-penalty 1.05
    python pipeline/2b_fallback.py --model "$name" --prompt short --repeat-penalty 1.15
    echo "=== evaluate $name ==="
    python pipeline/3_evaluate.py --model "$name" --prompt short && echo "EVAL_DONE"
    ollama stop "$name" 2>/dev/null
  } > "$log" 2>&1
  if grep -q "EVAL_DONE" "$log"; then
    echo "--- $tag: OK $(date)" >> "$QLOG"
  else
    echo "--- $tag: FAILED $(date) - see deploy_${tag}.log" >> "$QLOG"
  fi
done
echo "=== eval queue finished $(date) ===" >> "$QLOG"
cat "$QLOG"
