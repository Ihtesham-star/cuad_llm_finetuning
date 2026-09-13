#!/bin/bash
# Runs INSIDE WSL: merge one LoRA adapter -> fp16 -> GGUF f16, copy result to Windows.
# Usage: bash wsl_merge_convert.sh <adapter_dir_name>   (dir under cuad_extractor/models)
set -e
BASE=/mnt/c/Users/01/Desktop/cuad_extractor
ADAPTER="$1"
/root/venv/bin/python "$BASE/pipeline/merge_adapter.py" "$BASE/models/$ADAPTER" /root/merge_tmp
/root/venv/bin/python /root/llama.cpp/convert_hf_to_gguf.py /root/merge_tmp \
    --outfile /root/tmp-f16.gguf --outtype f16
cp /root/tmp-f16.gguf "$BASE/models/tmp-f16.gguf"
rm -rf /root/merge_tmp /root/tmp-f16.gguf
echo WSL_CONVERT_DONE
