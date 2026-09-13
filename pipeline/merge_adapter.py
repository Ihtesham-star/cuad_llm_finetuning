# Merge a saved LoRA adapter into fp16 weights (WSL side).
# Same path as 4_train.py --merge: base loaded 4-bit, save_pretrained_merged 16-bit,
# so all deployed variants are produced identically.
# Usage: /root/venv/bin/python merge_adapter.py <adapter_dir> <out_dir>
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from unsloth import FastLanguageModel

adapter, out = sys.argv[1], sys.argv[2]
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=adapter, max_seq_length=24576, load_in_4bit=True)
model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
print("MERGED_OK", out)
