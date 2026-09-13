# Stage 4 (v2, post-review): QLoRA fine-tune Qwen3-4B-Instruct-2507 on CUAD.
# Fixes vs v1: OOM env var set IN the script; over-length examples EXPLICITLY excluded
# and recorded (the one 28,353-token example; 30720 seq len OOMed, 24576 is proven);
# warmup_ratio instead of fixed steps (data-curve safe); checkpoints every 50 steps;
# corpus-wide think-block assert; --n-contracts validated.
# Usage (inside WSL): /root/venv/bin/python 4_train.py --seed 42 [--n-contracts N] [--merge]
import os
# must be set BEFORE torch is imported (unsloth imports torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json, argparse, hashlib

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-contracts", type=int, default=0)
parser.add_argument("--merge", action="store_true")
args = parser.parse_args()
if args.n_contracts < 0:
    raise SystemExit("--n-contracts must be >= 0 (0 = all)")

BASE = "/mnt/c/Users/01/Desktop/cuad_extractor"
DATA = f"{BASE}/data/train.jsonl"
TAG = f"cuad-4b-s{args.seed}" + (f"-n{args.n_contracts}" if args.n_contracts else "")
OUT_ADAPTER = f"{BASE}/models/{TAG}-lora"
MAX_LEN = 24576   # proven config (v1 completed 156 steps); 30720 OOMed in grad-ckpt buffers

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
)
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
if args.n_contracts:
    titles = sorted({r["title"] for r in rows},
                    key=lambda t: hashlib.md5(t.encode()).hexdigest())[:args.n_contracts]
    keep = set(titles)
    rows = [r for r in rows if r["title"] in keep]
print(f"training on {len(rows)} chunk-examples "
      f"({len({r['title'] for r in rows})} contracts), seed {args.seed}, tag {TAG}")

def to_text(ex):
    t = tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                      add_generation_prompt=False)
    t = t.replace("<|im_start|>assistant\n<think>\n\n</think>\n\n",
                  "<|im_start|>assistant\n")
    return {"text": t}

ds = Dataset.from_list(rows)
ds = ds.map(to_text, remove_columns=ds.column_names)
# corpus-wide guards (post-review): no think blocks anywhere; nothing exceeds MAX_LEN
assert all("<think>" not in x for x in ds["text"]), "think block present in rendered corpus"
# EXPLICIT exclusion of over-length examples (documented, not silently truncated/dropped).
# Tokenize EVERY example - a char-based prefilter (MAX_LEN*3 chars) missed the 28,353-token
# example (~65k chars) and let Unsloth silently drop it instead. ~1 min, runs once per job.
lens = [len(tokenizer.encode(x)) for x in ds["text"]]
excluded = [i for i, n in enumerate(lens) if n > MAX_LEN]
if excluded:
    print(f"EXCLUDING {len(excluded)} over-length example(s) (> {MAX_LEN} tokens): "
          f"indices {excluded}, token lengths {[lens[i] for i in excluded]} - "
          f"recorded in train_config.json")
    keep_idx = [i for i in range(len(ds)) if i not in set(excluded)]
    ds = ds.select(keep_idx)
print(f"length guard: {len(ds)} examples within {MAX_LEN} tokens (max {max(lens)})")

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,               # fixed steps would corrupt small-N curve points
        optim="adamw_8bit",
        logging_steps=5,
        save_strategy="steps",           # crash insurance for the overnight queue
        save_steps=50,
        save_total_limit=1,
        output_dir=f"/root/cuad_outputs_{TAG}",
        seed=args.seed,
        report_to="none",
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
stats = trainer.train()
print(f"done in {stats.metrics['train_runtime']/60:.1f} min, "
      f"final loss {stats.metrics['train_loss']:.4f}")

os.makedirs(OUT_ADAPTER, exist_ok=True)
model.save_pretrained(OUT_ADAPTER)
tokenizer.save_pretrained(OUT_ADAPTER)
json.dump({"tag": TAG, "seed": args.seed, "n_contracts": args.n_contracts or 408,
           "max_seq_length": MAX_LEN, "warmup_ratio": 0.06, "lr": 2e-4, "epochs": 2,
           "r": 16, "alpha": 32, "final_loss": round(stats.metrics["train_loss"], 4),
           "train_minutes": round(stats.metrics["train_runtime"] / 60, 1),
           "n_examples": len(ds), "n_excluded_overlength": len(excluded)},
          open(f"{OUT_ADAPTER}/train_config.json", "w"), indent=2)
print(f"adapter + train_config saved to {OUT_ADAPTER}")

if args.merge:
    MERGED = f"{BASE}/models/{TAG}-merged"
    model.save_pretrained_merged(MERGED, tokenizer, save_method="merged_16bit")
    print(f"merged model saved to {MERGED}")
print("ALL_STEPS_DONE")
