# Step 2b: deterministic fallback decode for chunks whose PRIMARY greedy decode failed.
# Failure mode: at temperature 0 some dense chunks enter a repetition loop (same 1-2
# sentences forever) and hit the num_predict cap mid-string -> invalid JSON, never cached.
# Policy (uniform across all models): one fallback decode with repeat_penalty 1.05,
# still temperature 0 (deterministic + repeatable). The fallback chunks and settings are
# recorded in _run_config.json, which 3_evaluate.py embeds in the published result.
# Usage: python pipeline/2b_fallback.py --model cuad-extractor --prompt short
import json, os, time, argparse, urllib.request, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, SYSTEM_SHORT, USER_SHORT, USER_DEFS, pred_dir, chunk_filename

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", choices=["defs", "short"], default="defs")
parser.add_argument("--repeat-penalty", type=float, default=1.05)
args = parser.parse_args()

PRED_DIR = pred_dir(args.model, args.prompt)
cfg_path = os.path.join(PRED_DIR, "_run_config.json")
if not os.path.exists(cfg_path):
    raise SystemExit("no _run_config.json - run the primary pass (2_predict.py) first")
config = json.load(open(cfg_path, encoding="utf-8"))

rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8")]
missing = [r for r in rows
           if not os.path.exists(os.path.join(PRED_DIR, chunk_filename(r["name"])))]
if not missing:
    print("nothing to do: all chunks cached")
    raise SystemExit(0)
print(f"fallback pass over {len(missing)} failed chunk(s), "
      f"repeat_penalty={args.repeat_penalty}, temperature={config['temperature']}")

def ask(chunk_text):
    user = (USER_SHORT if args.prompt == "short" else USER_DEFS) + chunk_text
    est = int(len(user) / 4.5) + 800 + config["num_predict"]
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM_SHORT},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "format": "json",
        "options": {"num_ctx": min(40960, max(8192, (est // 2048 + 1) * 2048)),
                    "temperature": config["temperature"],
                    "num_predict": config["num_predict"],
                    "repeat_penalty": args.repeat_penalty},
    }
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read())
    if "message" not in result:
        raise RuntimeError(result.get("error", str(result)[:200]))
    return json.loads(result["message"]["content"])

t0, recovered = time.time(), []
for i, r in enumerate(missing):
    try:
        chunk_text = r["messages"][1]["content"].split("CONTRACT EXCERPT:\n", 1)[1]
        pred = ask(chunk_text)
        with open(os.path.join(PRED_DIR, chunk_filename(r["name"])), "w",
                  encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False)
        recovered.append(r["name"])
        status = "recovered"
    except Exception as e:
        status = f"STILL FAILING: {e}"
    print(f"{i+1}/{len(missing)} {r['name'][:60]} | {status} | "
          f"{time.time()-t0:.0f}s elapsed", flush=True)

# auditable record: which chunks were decoded under the fallback, and with what settings
# (a list - escalation passes with stronger penalties append instead of overwriting)
config.setdefault("fallbacks", []).append(
    {"repeat_penalty": args.repeat_penalty,
     "temperature": config["temperature"],
     "reason": "primary greedy decode hit repetition loop / invalid JSON",
     "chunks_recovered": recovered,
     "chunks_still_failing": [r["name"] for r in missing
                              if r["name"] not in recovered]})
json.dump(config, open(cfg_path, "w", encoding="utf-8"), indent=2)
done = sum(1 for r in rows if os.path.exists(os.path.join(PRED_DIR, chunk_filename(r["name"]))))
print(f"coverage now {done}/{len(rows)}; fallback recorded in _run_config.json")
