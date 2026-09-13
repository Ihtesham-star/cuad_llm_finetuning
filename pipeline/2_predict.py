# Step 2: run a model over the test chunks and cache predictions.
#   --prompt defs  : zero-shot baseline (official clause definitions in prompt)
#   --prompt short : fine-tuned models (definitions live in the weights)
# Protocol discipline (post-review):
#   - num_predict / temperature are recorded in <PRED_DIR>/_run_config.json;
#     a cache created under different settings REFUSES to continue (use --fresh to wipe)
#   - failures are never cached; invalid cached JSON is swept and re-queried
# Usage: python pipeline/2_predict.py --model qwen3:14b --prompt defs
import json, os, time, argparse, urllib.request, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, SYSTEM_SHORT, USER_SHORT, USER_DEFS, pred_dir, chunk_filename

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", choices=["defs", "short"], default="defs")
parser.add_argument("--num-predict", type=int, default=12288)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--fresh", action="store_true", help="wipe cache and start over")
args = parser.parse_args()

PRED_DIR = pred_dir(args.model, args.prompt)
os.makedirs(PRED_DIR, exist_ok=True)
CONFIG = {"model": args.model, "prompt": args.prompt,
          "num_predict": args.num_predict, "temperature": args.temperature,
          "num_ctx_rule": "min(40960, max(8192, ceil((chars/4.5 + 800 + num_predict)/2048)*2048))"}
cfg_path = os.path.join(PRED_DIR, "_run_config.json")
if args.fresh:
    for f in os.listdir(PRED_DIR):
        os.remove(os.path.join(PRED_DIR, f))
if os.path.exists(cfg_path):
    old = json.load(open(cfg_path, encoding="utf-8"))
    if old != CONFIG:
        raise SystemExit(f"cache at {PRED_DIR} was built under different settings:\n"
                         f"  cached: {old}\n  requested: {CONFIG}\n"
                         f"Use --fresh to wipe it, or match the settings.")
else:
    json.dump(CONFIG, open(cfg_path, "w", encoding="utf-8"), indent=2)

# sweep invalid cached files (never legitimate: every valid prediction parses)
for f in list(os.listdir(PRED_DIR)):
    if f.startswith("_"):
        continue
    try:
        json.load(open(os.path.join(PRED_DIR, f), encoding="utf-8"))
    except Exception:
        os.remove(os.path.join(PRED_DIR, f))
        print(f"swept invalid cache entry {f}")

def ask(chunk_text):
    user = (USER_SHORT if args.prompt == "short" else USER_DEFS) + chunk_text
    est = int(len(user) / 4.5) + 800 + args.num_predict
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM_SHORT},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "format": "json",
        "options": {"num_ctx": min(40960, max(8192, (est // 2048 + 1) * 2048)),
                    "temperature": args.temperature, "num_predict": args.num_predict},
    }
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read())
    if "message" not in result:
        raise RuntimeError(result.get("error", str(result)[:200]))
    return json.loads(result["message"]["content"])

rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8")]
if args.limit:
    rows = rows[:args.limit]
t0, n_err = time.time(), 0
for i, r in enumerate(rows):
    pred_path = os.path.join(PRED_DIR, chunk_filename(r["name"]))
    if os.path.exists(pred_path):
        continue
    try:
        chunk_text = r["messages"][1]["content"].split("CONTRACT EXCERPT:\n", 1)[1]
        pred = ask(chunk_text)
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False)
        status = "ok"
    except Exception as e:
        n_err += 1
        status = f"ERROR (not cached): {e}"
    print(f"{i+1}/{len(rows)} {r['name'][:60]} | {status} | {time.time()-t0:.0f}s elapsed",
          flush=True)
done = sum(1 for r in rows if os.path.exists(os.path.join(PRED_DIR, chunk_filename(r["name"]))))
print(f"done: {done}/{len(rows)} chunks cached, {n_err} errors this run, "
      f"{(time.time()-t0)/60:.1f} min")
if done < len(rows):
    print(f"WARNING: cache incomplete ({done}/{len(rows)}) - evaluation will refuse "
          f"unless --allow-partial; at temperature 0 identical retries will fail "
          f"identically, so investigate the failing chunks instead of re-running blindly")
