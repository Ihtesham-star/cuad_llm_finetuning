# Step 3 (v2, post-review): score cached predictions against gold on the official test set.
# Fixes vs v1: coverage enforced and recorded; run-config embedded in the result;
# casefolded verbatim check; correct bootstrap percentile indices; dropped-key counter.
# Usage: python pipeline/3_evaluate.py --model cuad-extractor --prompt short
import json, os, argparse, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA_DIR, CLAUSES, CANON, norm_key, norm_ws, pred_dir, chunk_filename)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", choices=["defs", "short"], default="defs")
parser.add_argument("--allow-partial", action="store_true",
                    help="score despite missing prediction files (recorded in the result)")
args = parser.parse_args()

rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8")]

# ---- gold, merged per contract ----
gold = {}
for r in rows:
    g = gold.setdefault(r["title"], {c: set() for c in CLAUSES})
    for cl, quotes in json.loads(r["messages"][2]["content"]).items():
        g[cl].update(norm_ws(q) for q in quotes)

# ---- contract full texts, casefolded for the verbatim check ----
ctx = {c["title"]: norm_ws(c["paragraphs"][0]["context"]).casefold()
       for c in json.load(open(os.path.join(DATA_DIR, "CUAD_v1", "CUAD_v1.json"),
                               encoding="utf-8"))["data"] if c["title"] in gold}

# ---- predictions, merged per contract; coverage tracked ----
PRED_DIR = pred_dir(args.model, args.prompt)
run_config = None
cfg_path = os.path.join(PRED_DIR, "_run_config.json")
if os.path.exists(cfg_path):
    run_config = json.load(open(cfg_path, encoding="utf-8"))

pred = {t: {c: set() for c in CLAUSES} for t in gold}
n_scored, dropped_keys = 0, 0
missing = []
for r in rows:
    p_path = os.path.join(PRED_DIR, chunk_filename(r["name"]))
    if not os.path.exists(p_path):
        missing.append(r["name"])
        continue
    n_scored += 1
    d = json.load(open(p_path, encoding="utf-8"))
    for k, v in (d.items() if isinstance(d, dict) else []):
        cl = CANON.get(norm_key(k))
        if cl is None:
            dropped_keys += 1
            continue
        if not isinstance(v, list):
            continue
        for q in v:
            if isinstance(q, str) and q.strip():
                pred[r["title"]][cl].add(norm_ws(q))

if missing and not args.allow_partial:
    raise SystemExit(f"REFUSING to evaluate: {len(missing)}/{len(rows)} chunks missing "
                     f"predictions ({missing[:3]}...). Complete the prediction run or pass "
                     f"--allow-partial to score anyway (coverage will be recorded).")

def tok_f1(a, b):
    wa, wb = a.lower().split(), b.lower().split()
    if not wa or not wb: return 0.0
    common = {}
    for w in wa: common[w] = common.get(w, 0) + 1
    inter = sum(min(common.get(w, 0), wb.count(w)) for w in set(wb))
    if inter == 0: return 0.0
    p, r = inter / len(wa), inter / len(wb)
    return 2 * p * r / (p + r)

per_contract = {}
for t in gold:
    tp = fp = fn = 0
    ov_sum, ov_n, quotes, verb = 0.0, 0, 0, 0
    for cl in CLAUSES:
        g, p = gold[t][cl], pred[t][cl]
        if g and p:
            tp += 1
            ov_sum += max(tok_f1(pq, gq) for pq in p for gq in g)
            ov_n += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
        for pq in p:
            quotes += 1
            if pq.casefold() in ctx[t]:     # casefolded: ALL-CAPS headings re-cased by a
                verb += 1                   # model still count as verbatim copies
    per_contract[t] = [tp, fp, fn, ov_sum, ov_n, quotes, verb]

def metrics(titles):
    tp = sum(per_contract[t][0] for t in titles)
    fp = sum(per_contract[t][1] for t in titles)
    fn = sum(per_contract[t][2] for t in titles)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    return P, R, F

titles = list(gold)
P, R, F = metrics(titles)
ov_sum = sum(v[3] for v in per_contract.values())
ov_n = sum(v[4] for v in per_contract.values())
quotes = sum(v[5] for v in per_contract.values())
verb = sum(v[6] for v in per_contract.values())

random.seed(42)
fs = sorted(metrics(random.choices(titles, k=len(titles)))[2] for _ in range(1000))
lo, hi = fs[24], fs[974]    # 2.5th / 97.5th percentile of 1000 sorted resamples

result = {
    "model": args.model, "prompt": args.prompt,
    "contracts": len(titles), "decisions": len(titles) * len(CLAUSES),
    "chunks_scored": n_scored, "chunks_total": len(rows),
    "missing_chunks": missing,
    "dropped_prediction_keys": dropped_keys,
    "run_config": run_config,
    "detection_precision": round(P, 4), "detection_recall": round(R, 4),
    "detection_f1": round(F, 4), "detection_f1_ci95": [round(lo, 4), round(hi, 4)],
    "overlap_tokenF1_on_TP": round(ov_sum / ov_n, 4) if ov_n else None,
    "verbatim_rate_casefolded": round(verb / quotes, 4) if quotes else None,
    "total_predicted_quotes": quotes,
}
out = os.path.join(DATA_DIR,
                   f"eval_{args.model.replace(':', '_').replace('/', '_')}_{args.prompt}.json")
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
print("saved to", out)
