# Step 3b: STRICT full-span-coverage F1, approximating ContractEval's rule
# (arXiv 2508.03080) on the same official CUAD test set (102 contracts x 41 clauses).
# Their definition: TP = label non-empty AND the prediction fully covers the labeled
# span; failing to fully cover a non-empty label is an FN; predicting on an empty
# label is an FP. We apply it to our cached predictions per (contract, clause):
#   covered(g) = some predicted quote contains gold span g (whitespace-normalized,
#   casefolded substring)
#   strict_all: TP only if EVERY gold span for the clause is covered  (primary)
#   strict_any: TP if at least one gold span is covered               (lenient variant)
# Both are reported; strict_all is the harsher, closest reading of their rule.
# Usage: python pipeline/3b_strict_eval.py --model cuad-extractor --prompt short
import json, os, argparse, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, CLAUSES, CANON, norm_key, norm_ws, pred_dir, chunk_filename

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", choices=["defs", "short"], default="defs")
args = parser.parse_args()

rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8")]

gold = {}
for r in rows:
    g = gold.setdefault(r["title"], {c: set() for c in CLAUSES})
    for cl, quotes in json.loads(r["messages"][2]["content"]).items():
        g[cl].update(norm_ws(q) for q in quotes)

PRED_DIR = pred_dir(args.model, args.prompt)
pred = {t: {c: set() for c in CLAUSES} for t in gold}
missing = []
for r in rows:
    p_path = os.path.join(PRED_DIR, chunk_filename(r["name"]))
    if not os.path.exists(p_path):
        missing.append(r["name"])
        continue
    d = json.load(open(p_path, encoding="utf-8"))
    for k, v in (d.items() if isinstance(d, dict) else []):
        cl = CANON.get(norm_key(k))
        if cl is None or not isinstance(v, list):
            continue
        for q in v:
            if isinstance(q, str) and q.strip():
                pred[r["title"]][cl].add(norm_ws(q))
if missing:
    raise SystemExit(f"REFUSING: {len(missing)}/{len(rows)} chunks missing predictions")

def covered(g, P):
    gq = g.casefold()
    return any(gq in p.casefold() for p in P)

# per-contract counts for [strict_all, strict_any]
per_contract = {}
for t in gold:
    counts = [[0, 0, 0], [0, 0, 0]]   # [tp, fp, fn] per variant
    for cl in CLAUSES:
        G, P = gold[t][cl], pred[t][cl]
        if G:
            cov = [covered(g, P) for g in G] if P else []
            hits = (all(cov) if cov else False, any(cov) if cov else False)
            for i in range(2):
                if hits[i]:
                    counts[i][0] += 1
                else:
                    counts[i][2] += 1
        elif P:
            for i in range(2):
                counts[i][1] += 1
    per_contract[t] = counts

def metrics(titles, i):
    tp = sum(per_contract[t][i][0] for t in titles)
    fp = sum(per_contract[t][i][1] for t in titles)
    fn = sum(per_contract[t][i][2] for t in titles)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    return P, R, F

titles = list(gold)
result = {"model": args.model, "prompt": args.prompt, "contracts": len(titles),
          "metric": "full-span-coverage F1 (ContractEval-style approximation)",
          "note": ("TP requires predicted quote(s) to fully contain the gold span(s), "
                   "whitespace-normalized + casefolded substring match")}
for i, name in [(0, "strict_all_spans"), (1, "strict_any_span")]:
    P, R, F = metrics(titles, i)
    random.seed(42)
    fs = sorted(metrics(random.choices(titles, k=len(titles)), i)[2] for _ in range(1000))
    result[name] = {"precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
                    "f1_ci95": [round(fs[24], 4), round(fs[974], 4)]}

out = os.path.join(DATA_DIR,
                   f"eval_strict_{args.model.replace(':', '_').replace('/', '_')}_{args.prompt}.json")
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
print("saved to", out)
