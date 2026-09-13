# Step 1: convert CUAD into chunk-level training/eval JSONL using the OFFICIAL split.
#  - one example per contract-chunk; target = JSON with all 41 clause keys
#    (list of verbatim quotes; empty list = clause absent  -> explicit absence, NCCR-style)
#  - chunking: ~60k chars (~12-14k tokens) with 8k overlap; gold spans assigned by offsets
#  - training rows use the SHORT prompt (definitions live in the weights after fine-tuning);
#    the zero-shot baseline adds definitions at call time instead
import json, csv, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CHUNK, OVERLAP = 60_000, 8_000

D = json.load(open(os.path.join(DATA, "CUAD_v1", "CUAD_v1.json"), encoding="utf-8"))["data"]
manifest = json.load(open(os.path.join(DATA, "split_manifest.json"), encoding="utf-8"))
train_set, test_set = set(manifest["train"]), set(manifest["test"])

# canonical clause order from the official definitions file
defs = list(csv.reader(open(os.path.join(DATA, "category_descriptions.csv"), encoding="utf-8-sig")))[1:]
CLAUSES = [r[0].replace("Category: ", "").strip() for r in defs]
assert len(CLAUSES) == 41, len(CLAUSES)

SYSTEM_SHORT = ("You are a contract review system. Extract the requested clause types from the "
                "contract excerpt. Quote the contract text verbatim. A clause type not present "
                "in this excerpt gets an empty list. Reply with JSON only.")
USER_TASK = ("Extract all 41 CUAD clause types from this contract excerpt as JSON "
             "(keys: clause types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n")

def norm_name(s):
    # the qas ids title-case every word ("Termination For Convenience", "Ip Ownership...")
    # while the CSV uses natural casing - match on a normalized key, output the CSV name
    return re.sub(r"[^a-z0-9]", "", s.lower())

CANON = {norm_name(cl): cl for cl in CLAUSES}

def clause_of(qid):
    raw = qid.split("__")[-1]
    canon = CANON.get(norm_name(raw))
    if canon is None:
        raise SystemExit(f"unmapped clause name in qas id: {raw!r}")
    return canon

# post-review guards: fail LOUD before writing anything
all_titles = {c["title"] for c in D}
unknown = all_titles - train_set - test_set
assert not unknown, f"titles in data but not in split manifest (would corrupt the split): {sorted(unknown)[:3]}"
pre_max_span = max((len(a["text"]) for c in D for q in c["paragraphs"][0]["qas"]
                    for a in q.get("answers", [])), default=0)
assert pre_max_span < OVERLAP, (f"a gold span ({pre_max_span} chars) exceeds the overlap "
                                f"({OVERLAP}) and would be silently dropped from every chunk - "
                                f"increase OVERLAP before converting")

max_span = 0
rows_train, rows_test = [], []
assigned_distinct = set()   # (title, clause, answer_start) - loss check that can SEE loss
for c in D:
    title = c["title"]
    ctx = c["paragraphs"][0]["context"]
    qas = c["paragraphs"][0]["qas"]
    # collect gold spans per clause with offsets
    gold = {cl: [] for cl in CLAUSES}
    for q in qas:
        cl = clause_of(q["id"])   # raises on unmapped names instead of silently dropping
        for a in q.get("answers", []):
            gold[cl].append((a["answer_start"], a["text"]))
            max_span = max(max_span, len(a["text"]))
    # chunk
    starts = list(range(0, max(1, len(ctx) - OVERLAP), CHUNK - OVERLAP))
    for ci, s in enumerate(starts):
        e = min(s + CHUNK, len(ctx))
        chunk_text = ctx[s:e]
        chunk_gold = {cl: [] for cl in CLAUSES}
        for cl, spans in gold.items():
            for (st, txt) in spans:
                if st >= s and st + len(txt) <= e:
                    assigned_distinct.add((title, cl, st))
                    t = " ".join(txt.split())
                    if t not in chunk_gold[cl]:
                        chunk_gold[cl].append(t)
        row = {
            "name": f"{title}__chunk{ci}",
            "title": title,
            "chunk_index": ci,
            "n_chunks": len(starts),
            "messages": [
                {"role": "system", "content": SYSTEM_SHORT},
                {"role": "user", "content": USER_TASK + chunk_text},
                {"role": "assistant", "content": json.dumps(chunk_gold, ensure_ascii=False)},
            ],
        }
        (rows_train if title in train_set else rows_test).append(row)

for fname, rows in [("train.jsonl", rows_train), ("test.jsonl", rows_test)]:
    with open(os.path.join(DATA, fname), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{fname}: {len(rows)} chunk-examples")

print(f"train contracts: {len(train_set)} | test contracts: {len(test_set)}")
print(f"longest gold span: {max_span:,} chars (must be < overlap {OVERLAP:,} to never be cut: "
      f"{'OK' if max_span < OVERLAP else 'WARNING - increase overlap'})")
# loss check on DISTINCT spans (a per-chunk count can exceed the original via overlap
# duplication and therefore cannot detect loss - post-review fix)
orig_distinct = {(c["title"], q["id"].split("__")[-1] and clause_of(q["id"]), a["answer_start"])
                 for c in D for q in c["paragraphs"][0]["qas"] for a in q.get("answers", [])}
lost = orig_distinct - assigned_distinct
assert not lost, f"{len(lost)} gold spans assigned to NO chunk: {sorted(lost)[:3]}"
print(f"gold spans: {len(orig_distinct):,} distinct, all assigned to at least one chunk")
