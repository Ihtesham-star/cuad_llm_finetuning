# Single source of truth for prompts, clause canon, and naming.
# Imported by 1_convert / 2_predict / 3_evaluate / 4_train - the strings below are
# BYTE-IDENTICAL to what data/train.jsonl was built with; editing them breaks
# train/serve alignment for every already-trained model.
import csv, os, re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

_defs = list(csv.reader(open(os.path.join(DATA_DIR, "category_descriptions.csv"),
                             encoding="utf-8-sig")))[1:]
CLAUSES = [r[0].replace("Category: ", "").strip() for r in _defs]
assert len(CLAUSES) == 41, f"expected 41 clause types, got {len(CLAUSES)}"

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

CANON = {norm_key(c): c for c in CLAUSES}
assert len(CANON) == 41, "clause names collide after normalization"

DEF_LINES = "\n".join(f"- {r[0].replace('Category: ', '').strip()}: "
                      f"{r[1].replace('Description: ', '').strip()}" for r in _defs)

SYSTEM_SHORT = ("You are a contract review system. Extract the requested clause types from the "
                "contract excerpt. Quote the contract text verbatim. A clause type not present "
                "in this excerpt gets an empty list. Reply with JSON only.")
USER_SHORT = ("Extract all 41 CUAD clause types from this contract excerpt as JSON "
              "(keys: clause types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n")
USER_DEFS = ("Extract the following 41 clause types from the contract excerpt. For each clause "
             "type, return a list of VERBATIM quotes from the excerpt (copy exactly, do not "
             "paraphrase). Most clause types are NOT present in any given excerpt - return an "
             "empty list for those; never invent or stretch a quote. Return ONE JSON object "
             "with all 41 clause types as keys.\n\nCLAUSE TYPE DEFINITIONS:\n" + DEF_LINES +
             "\n\nCONTRACT EXCERPT:\n")

def pred_dir(model: str, prompt: str) -> str:
    return os.path.join(DATA_DIR, "predictions",
                        f"{model.replace(':', '_').replace('/', '_')}_{prompt}")

def chunk_filename(name: str) -> str:
    return f"{name}.json".replace("/", "_")

def norm_ws(s) -> str:
    return " ".join(str(s).split())
