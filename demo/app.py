# Local demo UI for the CUAD clause extractor. Talks to Ollama on localhost.
# Setup (once):  see "Local demo" in the repo README - download the GGUF, then
#                ollama create cuad-extractor -f demo/Modelfile
# Run:           pip install gradio && python demo/app.py
import json, os, time, html, urllib.request

import gradio as gr

MODEL = os.environ.get("CUAD_MODEL", "cuad-extractor")
MAX_CHARS = 60_000        # one pipeline chunk; longer contracts need chunk+merge
NUM_PREDICT = 12_288

# Prompts must match training byte-for-byte (pipeline/common.py).
SYSTEM_SHORT = ("You are a contract review system. Extract the requested clause types from the "
                "contract excerpt. Quote the contract text verbatim. A clause type not present "
                "in this excerpt gets an empty list. Reply with JSON only.")
USER_SHORT = ("Extract all 41 CUAD clause types from this contract excerpt as JSON "
              "(keys: clause types, values: lists of verbatim quotes).\n\nCONTRACT EXCERPT:\n")

EXAMPLES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "examples.json"), encoding="utf-8"))


def run_model(text, repeat_penalty):
    user = USER_SHORT + text
    est = int(len(user) / 4.5) + 800 + NUM_PREDICT
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_SHORT},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "format": "json",
        "options": {"num_ctx": min(40960, max(8192, (est // 2048 + 1) * 2048)),
                    "temperature": 0.0, "num_predict": NUM_PREDICT,
                    "repeat_penalty": repeat_penalty},
    }
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read())
    if "message" not in result:
        raise RuntimeError(result.get("error", "no message in Ollama response"))
    return json.loads(result["message"]["content"])


def render(pred, elapsed):
    found = {k: v for k, v in pred.items() if v}
    head = (f"**{len(found)} of 41 clause types found** ({elapsed:.1f}s). "
            f"Clause types not listed were reported absent.")
    if not found:
        return head, pred
    rows = []
    for cl, quotes in found.items():
        qs = "<br>".join(f"“{html.escape(q)}”" for q in quotes)
        rows.append(f"<tr><td style='white-space:nowrap;vertical-align:top;"
                    f"padding:6px 12px 6px 0'><b>{html.escape(cl)}</b></td>"
                    f"<td style='padding:6px 0'>{qs}</td></tr>")
    return head + "\n\n<table>" + "".join(rows) + "</table>", pred


def extract(text):
    text = (text or "").strip()
    if not text:
        return "Paste some contract text first.", {}
    if len(text) > MAX_CHARS:
        return (f"Input is {len(text):,} characters; this demo takes one chunk "
                f"(up to {MAX_CHARS:,}). Split longer contracts as in the pipeline."), {}
    t0 = time.time()
    err = None
    for rp in (1.0, 1.05, 1.15):     # anti-repetition-loop ladder from the eval protocol
        try:
            return render(run_model(text, rp), time.time() - t0)
        except Exception as e:
            err = e
    return f"Extraction failed: {err}", {}


DESC = """Paste contract text and the model extracts all
[41 CUAD clause types](https://www.atticusprojectai.org/cuad) in one pass, quoting the
contract verbatim. Runs entirely on this machine through Ollama - nothing leaves it.

[Weights & model card](https://huggingface.co/Ihteshamstar/qwen3-4b-cuad-extractor) ·
[Code & evaluation](https://github.com/Ihtesham-star/cuad_llm_finetuning)"""

with gr.Blocks(title="CUAD Clause Extractor") as demo:
    gr.Markdown("# CUAD Clause Extractor (Qwen3-4B fine-tune, local)")
    gr.Markdown(DESC)
    inp = gr.Textbox(lines=14, label="Contract text",
                     placeholder="Paste a contract or excerpt here (up to ~60k characters)...")
    btn = gr.Button("Extract clauses", variant="primary")
    out_md = gr.Markdown()
    with gr.Accordion("Raw JSON (all 41 clause types)", open=False):
        out_json = gr.JSON()
    gr.Examples(examples=[[e["text"]] for e in EXAMPLES],
                example_labels=[e["name"] for e in EXAMPLES],
                inputs=[inp], label="Example excerpts (CUAD test set)")
    btn.click(extract, inputs=[inp], outputs=[out_md, out_json])

demo.launch()
