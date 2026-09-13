# Fine-tuning Qwen3-4B on CUAD contract clause extraction

I fine-tuned Qwen3-4B-Instruct with QLoRA on [CUAD](https://www.atticusprojectai.org/cuad)
(510 commercial contracts, 41 clause types, annotated by lawyers). The model takes a
contract chunk and returns one JSON object with all 41 clause types, quoting the
contract verbatim, with an empty list for clause types that aren't there. The q4_K_M
quantization is 2.5 GB and runs on a consumer GPU through Ollama.

Weights and model card: [Ihteshamstar/qwen3-4b-cuad-extractor](https://huggingface.co/Ihteshamstar/qwen3-4b-cuad-extractor)

This is the public follow-up to a private project where the same recipe (small model +
QLoRA + explicit-absence training) replaced a cloud API for medical document extraction.
CUAD is the public, verifiable version of that claim.

## Results

Official CUAD split: 408 training contracts, 102 test contracts, no overlap. All
numbers below are on the 102 test contracts at full coverage (every chunk scored).

Detection F1 - for each (contract, clause type), did the model find the clause types
that are present and stay quiet about the ones that aren't. 95% CIs are bootstrap over
contracts.

| model | detection F1 | quote overlap (token F1) | verbatim rate |
|---|---|---|---|
| fine-tuned 4B, seed 42 | 0.900 [0.887-0.913] | 0.938 | 97.2% |
| Qwen3-14B zero-shot, official definitions in prompt | 0.816 [0.802-0.830] | 0.729 | 85.4% |

Three seeds under identical config give F1 0.8995 ± 0.0017, so the result is not a
lucky seed. Published weights are seed 42.

Strict full-span coverage - a true positive only when the predicted quotes fully
contain every gold span for that clause. This is the rule ContractEval
([arXiv:2508.03080](https://arxiv.org/abs/2508.03080)) used to benchmark zero-shot LLMs
on the same 102 test contracts:

| model | strict F1 |
|---|---|
| fine-tuned 4B, 3-seed mean | 0.678 |
| GPT-4.1 (ContractEval) | 0.641 |
| Qwen3-8B, best open model they tested (ContractEval) | 0.540 |
| Claude Sonnet 4 (ContractEval) | 0.523 |
| Qwen3-14B zero-shot (my baseline, same rule) | 0.410 |

Read that comparison with two caveats. My strict scorer is a reimplementation of their
stated rule (whitespace-normalized, casefolded substring match), not their published
harness. And ContractEval prompts one clause type per call with its definition, while
this model answers all 41 clause types in a single call, so the setups differ in both
directions.

Data curve, seed 42, detection F1 by number of training contracts:

| 50 | 150 | 300 | 408 |
|---|---|---|---|
| 0.746 | 0.862 | 0.885 | 0.900 |

The 14B zero-shot baseline (0.816) falls somewhere between 50 and 150 annotated
contracts. Training size also affected decoding stability: the 50-contract model hit
repetition loops on 94 of 150 test chunks; the full-data models on 9-12.

Raw evaluation outputs are in `results/`, each with the decode settings and per-chunk
fallback records embedded, so the numbers can be checked without rerunning anything.

## Approach

Contracts are chunked to ~60k characters with 8k overlap; gold spans are assigned to
chunks by character offset. Each chunk is one training example: fixed prompt in, gold
JSON out. About 70% of (contract, clause type) pairs in CUAD are absences, and the
empty lists are part of the training target - that is what teaches the model to answer
"not present" instead of stretching a quote.

Serving is Ollama with format json (grammar-constrained), temperature 0, and num_ctx
computed per request so the prompt never gets silently truncated. At evaluation time,
chunk outputs are merged per contract (union per clause type).

## Local demo

`demo/` has a small Gradio UI that runs against Ollama on your own machine - nothing
leaves it. Setup:

```
hf download Ihteshamstar/qwen3-4b-cuad-extractor gguf/cuad-4b-s42-q4_K_M.gguf --local-dir demo/dl
ollama create cuad-extractor -f demo/Modelfile      # after moving the gguf next to the Modelfile
pip install gradio
python demo/app.py
```

Paste a contract (or load one of the bundled CUAD test excerpts), get the found clause
types with their verbatim quotes. A ~5k-character excerpt takes a few seconds on a
consumer GPU.

## Repo layout

```
pipeline/
  common.py          prompts, clause-name canon, naming; imported by everything else,
                     byte-identical to what the training data was built with
  1_convert.py       CUAD json -> chunked train/test jsonl on the official split
  2_predict.py       runs a model over the test chunks, caches predictions; the cache
                     is keyed to the decode settings and refuses to mix protocols
  2b_fallback.py     re-decodes repetition-looped chunks with repeat_penalty
                     1.05/1.15/1.25, still temperature 0; records which chunks needed it
  3_evaluate.py      detection F1 + bootstrap CI, overlap, verbatim rate
  3b_strict_eval.py  strict full-span-coverage F1
  4_train.py         QLoRA: r=16, alpha=32, lr 2e-4, 2 epochs, 24576-token window
  merge_adapter.py   LoRA adapter -> merged fp16
  run_queue.sh       overnight queue: 3 seeds + data curve
  run_eval_queue.sh  per variant: merge -> GGUF -> ollama create -> predict -> evaluate
data/
  split_manifest.json         pins the official train/test split
  category_descriptions.csv   the 41 clause definitions (CUAD, CC BY 4.0)
results/                      raw eval jsons for every variant
```

To reproduce: download CUAD_v1.json from the Atticus Project into `data/CUAD_v1/` and
run the pipeline in order. The shell launchers have my WSL paths hardcoded; adjust
`BASE`. Training hardware was one RTX 5090, about 45 minutes per full run.

## Notes from the build

Things that went wrong and how the pipeline guards against them now:

- Ollama truncates a prompt that exceeds num_ctx without any error or warning. The
  model just never sees the end of the contract and the output still parses fine.
  num_ctx is now computed per request from chunk size plus output budget.
- An earlier version of the comparison had the two models' prediction caches built
  under different output caps, which is invisible after the fact. Every prediction
  directory now carries a `_run_config.json` fingerprint and `2_predict.py` refuses to
  add to a cache built under different settings.
- At temperature 0 the model occasionally gets stuck repeating the same one or two
  sentences until the token cap. One chunk repeated a 200-character span 201 times.
  A small repeat penalty breaks the loop and keeps decoding deterministic; every chunk
  that needed it is listed in the eval output instead of being silently retried.
- The rendered training text and Ollama's serving template disagreed about the empty
  think-block, which quietly costs accuracy. Training strips it to match how the model
  is actually served with thinking off.
- Unsloth drops training examples whose labels are all -100 after truncation, with one
  log line that's easy to miss. Lengths are now checked by tokenizing every example;
  the single over-length example (28,353 tokens) is excluded and recorded in the
  training config.
- Character-based length estimates are badly off for legal text. That over-length
  example was ~65k characters, which naive chars/4.5 puts at 14k tokens; it tokenizes
  to more than double that (~2.3 chars/token). The length guard tokenizes instead of
  estimating.

## Attribution

CUAD dataset by [The Atticus Project](https://www.atticusprojectai.org/cuad), CC BY
4.0. Base model Qwen3-4B-Instruct-2507, Apache 2.0. Trained with
[Unsloth](https://github.com/unslothai/unsloth). ContractEval numbers from
[arXiv:2508.03080](https://arxiv.org/abs/2508.03080).
