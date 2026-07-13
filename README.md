# GraSP: Graph Soft-Prompted KBQA with Two-Stage LLM Reasoning

GraSP is a Knowledge Base Question Answering (KBQA) framework that combines:

- **Live SPARQL retrieval** of a local subgraph around a question's topic entity from a Freebase-compatible endpoint (no gold reasoning chain — the graph is retrieved, not synthesized from the answer)
- A **second-order, query-conditioned relation-attention GNN** that encodes that subgraph
- A **per-entity soft prompt** (FFN projection of GNN node embeddings) injected into a local, LoRA-fine-tuned **light LLM** that selects the entities most relevant to the question
- An **iterative draft-and-refine loop**: a frozen, off-the-shelf **powerful LLM** (via the OpenAI API) decides whether the gathered evidence is enough to answer, or picks the next entity to explore, repeating until it can answer or an iteration limit is hit

This implementation is pure PyTorch and tracks a specific paper's equations closely — see the per-module docstrings for the exact equation each piece implements.

---

## Architecture

```
Question + Topic Entity (from a KBQA dataset item)
        │
        ▼
Live SPARQL retrieval (freebase_func.fetch_khop_triples)
  -- k-hop BFS from the topic entity; candidate relations/entities at each hop
     ranked by cosine similarity to the question and capped
        │
        ▼
SubgraphGNN (gnn_soft_prompt.py)
  -- second-order (two-hop edges added explicitly) + self-loops
  -- query-conditioned attention: a_ij = softmax(f(W(q||h_i), W(h_j||r_ij)))
        │
        ▼
Graph2Prefix: per-entity FFN -> soft-prompt tokens H_hat (one token per entity, padded per batch)
        │
        ▼
Light LLM / LLM_select (light_reasoning_llm.py)
  -- causal LM + LoRA, input = concat[instruction, question, candidate entities, H_hat]
  -- selects the entity/entities (Ê) most relevant to the question
        │
        ▼
Powerful LLM / LLM_ans (openai_llm.py, OpenAI API, frozen)
  -- sees the question + evidence (Ê and its relations) gathered so far
  -- either answers now (evidence sufficient / iteration limit reached), or
  -- picks one entity from the evidence as the next hop's topic entity
        │
        └── if "next entity": resolve name -> Freebase id, go back to SPARQL retrieval
        │
        ▼
Final answer
```

---

## Repository structure

```
.
├── subgraph.py             # KBQA item parsing, GraphData/GraphBatch, KBQAGraphDataset
├── freebase_func.py        # Live SPARQL retrieval (+ deterministic mock mode)
├── text_encoder.py         # Shared BERT-family sentence encoder (node features + retrieval ranking)
├── gnn_soft_prompt.py      # SubgraphGNN (query-conditioned attention) + Graph2Prefix
├── light_reasoning_llm.py  # LLM_select: causal LM + LoRA, soft-prompt injection, TwoStageReasoner
├── openai_llm.py           # LLM_ans: OpenAI API client, draft-and-refine decision parsing
├── prompts.py              # All prompt templates
├── train.py                # Trains SubgraphGNN + Graph2Prefix (+ optional light LLM LoRA adapter)
├── infer.py                # Runs the iterative draft-and-refine inference loop
├── eval.py                 # Scores infer.py's JSON predictions (Hits@1)
├── utils.py                # Evaluation helpers (matching, refusal detection)
├── requirements.txt
└── data/                   # WebQSP.json, WebQuestions.json, cwq.json, grailqa.json, ...
```

---

## Setup

### Requirements

- Python ≥ 3.9
- A Freebase-compatible SPARQL endpoint (e.g. a local Virtuoso instance loaded with a Freebase dump — see the ToG/RoG project READMEs for how to stand one up), **or** use `--mock_kg` for local smoke-testing without one
- An OpenAI API key for the answer-generation stage

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `torch`, `transformers`, `tqdm`, `SPARQLWrapper`, `openai`, `peft`. No PyTorch Geometric, no `sentence-transformers` (the text encoder does its own mean pooling over a plain `AutoModel`).

### Environment variables

- `FREEBASE_SPARQL_ENDPOINT` — your SPARQL endpoint URL (defaults to a placeholder address; you must point this at your own instance for real retrieval)
- `OPENAI_API_KEY` — required by `infer.py`'s powerful LLM (or pass `--openai_api_key`)
- `FREEBASE_MOCK=1` — bypass SPARQL and use deterministic synthetic KG data instead (same effect as `--mock_kg`); **smoke-testing only, never for real training/evaluation** (the synthetic "knowledge" is meaningless, though it's derived independently of the gold answer so it doesn't introduce label leakage)

---

## Paper-aligned default configuration

Running `train.py`/`infer.py` with no extra flags reproduces the paper's main experiment config:

| Setting | Default |
|---|---|
| Selection LLM (`LLM_select`) | `openai/gpt-oss-20b` |
| Answer LLM (`LLM_ans`) | `gpt-5.2` (OpenAI API) |
| LoRA target modules | `q_proj,v_proj` (query/value projections) |
| LoRA rank / alpha / dropout | `r=8`, `alpha=16`, `dropout=0.1` |
| LoRA training | on by default — only the LoRA adapter is trained, base weights frozen |
| Text encoder | `bert-base-uncased` (hidden size 768) |
| GNN hidden dim | `128` |
| GNN layers | `2` |
| Subgraph depth | `2` hops |

`GPT-OSS-20B` is a 20B-parameter model and needs real GPU memory to load — override `--light_model_name` with something small (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) for local development with `--mock_kg`.

---

## Training

```bash
python3 train.py \
  --data_path data/WebQSP.json \
  --output_dir checkpoints \
  --epochs 3 \
  --batch_size 2
```

Trainable modules:

1. **SubgraphGNN** — always trained
2. **Graph2Prefix** — always trained (per-entity FFN, paper Eq. 5)
3. **Light LLM's LoRA adapter** — trained by default (`--train_light_lm`, on unless you pass `--no-train_light_lm`); with `--no-use_lora` this becomes a full fine-tune instead of LoRA

**Loss**: `L = -log P(y | I_instr, Q, E, H_hat)`, where `y` is the ground-truth target entity. Since the subgraph is retrieved live via SPARQL (no gold reasoning chain), there's no per-hop gold intermediate entity to supervise against — `y` is the dataset's answer entity name, and this is single-step supervision (the iterative draft-and-refine loop is inference-only, see below).

Checkpoints (`last.pt`, `best.pt` in `--output_dir`) contain the GNN/Graph2Prefix state dicts, the relation vocabulary, the full config, and — if the light LLM was trained — its trainable (LoRA or full) parameter state, so `infer.py` can restore it exactly rather than starting from a fresh untrained adapter.

For local smoke-testing without a live SPARQL endpoint or GPU:

```bash
python3 train.py --mock_kg --max_examples 5 --epochs 1 \
  --light_model_name HuggingFaceTB/SmolLM2-135M-Instruct --device cpu
```

### Key CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--data_path` | `data/WebQSP.json` | KBQA training data |
| `--output_dir` | `checkpoints` | Where checkpoints + `relation2id.json` are saved |
| `--light_model_name` | `openai/gpt-oss-20b` | `LLM_select` backbone |
| `--use_lora` / `--no-use_lora` | `True` | LoRA fine-tuning vs. full fine-tuning |
| `--lora_r`, `--lora_alpha`, `--lora_dropout` | `8`, `16`, `0.1` | LoRA hyperparameters |
| `--lora_target_modules` | `q_proj,v_proj` | Comma-separated module suffixes LoRA targets |
| `--train_light_lm` / `--no-train_light_lm` | `True` | Fine-tune the light LLM at all, vs. keep it fully frozen |
| `--text_encoder` | `bert-base-uncased` | Sentence encoder for node features + SPARQL relation/entity ranking |
| `--graph_dim` | `128` | GNN hidden dim |
| `--gnn_layers` | `2` | Number of GNN layers |
| `--batch_size` | `2` | Training batch size |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_examples` | none | Cap on training examples (debugging) |
| `--max_hops` | `2` | SPARQL BFS depth from the topic entity |
| `--max_relations_per_hop` | `8` | Relation candidates kept per hop (after similarity ranking) |
| `--max_entities_per_relation` | `8` | Entity candidates kept per relation (after similarity ranking) |
| `--max_triples_per_example` | `60` | Overall retrieved-triple cap per example |
| `--mock_kg` | off | Use synthetic KG data instead of live SPARQL |

`--lr`/`--epochs`/`--batch_size`/`--weight_decay` aren't specified in the paper excerpt this repo was aligned to — these are this repo's own defaults, not verified against the paper's actual training run.

---

## Inference

```bash
export OPENAI_API_KEY=sk-...
python3 infer.py \
  --checkpoint checkpoints/best.pt \
  --example_id 0 \
  --num_examples 10 \
  > predictions.json
```

Per example, `infer.py` runs the **iterative draft-and-refine loop** (paper Eq. 7-9), one example at a time (hop count varies per question, so this isn't batched the way training is):

1. Retrieve a local subgraph from the current topic entity via SPARQL.
2. `SubgraphGNN` + `Graph2Prefix` encode it into a soft prompt, conditioned on the question.
3. The light LLM (`LLM_select`) selects the entity/entities most relevant to the question.
4. The powerful LLM (`LLM_ans`) sees the question and the evidence (selected entities + their relations) accumulated across all hops so far, and either:
   - answers directly, if the evidence is sufficient or `--max_iterations` has been reached, or
   - picks one entity from the evidence as the next hop's topic entity — its name is resolved back to a Freebase id and the loop repeats.

If the powerful LLM picks an entity that can't be resolved back to a Freebase id (or still won't commit to an answer once the iteration limit is reached), the loop forces an answer using whatever evidence was gathered rather than looping indefinitely.

Output is a JSON array printed to stdout, one record per example:

```json
{
  "qid": "...",
  "question": "...",
  "prediction": "...",
  "evidence": "...",
  "gold_answer": "...",
  "num_iterations": 2,
  "trace": [ { "iteration": 0, "topic_entity": "...", "selected_entities": "...", "decision": {...} }, ... ]
}
```

`gold_answer` and `prediction` are both included per record, so `eval.py` doesn't need to re-load the source dataset to score anything.

### Key CLI arguments

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Trained checkpoint |
| `--example_id` / `--num_examples` | `0` / `1` | Which examples to run, from the dataset |
| `--light_model_name` | checkpoint's value | Override `LLM_select` backbone |
| `--powerful_model` | `gpt-5.2` | OpenAI model id for `LLM_ans` |
| `--openai_api_key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--openai_base_url` | none | Override for Azure/proxy/self-hosted OpenAI-compatible endpoints |
| `--powerful_temperature` | `0.0` | Sampling temperature for `LLM_ans` |
| `--max_new_tokens_light` / `--max_new_tokens_powerful` | `64` / `128` | Generation length caps |
| `--max_iterations` | `3` | Draft-and-refine iteration limit |
| `--max_hops`, `--max_relations_per_hop`, `--max_entities_per_relation`, `--max_triples_per_example` | `2`, `8`, `8`, `60` | Same retrieval knobs as training, applied per hop |
| `--mock_kg` | off | Use synthetic KG data instead of live SPARQL |

---

## Evaluation

```bash
python3 eval.py --predictions predictions.json --show_errors 10
```

Scores `infer.py`'s output with **Hits@1** (the single predicted answer per question is either right or wrong — loose match: case/whitespace-insensitive, either string containing the other, since LLM answers are often more verbose than the gold entity name). Examples with an unresolved gold answer (`"unknown"`) or a detected refusal (`"I don't know"`-style responses, unless `--no_skip_refusals`) are excluded from the denominator and reported separately. Prints a summary, optionally the first N wrong examples, and writes a metrics JSON file (default `<predictions>_eval.json`).

---

## Design notes

- **No gold-answer leakage in retrieval**: the subgraph is retrieved live via SPARQL from the topic entity, independently of the gold answer — the retrieved neighborhood can (and often does) fail to contain the answer, same as a real deployment. The light LLM's training target is the answer entity name (there's no separate "organize the evidence" text target that could let it shortcut by copying a label).
- **Query-conditioned GNN attention**: relation/entity candidates during SPARQL retrieval, and node attention weights inside `SubgraphGNN`, are both conditioned on the question via the shared BERT text encoder — not arbitrary/alphabetical.
- **Variable-length soft prompt**: `Graph2Prefix` emits one soft-prompt token per entity in the retrieved subgraph (not a fixed-size pooled vector), padded per batch with a validity mask, matching the paper's `H_hat ∈ R^{n×d_LLM}`.
- **Mock mode** (`--mock_kg` / `FREEBASE_MOCK=1`) exists purely so the rest of the pipeline (graph construction, GNN, LoRA training loop, the draft-and-refine loop) can be exercised end-to-end without a live SPARQL endpoint. The synthetic edges are deterministic hashes of entity/relation strings, independent of the gold answer — but the "knowledge" itself is meaningless. Never use it for real training or evaluation numbers.
