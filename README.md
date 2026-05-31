# KBQA with Subgraph-GNN Soft Prompting and Two-Stage LLM Reasoning

This repository implements a **two-stage Knowledge Base Question Answering (KBQA)** framework that combines:

- **n-hop subgraph extraction**
- **Graph Neural Networks (GNNs)** for subgraph encoding
- **Soft prompts (prefix tuning)** injected into a lightweight LLM
- **Node selection by a small LLM**
- **Final answer generation by a large LLM via API (e.g., GPT-5)**

The framework is designed to be **modular**, **efficient**, and **LLM-cost-aware**, suitable for research and industrial KBQA scenarios.

---

## 📐 Overall Architecture

```

Question + Topic Entity
│
▼
n-hop Subgraph Extraction
│
▼
GNN Encoding (Subgraph → Graph Embedding)
│
▼
Graph Embedding → Soft Prompt (Prefix Tokens)
│
▼
Lightweight LLM (Node Selection)
│
▼
Selected Nodes + Evidence Edges
│
▼
Large LLM API (Answer Generation)
│
▼
Final Answer

```

---

## 📁 Repository Structure

```

.
├── prompts.py                  # All prompt templates (small LLM & big LLM)
├── subgraph.py                 # n-hop subgraph extraction
├── gnn_soft_prompt.py          # GNN + Graph-to-Prefix modules
├── light_reasoning_llm.py      # Lightweight LLM with soft prompt injection
├── train.py                    # Training loop (GNN + optional small LLM)
├── infer.py                    # Inference pipeline (two-stage reasoning)
├── dataset
├── README.md

````

---

## ⚙️ Environment Setup

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric
- HuggingFace Transformers

### Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers
```

---

## 🧠 Model Components

### 1. GNN Encoder

* Encodes an n-hop subgraph into a fixed-size graph embedding
* Implemented using `GATv2Conv`
* Trained to support downstream node selection

### 2. Soft Prompt Generator

* Maps graph embedding → `prefix_len` virtual tokens
* Injected into the lightweight LLM as **prefix embeddings**

### 3. Lightweight LLM (Node Selector)

* Input:
  **Soft Prompt + Question + Candidate Node List**
* Output (JSON only):

```json
{
  "selected_node_ids": [...],
  "rationale": "..."
}
```

### 4. Large LLM (Answer Generator)

* Input:

  * Question
  * Selected nodes
  * Evidence edges (triples)
* Output:

  * Final answer
  * Supporting evidence

---

## 🚀 Training

### Training Objective

The goal of training is to learn a **graph-aware soft prompt** that guides a lightweight LLM
to select the correct reasoning nodes from a subgraph given a question.

- Supervision: `gold_selected_node_ids`
- Loss: causal language modeling loss on structured JSON output
- The GNN is **always trainable**
- The lightweight LLM is **optionally trainable**

---

### 🔧 Training Configuration

A key parameter controls whether the lightweight LLM participates in training:

```python
train_small_llm = False  # or True
````

* `False` (recommended):

  * Train **GNN + soft prompt generator only**
  * Lightweight LLM is frozen
  * More stable and efficient

* `True`:

  * Train **GNN + soft prompt generator + lightweight LLM**
  * Higher capacity and cost
  * Can be replaced with LoRA for efficiency

---

### ▶️ Start Training (Command Line)

Train on the small JSON data already in the repo:

```bash
python train.py \
  --data_path data/WebQSP.json \
  --light_model_name google/flan-t5-small \
  --epochs 3 \
  --batch_size 2
```

By default, `train.py`:

* Initializes the GNN, soft prompt generator, and lightweight LLM
* Loads the training dataset
* Runs the training loop for multiple epochs
* Saves model checkpoints

You can configure training parameters (e.g., learning rate, number of epochs, whether to train
the lightweight LLM) directly inside `train.py` or via command-line arguments (optional).

---

## 🧪 Evaluation / Inference

Evaluation follows a **two-stage inference pipeline**:

1. The lightweight LLM selects relevant nodes from the subgraph
2. A large LLM (via API, e.g., GPT-5) generates the final answer using the selected evidence

---

### ▶️ Run Evaluation (Command Line)

Evaluation is launched using:

Run two-stage inference:

```bash
python infer.py \
  --data_path data/WebQSP.json \
  --checkpoint checkpoints/best.pt \
  --example_id 0 \
  --num_examples 1 \
  --light_model_name google/flan-t5-small \
  --powerful_model_name google/flan-t5-small
```

By default, `infer.py`:

* Loads trained GNN and soft prompt checkpoints
* Extracts n-hop subgraphs for each test question
* Uses the lightweight LLM to select relevant nodes
* Calls a large LLM API to generate final answers
* Outputs predicted answers and intermediate reasoning results

Example output:

```json
{
  "selected_node_ids": [12, 45],
  "final_answer": "Christopher Nolan"
}
```

---

### 🔌 Large LLM API Configuration

The large LLM is accessed via an API wrapper defined in `infer.py`:

```python
class BigLLMClient:
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        ...
```

You can connect this interface to:

* GPT-5
* Azure OpenAI
* Claude
* Any enterprise or internal LLM service

---

## 📝 Notes

* The training and evaluation scripts are intentionally kept simple for clarity
* The framework can be easily extended with:

  * Argument parsing (`argparse`)
  * LoRA-based fine-tuning
  * Distributed training



---

## 📌 Key Advantages

* Decouples reasoning structure from final answering
* Reduces large LLM context length and cost
* Enables graph-aware soft prompting
* Modular and extensible design


