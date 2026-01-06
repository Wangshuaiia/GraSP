# prompts
SMALL_LLM_SYSTEM_PROMPT = """You are a knowledge-graph reasoning assistant.
Given a question and a list of candidate nodes from a subgraph, select the nodes that are most relevant for answering the question.
Return ONLY valid JSON."""

SMALL_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Candidate nodes (choose from this list ONLY):
{candidate_block}

Instructions:
- Select the minimal set of nodes necessary.
- Output JSON with keys:
  - "selected_node_ids": a list of node ids from the candidate list
  - "rationale": one short sentence

Output format (JSON only):
{{"selected_node_ids":[...], "rationale":"..."}}
"""

BIG_LLM_SYSTEM_PROMPT = """You are an expert KBQA assistant.
You must answer the question using ONLY the provided subgraph evidence (nodes and edges).
If the evidence is insufficient, say you cannot determine the answer from the given evidence."""

BIG_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Selected nodes:
{selected_nodes_block}

Evidence edges (triples):
{edges_block}

Task:
- Reason over the evidence.
- Provide the final answer concisely.
- Also output a short "evidence" section listing the key triples you used.

Output format:
Answer: <final answer>
Evidence:
- <triple 1>
- <triple 2>
"""
