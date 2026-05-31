"""Prompt templates for GraSP.

The light LLM does evidence organization. The powerful LLM does final QA.
"""

LIGHT_LLM_SYSTEM_PROMPT = """You are a lightweight knowledge-graph reasoning assistant.
Your task is to read a question and a set of retrieved KG triples, remove noisy information,
and organize the graph evidence that may help answer the question.
Do not answer using unsupported facts. Keep the output concise."""

LIGHT_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Retrieved KG triples:
{triples_block}

Instructions:
1. Select the triples that are relevant to the question.
2. Group them into a short organized evidence summary.
3. Mention missing or insufficient evidence when necessary.

Output format:
Relevant graph information:
- ...
"""

POWERFUL_LLM_SYSTEM_PROMPT = """You are a strong KBQA assistant.
Use the organized graph information as grounded evidence. You may use your own knowledge only
when the graph evidence is incomplete, but make clear when the graph is insufficient."""

POWERFUL_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Organized graph information from the lightweight LLM:
{organized_graph_info}

Task:
Answer the question concisely. Then provide one short evidence sentence.

Output format:
Answer: ...
Evidence: ...
"""

# Backward-compatible names used by the earlier skeleton in README/train.py.
SMALL_LLM_SYSTEM_PROMPT = LIGHT_LLM_SYSTEM_PROMPT
SMALL_LLM_USER_PROMPT_TEMPLATE = LIGHT_LLM_USER_PROMPT_TEMPLATE
BIG_LLM_SYSTEM_PROMPT = POWERFUL_LLM_SYSTEM_PROMPT
BIG_LLM_USER_PROMPT_TEMPLATE = POWERFUL_LLM_USER_PROMPT_TEMPLATE
