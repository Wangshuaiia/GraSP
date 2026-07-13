"""Prompt templates for GraSP.

Stage 1 (LLM_select, local light LLM): given the question, retrieved KG triples, and the
candidate entity list E, select the entity/entities most relevant to the question (paper Eq. 7).

Stage 2 (LLM_ans, off-the-shelf API model): given the question and the evidence (entities +
relations) gathered so far, either answer directly (if sufficient, or the iteration limit is
reached) or pick one entity from the evidence as the next reasoning target to explore (paper
Eq. 8-9, the "draft-and-refine" loop).
"""

LIGHT_LLM_SYSTEM_PROMPT = """You are a lightweight knowledge-graph reasoning assistant.
Given a question, a set of retrieved KG triples, and a list of candidate entities, select the
entity or entities from the candidate list most likely to help answer the question.
Only output entity names from the candidate list, nothing else."""

LIGHT_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Retrieved KG triples:
{triples_block}

Candidate entities:
{entity_list}

Instructions:
Select the entity (or entities) from the candidate list above that are most relevant to
answering the question.

Output format:
<entity_1>, <entity_2>, ...
"""

POWERFUL_LLM_SYSTEM_PROMPT = """You are a strong KBQA assistant guiding a multi-hop reasoning process over a
knowledge graph. You will see the question and the entities/relations selected so far as evidence.
Decide whether this evidence is sufficient to answer the question.
- If it is sufficient, or you are told the iteration limit has been reached, answer the question,
  using your own knowledge to fill any gaps the graph evidence leaves, and make clear when the
  graph evidence was incomplete.
- Otherwise, pick exactly one entity from the evidence as the next reasoning target to explore
  further in the knowledge graph."""

POWERFUL_LLM_USER_PROMPT_TEMPLATE = """Question:
{question}

Evidence gathered so far (selected entities and their relations):
{evidence}

{limit_notice}
Respond in exactly one of these two formats -- nothing else:

Decision: ANSWER
Answer: <final answer>
Evidence: <one short supporting sentence>

Decision: ENTITY
Entity: <one entity name from the evidence above>
"""

# Backward-compatible names used by the earlier skeleton in README/train.py.
SMALL_LLM_SYSTEM_PROMPT = LIGHT_LLM_SYSTEM_PROMPT
SMALL_LLM_USER_PROMPT_TEMPLATE = LIGHT_LLM_USER_PROMPT_TEMPLATE
BIG_LLM_SYSTEM_PROMPT = POWERFUL_LLM_SYSTEM_PROMPT
BIG_LLM_USER_PROMPT_TEMPLATE = POWERFUL_LLM_USER_PROMPT_TEMPLATE
