"""Evaluation helpers for GraSP predictions (infer.py's JSON output).

infer.py already resolves each example down to a single `prediction` string and carries its own
`gold_answer` alongside it (semicolon-joined if there are multiple acceptable answers), so scoring
is a direct prediction-vs-gold comparison -- no separate dataset re-alignment step is needed (the
{...}-bracket extraction used by ToG-style eval scripts doesn't apply here: our answer LLM already
returns a clean "Answer: ..." field via openai_llm.parse_decision, so infer.py's `prediction` is
already the extracted answer text).
"""

from __future__ import annotations

from typing import Sequence

REFUSAL_PHRASES = ("i don't know", "i do not know", "i'm sorry", "i am sorry", "cannot determine", "not enough information")


def split_gold_answers(gold_answer: str) -> list[str]:
    """gold_answer is "; "-joined (subgraph.py's answer_text) when a question has multiple accepted answers."""
    return [a.strip() for a in (gold_answer or "").split(";") if a.strip()]


def is_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def exact_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    """Loose exact match: case/whitespace-insensitive, and accepts either string containing the
    other (LLM answers are often more verbose than the gold entity name, e.g. "Kansas City,
    Missouri" vs "Kansas City")."""
    clean_pred = (prediction or "").strip().replace(" ", "").lower()
    if not clean_pred:
        return False
    for answer in gold_answers:
        clean_answer = answer.strip().replace(" ", "").lower()
        if not clean_answer:
            continue
        if clean_pred == clean_answer or clean_pred in clean_answer or clean_answer in clean_pred:
            return True
    return False
