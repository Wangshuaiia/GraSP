"""OpenAI-backed "powerful LLM" (LLM_ans) for GraSP's stage-2 draft-and-refine answering.

Paper Eq. 8-9: given the question q, the selected entities Ê and their relations R̂, LLM_ans either
answers directly (if the evidence gathered so far is sufficient, or the iteration limit has been
reached) or picks one entity from the evidence as the next reasoning target. It never sees the
graph soft prompt -- only natural-language evidence -- which is what lets it be any off-the-shelf
API model instead of a locally hosted one.

The iteration loop itself (re-querying the KG from the newly chosen entity, tracking evidence
across hops) lives in infer.py, since it also needs freebase_func's retrieval layer to resolve a
chosen entity name back into a Freebase id.
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional

from openai import OpenAI

from prompts import POWERFUL_LLM_SYSTEM_PROMPT, POWERFUL_LLM_USER_PROMPT_TEMPLATE

Message = dict

_DECISION_RE = re.compile(r"Decision:\s*(ANSWER|ENTITY)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n\s*Evidence:|\Z)", re.IGNORECASE | re.DOTALL)
_EVIDENCE_RE = re.compile(r"Evidence:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ENTITY_RE = re.compile(r"Entity:\s*(.+)", re.IGNORECASE)

LIMIT_REACHED_NOTICE = (
    "The iteration limit has been reached: you must answer now using the available evidence and "
    "your own knowledge."
)


def build_decision_messages(question: str, evidence: str, limit_reached: bool) -> List[Message]:
    user = POWERFUL_LLM_USER_PROMPT_TEMPLATE.format(
        question=question,
        evidence=evidence,
        limit_notice=LIMIT_REACHED_NOTICE if limit_reached else "",
    )
    return [
        {"role": "system", "content": POWERFUL_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_decision(text: str) -> dict:
    """Parse LLM_ans's structured output (Eq. 8) into a decision dict.

    Falls back to treating the whole response as an answer if the model didn't follow the
    requested format -- a free-running agent loop must not crash on a malformed LLM response.
    """
    decision_match = _DECISION_RE.search(text)
    decision = decision_match.group(1).upper() if decision_match else "ANSWER"
    if decision == "ENTITY":
        entity_match = _ENTITY_RE.search(text)
        entity = entity_match.group(1).strip() if entity_match else None
        if not entity:
            return {"decision": "ANSWER", "answer": text.strip(), "evidence": "", "raw": text}
        return {"decision": "ENTITY", "entity": entity, "raw": text}

    answer_match = _ANSWER_RE.search(text)
    evidence_match = _EVIDENCE_RE.search(text)
    return {
        "decision": "ANSWER",
        "answer": answer_match.group(1).strip() if answer_match else text.strip(),
        "evidence": evidence_match.group(1).strip() if evidence_match else "",
        "raw": text,
    }


class OpenAIPowerfulLLM:
    """Calls the OpenAI chat completions API. Frozen/off-the-shelf -- never trained locally."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Pass api_key=... or set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=key, base_url=base_url)

    def _complete_one(self, messages: List[Message], max_new_tokens: int) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_new_tokens,
                    temperature=self.temperature,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # openai raises a mix of APIError/RateLimitError/etc.
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"OpenAI API call failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def decide(self, question: str, evidence: str, limit_reached: bool, max_new_tokens: int = 128) -> dict:
        """LLM_ans(q, Ê, R̂) (paper Eq. 8): returns {"decision": "ANSWER", "answer":..., "evidence":...}
        or {"decision": "ENTITY", "entity": ...}."""
        messages = build_decision_messages(question, evidence, limit_reached)
        raw = self._complete_one(messages, max_new_tokens)
        result = parse_decision(raw)
        if limit_reached and result["decision"] != "ANSWER":
            # Told to answer now but didn't follow the format -- treat the raw output as the
            # answer rather than surfacing an ENTITY-shaped non-answer or looping forever.
            result = {"decision": "ANSWER", "answer": raw.strip(), "evidence": "", "raw": raw}
        return result
