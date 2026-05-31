"""Light LLM and two-stage reasoning utilities for GraSP.

Stage 1:
    [GNN soft prompt] + [question + retrieved triples] -> organized graph information

Stage 2:
    [organized graph information] + [original question] -> final answer
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from prompts import (
    LIGHT_LLM_SYSTEM_PROMPT,
    LIGHT_LLM_USER_PROMPT_TEMPLATE,
    POWERFUL_LLM_SYSTEM_PROMPT,
    POWERFUL_LLM_USER_PROMPT_TEMPLATE,
)


def build_light_llm_input(question: str, triples_block: str) -> str:
    user = LIGHT_LLM_USER_PROMPT_TEMPLATE.format(
        question=question,
        triples_block=triples_block,
    )
    return f"System:\n{LIGHT_LLM_SYSTEM_PROMPT}\n\nUser:\n{user}\n\nAssistant:\n"


def build_powerful_llm_input(question: str, organized_graph_info: str) -> str:
    user = POWERFUL_LLM_USER_PROMPT_TEMPLATE.format(
        question=question,
        organized_graph_info=organized_graph_info,
    )
    return f"System:\n{POWERFUL_LLM_SYSTEM_PROMPT}\n\nUser:\n{user}\n\nAssistant:\n"


class SmallLLMSelector(nn.Module):
    """Seq2Seq LLM wrapper that supports soft-prompt embedding prepending."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: Optional[torch.device] = None,
        max_input_length: int = 512,
        max_target_length: int = 256,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.hidden = self.model.get_input_embeddings().embedding_dim

    def freeze_lm(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def unfreeze_lm(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

    def _embed_with_prefix(
        self,
        prefix: torch.Tensor,
        input_texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(input_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)
        text_embeds = self.model.get_input_embeddings()(encoded.input_ids)
        prefix = prefix.to(self.device)
        if prefix.dim() != 3:
            raise ValueError("prefix must have shape [batch_size, prefix_len, hidden_dim]")
        if prefix.size(0) != text_embeds.size(0):
            raise ValueError("prefix batch size must match input_texts batch size")
        if prefix.size(-1) != text_embeds.size(-1):
            raise ValueError(
                f"prefix hidden dim {prefix.size(-1)} != LLM hidden dim {text_embeds.size(-1)}"
            )
        prefix_mask = torch.ones(
            prefix.size()[:2],
            dtype=encoded.attention_mask.dtype,
            device=self.device,
        )
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, encoded.attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def forward_with_prefix(
        self,
        prefix: torch.Tensor,
        input_texts: Sequence[str],
        target_texts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 128,
    ):
        inputs_embeds, attention_mask = self._embed_with_prefix(prefix, input_texts)
        if target_texts is not None:
            labels = self.tokenizer(
                list(target_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
            ).to(self.device).input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_plain(self, input_texts: Sequence[str], max_new_tokens: int = 128) -> List[str]:
        encoded = self.tokenizer(
            list(input_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)
        generated = self.model.generate(**encoded, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class TwoStageReasoner(nn.Module):
    """End-to-end helper used by `infer.py`."""

    def __init__(self, gnn, graph2prefix, light_llm: SmallLLMSelector, powerful_llm: SmallLLMSelector):
        super().__init__()
        self.gnn = gnn
        self.graph2prefix = graph2prefix
        self.light_llm = light_llm
        self.powerful_llm = powerful_llm

    def make_prefix(self, graph_batch) -> torch.Tensor:
        _, graph_emb = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_type=getattr(graph_batch, "edge_type", None),
            batch=graph_batch.batch,
        )
        return self.graph2prefix(graph_emb)

    @torch.no_grad()
    def organize_graph_info(
        self,
        questions: Sequence[str],
        triples_texts: Sequence[str],
        graph_batch,
        max_new_tokens: int = 128,
    ) -> List[str]:
        prefix = self.make_prefix(graph_batch)
        light_inputs = [build_light_llm_input(q, tr) for q, tr in zip(questions, triples_texts)]
        return self.light_llm.forward_with_prefix(
            prefix=prefix,
            input_texts=light_inputs,
            target_texts=None,
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def answer(
        self,
        questions: Sequence[str],
        organized_graph_infos: Sequence[str],
        max_new_tokens: int = 128,
    ) -> List[str]:
        powerful_inputs = [
            build_powerful_llm_input(q, info) for q, info in zip(questions, organized_graph_infos)
        ]
        return self.powerful_llm.generate_plain(powerful_inputs, max_new_tokens=max_new_tokens)
