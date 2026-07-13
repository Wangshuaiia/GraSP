"""Light LLM (stage 1, entity selection) for GraSP.

Stage 1 (local, this file), paper Eq. 7:
    Ê = LLM_select(x_LLM),  x_LLM = concat[I_instr, Q, E, H_hat]

TwoStageReasoner here only owns stage 1 (GNN + Graph2Prefix + light LLM -> selected entities).
Stage 2 (LLM_ans, the iterative sufficiency-check / next-entity-or-answer decision, paper Eq. 8-9)
needs to resolve chosen entity names back into Freebase ids to re-query the KG for the next hop --
that orchestration lives in infer.py, which drives both this class and openai_llm.OpenAIPowerfulLLM.

LLM_select is a decoder-only causal LM (e.g. GPT-OSS-20B per the paper's experiment config), fine
-tuned with LoRA on the query/value projections of each self-attention layer (r=8, alpha=16,
dropout=0.1), base weights frozen -- see SmallLLMSelector's use_lora/lora_* args.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import LIGHT_LLM_SYSTEM_PROMPT, LIGHT_LLM_USER_PROMPT_TEMPLATE
from text_encoder import DEFAULT_TEXT_ENCODER, encode as encode_text

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")


def build_light_llm_input(question: str, triples_block: str, entity_list: str) -> str:
    user = LIGHT_LLM_USER_PROMPT_TEMPLATE.format(
        question=question,
        triples_block=triples_block,
        entity_list=entity_list,
    )
    return f"System:\n{LIGHT_LLM_SYSTEM_PROMPT}\n\nUser:\n{user}\n\nAssistant:\n"


class SmallLLMSelector(nn.Module):
    """Causal-LM (decoder-only) LLM_select wrapper: soft-prompt embedding injection + optional LoRA.

    Sequence layout, matching Eq. (6) x_LLM = concat[I_instr, Q, E, H_hat]:
        [ text tokens (instruction+question+candidate entities) ][ per-entity soft prompt H_hat ]
    and, only during training, a target segment appended after that for teacher forcing:
        [ ...x_LLM... ][ target tokens y ]

    The soft-prompt segment from Graph2Prefix is right-padded (real entities first, padding at the
    end, since different subgraphs have different entity counts). For a causal LM, generation
    continues from the sequence's last position, so trailing padding there would anchor the next-
    token prediction on a meaningless zero vector. We flip the soft-prompt segment (and its mask)
    per example before concatenating, so padding sits before the real entities instead -- the same
    left-padding convention used for the text segment -- guaranteeing the last position is always
    real content. Position ids are computed explicitly from the attention mask (skipping padded
    slots) since interior gaps from left-padding aren't handled automatically for inputs_embeds.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        device: Optional[torch.device] = None,
        max_input_length: int = 512,
        max_target_length: int = 32,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_lora = use_lora

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            model = get_peft_model(model, lora_config)  # freezes base weights, only LoRA params trainable
        self.model = model
        self.hidden = self.model.get_input_embeddings().embedding_dim
        self.dtype = self.model.get_input_embeddings().weight.dtype

    def freeze_lm(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def unfreeze_lm(self) -> None:
        """Restore the "correct" trainable set: with LoRA, only the adapter params (base stays
        frozen, matching "only the LoRA parameters are updated"); without LoRA, every parameter."""
        for name, p in self.model.named_parameters():
            p.requires_grad = ("lora_" in name) if self.use_lora else True
        self.model.train()

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def _embed_text(self, input_texts: Sequence[str], add_special_tokens: bool) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(input_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            add_special_tokens=add_special_tokens,
        ).to(self.device)
        embeds = self.model.get_input_embeddings()(encoded.input_ids)
        return embeds, encoded.attention_mask

    def _build_context(
        self,
        prefix: tuple[torch.Tensor, torch.Tensor],
        input_texts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # add_special_tokens=False: this text is a continuing prefix, not a complete sequence -- an
        # auto-added EOS here would land mid-sequence (before the soft prompt) and confuse the model.
        text_embeds, text_mask = self._embed_text(input_texts, add_special_tokens=False)

        prefix_embeds, prefix_mask = prefix
        prefix_embeds = prefix_embeds.to(self.device).to(self.dtype)
        prefix_mask = prefix_mask.to(self.device).to(text_mask.dtype)
        if prefix_embeds.dim() != 3:
            raise ValueError("prefix embeddings must have shape [batch_size, num_entities, hidden_dim]")
        if prefix_embeds.size(0) != text_embeds.size(0):
            raise ValueError("prefix batch size must match input_texts batch size")
        if prefix_embeds.size(-1) != text_embeds.size(-1):
            raise ValueError(
                f"prefix hidden dim {prefix_embeds.size(-1)} != LLM hidden dim {text_embeds.size(-1)}"
            )
        # Flip so padding precedes real entities (see class docstring) -- keeps the sequence's last
        # position real, which causal generation/position-id computation both depend on.
        prefix_embeds = prefix_embeds.flip(dims=[1])
        prefix_mask = prefix_mask.flip(dims=[1])

        # Eq. (6): x_LLM = concat[I_instr, Q, E, H_hat] -- textual context first, per-entity soft
        # prompt last.
        context_embeds = torch.cat([text_embeds, prefix_embeds], dim=1)
        context_mask = torch.cat([text_mask, prefix_mask], dim=1)
        return context_embeds, context_mask

    def forward_with_prefix(
        self,
        prefix: tuple[torch.Tensor, torch.Tensor],
        input_texts: Sequence[str],
        target_texts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 64,
    ):
        context_embeds, context_mask = self._build_context(prefix, input_texts)

        if target_texts is not None:
            target_embeds, target_mask = self._embed_text(target_texts, add_special_tokens=True)
            target_ids = self.tokenizer(
                list(target_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            ).to(self.device).input_ids

            inputs_embeds = torch.cat([context_embeds, target_embeds], dim=1)
            attention_mask = torch.cat([context_mask, target_mask], dim=1)
            labels = torch.full_like(attention_mask, fill_value=-100, dtype=torch.long)
            labels[:, -target_ids.size(1):] = target_ids.masked_fill(target_mask == 0, -100)

            position_ids = self._position_ids_from_mask(attention_mask)
            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )

        generated = self.model.generate(
            inputs_embeds=context_embeds,
            attention_mask=context_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # generate() with inputs_embeds (no input_ids) returns only the newly generated tokens.
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
        generated = self.model.generate(
            **encoded, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.batch_decode(generated[:, encoded.input_ids.size(1):], skip_special_tokens=True)


class TwoStageReasoner(nn.Module):
    """Stage 1 only: GNN + Graph2Prefix + light LLM -> selected entities (Ê)."""

    def __init__(
        self,
        gnn,
        graph2prefix,
        light_llm: SmallLLMSelector,
        text_encoder: str = DEFAULT_TEXT_ENCODER,
    ):
        super().__init__()
        self.gnn = gnn
        self.graph2prefix = graph2prefix
        self.light_llm = light_llm
        self.text_encoder = text_encoder

    def make_prefix(self, graph_batch, questions: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        query = encode_text(list(questions), model_name=self.text_encoder).to(graph_batch.x.device)
        node_emb, _ = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_type=getattr(graph_batch, "edge_type", None),
            batch=graph_batch.batch,
            query=query,
        )
        return self.graph2prefix(node_emb, graph_batch.batch)

    @torch.no_grad()
    def select_entities(
        self,
        questions: Sequence[str],
        triples_texts: Sequence[str],
        entity_lists: Sequence[str],
        graph_batch,
        max_new_tokens: int = 64,
    ) -> List[str]:
        """Ê = LLM_select(x_LLM) (paper Eq. 7): returns the selected-entity text per example."""
        prefix = self.make_prefix(graph_batch, questions)
        light_inputs = [
            build_light_llm_input(q, tr, ents) for q, tr, ents in zip(questions, triples_texts, entity_lists)
        ]
        return self.light_llm.forward_with_prefix(
            prefix=prefix,
            input_texts=light_inputs,
            target_texts=None,
            max_new_tokens=max_new_tokens,
        )
