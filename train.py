import json
import torch
import torch.nn as nn
from torch.optim import AdamW

from prompts import SMALL_LLM_SYSTEM_PROMPT, SMALL_LLM_USER_PROMPT_TEMPLATE
from gnn_soft_prompt import SubgraphGNN, Graph2Prefix
from light_reasoning_llm import SmallLLMSelector

def format_candidate_block(candidates):
    """
    candidates: list[dict] each: {"id": int/str, "name": str}
    """
    lines = []
    for c in candidates:
        lines.append(f'- id={c["id"]} | name="{c["name"]}"')
    return "\n".join(lines)

def build_small_llm_input(question, candidates):
    candidate_block = format_candidate_block(candidates)
    user = SMALL_LLM_USER_PROMPT_TEMPLATE.format(question=question, candidate_block=candidate_block)
    #  chat style 
    
    full = f"System:\n{SMALL_LLM_SYSTEM_PROMPT}\n\nUser:\n{user}\n\nAssistant:\n"
    return full

def build_small_llm_target(selected_node_ids, rationale=""):
    #  JSON
    obj = {"selected_node_ids": selected_node_ids, "rationale": rationale or "Most relevant nodes for answering."}
    return json.dumps(obj, ensure_ascii=False)

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def train_one_epoch(dataloader,
                    gnn,
                    g2p,
                    small_llm,
                    train_small_llm: bool,
                    optimizer,
                    device):
    gnn.train()
    g2p.train()
    if train_small_llm:
        small_llm.train()
    else:
        small_llm.eval()

    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        bvec = batch.batch.to(device)

        _, g = gnn(x, edge_index, bvec)          # [B, graph_dim]
        prefix = g2p(g)                          # [B, prefix_len, hidden]

        input_texts = []
        target_texts = []
        for q, cands, gold in zip(batch.question, batch.candidates, batch.gold_selected_node_ids):
            inp = build_small_llm_input(q, cands)
            tgt = build_small_llm_target(gold)
            input_texts.append(inp)
            target_texts.append(tgt)

        out = small_llm.forward_with_prefix(prefix, input_texts, target_texts=target_texts)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(g2p.parameters()) +
                                       (list(small_llm.parameters()) if train_small_llm else []),
                                       max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

def build_optimizer(gnn, g2p, small_llm, train_small_llm: bool, lr=2e-4, wd=0.01):
    params = []
    params += list(gnn.parameters())
    params += list(g2p.parameters())
    if train_small_llm:
        params += [p for p in small_llm.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=wd)

def setup_models(in_dim, graph_dim, small_model_name, prefix_len, device, train_small_llm: bool):
    gnn = SubgraphGNN(in_dim=in_dim, hid_dim=graph_dim, out_dim=graph_dim, num_layers=2).to(device)
    small_llm = SmallLLMSelector(small_model_name).to(device)
    g2p = Graph2Prefix(graph_dim=graph_dim, llm_hidden=small_llm.hidden, prefix_len=prefix_len).to(device)

    if not train_small_llm:
        freeze_module(small_llm)

    return gnn, g2p, small_llm
