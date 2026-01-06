import json
import torch

from prompts import BIG_LLM_SYSTEM_PROMPT, BIG_LLM_USER_PROMPT_TEMPLATE
from train import build_small_llm_input
from subgraph import extract_n_hop_subgraph

def parse_selector_json(text: str):
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {"selected_node_ids": [], "rationale": "parse_failed"}
    try:
        return json.loads(text[s:e+1])
    except Exception:
        return {"selected_node_ids": [], "rationale": "json_load_failed"}

def format_edges_as_triples(subset_nodes, sub_edge_index, sub_rel_ids, id2name, relid2name=None):
    lines = []
    src = sub_edge_index[0].tolist()
    dst = sub_edge_index[1].tolist()
    for i, (u, v) in enumerate(zip(src, dst)):
        hu = int(subset_nodes[u])
        hv = int(subset_nodes[v])
        rname = "related_to"
        if sub_rel_ids is not None and relid2name is not None:
            rname = relid2name[int(sub_rel_ids[i])]
        lines.append(f"({id2name.get(hu, str(hu))}, {rname}, {id2name.get(hv, str(hv))})")
    return "\n".join(lines)

def format_selected_nodes(selected_ids, id2name):
    lines = []
    for nid in selected_ids:
        try:
            nid_int = int(nid)
        except Exception:
            nid_int = nid
        lines.append(f"- id={nid} | name={id2name.get(nid_int, str(nid))}")
    return "\n".join(lines)

class BigLLMClient:
    def __init__(self):
        pass

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        # TODO: replace with real API call
        raise NotImplementedError("Connect to your API.")

@torch.no_grad()
def kbqa_infer_one(question: str,
                   center_entity_id: int,
                   candidates: list,
                   edge_index,
                   rel_ids,
                   node_features,
                   id2name: dict,
                   relid2name: dict | None,
                   num_hops: int,
                   gnn,
                   g2p,
                   small_llm,
                   big_llm: BigLLMClient,
                   device):
    subset, sub_edge_index, sub_rel_ids, center_mapping, edge_mask = extract_n_hop_subgraph(
        center_entity_id, edge_index, num_hops, rel_ids
    )
    sub_x = node_features[subset].to(device)  # [N_sub, in_dim]
    batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=device)

    _, g = gnn(sub_x, sub_edge_index.to(device), batch)
    prefix = g2p(g)  # [1, prefix_len, H]

    small_inp = build_small_llm_input(question, candidates)
    out_text = small_llm.forward_with_prefix(prefix, [small_inp], target_texts=None)[0]
    sel = parse_selector_json(out_text)
    selected_ids = sel.get("selected_node_ids", [])

    edges_block = format_edges_as_triples(subset, sub_edge_index, sub_rel_ids, id2name, relid2name)
    selected_nodes_block = format_selected_nodes(selected_ids, id2name)

    user_prompt = BIG_LLM_USER_PROMPT_TEMPLATE.format(
        question=question,
        selected_nodes_block=selected_nodes_block,
        edges_block=edges_block
    )

    answer = big_llm.chat(BIG_LLM_SYSTEM_PROMPT, user_prompt)
    return {
        "small_llm_raw": out_text,
        "selected_node_ids": selected_ids,
        "final_answer": answer
    }
