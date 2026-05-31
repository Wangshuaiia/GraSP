"""Run two-stage GraSP inference.

Flow:
  1. Build local graph from a KBQA example.
  2. Second-order GNN encodes the graph and produces soft prompts.
  3. Light LLM receives soft prompt + triples + question and organizes graph evidence.
  4. Powerful LLM receives organized evidence + original question and produces final answer.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from gnn_soft_prompt import Graph2Prefix, SubgraphGNN
from light_reasoning_llm import SmallLLMSelector, TwoStageReasoner
from subgraph import KBQAGraphDataset, collate_kbqa_examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/WebQSP.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--example_id", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--light_model_name", type=str, default=None)
    parser.add_argument("--powerful_model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens_light", type=int, default=128)
    parser.add_argument("--max_new_tokens_powerful", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    relation2id = ckpt.get("relation2id", {"<unk>": 0})

    light_model_name = args.light_model_name or ckpt.get("light_model_name") or cfg.get(
        "light_model_name", "google/flan-t5-small"
    )
    powerful_model_name = args.powerful_model_name or light_model_name

    dataset = KBQAGraphDataset(
        data_path=args.data_path,
        feature_dim=int(cfg.get("feature_dim", 256)),
        relation2id=relation2id,
        update_relation_vocab=False,
    )
    start = args.example_id
    end = min(len(dataset), start + args.num_examples)
    examples = [dataset[i] for i in range(start, end)]
    batch = collate_kbqa_examples(examples)
    graph = batch["graph"].to(device)

    light_llm = SmallLLMSelector(light_model_name, device=device)
    light_llm.freeze_lm()
    powerful_llm = SmallLLMSelector(powerful_model_name, device=device)
    powerful_llm.freeze_lm()

    gnn = SubgraphGNN(
        in_dim=int(cfg.get("feature_dim", 256)),
        hid_dim=int(cfg.get("graph_dim", 256)),
        out_dim=int(cfg.get("graph_dim", 256)),
        num_layers=int(cfg.get("gnn_layers", 2)),
        heads=int(cfg.get("gnn_heads", 4)),
        num_relations=len(relation2id) + 1,
        use_second_order=True,
    ).to(device)
    graph2prefix = Graph2Prefix(
        graph_dim=int(cfg.get("graph_dim", 256)),
        llm_hidden=light_llm.hidden,
        prefix_len=int(cfg.get("prefix_len", 8)),
    ).to(device)
    gnn.load_state_dict(ckpt["gnn"])
    graph2prefix.load_state_dict(ckpt["graph2prefix"])
    gnn.eval()
    graph2prefix.eval()

    reasoner = TwoStageReasoner(gnn, graph2prefix, light_llm, powerful_llm)
    organized_infos = reasoner.organize_graph_info(
        batch["questions"],
        batch["triples_texts"],
        graph,
        max_new_tokens=args.max_new_tokens_light,
    )
    answers = reasoner.answer(
        batch["questions"],
        organized_infos,
        max_new_tokens=args.max_new_tokens_powerful,
    )

    results = []
    for i, qid in enumerate(batch["qids"]):
        results.append(
            {
                "qid": qid,
                "question": batch["questions"][i],
                "retrieved_triples": batch["triples_texts"][i],
                "organized_graph_info": organized_infos[i],
                "prediction": answers[i],
                "gold_answer": batch["answer_texts"][i],
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
