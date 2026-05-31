"""Train GraSP's second-order GNN soft prompt.

The light LLM is frozen by default. The trainable modules are:
  1) SubgraphGNN: encodes first-order + second-order KG structure
  2) Graph2Prefix: maps graph embeddings to soft-prompt tokens

The supervision target is an automatically built "organized graph information" text from the
example's triples and answers. This matches the intended role of the light LLM.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_soft_prompt import Graph2Prefix, SubgraphGNN
from light_reasoning_llm import SmallLLMSelector, build_light_llm_input
from subgraph import KBQAGraphDataset, collate_kbqa_examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/WebQSP.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--light_model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--graph_dim", type=int, default=256)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_light_lm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = KBQAGraphDataset(
        data_path=args.data_path,
        feature_dim=args.feature_dim,
        max_examples=args.max_examples,
        update_relation_vocab=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_kbqa_examples,
    )

    light_llm = SmallLLMSelector(args.light_model_name, device=device)
    if not args.train_light_lm:
        light_llm.freeze_lm()

    gnn = SubgraphGNN(
        in_dim=args.feature_dim,
        hid_dim=args.graph_dim,
        out_dim=args.graph_dim,
        num_layers=args.gnn_layers,
        heads=args.gnn_heads,
        num_relations=len(dataset.relation2id) + 1,
        use_second_order=True,
    ).to(device)
    graph2prefix = Graph2Prefix(
        graph_dim=args.graph_dim,
        llm_hidden=light_llm.hidden,
        prefix_len=args.prefix_len,
    ).to(device)

    trainable = list(gnn.parameters()) + list(graph2prefix.parameters())
    if args.train_light_lm:
        trainable += [p for p in light_llm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        gnn.train()
        graph2prefix.train()
        if args.train_light_lm:
            light_llm.train()
        else:
            light_llm.eval()

        total_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            graph = batch["graph"].to(device)
            inputs = [
                build_light_llm_input(q, triples)
                for q, triples in zip(batch["questions"], batch["triples_texts"])
            ]
            targets = batch["evidence_targets"]

            _, graph_emb = gnn(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_type=getattr(graph, "edge_type", None),
                batch=graph.batch,
            )
            prefix = graph2prefix(graph_emb)
            outputs = light_llm.forward_with_prefix(prefix, inputs, target_texts=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch} avg_loss={avg_loss:.4f}")

        ckpt = {
            "gnn": gnn.state_dict(),
            "graph2prefix": graph2prefix.state_dict(),
            "relation2id": dataset.relation2id,
            "config": vars(args),
            "light_model_name": args.light_model_name,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    with open(os.path.join(args.output_dir, "relation2id.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.relation2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
