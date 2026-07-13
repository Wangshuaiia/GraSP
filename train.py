"""Train GraSP's second-order GNN soft prompt + light LLM entity selector (LLM_select).

Defaults match the paper's main experiment config: GPT-OSS-20B as the selection LLM, LoRA on the
query/value projections of each self-attention layer (r=8, alpha=16, dropout=0.1) with the LoRA
adapter as the only trainable part of the light LLM (base weights frozen), bert-base-uncased as
the text encoder (hidden size 768), 2-hop subgraphs, a two-layer GNN with hidden dim 128. The
Graph2Prefix FFN's output dim is derived from the light LLM's actual hidden size (2880 for
GPT-OSS-20B), not hardcoded. GPT-OSS-20B is a 20B-parameter model -- expect to need real GPU
memory; override --light_model_name with something small (e.g. a SmolLM2 checkpoint) for local
smoke-testing with --mock_kg.

The trainable modules are:
  1) SubgraphGNN: encodes first-order + second-order KG structure
  2) Graph2Prefix: per-entity FFN mapping node embeddings to soft-prompt tokens (paper Eq. 5)
  3) (--train_light_lm, on by default) the light LLM's LoRA adapter (or, with --no-use_lora, the
     full light LLM)

Loss (paper Eq. "Optimization", loss func): L = -log P(y | I_instr, Q, E, H_hat), where y is the
ground-truth target entity. We don't have per-hop gold intermediate entities (the KG is retrieved
live via SPARQL, not from a gold reasoning chain), so y is the dataset's answer entity name --
this is a single-step supervision matching the paper's Optimization section, which describes no
iteration; the iterative draft-and-refine loop (Eq. 8-9) is an inference-time behavior (see
infer.py), not part of training.
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
from text_encoder import DEFAULT_TEXT_ENCODER, encode as encode_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/WebQSP.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--light_model_name", type=str, default="openai/gpt-oss-20b", help="LLM_select backbone; paper default is GPT-OSS-20B (needs a real GPU -- override for local smoke-testing)")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True, help="Fine-tune the light LLM with LoRA instead of full fine-tuning; --no-use_lora to disable")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj", help="Comma-separated module name suffixes LoRA is applied to")
    parser.add_argument("--text_encoder", type=str, default=DEFAULT_TEXT_ENCODER, help="Pretrained sentence encoder for graph node features and SPARQL relation/entity ranking")
    parser.add_argument("--graph_dim", type=int, default=128, help="GNN hidden dim (paper default: 128)")
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_light_lm", action=argparse.BooleanOptionalAction, default=True, help="Fine-tune the light LLM (its LoRA adapter, if --use_lora); --no-train_light_lm to keep it fully frozen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_hops", type=int, default=2, help="SPARQL BFS depth from the topic entity")
    parser.add_argument("--max_relations_per_hop", type=int, default=8)
    parser.add_argument("--max_entities_per_relation", type=int, default=8)
    parser.add_argument("--max_triples_per_example", type=int, default=60)
    parser.add_argument(
        "--mock_kg", action="store_true",
        help="Use synthetic KG data instead of live SPARQL (pipeline smoke-testing only, not for real training)",
    )
    return parser.parse_args()


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def main():
    args = parse_args()
    if args.mock_kg:
        os.environ["FREEBASE_MOCK"] = "1"
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = KBQAGraphDataset(
        data_path=args.data_path,
        text_encoder=args.text_encoder,
        max_examples=args.max_examples,
        update_relation_vocab=True,
        max_hops=args.max_hops,
        max_relations_per_hop=args.max_relations_per_hop,
        max_entities_per_relation=args.max_entities_per_relation,
        max_triples_per_example=args.max_triples_per_example,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_kbqa_examples,
    )

    light_llm = SmallLLMSelector(
        args.light_model_name,
        device=device,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
    )
    if not args.train_light_lm:
        light_llm.freeze_lm()

    gnn = SubgraphGNN(
        in_dim=dataset.embedding_dim,
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
                build_light_llm_input(q, triples, ents)
                for q, triples, ents in zip(batch["questions"], batch["triples_texts"], batch["entity_lists"])
            ]
            targets = batch["answer_texts"]  # y: ground-truth target entity (paper Eq. loss func)
            query = encode_text(batch["questions"], model_name=args.text_encoder).to(device)

            node_emb, _ = gnn(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_type=getattr(graph, "edge_type", None),
                batch=graph.batch,
                query=query,
            )
            prefix = graph2prefix(node_emb, graph.batch)
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
            # Only the trainable (with --use_lora: just the LoRA adapter) params -- the rest of the
            # light LLM's weights come fresh from the pretrained checkpoint at load time, so without
            # this, any light-LLM fine-tuning would be silently discarded at inference.
            "light_llm_trainable_state": (
                {n: p.detach().cpu() for n, p in light_llm.model.named_parameters() if p.requires_grad}
                if args.train_light_lm
                else None
            ),
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    with open(os.path.join(args.output_dir, "relation2id.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.relation2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
