"""Run GraSP's iterative draft-and-refine inference (paper Section "Two-Stage Reasoning and
Answering", Eq. 7-9).

Per example, per hop:
  1. Retrieve a local subgraph from the current topic entity via SPARQL.
  2. Second-order GNN + Graph2Prefix encode it into a soft prompt (conditioned on the question).
  3. Light LLM (LLM_select, local) selects the entity/entities most relevant to the question (Ê).
  4. Powerful LLM (LLM_ans, OpenAI API, frozen off-the-shelf) sees the question and the evidence
     (Ê + their relations) accumulated so far, and either:
       - answers directly, if the evidence is sufficient or the iteration limit is reached, or
       - picks one entity from the evidence as the next hop's topic entity, and the loop repeats.

This loop is inherently per-example (different questions need different numbers of hops), so
unlike training this script processes one example at a time rather than batching them.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch

from freebase_func import fetch_khop_triples
from gnn_soft_prompt import Graph2Prefix, SubgraphGNN
from light_reasoning_llm import SmallLLMSelector, TwoStageReasoner
from openai_llm import OpenAIPowerfulLLM
from subgraph import (
    GraphBatch,
    KBQAGraphDataset,
    build_graph_from_triples,
    format_entity_list,
    format_triples,
)
from text_encoder import DEFAULT_TEXT_ENCODER, embedding_dim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/WebQSP.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--example_id", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--light_model_name", type=str, default=None, help="Overrides the checkpoint's light_model_name if set (paper default: GPT-OSS-20B)")
    parser.add_argument("--powerful_model", type=str, default="gpt-5.2", help="OpenAI model id for stage-2 answering (paper default: GPT-5.2)")
    parser.add_argument("--openai_api_key", type=str, default=None, help="Defaults to the OPENAI_API_KEY env var")
    parser.add_argument("--openai_base_url", type=str, default=None, help="Override for Azure/proxy/self-hosted OpenAI-compatible endpoints")
    parser.add_argument("--powerful_temperature", type=float, default=0.0)
    parser.add_argument("--text_encoder", type=str, default=None, help="Overrides the checkpoint's text_encoder if set")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens_light", type=int, default=64)
    parser.add_argument("--max_new_tokens_powerful", type=int, default=128)
    parser.add_argument("--max_iterations", type=int, default=3, help="Draft-and-refine iteration limit (paper Eq. 8-9)")
    parser.add_argument("--max_hops", type=int, default=2, help="SPARQL BFS depth from the topic entity, per iteration")
    parser.add_argument("--max_relations_per_hop", type=int, default=8)
    parser.add_argument("--max_entities_per_relation", type=int, default=8)
    parser.add_argument("--max_triples_per_example", type=int, default=60)
    parser.add_argument(
        "--mock_kg", action="store_true",
        help="Use synthetic KG data instead of live SPARQL (pipeline smoke-testing only, not for real inference)",
    )
    return parser.parse_args()


def run_iterative_reasoning(
    qid: str,
    question: str,
    topic_entity_mid: str,
    topic_entity_name: str,
    answer_text: str,
    reasoner: TwoStageReasoner,
    powerful_llm: OpenAIPowerfulLLM,
    relation2id: dict,
    text_encoder: str,
    device: torch.device,
    max_iterations: int,
    max_hops: int,
    max_relations_per_hop: int,
    max_entities_per_relation: int,
    max_triples_per_example: int,
    max_new_tokens_light: int,
    max_new_tokens_powerful: int,
) -> dict:
    evidence_log = []
    current_mid, current_name = topic_entity_mid, topic_entity_name
    final_answer, final_evidence = None, ""
    trace = []

    for iteration in range(max_iterations):
        triples, name_to_mid = fetch_khop_triples(
            topic_entity_id=current_mid,
            topic_entity_name=current_name,
            question=question,
            max_hops=max_hops,
            max_relations_per_hop=max_relations_per_hop,
            max_entities_per_relation=max_entities_per_relation,
            max_triples=max_triples_per_example,
            text_encoder=text_encoder,
        )
        if not triples:
            triples = [(current_name, "related_to", "no_triples_retrieved")]

        graph = build_graph_from_triples(
            triples, relation2id=relation2id, text_encoder=text_encoder, update_relation_vocab=False
        )
        graph_batch = GraphBatch.from_data_list([graph]).to(device)
        entity_list = format_entity_list(graph.node_names)
        triples_text = format_triples(triples)

        selected = reasoner.select_entities(
            [question], [triples_text], [entity_list], graph_batch, max_new_tokens=max_new_tokens_light
        )[0]

        # R\hat: triples touching any selected-entity mention (simple substring match -- the light
        # LLM's raw output is free text, not guaranteed to exactly match a candidate name).
        selected_names = [s.strip() for s in re.split(r"[;,\n]", selected) if s.strip()]
        relevant_triples = [
            t for t in triples
            if any(name.lower() in t[0].lower() or name.lower() in t[2].lower() for name in selected_names)
        ] or triples[:5]

        evidence_log.append(f"[hop {iteration}] Selected entities: {selected}\n{format_triples(relevant_triples)}")
        evidence_text = "\n\n".join(evidence_log)

        limit_reached = iteration == max_iterations - 1
        decision = powerful_llm.decide(question, evidence_text, limit_reached, max_new_tokens=max_new_tokens_powerful)
        trace.append(
            {"iteration": iteration, "topic_entity": current_name, "selected_entities": selected, "decision": decision}
        )

        if decision["decision"] == "ANSWER" or limit_reached:
            final_answer = decision.get("answer") or decision.get("raw", "")
            final_evidence = decision.get("evidence", "")
            break

        next_name = decision.get("entity")
        next_mid = name_to_mid.get(next_name) if next_name else None
        if not next_mid:
            # Couldn't resolve the chosen entity to a KG id -- force an answer instead of looping blindly.
            forced = powerful_llm.decide(question, evidence_text, limit_reached=True, max_new_tokens=max_new_tokens_powerful)
            final_answer = forced.get("answer") or forced.get("raw", "")
            final_evidence = forced.get("evidence", "")
            trace.append({"iteration": iteration, "note": f"could not resolve entity {next_name!r} to a KG id; forced answer"})
            break
        current_mid, current_name = next_mid, next_name

    return {
        "qid": qid,
        "question": question,
        "prediction": final_answer,
        "evidence": final_evidence,
        "gold_answer": answer_text,
        "num_iterations": len(trace),
        "trace": trace,
    }


def main():
    args = parse_args()
    if args.mock_kg:
        os.environ["FREEBASE_MOCK"] = "1"
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    relation2id = ckpt.get("relation2id", {"<unk>": 0})

    light_model_name = args.light_model_name or ckpt.get("light_model_name") or cfg.get(
        "light_model_name", "openai/gpt-oss-20b"
    )
    text_encoder = args.text_encoder or cfg.get("text_encoder", DEFAULT_TEXT_ENCODER)

    dataset = KBQAGraphDataset(
        data_path=args.data_path,
        text_encoder=text_encoder,
        relation2id=relation2id,
        update_relation_vocab=False,
        max_hops=args.max_hops,
        max_relations_per_hop=args.max_relations_per_hop,
        max_entities_per_relation=args.max_entities_per_relation,
        max_triples_per_example=args.max_triples_per_example,
    )
    start = args.example_id
    end = min(len(dataset), start + args.num_examples)

    light_llm = SmallLLMSelector(
        light_model_name,
        device=device,
        use_lora=bool(cfg.get("use_lora", True)),
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
        lora_target_modules=str(cfg.get("lora_target_modules", "q_proj,v_proj")).split(","),
    )
    light_llm_trainable_state = ckpt.get("light_llm_trainable_state")
    if light_llm_trainable_state:
        missing, unexpected = light_llm.model.load_state_dict(light_llm_trainable_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading light_llm_trainable_state: {unexpected}")
    light_llm.freeze_lm()
    powerful_llm = OpenAIPowerfulLLM(
        model=args.powerful_model,
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        temperature=args.powerful_temperature,
    )

    gnn = SubgraphGNN(
        in_dim=embedding_dim(text_encoder),
        hid_dim=int(cfg.get("graph_dim", 128)),
        out_dim=int(cfg.get("graph_dim", 128)),
        num_layers=int(cfg.get("gnn_layers", 2)),
        heads=int(cfg.get("gnn_heads", 4)),
        num_relations=len(relation2id) + 1,
        use_second_order=True,
    ).to(device)
    graph2prefix = Graph2Prefix(
        graph_dim=int(cfg.get("graph_dim", 128)),
        llm_hidden=light_llm.hidden,
    ).to(device)
    gnn.load_state_dict(ckpt["gnn"])
    graph2prefix.load_state_dict(ckpt["graph2prefix"])
    gnn.eval()
    graph2prefix.eval()

    reasoner = TwoStageReasoner(gnn, graph2prefix, light_llm, text_encoder=text_encoder)

    results = []
    for i in range(start, end):
        ex = dataset[i]
        results.append(
            run_iterative_reasoning(
                qid=ex.qid,
                question=ex.question,
                topic_entity_mid=ex.topic_entity_mid,
                topic_entity_name=ex.topic_entity,
                answer_text=ex.answer_text,
                reasoner=reasoner,
                powerful_llm=powerful_llm,
                relation2id=relation2id,
                text_encoder=text_encoder,
                device=device,
                max_iterations=args.max_iterations,
                max_hops=args.max_hops,
                max_relations_per_hop=args.max_relations_per_hop,
                max_entities_per_relation=args.max_entities_per_relation,
                max_triples_per_example=args.max_triples_per_example,
                max_new_tokens_light=args.max_new_tokens_light,
                max_new_tokens_powerful=args.max_new_tokens_powerful,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
