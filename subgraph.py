"""Data and subgraph utilities for GraSP.

Each example's local graph is built by walking a live Freebase KG outward from the question's
topic entity via SPARQL (see `freebase_func.fetch_khop_triples`). The question and gold answer
still come from the KBQA parse -- they are the supervision label, not part of the retrieved
graph -- but the triples themselves are real KG edges/entities, not synthesized from the parse.

This file uses only PyTorch data containers, not PyTorch Geometric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from freebase_func import fetch_khop_triples
from text_encoder import DEFAULT_TEXT_ENCODER, embedding_dim, encode

Triple = Tuple[str, str, str]


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_names: List[str]
    triples: List[Triple]

    def to(self, device: torch.device | str):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        return self


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    batch: torch.Tensor
    node_names: List[List[str]]
    triples: List[List[Triple]]

    def to(self, device: torch.device | str):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        self.batch = self.batch.to(device)
        return self

    @staticmethod
    def from_data_list(graphs: Sequence[GraphData]) -> "GraphBatch":
        xs = []
        edge_indices = []
        edge_types = []
        batches = []
        node_offset = 0
        node_names = []
        triples = []
        for gid, g in enumerate(graphs):
            xs.append(g.x)
            num_nodes = g.x.size(0)
            if g.edge_index.numel() > 0:
                edge_indices.append(g.edge_index + node_offset)
                edge_types.append(g.edge_type)
            batches.append(torch.full((num_nodes,), gid, dtype=torch.long))
            node_offset += num_nodes
            node_names.append(g.node_names)
            triples.append(g.triples)

        x = torch.cat(xs, dim=0) if xs else torch.empty((0, 0))
        edge_index = (
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.empty((0,), dtype=torch.long)
        batch = torch.cat(batches, dim=0) if batches else torch.empty((0,), dtype=torch.long)
        return GraphBatch(x=x, edge_index=edge_index, edge_type=edge_type, batch=batch, node_names=node_names, triples=triples)


@dataclass
class KBQAExample:
    qid: str
    question: str
    topic_entity: str
    topic_entity_mid: str
    answer_text: str
    triples: List[Triple]
    graph: GraphData


def _first_parse(item: dict) -> dict:
    parses = item.get("Parses") or []
    if not parses:
        return {}
    for p in parses:
        if p.get("AnnotatorComment", {}).get("ParseQuality") == "Complete":
            return p
    return parses[0]


def _answer_names(parse: dict) -> List[str]:
    answers = []
    for ans in parse.get("Answers", []) or []:
        name = ans.get("EntityName") or ans.get("AnswerArgument")
        if name is not None:
            answers.append(str(name).replace("\n", " ").strip())
    return answers


def build_triples_from_kbqa_item(
    item: dict,
    max_hops: int = 2,
    max_relations_per_hop: int = 8,
    max_entities_per_relation: int = 10,
    max_triples: int = 60,
    text_encoder: str = DEFAULT_TEXT_ENCODER,
) -> Tuple[str, str, str, str, str, List[Triple]]:
    """Convert a WebQSP/WebQuestions-style item into local evidence triples.

    The topic entity's local neighborhood is retrieved live from Freebase via SPARQL
    (`freebase_func.fetch_khop_triples`), keyed off the parse's `TopicEntityMid`, and pruned at
    each hop by cosine similarity between the question and candidate relations/entities. `question`
    and `answer_text` still come from the parse -- they are the supervision label, not graph
    input -- so the retrieved subgraph is independent of the gold answer/relation chain.

    Returns (qid, question, topic_entity_mid, topic_entity_name, answer_text, triples).
    `topic_entity_mid` is returned so callers can re-run `fetch_khop_triples` from a different
    entity later (the iterative multi-hop reasoning loop in infer.py).
    """
    parse = _first_parse(item)
    qid = str(item.get("QuestionId") or item.get("id") or "unknown")
    question = str(item.get("ProcessedQuestion") or item.get("RawQuestion") or item.get("question") or "")

    topic_entity_mid = parse.get("TopicEntityMid") or next(iter((item.get("topic_entity") or {}).keys()), None)
    topic_entity_name = (
        parse.get("TopicEntityName")
        or next(iter((item.get("topic_entity") or {}).values()), None)
        or next(iter((item.get("qid_topic_entity") or {}).values()), None)
        or topic_entity_mid
        or "topic_entity"
    )
    topic_entity_name = str(topic_entity_name).replace("\n", " ").strip()

    answers = _answer_names(parse)
    answer_text = "; ".join(answers) if answers else "unknown"

    if not topic_entity_mid:
        raise ValueError(f"No TopicEntityMid found for question {qid!r}; cannot query Freebase via SPARQL.")
    topic_entity_mid = str(topic_entity_mid)

    triples, _name_to_mid = fetch_khop_triples(
        topic_entity_id=topic_entity_mid,
        topic_entity_name=topic_entity_name,
        question=question,
        max_hops=max_hops,
        max_relations_per_hop=max_relations_per_hop,
        max_entities_per_relation=max_entities_per_relation,
        max_triples=max_triples,
        text_encoder=text_encoder,
    )
    if not triples:
        triples = [(topic_entity_name, "related_to", "no_triples_retrieved")]

    return qid, question, topic_entity_mid, topic_entity_name, answer_text, triples


def format_triples(triples: Sequence[Triple], max_triples: Optional[int] = None) -> str:
    rows = []
    for h, r, t in list(triples)[:max_triples]:
        rows.append(f"({h}, {r}, {t})")
    return "\n".join(rows) if rows else "(no triples retrieved)"


def format_entity_list(names: Sequence[str], max_entities: Optional[int] = None) -> str:
    """Render the candidate entity list `E` (paper Eq. 6) for the light LLM's prompt."""
    deduped = list(dict.fromkeys(names))[:max_entities]
    return ", ".join(deduped) if deduped else "(no candidate entities)"


def build_graph_from_triples(
    triples: Sequence[Triple],
    relation2id: Dict[str, int],
    text_encoder: str = DEFAULT_TEXT_ENCODER,
    add_reverse_edges: bool = True,
    update_relation_vocab: bool = True,
) -> GraphData:
    node2id: Dict[str, int] = {}

    def get_node_id(name: str) -> int:
        if name not in node2id:
            node2id[name] = len(node2id)
        return node2id[name]

    edge_pairs = []
    edge_types = []
    for h, r, t in triples:
        u, v = get_node_id(h), get_node_id(t)
        if r not in relation2id:
            if update_relation_vocab:
                relation2id[r] = len(relation2id)
            else:
                relation2id.setdefault("<unk>", 0)
        rid = relation2id.get(r, relation2id.get("<unk>", 0))
        edge_pairs.append((u, v))
        edge_types.append(rid)
        if add_reverse_edges:
            rr = f"reverse::{r}"
            if rr not in relation2id:
                if update_relation_vocab:
                    relation2id[rr] = len(relation2id)
                else:
                    relation2id.setdefault("<unk>", 0)
            edge_pairs.append((v, u))
            edge_types.append(relation2id.get(rr, relation2id.get("<unk>", 0)))

    if not node2id:
        node2id["empty_graph"] = 0

    names = [None] * len(node2id)
    for name, idx in node2id.items():
        names[idx] = name

    x = encode(names, model_name=text_encoder)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return GraphData(x=x, edge_index=edge_index, edge_type=edge_type, node_names=names, triples=list(triples))


class KBQAGraphDataset(Dataset):
    """Dataset that turns the repo JSON KBQA data into local graph examples."""

    def __init__(
        self,
        data_path: str,
        text_encoder: str = DEFAULT_TEXT_ENCODER,
        max_examples: Optional[int] = None,
        relation2id: Optional[Dict[str, int]] = None,
        update_relation_vocab: bool = True,
        max_hops: int = 2,
        max_relations_per_hop: int = 8,
        max_entities_per_relation: int = 10,
        max_triples_per_example: int = 60,
    ) -> None:
        self.data_path = data_path
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim(text_encoder)
        self.relation2id = relation2id if relation2id is not None else {"<unk>": 0}
        self.update_relation_vocab = update_relation_vocab

        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw_items = raw.get("data") or raw.get("examples") or raw.get("questions") or []
        else:
            raw_items = raw
        if max_examples is not None:
            raw_items = raw_items[:max_examples]

        self.examples: List[KBQAExample] = []
        for item in raw_items:
            qid, question, topic_mid, topic, answer_text, triples = build_triples_from_kbqa_item(
                item,
                max_hops=max_hops,
                max_relations_per_hop=max_relations_per_hop,
                max_entities_per_relation=max_entities_per_relation,
                max_triples=max_triples_per_example,
                text_encoder=text_encoder,
            )
            graph = build_graph_from_triples(
                triples,
                relation2id=self.relation2id,
                text_encoder=text_encoder,
                update_relation_vocab=update_relation_vocab,
            )
            self.examples.append(
                KBQAExample(
                    qid=qid,
                    question=question,
                    topic_entity=topic,
                    topic_entity_mid=topic_mid,
                    answer_text=answer_text,
                    triples=triples,
                    graph=graph,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> KBQAExample:
        return self.examples[idx]


def collate_kbqa_examples(examples: Sequence[KBQAExample]) -> dict:
    graph_batch = GraphBatch.from_data_list([ex.graph for ex in examples])
    return {
        "qids": [ex.qid for ex in examples],
        "questions": [ex.question for ex in examples],
        "topic_entities": [ex.topic_entity for ex in examples],
        "topic_entity_mids": [ex.topic_entity_mid for ex in examples],
        "answer_texts": [ex.answer_text for ex in examples],
        "triples": [ex.triples for ex in examples],
        "triples_texts": [format_triples(ex.triples) for ex in examples],
        "entity_lists": [format_entity_list(ex.graph.node_names) for ex in examples],
        "graph": graph_batch,
    }


def extract_n_hop_subgraph(
    center_entity_id: int,
    edge_index: torch.Tensor,
    num_hops: int,
    rel_ids: Optional[torch.Tensor] = None,
):
    """Pure-PyTorch compatibility helper for older `infer.py` skeletons.

    Returns: subset, sub_edge_index, sub_rel_ids, center_mapping, edge_mask.
    """
    if edge_index.numel() == 0:
        subset = torch.tensor([center_entity_id], dtype=torch.long, device=edge_index.device)
        return subset, torch.empty((2, 0), dtype=torch.long, device=edge_index.device), None, 0, torch.empty((0,), dtype=torch.bool, device=edge_index.device)

    frontier = {int(center_entity_id)}
    visited = {int(center_entity_id)}
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj = {}
    for u, v in zip(src, dst):
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))
    for _ in range(num_hops):
        new_frontier = set()
        for u in frontier:
            for v in adj.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    new_frontier.add(v)
        frontier = new_frontier

    subset_list = sorted(visited)
    subset = torch.tensor(subset_list, dtype=torch.long, device=edge_index.device)
    old2new = {old: i for i, old in enumerate(subset_list)}
    keep_mask_list = []
    new_edges = []
    for i, (u, v) in enumerate(zip(src, dst)):
        keep = int(u) in old2new and int(v) in old2new
        keep_mask_list.append(keep)
        if keep:
            new_edges.append((old2new[int(u)], old2new[int(v)]))
    edge_mask = torch.tensor(keep_mask_list, dtype=torch.bool, device=edge_index.device)
    sub_edge_index = (
        torch.tensor(new_edges, dtype=torch.long, device=edge_index.device).t().contiguous()
        if new_edges
        else torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    )
    sub_rel_ids = rel_ids[edge_mask] if rel_ids is not None else None
    center_mapping = old2new[int(center_entity_id)]
    return subset, sub_edge_index, sub_rel_ids, center_mapping, edge_mask
