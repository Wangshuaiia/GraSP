"""Data and subgraph utilities for GraSP.

The repository currently contains KBQA JSON files but no full external KG dump. To make the
pipeline runnable, this module builds a local graph from each example's topic entity,
inferential relation chain, constraints, and gold answers. Later, you can replace
`build_triples_from_kbqa_item` with a real KG retriever while keeping the same model code.

This file uses only PyTorch data containers, not PyTorch Geometric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
    answer_text: str
    triples: List[Triple]
    graph: GraphData
    evidence_target: str


def hash_text_embedding(text: str, dim: int = 256) -> torch.Tensor:
    """Deterministic lightweight text feature for graph nodes.

    This avoids requiring BERT embeddings just to run the repo. Replace this with pretrained
    entity embeddings when you connect to a real KG.
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    tokens = text.lower().replace("_", " ").replace(".", " ").split()
    if not tokens:
        vec[0] = 1.0
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return F.normalize(vec, p=2, dim=0)


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


def build_triples_from_kbqa_item(item: dict) -> Tuple[str, str, str, List[Triple]]:
    """Convert a WebQSP/WebQuestions-style item into local evidence triples."""
    parse = _first_parse(item)
    qid = str(item.get("QuestionId") or item.get("id") or "unknown")
    question = str(item.get("ProcessedQuestion") or item.get("RawQuestion") or item.get("question") or "")
    topic_entity = (
        parse.get("TopicEntityName")
        or next(iter((item.get("topic_entity") or {}).values()), None)
        or next(iter((item.get("qid_topic_entity") or {}).values()), None)
        or "topic_entity"
    )
    topic_entity = str(topic_entity).replace("\n", " ").strip()

    rel_chain = [str(r) for r in (parse.get("InferentialChain") or [])]
    answers = _answer_names(parse)
    answer_text = "; ".join(answers) if answers else "unknown"

    triples: List[Triple] = []
    if answers:
        for ans in answers:
            if len(rel_chain) <= 1:
                rel = rel_chain[0] if rel_chain else "related_to"
                triples.append((topic_entity, rel, ans))
            else:
                head = topic_entity
                for hop_id, rel in enumerate(rel_chain[:-1], start=1):
                    mid = f"intermediate_node_{hop_id}_for_{topic_entity}"
                    triples.append((head, rel, mid))
                    head = mid
                triples.append((head, rel_chain[-1], ans))
    else:
        triples.append((topic_entity, "related_to", "unknown_answer"))

    for c in parse.get("Constraints", []) or []:
        pred = str(c.get("NodePredicate") or "constraint")
        ent = str(c.get("EntityName") or c.get("Argument") or c.get("ValueType") or "constraint_value")
        triples.append((topic_entity, pred, ent))

    return qid, question, topic_entity, answer_text, triples


def format_triples(triples: Sequence[Triple], max_triples: Optional[int] = None) -> str:
    rows = []
    for h, r, t in list(triples)[:max_triples]:
        rows.append(f"({h}, {r}, {t})")
    return "\n".join(rows) if rows else "(no triples retrieved)"


def build_evidence_target(question: str, triples: Sequence[Triple], answer_text: str) -> str:
    return (
        "Relevant graph information:\n"
        + "\n".join(f"- ({h}, {r}, {t})" for h, r, t in triples)
        + f"\nLikely answer from graph: {answer_text}"
    )


def build_graph_from_triples(
    triples: Sequence[Triple],
    relation2id: Dict[str, int],
    feature_dim: int = 256,
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

    x = torch.stack([hash_text_embedding(name, feature_dim) for name in names], dim=0)
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
        feature_dim: int = 256,
        max_examples: Optional[int] = None,
        relation2id: Optional[Dict[str, int]] = None,
        update_relation_vocab: bool = True,
    ) -> None:
        self.data_path = data_path
        self.feature_dim = feature_dim
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
            qid, question, topic, answer_text, triples = build_triples_from_kbqa_item(item)
            graph = build_graph_from_triples(
                triples,
                relation2id=self.relation2id,
                feature_dim=feature_dim,
                update_relation_vocab=update_relation_vocab,
            )
            evidence_target = build_evidence_target(question, triples, answer_text)
            self.examples.append(
                KBQAExample(
                    qid=qid,
                    question=question,
                    topic_entity=topic,
                    answer_text=answer_text,
                    triples=triples,
                    graph=graph,
                    evidence_target=evidence_target,
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
        "answer_texts": [ex.answer_text for ex in examples],
        "triples": [ex.triples for ex in examples],
        "triples_texts": [format_triples(ex.triples) for ex in examples],
        "evidence_targets": [ex.evidence_target for ex in examples],
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
