"""Second-order GNN soft-prompt modules for GraSP.

This file only handles graph encoding:
    KG graph -> second-order relation-aware GNN -> graph embedding -> soft prompt tokens

The second-order part explicitly adds two-hop edges u->w whenever u->v and v->w exist.
Those edges allow the GNN to encode neighbor-of-neighbor information as soft prompts.

This implementation is pure PyTorch, so the repository can run without PyTorch Geometric.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_two_hop_edges(
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    two_hop_rel_id: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Add explicit two-hop edges.

    For every path u -> v -> w, this function adds u -> w. Original edges are kept.
    Duplicate edges are removed. Two-hop edges receive relation id `two_hop_rel_id`.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_type

    device = edge_index.device
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()

    out_neighbors = [[] for _ in range(num_nodes)]
    original_pairs = set()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            out_neighbors[u].append(v)
            original_pairs.add((u, v))

    two_hop_pairs = []
    seen_two_hop = set()
    for u in range(num_nodes):
        for v in out_neighbors[u]:
            for w in out_neighbors[v]:
                if u == w:
                    continue
                pair = (u, w)
                if pair not in original_pairs and pair not in seen_two_hop:
                    seen_two_hop.add(pair)
                    two_hop_pairs.append(pair)

    if not two_hop_pairs:
        return edge_index, edge_type

    two_hop_edge_index = torch.tensor(two_hop_pairs, dtype=torch.long, device=device).t().contiguous()
    new_edge_index = torch.cat([edge_index, two_hop_edge_index], dim=1)

    if edge_type is None:
        return new_edge_index, None

    two_hop_type = torch.full(
        (two_hop_edge_index.size(1),),
        fill_value=int(two_hop_rel_id),
        dtype=edge_type.dtype,
        device=device,
    )
    new_edge_type = torch.cat([edge_type, two_hop_type], dim=0)
    return new_edge_index, new_edge_type


def add_self_loop_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    self_loop_rel_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    loops = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    loop_types = torch.full((num_nodes,), int(self_loop_rel_id), dtype=torch.long, device=device)
    return torch.cat([edge_index, loops], dim=1), torch.cat([edge_type, loop_types], dim=0)


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    out = x.new_zeros((num_graphs, x.size(-1)))
    count = x.new_zeros((num_graphs, 1))
    out.index_add_(0, batch, x)
    count.index_add_(0, batch, torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device))
    return out / count.clamp(min=1.0)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    rows = []
    for gid in range(num_graphs):
        mask = batch == gid
        if mask.any():
            rows.append(x[mask].max(dim=0).values)
        else:
            rows.append(x.new_zeros(x.size(-1)))
    return torch.stack(rows, dim=0)


class RelGraphLayer(nn.Module):
    """A simple relation-aware graph message-passing layer.

    For edge u -> v, it sends a message from u to v using node and relation embeddings.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.rel_lin = nn.Linear(hidden_dim, hidden_dim)
        self.gate_lin = nn.Linear(hidden_dim * 3, 1)
        self.out_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0], edge_index[1]
        src_x = x[src]
        dst_x = x[dst]
        rel_x = edge_attr

        raw_msg = self.msg_lin(src_x) + self.rel_lin(rel_x)
        gate = torch.sigmoid(self.gate_lin(torch.cat([src_x, rel_x, dst_x], dim=-1)))
        msg = raw_msg * gate

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones((dst.size(0), 1), dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp(min=1.0)

        h = self.out_lin(torch.cat([x, agg], dim=-1))
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(x + h)


class SubgraphGNN(nn.Module):
    """Relation-aware GNN that encodes first-order and second-order graph structure."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,  # kept for API compatibility; not used in this pure PyTorch layer
        num_relations: int = 512,
        dropout: float = 0.1,
        use_second_order: bool = True,
    ) -> None:
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_relations = max(1, int(num_relations))
        self.two_hop_rel_id = self.num_relations
        self.use_second_order = use_second_order
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hid_dim)
        # +1 slot is reserved for self-loop and explicit two-hop edges.
        self.rel_emb = nn.Embedding(self.num_relations + 1, hid_dim)
        self.layers = nn.ModuleList([RelGraphLayer(hid_dim, dropout=dropout) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hid_dim * 2, out_dim)

    def _prepare_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        device = x.device

        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device).long()
            if edge_type is None:
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
            else:
                edge_type = edge_type.to(device).long().clamp(min=0, max=self.num_relations - 1)

        if self.use_second_order:
            edge_index, edge_type = add_two_hop_edges(
                edge_index=edge_index,
                edge_type=edge_type,
                num_nodes=num_nodes,
                two_hop_rel_id=self.two_hop_rel_id,
            )

        edge_index, edge_type = add_self_loop_edges(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            self_loop_rel_id=self.two_hop_rel_id,
        )
        return edge_index, edge_type

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return node embeddings and graph embeddings.

        Args:
            x: node features, [N, in_dim]
            edge_index: graph edges, [2, E], source -> destination
            batch: graph id for each node, [N]
            edge_type: relation id for each edge, [E]
        """
        x = self.input_proj(x.float())
        edge_index, edge_type = self._prepare_edges(x, edge_index, edge_type)
        edge_attr = self.rel_emb(edge_type)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_emb = self.out_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return x, graph_emb


class Graph2Prefix(nn.Module):
    """Map graph embeddings to virtual soft-prompt token embeddings."""

    def __init__(
        self,
        graph_dim: int,
        llm_hidden: int,
        prefix_len: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.llm_hidden = llm_hidden
        self.net = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, prefix_len * llm_hidden),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        bsz = graph_emb.size(0)
        prefix = self.net(graph_emb)
        return prefix.view(bsz, self.prefix_len, self.llm_hidden)


def graph_to_soft_prompt(
    gnn: SubgraphGNN,
    graph2prefix: Graph2Prefix,
    graph_batch,
) -> torch.Tensor:
    """Convenience function: GraphBatch -> soft prompt."""
    _, graph_emb = gnn(
        x=graph_batch.x,
        edge_index=graph_batch.edge_index,
        edge_type=getattr(graph_batch, "edge_type", None),
        batch=graph_batch.batch,
    )
    return graph2prefix(graph_emb)
