"""Second-order GNN soft-prompt modules for GraSP.

This file only handles graph encoding:
    KG graph + question query -> second-order relation-aware GNN -> graph embedding -> soft prompt

The second-order part explicitly adds two-hop edges u->w whenever u->v and v->w exist.
Those edges allow the GNN to encode neighbor-of-neighbor information as soft prompts.

RelGraphLayer implements the paper's query-conditioned attention:
    h_i^(l+1) = sigma( sum_j a_ij W^(l) h_j^(l) + b^(l) )
    a_ij = softmax_j( f(W(q||h_i), W(h_j||r_ij)) ),  f = inner product
See scatter_softmax for the per-target-node softmax normalization over neighbors.

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


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` grouped by `index` (e.g. normalize attention over each target node's
    incoming edges). Numerically stable via a per-group max subtraction."""
    if scores.numel() == 0:
        return scores
    max_per_group = scores.new_full((num_nodes,), float("-inf"))
    max_per_group.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    exp_scores = (scores - max_per_group[index]).exp()
    sum_per_group = torch.zeros(num_nodes, dtype=exp_scores.dtype, device=exp_scores.device)
    sum_per_group.index_add_(0, index, exp_scores)
    return exp_scores / (sum_per_group[index] + 1e-12)


class RelGraphLayer(nn.Module):
    """Query-conditioned relation-attention layer.

    For a target entity e_i and neighbor e_j (edge j -> i):
        h_i^(l+1) = sigma( sum_j a_ij W^(l) h_j^(l) + b^(l) )
        a_ij = softmax_j( f(W(q || h_i), W(h_j || r_ij)) ),  f = inner product

    `attn_proj` is the shared W projecting both the (query, target) side and the
    (neighbor, relation) side of the attention score. `msg_lin` is the layer-specific W^(l)
    applied to the message value h_j.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        target_side = self.attn_proj(torch.cat([query, x], dim=-1))             # W(q_i || h_i), per node
        neighbor_side = self.attn_proj(torch.cat([x[src], edge_attr], dim=-1))  # W(h_j || r_ij), per edge

        scores = (target_side[dst] * neighbor_side).sum(dim=-1)  # f(.,.) = inner product
        attn = scatter_softmax(scores, dst, num_nodes)           # softmax over j in N(i)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        messages = self.msg_lin(x[src]) * attn.unsqueeze(-1)  # a_ij * W^(l) h_j
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)  # sum_j

        return F.relu(agg + self.bias)  # sigma( ... + b^(l) )


class SubgraphGNN(nn.Module):
    """Relation-aware GNN that encodes first-order and second-order graph structure."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,  # kept for API/checkpoint compatibility; the paper's attention is single-head
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
        query: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return node embeddings and graph embeddings.

        Args:
            x: node features, [N, in_dim]
            edge_index: graph edges, [2, E], source -> destination
            batch: graph id for each node, [N]
            edge_type: relation id for each edge, [E]
            query: per-graph query vector `q` in a_ij = softmax(f(W(q||h_i), W(h_j||r_ij))) --
                typically the question's text-encoder embedding, shape [num_graphs, in_dim].
                Defaults to zeros (attention degenerates to being question-agnostic) if omitted.
        """
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        if query is None:
            query = x.new_zeros((num_graphs, x.size(-1)))

        x = self.input_proj(x.float())
        query = self.input_proj(query.float())
        query_per_node = query[batch] if batch.numel() else x.new_zeros((x.size(0), query.size(-1)))

        edge_index, edge_type = self._prepare_edges(x, edge_index, edge_type)
        edge_attr = self.rel_emb(edge_type)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, query_per_node)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_emb = self.out_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return x, graph_emb


def pad_node_embeddings(node_emb: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unflatten a [N_total, d] node embedding matrix (many graphs concatenated, `batch` gives each
    node's graph id) into a padded [B, max_n, d] tensor plus a boolean [B, max_n] validity mask,
    since different subgraphs have different entity counts."""
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    d = node_emb.size(-1)
    counts = torch.bincount(batch, minlength=num_graphs) if batch.numel() else torch.zeros(num_graphs, dtype=torch.long)
    max_n = int(counts.max().item()) if counts.numel() else 0
    padded = node_emb.new_zeros((num_graphs, max_n, d))
    mask = torch.zeros((num_graphs, max_n), dtype=torch.bool, device=node_emb.device)
    if batch.numel() == 0:
        return padded, mask

    order = torch.argsort(batch, stable=True)
    sorted_batch = batch[order]
    first_occurrence = torch.searchsorted(sorted_batch, sorted_batch)
    pos_in_sorted = torch.arange(sorted_batch.size(0), device=batch.device) - first_occurrence
    pos = torch.empty_like(pos_in_sorted)
    pos[order] = pos_in_sorted

    padded[batch, pos] = node_emb
    mask[batch, pos] = True
    return padded, mask


class Graph2Prefix(nn.Module):
    """Per-entity FFN projecting GNN node embeddings into the LLM's embedding space (paper Eq. 5):

        H_hat = FFN(H),  H in R^{n x d} (one subgraph's entity embeddings), H_hat in R^{n x d_LLM}

    Since subgraphs in a batch have different entity counts n, the output is padded to the batch's
    max entity count and returned with a validity mask (analogous to a token attention mask).
    """

    def __init__(
        self,
        graph_dim: int,
        llm_hidden: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.llm_hidden = llm_hidden
        self.net = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, llm_hidden),
        )

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hat = self.net(node_emb)  # FFN(H), per-entity: [N_total, llm_hidden]
        return pad_node_embeddings(h_hat, batch)


def graph_to_soft_prompt(
    gnn: SubgraphGNN,
    graph2prefix: Graph2Prefix,
    graph_batch,
    query: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience function: GraphBatch -> (soft prompt embeddings, validity mask)."""
    node_emb, _ = gnn(
        x=graph_batch.x,
        edge_index=graph_batch.edge_index,
        edge_type=getattr(graph_batch, "edge_type", None),
        batch=graph_batch.batch,
        query=query,
    )
    return graph2prefix(node_emb, graph_batch.batch)
