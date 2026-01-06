import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, subgraph

from transformers import AutoTokenizer, AutoModel


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over the token dimension with attention mask.

    Args:
        last_hidden_state: Tensor of shape [B, T, H]
        attention_mask: Tensor of shape [B, T]

    Returns:
        Tensor of shape [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
    return summed / denom


@torch.no_grad()
def _bert_encode_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size=64,
    max_length=64
):
    """
    Encode a list of texts into L2-normalized embeddings using BERT
    with mean pooling.

    Args:
        texts: List[str]
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch.device
        batch_size: encoding batch size
        max_length: max token length

    Returns:
        Tensor of shape [N, H], L2-normalized
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        outputs = model(**tokenized)
        pooled = _mean_pool(
            outputs.last_hidden_state,
            tokenized["attention_mask"]
        )  # [B, H]

        pooled = F.normalize(pooled, p=2, dim=-1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)  # [N, H]


def extract_n_hop_subgraph_with_bert_filter(
    question: str,
    center_node_id: int,
    edge_index: torch.Tensor,
    num_hops: int,
    entity_texts: dict,
    top_k: int,
    rel_ids: torch.Tensor | None = None,
    bert_model_name: str = "bert-base-uncased",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    max_length: int = 64,
):
    """
    Extract an n-hop subgraph and filter nodes using BERT-based
    question–entity semantic similarity.

    Pipeline:
        1) Encode the question and entity texts using BERT
        2) Compute cosine similarity between question and entities
        3) Select top-K most similar entities (always keep the center node)
        4) Extract n-hop subgraph around the center node
        5) Filter the subgraph to keep only top-K relevant nodes and edges
    """
    device = torch.device(device)

    # Initialize BERT
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name).to(device)
    model.eval()

    # Encode the question
    question_emb = _bert_encode_texts(
        [question],
        tokenizer,
        model,
        device,
        batch_size=1,
        max_length=max_length
    ).to(device)  # [1, H]

    # Encode entity texts
    node_ids = list(entity_texts.keys())
    texts = [entity_texts[nid] for nid in node_ids]

    entity_embs = _bert_encode_texts(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length
    ).to(device)  # [N, H]

    # Cosine similarity (dot product since embeddings are normalized)
    similarities = (entity_embs @ question_emb.T).squeeze(-1)  # [N]

    # Select top-K most relevant entities
    k = min(top_k, similarities.numel())
    _, topk_indices = torch.topk(similarities, k=k, largest=True)

    kept_node_ids = {node_ids[i] for i in topk_indices.tolist()}
    kept_node_ids.add(center_node_id)

    sim_scores = {
        node_ids[i]: float(similarities[i].item())
        for i in range(len(node_ids))
    }

    # Extract n-hop subgraph around the center node
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=torch.tensor([center_node_id], dtype=torch.long),
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True
    )

    sub_rel_ids = None
    if rel_ids is not None:
        sub_rel_ids = rel_ids[edge_mask]

    # Filter nodes inside the subgraph based on semantic relevance
    keep_mask = torch.tensor(
        [int(n.item()) in kept_node_ids for n in subset],
        dtype=torch.bool
    )

    filtered_edge_index, filtered_edge_mask = subgraph(
        subset=keep_mask,
        edge_index=sub_edge_index,
        relabel_nodes=True,
        num_nodes=subset.size(0),
        return_edge_mask=True
    )

    filtered_subset = subset[keep_mask]

    # Recompute center node index after filtering
    center_positions = (filtered_subset == center_node_id).nonzero(as_tuple=True)[0]
    if center_positions.numel() == 0:
        raise ValueError("center_node_id was unexpectedly filtered out.")

    filtered_mapping = int(center_positions.item())

    # Combine original edge mask with filtered edge mask
    final_edge_mask = edge_mask.clone()
    sub_edge_positions = edge_mask.nonzero(as_tuple=True)[0]
    final_edge_mask[sub_edge_positions[~filtered_edge_mask]] = False

    filtered_rel_ids = None
    if sub_rel_ids is not None:
        filtered_rel_ids = sub_rel_ids[filtered_edge_mask]

    return (
        filtered_subset,
        filtered_edge_index,
        filtered_rel_ids,
        filtered_mapping,
        final_edge_mask,
        kept_node_ids,
        sim_scores
    )
