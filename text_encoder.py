"""Shared pretrained sentence encoder for GraSP.

Backs two different uses with the same embedding space:
  - subgraph.py: graph node features (replacing the earlier hash-based projection).
  - freebase_func.py: cosine-similarity ranking of candidate relations/entities against the
    question during SPARQL retrieval, instead of arbitrary alphabetical/first-N capping.

Mean-pooling implementation follows the standard sentence-transformers recipe (AutoModel +
attention-masked mean pooling), so any BERT-family checkpoint on the HF Hub works without adding
the separate `sentence-transformers` package as a dependency -- including plain `bert-base-uncased`
(paper's experiment config), not just checkpoints purpose-built for mean pooling.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Paper's experiment config: "We adopt bert-base-uncased as the text encoder, which has a hidden
# size of 768."
DEFAULT_TEXT_ENCODER = "bert-base-uncased"

_tokenizers: Dict[str, AutoTokenizer] = {}
_models: Dict[str, AutoModel] = {}


def _load(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    if model_name not in _models:
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _models[model_name] = model
    return _tokenizers[model_name], _models[model_name]


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embedding_dim(model_name: str = DEFAULT_TEXT_ENCODER) -> int:
    _, model = _load(model_name)
    return model.config.hidden_size


@torch.no_grad()
def encode(texts: Sequence[str], model_name: str = DEFAULT_TEXT_ENCODER) -> torch.Tensor:
    """Batch-encode texts into L2-normalized, mean-pooled embeddings."""
    if not texts:
        return torch.empty((0, embedding_dim(model_name)))
    tokenizer, model = _load(model_name)
    encoded = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    model_output = model(**encoded)
    embeddings = _mean_pooling(model_output, encoded["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def rank_by_similarity(query: str, candidates: Sequence[str], model_name: str = DEFAULT_TEXT_ENCODER) -> List[int]:
    """Return indices into `candidates`, sorted by cosine similarity to `query` (descending)."""
    if not candidates:
        return []
    query_emb = encode([query], model_name=model_name)
    candidate_emb = encode(list(candidates), model_name=model_name)
    similarities = (candidate_emb @ query_emb.T).squeeze(1)  # both L2-normalized -> dot == cosine
    return torch.argsort(similarities, descending=True).tolist()
