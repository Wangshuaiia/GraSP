"""Minimal SPARQL-based Freebase retriever for GraSP.

Given a topic entity id (a Freebase mid, e.g. "m.055d_1d"), `fetch_khop_triples` walks outward
over the live KG via SPARQL and returns real (head, relation, tail) triples -- both endpoints and
edges come straight from the endpoint, not from a gold parse. This replaces the earlier
placeholder that synthesized triples out of the dataset's gold answer/relation chain.

Requires a running Freebase-compatible SPARQL endpoint (see the ToG/RoG project READMEs for how
to stand up a local Virtuoso instance loaded with a Freebase dump). Point FREEBASE_SPARQL_ENDPOINT
at it; the default below is just a placeholder.

Breadth at each hop is controlled by simple caps (max_hops / max_relations_per_hop /
max_entities_per_relation / max_triples). When a `question` is supplied, candidates are ranked by
cosine similarity to the question (via text_encoder.py) before capping, instead of an arbitrary
alphabetical/first-N cut -- no LLM is involved, just embedding similarity.

Mock mode: set FREEBASE_MOCK=1 (or pass --mock_kg to train.py/infer.py) to bypass SPARQL entirely
and generate a small synthetic KG neighborhood per entity instead. This is for exercising the
rest of the pipeline (graph construction, GNN, LLM training loop) when no live endpoint is
reachable. The synthetic edges are derived only from the entity id/relation strings (deterministic
hashing), never from the dataset's gold answer, so it doesn't reintroduce answer leakage -- but the
"knowledge" itself is meaningless. Do not use it for real training or evaluation.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from SPARQLWrapper import JSON, SPARQLWrapper

from text_encoder import DEFAULT_TEXT_ENCODER, rank_by_similarity

SPARQL_ENDPOINT = os.environ.get("FREEBASE_SPARQL_ENDPOINT", "http://192.168.80.12:8890/sparql")

_MOCK_RELATION_POOL = [
    "common.topic.notable_for",
    "people.person.profession",
    "location.location.containedby",
    "organization.organization.founders",
    "film.film.genre",
    "book.book.author",
    "people.person.nationality",
    "sports.pro_athlete.teams",
    "music.artist.genre",
    "business.business_operation.industry",
]

_MOCK_NAME_WORDS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "River", "Mountain", "City", "Studio", "Records", "Museum", "Festival", "Institute",
]

_mock_warned = False


def _mock_enabled() -> bool:
    """Read dynamically (not frozen at import time) so callers can flip FREEBASE_MOCK after import."""
    enabled = os.environ.get("FREEBASE_MOCK", "0") == "1"
    global _mock_warned
    if enabled and not _mock_warned:
        # stderr, not stdout: infer.py prints its JSON results to stdout (e.g. `infer.py > preds.json`
        # for eval.py to consume), and this banner would corrupt that output if mixed in.
        print(
            "[freebase_func] FREEBASE_MOCK=1: using synthetic mock KG data instead of live SPARQL. "
            "Pipeline smoke-testing only -- do NOT use for real training/evaluation.",
            file=sys.stderr,
        )
        _mock_warned = True
    return enabled


def _mock_relations(entity_id: str) -> Tuple[List[str], List[str]]:
    head_rng = random.Random(f"head::{entity_id}")
    tail_rng = random.Random(f"tail::{entity_id}")
    head = head_rng.sample(_MOCK_RELATION_POOL, head_rng.randint(2, 4))
    tail = tail_rng.sample(_MOCK_RELATION_POOL, tail_rng.randint(1, 3))
    return head, tail


def _mock_neighbor_ids(entity_id: str, relation: str, n: int = 3) -> List[str]:
    rng = random.Random(f"{entity_id}::{relation}")
    return [f"m.mock_{rng.randrange(10**6):06d}" for _ in range(rng.randint(1, n))]


def _mock_entity_name(entity_id: str) -> str:
    if not entity_id.startswith("m.mock_"):
        return entity_id
    rng = random.Random(entity_id)
    return f"{rng.choice(_MOCK_NAME_WORDS)} {rng.choice(_MOCK_NAME_WORDS)}"

# Safety cap on how many raw SPARQL candidates get their names resolved + similarity-scored per
# relation, before max_entities_per_relation trims further. Keeps hub entities (e.g. "United
# States") with thousands of edges from triggering thousands of id2entity_name lookups.
_MAX_CANDIDATES_TO_SCORE = 50

NS_PREFIX = "http://rdf.freebase.com/ns/"

SPARQL_HEAD_RELATIONS = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
  ns:%s ?relation ?x .
}"""

SPARQL_TAIL_RELATIONS = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
  ?x ?relation ns:%s .
}"""

# entity --relation--> ?tailEntity
SPARQL_FORWARD_ENTITIES = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  ns:%s ns:%s ?tailEntity .
}"""

# ?headEntity --relation--> entity
SPARQL_BACKWARD_ENTITIES = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?headEntity
WHERE {
  ?headEntity ns:%s ns:%s .
}"""

SPARQL_ENTITY_NAME = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?name
WHERE {
  { ns:%s ns:type.object.name ?name . }
  UNION
  { ns:%s <http://www.w3.org/2002/07/owl#sameAs> ?name . }
}"""

_relation_cache: Dict[str, Tuple[List[str], List[str]]] = {}
_entity_cache: Dict[Tuple[str, str, bool], List[str]] = {}
_name_cache: Dict[str, str] = {}


def reset_cache() -> None:
    """Clear the in-process SPARQL result caches (mainly useful for tests)."""
    _relation_cache.clear()
    _entity_cache.clear()
    _name_cache.clear()


def check_end_word(s: str) -> bool:
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation: str) -> bool:
    return bool(
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
    )


def execute_sparql(query: str, timeout: int = 15) -> List[dict]:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    try:
        results = sparql.query().convert()
    except Exception as exc:  # SPARQLWrapper raises a mix of urllib/SPARQL exceptions
        raise RuntimeError(
            f"SPARQL query against {SPARQL_ENDPOINT} failed ({exc}). Check that a Freebase "
            "SPARQL endpoint is running and reachable, or set the FREEBASE_SPARQL_ENDPOINT "
            "environment variable to the correct address."
        ) from exc
    return results["results"]["bindings"]


def _strip_ns(uri: str) -> str:
    return uri.replace(NS_PREFIX, "")


def id2entity_name(entity_id: str) -> str:
    """Resolve a Freebase mid to its human-readable name, falling back to the raw id.

    Falling back to the id (instead of a shared placeholder like "UnName_Entity") matters here:
    node identity in build_graph_from_triples is keyed by name string, so collapsing every
    unnamed/CVT node onto one placeholder string would incorrectly merge distinct graph nodes.
    """
    if entity_id in _name_cache:
        return _name_cache[entity_id]
    if _mock_enabled():
        name = _mock_entity_name(entity_id)
        _name_cache[entity_id] = name
        return name
    if not entity_id.startswith(("m.", "g.")):
        _name_cache[entity_id] = entity_id
        return entity_id
    bindings = execute_sparql(SPARQL_ENTITY_NAME % (entity_id, entity_id))
    name = bindings[0]["name"]["value"] if bindings else entity_id
    _name_cache[entity_id] = name
    return name


def get_relations(entity_id: str) -> Tuple[List[str], List[str]]:
    """Return (outgoing_relations, incoming_relations) for an entity."""
    if entity_id in _relation_cache:
        return _relation_cache[entity_id]
    if _mock_enabled():
        result = _mock_relations(entity_id)
        _relation_cache[entity_id] = result
        return result
    head = [_strip_ns(b["relation"]["value"]) for b in execute_sparql(SPARQL_HEAD_RELATIONS % entity_id)]
    tail = [_strip_ns(b["relation"]["value"]) for b in execute_sparql(SPARQL_TAIL_RELATIONS % entity_id)]
    result = (head, tail)
    _relation_cache[entity_id] = result
    return result


def _filter_entity_ids(bindings: List[dict], var: str) -> List[str]:
    ids = []
    for b in bindings:
        raw = b[var]["value"]
        if raw.startswith(NS_PREFIX):
            stripped = _strip_ns(raw)
            if stripped.startswith(("m.", "g.")):
                ids.append(stripped)
    return ids


def get_forward_entities(entity_id: str, relation: str) -> List[str]:
    """Entities reachable as entity --relation--> ?tailEntity."""
    key = (entity_id, relation, True)
    if key in _entity_cache:
        return _entity_cache[key]
    if _mock_enabled():
        ids = _mock_neighbor_ids(entity_id, relation)
        _entity_cache[key] = ids
        return ids
    bindings = execute_sparql(SPARQL_FORWARD_ENTITIES % (entity_id, relation))
    ids = _filter_entity_ids(bindings, "tailEntity")
    _entity_cache[key] = ids
    return ids


def get_backward_entities(entity_id: str, relation: str) -> List[str]:
    """Entities reachable as ?headEntity --relation--> entity."""
    key = (entity_id, relation, False)
    if key in _entity_cache:
        return _entity_cache[key]
    if _mock_enabled():
        ids = _mock_neighbor_ids(entity_id, f"reverse::{relation}")
        _entity_cache[key] = ids
        return ids
    bindings = execute_sparql(SPARQL_BACKWARD_ENTITIES % (relation, entity_id))
    ids = _filter_entity_ids(bindings, "headEntity")
    _entity_cache[key] = ids
    return ids


def _rank_relations(relations: Sequence[str], question: Optional[str], text_encoder: str) -> List[str]:
    relations = list(relations)
    if not question or not relations:
        return sorted(relations)
    order = rank_by_similarity(question, relations, model_name=text_encoder)
    return [relations[i] for i in order]


def _rank_entities(entity_ids: Sequence[str], question: Optional[str], text_encoder: str) -> List[str]:
    candidates = list(entity_ids)[:_MAX_CANDIDATES_TO_SCORE]
    if not question or not candidates:
        return candidates
    names = [id2entity_name(eid) for eid in candidates]
    order = rank_by_similarity(question, names, model_name=text_encoder)
    return [candidates[i] for i in order]


def fetch_khop_triples(
    topic_entity_id: str,
    topic_entity_name: Optional[str] = None,
    question: Optional[str] = None,
    max_hops: int = 2,
    max_relations_per_hop: int = 8,
    max_entities_per_relation: int = 10,
    max_triples: int = 60,
    text_encoder: str = DEFAULT_TEXT_ENCODER,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, str]]:
    """BFS a live Freebase KG outward from a topic entity via SPARQL.

    Returns ((head_name, relation, tail_name) triples with human-readable names, a name->mid map
    covering every entity discovered during this call). The name->mid map lets a caller resolve an
    entity a downstream LLM picked (by name) back into a Freebase id to re-query from -- needed for
    the iterative multi-hop "draft-and-refine" loop (see infer.py).

    At each hop, candidate relations and entities are ranked by cosine similarity to `question` (if
    given) and capped by max_relations_per_hop / max_entities_per_relation; without a question
    they're capped in a stable (alphabetical / SPARQL-order) but otherwise arbitrary way.
    """
    topic_name = topic_entity_name or id2entity_name(topic_entity_id)

    triples: List[Tuple[str, str, str]] = []
    name_to_mid: Dict[str, str] = {topic_name: topic_entity_id}
    seen_triples = set()
    visited = {topic_entity_id}
    frontier = [(topic_entity_id, topic_name)]

    for _ in range(max_hops):
        if not frontier or len(triples) >= max_triples:
            break
        next_frontier: List[Tuple[str, str]] = []

        for entity_id, entity_name in frontier:
            if len(triples) >= max_triples:
                break
            head_relations, tail_relations = get_relations(entity_id)
            head_relations = _rank_relations(
                [r for r in head_relations if not abandon_rels(r)], question, text_encoder
            )[:max_relations_per_hop]
            tail_relations = _rank_relations(
                [r for r in tail_relations if not abandon_rels(r)], question, text_encoder
            )[:max_relations_per_hop]

            for relation in head_relations:
                if len(triples) >= max_triples:
                    break
                candidates = _rank_entities(get_forward_entities(entity_id, relation), question, text_encoder)
                for tail_id in candidates[:max_entities_per_relation]:
                    key = (entity_id, relation, tail_id)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    tail_name = id2entity_name(tail_id)
                    name_to_mid[tail_name] = tail_id
                    triples.append((entity_name, relation, tail_name))
                    if tail_id not in visited:
                        visited.add(tail_id)
                        next_frontier.append((tail_id, tail_name))
                    if len(triples) >= max_triples:
                        break

            for relation in tail_relations:
                if len(triples) >= max_triples:
                    break
                candidates = _rank_entities(get_backward_entities(entity_id, relation), question, text_encoder)
                for head_id in candidates[:max_entities_per_relation]:
                    key = (head_id, relation, entity_id)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    head_name = id2entity_name(head_id)
                    name_to_mid[head_name] = head_id
                    triples.append((head_name, relation, entity_name))
                    if head_id not in visited:
                        visited.add(head_id)
                        next_frontier.append((head_id, head_name))
                    if len(triples) >= max_triples:
                        break

        frontier = next_frontier

    return triples, name_to_mid
