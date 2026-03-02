from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ToyDocument:
    doc_id: str
    title: str
    text: str


# A tiny in-memory corpus for toy RAG / Graph-RAG.
# In a real system this would be a large vector index.
_CORPUS: List[ToyDocument] = [
    ToyDocument(
        doc_id="d1",
        title="Heart Disease Basics",
        text=(
            "Heart disease refers to several types of heart conditions. "
            "The most common type is coronary artery disease, which affects "
            "blood flow to the heart."
        ),
    ),
    ToyDocument(
        doc_id="d2",
        title="Healthy Diet",
        text=(
            "A healthy diet includes vegetables, fruits, whole grains, and "
            "lean proteins. Limiting salt and sugar can reduce risk of "
            "heart disease."
        ),
    ),
    ToyDocument(
        doc_id="d3",
        title="Graph Databases",
        text=(
            "Graph databases store data as nodes and edges. They are useful "
            "for representing relationships in knowledge graphs."
        ),
    ),
    ToyDocument(
        doc_id="d4",
        title="Large Language Models",
        text=(
            "Large language models (LLMs) can answer questions using context. "
            "They may hallucinate if context is missing or ambiguous."
        ),
    ),
]


# A tiny knowledge graph for the Graph-RAG toy.
# Keys are node names, values are lists of (relation, neighbor).
_KG: Dict[str, List[Dict[str, str]]] = {
    "heart_disease": [
        {"relation": "risk_factor", "neighbor": "high_blood_pressure"},
        {"relation": "reduced_by", "neighbor": "healthy_diet"},
    ],
    "healthy_diet": [
        {"relation": "reduces_risk_of", "neighbor": "heart_disease"},
    ],
    "graph_database": [
        {"relation": "stores", "neighbor": "knowledge_graph"},
    ],
}


def _simple_score(query: str, text: str) -> int:
    """Very small scoring function based on token overlap.

    This keeps the toy retriever deterministic and dependency-free.
    """

    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    return len(q_tokens & t_tokens)


def retrieve_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k documents with a crude lexical score."""

    scored = [
        (doc, _simple_score(query, f"{doc.title} {doc.text}")) for doc in _CORPUS
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs: List[Dict[str, Any]] = []
    for doc, score in scored[:k]:
        top_docs.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "score": score,
            }
        )
    return top_docs


def retrieve_graph_context(query: str) -> Dict[str, Any]:
    """Return a tiny subgraph relevant to the query.

    We match nodes whose name tokens overlap with the query.
    """

    q_tokens = set(query.lower().split())
    nodes = []
    edges = []
    for node, rels in _KG.items():
        if q_tokens & set(node.split("_")):
            nodes.append(node)
            for rel in rels:
                edges.append(
                    {
                        "source": node,
                        "relation": rel["relation"],
                        "target": rel["neighbor"],
                    }
                )
    return {"nodes": nodes, "edges": edges}
