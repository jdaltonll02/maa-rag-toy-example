from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class QAExample:
    question: str
    answer: str


# A tiny synthetic QA set with different question types.
_DATA: List[QAExample] = [
    QAExample(
        question="What lifestyle change can reduce the risk of heart disease?",
        answer="Eating a healthy diet can reduce the risk of heart disease.",
    ),
    QAExample(
        question="What structure do graph databases use to represent data?",
        answer="Graph databases represent data as nodes and edges.",
    ),
    QAExample(
        question="What is a potential problem with large language models when context is missing?",
        answer="Large language models may hallucinate when context is missing.",
    ),
]


def get_dataset() -> List[QAExample]:
    return list(_DATA)


def _normalize(text: str) -> List[str]:
    return [t for t in text.lower().replace(".", " ").split() if t]


def f1_score(pred: str, gold: str) -> float:
    """Simple token-level F1 used as answer quality reward."""

    p_tokens = _normalize(pred)
    g_tokens = _normalize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def make_claims(answer: str) -> List[str]:
    """Very small helper: split answer into clause-like segments."""

    clauses = [c.strip() for c in answer.split(".")]
    return [c for c in clauses if c]


def iter_questions_and_answers() -> List[Tuple[str, str]]:
    return [(ex.question, ex.answer) for ex in _DATA]
