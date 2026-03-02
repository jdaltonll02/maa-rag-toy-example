from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:  # optional: external QA datasets
    from datasets import load_dataset  # type: ignore[import]
except Exception:  # pragma: no cover - datasets may be missing
    load_dataset = None


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


def _load_nq_open(max_examples: int = 500) -> List[QAExample]:
    if load_dataset is None:
        return []
    ds = load_dataset("google-research-datasets/nq_open", split=f"validation[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        # "answer" is typically a list of acceptable answers; take the first as gold.
        ans_list = row.get("answer", []) or []
        answer = str(ans_list[0]).strip() if ans_list else ""
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_popqa(max_examples: int = 500) -> List[QAExample]:
    if load_dataset is None:
        return []
    ds = load_dataset("akariasai/PopQA", split=f"test[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        # "possible_answers" is a list; use the first as the gold.
        ans_list = row.get("possible_answers", []) or []
        answer = str(ans_list[0]).strip() if ans_list else ""
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_ambigqa(max_examples: int = 500) -> List[QAExample]:
    if load_dataset is None:
        return []
    ds = load_dataset("sewon/ambig_qa", split=f"validation[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        # Use the NQ-style answer field, which is simpler and
        # avoids dealing with nested annotation structures.
        answer = str(row.get("nq_answer", "")).strip()
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_hotpotqa(max_examples: int = 500) -> List[QAExample]:
    """Load multi-hop HotpotQA questions.

    Uses the fullwiki validation split with standard fields.
    """

    if load_dataset is None:
        return []
    ds = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        answer = str(row.get("answer", "")).strip()
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_2wiki(max_examples: int = 500) -> List[QAExample]:
    """Load 2WikiMultiHopQA questions via the HF dataset used in the baseline."""

    if load_dataset is None:
        return []
    try:
        ds = load_dataset("voidful/2WikiMultihopQA", split=f"test[:{max_examples}]")
    except Exception:
        # In some environments this dataset may not parse cleanly;
        # fall back to an empty list so the rest still works.
        return []
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        answer = str(row.get("answer", "")).strip()
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_musique(max_examples: int = 500) -> List[QAExample]:
    """Load Musique multi-hop questions.

    Uses the HF dataset id from the baseline script.
    """

    if load_dataset is None:
        return []
    ds = load_dataset("bdsaglam/musique", split=f"validation[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        question = str(row["question"]).strip()
        answer = str(row.get("answer", "")).strip()
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def _load_bamboogle(max_examples: int = 500) -> List[QAExample]:
    """Load Bamboogle multi-hop questions from HF."""

    if load_dataset is None:
        return []
    ds = load_dataset("chiayewken/bamboogle", split=f"test[:{max_examples}]")
    out: List[QAExample] = []
    for row in ds:
        # Be defensive about field names; skip rows that don't fit.
        raw_q = row.get("question") or row.get("query") or row.get("prompt")
        raw_a = row.get("answer") or row.get("answers")
        question = str(raw_q).strip() if raw_q is not None else ""
        # If answers is a list, take the first one.
        if isinstance(raw_a, list):
            ans_val = raw_a[0] if raw_a else ""
        else:
            ans_val = raw_a or ""
        answer = str(ans_val).strip()
        if question and answer:
            out.append(QAExample(question=question, answer=answer))
    return out


def get_dataset(
    use_external: bool = False,
    max_per_dataset: int = 200,
    mode: str = "single-hop",
) -> List[QAExample]:
    """Return QA examples for the toy environment.

    Parameters
    ----------
    use_external:
        When False, return only the small built-in toy set.
        When True and ``datasets`` is available, augment the toy set with
        samples drawn from the baseline datasets.
    max_per_dataset:
        Maximum number of examples to pull from each external dataset.
    mode:
        "single-hop"   -> NQ-Open, PopQA, AmbigQA.
        "multi-hop"    -> HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle.
        "mixed"        -> both single- and multi-hop.
    """

    toy = list(_DATA)
    if not use_external or load_dataset is None:
        return toy

    mode = mode.lower()
    external: List[QAExample] = []

    if mode in {"single-hop", "mixed"}:
        external.extend(_load_nq_open(max_examples=max_per_dataset))
        external.extend(_load_popqa(max_examples=max_per_dataset))
        external.extend(_load_ambigqa(max_examples=max_per_dataset))

    if mode in {"multi-hop", "mixed"}:
        external.extend(_load_hotpotqa(max_examples=max_per_dataset))
        external.extend(_load_2wiki(max_examples=max_per_dataset))
        external.extend(_load_musique(max_examples=max_per_dataset))
        external.extend(_load_bamboogle(max_examples=max_per_dataset))

    return toy + external


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


def iter_questions_and_answers(
    use_external: bool = False,
    max_per_dataset: int = 200,
    mode: str = "single-hop",
) -> List[Tuple[str, str]]:
    examples = get_dataset(use_external=use_external, max_per_dataset=max_per_dataset, mode=mode)
    return [(ex.question, ex.answer) for ex in examples]
