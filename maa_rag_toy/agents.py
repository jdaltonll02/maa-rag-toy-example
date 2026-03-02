from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import retriever
from . import data as toy_data

try:  # optional lightweight HF support
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - library may be missing in some envs
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


@dataclass
class ToyState:
    """State container for the toy environment.

    This mirrors the paper's (q, H_t, D_t, A_t, C_t) abstraction
    using plain Python fields.
    """

    question: str
    history: List[str]
    docs: List[Dict[str, Any]]
    answer: Optional[str]
    verification_score: float


class ToyAgentBase:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, state: ToyState) -> ToyState:
        return self.run(state)

    def run(self, state: ToyState) -> ToyState:  # pragma: no cover - interface
        raise NotImplementedError


class ToyRetrieverAgent(ToyAgentBase):
    def __init__(self, top_k: int = 3) -> None:
        super().__init__(name="retrieve")
        self.top_k = top_k

    def run(self, state: ToyState) -> ToyState:
        docs = retriever.retrieve_docs(state.question, k=self.top_k)
        state.docs = docs
        state.history.append(f"Retrieved {len(docs)} documents.")
        return state


class ToyReasoningAgent(ToyAgentBase):
    """Toy Self-Graph Reasoning agent.

    For simplicity we construct a pseudo-graph description from
    retrieved docs instead of full graph generation.
    """

    def __init__(self) -> None:
        super().__init__(name="reason")

    def run(self, state: ToyState) -> ToyState:
        if not state.docs:
            state.history.append("Reasoner: no docs, skipping.")
            return state
        bullets = []
        for doc in state.docs:
            bullets.append(f"Node: {doc['title']} -> {doc['text'][:80]}...")
        reasoning = "\n".join(bullets)
        state.history.append("Reasoning graph built:\n" + reasoning)
        return state


class ToyVerificationAgent(ToyAgentBase):
    """Toy verification agent.

    Approximates NLI-based verification by measuring lexical overlap
    between answer claims and retrieved docs.
    """

    def __init__(self) -> None:
        super().__init__(name="verify")

    def run(self, state: ToyState) -> ToyState:
        if not state.answer or not state.docs:
            state.verification_score = 0.0
            state.history.append("Verifier: missing answer or docs; score = 0.0")
            return state

        claims = toy_data.make_claims(state.answer)
        doc_text = "\n".join(d["text"] for d in state.docs)
        from .data import f1_score

        scores = [f1_score(c, doc_text) for c in claims]
        state.verification_score = max(scores) if scores else 0.0
        state.history.append(
            f"Verifier: computed verification score = {state.verification_score:.3f}"
        )
        return state


class ToyAnswerGenBase(ToyAgentBase):
    def __init__(self, name: str, hf_model_name: str = "gpt2") -> None:
        super().__init__(name=name)
        self.hf_model_name = hf_model_name
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            return
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.hf_model_name)

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            # Fallback: deterministic placeholder behavior for environments
            return prompt[-200:]
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() or text.strip()


class ToyAnswerGenRAGAgent(ToyAnswerGenBase):
    def __init__(self, hf_model_name: str = "gpt2") -> None:
        super().__init__(name="answer_rag", hf_model_name=hf_model_name)

    def run(self, state: ToyState) -> ToyState:
        context = "\n".join(d["text"] for d in state.docs)
        prompt = (
            "You are a helpful medical assistant. Use the following documents "
            "to answer the question.\n\nDocuments:\n" + context + "\n\nQuestion: "
            + state.question
            + "\nAnswer:"
        )
        answer = self._generate(prompt)
        state.answer = answer
        state.history.append("Answer (RAG) generated.")
        return state


class ToyAnswerGenGraphRAGAgent(ToyAnswerGenBase):
    def __init__(self, hf_model_name: str = "gpt2") -> None:
        super().__init__(name="answer_graph_rag", hf_model_name=hf_model_name)

    def run(self, state: ToyState) -> ToyState:
        graph_ctx = retriever.retrieve_graph_context(state.question)
        prompt = (
            "You are a knowledge-graph assistant. Use the following graph "
            "structure to answer the question.\n\nGraph nodes: "
            + ", ".join(graph_ctx.get("nodes", []))
            + "\nGraph edges: "
            + ", ".join(
                f"{e['source']} -[{e['relation']}]-> {e['target']}"
                for e in graph_ctx.get("edges", [])
            )
            + "\n\nQuestion: "
            + state.question
            + "\nAnswer:"
        )
        answer = self._generate(prompt)
        state.answer = answer
        state.history.append("Answer (Graph RAG) generated.")
        return state


class ToyAnswerGenContextLLMAgent(ToyAnswerGenBase):
    def __init__(self, hf_model_name: str = "gpt2") -> None:
        super().__init__(name="answer_context_llm", hf_model_name=hf_model_name)

    def run(self, state: ToyState) -> ToyState:
        prompt = (
            "You are a helpful assistant. Answer the question based only on "
            "general knowledge and the question itself.\n\nQuestion: "
            + state.question
            + "\nAnswer:"
        )
        answer = self._generate(prompt)
        state.answer = answer
        state.history.append("Answer (context LLM) generated.")
        return state
