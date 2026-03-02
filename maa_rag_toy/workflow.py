from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple

from .agents import (
    ToyState,
    ToyRetrieverAgent,
    ToyReasoningAgent,
    ToyVerificationAgent,
    ToyAnswerGenRAGAgent,
    ToyAnswerGenGraphRAGAgent,
    ToyAnswerGenContextLLMAgent,
)
from .data import f1_score


class MacroAction(Enum):
    RETRIEVE = 0
    REASON = 1
    VERIFY = 2
    ANSWER = 3  # low-level chooses which answer mode
    TERMINATE = 4


@dataclass
class EnvConfig:
    max_steps: int = 6
    lambda_step: float = 0.01
    lambda_cost: float = 0.0  # placeholder for token/latency cost if needed


@dataclass
class MAARagToyEnv:
    """Toy environment for MAA-RAG formulated as a Semi-MDP.

    The state is a ToyState plus a small step counter; rewards are based on
    answer quality at termination and small step penalties otherwise.
    """

    config: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self) -> None:
        self.retriever = ToyRetrieverAgent()
        self.reasoner = ToyReasoningAgent()
        self.verifier = ToyVerificationAgent()
        self.answer_rag = ToyAnswerGenRAGAgent()
        self.answer_graph = ToyAnswerGenGraphRAGAgent()
        self.answer_context = ToyAnswerGenContextLLMAgent()
        self._step_count = 0
        self._gold_answer: str = ""
        self.state: ToyState | None = None

    # ------------------------- core API -------------------------

    def reset(self, question: str, gold_answer: str) -> ToyState:
        self._step_count = 0
        self._gold_answer = gold_answer
        self.state = ToyState(
            question=question,
            history=[],
            docs=[],
            answer=None,
            verification_score=0.0,
        )
        return self.state

    def step(
        self,
        macro_action: MacroAction,
        low_level_choice: int | None = None,
    ) -> Tuple[ToyState, float, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()"
        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count}

        # Execute macro-action
        if macro_action == MacroAction.RETRIEVE:
            self.state = self.retriever(self.state)
        elif macro_action == MacroAction.REASON:
            self.state = self.reasoner(self.state)
        elif macro_action == MacroAction.VERIFY:
            self.state = self.verifier(self.state)
        elif macro_action == MacroAction.ANSWER:
            # low_level_choice selects answer mode: 0=RAG, 1=Graph RAG, 2=Context LLM
            if low_level_choice == 1:
                self.state = self.answer_graph(self.state)
                info["answer_mode"] = "graph_rag"
            elif low_level_choice == 2:
                self.state = self.answer_context(self.state)
                info["answer_mode"] = "context_llm"
            else:
                self.state = self.answer_rag(self.state)
                info["answer_mode"] = "rag"
        elif macro_action == MacroAction.TERMINATE:
            pass

        done = macro_action == MacroAction.TERMINATE or self._step_count >= self.config.max_steps

        # Reward: small step penalty every step; on termination, add answer quality.
        reward = -self.config.lambda_step
        if done and self.state.answer:
            qa_reward = f1_score(self.state.answer, self._gold_answer)
            reward += qa_reward
            info["qa_reward"] = qa_reward

        return self.state, reward, done, info

    # -------------------- feature encoding ----------------------

    def encode_state(self) -> List[float]:
        """Encode current state into a small numeric feature vector.

        This is used by the toy PPO trainer and is not meant to
        capture all information, just a few coarse statistics.
        """

        assert self.state is not None
        s = self.state
        q_len = float(len(s.question.split()))
        num_docs = float(len(s.docs))
        ans_len = float(len(s.answer.split())) if s.answer else 0.0
        verif = float(s.verification_score)
        step = float(self._step_count)
        return [q_len, num_docs, ans_len, verif, step]
