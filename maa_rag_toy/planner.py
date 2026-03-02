from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .workflow import MacroAction


@dataclass
class PlannerConfig:
    state_dim: int = 5
    hidden_dim: int = 32
    num_macro_actions: int = len(MacroAction)
    num_answer_modes: int = 3  # RAG, Graph-RAG, Context-LLM


class HighLevelPolicyNet(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.num_macro_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        h = torch.tanh(self.fc1(x))
        logits = self.fc2(h)
        return logits


class LowLevelPolicyNet(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.num_answer_modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        h = torch.tanh(self.fc1(x))
        logits = self.fc2(h)
        return logits


class ValueNet(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
        h = torch.tanh(self.fc1(x))
        value = self.fc2(h)
        return value.squeeze(-1)


class HierarchicalPlanner:
    """Hierarchical planner with separate high- and low-level policies.

    - High-level selects a macro-action z in {retrieve, reason, verify, answer, terminate}.
    - When z == ANSWER, the low-level policy selects the answer mode
      in {RAG, Graph-RAG, Context-LLM}.
    """

    def __init__(self, cfg: PlannerConfig | None = None, device: str = "cpu") -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for the hierarchical planner.")
        self.cfg = cfg or PlannerConfig()
        self.device = torch.device(device)
        self.high = HighLevelPolicyNet(self.cfg).to(self.device)
        self.low = LowLevelPolicyNet(self.cfg).to(self.device)
        self.value = ValueNet(self.cfg).to(self.device)

    def select_actions(
        self, state_vec: List[float]
    ) -> Tuple[int, int | None, float, float]:
        """Sample macro and (optional) low-level action, returning indices and log-probs."""

        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        logits_high = self.high(x)
        probs_high = torch.distributions.Categorical(logits=logits_high)
        macro_idx = int(probs_high.sample().item())
        logp_high = float(probs_high.log_prob(torch.tensor(macro_idx, device=self.device)).item())

        low_idx: int | None = None
        logp_low: float = 0.0
        if macro_idx == MacroAction.ANSWER.value:
            logits_low = self.low(x)
            probs_low = torch.distributions.Categorical(logits=logits_low)
            low_idx = int(probs_low.sample().item())
            logp_low = float(probs_low.log_prob(torch.tensor(low_idx, device=self.device)).item())

        return macro_idx, low_idx, logp_high, logp_low

    def evaluate_state(self, state_vec: List[float]) -> float:
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.value(x)
        return float(v.item())

    def parameters_high(self):  # type: ignore[no-untyped-def]
        return self.high.parameters()

    def parameters_low(self):  # type: ignore[no-untyped-def]
        return self.low.parameters()

    def parameters_value(self):  # type: ignore[no-untyped-def]
        return self.value.parameters()
