from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import json
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import Adam
except Exception:  # pragma: no cover
    torch = None
    F = None
    Adam = None

from .workflow import MAARagToyEnv, EnvConfig, MacroAction
from .planner import HierarchicalPlanner
from .data import get_dataset


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 10
    steps_per_epoch: int = 64

    @staticmethod
    def default() -> "PPOConfig":
        """Baseline configuration used in the paper-style toy setup."""

        return PPOConfig()

    @staticmethod
    def conservative() -> "PPOConfig":
        """Smaller updates, useful for stability checks."""

        return PPOConfig(
            clip_ratio=0.1,
            lr=1e-4,
            train_iters=5,
            steps_per_epoch=48,
        )

    @staticmethod
    def aggressive() -> "PPOConfig":
        """Larger, faster updates for quick experimentation."""

        return PPOConfig(
            clip_ratio=0.3,
            lr=5e-4,
            train_iters=15,
            steps_per_epoch=96,
        )

    @staticmethod
    def small_batch() -> "PPOConfig":
        """Tiny batch size to test variance / debugging runs."""

        return PPOConfig(
            steps_per_epoch=16,
            train_iters=8,
        )


@dataclass
class Transition:
    state: List[float]
    macro_idx: int
    low_idx: int | None
    reward: float
    done: bool
    logp_high: float
    logp_low: float
    value: float


def collect_trajectories(
    env: MAARagToyEnv,
    planner: HierarchicalPlanner,
    cfg: PPOConfig,
    use_external: bool = False,
    max_per_dataset: int = 200,
    mode: str = "single-hop",
) -> List[Transition]:
    """Roll out trajectories across all QA examples once per epoch."""

    dataset = get_dataset(
        use_external=use_external,
        max_per_dataset=max_per_dataset,
        mode=mode,
    )
    transitions: List[Transition] = []
    for qa in dataset:
        state = env.reset(qa.question, qa.answer)
        done = False
        while not done and len(transitions) < cfg.steps_per_epoch:
            s_vec = env.encode_state()
            macro_idx, low_idx, logp_h, logp_l = planner.select_actions(s_vec)
            value = planner.evaluate_state(s_vec)
            _, reward, done, _ = env.step(MacroAction(macro_idx), low_idx)
            transitions.append(
                Transition(
                    state=list(s_vec),
                    macro_idx=macro_idx,
                    low_idx=low_idx,
                    reward=reward,
                    done=done,
                    logp_high=logp_h,
                    logp_low=logp_l,
                    value=value,
                )
            )
            if done:
                break
    return transitions


def compute_gae(transitions: List[Transition], gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    assert torch is not None and F is not None
    rewards = [t.reward for t in transitions]
    values = [t.value for t in transitions] + [0.0]
    dones = [t.done for t in transitions]

    advs: List[float] = []
    gae = 0.0
    for t in reversed(range(len(transitions))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        advs.insert(0, gae)
    returns = [a + v for a, v in zip(advs, values[:-1])]

    adv_tensor = torch.tensor(advs, dtype=torch.float32)
    ret_tensor = torch.tensor(returns, dtype=torch.float32)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    return adv_tensor, ret_tensor


def train(
    num_epochs: int = 5,
    device: str = "cpu",
    use_external: bool = False,
    max_per_dataset: int = 200,
    mode: str = "single-hop",
) -> None:
    if torch is None or F is None or Adam is None:
        raise RuntimeError("PyTorch is required to run PPO training.")

    env = MAARagToyEnv(EnvConfig())
    planner = HierarchicalPlanner(device=device)
    optim_high = Adam(planner.parameters_high(), lr=PPOConfig.lr)
    optim_low = Adam(planner.parameters_low(), lr=PPOConfig.lr)
    optim_val = Adam(planner.parameters_value(), lr=PPOConfig.lr)

    # Choose a PPO configuration here. The default mirrors the
    # original toy settings; others explore different trade-offs.
    # ppo_cfg = PPOConfig.default()
    # ppo_cfg = PPOConfig.conservative()
    # ppo_cfg = PPOConfig.aggressive()
    # ppo_cfg = PPOConfig.small_batch()
    ppo_cfg = PPOConfig.default()

    # Prepare logging of per-epoch metrics
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "maa_rag_toy_rl_metrics.json"
    epoch_logs = []

    for epoch in range(num_epochs):
        transitions = collect_trajectories(
            env,
            planner,
            ppo_cfg,
            use_external=use_external,
            max_per_dataset=max_per_dataset,
            mode=mode,
        )
        if not transitions:
            print("No transitions collected; skipping epoch.")
            continue

        states = torch.tensor([t.state for t in transitions], dtype=torch.float32)
        macro_indices = torch.tensor([t.macro_idx for t in transitions], dtype=torch.long)
        low_indices = torch.tensor(
            [t.low_idx if t.low_idx is not None else -1 for t in transitions], dtype=torch.long
        )
        old_logp_high = torch.tensor([t.logp_high for t in transitions], dtype=torch.float32)
        old_logp_low = torch.tensor([t.logp_low for t in transitions], dtype=torch.float32)

        adv_tensor, ret_tensor = compute_gae(transitions, ppo_cfg.gamma, ppo_cfg.lam)

        for _ in range(ppo_cfg.train_iters):
            logits_high = planner.high(states)
            dist_high = torch.distributions.Categorical(logits=logits_high)
            logp_high = dist_high.log_prob(macro_indices)
            ratio_high = torch.exp(logp_high - old_logp_high)
            clipped_high = torch.clamp(ratio_high, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio)
            loss_pi_high = -(torch.min(ratio_high * adv_tensor, clipped_high * adv_tensor)).mean()

            # Low-level loss only for steps where macro == ANSWER
            mask_answer = (macro_indices == MacroAction.ANSWER.value)
            if mask_answer.any():
                logits_low = planner.low(states[mask_answer])
                dist_low = torch.distributions.Categorical(logits=logits_low)
                logp_low = dist_low.log_prob(low_indices[mask_answer])
                ratio_low = torch.exp(logp_low - old_logp_low[mask_answer])
                clipped_low = torch.clamp(ratio_low, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio)
                adv_low = adv_tensor[mask_answer]
                loss_pi_low = -(torch.min(ratio_low * adv_low, clipped_low * adv_low)).mean()
            else:
                loss_pi_low = torch.tensor(0.0)

            values_pred = planner.value(states)
            loss_v = F.mse_loss(values_pred, ret_tensor)

            optim_high.zero_grad()
            optim_low.zero_grad()
            optim_val.zero_grad()

            total_loss = loss_pi_high + loss_pi_low + 0.5 * loss_v
            total_loss.backward()

            optim_high.step()
            optim_low.step()
            optim_val.step()

        avg_return = float(ret_tensor.mean().item())
        avg_reward = float(sum(t.reward for t in transitions) / len(transitions))
        print(f"Epoch {epoch}: avg return {avg_return:.3f}, steps {len(transitions)}")

        epoch_logs.append(
            {
                "epoch": epoch,
                "avg_return": avg_return,
                "avg_reward": avg_reward,
                "num_steps": len(transitions),
            }
        )

    # Save metrics to JSON file for offline inspection
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(epoch_logs, f, indent=2, ensure_ascii=False)
