from __future__ import annotations

from .rl_hppo import train


def main() -> None:
    # Increase the number of training epochs for PPO.
    # Adjust this value to control total training time.
    # use_external=True enables loading baseline datasets via `datasets`.
    # mode can be "single-hop", "multi-hop", or "mixed".
    train(num_epochs=100, use_external=True, max_per_dataset=50, mode="mixed")


if __name__ == "__main__":
    main()
