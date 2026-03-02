from __future__ import annotations

from .rl_hppo import train


def main() -> None:
    train(num_epochs=3)


if __name__ == "__main__":
    main()
