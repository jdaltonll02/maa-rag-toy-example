from __future__ import annotations

from .workflow import MAARagToyEnv, EnvConfig, MacroAction
from .planner import HierarchicalPlanner
from .data import iter_questions_and_answers


def main() -> None:
    env = MAARagToyEnv(EnvConfig())
    planner = HierarchicalPlanner()

    for idx, (q, gold) in enumerate(iter_questions_and_answers()):
        print("=" * 60)
        print(f"Example {idx+1}: {q}")
        state = env.reset(q, gold)
        done = False
        while not done:
            s_vec = env.encode_state()
            macro_idx, low_idx, _, _ = planner.select_actions(s_vec)
            macro = MacroAction(macro_idx)
            state, reward, done, info = env.step(macro, low_idx)
            print(f"  Action: {macro.name}, low={low_idx}, reward={reward:.3f}")
            if done:
                print("  Final answer:")
                print("  ", state.answer)
                print("  Verification score:", state.verification_score)
                print("  Info:", info)
                print("  History:")
                for h in state.history:
                    print("   -", h)
                break


if __name__ == "__main__":
    main()
