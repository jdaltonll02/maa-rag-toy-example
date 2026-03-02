from __future__ import annotations

import json
from pathlib import Path

from .workflow import MAARagToyEnv, EnvConfig, MacroAction
from .planner import HierarchicalPlanner
from .data import iter_questions_and_answers


def main() -> None:
    env = MAARagToyEnv(EnvConfig())
    planner = HierarchicalPlanner()

    # Prepare output directory and JSON results file path
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "maa_rag_toy_inference_results.json"

    all_results = []

    for idx, (q, gold) in enumerate(iter_questions_and_answers()):
        print("=" * 60)
        print(f"Example {idx+1}: {q}")
        state = env.reset(q, gold)
        done = False
        steps = []
        while not done:
            s_vec = env.encode_state()
            macro_idx, low_idx, _, _ = planner.select_actions(s_vec)
            macro = MacroAction(macro_idx)
            state, reward, done, info = env.step(macro, low_idx)
            steps.append(
                {
                    "step": info.get("step"),
                    "macro_action": macro.name,
                    "low_level_choice": low_idx,
                    "reward": reward,
                    "info": info,
                }
            )
            print(f"  Action: {macro.name}, low={low_idx}, reward={reward:.3f}")
            if done:
                print("  Final answer:")
                print("  ", state.answer)
                print("  Verification score:", state.verification_score)
                print("  Info:", info)
                print("  History:")
                for h in state.history:
                    print("   -", h)
                all_results.append(
                    {
                        "example_index": idx,
                        "question": q,
                        "gold_answer": gold,
                        "final_answer": state.answer,
                        "verification_score": state.verification_score,
                        "steps": steps,
                        "history": list(state.history),
                    }
                )
                break

    # Save aggregated results to JSON for offline analysis
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
