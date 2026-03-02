# MAA-RAG Toy: Setup, Training, and Testing

This document describes how to set up the environment, train the hierarchical planner, and run/testing the toy MAA-RAG example.

## 1. Setup

1. **Clone and enter the repository** (if not already):

   ```bash
   git clone <your-fork-or-repo-url>
   cd agentic_rag_improved
   ```

2. **Create a virtual environment** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:

   - PowerShell / CMD (Windows):

     ```bash
     .\.venv\Scripts\activate
     ```

   - Git Bash (Windows):

     ```bash
     source .venv/Scripts/activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

This installs the minimal requirements for the toy example:

- `torch` – for the planner networks and PPO.
- `transformers` – optional, for answer-generation agents using a
  small causal LM (e.g. `gpt2`).
- `datasets` – optional but recommended, to load real QA benchmarks
  (NQ-Open, PopQA, AmbigQA, HotpotQA, 2WikiMultiHopQA, Musique,
  Bamboogle) used in the extended toy experiments.

If `transformers` or model weights are not available, the answer
agents fall back to deterministic string behavior. If `datasets` is
missing, the code simply uses the tiny built-in toy QA set.

## 2. Training (Toy Hierarchical PPO)

The toy PPO training loop lives in `maa_rag_toy/rl_hppo.py` and is wrapped by `maa_rag_toy/run_toy_rl.py`.

From the repository root (with the virtual environment activated):

```bash
python -m maa_rag_toy.run_toy_rl
```

What this does (with the default settings in `run_toy_rl.py`):

- Builds the toy environment (`MAARagToyEnv`) and `HierarchicalPlanner`.
- Rolls out trajectories over a **mixture** of QA tasks:
  - The tiny synthetic toy set from `maa_rag_toy/data.py`.
  - Samples from single-hop benchmarks (NQ-Open, PopQA, AmbigQA).
  - Samples from multi-hop benchmarks (HotpotQA, Musique, and
    others where available), controlled via the `mode` and
    `max_per_dataset` arguments to `train`.
- Optimizes high-level and low-level policies and a value network
  using PPO.
- Prints per-epoch average return and number of steps.
- Writes per-epoch metrics to `results/maa_rag_toy_rl_metrics.json`.

You can open that JSON file to inspect fields like `epoch`, `avg_return`, `avg_reward`, and `num_steps`.

## 3. Testing / Inference

The main way to "test" the pipeline is to run the toy inference script, which exercises the hierarchical planner and agents on the synthetic QA data.

From the repository root (with the virtual environment activated):

```bash
python -m maa_rag_toy.run_toy_inference
```

What this does (with the default settings in `run_toy_inference.py`):

- Iterates over the toy QA examples from `maa_rag_toy/data.py` **plus**
  sampled questions from single-hop and multi-hop benchmarks, since
  the script calls `iter_questions_and_answers(use_external=True,
  mode="mixed", max_per_dataset=...)`.
- For each question, repeatedly:
  - Encodes the environment state.
  - Samples a macro-action and, when appropriate, an answer mode.
  - Steps the environment and accumulates rewards.
- Prints, for each example:
  - The sequence of macro-actions and rewards.
  - The final answer (if any) and verification score.
  - A textual history summarizing retrieval, reasoning, and verification.
- Writes a structured log of all trajectories and final answers to `results/maa_rag_toy_inference_results.json`.

## 4. Result Artifacts

After running the above commands, you should see:

- `results/maa_rag_toy_inference_results.json`
  - One entry per QA example with question, gold answer, final answer, verification score, per-step actions, and history.
- `results/maa_rag_toy_rl_metrics.json`
  - One entry per training epoch with average return, average reward, and number of steps.

These JSON files are intended for quick inspection and simple downstream analysis (e.g., plotting learning curves or examining action sequences) for this toy, in-memory example.