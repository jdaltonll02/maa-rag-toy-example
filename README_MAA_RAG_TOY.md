# Toy MAA-RAG (Hierarchical Semi-MDP Adaptive RAG)

This directory contains a **toy implementation** of the proposed
Multi-Agent Adaptive RAG (MAA-RAG) method, built as a small, self-
contained extension to the MAO-ARAG baseline implemented in this
repository.

The goal is **illustration**, not performance: the code uses a tiny
in-memory corpus, simple lexical rewards, and small neural policies to
demonstrate the ideas in the paper (hierarchical Semi-MDP, adaptive
pipelines, reasoning and verification) in a way that is easy to
inspect and, later, port into the main system.

## 1. System Overview

- **State**  \(s_t = (q, H_t, D_t, A_t, C_t)\):
  - `q`: question string.
  - `H_t`: textual interaction history.
  - `D_t`: retrieved documents.
  - `A_t`: partial or final answer.
  - `C_t`: verification score.
- **Macro-actions**  \(Z = {\text{retrieve}, \text{reason}, \text{verify}, \text{answer}, \text{terminate}}\):
  - Implemented as `MacroAction` enum in
    [agentic_rag_improved/maa_rag_toy/workflow.py](agentic_rag_improved/maa_rag_toy/workflow.py).
  - `ANSWER` macro-action has low-level choices among `{RAG, Graph-RAG, Context-LLM}`.
- **Answer modes**:
  - `RAG`: use retrieved documents.
  - `Graph-RAG`: use a tiny knowledge graph view.
  - `Context-LLM`: rely mainly on the question and model prior.

This matches the hierarchical action space and options view in the
description: high-level chooses a macro-option, low-level (for
`ANSWER`) selects a primitive answer-generation mode.

## 2. Components and Code Structure

### 2.1 Data and Metrics

- [agentic_rag_improved/maa_rag_toy/data.py](agentic_rag_improved/maa_rag_toy/data.py)
  - `QAExample`: small dataclass for `(question, answer)`.
  - `get_dataset(...)` and `iter_questions_and_answers(...)`:
    - Always include a tiny synthetic toy QA set.
    - Optionally augment with **baseline QA benchmarks** via the
      Hugging Face `datasets` library:
      - **Single-hop**: NQ-Open, PopQA, AmbigQA.
      - **Multi-hop**: HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle.
    - A `mode` flag selects between `"single-hop"`,
      `"multi-hop"`, or `"mixed"` (both), and
      `max_per_dataset` controls how many examples are sampled
      from each external dataset (default used in the scripts is 50).
  - `f1_score` and `exact_match` implement token-level F1 and EM,
    used for reward shaping.
  - `make_claims(answer)` decomposes an answer into coarse clauses
    (used by the verification agent as a proxy for claim extraction).

### 2.2 Toy Retriever and Knowledge Graph

- [agentic_rag_improved/maa_rag_toy/retriever.py](agentic_rag_improved/maa_rag_toy/retriever.py)
  - `_CORPUS`: a few `ToyDocument` entries (e.g., heart disease,
    healthy diet, graph databases, LLMs).
  - `_KG`: tiny hand-crafted knowledge graph for Graph-RAG-style
    reasoning.
  - `_simple_score(query, text)`: lexical overlap scorer (no external
    dependencies).
  - `retrieve_docs(query, k)`: top-k lexical retriever (RAG backend).
  - `retrieve_graph_context(query)`: returns subgraph nodes and edges
    relevant to the query (Graph-RAG backend).

### 2.3 Executor Agents

- [agentic_rag_improved/maa_rag_toy/agents.py](agentic_rag_improved/maa_rag_toy/agents.py)
  - `ToyState` mirrors the theoretical state tuple, storing question,
    history, docs, answer, and verification score.
  - `ToyRetrieverAgent`: fills `docs` using `retrieve_docs`.
  - `ToyReasoningAgent`: builds a textual “self-graph reasoning”
    summary from retrieved docs, appending it to `history`.
  - `ToyVerificationAgent`: approximates NLI-based verification by
    scoring overlap between answer claims and retrieved docs; updates
    `verification_score`.
  - `ToyAnswerGenRAGAgent`: generates answers from retrieved docs
    (RAG mode).
  - `ToyAnswerGenGraphRAGAgent`: uses the toy graph context
    (Graph-RAG mode).
  - `ToyAnswerGenContextLLMAgent`: answers from question/context only
    (context-LLM mode).
  - All answer agents share `ToyAnswerGenBase`, which optionally uses a
    small HF causal LM (`gpt2` by default); if `transformers`/PyTorch
    are missing, they fall back to deterministic string behavior so the
    pipeline remains runnable.

### 2.4 HSMDP Environment

- [agentic_rag_improved/maa_rag_toy/workflow.py](agentic_rag_improved/maa_rag_toy/workflow.py)
  - `MacroAction` enum defines `{RETRIEVE, REASON, VERIFY, ANSWER, TERMINATE}`.
  - `EnvConfig` configures max steps and step penalty.
  - `MAARagToyEnv`:
    - `reset(question, gold_answer)` initializes `ToyState` and
      internal counters.
    - `step(macro_action, low_level_choice)` executes the selected
      macro-action via the appropriate agent; if `macro_action` is
      `ANSWER`, the low-level choice selects `{RAG, Graph-RAG,
      Context-LLM}`.
    - `encode_state()` returns a small feature vector for RL:
      `[question_length, num_docs, answer_length, verification_score, step]`.
  - **Reward**: at each step, a small negative step cost; on
    termination (either explicit `TERMINATE` or max steps), an F1-based
    reward between `state.answer` and the gold answer from `data.py` is
    added. This corresponds to the paper’s formulation with an
    answer-quality term plus cost/latency penalty.

### 2.5 Hierarchical Planner Networks

- [agentic_rag_improved/maa_rag_toy/planner.py](agentic_rag_improved/maa_rag_toy/planner.py)
  - `PlannerConfig`: dims for state, hidden layer, macro-actions, and
    answer modes.
  - `HighLevelPolicyNet`: MLP producing logits over macro-actions
    \(\pi_H(z_t \mid s_t)\).
  - `LowLevelPolicyNet`: MLP producing logits over answer modes when
    `z_t = ANSWER` (\(\pi_L(a_t \mid s_t, z_t)\)).
  - `ValueNet`: state-value estimator \(V(s_t)\).
  - `HierarchicalPlanner`:
    - `select_actions(state_vec)` samples a macro-action index, and
      when needed, a low-level answer-mode index; returns log-probs for
      PPO.
    - `evaluate_state(state_vec)` computes the baseline value.

## 3. Training: Toy Hierarchical PPO

> The main system uses PPO (via `verl`) to train an agentic planner.
> This toy example includes a minimal hierarchical PPO loop so that
> training behaviour can be prototyped and debugged in a small
> environment before being integrated with the full MAO-ARAG stack.

- [agentic_rag_improved/maa_rag_toy/rl_hppo.py](agentic_rag_improved/maa_rag_toy/rl_hppo.py)
  - `PPOConfig`: hyperparameters (\(\gamma, \lambda, \epsilon,\) learning
    rate, etc.).
  - `collect_trajectories(env, planner, cfg, use_external, max_per_dataset, mode)`:
    - Loops over either the tiny toy QA set alone, or the toy set
      **plus** samples from the above single-hop / multi-hop QA
      benchmarks, depending on the flags.
    - For each example, repeatedly:
      - Encodes state via `env.encode_state()`.
      - Samples macro and low-level actions from `HierarchicalPlanner`.
      - Steps the environment and records transitions
        `(s_t, z_t, a_t, r_t, done, logp_H, logp_L, V(s_t))`.
  - `compute_gae(...)`: computes advantages and returns with
    Generalized Advantage Estimation (GAE), following the PPO
    formulation.
  - `train(num_epochs=5, device="cpu", use_external=False, max_per_dataset=200, mode="single-hop")`:
    - Initializes `MAARagToyEnv` and `HierarchicalPlanner`.
    - For each epoch:
      - Collects transitions from the chosen QA mixture.
      - Optimizes high-level policy, low-level policy (only on
        `ANSWER` steps), and value network using PPO’s clipped objective.
      - Prints average return per epoch.

This provides a complete, albeit tiny, instantiation of hierarchical
PPO for adaptive RAG workflows.

## 4. Running the Toy Inference Demo

From the repository root (where `pyproject.toml` lives), run:

```bash
python -m agentic_rag_improved.maa_rag_toy.run_toy_inference
```

By default, the script is configured to **use_external=True** and
`mode="mixed"`, so it will:

- Iterate over the toy QA examples **plus** sampled questions from
  the single-hop and multi-hop benchmarks listed above.
- For each question, sample macro-actions and answer modes from the
  hierarchical planner.
- Print the chosen actions, final answer, verification score, and
  reasoning history.

## 5. Running the Toy Hierarchical PPO Demo

To run PPO training (CPU-only, toy scale, but now over a richer
mixture of QA tasks):

```bash
python -m agentic_rag_improved.maa_rag_toy.run_toy_rl
```

This will:

- Construct `MAARagToyEnv` and `HierarchicalPlanner`.
- Collect trajectories across the configured QA mixture (toy +
  selected external datasets, depending on `use_external` and
  `mode` inside `run_toy_rl.py`).
- Optimize separate high-level and low-level policies and a value
  function using PPO-style clipped objectives.
- Print average return per epoch and log metrics to
  `results/maa_rag_toy_rl_metrics.json`.

> Note: This example requires PyTorch. The answer-generating
> agents optionally use `transformers` and a small local HF model
> (default `gpt2`). If `transformers` or the model are not
> available, they fall back to deterministic string-based behavior,
> which is sufficient for structural testing.
>
> When using the real QA benchmarks, the toy code also relies on
> the Hugging Face `datasets` library to load NQ-Open, PopQA,
> AmbigQA, HotpotQA, 2WikiMultiHopQA, Musique, and Bamboogle.

## 6. Relation to MAO-ARAG Code

This toy code does **not** modify the existing MAO-ARAG implementation.
Instead, it mirrors its multi-agent orchestration concepts in a tiny,
readable example:

- MAO-ARAG planner and workflows:
  - [qa_manager/PlanningAgent.py](qa_manager/PlanningAgent.py)
  - [qa_manager/BaseAgent4.py](qa_manager/BaseAgent4.py)
  - [qa_manager/qa.py](qa_manager/qa.py)
- MAO-ARAG PPO trainer:
  - [verl/trainer/ppo/ray_trainer_agentic_rag_2.py](verl/trainer/ppo/ray_trainer_agentic_rag_2.py)

The toy MAA-RAG emphasizes:

- Fewer, more capable agents with a larger action space (answer-mode
  choice, reasoning, verification) instead of many micro-agents.
- Hierarchical control: high-level macro-action selection and low-level
  answer-mode selection.
- Reasoning and verification integrated into the pipeline and reward.
