# Findings: Continuous Latent Recurrence vs. Discrete Token-Space Reasoning

## 1. Executive Summary

This document evaluates two paradigms for model test-time reasoning:
1. **Discrete Token-Space Reasoning**: Natural language generation inside scratchpads (`<think> ... </think>`) scaled via reinforcement learning (e.g., DeepSeek-R1, OpenAI o1/o3, Gemini Thinking).
2. **Continuous Latent Recurrence**: State-space recurrence over continuous vectors ($\mathbf{z}, \mathbf{y} \in \mathbb{R}^d$) directly within transformer layers (Tiny Recursive Gemma / Samsung TRM).

---

## 2. Structural Comparison

| Dimension | Discrete Token-Space Thinking | Continuous Latent Recurrence (TRM) |
| :--- | :--- | :--- |
| **Medium** | Text tokens ($\mathcal{V}^*$) | Latent vectors ($\mathbf{z}, \mathbf{y} \in \mathbb{R}^{1 \times d}$) |
| **Emission** | 500–5,000+ thought tokens | 0 thought tokens; solution only |
| **Memory Bound** | $\mathcal{O}(T)$ to $\mathcal{O}(T^2)$ KV-cache growth | Strict $\mathcal{O}(1)$ via detached graph unrolling |
| **Execution Bottleneck** | Memory-bandwidth bound (sequential decoding) | Compute-bound (parallel matrix multiplications) |
| **State Separation** | Unified token sequence (reasoning & solution mixed) | Dual-latent: Reasoning $\mathbf{z}$ ($n$ passes), Solution $\mathbf{y}$ (1 pass) |
| **Dynamic Compute** | End-of-thought token (`</think>`) | ACT classifier head ($h_t \ge \tau$) + cosine drift |
| **Optimization** | Large-scale RL (PPO/GRPO) + rule verifiers | Multi-step deep supervision ($w_t = t^\gamma / \sum j^\gamma$) + LoRA |
| **Interpretability** | Transparent (human-readable reasoning trace) | Opaque (dense vector trajectories) |

---

## 3. Objective Assessment: Discrete Token-Space Thinking

### Strengths
- **Empirical Scalability**: Proven power-law scaling with test-time compute via RL across math, code, and formal logic.
- **Full Transparency**: Step-by-step reasoning traces allow runtime safety checks, auditing, and causal debugging.
- **Expressive Generalization**: Leverages entire natural language vocabulary for complex abstractions, backtracking, and counterfactuals.
- **Error Recovery via Search**: Can verbalize mistakes ("Wait, that violates constraint X...") and re-route plans explicitly.

### Weaknesses & Failure Modes
- **Memory Bandwidth Bottleneck**: Autoregressive decoding requires loading full model weights per token generated, capping serving throughput.
- **KV-Cache Bloat**: Long reasoning traces consume quadratic memory and reduce effective context window for inputs.
- **Serving Cost & Latency**: Generates 3x to 10x more tokens than final answers, multiplying compute cost and latency.
- **Commitment Trap**: Sampling a catastrophic token forces the model to either compound hallucinations or burn extensive tokens backtracking.

---

## 4. Objective Assessment: Continuous Latent Recurrence

### Strengths
- **Serving Efficiency**: Emits ~2.5x–3.4x fewer tokens with ~60% lower latency on standardized benchmarks.
- **Strict $\mathcal{O}(1)$ Memory Scaling**: Detached unrolling (`stop_gradient`) limits backpropagation and KV footprint to depth 1 (94 MB peak RSS).
- **Compute Density**: Shifts work from memory-bound token decoding into tensor-core-dense matrix multiplications.
- **Continuous State Fluidity**: Latents adjust continuously without discrete sampling commitments or token discretization noise.
- **Edge Feasibility**: Enables 2B–4B parameter models to execute multi-step logic on constrained hardware without OOM panics.

### Weaknesses & Failure Modes
- **Zero Interpretability**: Reasoning happens in latent space; intermediate states cannot be read, audited, or steered by users.
- **Pretrained Manifold Drift**: Raw vector injection causes vocabulary drift in pretrained LLMs unless stabilized by RMSNorm and targeted LoRA.
- **Optimization Sensitivity**: Requires joint tuning of recurrent depth ($T$), reasoning steps ($n$), power-decay exponent ($\gamma$), and halting threshold ($\tau$).
- **RL Integration Gap**: Harder to train with arbitrary black-box verifiers compared to token rollouts in RL environments.

---

## 5. Conclusion & Deployment Fit

- **Use Discrete Thinking When**: Tasks require explainability, verifiable proof chains, interactive clarification, or open-domain creative synthesis where compute cost is secondary.
- **Use Continuous Recurrence When**: Tasks target edge deployment, strict latency ceilings, cost-sensitive serving, or closed-domain tasks (algorithmic execution, deterministic code generation, puzzles).
