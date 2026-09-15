# Gemini & Agent Directives

## 1. Always Check For & Use Latest Gemini Models (Gemini 3.8+)
- **CRITICAL DIRECTIVE**: The agent must **ALWAYS check for and use the latest available Gemini model generation** (currently **Gemini 3.8**, including `gemini-3.8-flash` and `gemini-3.8-pro`).
- **PROHIBITION**: Never use or downgrade to legacy versions (such as `gemini-2.5-flash`, `gemini-2.0`, or `gemini-1.5`).
- **SCOPE**: Applies across all APIs, Next.js routes (`/api/infer`, `/api/judge`, `/api/cloud`), synthetic data generators, benchmark judges, and documentation.

## 2. Always Use Gemma 4 Models for Recursive Reasoning
- **CRITICAL DIRECTIVE**: For all continuous latent reasoning, LoRA adapters, evaluation suites, and training jobs, the agent must **ALWAYS use Gemma 4 models** (defaulting to `google/gemma-4-E2B-it-qat-q4_0-unquantized` or Gemma 4 variants).
- **PROHIBITION**: Never downgrade or switch to Gemma 2.

## 3. Strict Apple Silicon Hardware Guardrails
- **CRITICAL DIRECTIVE**: Never execute heavy model training, large weight loading, or unrolled autoregressive loops locally on Apple Silicon to prevent macOS kernel memory watchdog reboots.
- **LOCAL WORKFLOW**: Local runs are strictly restricted to fast isolated unit tests (`uv run pytest`, ~3.0s, mock tensors only for fast isolated syntax/interface testing).
- **CLOUD WORKFLOW**: All PyTorch training and heavy recurrence backpropagation must run on Google Cloud Vertex AI Custom Training on dedicated NVIDIA L4 (24GB VRAM) GPUs (`davenport-boutique`, `us-central1`).

## 4. Strict "Never Mock" Directive (Real Execution Only)
- **CRITICAL DIRECTIVE**: Never mock, simulate, or hardcode fake data, synthetic job progress, fake trajectories, or dummy fallbacks across APIs, routes, evaluations, or benchmarks.
- **PROHIBITION**:
  - Never set `continuousCode = cotCode` or duplicate results across distinct model paradigms.
  - Never use `Math.random()`, fake progress loops (`progress += 30`), or synthetic loss curves to simulate training/benchmarks.
  - Never use silent mock fallbacks (such as `simulateInference()` or `synthesizeJudgeScores()`) when an API call or backend service fails.
- **ERROR HANDLING**: If an API, cloud runner, or service call fails (e.g., missing credentials, network timeout, service error), always propagate and surface the true underlying error directly so it can be diagnosed and fixed.
- **SCOPE**: Applies across all Next.js API routes (`/api/infer`, `/api/cloud`, `/api/judge`, `/api/benchmarks`), web components, Python evaluation scripts, and cloud orchestration.

## 5. Strict Evaluation & Benchmark Integrity (Anti-Template / Anti-Cheating)
- **CRITICAL DIRECTIVE**: Never wrap `canonical_solution` or ground-truth task answers in candidate template strings (such as `f"{prompt}\n{canonical}"` or simulated thought tags) to fake evaluation pass rates.
- **PROHIBITION**:
  - Never synthesize ACT halting probabilities $h_t$ using index modulo formulas (e.g., `0.55 + 0.35 * (idx % 5)`).
  - All candidate solutions evaluated in `eval/judge_evaluator.py`, `scripts/evaluate.py`, and `scripts/run_cloud_benchmark_sweep.py` must originate from true model rollouts (greedy or sampled generation from the model pipeline).
  - If a model is not loaded or weights are unavailable, mark the benchmark run explicitly as unexecuted or surface the exact runtime error—never populate benchmark matrices with synthetic 100% pass rates.

## 6. Causal Mask & Latent Conditioning Invariant
- **CRITICAL DIRECTIVE**: In autoregressive causal transformers (Gemma 4), continuous latent embeddings $(z, y)$ must **ALWAYS be placed prior to the target tokens** (`[prompt_embeds, z_inj, y_inj, target_embeds]`), or provided through a custom attention mask.
- **PROHIBITION**: Never append latents after target tokens (`[inputs_embeds, y, z]`). Because standard causal attention is lower-triangular, tokens before $(z, y)$ cannot attend to them, resulting in zero gradient $\frac{\partial \mathcal{L}_{\text{target}}}{\partial z} = 0$ and completely disconnecting the model from latent thoughts.

## 7. Top-K Layer Recycling for Latent Recurrence
- **CRITICAL DIRECTIVE**: To prevent quadratic compute explosion and Unified Memory exhaustion, continuous recurrence must prioritize **Top-K Layer Recycling** (recursing through top 2 layers, e.g. Layers 16 & 17, with base representations cached) rather than unrolling all 18 layers of Gemma 4 for every recurrent iteration.
- **PROHIBITION**: Never execute full 18-layer multi-step recurrence loops ($T \times n > 2$) locally on Apple Silicon.

## 8. RoPE (Rotary Position Embedding) Decoupling Invariant
- **CRITICAL DIRECTIVE**: When injecting continuous latent vectors into the sequence, rotary position IDs must be decoupled or pinned so that target token positions do not suffer from index drift or phase distortion relative to pretrained Gemma 4 weights.

