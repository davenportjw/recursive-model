# Agent Directives & Engineering Standards

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
