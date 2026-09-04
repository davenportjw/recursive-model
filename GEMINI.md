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
- **LOCAL WORKFLOW**: Local runs are strictly restricted to fast isolated unit tests (`uv run pytest`, ~3.0s, mock tensors only).
- **CLOUD WORKFLOW**: All PyTorch training and heavy recurrence backpropagation must run on Google Cloud Vertex AI Custom Training on dedicated NVIDIA L4 (24GB VRAM) GPUs (`davenport-boutique`, `us-central1`).
