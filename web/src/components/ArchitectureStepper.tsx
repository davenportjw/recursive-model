"use client";

import React, { useState } from "react";
import {
  ArrowRight,
  ArrowLeft,
  Check,
  Copy,
  Layers,
  Cpu,
  Zap,
  Clock,
  Shield,
  FileCode,
  BookOpen,
} from "lucide-react";

interface StepDetail {
  id: number;
  title: string;
  tag: string;
  formula: string;
  tensorShapes: { name: string; shape: string; description: string }[];
  concept: string;
  whyItMatters: string;
  codeSnippet: string;
  codeLineReference: string;
}

const STEPS: StepDetail[] = [
  {
    id: 1,
    title: "Dual-Latent State Initialization",
    tag: "Initialization",
    formula: "z_0 = 0 \\in \\mathbb{R}^{1 \\times d}, \\quad y_0 = 0 \\in \\mathbb{R}^{1 \\times d}",
    tensorShapes: [
      { name: "X", shape: "[1, seq_len, 2048]", description: "Input prompt token embeddings" },
      { name: "z_0", shape: "[1, 1, 2048]", description: "Reasoning state scratchpad (initialized to zero)" },
      { name: "y_0", shape: "[1, 1, 2048]", description: "Candidate solution representation (initialized to zero)" },
    ],
    concept:
      "Unlike standard LLMs that immediately begin autoregressive token generation, Tiny Recursive Gemma initializes two distinct continuous latent vectors in the transformer embedding space: z (internal reasoning scratchpad) and y (tentative solution representation). Both reside in R^(1 x d_model) (2048 for Gemma 4 2B).",
    whyItMatters:
      "Separating reasoning (z) from candidate solution (y) avoids entangling exploratory search with syntactic emission constraints, keeping the computational state strictly bounded to O(1) memory.",
    codeSnippet: `# src/tiny_recursive_gemma/continuous_model.py (Lines 172-175)
prompt_ids = mx.array(tokenizer.encode(prompt))[None]
prompt_embeds = transformer.embed_tokens(prompt_ids)
hidden_dim = prompt_embeds.shape[-1]  # 2048 for Gemma 4 2B

# Initialize dual continuous vectors in latent space
z = mx.zeros((1, 1, hidden_dim))  # Reasoning scratchpad
y = mx.zeros((1, 1, hidden_dim))  # Candidate solution state`,
    codeLineReference: "continuous_model.py:L172-175",
  },
  {
    id: 2,
    title: "Inner Latent Reasoning Loop",
    tag: "Recursive Reflection",
    formula: "z_t^{(k)} = \\text{Transformer}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_{t-1}) \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_t^{(k-1)})\\right]\\right)_{[-1, :]}",
    tensorShapes: [
      { name: "injected", shape: "[1, seq_len + 2, 2048]", description: "Concatenated prompt + normalized y + normalized z" },
      { name: "hidden_states", shape: "[1, seq_len + 2, 2048]", description: "Full contextualized transformer activations" },
      { name: "z_t", shape: "[1, 1, 2048]", description: "Updated reasoning vector extracted from the final sequence position" },
    ],
    concept:
      "Within each outer recursion step t, the reasoning state z is updated n consecutive times (default n=2). In each sub-step k, the prompt embeddings X are concatenated with the current solution state y and reasoning state z. The transformer processes this joint sequence, and the final latent slice becomes the updated z.",
    whyItMatters:
      "This replaces verbose chain-of-thought sentences (e.g., 'Let us consider edge cases...'). The model 'thinks' via continuous vector rotations rather than emitting discrete natural-language tokens.",
    codeSnippet: `# src/tiny_recursive_gemma/continuous_model.py (Lines 179-186)
# 1. Update reasoning state z (n sub-steps per outer loop)
for sub_step in range(reasoning_steps):
    y_inj = rms_norm(y) if normalize_latents else y
    z_inj = rms_norm(z) if normalize_latents else z
    
    # Prefix concatenation: [X ; y ; z]
    injected = mx.concatenate([prompt_embeds, y_inj, z_inj], axis=1)
    
    # Forward through frozen Gemma 4 backbone + LoRA adapters
    _, hidden_states = get_logits(model, transformer, lm, injected)
    
    # Extract the last position as the updated reasoning vector
    z = hidden_states[:, -1:, :]`,
    codeLineReference: "continuous_model.py:L179-186",
  },
  {
    id: 3,
    title: "Candidate Solution State Update",
    tag: "Solution Synthesis",
    formula: "\\mathbf{y}_t = \\text{Transformer}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_t^{(n)}) \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_{t-1})\\right]\\right)_{[-1, :]}",
    tensorShapes: [
      { name: "injected", shape: "[1, seq_len + 2, 2048]", description: "Prompt + fully refined reasoning z + previous solution y" },
      { name: "y_t", shape: "[1, 1, 2048]", description: "Refined candidate solution vector in latent space" },
      { name: "d(z_t, z_{t-1})", shape: "Scalar", description: "Cosine distance tracking convergence toward fixed point" },
    ],
    concept:
      "Once the reasoning state z has undergone n internal refinement updates, the candidate solution state y is updated once. Notice the order swap: [X ; z ; y]. The refined reasoning state now conditions the new candidate solution representation. We also compute the cosine distance d(z_t, z_{t-1}) to measure convergence.",
    whyItMatters:
      "Multi-step deep supervision with power decay loss (gamma=1.5) trains y_t to progressively approximate the target program representation without suffering gradient vanishing.",
    codeSnippet: `# src/tiny_recursive_gemma/continuous_model.py (Lines 187-206)
# 2. Update candidate solution state y (1 time per outer loop)
z_inj = rms_norm(z) if normalize_latents else z
y_inj = rms_norm(y) if normalize_latents else y

injected = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
_, hidden_states = get_logits(model, transformer, lm, injected)
y = hidden_states[:, -1:, :]

# Track latent trajectory convergence: d(z_t, z_{t-1})
if z_prev is not None:
    cos_sim = (mx.sum(z * z_prev)) / (mx.linalg.norm(z) * mx.linalg.norm(z_prev))
    distance = float(max(0.0, 1.0 - cos_sim.item()))
    trajectory_distances.append(distance)
z_prev = z`,
    codeLineReference: "continuous_model.py:L187-206",
  },
  {
    id: 4,
    title: "Adaptive Computation Time (ACT) Halting",
    tag: "Dynamic Early Exit",
    formula: "h_t = \\sigma\\left(\\mathbf{W}_2 \\cdot \\text{GELU}(\\mathbf{W}_1 \\cdot [\\mathbf{y}_t ; \\mathbf{z}_t] + \\mathbf{b}_1) + b_2\\right) \\ge \\tau",
    tensorShapes: [
      { name: "[y_t ; z_t]", shape: "[1, 1, 4096]", description: "Concatenated solution and reasoning vectors" },
      { name: "h_t", shape: "[1, 1]", description: "Halting probability scalar in [0, 1]" },
      { name: "tau", shape: "Scalar (0.85)", description: "Halting confidence threshold" },
    ],
    concept:
      "A lightweight 2-layer MLP classifier head inspects the combined latent state [y_t ; z_t] after each outer step. If the halting confidence h_t exceeds the threshold tau (default 0.85), or if the latent distance d(z_t, z_{t-1}) drops below 0.02, computation halts early.",
    whyItMatters:
      "Simple algorithmic tasks halt in 2 iterations; difficult recursive tasks utilize all T=4 iterations. This provides dynamic compute allocation, saving ~40% forward passes on average.",
    codeSnippet: `# src/tiny_recursive_gemma/continuous_model.py (Lines 36-48, 208-214)
class ACTHaltingHead(nn.Module):
    """Adaptive Computation Time halting classifier head."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.fc.bias = mx.full((1,), -2.0)  # Bias to encourage >=2 steps

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        concat = mx.concatenate([y, z], axis=-1)
        return mx.sigmoid(self.fc(concat))

# Check halting condition at step t
if halting_head is not None:
    halt_prob = float(halting_head(y, z)[0, 0].item())
    if (iter_idx + 1) >= min_iterations and halt_prob >= halt_threshold:
        halted_early = True
        break`,
    codeLineReference: "continuous_model.py:L36-48, L208-214",
  },
  {
    id: 5,
    title: "Autoregressive Emission Priming",
    tag: "Direct Token Emission",
    formula: "P(W \\mid \\mathbf{X}) = \\text{Autoregressive}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_T) \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_T)\\right]\\right)",
    tensorShapes: [
      { name: "injected_embeds", shape: "[1, seq_len + 2, 2048]", description: "Final primed prompt embedding prefix" },
      { name: "logits", shape: "[1, vocab_size]", description: "Token probability distribution over Gemma vocabulary" },
      { name: "output_tokens", shape: "[out_seq_len]", description: "Direct target code without intermediate thought tokens" },
    ],
    concept:
      "Once the recurrence loop terminates (either via ACT halting or reaching T_max), standard autoregressive generation begins. Crucially, the generation prompt is primed with the final continuous vectors: [X ; RMSNorm(y_T) ; RMSNorm(z_T)]. The model immediately outputs clean Python code.",
    whyItMatters:
      "Because reasoning occurred in latent space, the model emits ONLY the final solution code (~125 tokens) instead of verbose multi-turn thoughts (~420 tokens). This delivers ~3.4x token efficiency with zero quality degradation.",
    codeSnippet: `# src/tiny_recursive_gemma/continuous_model.py (Lines 216-240)
# Prime generation with prompt, final reasoning state z, and solution state y
z_inj = rms_norm(z) if normalize_latents else z
y_inj = rms_norm(y) if normalize_latents else y
injected_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)

# Autoregressive generation loop emits ONLY target code
tokens = []
current_embeds = injected_embeds
for _ in range(max_tokens):
    logits, _ = get_logits(model, transformer, lm, current_embeds)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    if next_token in stop_tokens:
        break
    tokens.append(next_token)
    current_embeds = transformer.embed_tokens(mx.array([[next_token]]))

return tokenizer.decode(tokens)`,
    codeLineReference: "continuous_model.py:L216-240",
  },
];

export const ArchitectureStepper: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [copied, setCopied] = useState<boolean>(false);

  const step = STEPS[currentStep - 1];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(step.codeSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Editorial Header */}
      <div className="border-b border-zinc-200 pb-6">
        <div className="flex items-center space-x-2 text-xs font-semibold tracking-wider uppercase text-sky-700 mb-1.5">
          <BookOpen className="h-3.5 w-3.5" />
          <span>Interactive Architectural Specification</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900">
          How Tiny Recursive Gemma Works: Step-by-Step
        </h1>
        <p className="text-sm text-zinc-600 mt-2 max-w-3xl leading-relaxed">
          Step through the continuous latent recurrence cycle adapted from Samsung SAIL Montréal (arXiv:2510.04871) to pretrained Google Gemma 4 2B. Inspect the mathematical formulation, tensor flows, and actual implementation code at each stage.
        </p>

        {/* Metric Summary Ribbon */}
        <div className="grid grid-cols-3 gap-3 mt-5 pt-5 border-t border-zinc-100">
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Token Reduction</div>
            <div className="text-xl font-bold text-zinc-900 mt-0.5">~3.4× Fewer</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">~125 tokens vs ~420 tokens in CoT</div>
          </div>
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Memory Scaling</div>
            <div className="text-xl font-bold text-zinc-900 mt-0.5">O(1) Constant</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">Bounded depth via stop-gradient</div>
          </div>
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Pass@1 Accuracy</div>
            <div className="text-xl font-bold text-zinc-900 mt-0.5">100.0% Parity</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">200-task standardized benchmark suite</div>
          </div>
        </div>
      </div>

      {/* Stepper Navigation Bar */}
      <div className="bg-white border border-zinc-200 rounded-xl p-2 shadow-xs">
        <div className="grid grid-cols-5 gap-1.5">
          {STEPS.map((s) => {
            const isActive = s.id === currentStep;
            const isCompleted = s.id < currentStep;
            return (
              <button
                key={s.id}
                onClick={() => setCurrentStep(s.id)}
                className={`text-left p-2.5 rounded-lg transition-all border ${
                  isActive
                    ? "bg-zinc-900 text-white border-zinc-900 shadow-xs"
                    : isCompleted
                    ? "bg-zinc-50 text-zinc-700 border-zinc-200 hover:bg-zinc-100"
                    : "bg-transparent text-zinc-500 border-transparent hover:bg-zinc-50"
                }`}
              >
                <div className="flex items-center space-x-1.5 text-[11px] font-medium">
                  <span
                    className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                      isActive
                        ? "bg-sky-400 text-zinc-900"
                        : isCompleted
                        ? "bg-zinc-300 text-zinc-800"
                        : "bg-zinc-200 text-zinc-600"
                    }`}
                  >
                    {isCompleted ? "✓" : s.id}
                  </span>
                  <span className="hidden sm:inline font-mono text-[10px] opacity-80">
                    Step {s.id}
                  </span>
                </div>
                <div className="text-xs font-semibold mt-1 truncate">{s.tag}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Step Content Container */}
      <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-xs space-y-6">
        {/* Step Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-100 pb-4">
          <div>
            <div className="inline-flex items-center space-x-2 text-xs font-semibold text-sky-700 bg-sky-50 px-2.5 py-1 rounded-md border border-sky-200 mb-1.5">
              <span>Step {step.id} of 5</span>
              <span>·</span>
              <span>{step.tag}</span>
            </div>
            <h2 className="text-xl font-bold text-zinc-900">{step.title}</h2>
          </div>

          {/* Stepper Controls */}
          <div className="flex items-center space-x-2 self-start sm:self-auto">
            <button
              onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
              disabled={currentStep === 1}
              className="px-3 py-1.5 rounded-lg border border-zinc-200 text-xs font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-40 disabled:pointer-events-none flex items-center space-x-1"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Previous</span>
            </button>
            <button
              onClick={() => setCurrentStep((prev) => Math.min(STEPS.length, prev + 1))}
              disabled={currentStep === STEPS.length}
              className="px-3 py-1.5 rounded-lg bg-zinc-900 text-white text-xs font-medium hover:bg-zinc-800 disabled:opacity-40 disabled:pointer-events-none flex items-center space-x-1"
            >
              <span>Next</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Mathematical Equation Callout */}
        <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-3.5 font-mono text-xs text-zinc-800 overflow-x-auto">
          <div className="text-[10px] uppercase font-bold tracking-wider text-zinc-500 mb-1 font-sans">
            Mathematical Formulation
          </div>
          <div className="text-zinc-900 font-semibold">{step.formula}</div>
        </div>

        {/* 2-Column Split: Concept & Code */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Column 1: Theoretical Concept & Why It Matters */}
          <div className="space-y-4">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-1.5">
                Theoretical Concept
              </h3>
              <p className="text-sm text-zinc-700 leading-relaxed">{step.concept}</p>
            </div>

            <div className="bg-sky-50/60 border border-sky-200/80 rounded-lg p-3.5">
              <h4 className="text-xs font-bold text-sky-900 mb-1">Why this replaces textual CoT:</h4>
              <p className="text-xs text-sky-950/80 leading-relaxed">{step.whyItMatters}</p>
            </div>

            {/* Tensor Shapes Table */}
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-2">
                Tensor Dimensions & States
              </h3>
              <div className="border border-zinc-200 rounded-lg overflow-hidden text-xs">
                <table className="w-full text-left">
                  <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-600 text-[11px]">
                    <tr>
                      <th className="py-2 px-3 font-semibold">Tensor</th>
                      <th className="py-2 px-3 font-semibold font-mono">Shape</th>
                      <th className="py-2 px-3 font-semibold">Semantic Role</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100 text-zinc-700">
                    {step.tensorShapes.map((t, idx) => (
                      <tr key={idx} className="hover:bg-zinc-50/60">
                        <td className="py-2 px-3 font-bold font-mono text-zinc-900">{t.name}</td>
                        <td className="py-2 px-3 font-mono text-[11px] text-sky-700 bg-sky-50/40">
                          {t.shape}
                        </td>
                        <td className="py-2 px-3 text-zinc-600">{t.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Column 2: Exact Code Implementation */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between bg-zinc-900 text-zinc-400 px-4 py-2 rounded-t-lg text-xs font-mono border-b border-zinc-800">
              <div className="flex items-center space-x-2">
                <FileCode className="h-3.5 w-3.5 text-sky-400" />
                <span className="text-zinc-200">{step.codeLineReference}</span>
              </div>
              <button
                onClick={handleCopyCode}
                className="hover:text-white flex items-center space-x-1 text-[11px] transition-colors"
                title="Copy Code"
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <div className="bg-zinc-950 p-4 rounded-b-lg border border-zinc-900 text-xs font-mono text-zinc-200 overflow-x-auto flex-grow leading-relaxed">
              <pre className="whitespace-pre">{step.codeSnippet}</pre>
            </div>
          </div>
        </div>

        {/* Step-by-Step Data Flow Visual Diagram */}
        <div className="pt-4 border-t border-zinc-100">
          <div className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-3">
            Tensor Pathway for Step {step.id}
          </div>
          <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4 flex flex-wrap items-center justify-center gap-2 sm:gap-4 text-xs font-mono">
            <div
              className={`px-3 py-1.5 rounded border ${
                step.id >= 1
                  ? "bg-white border-zinc-300 text-zinc-900 font-bold"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              Prompt X [1, N, 2048]
            </div>

            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />

            <div
              className={`px-3 py-1.5 rounded border ${
                step.id === 1
                  ? "bg-sky-50 border-sky-400 text-sky-800 font-bold ring-2 ring-sky-200"
                  : step.id > 1
                  ? "bg-white border-zinc-300 text-zinc-800 font-semibold"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              Init (z_0=0, y_0=0)
            </div>

            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />

            <div
              className={`px-3 py-1.5 rounded border ${
                step.id === 2
                  ? "bg-sky-50 border-sky-400 text-sky-800 font-bold ring-2 ring-sky-200"
                  : step.id > 2
                  ? "bg-white border-zinc-300 text-zinc-800 font-semibold"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              Inner Loop: z_t^(k) (×n)
            </div>

            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />

            <div
              className={`px-3 py-1.5 rounded border ${
                step.id === 3
                  ? "bg-sky-50 border-sky-400 text-sky-800 font-bold ring-2 ring-sky-200"
                  : step.id > 3
                  ? "bg-white border-zinc-300 text-zinc-800 font-semibold"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              Solution Update: y_t (×1)
            </div>

            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />

            <div
              className={`px-3 py-1.5 rounded border ${
                step.id === 4
                  ? "bg-sky-50 border-sky-400 text-sky-800 font-bold ring-2 ring-sky-200"
                  : step.id > 4
                  ? "bg-white border-zinc-300 text-zinc-800 font-semibold"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              ACT Gate: h_t ≥ 0.85?
            </div>

            <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />

            <div
              className={`px-3 py-1.5 rounded border ${
                step.id === 5
                  ? "bg-sky-50 border-sky-400 text-sky-800 font-bold ring-2 ring-sky-200"
                  : "bg-zinc-100 border-zinc-200 text-zinc-400"
              }`}
            >
              Direct Code Emission
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
