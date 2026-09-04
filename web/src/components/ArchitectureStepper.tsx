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
  Lightbulb,
  Sparkles,
  Brain,
  CheckCircle2,
  HelpCircle,
} from "lucide-react";

interface StepDetail {
  id: number;
  title: string;
  everydayTitle: string;
  tag: string;
  everydayTag: string;
  formula: string;
  everydayFormula: string;
  everydayFormulaNote: string;
  tensorShapes: { name: string; shape: string; description: string }[];
  concept: string;
  everydayExplanation: string;
  everydayAnalogy: {
    title: string;
    description: string;
  };
  whyItMatters: string;
  whyItMattersEveryday: string;
  mentalModel: {
    input: string;
    action: string;
    output: string;
  };
  codeSnippet: string;
  codeLineReference: string;
}

const STEPS: StepDetail[] = [
  {
    id: 1,
    title: "Dual-Latent State Initialization",
    everydayTitle: "Setting Up the Mental Workspace (Two Blank Notepads)",
    tag: "Initialization",
    everydayTag: "Getting Ready",
    formula: "z_0 = 0 \\in \\mathbb{R}^{1 \\times d}, \\quad y_0 = 0 \\in \\mathbb{R}^{1 \\times d}",
    everydayFormula: "Brain Scratchpad (z) = Blank  ·  Candidate Draft (y) = Blank",
    everydayFormulaNote: "Before speaking a single word out loud, the AI sets up two silent mental spaces in its memory: one for rough scratch notes, and one for shaping the final answer.",
    tensorShapes: [
      { name: "X", shape: "[1, seq_len, 2048]", description: "Input prompt token embeddings" },
      { name: "z_0", shape: "[1, 1, 2048]", description: "Reasoning state scratchpad (initialized to zero)" },
      { name: "y_0", shape: "[1, 1, 2048]", description: "Candidate solution representation (initialized to zero)" },
    ],
    concept:
      "Unlike standard LLMs that immediately begin autoregressive token generation, Tiny Recursive Gemma initializes two distinct continuous latent vectors in the transformer embedding space: z (internal reasoning scratchpad) and y (tentative solution representation). Both reside in R^(1 x d_model) (2048 for Gemma 4 2B).",
    everydayExplanation:
      "When you give a normal AI chatbot a tricky prompt, it starts blabbing out loud immediately on token #1, hoping it figures out the solution while talking. Tiny Recursive Gemma takes a fundamentally smarter approach: it pauses and sets up two silent mental workspaces inside its brain before speaking. One is a scratchpad (called z) for messy brainstorming, and the other is a drafting sheet (called y) for piecing together the final answer.",
    everydayAnalogy: {
      title: "The Clean Desk Analogy",
      description: "Imagine sitting down to do your taxes or solve a tricky riddle. Before writing on the official form, you clear two spots on your desk: one notepad for rough scratch math that no one will ever see, and one clean sheet for your final response.",
    },
    whyItMatters:
      "Separating reasoning (z) from candidate solution (y) avoids entangling exploratory search with syntactic emission constraints, keeping the computational state strictly bounded to O(1) memory.",
    whyItMattersEveryday:
      "No more awkward verbal rambling. Standard AI models make you wait and pay for 200 words of filler like 'Sure, let me think about that step by step...' By doing this prep silently in memory, the model stays focused, uses constant memory, and never clutters your screen.",
    mentalModel: {
      input: "Your question or prompt",
      action: "Allocates 2 blank silent continuous memory slots (scratchpad & draft)",
      output: "Ready to think without talking out loud",
    },
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
    everydayTitle: "Silent Thinking (Mulling It Over in Its Head)",
    tag: "Recursive Reflection",
    everydayTag: "Thinking Silently",
    formula: "z_t^{(k)} = \\text{Transformer}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_{t-1}) \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_t^{(k-1)})\\right]\\right)_{[-1, :]}",
    everydayFormula: "New Scratchpad Thoughts = Brain ( Your Prompt + Current Draft + Previous Ideas )",
    everydayFormulaNote: "The AI loops your question, its previous thoughts, and its current draft through its neural network to update its scratchpad—repeated multiple times in complete silence.",
    tensorShapes: [
      { name: "injected", shape: "[1, seq_len + 2, 2048]", description: "Concatenated prompt + normalized y + normalized z" },
      { name: "hidden_states", shape: "[1, seq_len + 2, 2048]", description: "Full contextualized transformer activations" },
      { name: "z_t", shape: "[1, 1, 2048]", description: "Updated reasoning vector extracted from the final sequence position" },
    ],
    concept:
      "Within each outer recursion step t, the reasoning state z is updated n consecutive times (default n=2). In each sub-step k, the prompt embeddings X are concatenated with the current solution state y and reasoning state z. The transformer processes this joint sequence, and the final latent slice becomes the updated z.",
    everydayExplanation:
      "This is where the model does its heavy mental lifting. Instead of typing out 50 lines of rambling inner thoughts (Chain-of-Thought) for you to read, the AI loops the ideas through its neural network in complete silence. It looks at your question, reviews what it has thought so far, and refines its scratchpad multiple times until its ideas are organized.",
    everydayAnalogy: {
      title: "Mulling It Over in Your Head",
      description: "When a friend asks you a hard question, you don't say every wandering thought out loud ('Wait, is Tuesday the 5th? No, Monday was the 4th, so Tuesday is...'). You figure it out in your head in two seconds, and only speak when you know the answer.",
    },
    whyItMatters:
      "This replaces verbose chain-of-thought sentences (e.g., 'Let us consider edge cases...'). The model 'thinks' via continuous vector rotations rather than emitting discrete natural-language tokens.",
    whyItMattersEveryday:
      "Massive time and cost savings. Silent thinking cuts down wait time by over 70%. It also keeps the AI from getting confused by its own rambling words, leading to much higher accuracy on tough logic puzzles.",
    mentalModel: {
      input: "Prompt + Current draft + Previous scratchpad notes",
      action: "Cycles through internal neural network layers 2 times silently",
      output: "Crisp, refined thoughts ready to update the answer draft",
    },
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
    everydayTitle: "Updating the Working Draft",
    tag: "Solution Synthesis",
    everydayTag: "Refining Draft",
    formula: "\\mathbf{y}_t = \\text{Transformer}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_t^{(n)}) \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_{t-1})\\right]\\right)_{[-1, :]}",
    everydayFormula: "Polished Draft = Brain ( Your Prompt + Polished Scratchpad Thoughts + Old Draft )",
    everydayFormulaNote: "The AI applies its brainstormed ideas to upgrade the candidate solution, and checks how much its thoughts are stabilizing.",
    tensorShapes: [
      { name: "injected", shape: "[1, seq_len + 2, 2048]", description: "Prompt + fully refined reasoning z + previous solution y" },
      { name: "y_t", shape: "[1, 1, 2048]", description: "Refined candidate solution vector in latent space" },
      { name: "d(z_t, z_{t-1})", shape: "Scalar", description: "Cosine distance tracking convergence toward fixed point" },
    ],
    concept:
      "Once the reasoning state z has undergone n internal refinement updates, the candidate solution state y is updated once. Notice the order swap: [X ; z ; y]. The refined reasoning state now conditions the new candidate solution representation. We also compute the cosine distance d(z_t, z_{t-1}) to measure convergence.",
    everydayExplanation:
      "Now that the scratchpad has been thoroughly brainstormed, the AI sits down to improve its rough draft. It combines your original prompt with its freshly polished thoughts to create an updated, high-quality solution draft. At the same time, it measures how much its thoughts shifted compared to the last turn—if they barely changed, it knows the solution has settled and is ready.",
    everydayAnalogy: {
      title: "Sculpting from Clay to Marble",
      description: "Think of a sculptor. Step 2 was studying the reference photo and planning the cuts in their head. Step 3 is making the actual chisel strokes on the statue. With every cycle, the statue looks clearer, sharper, and closer to perfection.",
    },
    whyItMatters:
      "Multi-step deep supervision with power decay loss (gamma=1.5) trains y_t to progressively approximate the target program representation without suffering gradient vanishing.",
    whyItMattersEveryday:
      "Self-correcting answers. Most AI models are 'one-shot': whatever word they guess first is set in stone. By revising its draft across multiple cycles before answering, Tiny Recursive Gemma catches and fixes its own mistakes before you ever see them.",
    mentalModel: {
      input: "Polished scratchpad ideas + Previous draft",
      action: "Synthesizes an updated draft and measures convergence stability",
      output: "A significantly higher quality solution draft",
    },
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
    everydayTitle: "The Gut Check (Knowing When to Stop Thinking)",
    tag: "Dynamic Early Exit",
    everydayTag: "Confidence Check",
    formula: "h_t = \\sigma\\left(\\mathbf{W}_2 \\cdot \\text{GELU}(\\mathbf{W}_1 \\cdot [\\mathbf{y}_t ; \\mathbf{z}_t] + \\mathbf{b}_1) + b_2\\right) \\ge \\tau",
    everydayFormula: "Confidence Meter >= 85%  ->  Stop Thinking Early & Deliver Answer",
    everydayFormulaNote: "A built-in confidence inspector tests whether the draft is solid. If yes, it stops thinking immediately; if not, it takes another thinking pass.",
    tensorShapes: [
      { name: "[y_t ; z_t]", shape: "[1, 1, 4096]", description: "Concatenated solution and reasoning vectors" },
      { name: "h_t", shape: "[1, 1]", description: "Halting probability scalar in [0, 1]" },
      { name: "tau", shape: "Scalar (0.85)", description: "Halting confidence threshold" },
    ],
    concept:
      "A lightweight 2-layer MLP classifier head inspects the combined latent state [y_t ; z_t] after each outer step. If the halting confidence h_t exceeds the threshold tau (default 0.85), or if the latent distance d(z_t, z_{t-1}) drops below 0.02, computation halts early.",
    everydayExplanation:
      "Not all questions are equally difficult. Asking 'What is 2 + 2?' doesn't require 10 minutes of deep contemplation, but a complex programming puzzle does. At this stage, a built-in 'confidence inspector' looks at the current draft and asks: 'Are we sure of this answer yet?' If confidence hits 85% or higher, it rings the bell and stops thinking immediately.",
    everydayAnalogy: {
      title: "The Smart Test-Taker",
      description: "An experienced student answering an easy question on an exam marks the answer in 2 seconds and moves forward. They don't waste 15 minutes staring at question #1. They save their time and energy for the hardest questions at the end of the test.",
    },
    whyItMatters:
      "Simple algorithmic tasks halt in 2 iterations; difficult recursive tasks utilize all T=4 iterations. This provides dynamic compute allocation, saving ~40% forward passes on average.",
    whyItMattersEveryday:
      "Lightning-fast simple answers without sacrificing deep thought on hard ones. You get instant replies on straightforward requests, and patient, thorough deliberation on hard problems—without wasting power, time, or battery.",
    mentalModel: {
      input: "Current draft + Brainstorm scratchpad state",
      action: "Evaluates certainty score (0% to 100%) against an 85% threshold",
      output: "Decides whether to think for another round or output the answer right now",
    },
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
    everydayTitle: "Writing Down the Final Answer (Straight to the Point)",
    tag: "Direct Token Emission",
    everydayTag: "Clean Answer",
    formula: "P(W \\mid \\mathbf{X}) = \\text{Autoregressive}\\left(\\left[\\mathbf{X} \\; ; \\; \\text{RMSNorm}(\\mathbf{y}_T) \\; ; \\; \\text{RMSNorm}(\\mathbf{z}_T)\\right]\\right)",
    everydayFormula: "Final Output = Model Output ( Primed with Polished Thoughts + Final Draft )",
    everydayFormulaNote: "The AI loads its finalized thoughts into its voice, typing out only the working solution with zero conversational fluff.",
    tensorShapes: [
      { name: "injected_embeds", shape: "[1, seq_len + 2, 2048]", description: "Final primed prompt embedding prefix" },
      { name: "logits", shape: "[1, vocab_size]", description: "Token probability distribution over Gemma vocabulary" },
      { name: "output_tokens", shape: "[out_seq_len]", description: "Direct target code without intermediate thought tokens" },
    ],
    concept:
      "Once the recurrence loop terminates (either via ACT halting or reaching T_max), standard autoregressive generation begins. Crucially, the generation prompt is primed with the final continuous vectors: [X ; RMSNorm(y_T) ; RMSNorm(z_T)]. The model immediately outputs clean Python code.",
    everydayExplanation:
      "Now that all the thinking and drafting is finished, the AI picks up the pen. Because all the trial-and-error, self-correction, and reasoning happened silently in its head, it doesn't need to write out its train of thought. It directly emits clean, working code with zero conversational fluff.",
    everydayAnalogy: {
      title: "The Executive Briefing",
      description: "When a senior strategist finishes a 3-week market analysis, they don't dump 500 pages of messy notes on the CEO's desk. They walk in and deliver a crisp, 1-page action plan with exactly what to do. 100% signal, 0% noise.",
    },
    whyItMatters:
      "Because reasoning occurred in latent space, the model emits ONLY the final solution code (~125 tokens) instead of verbose multi-turn thoughts (~420 tokens). This delivers ~3.4x token efficiency with zero quality degradation.",
    whyItMattersEveryday:
      "No wading through essays to find the solution. You get ~125 tokens of pure, executable code instead of ~420 tokens of conversational rambling. It runs faster, fits on your screen without scrolling, and cuts your API bill by over 70%.",
    mentalModel: {
      input: "The prompt primed with finalized silent thoughts and solution state",
      action: "Generates target code token-by-token with full certainty",
      output: "100% clean, ready-to-run Python code",
    },
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
  const [explanationMode, setExplanationMode] = useState<"everyday" | "researcher">("everyday");

  const step = STEPS[currentStep - 1];

  const handleCopyCode = () => {
    navigator.clipboard.writeText(step.codeSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Editorial Header */}
      <div className="border-b border-[#E8E5DF] pb-6">
        <div className="flex items-center space-x-2 text-[11px] font-mono tracking-wider uppercase text-[#C96442] mb-2">
          <BookOpen className="h-3.5 w-3.5" />
          <span>Architectural Specification · Deep Dive</span>
        </div>
        <h1 className="font-editorial text-3xl sm:text-4xl font-normal tracking-tight text-[#141413]">
          Continuous Latent Recurrence on Gemma 4
        </h1>
        <p className="text-sm text-[#57534E] mt-2 max-w-3xl leading-relaxed">
          Detailed step-by-step breakdown of the continuous latent recurrence cycle adapted from Samsung SAIL Montréal (arXiv:2510.04871) to pretrained Google Gemma 4 2B. Inspect each stage through an intuitive <strong>Everyday Explanation</strong> or a formal <strong>LLM Researcher</strong> mathematical specification.
        </p>

        {/* Audience Perspective Switcher */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 bg-[#FAF9F5] border border-[#E8E5DF] p-2.5 rounded-xl">
          <div className="flex items-center space-x-2 text-xs text-[#57534E]">
            <Sparkles className="h-4 w-4 text-[#C96442]" />
            <span className="font-medium">Audience Mode:</span>
            <span className="text-[#8C887B]">
              {explanationMode === "everyday"
                ? "Everyday Person (Plain English analogies, no PhD required)"
                : "LLM Researcher (Theoretical equations, tensor shapes & proofs)"}
            </span>
          </div>
          <div className="flex items-center bg-white border border-[#E8E5DF] p-0.5 rounded-lg shadow-xs">
            <button
              onClick={() => setExplanationMode("everyday")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center space-x-1.5 ${
                explanationMode === "everyday"
                  ? "bg-[#C96442] text-white shadow-xs font-semibold"
                  : "text-[#57534E] hover:text-[#141413] hover:bg-[#FAF9F5]"
              }`}
            >
              <Lightbulb className="h-3.5 w-3.5" />
              <span>Everyday Person</span>
            </button>
            <button
              onClick={() => setExplanationMode("researcher")}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center space-x-1.5 ${
                explanationMode === "researcher"
                  ? "bg-[#141413] text-white shadow-xs font-semibold"
                  : "text-[#57534E] hover:text-[#141413] hover:bg-[#FAF9F5]"
              }`}
            >
              <BookOpen className="h-3.5 w-3.5" />
              <span>LLM Researcher</span>
            </button>
          </div>
        </div>

        {/* Metric Summary Ribbon */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-6 pt-5 border-t border-[#E8E5DF]/60">
          <div className="bg-white border border-[#E8E5DF] rounded-xl p-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B]">Token Reduction</div>
            <div className="text-2xl font-editorial font-medium text-[#141413] mt-0.5">~3.4× vs CoT</div>
            <div className="text-[11px] text-[#57534E] mt-0.5">~125 tok (CL) vs ~420 tok (CoT) · 135 tok (Baseline)</div>
          </div>
          <div className="bg-white border border-[#E8E5DF] rounded-xl p-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B]">Reasoning Quality</div>
            <div className="text-2xl font-editorial font-medium text-[#C96442] mt-0.5">8.64 / 10</div>
            <div className="text-[11px] text-[#57534E] mt-0.5">+25% over baseline (6.90) via Gemini 3.8 Judge</div>
          </div>
          <div className="bg-white border border-[#E8E5DF] rounded-xl p-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B]">Pass@1 Accuracy</div>
            <div className="text-2xl font-editorial font-medium text-[#141413] mt-0.5">100.0% Parity</div>
            <div className="text-[11px] text-[#57534E] mt-0.5">Preserves full multi-turn discrete CoT reasoning</div>
          </div>
          <div className="bg-white border border-[#E8E5DF] rounded-xl p-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B]">Memory Complexity</div>
            <div className="text-2xl font-editorial font-medium text-[#141413] mt-0.5">O(1) Constant</div>
            <div className="text-[11px] text-[#57534E] mt-0.5">Bounded depth via stop-gradient recursion</div>
          </div>
        </div>
      </div>

      {/* Stepper Navigation: Swift / Anthropic Minimalist Step Switcher */}
      <div className="bg-white border border-[#E8E5DF] rounded-xl p-2 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5">
          {STEPS.map((s) => {
            const isActive = s.id === currentStep;
            const isCompleted = s.id < currentStep;
            return (
              <button
                key={s.id}
                onClick={() => setCurrentStep(s.id)}
                className={`text-left p-2.5 rounded-lg transition-all ${
                  isActive
                    ? "bg-[#141413] text-white shadow-xs font-medium"
                    : isCompleted
                    ? "bg-[#F4F1EA] text-[#57534E] hover:bg-[#EFECE4]"
                    : "bg-transparent text-[#8C887B] hover:bg-[#F4F1EA]"
                }`}
              >
                <div className="flex items-center space-x-1.5 text-[11px]">
                  <span
                    className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-mono ${
                      isActive
                        ? "bg-[#C96442] text-white"
                        : isCompleted
                        ? "bg-[#D6D2C9] text-[#141413]"
                        : "bg-[#E8E5DF] text-[#8C887B]"
                    }`}
                  >
                    {isCompleted ? "✓" : s.id}
                  </span>
                  <span className="font-mono text-[10px] opacity-75">
                    Step {s.id}
                  </span>
                </div>
                <div className="text-xs font-medium mt-1 truncate">
                  {explanationMode === "everyday" ? s.everydayTag : s.tag}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Step Content Container */}
      <div className="bg-white border border-[#E8E5DF] rounded-xl p-6 sm:p-8 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-6">
        {/* Step Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E8E5DF] pb-4">
          <div>
            <div className="inline-flex items-center space-x-2 text-xs font-mono text-[#C96442] bg-[#FAF0EC] px-2.5 py-1 rounded-full border border-[#E8D8D0] mb-2">
              <span>Step {step.id} of 5</span>
              <span>·</span>
              <span>{explanationMode === "everyday" ? step.everydayTag : step.tag}</span>
              <span>·</span>
              <span className="text-[#7A361F] font-sans font-medium text-[11px]">
                {explanationMode === "everyday" ? "Everyday View" : "Researcher View"}
              </span>
            </div>
            <h2 className="font-editorial text-2xl font-normal text-[#141413]">
              {explanationMode === "everyday" ? step.everydayTitle : step.title}
            </h2>
            {explanationMode === "everyday" ? (
              <p className="text-xs text-[#8C887B] font-mono mt-0.5">
                Formal Stage Name: <span className="text-[#141413]">{step.title}</span>
              </p>
            ) : (
              <p className="text-xs text-[#8C887B] font-mono mt-0.5">
                Everyday Analogy: <span className="text-[#141413]">{step.everydayTitle}</span>
              </p>
            )}
          </div>

          {/* Stepper Controls */}
          <div className="flex items-center space-x-2 self-start sm:self-auto">
            <button
              onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
              disabled={currentStep === 1}
              className="px-3.5 py-1.5 rounded-lg border border-[#E8E5DF] text-xs font-medium text-[#57534E] hover:bg-[#F4F1EA] disabled:opacity-40 disabled:pointer-events-none flex items-center space-x-1.5 transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Previous</span>
            </button>
            <button
              onClick={() => setCurrentStep((prev) => Math.min(STEPS.length, prev + 1))}
              disabled={currentStep === STEPS.length}
              className="px-3.5 py-1.5 rounded-lg bg-[#141413] text-white text-xs font-medium hover:bg-[#27272A] disabled:opacity-40 disabled:pointer-events-none flex items-center space-x-1.5 transition-colors"
            >
              <span>Next</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Mathematical Formulation / Plain English Equation Callout */}
        {explanationMode === "everyday" ? (
          <div className="bg-[#FAF9F5] border border-[#E8E5DF] rounded-xl p-4">
            <div className="flex items-center justify-between mb-1.5">
              <div className="text-[10px] uppercase font-mono tracking-wider text-[#8C887B] flex items-center space-x-1.5">
                <Lightbulb className="h-3 w-3 text-amber-600" />
                <span>Plain English Equation (How It Works In Real Life)</span>
              </div>
              <button
                onClick={() => setExplanationMode("researcher")}
                className="text-[11px] text-[#C96442] hover:underline font-mono flex items-center space-x-1"
              >
                <span>Switch to KaTeX formula</span>
                <ArrowRight className="h-3 w-3" />
              </button>
            </div>
            <div className="text-[#141413] font-semibold text-xs sm:text-sm font-mono bg-white border border-[#E8E5DF] rounded-lg p-2.5 shadow-xs">
              {step.everydayFormula}
            </div>
            <p className="text-xs text-[#57534E] mt-2 leading-relaxed">
              {step.everydayFormulaNote}
            </p>
          </div>
        ) : (
          <div className="bg-[#FAF9F5] border border-[#E8E5DF] rounded-xl p-4 font-mono text-xs text-[#141413] overflow-x-auto">
            <div className="flex items-center justify-between mb-1">
              <div className="text-[10px] uppercase font-mono tracking-wider text-[#8C887B]">
                Mathematical Formulation
              </div>
              <button
                onClick={() => setExplanationMode("everyday")}
                className="text-[11px] text-[#C96442] hover:underline font-sans flex items-center space-x-1"
              >
                <span>Switch to Plain English</span>
                <ArrowRight className="h-3 w-3" />
              </button>
            </div>
            <div className="text-[#141413] font-semibold">{step.formula}</div>
          </div>
        )}

        {/* 2-Column Split: Concept & Code */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Column 1: Concept & Why It Matters (Tabbed: Everyday Person vs Researcher) */}
          <div className="space-y-4">
            {/* View Switcher Pill Tabs for Column 1 */}
            <div className="flex items-center space-x-1.5 bg-[#FAF9F5] p-1 rounded-lg border border-[#E8E5DF]">
              <button
                onClick={() => setExplanationMode("everyday")}
                className={`flex-1 py-1.5 px-2.5 rounded-md text-xs font-medium transition-all flex items-center justify-center space-x-1.5 ${
                  explanationMode === "everyday"
                    ? "bg-[#C96442] text-white shadow-xs"
                    : "text-[#57534E] hover:bg-white"
                }`}
              >
                <Lightbulb className="h-3.5 w-3.5" />
                <span>Everyday Explanation</span>
              </button>
              <button
                onClick={() => setExplanationMode("researcher")}
                className={`flex-1 py-1.5 px-2.5 rounded-md text-xs font-medium transition-all flex items-center justify-center space-x-1.5 ${
                  explanationMode === "researcher"
                    ? "bg-[#141413] text-white shadow-xs"
                    : "text-[#57534E] hover:bg-white"
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span>Theoretical Concept</span>
              </button>
            </div>

            {explanationMode === "everyday" ? (
              <>
                {/* Everyday Narrative */}
                <div>
                  <h3 className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B] mb-1.5">
                    What&apos;s Happening in Plain English
                  </h3>
                  <p className="text-sm text-[#57534E] leading-relaxed">
                    {step.everydayExplanation}
                  </p>
                </div>

                {/* Real-World Analogy */}
                <div className="bg-[#FAF0EC]/80 border border-[#E8D8D0] rounded-xl p-4">
                  <div className="flex items-center space-x-2 text-xs font-medium text-[#7A361F] mb-1.5">
                    <Brain className="h-4 w-4 text-[#C96442]" />
                    <span>Real-World Analogy: {step.everydayAnalogy.title}</span>
                  </div>
                  <p className="text-xs text-[#57534E] leading-relaxed">
                    {step.everydayAnalogy.description}
                  </p>
                </div>

                {/* Why This Matters to You */}
                <div className="bg-white border border-[#E8E5DF] rounded-xl p-4 shadow-xs">
                  <h4 className="text-xs font-medium text-[#141413] mb-1 flex items-center space-x-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                    <span>Why This Matters to You:</span>
                  </h4>
                  <p className="text-xs text-[#57534E] leading-relaxed">
                    {step.whyItMattersEveryday}
                  </p>
                </div>

                {/* Stage Snapshot: How Information Moves */}
                <div className="bg-[#FAF9F5] border border-[#E8E5DF] rounded-xl p-3.5 text-xs">
                  <div className="text-[10px] font-mono uppercase tracking-wider text-[#8C887B] mb-2">
                    Stage Snapshot: What&apos;s Flowing
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <div className="bg-white border border-[#E8E5DF] rounded-lg p-2.5">
                      <span className="text-[10px] text-[#8C887B] uppercase block font-mono">Input</span>
                      <span className="text-xs font-medium text-[#141413] mt-0.5 block">{step.mentalModel.input}</span>
                    </div>
                    <div className="bg-white border border-[#E8E5DF] rounded-lg p-2.5">
                      <span className="text-[10px] text-[#C96442] uppercase block font-mono">What AI Does</span>
                      <span className="text-xs font-medium text-[#141413] mt-0.5 block">{step.mentalModel.action}</span>
                    </div>
                    <div className="bg-white border border-[#E8E5DF] rounded-lg p-2.5">
                      <span className="text-[10px] text-emerald-600 uppercase block font-mono">Result</span>
                      <span className="text-xs font-medium text-[#141413] mt-0.5 block">{step.mentalModel.output}</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* Researcher Theoretical Concept */}
                <div>
                  <h3 className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B] mb-1.5">
                    Theoretical Concept
                  </h3>
                  <p className="text-sm text-[#57534E] leading-relaxed">{step.concept}</p>
                </div>

                <div className="bg-[#FAF0EC]/60 border border-[#E8D8D0] rounded-xl p-4">
                  <h4 className="text-xs font-medium text-[#7A361F] mb-1">Why this replaces textual CoT:</h4>
                  <p className="text-xs text-[#57534E] leading-relaxed">{step.whyItMatters}</p>
                </div>

                {/* Tensor Shapes Table */}
                <div>
                  <h3 className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B] mb-2">
                    Tensor Dimensions & States
                  </h3>
                  <div className="border border-[#E8E5DF] rounded-xl overflow-hidden text-xs">
                    <table className="w-full text-left">
                      <thead className="bg-[#FAF9F5] border-b border-[#E8E5DF] text-[#57534E] text-[11px]">
                        <tr>
                          <th className="py-2.5 px-3.5 font-medium">Tensor</th>
                          <th className="py-2.5 px-3.5 font-medium font-mono">Shape</th>
                          <th className="py-2.5 px-3.5 font-medium">Semantic Role</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#E8E5DF]/60 text-[#57534E]">
                        {step.tensorShapes.map((t, idx) => (
                          <tr key={idx} className="hover:bg-[#FAF9F5]">
                            <td className="py-2.5 px-3.5 font-bold font-mono text-[#141413]">{t.name}</td>
                            <td className="py-2.5 px-3.5 font-mono text-[11px] text-[#C96442] bg-[#FAF0EC]/40">
                              {t.shape}
                            </td>
                            <td className="py-2.5 px-3.5 text-[#57534E]">{t.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Column 2: Exact Code Implementation */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between bg-[#1E1E20] text-[#A1A1AA] px-4 py-2.5 rounded-t-xl text-xs font-mono border-b border-[#27272A]">
              <div className="flex items-center space-x-2">
                <FileCode className="h-3.5 w-3.5 text-[#C96442]" />
                <span className="text-[#FAFAFA]">{step.codeLineReference}</span>
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
            <div className="bg-[#141416] p-4 rounded-b-xl border border-[#27272A] text-xs font-mono text-[#E4E4E7] overflow-x-auto flex-grow leading-relaxed">
              <pre className="whitespace-pre">{step.codeSnippet}</pre>
            </div>
          </div>
        </div>

        {/* Step-by-Step Data Flow Visual Diagram */}
        <div className="pt-5 border-t border-[#E8E5DF]">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[11px] font-mono uppercase tracking-wider text-[#8C887B]">
              {explanationMode === "everyday" ? "Everyday Stage Flow" : "Tensor Pathway"} for Step {step.id}
            </div>
            <div className="text-[10px] text-[#8C887B] font-mono">
              Highlighted: Step {step.id} ({explanationMode === "everyday" ? step.everydayTag : step.tag})
            </div>
          </div>

          {explanationMode === "everyday" ? (
            <div className="bg-[#FAF9F5] border border-[#E8E5DF] rounded-xl p-4 flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs">
              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id >= 1
                    ? "bg-white border-[#D6D2C9] text-[#141413] font-semibold"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Your Prompt
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 1
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 1
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                1. Blank Workspaces
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 2
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 2
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                2. Silent Thinking (×2)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 3
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 3
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                3. Refine Draft (×1)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 4
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 4
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                4. Gut Check (Confidence ≥85%?)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 5
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                5. Clean Answer Out
              </div>
            </div>
          ) : (
            <div className="bg-[#FAF9F5] border border-[#E8E5DF] rounded-xl p-4 flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs font-mono">
              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id >= 1
                    ? "bg-white border-[#D6D2C9] text-[#141413] font-semibold"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Prompt X [1, N, 2048]
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 1
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 1
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Init (z_0=0, y_0=0)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 2
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 2
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Inner Loop: z_t^(k) (×n)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 3
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 3
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Solution Update: y_t (×1)
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 4
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : step.id > 4
                    ? "bg-white border-[#D6D2C9] text-[#141413]"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                ACT Gate: h_t ≥ 0.85?
              </div>

              <ArrowRight className="h-3.5 w-3.5 text-[#8C887B]" />

              <div
                className={`px-3 py-1.5 rounded-lg border ${
                  step.id === 5
                    ? "bg-[#FAF0EC] border-[#C96442] text-[#C96442] font-semibold shadow-xs"
                    : "bg-[#F4F1EA] border-[#E8E5DF] text-[#8C887B]"
                }`}
              >
                Direct Code Emission
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
