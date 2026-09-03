"use client";

import React, { useState } from "react";
import { ARCHITECTURE_COMPARISONS } from "@/lib/researchData";
import {
  Cpu,
  Zap,
  Layers,
  RefreshCw,
  GitBranch,
  ArrowRight,
  BookOpen,
  Cloud,
  FileCode,
  ShieldAlert,
  Server,
  Activity,
  CheckCircle2,
  Lock
} from "lucide-react";

export const ArchitectureExplainer: React.FC = () => {
  const [activeSection, setActiveSection] = useState<"math" | "cloud" | "comparison" | "safeguards">("math");

  return (
    <div className="space-y-8">
      {/* Title & Introduction */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-xl">
        <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">
          <BookOpen className="h-4 w-4" />
          <span>Researcher Architectural Specification</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Tiny Recursive Gemma: Latent-Space Recursion Architecture
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl mt-2 leading-relaxed">
          Transferring Samsung SAIL Montréal’s <strong>Tiny Recursive Model (TRM)</strong> (arXiv:2510.04871) to pretrained <strong>Google Gemma 4 / 2B</strong>.
          By replacing token-space autoregressive reasoning traces with continuous hidden-state iterations, the model achieves <strong>~3.4x token savings</strong> and <strong>O(1) constant memory scaling</strong>.
        </p>

        {/* Section Tabs */}
        <div className="flex flex-wrap items-center gap-2 mt-6 pt-6 border-t border-slate-800">
          <button
            onClick={() => setActiveSection("math")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              activeSection === "math"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <RefreshCw className="h-4 w-4" />
            <span>Mathematical Formulation</span>
          </button>

          <button
            onClick={() => setActiveSection("cloud")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              activeSection === "cloud"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Cloud className="h-4 w-4" />
            <span>Google Cloud Topology</span>
          </button>

          <button
            onClick={() => setActiveSection("comparison")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              activeSection === "comparison"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers className="h-4 w-4" />
            <span>Paradigm Matrix (TRM vs DiffusionGemma)</span>
          </button>

          <button
            onClick={() => setActiveSection("safeguards")}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              activeSection === "safeguards"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <ShieldAlert className="h-4 w-4" />
            <span>Hardware Safeguards</span>
          </button>
        </div>
      </div>

      {/* SECTION 1: MATHEMATICAL FORMULATION */}
      {activeSection === "math" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Dual Latent Update Flow */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl">
              <h2 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
                <RefreshCw className="h-4 w-4 text-cyan-400" />
                <span>1. Dual-Latent Recurrence Equations</span>
              </h2>
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                Maintains distinct continuous vectors: reasoning vector <code className="text-cyan-300">z_t</code> updated <code className="text-slate-200">n</code> times, and candidate solution vector <code className="text-emerald-300">y_t</code> updated once per recurrence step.
              </p>

              <div className="bg-slate-950 rounded-2xl p-4 border border-slate-850 font-mono text-xs text-slate-300 space-y-3">
                <div className="text-slate-500">// Initialization:</div>
                <div className="text-slate-400">z_0 = 0, y_0 = 0  in R^(1 x d_model)</div>

                <div className="text-slate-500 mt-2">// Step A: Inner Reasoning Loop (k = 1..n):</div>
                <div className="text-cyan-300 pl-2">
                  z_t^(k) = Transformer([X ; RMSNorm(y_(t-1)) ; RMSNorm(z_t^(k-1))])[-1, :]
                </div>

                <div className="text-slate-500 mt-2">// Step B: Solution Update (k = n):</div>
                <div className="text-emerald-400 pl-2">
                  y_t = Transformer([X ; RMSNorm(z_t^(n)) ; RMSNorm(y_(t-1))])[-1, :]
                </div>

                <div className="text-slate-500 mt-2">// Step C: Autoregressive Emission Priming:</div>
                <div className="text-purple-300 pl-2">
                  P(W | X) = Autoregressive([X ; RMSNorm(y_T) ; RMSNorm(z_T)])
                </div>
              </div>
            </div>

            {/* Deep Supervision & ACT Head */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl">
              <h2 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
                <Zap className="h-4 w-4 text-emerald-400" />
                <span>2. Multi-Step Deep Supervision & ACT</span>
              </h2>
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                Intermediate solutions <code className="text-slate-200">y_t</code> are supervised with monotonically increasing power-decay weights to reward late-stage refinement while preserving early gradients.
              </p>

              <div className="bg-slate-950 rounded-2xl p-4 border border-slate-850 font-mono text-xs text-slate-300 space-y-3">
                <div className="text-slate-500">// Power Decay Loss Weights:</div>
                <div className="text-cyan-300">
                  w_t = (t^γ) / Σ (j^γ)  (default γ = 1.5)
                </div>
                <div className="text-slate-400 pl-2 text-[11px]">
                  T=3: w_1 = 0.111, w_2 = 0.313, w_3 = 0.576
                </div>

                <div className="text-slate-500 mt-2">// Adaptive Computation Time (ACT) Gate:</div>
                <div className="text-emerald-400">
                  h_t = σ(W_2 · GELU(W_1 · z_t + b_1) + b_2) ∈ [0, 1]
                </div>
                <div className="text-slate-400 pl-2 text-[11px]">
                  Halts when h_t &ge; τ (default τ = 0.85) or cosine_dist &lt; 0.02
                </div>

                <div className="text-slate-500 mt-2">// Combined Optimization Objective:</div>
                <div className="text-purple-300">
                  L_total = Σ w_t · L_LM(y_t) + λ · L_BCE(h_t, h_t*)
                </div>
              </div>
            </div>
          </div>

          {/* O(1) Memory Proof Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl">
            <h2 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
              <Layers className="h-4 w-4 text-cyan-400" />
              <span>3. Theorem: O(1) Constant Memory Scaling via Stop-Gradient Unrolling</span>
            </h2>
            <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
              Standard Backpropagation Through Time (BPTT) unrolling over <code className="text-cyan-300">T</code> steps stores an unrolled graph of depth <code className="text-slate-400">O(T · (n+1) · L)</code>, triggering severe GPU/VRAM OOMs.
              By applying <code className="text-cyan-400">stop_gradient</code> (<code className="text-cyan-400">.detach()</code>) to previous latent vectors <code className="text-slate-200">z_(t-1)</code> and <code className="text-slate-200">y_(t-1)</code>, each recurrence step operates on bounded depth 1:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-slate-950 p-4 rounded-2xl border border-slate-850">
                <span className="text-xs font-bold text-emerald-400 block mb-1">Prefix Steps (1 to T-1)</span>
                <p className="text-[11px] text-slate-400">
                  Forward unrolling without tracking gradients. Graph size: 0 MB. Physical VRAM allocation is bounded and constant.
                </p>
              </div>
              <div className="bg-slate-950 p-4 rounded-2xl border border-slate-850">
                <span className="text-xs font-bold text-cyan-400 block mb-1">Step T (Active Gradients)</span>
                <p className="text-[11px] text-slate-400">
                  Only the active recurrent step tracks gradients. Mathematically exact for convergent fixed points.
                </p>
              </div>
              <div className="bg-slate-950 p-4 rounded-2xl border border-slate-850">
                <span className="text-xs font-bold text-purple-400 block mb-1">Selective LoRA Layering</span>
                <p className="text-[11px] text-slate-400">
                  Adapters restricted to top attention projections (layers 24-25 in Gemma 2B), preventing manifold drift.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 2: GOOGLE CLOUD TOPOLOGY */}
      {activeSection === "cloud" && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-xl">
            <h2 className="text-lg font-bold text-white mb-2 flex items-center space-x-2">
              <Cloud className="h-5 w-5 text-cyan-400" />
              <span>Production Google Cloud Topology (davenport-boutique &bull; us-central1)</span>
            </h2>
            <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
              To eliminate local Apple Silicon memory pressure and kernel panics, model training and heavy recurrent unrolls are offloaded to dedicated Google Cloud infrastructure:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              {/* Cloud Run */}
              <div className="bg-slate-950 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase mb-2">
                    <Server className="h-4 w-4" />
                    <span>Tier 1: Serving & Showcase</span>
                  </div>
                  <h3 className="text-sm font-bold text-white mb-2">Google Cloud Run v2</h3>
                  <div className="text-xs text-slate-400 space-y-1 font-mono text-[11px]">
                    <div>&bull; Service: tiny-recursive-gemma-web</div>
                    <div>&bull; Region: us-central1</div>
                    <div>&bull; Auth: HTTP Basic + SmartRouter Secret</div>
                    <div>&bull; Vertex AI Project Auth (ADC)</div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] text-emerald-400 flex items-center space-x-1">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>Deployed & Live (HTTP 200)</span>
                </div>
              </div>

              {/* Vertex AI */}
              <div className="bg-slate-950 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase mb-2">
                    <Cpu className="h-4 w-4" />
                    <span>Tier 2: Model Training</span>
                  </div>
                  <h3 className="text-sm font-bold text-white mb-2">Vertex AI Custom Jobs</h3>
                  <div className="text-xs text-slate-400 space-y-1 font-mono text-[11px]">
                    <div>&bull; Machine: g2-standard-4 (4 vCPU, 16GB)</div>
                    <div>&bull; GPU: 1x NVIDIA L4 (24GB VRAM)</div>
                    <div>&bull; Image: PyTorch 2.4 GPU (py311)</div>
                    <div>&bull; Script: cloud/train_torch_trm.py</div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] text-cyan-400 flex items-center space-x-1">
                  <Activity className="h-3.5 w-3.5" />
                  <span>On-Demand Ephemeral GPU</span>
                </div>
              </div>

              {/* Cloud Storage */}
              <div className="bg-slate-950 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center space-x-2 text-amber-400 text-xs font-bold uppercase mb-2">
                    <Lock className="h-4 w-4" />
                    <span>Tier 3: Staging & Artifacts</span>
                  </div>
                  <h3 className="text-sm font-bold text-white mb-2">Google Cloud Storage</h3>
                  <div className="text-xs text-slate-400 space-y-1 font-mono text-[11px]">
                    <div>&bull; Bucket: gs://davenport-boutique-vertex-staging/</div>
                    <div>&bull; /source: Automated tarball staging</div>
                    <div>&bull; /checkpoints: LoRA adapters + ACT</div>
                    <div>&bull; Retention: Immutable versioning</div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800 text-[10px] text-amber-400 flex items-center space-x-1">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>Staging Bucket Ready</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 3: PARADIGM COMPARISON MATRIX */}
      {activeSection === "comparison" && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="p-4 sm:p-6 border-b border-slate-800">
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <Layers className="h-5 w-5 text-cyan-400" />
              <span>Paradigm Matrix: Discrete CoT vs Continuous TRM vs Google DiffusionGemma</span>
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead className="bg-slate-950/85 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-3.5 px-4 font-semibold">Architectural Dimension</th>
                  <th className="py-3.5 px-4 font-semibold text-amber-300">Discrete CoT</th>
                  <th className="py-3.5 px-4 font-semibold text-cyan-300 bg-cyan-950/20">Continuous Latent TRM</th>
                  <th className="py-3.5 px-4 font-semibold text-purple-300">Google DiffusionGemma</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {ARCHITECTURE_COMPARISONS.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-4 font-semibold text-slate-200">{row.feature}</td>
                    <td className="py-3.5 px-4 text-slate-300">{row.discrete}</td>
                    <td className="py-3.5 px-4 text-cyan-300 font-medium bg-cyan-950/10">{row.continuous}</td>
                    <td className="py-3.5 px-4 text-purple-200">{row.diffusion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SECTION 4: HARDWARE SAFEGUARDS */}
      {activeSection === "safeguards" && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <h2 className="text-base font-bold text-white flex items-center space-x-2">
            <ShieldAlert className="h-4 w-4 text-amber-400" />
            <span>Local Apple Silicon Stability & Isolation Architecture</span>
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
            Consumer Apple Silicon shares non-pageable wired memory between CPU and GPU. Unrolling multi-step recurrent loops with unquantized 2B models can trigger kernel memory watchdog timeouts. The following architectural safeguards ensure complete system stability:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-850">
              <h3 className="text-xs font-bold text-emerald-400 mb-1">1. Darwin vm_stat Headroom Guard</h3>
              <p className="text-[11px] text-slate-400">
                Evaluates system memory headroom before every forward unroll. Automatically clears MLX Metal buffers (<code className="text-cyan-300">mx.metal.clear_cache()</code>) and triggers garbage collection if headroom drops below 2.0GB.
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-850">
              <h3 className="text-xs font-bold text-cyan-400 mb-1">2. Strict Pytest Sandbox Isolation</h3>
              <p className="text-[11px] text-slate-400">
                Heavy weights are excluded from local unit tests (<code className="text-cyan-300">RUN_HEAVY_TESTS=1</code> guard). Unit tests run entirely in 3 seconds using lightweight mock tensors with 0 RAM pressure.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
