"use client";

import React from "react";
import { ARCHITECTURE_COMPARISONS } from "@/lib/researchData";
import { Cpu, Zap, Layers, RefreshCw, GitBranch, ArrowRight, BookOpen } from "lucide-react";

export const ArchitectureExplainer: React.FC = () => {
  return (
    <div className="space-y-8">
      {/* Title & Introduction */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-xl">
        <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2">
          <BookOpen className="h-4 w-4" />
          <span>Architectural Deep Dive</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Continuous Latent TRM vs. DiffusionGemma
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl mt-2 leading-relaxed">
          Google’s experimental <strong>DiffusionGemma</strong> (built on Gemma 4) introduces a non-autoregressive token-canvas denoising paradigm, while <strong>Tiny Recursive Gemma</strong> adapts Samsung SAIL Montréal’s <strong>Tiny Recursive Model (TRM)</strong> to latent-space pre-computation. Below is an engineering comparison of how both paradigms push beyond conventional autoregressive generation.
        </p>
      </div>

      {/* Structural Comparison Matrix */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 sm:p-6 border-b border-slate-800">
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Layers className="h-5 w-5 text-cyan-400" />
            <span>Paradigm Matrix: Discrete CoT vs Continuous TRM vs DiffusionGemma</span>
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

      {/* The Dual-Latent Update Mechanism Flow */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
            <RefreshCw className="h-4 w-4 text-cyan-400" />
            <span>Dual-Latent Recurrence Mechanism (z and y)</span>
          </h3>
          <p className="text-xs text-slate-400 mb-4 leading-relaxed">
            Maintains two continuous latent states: <strong>z</strong> (internal reasoning scratchpad) and <strong>y</strong> (target solution draft), initialized to zeros.
          </p>

          <div className="bg-slate-950 rounded-xl p-4 border border-slate-850 font-mono text-xs text-slate-300 space-y-3">
            <div className="text-slate-500">// 1. For T iterations:</div>
            <div className="text-cyan-300 pl-4">
              for iter in range(T):
            </div>
            <div className="text-slate-400 pl-8">
              // Update reasoning state z (n times):
            </div>
            <div className="text-emerald-400 pl-8">
              z = Transformer([prompt, y, z])[:, -1:, :]
            </div>
            <div className="text-slate-400 pl-8 mt-2">
              // Update solution state y (1 time):
            </div>
            <div className="text-emerald-400 pl-8">
              y = Transformer([prompt, z, y])[:, -1:, :]
            </div>
            <div className="text-slate-500 mt-2">// 2. Prime generation:</div>
            <div className="text-purple-300 pl-4">
              output = Autoregressive([prompt, z, y])
            </div>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-2 flex items-center space-x-2">
            <Zap className="h-4 w-4 text-emerald-400" />
            <span>Gradient-Free Premature Recursion (Stop-Gradient)</span>
          </h3>
          <p className="text-xs text-slate-400 mb-4 leading-relaxed">
            Unrolling T steps directly triples computational graph size and causes Out-Of-Memory (OOM) failures. TRM solves this with stop-gradient prefix unrolling:
          </p>

          <div className="space-y-3 text-xs">
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <span className="font-bold text-slate-200 block mb-1">1. Steps 1 to T-1 (Gradient-Free):</span>
              <p className="text-slate-400 text-[11px]">
                Executed under <code className="text-cyan-300">mx.stop_gradient</code> (or <code className="text-cyan-300">torch.no_grad()</code>). Zero computational graph is stored. VRAM usage is strictly O(1).
              </p>
            </div>

            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <span className="font-bold text-slate-200 block mb-1">2. Final Step T (Active Gradients):</span>
              <p className="text-slate-400 text-[11px]">
                Only the final update and token prediction track gradients. This is mathematically exact for convergent reasoning states and requires zero fixed-point approximations.
              </p>
            </div>

            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <span className="font-bold text-slate-200 block mb-1">3. Selective LoRA Targeting:</span>
              <p className="text-slate-400 text-[11px]">
                Adapters are isolated to top transformer layers (layers 24-25 in Gemma 4 2B), dedicating a compact 2-layer recursive reasoning engine while keeping base model capabilities intact.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
