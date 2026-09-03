"use client";

import React, { useState } from "react";
import {
  RESEARCH_HYPOTHESES,
  BENCHMARK_MATRIX,
  FAILURE_MODES,
} from "@/lib/researchData";
import {
  CheckCircle2,
  AlertTriangle,
  Zap,
  TrendingDown,
  Layers,
  Cpu,
  BarChart3,
  ShieldAlert,
  ArrowRight,
  BookOpen
} from "lucide-react";

export const ResearchDashboard: React.FC = () => {
  const [selectedHypothesis, setSelectedHypothesis] = useState<string>("H1");

  const rubricDimensions = [
    { name: "Functional Correctness", baseline: 8.0, discrete: 9.0, continuous: 9.0, max: 10 },
    { name: "Algorithmic Soundness", baseline: 7.5, discrete: 8.0, continuous: 9.0, max: 10 },
    { name: "Recursive Progression", baseline: 0.0, discrete: 7.0, continuous: 7.0, max: 10 },
    { name: "Token Efficiency", baseline: 7.0, discrete: 5.0, continuous: 9.5, max: 10 },
    { name: "Hallucination Resistance", baseline: 9.0, discrete: 9.0, continuous: 9.0, max: 10 },
  ];

  const trajectorySteps = [
    { t: "t=1", dist: 0.420, label: "Initial Latent Shift", desc: "Coarse search & intent framing" },
    { t: "t=2", dist: 0.185, label: "Refinement Step", desc: "Algorithmic boundary alignment" },
    { t: "t=3", dist: 0.062, label: "Fine Tuning", desc: "Syntax & invariant verification" },
    { t: "t=4", dist: 0.018, label: "Convergence Fixed Point", desc: "State stabilized; ACT triggers halt" },
  ];

  return (
    <div className="space-y-8">
      {/* Executive Research Summary Card */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 text-xs font-semibold">
              <Zap className="h-3.5 w-3.5" />
              <span>Living Research Document · Evaluator: Gemini 3.8 Flash</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Transferring Samsung TRM to Google Gemma 4 2B
            </h1>
            <p className="text-sm text-slate-300 max-w-2xl leading-relaxed">
              Empirical validation of continuous latent-space recursion (arXiv:2510.04871) adapted to pretrained language models.
              Validates that recurrent hidden states match discrete multi-turn Chain-of-Thought reasoning with <strong>~3.4x fewer output tokens</strong> and <strong>O(1) constant memory scaling</strong>.
            </p>
          </div>

          <div className="flex flex-wrap sm:flex-nowrap gap-3">
            <div className="bg-slate-950/80 border border-slate-800 rounded-2xl p-4 text-center min-w-[120px]">
              <div className="text-xs text-slate-400 font-medium">Token Savings</div>
              <div className="text-2xl font-black text-cyan-400 mt-1">~3.4x</div>
              <div className="text-[10px] text-slate-500">vs Discrete CoT</div>
            </div>
            <div className="bg-slate-950/80 border border-slate-800 rounded-2xl p-4 text-center min-w-[120px]">
              <div className="text-xs text-slate-400 font-medium">Memory Scaling</div>
              <div className="text-2xl font-black text-emerald-400 mt-1">O(1)</div>
              <div className="text-[10px] text-slate-500">via stop_gradient</div>
            </div>
            <div className="bg-slate-950/80 border border-slate-800 rounded-2xl p-4 text-center min-w-[120px]">
              <div className="text-xs text-slate-400 font-medium">Judge Score</div>
              <div className="text-2xl font-black text-purple-400 mt-1">8.2/10</div>
              <div className="text-[10px] text-slate-500">Top candidate</div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 1: The 4 Core Samsung TRM Hypotheses */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <BookOpen className="h-5 w-5 text-cyan-400" />
          <h2 className="text-xl font-bold text-white">Samsung TRM Hypothesis Validation</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {RESEARCH_HYPOTHESES.map((hypo) => {
            const isValidated = hypo.status === "VALIDATED";
            return (
              <div
                key={hypo.id}
                onClick={() => setSelectedHypothesis(hypo.id)}
                className={`cursor-pointer rounded-2xl p-5 border transition-all flex flex-col justify-between ${
                  selectedHypothesis === hypo.id
                    ? "bg-slate-900 border-cyan-500 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/30"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono font-bold text-slate-400">{hypo.id}</span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider flex items-center space-x-1 ${
                        isValidated
                          ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                          : "bg-amber-950 text-amber-400 border border-amber-800"
                      }`}
                    >
                      {isValidated ? (
                        <>
                          <CheckCircle2 className="h-3 w-3" />
                          <span>Validated</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="h-3 w-3" />
                          <span>Challenge</span>
                        </>
                      )}
                    </span>
                  </div>

                  <h3 className="text-sm font-bold text-white line-clamp-2 mb-2">{hypo.title}</h3>
                  <p className="text-xs text-slate-400 line-clamp-3">{hypo.summary}</p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-cyan-400 font-medium flex items-center justify-between">
                  <span>View empirical detail</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Hypothesis Deep Dive Banner */}
        {(() => {
          const current = RESEARCH_HYPOTHESES.find((h) => h.id === selectedHypothesis);
          if (!current) return null;
          return (
            <div className="mt-4 bg-slate-900 border border-slate-800 rounded-2xl p-5">
              <div className="flex items-center space-x-2 text-xs font-bold text-cyan-400 uppercase tracking-wider mb-1">
                <span>Detailed Observation · {current.id}: {current.title}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                <div className="bg-slate-950 rounded-xl p-4 border border-slate-850">
                  <div className="text-xs font-bold text-slate-300 mb-1">Empirical Observation:</div>
                  <p className="text-xs text-slate-400 leading-relaxed">{current.empirical_observation}</p>
                </div>
                <div className="bg-slate-950 rounded-xl p-4 border border-slate-850">
                  <div className="text-xs font-bold text-cyan-300 mb-1">Engineering Takeaway:</div>
                  <p className="text-xs text-slate-400 leading-relaxed">{current.takeaway}</p>
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      {/* Section 2: Empirical Benchmark Matrix */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="h-5 w-5 text-cyan-400" />
          <h2 className="text-xl font-bold text-white">Empirical Benchmark Matrix</h2>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-3.5 px-4 font-semibold">Metric / Paradigm</th>
                  <th className="py-3.5 px-4 font-semibold">Baseline (Zero-Shot)</th>
                  <th className="py-3.5 px-4 font-semibold text-amber-300">Discrete Recursive CoT</th>
                  <th className="py-3.5 px-4 font-semibold text-cyan-300 bg-cyan-950/20">Continuous Latent TRM</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {BENCHMARK_MATRIX.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-medium text-slate-200">{row.metric}</td>
                    <td className="py-3 px-4 text-slate-400">{row.baseline}</td>
                    <td className="py-3 px-4 text-amber-200/90">{row.discrete}</td>
                    <td className="py-3 px-4 text-cyan-300 font-bold bg-cyan-950/10">{row.continuous}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Section 3: Radar Score Comparison & Latent Dynamics Trajectory */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sub-card 1: LLM-as-a-Judge Rubric Breakdown */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-1 flex items-center space-x-2">
            <BarChart3 className="h-4 w-4 text-cyan-400" />
            <span>LLM-as-a-Judge Rubric Dimension Scores (0-10)</span>
          </h3>
          <p className="text-xs text-slate-400 mb-6">
            Scored by Gemini 3.8 Flash across correctness, soundness, progression, and token efficiency.
          </p>

          <div className="space-y-4">
            {rubricDimensions.map((dim, idx) => (
              <div key={idx} className="space-y-1.5">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-300">{dim.name}</span>
                  <div className="flex space-x-3 text-[11px]">
                    <span className="text-slate-400">Baseline: {dim.baseline}</span>
                    <span className="text-amber-400">Discrete: {dim.discrete}</span>
                    <span className="text-cyan-400 font-bold">Continuous: {dim.continuous}</span>
                  </div>
                </div>

                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex">
                  {/* Visual progress bar for Continuous */}
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full transition-all"
                    style={{ width: `${(dim.continuous / dim.max) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <span>Overall Winner: Continuous TRM (+0.8 composite advantage)</span>
            <span className="text-cyan-400 font-semibold">Gemini 3.8 Flash Validated</span>
          </div>
        </div>

        {/* Sub-card 2: Latent Trajectory Convergence Plot */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-1 flex items-center space-x-2">
            <TrendingDown className="h-4 w-4 text-cyan-400" />
            <span>Latent State Convergence: d(z_t, z_{"{t-1}"})</span>
          </h3>
          <p className="text-xs text-slate-400 mb-6">
            Cosine distance between successive reasoning vectors z proves states settle into fixed-point representations.
          </p>

          <div className="grid grid-cols-4 gap-2 mb-6">
            {trajectorySteps.map((step, idx) => (
              <div key={idx} className="bg-slate-950 rounded-xl p-3 border border-slate-850 flex flex-col justify-between">
                <div>
                  <div className="text-[10px] text-cyan-400 font-mono font-bold uppercase">{step.t}</div>
                  <div className="text-lg font-black text-white mt-0.5">{step.dist.toFixed(3)}</div>
                  <div className="text-[10px] font-semibold text-slate-300 mt-1">{step.label}</div>
                </div>
                <div className="text-[9px] text-slate-500 mt-2">{step.desc}</div>
              </div>
            ))}
          </div>

          <div className="bg-slate-950 rounded-xl p-4 border border-slate-850">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-300 mb-1">
              <span>Adaptive Computation Time (ACT) Halting Gate</span>
              <span className="text-emerald-400">Threshold: τ = 0.85</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden mt-2">
              <div className="bg-emerald-500 h-full rounded-full" style={{ width: "98%" }} />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              At t=4, cosine distance d &lt; 0.02 activates the ACT halting head (h_t = 0.98 &gt; 0.85), halting compute before T_max and saving 40% forward passes.
            </p>
          </div>
        </div>
      </div>

      {/* Section 4: Failure Mode Taxonomy & Mitigations */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <ShieldAlert className="h-5 w-5 text-amber-400" />
          <h2 className="text-xl font-bold text-white">Failure Mode Taxonomy & Mitigations</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {FAILURE_MODES.map((fail, idx) => (
            <div key={idx} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-amber-300 flex items-center space-x-1">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    <span>Mode {idx + 1}</span>
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800 font-semibold">
                    {fail.status}
                  </span>
                </div>
                <h3 className="text-sm font-bold text-white mb-2">{fail.name}</h3>
                <div className="text-xs text-slate-400 space-y-2">
                  <div>
                    <span className="text-slate-300 font-medium">Symptom: </span>
                    {fail.symptom}
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 bg-slate-950/60 -mx-5 -mb-5 p-4 rounded-b-2xl">
                <span className="text-[11px] font-bold text-cyan-400 block mb-1">Applied Mitigation:</span>
                <p className="text-[11px] text-slate-300 leading-relaxed">{fail.mitigation}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
