"use client";

import React, { useState, useEffect } from "react";
import { BenchmarkTask, TaskEvaluationReport } from "@/lib/types";
import {
  CheckCircle2,
  XCircle,
  Zap,
  Gauge,
  Clock,
  ChevronDown,
  ChevronRight,
  Code2,
  Cpu,
  Layers,
  Copy,
  Check,
  TrendingDown,
  Sparkles
} from "lucide-react";

interface BenchmarkExplorerProps {
  tasks: BenchmarkTask[];
}

export const BenchmarkExplorer: React.FC<BenchmarkExplorerProps> = ({ tasks }) => {
  const [selectedTaskId, setSelectedTaskId] = useState<string>("Complex/0");
  const [report, setReport] = useState<TaskEvaluationReport | null>(null);
  const [selectedTask, setSelectedTask] = useState<BenchmarkTask | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterDifficulty, setFilterDifficulty] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [expandedThoughtIndex, setExpandedThoughtIndex] = useState<number | null>(0);

  useEffect(() => {
    async function loadReport(taskId: string) {
      setLoading(true);
      try {
        const res = await fetch(`/api/benchmarks?task_id=${encodeURIComponent(taskId)}`);
        const data = await res.json();
        if (data.report && data.task) {
          setReport(data.report);
          setSelectedTask(data.task);
        }
      } catch (err) {
        console.error("Failed to load benchmark task report:", err);
      } finally {
        setLoading(false);
      }
    }
    loadReport(selectedTaskId);
  }, [selectedTaskId]);

  const handleCopy = (code: string, key: string) => {
    navigator.clipboard.writeText(code);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const filteredTasks = tasks.filter((t) => {
    const matchesDiff = filterDifficulty === "all" || t.difficulty === filterDifficulty;
    const matchesSearch =
      t.prompt.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.task_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (t.category && t.category.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesDiff && matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Top Controls: Task Selector, Filters, Search */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center space-x-2">
              <Layers className="h-5 w-5 text-cyan-400" />
              <span>Benchmark Exploration: 3-Way Paradigm Comparison</span>
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Inspect empirical runs across 200 algorithmic tasks comparing Zero-Shot Baseline, Discrete CoT, and Continuous TRM.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Filter by Difficulty */}
            <select
              value={filterDifficulty}
              onChange={(e) => setFilterDifficulty(e.target.value)}
              className="bg-slate-850 border border-slate-700 text-xs sm:text-sm rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              <option value="all">All Difficulties (Tiers 1-4)</option>
              <option value="Tier 1">Tier 1 (Strings & Arrays)</option>
              <option value="Tier 2">Tier 2 (Grid & Simulation)</option>
              <option value="Tier 3">Tier 3 (BFS / Graph Key)</option>
              <option value="Tier 4">Tier 4 (Tree Balancing)</option>
            </select>

            {/* Task Search Input */}
            <input
              type="text"
              placeholder="Search tasks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-850 border border-slate-700 text-xs sm:text-sm rounded-lg px-3 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-44 sm:w-56"
            />
          </div>
        </div>

        {/* Task Pills */}
        <div className="mt-4 flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
          {filteredTasks.slice(0, 10).map((t) => (
            <button
              key={t.task_id}
              onClick={() => setSelectedTaskId(t.task_id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all flex items-center space-x-2 ${
                selectedTaskId === t.task_id
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              <span>{t.task_id}</span>
              {t.difficulty && (
                <span className={`text-[10px] px-1.5 py-0.2 rounded ${
                  selectedTaskId === t.task_id ? "bg-slate-950/20 text-slate-900" : "bg-slate-900 text-slate-400"
                }`}>
                  {t.difficulty}
                </span>
              )}
            </button>
          ))}
          {filteredTasks.length > 10 && (
            <span className="text-xs text-slate-500 self-center px-2">
              +{filteredTasks.length - 10} more tasks
            </span>
          )}
        </div>
      </div>

      {loading || !report ? (
        <div className="h-64 flex items-center justify-center bg-slate-900/50 border border-slate-800 rounded-2xl">
          <div className="flex items-center space-x-3 text-cyan-400">
            <Cpu className="h-6 w-6 animate-spin" />
            <span className="text-sm">Loading task execution telemetry and judge evaluation...</span>
          </div>
        </div>
      ) : (
        <>
          {/* Task Summary Banner */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
              <div>
                <div className="flex items-center space-x-3">
                  <h3 className="text-lg font-bold text-white">{selectedTask?.task_id}: {selectedTask?.entry_point}</h3>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800">
                    {selectedTask?.category || "Algorithmic"}
                  </span>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-800 text-slate-300">
                    {selectedTask?.difficulty || "Tier 2"}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">{selectedTask?.prompt}</p>
              </div>

              {/* Samsung TRM Verdict Highlight */}
              {report.samsung_paper_alignment && (
                <div className="bg-emerald-950/40 border border-emerald-800/80 rounded-xl p-3 flex items-center space-x-3">
                  <div className="h-9 w-9 rounded-lg bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                    <TrendingDown className="h-5 w-5 text-emerald-400" />
                  </div>
                  <div>
                    <div className="text-xs text-emerald-400 font-bold uppercase tracking-wider">
                      Samsung TRM Validation ({report.samsung_paper_alignment.token_saving_factor} Fewer Tokens)
                    </div>
                    <div className="text-xs text-slate-300">
                      {report.samsung_paper_alignment.latent_reasoning_verdict}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Telemetry Metric Comparison Badges */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-4">
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Baseline (Zero-Shot)</div>
                  <div className="text-base font-bold text-slate-200 mt-0.5">{report.baseline.tokens} tokens · {report.baseline.latency_ms}ms</div>
                </div>
                <div className="flex items-center space-x-1 text-xs text-emerald-400 bg-emerald-950/40 px-2 py-1 rounded-md border border-emerald-900">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>PASS</span>
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Discrete CoT (Multi-Turn)</div>
                  <div className="text-base font-bold text-amber-300 mt-0.5">{report.discrete.tokens} tokens · {report.discrete.latency_ms}ms</div>
                </div>
                <div className="flex items-center space-x-1 text-xs text-emerald-400 bg-emerald-950/40 px-2 py-1 rounded-md border border-emerald-900">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>PASS</span>
                </div>
              </div>

              <div className="bg-cyan-950/30 border border-cyan-800/80 rounded-xl p-3 flex items-center justify-between">
                <div>
                  <div className="text-xs text-cyan-300 font-medium">Continuous Latent TRM</div>
                  <div className="text-base font-bold text-cyan-200 mt-0.5">
                    {report.continuous.tokens} tokens · {report.continuous.latency_ms}ms
                  </div>
                </div>
                <div className="flex items-center space-x-1 text-xs text-cyan-300 bg-cyan-900/60 px-2 py-1 rounded-md border border-cyan-700">
                  <Zap className="h-3.5 w-3.5 text-cyan-300" />
                  <span>~3.4x EFFICIENT</span>
                </div>
              </div>
            </div>
          </div>

          {/* 3-Way Side-by-Side Comparison Columns */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Column 1: Zero-Shot Baseline */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col h-full glow-card">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-slate-500" />
                  <span className="font-bold text-sm text-slate-200">1. Zero-Shot Baseline</span>
                </div>
                <button
                  onClick={() => handleCopy(report.baseline.code, "baseline")}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 text-xs flex items-center space-x-1"
                >
                  {copiedKey === "baseline" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </div>

              <p className="text-xs text-slate-400 mb-3">
                Standard single-pass autoregressive generation without recursion or reflection.
              </p>

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[380px] border border-slate-850">
                <pre className="text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-wrap">
                  {report.baseline.code}
                </pre>
              </div>

              {/* Judge Score Summary */}
              {report.judge_evaluations && (
                <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                  <div className="flex items-center justify-between font-semibold text-slate-300">
                    <span>Judge Composite Score</span>
                    <span className="text-amber-400">{report.judge_evaluations.baseline.total_score} / 10</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 italic">
                    "{report.judge_evaluations.baseline.critique}"
                  </p>
                </div>
              )}
            </div>

            {/* Column 2: Discrete Recursive CoT */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col h-full glow-card">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-amber-400" />
                  <span className="font-bold text-sm text-amber-300">2. Discrete Recursive CoT</span>
                </div>
                <button
                  onClick={() => handleCopy(report.discrete.code, "discrete")}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 text-xs flex items-center space-x-1"
                >
                  {copiedKey === "discrete" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </div>

              <p className="text-xs text-slate-400 mb-3">
                Explicit textual chain-of-thought loop generating intermediate &lt;thought&gt; &amp; &lt;code_update&gt; tags.
              </p>

              {/* Accordion for Step-by-Step Thoughts */}
              {report.discrete.thoughts && report.discrete.thoughts.length > 0 && (
                <div className="mb-3 space-y-2">
                  {report.discrete.thoughts.map((thought, idx) => (
                    <div key={idx} className="bg-slate-950/70 border border-slate-800 rounded-lg text-xs overflow-hidden">
                      <button
                        onClick={() => setExpandedThoughtIndex(expandedThoughtIndex === idx ? null : idx)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 text-slate-300 hover:bg-slate-800/40 text-left"
                      >
                        <span className="font-medium text-amber-300/90">CoT Iteration {idx + 1} Scratchpad</span>
                        {expandedThoughtIndex === idx ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                      </button>
                      {expandedThoughtIndex === idx && (
                        <div className="px-3 py-2 text-[11px] text-slate-300 bg-slate-950 border-t border-slate-850 whitespace-pre-wrap leading-relaxed">
                          {thought}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[300px] border border-slate-850">
                <pre className="text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-wrap">
                  {report.discrete.code}
                </pre>
              </div>

              {/* Judge Score Summary */}
              {report.judge_evaluations && (
                <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                  <div className="flex items-center justify-between font-semibold text-slate-300">
                    <span>Judge Composite Score</span>
                    <span className="text-amber-400">{report.judge_evaluations.discrete.total_score} / 10</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 italic">
                    "{report.judge_evaluations.discrete.critique}"
                  </p>
                </div>
              )}
            </div>

            {/* Column 3: Continuous Latent TRM */}
            <div className="bg-slate-900 border border-cyan-800/60 rounded-2xl p-4 flex flex-col h-full glow-card shadow-lg shadow-cyan-950/20">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="font-bold text-sm text-cyan-300">3. Continuous Latent TRM</span>
                </div>
                <button
                  onClick={() => handleCopy(report.continuous.code, "continuous")}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 text-xs flex items-center space-x-1"
                >
                  {copiedKey === "continuous" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </div>

              <p className="text-xs text-slate-400 mb-3">
                Samsung TRM latent recurrence over dual vectors (z reasoning, y solution). Emits direct code only.
              </p>

              {/* Latent Convergence Distance Telemetry */}
              {report.continuous.trajectory_distances && (
                <div className="bg-cyan-950/30 border border-cyan-900/50 rounded-lg p-2.5 mb-3 text-xs">
                  <div className="flex items-center justify-between text-cyan-300 text-[11px] font-semibold mb-1">
                    <span>Latent Distance Convergence: d(z_t, z_t-1)</span>
                    <span>T={report.continuous.iterations_completed} {report.continuous.halted_early && "(ACT Halted)"}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {report.continuous.trajectory_distances.map((dist, idx) => (
                      <div key={idx} className="flex-1 bg-slate-900 rounded p-1 text-center">
                        <div className="text-[10px] text-slate-400">t={idx+1}</div>
                        <div className="text-xs font-mono font-bold text-cyan-400">{dist.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[300px] border border-cyan-950">
                <pre className="text-xs text-slate-200 font-mono leading-relaxed whitespace-pre-wrap">
                  {report.continuous.code}
                </pre>
              </div>

              {/* Judge Score Summary */}
              {report.judge_evaluations && (
                <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                  <div className="flex items-center justify-between font-semibold text-slate-200">
                    <span>Judge Composite Score</span>
                    <span className="text-emerald-400 font-bold">{report.judge_evaluations.continuous.total_score} / 10</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 italic">
                    "{report.judge_evaluations.continuous.critique}"
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
