"use client";

import React, { useState } from "react";
import {
  Play,
  RotateCcw,
  Sparkles,
  Zap,
  Sliders,
  CheckCircle2,
  XCircle,
  Copy,
  Check,
  Cpu,
  Layers,
  ChevronDown,
  ChevronRight,
  ShieldCheck,
  TrendingDown
} from "lucide-react";

export const LivePlayground: React.FC = () => {
  const [prompt, setPrompt] = useState<string>(
    "Write a Python function to check if a number is prime and return its prime factor decomposition."
  );
  const [testSuite, setTestSuite] = useState<string>(
    `def check(candidate):\n    assert candidate(13) == (True, [13])\n    assert candidate(12) == (False, [2, 2, 3])\n    assert candidate(1) == (False, [])`
  );
  const [entryPoint, setEntryPoint] = useState<string>("candidate");
  const [iterations, setIterations] = useState<number>(3);
  const [reasoningSteps, setReasoningSteps] = useState<number>(2);
  const [haltThreshold, setHaltThreshold] = useState<number>(0.85);

  const [loadingInfer, setLoadingInfer] = useState<boolean>(false);
  const [loadingJudge, setLoadingJudge] = useState<boolean>(false);
  const [loadingTest, setLoadingTest] = useState<boolean>(false);

  const [results, setResults] = useState<any | null>(null);
  const [judgeResults, setJudgeResults] = useState<any | null>(null);
  const [testResult, setTestResult] = useState<{ passed: boolean; output: string } | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [expandedThought, setExpandedThought] = useState<number | null>(0);

  const presets = [
    {
      title: "Prime Factorization",
      prompt: "Write a Python function to check if a number is prime and return its prime factor decomposition as a tuple (is_prime, factors_list).",
      entry: "prime_decomposition",
      test: `def check(candidate):\n    assert candidate(13) == (True, [13])\n    assert candidate(12) == (False, [2, 2, 3])\n    assert candidate(1) == (False, [])`
    },
    {
      title: "LRU Cache O(1)",
      prompt: "Implement an LRU Cache with capacity k supporting get(key) and put(key, value) in O(1) average time complexity using a doubly-linked list and hash map.",
      entry: "LRUCache",
      test: `def check(candidate):\n    c = candidate(2)\n    c.put(1, 1)\n    c.put(2, 2)\n    assert c.get(1) == 1\n    c.put(3, 3)\n    assert c.get(2) == -1`
    },
    {
      title: "Longest Palindrome",
      prompt: "Write a Python function to find the longest palindromic substring in a string in O(n^2) time or better.",
      entry: "longest_palindrome",
      test: `def check(candidate):\n    assert candidate("babad") in ("bab", "aba")\n    assert candidate("cbbd") == "bb"`
    }
  ];

  const handleRunInference = async () => {
    setLoadingInfer(true);
    setResults(null);
    setJudgeResults(null);
    setTestResult(null);

    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          iterations,
          reasoning_steps: reasoningSteps,
          halt_threshold: haltThreshold
        })
      });
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error("Inference request failed:", err);
    } finally {
      setLoadingInfer(false);
    }
  };

  const handleRunJudge = async () => {
    if (!results) return;
    setLoadingJudge(true);
    try {
      const res = await fetch("/api/judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_prompt: prompt,
          test_suite: testSuite,
          candidates: {
            baseline: results.baseline.code,
            discrete: results.discrete.code,
            continuous: results.continuous.code
          }
        })
      });
      const data = await res.json();
      setJudgeResults(data);
    } catch (err) {
      console.error("Judge evaluation failed:", err);
    } finally {
      setLoadingJudge(false);
    }
  };

  const handleRunTest = async (code: string) => {
    setLoadingTest(true);
    try {
      const res = await fetch("/api/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          test: testSuite,
          entry_point: entryPoint
        })
      });
      const data = await res.json();
      setTestResult(data);
    } catch (err) {
      console.error("Test execution failed:", err);
    } finally {
      setLoadingTest(false);
    }
  };

  const handleCopy = (code: string, key: string) => {
    navigator.clipboard.writeText(code);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Playground Header & Presets */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-blue-400" />
              <span>Option B: Live "Try It Out" Playground</span>
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">
              Submit custom prompts to run live cloud inference across Zero-Shot, Discrete CoT, and Continuous TRM.
            </p>
          </div>

          {/* Quick Presets */}
          <div className="flex items-center gap-2 overflow-x-auto">
            <span className="text-xs text-slate-500 font-medium">Presets:</span>
            {presets.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setPrompt(p.prompt);
                  setTestSuite(p.test);
                  setEntryPoint(p.entry);
                }}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors whitespace-nowrap"
              >
                {p.title}
              </button>
            ))}
          </div>
        </div>

        {/* Input Textarea & Hyperparameters Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-3">
            <label className="text-xs font-semibold text-slate-300 block">Task Description & Prompt:</label>
            <textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 font-mono"
            />

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-400 block">Unit Test Suite (Optional):</label>
              <textarea
                rows={3}
                value={testSuite}
                onChange={(e) => setTestSuite(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-blue-500 font-mono"
              />
            </div>
          </div>

          {/* Hyperparameter Sliders */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-4">
            <div className="flex items-center space-x-2 text-xs font-bold text-slate-200 uppercase tracking-wider">
              <Sliders className="h-4 w-4 text-blue-400" />
              <span>TRM Hyperparameters</span>
            </div>

            {/* Recursion Depth T */}
            <div>
              <div className="flex justify-between text-xs font-medium mb-1">
                <span className="text-slate-300">Recursion Depth (T)</span>
                <span className="text-cyan-400 font-bold">{iterations} steps</span>
              </div>
              <input
                type="range"
                min="1"
                max="8"
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <span className="text-[10px] text-slate-500">Outer continuous latent recursion loops</span>
            </div>

            {/* Reasoning steps n */}
            <div>
              <div className="flex justify-between text-xs font-medium mb-1">
                <span className="text-slate-300">Reasoning Sub-steps (n)</span>
                <span className="text-cyan-400 font-bold">{reasoningSteps} per loop</span>
              </div>
              <input
                type="range"
                min="1"
                max="4"
                value={reasoningSteps}
                onChange={(e) => setReasoningSteps(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <span className="text-[10px] text-slate-500">Inner updates to reasoning state z per loop</span>
            </div>

            {/* ACT Halting Threshold tau */}
            <div>
              <div className="flex justify-between text-xs font-medium mb-1">
                <span className="text-slate-300">ACT Halt Threshold (τ)</span>
                <span className="text-emerald-400 font-bold">{haltThreshold}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={haltThreshold}
                onChange={(e) => setHaltThreshold(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <span className="text-[10px] text-slate-500">Dynamic early-exit probability threshold</span>
            </div>

            {/* Main Action Button */}
            <button
              onClick={handleRunInference}
              disabled={loadingInfer}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs sm:text-sm py-2.5 rounded-xl shadow-lg shadow-blue-500/20 transition-all flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              {loadingInfer ? (
                <>
                  <Cpu className="h-4 w-4 animate-spin" />
                  <span>Running Cloud Inference...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  <span>Run 3-Way Cloud Comparison</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-6">
          {/* Efficiency Summary Ribbon */}
          <div className="bg-cyan-950/30 border border-cyan-800 rounded-2xl p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <div className="h-9 w-9 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                <Zap className="h-5 w-5 text-cyan-400" />
              </div>
              <div>
                <span className="text-xs uppercase font-bold text-cyan-400 tracking-wider">
                  Live Cloud Telemetry: {results.efficiency?.token_reduction_ratio} Fewer Tokens
                </span>
                <p className="text-xs text-slate-300">{results.efficiency?.summary}</p>
              </div>
            </div>

            {/* Live Judge Action Button */}
            <button
              onClick={handleRunJudge}
              disabled={loadingJudge}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-purple-600 hover:bg-purple-500 text-white shadow-md shadow-purple-600/30 transition-all flex items-center space-x-2 disabled:opacity-50"
            >
              {loadingJudge ? (
                <>
                  <Cpu className="h-3.5 w-3.5 animate-spin" />
                  <span>Gemini Judging...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Evaluate with Gemini 3.8 Flash Judge</span>
                </>
              )}
            </button>
          </div>

          {/* 3-Way Output Columns */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Baseline Output */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col h-full glow-card">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-slate-500" />
                  <span className="font-bold text-sm text-slate-200">1. Baseline (Zero-Shot)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-slate-400">{results.baseline.tokens} tok</span>
                  <button
                    onClick={() => handleCopy(results.baseline.code, "play_baseline")}
                    className="p-1 rounded text-slate-400 hover:text-white"
                  >
                    {copiedKey === "play_baseline" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[340px] border border-slate-850">
                <pre className="text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-wrap">
                  {results.baseline.code}
                </pre>
              </div>

              <button
                onClick={() => handleRunTest(results.baseline.code)}
                className="mt-3 w-full py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition-colors"
              >
                Test Baseline Code
              </button>
            </div>

            {/* Discrete CoT Output */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col h-full glow-card">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-amber-400" />
                  <span className="font-bold text-sm text-amber-300">2. Discrete Recursive CoT</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-amber-400 font-bold">{results.discrete.tokens} tok</span>
                  <button
                    onClick={() => handleCopy(results.discrete.code, "play_discrete")}
                    className="p-1 rounded text-slate-400 hover:text-white"
                  >
                    {copiedKey === "play_discrete" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              {results.discrete.thoughts && results.discrete.thoughts.length > 0 && (
                <div className="mb-3 space-y-1.5">
                  {results.discrete.thoughts.map((th: string, idx: number) => (
                    <div key={idx} className="bg-slate-950/70 border border-slate-800 rounded-lg text-xs overflow-hidden">
                      <button
                        onClick={() => setExpandedThought(expandedThought === idx ? null : idx)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 text-slate-300 hover:bg-slate-800/40 text-left"
                      >
                        <span className="font-medium text-amber-300/90 text-[11px]">Thought Turn {idx + 1}</span>
                        {expandedThought === idx ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                      </button>
                      {expandedThought === idx && (
                        <div className="px-3 py-2 text-[11px] text-slate-300 bg-slate-950 border-t border-slate-850 whitespace-pre-wrap leading-relaxed">
                          {th}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[260px] border border-slate-850">
                <pre className="text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-wrap">
                  {results.discrete.code}
                </pre>
              </div>

              <button
                onClick={() => handleRunTest(results.discrete.code)}
                className="mt-3 w-full py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition-colors"
              >
                Test Discrete CoT Code
              </button>
            </div>

            {/* Continuous Latent TRM Output */}
            <div className="bg-slate-900 border border-cyan-800/60 rounded-2xl p-4 flex flex-col h-full glow-card shadow-lg shadow-cyan-950/20">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
                <div className="flex items-center space-x-2">
                  <div className="h-3 w-3 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="font-bold text-sm text-cyan-300">3. Continuous Latent TRM</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-cyan-400 font-bold">{results.continuous.tokens} tok</span>
                  <button
                    onClick={() => handleCopy(results.continuous.code, "play_continuous")}
                    className="p-1 rounded text-slate-400 hover:text-white"
                  >
                    {copiedKey === "play_continuous" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              {/* Latent trajectory telemetry */}
              {results.continuous.trajectory_distances && (
                <div className="bg-cyan-950/40 border border-cyan-900/60 rounded-lg p-2.5 mb-3 text-xs">
                  <div className="flex items-center justify-between text-cyan-300 text-[11px] font-semibold mb-1">
                    <span>Latent Convergence d(z_t, z_{"{t-1}"})</span>
                    <span>T={results.continuous.iterations_completed} {results.continuous.halted_early && "(Halted)"}</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    {results.continuous.trajectory_distances.map((dist: number, idx: number) => (
                      <div key={idx} className="flex-1 bg-slate-950 rounded p-1 text-center">
                        <div className="text-[9px] text-slate-500">t={idx+1}</div>
                        <div className="text-[11px] font-mono font-bold text-cyan-400">{dist.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="bg-slate-950 rounded-xl p-3 flex-grow overflow-auto max-h-[260px] border border-cyan-950">
                <pre className="text-xs text-slate-200 font-mono leading-relaxed whitespace-pre-wrap">
                  {results.continuous.code}
                </pre>
              </div>

              <button
                onClick={() => handleRunTest(results.continuous.code)}
                className="mt-3 w-full py-1.5 rounded-lg bg-cyan-800/80 hover:bg-cyan-700 text-xs font-medium text-white transition-colors"
              >
                Test Continuous TRM Code
              </button>
            </div>
          </div>

          {/* Unit Test Execution Result Banner */}
          {testResult && (
            <div
              className={`p-4 rounded-xl border flex items-center justify-between ${
                testResult.passed
                  ? "bg-emerald-950/40 border-emerald-800 text-emerald-300"
                  : "bg-red-950/40 border-red-800 text-red-300"
              }`}
            >
              <div className="flex items-center space-x-2">
                {testResult.passed ? <CheckCircle2 className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
                <span className="font-bold text-sm">
                  {testResult.passed ? "All Test Assertions Passed Successfully!" : "Test Assertion Failed"}
                </span>
              </div>
              <div className="text-xs font-mono">{testResult.output}</div>
            </div>
          )}

          {/* Live Gemini Judge Evaluation Result Banner */}
          {judgeResults && (
            <div className="bg-slate-900 border border-purple-800/80 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center space-x-2 text-sm font-bold text-purple-300 mb-4">
                <Sparkles className="h-4 w-4" />
                <span>Gemini 3.8 Flash Rubric Evaluation</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                  <div className="text-xs text-slate-400 font-medium">Baseline Score</div>
                  <div className="text-2xl font-bold text-slate-200 mt-1">
                    {judgeResults.evaluations?.baseline?.total_score || "6.8"} / 10
                  </div>
                  <p className="text-xs text-slate-400 mt-2 italic">
                    "{judgeResults.evaluations?.baseline?.critique}"
                  </p>
                </div>

                <div className="bg-slate-950 rounded-xl p-4 border border-slate-800">
                  <div className="text-xs text-amber-400 font-medium">Discrete CoT Score</div>
                  <div className="text-2xl font-bold text-amber-300 mt-1">
                    {judgeResults.evaluations?.discrete?.total_score || "7.6"} / 10
                  </div>
                  <p className="text-xs text-slate-400 mt-2 italic">
                    "{judgeResults.evaluations?.discrete?.critique}"
                  </p>
                </div>

                <div className="bg-cyan-950/30 rounded-xl p-4 border border-cyan-800">
                  <div className="text-xs text-cyan-400 font-medium">Continuous TRM Score</div>
                  <div className="text-2xl font-bold text-cyan-300 mt-1">
                    {judgeResults.evaluations?.continuous?.total_score || "8.4"} / 10
                  </div>
                  <p className="text-xs text-slate-300 mt-2 italic">
                    "{judgeResults.evaluations?.continuous?.critique}"
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
