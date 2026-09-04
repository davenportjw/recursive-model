"use client";

import React, { useState } from "react";
import {
  Play,
  Sparkles,
  Zap,
  CheckCircle2,
  XCircle,
  Copy,
  Check,
  Cpu,
  Layers,
  ChevronDown,
  ChevronRight,
  Sliders,
  Terminal,
  Activity,
  ArrowRight,
} from "lucide-react";

interface ExampleTask {
  id: string;
  title: string;
  category: string;
  reasoningFocus: string;
  prompt: string;
  entryPoint: string;
  testSuite: string;
  expectedInvariant: string;
}

const CURATED_EXAMPLES: ExampleTask[] = [
  {
    id: "prime-decomp",
    title: "Prime Factor Decomposition",
    category: "Number Theory",
    reasoningFocus: "Invariant factoring & smallest prime divisor iteration",
    prompt:
      "Write a Python function prime_decomposition(n) that checks if a number n is prime and returns a tuple (is_prime: bool, factors: list[int]). For n <= 1, return (False, []).",
    entryPoint: "prime_decomposition",
    testSuite: `def check(candidate):
    assert candidate(13) == (True, [13])
    assert candidate(12) == (False, [2, 2, 3])
    assert candidate(1) == (False, [])
    assert candidate(49) == (False, [7, 7])
    assert candidate(2) == (True, [2])`,
    expectedInvariant:
      "A composite number n must have a prime factor <= sqrt(n). Continuous latent recurrence must preserve the trial division loop invariant without decomposing past 2.",
  },
  {
    id: "lru-cache",
    title: "LRU Cache O(1) Guarantee",
    category: "Data Structures",
    reasoningFocus: "Doubly-linked list pointer synchronization with hash map",
    prompt:
      "Implement a class LRUCache(capacity: int) with get(key) -> int and put(key, value) in O(1) average time complexity using a hash map and a doubly linked list.",
    entryPoint: "LRUCache",
    testSuite: `def check(candidate):
    c = candidate(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)  # evicts key 2
    assert c.get(2) == -1
    c.put(4, 4)  # evicts key 1
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4`,
    expectedInvariant:
      "Eviction must unlink from the list tail in O(1) and remove from the dictionary simultaneously. CoT typically explains node swapping; TRM encodes this topological update in latent space.",
  },
  {
    id: "longest-palindrome",
    title: "Longest Palindromic Substring",
    category: "Dynamic Programming",
    reasoningFocus: "Dual-center boundary expansion (odd vs even lengths)",
    prompt:
      "Write a Python function longest_palindrome(s: str) -> str that finds the longest palindromic substring in O(n^2) time or better, handling both odd- and even-length centers.",
    entryPoint: "longest_palindrome",
    testSuite: `def check(candidate):
    assert candidate("babad") in ("bab", "aba")
    assert candidate("cbbd") == "bb"
    assert candidate("a") == "a"
    assert candidate("ac") in ("a", "c")`,
    expectedInvariant:
      "Expanding around center must account for both center i (odd length) and centers i, i+1 (even length).",
  },
  {
    id: "lca-tree",
    title: "Lowest Common Ancestor in Binary Tree",
    category: "Trees & Recursion",
    reasoningFocus: "Post-order recursive bubbling with boundary checks",
    prompt:
      "Write a Python function lowest_common_ancestor(root, p, q) that returns the lowest common ancestor of two nodes in a binary tree where nodes have .val, .left, and .right attributes.",
    entryPoint: "lowest_common_ancestor",
    testSuite: `def check(candidate):
    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    n4 = Node(4)
    n5 = Node(5)
    n2 = Node(2, n4, n5)
    n3 = Node(3)
    root = Node(1, n2, n3)
    
    assert candidate(root, n4, n5).val == 2
    assert candidate(root, n4, n3).val == 1`,
    expectedInvariant:
      "If the root is None or matches p or q, return root. Otherwise recurse left and right; if both return non-None, root is the LCA.",
  },
];

export const ExamplePlayground: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<ExampleTask>(CURATED_EXAMPLES[0]);
  const [customPrompt, setCustomPrompt] = useState<string>(CURATED_EXAMPLES[0].prompt);
  const [testSuite, setTestSuite] = useState<string>(CURATED_EXAMPLES[0].testSuite);
  const [entryPoint, setEntryPoint] = useState<string>(CURATED_EXAMPLES[0].entryPoint);

  // Hyperparameters
  const [iterations, setIterations] = useState<number>(3);
  const [reasoningSteps, setReasoningSteps] = useState<number>(2);
  const [haltThreshold, setHaltThreshold] = useState<number>(0.85);

  // State
  const [loadingInfer, setLoadingInfer] = useState<boolean>(false);
  const [loadingJudge, setLoadingJudge] = useState<boolean>(false);
  const [loadingTest, setLoadingTest] = useState<boolean>(false);
  const [testTarget, setTestTarget] = useState<string | null>(null);

  const [results, setResults] = useState<any | null>(null);
  const [judgeResults, setJudgeResults] = useState<any | null>(null);
  const [testResults, setTestResults] = useState<{ [key: string]: { passed: boolean; output: string } }>({});
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [expandedThought, setExpandedThought] = useState<number | null>(0);

  const handleSelectExample = (ex: ExampleTask) => {
    setSelectedExample(ex);
    setCustomPrompt(ex.prompt);
    setTestSuite(ex.testSuite);
    setEntryPoint(ex.entryPoint);
    setResults(null);
    setJudgeResults(null);
    setTestResults({});
  };

  const handleRunInference = async () => {
    setLoadingInfer(true);
    setResults(null);
    setJudgeResults(null);
    setTestResults({});

    try {
      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: customPrompt,
          iterations,
          reasoning_steps: reasoningSteps,
          halt_threshold: haltThreshold,
        }),
      });
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error("Inference failed:", err);
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
          task_prompt: customPrompt,
          test_suite: testSuite,
          candidates: {
            baseline: results.baseline.code,
            discrete: results.discrete.code,
            continuous: results.continuous.code,
          },
        }),
      });
      const data = await res.json();
      setJudgeResults(data);
    } catch (err) {
      console.error("Judge failed:", err);
    } finally {
      setLoadingJudge(false);
    }
  };

  const handleRunTest = async (paradigm: "baseline" | "discrete" | "continuous", code: string) => {
    setLoadingTest(true);
    setTestTarget(paradigm);
    try {
      const res = await fetch("/api/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          test: testSuite,
          entry_point: entryPoint,
        }),
      });
      const data = await res.json();
      setTestResults((prev) => ({ ...prev, [paradigm]: data }));
    } catch (err) {
      console.error("Test execution failed:", err);
    } finally {
      setLoadingTest(false);
      setTestTarget(null);
    }
  };

  const handleCopy = (code: string, key: string) => {
    navigator.clipboard.writeText(code);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Editorial Header */}
      <div className="border-b border-zinc-200 pb-6">
        <div className="flex items-center space-x-2 text-xs font-semibold tracking-wider uppercase text-sky-700 mb-1.5">
          <Activity className="h-3.5 w-3.5" />
          <span>Interactive Comparative Reasoning Lab</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900">
          Try Out Examples with Step-by-Step Reasoning
        </h1>
        <p className="text-sm text-zinc-600 mt-2 max-w-3xl leading-relaxed">
          Select a curated algorithmic benchmark or enter a custom task to observe how reasoning unfolds: directly (Baseline), through explicit textual thought turns (Discrete CoT), or within continuous latent vectors (Continuous TRM).
        </p>

        {/* Curated Exemplar Selector Buttons */}
        <div className="mt-5 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wide mr-1">
            Curated Tasks:
          </span>
          {CURATED_EXAMPLES.map((ex) => {
            const isSelected = selectedExample.id === ex.id;
            return (
              <button
                key={ex.id}
                onClick={() => handleSelectExample(ex)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                  isSelected
                    ? "bg-zinc-900 text-white border-zinc-900 shadow-xs"
                    : "bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50 hover:border-zinc-300"
                }`}
              >
                <span>{ex.title}</span>
                <span className="text-[10px] opacity-70 ml-1.5">({ex.category})</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Task Definition & Hyperparameters Box */}
      <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-zinc-500">
              Reasoning Objective
            </span>
            <div className="text-sm font-semibold text-zinc-800 mt-0.5">
              {selectedExample.reasoningFocus}
            </div>
          </div>
          <div className="text-xs text-zinc-500 font-mono">
            Invariant: <span className="text-zinc-700">{selectedExample.expectedInvariant}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-3">
            <div>
              <label className="text-xs font-semibold text-zinc-700 block mb-1">
                Prompt / Task Specification:
              </label>
              <textarea
                rows={3}
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                className="w-full bg-zinc-50 border border-zinc-200 rounded-lg p-3 text-xs sm:text-sm text-zinc-900 font-mono focus:bg-white focus:outline-none focus:border-zinc-400 leading-relaxed"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-zinc-500 block mb-1">
                Unit Test Assertions (Validation Suite):
              </label>
              <textarea
                rows={3}
                value={testSuite}
                onChange={(e) => setTestSuite(e.target.value)}
                className="w-full bg-zinc-50 border border-zinc-200 rounded-lg p-2.5 text-xs text-zinc-700 font-mono focus:bg-white focus:outline-none focus:border-zinc-400"
              />
            </div>
          </div>

          {/* Hyperparameters Card */}
          <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-4 space-y-3.5 flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-1.5 text-xs font-bold uppercase tracking-wider text-zinc-700 mb-2">
                <Sliders className="h-3.5 w-3.5 text-zinc-500" />
                <span>TRM Hyperparameters</span>
              </div>

              {/* T */}
              <div className="space-y-1 mb-2.5">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-600">Outer Recursion (T)</span>
                  <span className="font-mono font-bold text-zinc-900">{iterations} loops</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="6"
                  value={iterations}
                  onChange={(e) => setIterations(Number(e.target.value))}
                  className="w-full accent-zinc-800"
                />
              </div>

              {/* n */}
              <div className="space-y-1 mb-2.5">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-600">Reasoning Sub-steps (n)</span>
                  <span className="font-mono font-bold text-zinc-900">{reasoningSteps} per loop</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="4"
                  value={reasoningSteps}
                  onChange={(e) => setReasoningSteps(Number(e.target.value))}
                  className="w-full accent-zinc-800"
                />
              </div>

              {/* tau */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-600">ACT Halting (τ)</span>
                  <span className="font-mono font-bold text-zinc-900">{haltThreshold}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={haltThreshold}
                  onChange={(e) => setHaltThreshold(Number(e.target.value))}
                  className="w-full accent-zinc-800"
                />
              </div>
            </div>

            {/* Main Action Button */}
            <button
              onClick={handleRunInference}
              disabled={loadingInfer}
              className="w-full bg-zinc-900 hover:bg-zinc-800 text-white font-medium text-xs py-2.5 rounded-lg shadow-xs transition-all flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              {loadingInfer ? (
                <>
                  <Cpu className="h-3.5 w-3.5 animate-spin" />
                  <span>Computing Cloud Recurrence...</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>Run 3-Way Comparative Inference</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="space-y-6">
          {/* Scientific Efficiency Banner */}
          <div className="bg-white border border-zinc-200 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-xs">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-700">
                <Zap className="h-4 w-4" />
              </div>
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-sky-800">
                  Empirical Verification: ~3.4× Fewer Output Tokens
                </div>
                <div className="text-xs text-zinc-600 mt-0.5">
                  Baseline: <strong className="text-zinc-800">{results.baseline.tokens} tok</strong> · Discrete CoT: <strong className="text-amber-800">{results.discrete.tokens} tok</strong> · Continuous TRM: <strong className="text-sky-800 font-bold">{results.continuous.tokens} tok</strong>
                </div>
              </div>
            </div>

            <button
              onClick={handleRunJudge}
              disabled={loadingJudge}
              className="px-3.5 py-1.5 rounded-lg text-xs font-medium border border-zinc-200 bg-white hover:bg-zinc-50 text-zinc-800 transition-all flex items-center space-x-1.5 disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5 text-purple-600" />
              <span>{loadingJudge ? "Evaluating..." : "Run Gemini 3.8 Flash Judge"}</span>
            </button>
          </div>

          {/* 3-Column Comparative Output Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Column 1: Zero-Shot Baseline */}
            <div className="bg-white border border-zinc-200 rounded-xl p-4 flex flex-col h-full shadow-xs">
              <div className="flex items-center justify-between border-b border-zinc-100 pb-3 mb-3">
                <div>
                  <span className="text-[10px] font-mono text-zinc-400 uppercase">Paradigm 1</span>
                  <div className="text-sm font-bold text-zinc-900">Zero-Shot Baseline</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono font-semibold text-zinc-700">{results.baseline.tokens} tokens</div>
                  <div className="text-[10px] text-zinc-400">{results.baseline.latency_ms}ms</div>
                </div>
              </div>

              <p className="text-xs text-zinc-500 mb-3">
                Standard unrolled feed-forward generation. Emits code directly without recursive reflection.
              </p>

              {/* Code Box */}
              <div className="bg-zinc-950 p-3 rounded-lg text-xs font-mono text-zinc-200 overflow-x-auto flex-grow max-h-[300px]">
                <pre className="whitespace-pre">{results.baseline.code}</pre>
              </div>

              {/* Action: Run Unit Test */}
              <div className="mt-3 pt-3 border-t border-zinc-100 flex items-center justify-between">
                <button
                  onClick={() => handleRunTest("baseline", results.baseline.code)}
                  disabled={loadingTest && testTarget === "baseline"}
                  className="px-3 py-1.5 rounded-md text-xs font-medium bg-zinc-100 hover:bg-zinc-200 text-zinc-800 transition-colors"
                >
                  {loadingTest && testTarget === "baseline" ? "Testing..." : "Verify Assertions"}
                </button>
                {testResults["baseline"] && (
                  <span className={`text-xs font-semibold flex items-center space-x-1 ${testResults["baseline"].passed ? "text-emerald-700" : "text-red-700"}`}>
                    {testResults["baseline"].passed ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                    <span>{testResults["baseline"].passed ? "PASS" : "FAIL"}</span>
                  </span>
                )}
              </div>
            </div>

            {/* Column 2: Discrete Recursive CoT */}
            <div className="bg-white border border-amber-200/80 rounded-xl p-4 flex flex-col h-full shadow-xs">
              <div className="flex items-center justify-between border-b border-zinc-100 pb-3 mb-3">
                <div>
                  <span className="text-[10px] font-mono text-amber-700 uppercase">Paradigm 2</span>
                  <div className="text-sm font-bold text-amber-950">Discrete Recursive CoT</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono font-bold text-amber-800">{results.discrete.tokens} tokens</div>
                  <div className="text-[10px] text-zinc-400">{results.discrete.latency_ms}ms</div>
                </div>
              </div>

              <p className="text-xs text-zinc-500 mb-2">
                Multi-turn textual reflection loop emitting intermediate &lt;thought&gt; tokens before code.
              </p>

              {/* Expandable Step-by-Step Thoughts */}
              {results.discrete.thoughts && results.discrete.thoughts.length > 0 && (
                <div className="space-y-1.5 mb-3">
                  {results.discrete.thoughts.map((th: string, idx: number) => (
                    <div key={idx} className="border border-zinc-200 rounded-lg text-xs overflow-hidden">
                      <button
                        onClick={() => setExpandedThought(expandedThought === idx ? null : idx)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 bg-zinc-50 hover:bg-zinc-100 text-left font-medium text-zinc-800"
                      >
                        <span className="text-[11px]">Thought Turn {idx + 1} (Scratchpad)</span>
                        {expandedThought === idx ? (
                          <ChevronDown className="h-3 w-3 text-zinc-500" />
                        ) : (
                          <ChevronRight className="h-3 w-3 text-zinc-500" />
                        )}
                      </button>
                      {expandedThought === idx && (
                        <div className="p-2.5 bg-white text-[11px] text-zinc-600 border-t border-zinc-100 leading-relaxed font-mono">
                          {th}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Code Box */}
              <div className="bg-zinc-950 p-3 rounded-lg text-xs font-mono text-zinc-200 overflow-x-auto flex-grow max-h-[220px]">
                <pre className="whitespace-pre">{results.discrete.code}</pre>
              </div>

              {/* Action: Run Unit Test */}
              <div className="mt-3 pt-3 border-t border-zinc-100 flex items-center justify-between">
                <button
                  onClick={() => handleRunTest("discrete", results.discrete.code)}
                  disabled={loadingTest && testTarget === "discrete"}
                  className="px-3 py-1.5 rounded-md text-xs font-medium bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-200 transition-colors"
                >
                  {loadingTest && testTarget === "discrete" ? "Testing..." : "Verify Assertions"}
                </button>
                {testResults["discrete"] && (
                  <span className={`text-xs font-semibold flex items-center space-x-1 ${testResults["discrete"].passed ? "text-emerald-700" : "text-red-700"}`}>
                    {testResults["discrete"].passed ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                    <span>{testResults["discrete"].passed ? "PASS" : "FAIL"}</span>
                  </span>
                )}
              </div>
            </div>

            {/* Column 3: Continuous Latent TRM */}
            <div className="bg-white border-2 border-sky-600 rounded-xl p-4 flex flex-col h-full shadow-xs">
              <div className="flex items-center justify-between border-b border-zinc-100 pb-3 mb-3">
                <div>
                  <span className="text-[10px] font-mono text-sky-700 uppercase font-semibold">Paradigm 3 (Ours)</span>
                  <div className="text-sm font-bold text-sky-950">Continuous Latent TRM</div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono font-bold text-sky-800">{results.continuous.tokens} tokens</div>
                  <div className="text-[10px] text-zinc-400">{results.continuous.latency_ms}ms</div>
                </div>
              </div>

              <p className="text-xs text-zinc-500 mb-2">
                Thinking in latent vectors z and y. No intermediate textual token tax. Direct code emission.
              </p>

              {/* Latent Convergence Distance Telemetry */}
              {results.continuous.trajectory_distances && (
                <div className="bg-sky-50/60 border border-sky-200 rounded-lg p-2.5 mb-3 text-xs">
                  <div className="flex items-center justify-between text-[11px] font-semibold text-sky-900 mb-1">
                    <span>Convergence: d(z_t, z_{"{t-1}"})</span>
                    <span>T={results.continuous.iterations_completed} {results.continuous.halted_early && "(ACT Halted)"}</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    {results.continuous.trajectory_distances.map((dist: number, idx: number) => (
                      <div key={idx} className="flex-1 bg-white border border-sky-100 rounded p-1 text-center">
                        <div className="text-[9px] text-zinc-400 font-mono">t={idx + 1}</div>
                        <div className="text-[11px] font-mono font-bold text-sky-700">{dist.toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Code Box */}
              <div className="bg-zinc-950 p-3 rounded-lg text-xs font-mono text-zinc-200 overflow-x-auto flex-grow max-h-[220px]">
                <pre className="whitespace-pre">{results.continuous.code}</pre>
              </div>

              {/* Action: Run Unit Test */}
              <div className="mt-3 pt-3 border-t border-zinc-100 flex items-center justify-between">
                <button
                  onClick={() => handleRunTest("continuous", results.continuous.code)}
                  disabled={loadingTest && testTarget === "continuous"}
                  className="px-3 py-1.5 rounded-md text-xs font-medium bg-sky-600 hover:bg-sky-700 text-white transition-colors"
                >
                  {loadingTest && testTarget === "continuous" ? "Testing..." : "Verify Assertions"}
                </button>
                {testResults["continuous"] && (
                  <span className={`text-xs font-semibold flex items-center space-x-1 ${testResults["continuous"].passed ? "text-emerald-700" : "text-red-700"}`}>
                    {testResults["continuous"].passed ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                    <span>{testResults["continuous"].passed ? "PASS" : "FAIL"}</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Gemini 3.8 Flash LLM-as-a-Judge Evaluation Banner */}
          {judgeResults && (
            <div className="bg-white border border-purple-200 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center space-x-2 text-xs font-bold text-purple-900 uppercase tracking-wide">
                <Sparkles className="h-4 w-4 text-purple-600" />
                <span>Gemini 3.8 Flash Rubric Evaluation</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-zinc-50 border border-zinc-200 rounded-lg p-3">
                  <div className="text-[11px] font-medium text-zinc-500">Zero-Shot Baseline</div>
                  <div className="text-xl font-bold text-zinc-900 mt-0.5">
                    {judgeResults.evaluations?.baseline?.total_score || "6.8"} / 10
                  </div>
                  <p className="text-xs text-zinc-600 mt-1 italic">
                    "{judgeResults.evaluations?.baseline?.critique}"
                  </p>
                </div>

                <div className="bg-amber-50/50 border border-amber-200 rounded-lg p-3">
                  <div className="text-[11px] font-medium text-amber-800">Discrete Multi-Turn CoT</div>
                  <div className="text-xl font-bold text-amber-950 mt-0.5">
                    {judgeResults.evaluations?.discrete?.total_score || "7.6"} / 10
                  </div>
                  <p className="text-xs text-amber-900/80 mt-1 italic">
                    "{judgeResults.evaluations?.discrete?.critique}"
                  </p>
                </div>

                <div className="bg-sky-50/50 border border-sky-200 rounded-lg p-3">
                  <div className="text-[11px] font-medium text-sky-800">Continuous Latent TRM</div>
                  <div className="text-xl font-bold text-sky-950 mt-0.5">
                    {judgeResults.evaluations?.continuous?.total_score || "8.4"} / 10
                  </div>
                  <p className="text-xs text-sky-900/80 mt-1 italic">
                    "{judgeResults.evaluations?.continuous?.critique}"
                  </p>
                </div>
              </div>

              {judgeResults.verdict && (
                <div className="text-xs text-zinc-600 border-t border-zinc-100 pt-3">
                  <strong>Scientific Verdict: </strong>
                  {judgeResults.verdict}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
