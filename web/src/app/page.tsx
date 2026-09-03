"use client";

import React, { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { BenchmarkExplorer } from "@/components/BenchmarkExplorer";
import { ResearchDashboard } from "@/components/ResearchDashboard";
import { LivePlayground } from "@/components/LivePlayground";
import { ArchitectureExplainer } from "@/components/ArchitectureExplainer";
import { BenchmarkTask } from "@/lib/types";
import { SAMPLE_TASKS } from "@/lib/benchmarkData";
import { Activity, BookOpen, Layers, Terminal, Sparkles, ExternalLink } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"option-a" | "option-b" | "deep-dive">("option-a");
  const [optionASubTab, setOptionASubTab] = useState<"explorer" | "findings">("explorer");
  const [tasks, setTasks] = useState<BenchmarkTask[]>(SAMPLE_TASKS);

  useEffect(() => {
    async function fetchTasks() {
      try {
        const res = await fetch("/api/benchmarks");
        const data = await res.json();
        if (data.tasks && data.tasks.length > 0) {
          setTasks(data.tasks);
        }
      } catch (err) {
        console.warn("Using sample tasks due to fetch error:", err);
      }
    }
    fetchTasks();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-between">
      <div>
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
          {/* Option A View: Benchmark Explorer + Research Dashboard */}
          {activeTab === "option-a" && (
            <div className="space-y-6">
              {/* Option A Sub-Navigation */}
              <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
                <button
                  onClick={() => setOptionASubTab("explorer")}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                    optionASubTab === "explorer"
                      ? "bg-slate-800 text-cyan-400 border border-slate-700 shadow-md"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Layers className="h-4 w-4" />
                  <span>Benchmark Explorer (3-Way Comparison)</span>
                </button>

                <button
                  onClick={() => setOptionASubTab("findings")}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                    optionASubTab === "findings"
                      ? "bg-slate-800 text-cyan-400 border border-slate-700 shadow-md"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Activity className="h-4 w-4" />
                  <span>Digestible Research Findings</span>
                </button>
              </div>

              {optionASubTab === "explorer" ? (
                <BenchmarkExplorer tasks={tasks} />
              ) : (
                <ResearchDashboard />
              )}
            </div>
          )}

          {/* Option B View: Live Try-It-Out Playground */}
          {activeTab === "option-b" && <LivePlayground />}

          {/* Deep Dive: TRM vs DiffusionGemma View */}
          {activeTab === "deep-dive" && <ArchitectureExplainer />}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/60 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <span>Tiny Recursive Gemma</span>
            <span>·</span>
            <span>Based on Samsung SAIL Montréal (arXiv:2510.04871)</span>
            <span>·</span>
            <span>Google Gemma 4 2B Architecture</span>
          </div>

          <div className="flex items-center space-x-4">
            <a
              href="https://arxiv.org/abs/2510.04871"
              target="_blank"
              rel="noreferrer"
              className="hover:text-slate-300 flex items-center space-x-1 transition-colors"
            >
              <span>TRM Paper</span>
              <ExternalLink className="h-3 w-3" />
            </a>
            <a
              href="https://developers.googleblog.com/diffusiongemma-the-developer-guide/"
              target="_blank"
              rel="noreferrer"
              className="hover:text-slate-300 flex items-center space-x-1 transition-colors"
            >
              <span>DiffusionGemma</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
