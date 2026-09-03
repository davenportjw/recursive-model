"use client";

import React from "react";
import { Sparkles, Terminal, Activity, BookOpen, Cloud, Cpu, ArrowUpRight } from "lucide-react";

interface NavbarProps {
  activeTab: "option-a" | "option-b" | "deep-dive";
  setActiveTab: (tab: "option-a" | "option-b" | "deep-dive") => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Cpu className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-base tracking-tight text-white">Tiny Recursive Gemma</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800">
                  Samsung TRM on Gemma 2B
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Continuous Latent Reasoning vs Discrete Multi-Turn CoT
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab("option-a")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "option-a"
                  ? "bg-cyan-600 text-white shadow-md shadow-cyan-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Activity className="h-4 w-4" />
              <span>Option A: Benchmark & Research</span>
            </button>

            <button
              onClick={() => setActiveTab("option-b")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "option-b"
                  ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Terminal className="h-4 w-4" />
              <span>Option B: Try It Out (Cloud)</span>
            </button>

            <button
              onClick={() => setActiveTab("deep-dive")}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "deep-dive"
                  ? "bg-slate-700 text-white shadow-md"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <BookOpen className="h-4 w-4" />
              <span>Architecture &amp; Docs</span>
            </button>
          </nav>

          {/* Cloud Status Indicator */}
          <div className="hidden lg:flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-300">
              <Cloud className="h-3.5 w-3.5 text-emerald-400 animate-pulse" />
              <span>Google Cloud Run: Active</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
