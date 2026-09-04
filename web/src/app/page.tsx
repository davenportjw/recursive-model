"use client";

import React, { useState } from "react";
import { Navbar, NavTab } from "@/components/Navbar";
import { ArchitectureStepper } from "@/components/ArchitectureStepper";
import { ExamplePlayground } from "@/components/ExamplePlayground";
import { BenchmarkStory } from "@/components/BenchmarkStory";
import { CloudSystem } from "@/components/CloudSystem";
import { ExternalLink, BookOpen, GitFork } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<NavTab>("architecture");

  return (
    <div className="min-h-screen bg-[#fbfbfd] text-zinc-900 flex flex-col justify-between">
      <div>
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
          {/* 1. Architecture & Code Walkthrough */}
          {activeTab === "architecture" && <ArchitectureStepper />}

          {/* 2. Interactive Examples & Reasoning Lab */}
          {activeTab === "examples" && <ExamplePlayground />}

          {/* 3. Empirical Results (200-Task Suite) */}
          {activeTab === "benchmarks" && <BenchmarkStory />}

          {/* 4. Cloud Infrastructure & Vertex AI Dispatch */}
          {activeTab === "cloud" && <CloudSystem />}
        </main>
      </div>

      {/* Academic Paper Footer */}
      <footer className="border-t border-zinc-200 bg-white py-6 mt-16 text-xs text-zinc-500">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-semibold text-zinc-700">Tiny Recursive Gemma</span>
            <span>·</span>
            <span>Samsung SAIL Montréal (arXiv:2510.04871)</span>
            <span>·</span>
            <span>Google Gemma 4 2B Architecture</span>
          </div>

          <div className="flex items-center space-x-4">
            <a
              href="https://arxiv.org/abs/2510.04871"
              target="_blank"
              rel="noreferrer"
              className="hover:text-zinc-900 flex items-center space-x-1 transition-colors"
            >
              <BookOpen className="h-3 w-3" />
              <span>arXiv:2510.04871</span>
              <ExternalLink className="h-2.5 w-2.5 opacity-60" />
            </a>
            <a
              href="https://github.com/davenportjw/recursive-model"
              target="_blank"
              rel="noreferrer"
              className="hover:text-zinc-900 flex items-center space-x-1 transition-colors"
            >
              <GitFork className="h-3 w-3" />
              <span>Repository</span>
              <ExternalLink className="h-2.5 w-2.5 opacity-60" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
