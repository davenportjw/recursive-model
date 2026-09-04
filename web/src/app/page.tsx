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
    <div className="min-h-screen bg-[#FAF9F5] text-[#141413] flex flex-col justify-between">
      <div>
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="max-w-5xl mx-auto px-4 sm:px-6 pt-20 pb-16">
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

      {/* Editorial Research Footer */}
      <footer className="border-t border-[#E8E5DF] bg-[#FAF9F5] py-8 text-xs text-[#8C887B]">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="font-editorial text-sm font-medium text-[#141413]">
              Tiny Recursive Gemma
            </span>
            <span>·</span>
            <span>Samsung SAIL Montréal (arXiv:2510.04871)</span>
            <span>·</span>
            <span className="font-mono text-[11px]">Gemma 4 2B Backbone</span>
          </div>

          <div className="flex items-center space-x-5">
            <a
              href="https://arxiv.org/abs/2510.04871"
              target="_blank"
              rel="noreferrer"
              className="hover:text-[#141413] flex items-center space-x-1.5 transition-colors"
            >
              <BookOpen className="h-3 w-3" />
              <span>arXiv:2510.04871</span>
              <ExternalLink className="h-2.5 w-2.5 opacity-60" />
            </a>
            <a
              href="https://github.com/davenportjw/recursive-model"
              target="_blank"
              rel="noreferrer"
              className="hover:text-[#141413] flex items-center space-x-1.5 transition-colors"
            >
              <GitFork className="h-3 w-3" />
              <span>Source</span>
              <ExternalLink className="h-2.5 w-2.5 opacity-60" />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
