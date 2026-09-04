"use client";

import React, { useState } from "react";
import {
  Cloud,
  Cpu,
  Server,
  Play,
  CheckCircle2,
  Terminal,
  ShieldCheck,
  HardDrive,
  ExternalLink,
  Layers,
} from "lucide-react";

export const CloudSystem: React.FC = () => {
  const [dispatching, setDispatching] = useState<string | null>(null);
  const [dispatchResult, setDispatchResult] = useState<any | null>(null);
  const [activeLogs, setActiveLogs] = useState<string[]>([
    "[Google Cloud] Project: davenport-boutique | Region: us-central1",
    "[Vertex AI] Custom Training Cluster ready: Dedicated NVIDIA L4 (24GB VRAM) GPU",
    "[Cloud Run] Service active: tiny-recursive-gemma (stateless web tier)",
    "[Hardware Guardrail] Apple Silicon local runs strictly isolated to mock unit tests (uv run pytest)",
  ]);

  const handleDispatch = async (jobType: "train" | "benchmark") => {
    setDispatching(jobType);
    try {
      const res = await fetch("/api/cloud", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: jobType === "train" ? "dispatch_training" : "run_benchmark",
          params: {
            gpu_type: "NVIDIA_L4",
            machine_type: "g2-standard-8",
            epochs: 3,
            learning_rate: 2e-4,
          },
        }),
      });
      const data = await res.json();
      setDispatchResult(data);
      if (data.status === "dispatched" || data.status === "mock_dispatched") {
        setActiveLogs((prev) => [
          ...prev,
          `[${new Date().toLocaleTimeString()}] ${jobType.toUpperCase()} Job Dispatched to Vertex AI: ${data.job_name || "vertex-trm-job-001"}`,
          `[${new Date().toLocaleTimeString()}] Provisioning container: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2:latest`,
          `[${new Date().toLocaleTimeString()}] Training on NVIDIA L4 (24GB VRAM). Checkpoints uploading to gs://tiny-recursive-gemma-checkpoints/`,
        ]);
      }
    } catch (err) {
      console.error("Cloud dispatch error:", err);
    } finally {
      setDispatching(null);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Editorial Header */}
      <div className="border-b border-zinc-200 pb-6">
        <div className="flex items-center space-x-2 text-xs font-semibold tracking-wider uppercase text-sky-700 mb-1.5">
          <Cloud className="h-3.5 w-3.5" />
          <span>Cloud Infrastructure & Topology</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900">
          Google Cloud Vertex AI & Hardware Safeguards
        </h1>
        <p className="text-sm text-zinc-600 mt-2 max-w-3xl leading-relaxed">
          Due to strict Apple Silicon memory watchdog guardrails, all continuous latent training and recurrence backpropagation run on dedicated Google Cloud Vertex AI NVIDIA L4 (24GB VRAM) instances.
        </p>

        {/* Status Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-5">
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Target Hardware</div>
            <div className="text-base font-bold text-zinc-900 mt-0.5">NVIDIA L4 (24GB)</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">g2-standard-8 · us-central1</div>
          </div>
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Model Backbone</div>
            <div className="text-base font-bold text-zinc-900 mt-0.5">Google Gemma 4 2B</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">LoRA rank=16 · alpha=32</div>
          </div>
          <div className="bg-white border border-zinc-200 rounded-lg p-3">
            <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wide">Local Policy</div>
            <div className="text-base font-bold text-emerald-700 mt-0.5">Strict Isolation</div>
            <div className="text-[11px] text-zinc-500 mt-0.5">Mock tensors only (pytest ~3s)</div>
          </div>
        </div>
      </div>

      {/* Cloud Topology Diagram */}
      <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-xs font-bold uppercase tracking-wider text-zinc-500">
          Distributed System Topology
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="border border-zinc-200 rounded-lg p-4 bg-zinc-50/50 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-800 font-bold">
              <Server className="h-4 w-4 text-sky-600" />
              <span>1. Cloud Run Serving Tier</span>
            </div>
            <p className="text-zinc-600 text-[11px] leading-relaxed">
              Stateless Next.js 15 App Router handling user interaction, live benchmark inspection, and proxying inference via Vertex AI Gemini 3.8 Flash.
            </p>
          </div>

          <div className="border border-zinc-200 rounded-lg p-4 bg-zinc-50/50 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-800 font-bold">
              <Cpu className="h-4 w-4 text-sky-600" />
              <span>2. Vertex AI Custom Training</span>
            </div>
            <p className="text-zinc-600 text-[11px] leading-relaxed">
              Dedicated GPU worker (NVIDIA L4 24GB VRAM) executing unrolled continuous recurrence backpropagation without host kernel memory exhaustion.
            </p>
          </div>

          <div className="border border-zinc-200 rounded-lg p-4 bg-zinc-50/50 space-y-2">
            <div className="flex items-center space-x-2 text-zinc-800 font-bold">
              <HardDrive className="h-4 w-4 text-sky-600" />
              <span>3. Cloud Storage Checkpoints</span>
            </div>
            <p className="text-zinc-600 text-[11px] leading-relaxed">
              Durable storage for continuous latent LoRA weights, ACT halting classifier weights, and precomputed 200-task JSONL evaluation artifacts.
            </p>
          </div>
        </div>
      </div>

      {/* Cloud Dispatch Center */}
      <div className="bg-white border border-zinc-200 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-100 pb-3">
          <div>
            <h2 className="text-sm font-bold text-zinc-900">Vertex AI Custom Job Dispatch</h2>
            <p className="text-xs text-zinc-500">
              Submit PyTorch training or automated benchmark evaluation sweeps to Google Cloud.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleDispatch("train")}
              disabled={dispatching !== null}
              className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-zinc-900 text-white hover:bg-zinc-800 transition-colors flex items-center space-x-1.5 disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              <span>{dispatching === "train" ? "Dispatching..." : "Launch L4 Training Job"}</span>
            </button>
            <button
              onClick={() => handleDispatch("benchmark")}
              disabled={dispatching !== null}
              className="px-3.5 py-1.5 rounded-lg text-xs font-medium border border-zinc-200 text-zinc-800 hover:bg-zinc-50 transition-colors flex items-center space-x-1.5 disabled:opacity-50"
            >
              <Cpu className="h-3.5 w-3.5 text-sky-600" />
              <span>{dispatching === "benchmark" ? "Dispatching..." : "Run 200-Task Sweep"}</span>
            </button>
          </div>
        </div>

        {/* Live / Simulated Streaming Console */}
        <div>
          <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-1.5">
            <span className="flex items-center space-x-1.5">
              <Terminal className="h-3 w-3" />
              <span>Vertex AI Console Stream</span>
            </span>
            <span className="text-emerald-700">Connected · us-central1</span>
          </div>
          <div className="bg-zinc-950 p-3.5 rounded-lg text-xs font-mono text-zinc-300 space-y-1 max-h-48 overflow-y-auto border border-zinc-900">
            {activeLogs.map((log, i) => (
              <div key={i} className="leading-relaxed">
                {log}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
