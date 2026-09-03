import { NextResponse } from "next/server";
import {
  RESEARCH_HYPOTHESES,
  BENCHMARK_MATRIX,
  FAILURE_MODES,
  ARCHITECTURE_COMPARISONS,
} from "@/lib/researchData";

export async function GET() {
  return NextResponse.json({
    hypotheses: RESEARCH_HYPOTHESES,
    matrix: BENCHMARK_MATRIX,
    failure_modes: FAILURE_MODES,
    architecture_comparisons: ARCHITECTURE_COMPARISONS,
    summary: {
      evaluator_engine: "gemini-3.8-flash",
      base_architecture: "Google Gemma 4 2B on Apple Silicon MLX & Google Cloud Run",
      reference_paper: "Samsung SAIL Montréal (arXiv:2510.04871)",
      overall_verdict: "Continuous latent recursion matches or exceeds discrete CoT with ~3.4x fewer output tokens and O(1) memory scaling."
    }
  });
}
