import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    service: "tiny-recursive-gemma-web",
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || "production",
    gcp_project: process.env.GOOGLE_CLOUD_PROJECT || "davenport-boutique",
    gcp_region: process.env.GOOGLE_CLOUD_REGION || "us-central1",
    directives: {
      no_mocking_enforced: true,
      latest_gemini_models: "gemini-3.8-flash / gemini-3.8-pro",
      recursive_model: "Gemma 4 2B Continuous Latent TRM",
    },
    endpoints: {
      health: { path: "/api/health", methods: ["GET"], description: "System health check and route discovery" },
      infer: { path: "/api/infer", methods: ["GET", "POST"], description: "Real 3-way generation (Baseline, Discrete CoT, Continuous TRM)" },
      judge: { path: "/api/judge", methods: ["GET", "POST"], description: "Gemini 3.8 Flash LLM-as-a-judge evaluation" },
      execute: { path: "/api/execute", methods: ["GET", "POST"], description: "Sandboxed Python code execution and test verification" },
      benchmarks: { path: "/api/benchmarks", methods: ["GET"], description: "200-task benchmark suite metadata and evaluation reports" },
      cloud: { path: "/api/cloud", methods: ["GET", "POST"], description: "Vertex AI GPU jobs and Cloud Run status" },
      research: { path: "/api/research", methods: ["GET"], description: "Research hypotheses, empirical matrix, and paper alignment" },
      auth_me: { path: "/api/auth/me", methods: ["GET"], description: "Current session and IAP user identity" },
    },
  });
}
