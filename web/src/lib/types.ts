export interface BenchmarkTask {
  task_id: string;
  source?: string;
  difficulty?: "Tier 1" | "Tier 2" | "Tier 3" | "Tier 4";
  category?: string;
  prompt: string;
  entry_point: string;
  canonical_solution?: string;
  test: string;
}

export interface CandidateResult {
  code: string;
  tokens: number;
  latency_ms: number;
  passed: boolean;
  test_output?: string;
  thoughts?: string[];
  code_updates?: string[];
  iterations_completed?: number;
  halted_early?: boolean;
  trajectory_distances?: number[];
}

export interface JudgeScoreItem {
  functional_correctness: number;
  algorithmic_soundness: number;
  recursive_progression: number;
  token_efficiency: number;
  hallucination_resistance: number;
  total_score: number;
  critique: string;
}

export interface TaskEvaluationReport {
  task_id: string;
  baseline: CandidateResult;
  discrete: CandidateResult;
  continuous: CandidateResult;
  judge_evaluations?: {
    baseline: JudgeScoreItem;
    discrete: JudgeScoreItem;
    continuous: JudgeScoreItem;
  };
  samsung_paper_alignment?: {
    did_continuous_match_discrete: boolean;
    token_saving_factor: string;
    latent_reasoning_verdict: string;
  };
}

export interface ResearchHypothesis {
  id: string;
  title: string;
  status: "VALIDATED" | "CONFIRMED CHALLENGE" | "EXPLORING";
  confidence: "High" | "Medium-High" | "Medium";
  summary: string;
  empirical_observation: string;
  takeaway: string;
}

export interface FailureModeItem {
  name: string;
  symptom: string;
  mitigation: string;
  status: string;
}

export interface BenchmarkMatrixRow {
  metric: string;
  baseline: string;
  discrete: string;
  continuous: string;
}
