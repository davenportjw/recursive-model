import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { SAMPLE_TASKS, PRECOMPUTED_REPORTS } from "@/lib/benchmarkData";
import { BenchmarkTask } from "@/lib/types";

function getFilesystemTasks(): BenchmarkTask[] {
  try {
    const filePath = path.join(process.cwd(), "..", "eval", "complex_tasks_200.jsonl");
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, "utf-8");
      const lines = content.split("\n").filter((l) => l.trim().length > 0);
      return lines.map((line) => JSON.parse(line));
    }
  } catch (e) {
    console.warn("Could not read tasks from filesystem:", e);
  }
  return SAMPLE_TASKS;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const taskId = searchParams.get("task_id");
  const allTasks = getFilesystemTasks();

  if (taskId) {
    const report = PRECOMPUTED_REPORTS[taskId];
    const task = allTasks.find((t) => t.task_id === taskId) || SAMPLE_TASKS.find((t) => t.task_id === taskId);

    if (task) {
      if (report) {
        return NextResponse.json({ task, report });
      }
      return NextResponse.json({
        task,
        report: null,
        message: "No precomputed evaluation report found for this task. Run evaluate.py or judge_evaluator.py to generate real benchmark metrics."
      });
    }

    return NextResponse.json({ error: "Task not found" }, { status: 404 });
  }

  return NextResponse.json({
    total: allTasks.length,
    tasks: allTasks
  });
}
