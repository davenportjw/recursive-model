import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    endpoint: "/api/execute",
    methods: ["GET", "POST"],
    runtime: "Python 3 sandboxed execution",
    timeout_ms: 4000,
    description: "Executes Python code against candidate unit tests to determine functional pass/fail status."
  });
}

export async function POST(request: NextRequest) {
  try {
    const { code, test, entry_point } = await request.json();

    if (!code || !test) {
      return NextResponse.json({ passed: false, output: "Missing code or test suite" }, { status: 400 });
    }

    const scriptContent = `
${code}

${test}

if __name__ == '__main__':
    try:
        check(${entry_point || "candidate"})
        print("TEST_PASSED_SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()
`;

    const tmpDir = os.tmpdir();
    const scriptPath = path.join(tmpDir, `test_${Date.now()}_${Math.random().toString(36).substring(7)}.py`);
    fs.writeFileSync(scriptPath, scriptContent);

    return new Promise<NextResponse>((resolve) => {
      exec(`python3 ${scriptPath}`, { timeout: 4000 }, (error, stdout, stderr) => {
        try {
          fs.unlinkSync(scriptPath);
        } catch {}

        if (error || stderr) {
          const out = stderr || stdout || (error ? error.message : "Execution failed");
          resolve(NextResponse.json({
            passed: false,
            output: out.slice(0, 1000)
          }));
        } else {
          resolve(NextResponse.json({
            passed: stdout.includes("TEST_PASSED_SUCCESS"),
            output: stdout.trim()
          }));
        }
      });
    });

  } catch (error: any) {
    return NextResponse.json({ passed: false, output: error.message }, { status: 500 });
  }
}
