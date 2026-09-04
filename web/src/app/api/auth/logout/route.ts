import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const response = NextResponse.json({ status: "success" });
  response.cookies.set({
    name: "session",
    value: "",
    maxAge: 0,
    path: "/",
    httpOnly: true,
  });
  return response;
}

export async function GET(request: NextRequest) {
  const url = new URL("/login", request.url);
  const response = NextResponse.redirect(url, { status: 302 });
  response.cookies.set({
    name: "session",
    value: "",
    maxAge: 0,
    path: "/",
    httpOnly: true,
  });
  return response;
}
