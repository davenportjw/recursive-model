import { NextRequest, NextResponse } from "next/server";
import { verifySessionToken } from "@/lib/auth";

export async function GET(request: NextRequest) {
  const isLocalDev = process.env.LOCAL_DEV === "true";
  const sessionCookie = request.cookies.get("session")?.value;

  if (isLocalDev && (!sessionCookie || sessionCookie === "dev-admin-session-cookie")) {
    return NextResponse.json({
      authenticated: true,
      user: {
        email: "dev-admin@google.com",
        name: "Local Developer",
        picture: "",
      },
    });
  }

  if (!sessionCookie) {
    return NextResponse.json({ authenticated: false, user: null }, { status: 401 });
  }

  const payload = await verifySessionToken(sessionCookie);
  if (!payload) {
    return NextResponse.json({ authenticated: false, user: null }, { status: 401 });
  }

  return NextResponse.json({
    authenticated: true,
    user: {
      email: payload.email,
      name: payload.name,
      picture: payload.picture || "",
    },
  });
}
