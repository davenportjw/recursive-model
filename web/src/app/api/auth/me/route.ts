import { NextRequest, NextResponse } from "next/server";
import { verifySessionToken, extractIapUserEmail, getAllowedEmailDomains, isEmailAuthorized } from "@/lib/auth";

export async function GET(request: NextRequest) {
  const isLocalDev = process.env.LOCAL_DEV === "true";
  const sessionCookie = request.cookies.get("session")?.value;

  // 1. Check Cloud Run IAP headers or middleware forwarded headers
  const iapEmail =
    request.headers.get("x-user-email") ||
    extractIapUserEmail(request.headers);

  if (iapEmail) {
    const allowed = isEmailAuthorized(iapEmail, getAllowedEmailDomains());
    if (allowed) {
      return NextResponse.json({
        authenticated: true,
        user: {
          email: iapEmail,
          name: request.headers.get("x-user-name") || iapEmail.split("@")[0],
          picture: "",
        },
      });
    }
  }

  // 2. Local dev bypass
  if (isLocalDev && (!sessionCookie || sessionCookie === "dev-admin-session-cookie")) {
    return NextResponse.json({
      authenticated: true,
      user: {
        email: "davenportjw@cloudadvocacyorg.joonix.net",
        name: "davenportjw",
        picture: "",
      },
    });
  }

  // 3. Session cookie check
  if (!sessionCookie) {
    return NextResponse.json({ authenticated: false, user: null }, { status: 401 });
  }

  const payload = await verifySessionToken(sessionCookie);
  if (!payload || !payload.email) {
    return NextResponse.json({ authenticated: false, user: null }, { status: 401 });
  }

  return NextResponse.json({
    authenticated: true,
    user: {
      email: payload.email,
      name: payload.name || payload.email.split("@")[0],
      picture: payload.picture || "",
    },
  });
}
