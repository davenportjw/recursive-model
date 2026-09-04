import { NextRequest, NextResponse } from "next/server";
import {
  getAllowedEmailDomains,
  isEmailAuthorized,
  verifyGoogleIdToken,
  createSessionToken,
} from "@/lib/auth";

export async function POST(request: NextRequest) {
  try {
    let idToken = "";

    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const body = await request.json();
      idToken = body.idToken;
    } else if (contentType.includes("application/x-www-form-urlencoded")) {
      const formData = await request.formData();
      idToken = (formData.get("idToken") as string) || "";
    } else {
      const body = await request.text();
      try {
        const parsed = JSON.parse(body);
        idToken = parsed.idToken;
      } catch {
        const params = new URLSearchParams(body);
        idToken = params.get("idToken") || "";
      }
    }

    if (!idToken || !idToken.trim()) {
      return NextResponse.json(
        { error: "Missing idToken parameter" },
        { status: 400 }
      );
    }

    const isLocalDev = process.env.LOCAL_DEV === "true";
    const allowedDomains = getAllowedEmailDomains();

    if (!isLocalDev && allowedDomains.length === 0) {
      console.error("[Auth] Access denied because ALLOWED_EMAIL_DOMAINS is not set");
      return NextResponse.json(
        { error: "Authentication system is misconfigured: ALLOWED_EMAIL_DOMAINS must be explicitly set" },
        { status: 500 }
      );
    }

    // Verify token
    const userInfo = await verifyGoogleIdToken(idToken.trim());
    if (!userInfo) {
      return NextResponse.json(
        { error: "Invalid ID token or verification failed" },
        { status: 401 }
      );
    }

    // Check email authorization
    if (!isEmailAuthorized(userInfo.email, allowedDomains)) {
      console.warn(`[Auth] Access denied for unauthorized email: ${userInfo.email}`);
      return NextResponse.json(
        { error: "Access restricted to authorized domains or specific email addresses" },
        { status: 403 }
      );
    }

    // Set session expiration: 5 days (matching smartrouter)
    const expiresInSeconds = 5 * 24 * 60 * 60;
    const exp = Math.floor(Date.now() / 1000) + expiresInSeconds;

    const sessionCookieValue =
      isLocalDev && idToken === "dev-admin-token"
        ? "dev-admin-session-cookie"
        : await createSessionToken({
            email: userInfo.email,
            name: userInfo.name,
            picture: userInfo.picture,
            exp,
          });

    const response = NextResponse.json({
      status: "success",
      user: {
        email: userInfo.email,
        name: userInfo.name,
        picture: userInfo.picture,
      },
    });

    response.cookies.set({
      name: "session",
      value: sessionCookieValue,
      maxAge: expiresInSeconds,
      path: "/",
      httpOnly: true,
      secure: process.env.NODE_ENV === "production" && !isLocalDev,
      sameSite: "lax",
    });

    return response;
  } catch (error: any) {
    console.error("[Auth] Error in createSession:", error);
    return NextResponse.json(
      { error: error?.message || "Internal server error" },
      { status: 500 }
    );
  }
}
