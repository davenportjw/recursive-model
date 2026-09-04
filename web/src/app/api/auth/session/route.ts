import { NextRequest, NextResponse } from "next/server";
import {
  getAllowedEmailDomains,
  isEmailAuthorized,
  verifyGoogleIdToken,
  createSessionToken,
  extractIapUserEmail,
} from "@/lib/auth";

export async function POST(request: NextRequest) {
  try {
    let idToken = "";

    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        const body = await request.json();
        idToken = body?.idToken || "";
      } catch {
        // no json body
      }
    } else if (contentType.includes("application/x-www-form-urlencoded")) {
      try {
        const formData = await request.formData();
        idToken = (formData.get("idToken") as string) || "";
      } catch {
        // no form data
      }
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

    // Check 1: Cloud Run Identity-Aware Proxy (IAP) header
    const iapEmail = extractIapUserEmail(request.headers);
    if (iapEmail) {
      if (!isEmailAuthorized(iapEmail, allowedDomains)) {
        return NextResponse.json(
          { error: `Access restricted. ${iapEmail} is not authorized.` },
          { status: 403 }
        );
      }

      const expiresInSeconds = 5 * 24 * 60 * 60;
      const exp = Math.floor(Date.now() / 1000) + expiresInSeconds;
      const userName = iapEmail.split("@")[0];

      const sessionCookieValue = await createSessionToken({
        email: iapEmail,
        name: userName,
        picture: "",
        exp,
      });

      const response = NextResponse.json({
        status: "success",
        source: "iap",
        user: {
          email: iapEmail,
          name: userName,
          picture: "",
        },
      });

      response.cookies.set({
        name: "session",
        value: sessionCookieValue,
        maxAge: expiresInSeconds,
        path: "/",
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
      });

      return response;
    }

    // Check 2: Missing idToken when no IAP header is present
    if (!idToken || !idToken.trim()) {
      return NextResponse.json(
        { error: "Missing idToken parameter and no Cloud Run IAP identity detected." },
        { status: 400 }
      );
    }

    // Check 3: Token verification (Google OAuth ID Token or local dev bypass)
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
      source: "token",
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

export async function GET(request: NextRequest) {
  // Allow GET /api/auth/session to auto-initialize session if IAP header is present
  return POST(request);
}
