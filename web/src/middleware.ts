import { NextRequest, NextResponse } from "next/server";
import {
  verifySessionToken,
  isEmailAuthorized,
  getAllowedEmailDomains,
  extractIapUserEmail,
  createSessionToken,
} from "./lib/auth";

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 1. Bypass authentication for Next.js internal files and static assets
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/static") ||
    pathname === "/favicon.ico" ||
    pathname === "/robots.txt"
  ) {
    return NextResponse.next();
  }

  // 2. Bypass authentication for health check and auth callback endpoints
  if (pathname.startsWith("/api/auth/") || pathname === "/api/health") {
    return NextResponse.next();
  }

  // 3. Local Development bypass (matching smartrouter's LOCAL_DEV="true" convention)
  const isLocalDev = process.env.LOCAL_DEV === "true";
  if (isLocalDev) {
    return NextResponse.next();
  }

  // 4. Machine-to-Machine Secret (SmartRouter convention: X-Shared-Secret or Bearer token)
  const sharedSecret = process.env.BACKEND_SHARED_SECRET;
  if (sharedSecret) {
    const xSharedSecret = request.headers.get("X-Shared-Secret");
    if (xSharedSecret && xSharedSecret === sharedSecret) {
      return NextResponse.next();
    }

    const authHeader =
      request.headers.get("authorization") || request.headers.get("Authorization");
    if (authHeader && authHeader.startsWith("Bearer ")) {
      const bearerToken = authHeader.substring(7).trim();
      if (bearerToken === sharedSecret) {
        return NextResponse.next();
      }
    }
  }

  const allowedDomains = getAllowedEmailDomains();

  // 5. Cloud Run Identity-Aware Proxy (IAP) Header Detection
  // Automatically authenticate when request carries Google Cloud IAP identity headers
  const iapEmail = extractIapUserEmail(request.headers);
  if (iapEmail) {
    if (isEmailAuthorized(iapEmail, allowedDomains)) {
      const userName = iapEmail.split("@")[0];

      // If user was heading to /login, redirect directly to dashboard
      const response =
        pathname === "/login"
          ? NextResponse.redirect(new URL("/", request.url))
          : NextResponse.next();

      // Pass verified user headers to downstream routes
      response.headers.set("x-user-email", iapEmail);
      response.headers.set("x-user-name", userName);

      // Auto-issue session cookie if not present or expired
      const existingCookie = request.cookies.get("session")?.value;
      if (!existingCookie) {
        const expiresInSeconds = 5 * 24 * 60 * 60; // 5 days
        const exp = Math.floor(Date.now() / 1000) + expiresInSeconds;
        const token = await createSessionToken({
          email: iapEmail,
          name: userName,
          picture: "",
          exp,
        });

        response.cookies.set({
          name: "session",
          value: token,
          maxAge: expiresInSeconds,
          path: "/",
          httpOnly: true,
          secure: process.env.NODE_ENV === "production",
          sameSite: "lax",
        });
      }

      return response;
    } else {
      console.warn(`[Middleware] Cloud Run IAP email unauthorized: ${iapEmail}`);
      return new NextResponse(
        JSON.stringify({
          error: `Access restricted. ${iapEmail} is not in ALLOWED_EMAIL_DOMAINS.`,
        }),
        {
          status: 403,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
  }

  // 6. User Session Cookie Verification
  const sessionCookie = request.cookies.get("session")?.value;
  if (sessionCookie) {
    const payload = await verifySessionToken(sessionCookie);
    if (payload && payload.email) {
      if (isEmailAuthorized(payload.email, allowedDomains)) {
        // If user already has a valid session and visits /login, redirect to dashboard
        if (pathname === "/login") {
          return NextResponse.redirect(new URL("/", request.url));
        }

        const requestHeaders = new Headers(request.headers);
        requestHeaders.set("x-user-email", payload.email);
        requestHeaders.set("x-user-name", payload.name || payload.email.split("@")[0]);

        return NextResponse.next({
          request: {
            headers: requestHeaders,
          },
        });
      } else {
        console.warn(`[Middleware] Session user email unauthorized: ${payload.email}`);
      }
    }
  }

  // 7. Allow access to /login page for unauthenticated users
  if (pathname === "/login") {
    return NextResponse.next();
  }

  // 8. Unauthorized Handling
  // If this is an API call, return JSON 401
  if (pathname.startsWith("/api/")) {
    return new NextResponse(
      JSON.stringify({
        error: "Unauthorized: Active session or valid shared secret required",
      }),
      {
        status: 401,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // If this is a page request, redirect to /login
  const loginUrl = new URL("/login", request.url);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for static files.
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
