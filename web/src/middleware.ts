import { NextRequest, NextResponse } from "next/server";
import { verifySessionToken, isEmailAuthorized, getAllowedEmailDomains } from "./lib/auth";

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

  // 2. Bypass authentication for public auth endpoints and login page
  if (pathname === "/login" || pathname.startsWith("/api/auth/")) {
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

  // 5. User Session Cookie Verification
  const sessionCookie = request.cookies.get("session")?.value;
  if (sessionCookie) {
    const payload = await verifySessionToken(sessionCookie);
    if (payload && payload.email) {
      const allowedDomains = getAllowedEmailDomains();
      if (isEmailAuthorized(payload.email, allowedDomains)) {
        // Authorized: pass user identity forward via request headers
        const requestHeaders = new Headers(request.headers);
        requestHeaders.set("x-user-email", payload.email);
        requestHeaders.set("x-user-name", payload.name || "");

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

  // 6. Unauthorized Handling
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
