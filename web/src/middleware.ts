import { NextRequest, NextResponse } from "next/server";

export function middleware(request: NextRequest) {
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

  // 2. Local Development bypass (matching smartrouter's LOCAL_DEV="true" convention)
  const isLocalDev = process.env.LOCAL_DEV === "true";
  if (isLocalDev) {
    return NextResponse.next();
  }

  // 3. Read configured credentials
  const expectedUser = process.env.BASIC_AUTH_USER || "admin";
  const expectedPassword = process.env.BASIC_AUTH_PASSWORD;
  const sharedSecret = process.env.BACKEND_SHARED_SECRET;

  // If no password or secret is configured and not in local dev, fail closed for security
  if (!expectedPassword && !sharedSecret) {
    console.warn("[Auth] Neither BASIC_AUTH_PASSWORD nor BACKEND_SHARED_SECRET configured.");
  }

  // 4. Check SmartRouter-style shared secret headers (for API requests / programmatic clients)
  const xSharedSecret = request.headers.get("X-Shared-Secret");
  if (sharedSecret && xSharedSecret && xSharedSecret === sharedSecret) {
    return NextResponse.next();
  }

  const authHeader = request.headers.get("authorization") || request.headers.get("Authorization");

  if (sharedSecret && authHeader && authHeader.startsWith("Bearer ")) {
    const bearerToken = authHeader.substring(7).trim();
    if (bearerToken === sharedSecret) {
      return NextResponse.next();
    }
  }

  // 5. Check HTTP Basic Authentication
  if (expectedPassword && authHeader && authHeader.startsWith("Basic ")) {
    try {
      const base64Credentials = authHeader.substring(6).trim();
      const credentials = atob(base64Credentials);
      const colonIndex = credentials.indexOf(":");

      if (colonIndex !== -1) {
        const username = credentials.substring(0, colonIndex);
        const password = credentials.substring(colonIndex + 1);

        if (username === expectedUser && password === expectedPassword) {
          return NextResponse.next();
        }
      }
    } catch (e) {
      // Invalid base64 encoding, fall through to 401
    }
  }

  // 6. Return 401 Unauthorized with standard Basic Auth challenge
  if (pathname.startsWith("/api/")) {
    return new NextResponse(
      JSON.stringify({ error: "Unauthorized: Invalid credentials or missing Authorization header" }),
      {
        status: 401,
        headers: {
          "Content-Type": "application/json",
          "WWW-Authenticate": 'Basic realm="Tiny Recursive Gemma Dashboard", charset="UTF-8"',
        },
      }
    );
  }

  return new NextResponse("Authentication Required: Access to Tiny Recursive Gemma is restricted.", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="Tiny Recursive Gemma Dashboard", charset="UTF-8"',
      "Content-Type": "text/plain; charset=UTF-8",
    },
  });
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for static files.
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
