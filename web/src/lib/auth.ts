/**
 * Authentication and authorization utilities matching the pattern from smartrouter.
 * Enforces email and domain allowlisting via ALLOWED_EMAIL_DOMAINS.
 * Uses 100% universal Web Crypto & standard Web APIs for Edge Runtime & Node compatibility.
 */

export interface SessionPayload {
  email: string;
  name: string;
  picture?: string;
  exp: number; // Unix timestamp in seconds
}

/**
 * Checks if the provided email is authorized based on a list of specific emails or domains.
 * Exact logic ported from smartrouter/frontend/auth/auth.go:
 * - Case-insensitive
 * - If entry contains '@', matches specific email address exactly
 * - If entry does not contain '@', matches entire domain suffix
 */
export function isEmailAuthorized(email: string, allowedList: string[]): boolean {
  const normalizedEmail = email.toLowerCase().trim();
  for (const rawEntry of allowedList) {
    let entry = rawEntry.toLowerCase().trim();
    if (!entry) continue;

    // Trim leading '@' if user input includes it (e.g., @gmail.com)
    if (entry.startsWith("@")) {
      entry = entry.substring(1);
    }

    if (entry.includes("@")) {
      // Match specific email address exactly
      if (normalizedEmail === entry) {
        return true;
      }
    } else {
      // Match domain suffix (@domain.com) or any organizational subdomain (.domain.com)
      if (
        normalizedEmail.endsWith("@" + entry) ||
        normalizedEmail.endsWith("." + entry)
      ) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Extracts authenticated user email from Google Cloud Identity-Aware Proxy (IAP)
 * or Cloud Run authenticated ingress headers.
 */
export function extractIapUserEmail(headers: Headers): string | null {
  // 1. Direct Google Authenticated User Email header
  const authEmail =
    headers.get("x-goog-authenticated-user-email") ||
    headers.get("x-forwarded-user-email") ||
    headers.get("x-user-email") ||
    headers.get("x-goog-user-email");

  if (authEmail && authEmail.trim()) {
    // Header format is typically "accounts.google.com:username@domain.com"
    const cleaned = authEmail.includes(":")
      ? authEmail.split(":").pop()!.trim()
      : authEmail.trim();
    if (cleaned.includes("@")) {
      return cleaned.toLowerCase();
    }
  }

  // 2. Google IAP JWT Assertion header (x-goog-iap-jwt-assertion)
  const iapJwt = headers.get("x-goog-iap-jwt-assertion");
  if (iapJwt && iapJwt.includes(".")) {
    try {
      const parts = iapJwt.split(".");
      if (parts.length >= 2) {
        // Base64URL decode the JWT payload
        const base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
        const jsonStr =
          typeof atob !== "undefined"
            ? atob(base64)
            : Buffer.from(base64, "base64").toString("utf-8");
        const payload = JSON.parse(jsonStr);
        if (payload.email && typeof payload.email === "string") {
          return payload.email.toLowerCase().trim();
        }
      }
    } catch (e) {
      console.warn("[Auth] Failed to parse x-goog-iap-jwt-assertion payload:", e);
    }
  }

  return null;
}

/**
 * Parses the ALLOWED_EMAIL_DOMAINS environment variable into a clean string array.
 */
export function getAllowedEmailDomains(): string[] {
  const envVal = process.env.ALLOWED_EMAIL_DOMAINS || "";
  if (!envVal.trim()) {
    // In local dev, default to google.com and joonix.net if not set
    if (process.env.LOCAL_DEV === "true") {
      return [
        "google.com",
        "joonix.net",
        "cloudadvocacyorg.joonix.net",
        "davenportjw@cloudadvocacyorg.joonix.net",
        "davenportjw@gmail.com",
      ];
    }
    return [];
  }
  return envVal
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

const DEFAULT_SECRET = "tiny-recursive-gemma-insecure-default-secret-key-32chars!";

function getSigningKey(): string {
  return (
    process.env.SESSION_SECRET ||
    process.env.BACKEND_SHARED_SECRET ||
    DEFAULT_SECRET
  );
}

function base64UrlEncode(str: string): string {
  if (typeof btoa === "function") {
    return btoa(unescape(encodeURIComponent(str)))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
  }
  return Buffer.from(str, "utf-8").toString("base64url");
}

function base64UrlDecode(str: string): string {
  if (typeof atob === "function") {
    let base64 = str.replace(/-/g, "+").replace(/_/g, "/");
    while (base64.length % 4) {
      base64 += "=";
    }
    return decodeURIComponent(escape(atob(base64)));
  }
  return Buffer.from(str, "base64url").toString("utf-8");
}

function arrayBufferToBase64Url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

/**
 * Creates an HMAC-SHA256 signature using Web Crypto API.
 */
async function createHmacSignature(data: string, secret: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(data));
  return arrayBufferToBase64Url(signature);
}

/**
 * Generates a signed session token.
 */
export async function createSessionToken(payload: SessionPayload): Promise<string> {
  const secret = getSigningKey();
  const header = base64UrlEncode(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const body = base64UrlEncode(JSON.stringify(payload));
  const data = `${header}.${body}`;
  const signature = await createHmacSignature(data, secret);
  return `${data}.${signature}`;
}

/**
 * Verifies a signed session token and checks expiration.
 */
export async function verifySessionToken(token: string): Promise<SessionPayload | null> {
  if (!token || typeof token !== "string") return null;

  // Handle local dev mock session cookie
  if (process.env.LOCAL_DEV === "true" && token === "dev-admin-session-cookie") {
    return {
      email: "dev-admin@google.com",
      name: "Local Developer",
      picture: "",
      exp: Math.floor(Date.now() / 1000) + 86400 * 5,
    };
  }

  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [header, body, signature] = parts;
  const data = `${header}.${body}`;
  const secret = getSigningKey();

  try {
    const expectedSig = await createHmacSignature(data, secret);
    if (signature !== expectedSig) {
      return null;
    }

    const payload: SessionPayload = JSON.parse(base64UrlDecode(body));

    const nowSeconds = Math.floor(Date.now() / 1000);
    if (payload.exp && payload.exp < nowSeconds) {
      return null;
    }

    return payload;
  } catch (err) {
    return null;
  }
}

/**
 * Verifies a Google ID token by querying Google's official tokeninfo endpoint
 * or resolving dev-admin-token in local dev mode.
 */
export async function verifyGoogleIdToken(
  idToken: string
): Promise<{ email: string; name: string; picture?: string } | null> {
  // Support local developer mock token
  if (process.env.LOCAL_DEV === "true" && idToken === "dev-admin-token") {
    return {
      email: "dev-admin@google.com",
      name: "Local Developer",
      picture: "",
    };
  }

  try {
    const res = await fetch(
      `https://oauth2.googleapis.com/tokeninfo?id_token=${encodeURIComponent(idToken)}`,
      {
        method: "GET",
      }
    );

    if (!res.ok) {
      console.error("[Auth] Google tokeninfo verification failed with status:", res.status);
      return null;
    }

    const data = await res.json();
    if (!data.email || data.email_verified === "false" || data.email_verified === false) {
      console.warn("[Auth] Token does not contain verified email:", data.email);
      return null;
    }

    return {
      email: data.email,
      name: data.name || data.email.split("@")[0],
      picture: data.picture || "",
    };
  } catch (err) {
    console.error("[Auth] Error verifying Google ID token:", err);
    return null;
  }
}
