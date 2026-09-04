"use client";

import React, { useState, useEffect } from "react";
import Script from "next/script";
import { Sparkles, Shield, AlertCircle, Loader2, Cloud, ArrowRight, Cpu } from "lucide-react";

export default function LoginPage() {
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLocalDev, setIsLocalDev] = useState(false);
  const [googleClientId, setGoogleClientId] = useState<string>("");

  useEffect(() => {
    // 1. Check if user is already authenticated (via Cloud Run IAP proxy or session cookie)
    fetch("/api/auth/me")
      .then((res) => {
        if (res.ok) {
          return res.json();
        }
        return null;
      })
      .then((data) => {
        if (data?.authenticated) {
          window.location.href = "/";
        }
      })
      .catch(() => {
        // Not yet authenticated
      });

    // 2. Check local dev status
    const isDev =
      process.env.NODE_ENV === "development" ||
      process.env.NEXT_PUBLIC_LOCAL_DEV === "true" ||
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1";
    setIsLocalDev(isDev);

    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "";
    setGoogleClientId(clientId);
  }, []);

  const handleCloudIdentitySignIn = async (idToken?: string) => {
    setLoading(true);
    setErrorMessage(null);

    try {
      const res = await fetch("/api/auth/session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(idToken ? { idToken } : {}),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Authentication failed. Make sure your account is in ALLOWED_EMAIL_DOMAINS.");
      }

      // Successfully authenticated via Cloud Run IAP proxy or token
      window.location.href = "/";
    } catch (err: any) {
      console.error("Auth error:", err);
      setErrorMessage(err.message || "Failed to authenticate.");
      setLoading(false);
    }
  };

  const signInAsDeveloper = () => {
    handleCloudIdentitySignIn("dev-admin-token");
  };

  const handleGoogleCredentialResponse = (response: any) => {
    if (response && response.credential) {
      handleCloudIdentitySignIn(response.credential);
    } else {
      setErrorMessage("No credential received from Google.");
    }
  };

  return (
    <>
      {googleClientId && (
        <Script
          src="https://accounts.google.com/gsi/client"
          strategy="afterInteractive"
          onLoad={() => {
            if ((window as any).google?.accounts?.id && googleClientId) {
              try {
                (window as any).google.accounts.id.initialize({
                  client_id: googleClientId,
                  callback: handleGoogleCredentialResponse,
                });
                (window as any).google.accounts.id.renderButton(
                  document.getElementById("google-btn-container"),
                  { theme: "outline", size: "large", width: "100%" }
                );
              } catch (e) {
                console.warn("Failed to init Google GIS:", e);
              }
            }
          }}
        />
      )}

      <div className="min-h-screen bg-[#fbfbfd] text-zinc-900 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
          <div className="inline-flex items-center justify-center p-2.5 rounded-xl bg-zinc-900 text-white shadow-xs mb-3">
            <Cpu className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">
            Tiny Recursive Gemma
          </h1>
          <p className="mt-1 text-xs text-zinc-500 font-medium">
            Continuous Latent Recurrence Research Portal (Gemma 4 2B)
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md px-4 sm:px-0">
          <div className="bg-white py-8 px-6 shadow-xs rounded-xl border border-zinc-200 sm:px-10">
            <div className="space-y-6">
              <div className="text-center">
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-zinc-100 text-zinc-700 border border-zinc-200 mb-2">
                  <Shield className="w-3 h-3 text-sky-700" />
                  <span>Google Cloud Identity-Aware Proxy (IAP)</span>
                </div>
                <h2 className="text-base font-semibold text-zinc-900">Sign In to Continue</h2>
                <p className="text-xs text-zinc-500 mt-1 leading-relaxed">
                  Access is authenticated via Google Cloud IAP and restricted to authorized organizational domains.
                </p>
              </div>

              {/* Error message */}
              {errorMessage && (
                <div className="rounded-lg bg-red-50 border border-red-200 p-3.5">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-semibold text-red-900">Authentication Failed</h4>
                      <p className="text-xs text-red-700 mt-0.5">{errorMessage}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Sign in with Google / Cloud Run Identity */}
              {googleClientId ? (
                <div id="google-btn-container" className="w-full flex justify-center min-h-[44px]" />
              ) : (
                <button
                  type="button"
                  onClick={() => handleCloudIdentitySignIn()}
                  disabled={loading}
                  className="w-full flex justify-center items-center gap-2.5 rounded-lg bg-zinc-900 px-4 py-2.5 text-xs font-medium text-white shadow-xs hover:bg-zinc-800 transition disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin text-zinc-400" />
                      <span>Authenticating Cloud Identity...</span>
                    </>
                  ) : (
                    <>
                      <Cloud className="h-4 w-4 text-sky-400" />
                      <span>Authenticate with Google Cloud</span>
                    </>
                  )}
                </button>
              )}

              {/* Developer Bypass (Visible only during local dev) */}
              {isLocalDev && (
                <div className="pt-3 border-t border-zinc-100">
                  <button
                    type="button"
                    onClick={signInAsDeveloper}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-1.5 rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-800 px-4 py-2 text-xs font-medium transition border border-zinc-200"
                  >
                    <span>Developer Session Bypass</span>
                    <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 text-center text-[11px] text-zinc-400">
            Samsung SAIL Montréal (arXiv:2510.04871) · Google Cloud Vertex AI &amp; Cloud Run
          </div>
        </div>
      </div>
    </>
  );
}
