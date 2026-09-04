"use client";

import React, { useState, useEffect } from "react";
import Script from "next/script";
import { Sparkles, Shield, AlertCircle, Loader2, Cloud, ArrowRight } from "lucide-react";

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

      <div className="min-h-screen bg-slate-950 flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Ambient background glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="sm:mx-auto sm:w-full sm:max-w-md relative z-10 text-center">
          <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl mb-4">
            <Sparkles className="w-8 h-8 text-purple-400 animate-pulse" />
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            <span className="bg-gradient-to-r from-purple-400 via-indigo-300 to-sky-400 bg-clip-text text-transparent">
              Tiny Recursive Gemma
            </span>
          </h1>
          <p className="mt-2 text-sm text-slate-400 font-medium">
            Latent-Space Continuous Reasoning &amp; Benchmark Console
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md relative z-10 px-4 sm:px-0">
          <div className="bg-slate-900/90 backdrop-blur-xl py-8 px-6 shadow-2xl rounded-2xl border border-slate-800 sm:px-10">
            <div className="space-y-6">
              <div className="text-center">
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700/60 mb-2">
                  <Shield className="w-3.5 h-3.5 text-indigo-400" />
                  Cloud Run Identity-Aware Access
                </div>
                <h2 className="text-lg font-semibold text-slate-100">Sign In</h2>
                <p className="text-xs text-slate-400 mt-1">
                  Access is protected by Google Cloud Identity-Aware Proxy (IAP) and restricted to authorized organizational domains.
                </p>
              </div>

              {/* Error message */}
              {errorMessage && (
                <div className="rounded-xl bg-red-950/50 border border-red-800/60 p-4">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-semibold text-red-300">Access Denied</h4>
                      <p className="text-xs text-red-200/80 mt-1">{errorMessage}</p>
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
                  className="w-full flex justify-center items-center gap-3 rounded-xl bg-white px-4 py-3 text-sm font-semibold text-slate-900 shadow-md hover:bg-slate-100 transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin text-slate-600" />
                      <span>Authenticating Cloud Identity...</span>
                    </>
                  ) : (
                    <>
                      <svg className="h-5 w-5" viewBox="0 0 24 24">
                        <path
                          fill="#4285F4"
                          d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                        />
                        <path
                          fill="#34A853"
                          d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                        />
                        <path
                          fill="#FBBC05"
                          d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                        />
                        <path
                          fill="#EA4335"
                          d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                        />
                      </svg>
                      <span>Sign In with Google Cloud Identity</span>
                    </>
                  )}
                </button>
              )}

              {/* Developer Bypass (Visible only during local dev) */}
              {isLocalDev && (
                <div className="pt-2 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={signInAsDeveloper}
                    disabled={loading}
                    className="w-full flex items-center justify-center gap-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-4 py-2.5 text-xs font-medium transition border border-slate-700/60"
                  >
                    <span>Bypass Auth (Local Developer)</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 text-center text-xs text-slate-500">
            Powered by Google Cloud Run &amp; Vertex AI · SmartRouter Security Protocol
          </div>
        </div>
      </div>
    </>
  );
}
