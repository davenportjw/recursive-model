"use client";

import React, { useEffect, useState } from "react";
import {
  Layers,
  Activity,
  BookOpen,
  Cloud,
  Cpu,
  User,
  LogOut,
  Play,
  BarChart3,
} from "lucide-react";

export type NavTab = "architecture" | "examples" | "benchmarks" | "cloud";

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
}

interface UserProfile {
  email: string;
  name: string;
  picture?: string;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const [user, setUser] = useState<UserProfile | null>(null);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.authenticated && data?.user) {
          setUser(data.user);
        }
      })
      .catch(() => {
        // Ignore network errors or unauthenticated state
      });
  }, []);

  const handleSignOut = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } finally {
      window.location.href = "/login";
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b border-zinc-200 bg-white/95 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Academic Masthead */}
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-zinc-900 flex items-center justify-center text-white">
              <Cpu className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-sm tracking-tight text-zinc-900">
                  Tiny Recursive Gemma
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium bg-zinc-100 text-zinc-600 border border-zinc-200">
                  Samsung TRM × Gemma 4
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 hidden sm:block">
                Continuous Latent-Space Recurrence (Samsung SAIL Montréal)
              </p>
            </div>
          </div>

          {/* Clean Academic Navigation Tabs */}
          <nav className="flex items-center space-x-1 bg-zinc-100 p-1 rounded-lg border border-zinc-200/80">
            <button
              onClick={() => setActiveTab("architecture")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === "architecture"
                  ? "bg-white text-zinc-900 shadow-xs border border-zinc-200/60 font-semibold"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50"
              }`}
            >
              <Layers className="h-3.5 w-3.5 text-sky-700" />
              <span>1. Architecture &amp; Code</span>
            </button>

            <button
              onClick={() => setActiveTab("examples")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === "examples"
                  ? "bg-white text-zinc-900 shadow-xs border border-zinc-200/60 font-semibold"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50"
              }`}
            >
              <Play className="h-3.5 w-3.5 text-zinc-800" />
              <span>2. Interactive Examples</span>
            </button>

            <button
              onClick={() => setActiveTab("benchmarks")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === "benchmarks"
                  ? "bg-white text-zinc-900 shadow-xs border border-zinc-200/60 font-semibold"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50"
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5 text-zinc-800" />
              <span>3. Empirical Results</span>
            </button>

            <button
              onClick={() => setActiveTab("cloud")}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === "cloud"
                  ? "bg-white text-zinc-900 shadow-xs border border-zinc-200/60 font-semibold"
                  : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50"
              }`}
            >
              <Cloud className="h-3.5 w-3.5 text-zinc-800" />
              <span>4. Cloud System</span>
            </button>
          </nav>

          {/* User Profile & Actions */}
          <div className="flex items-center space-x-2">
            {user ? (
              <div className="flex items-center space-x-2">
                <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-xs text-zinc-700">
                  <div className="w-4 h-4 rounded-full bg-zinc-800 flex items-center justify-center text-[9px] font-bold text-white">
                    {user.name?.[0]?.toUpperCase() || user.email[0]?.toUpperCase()}
                  </div>
                  <span className="font-medium max-w-[130px] truncate">{user.email}</span>
                </div>

                <button
                  onClick={handleSignOut}
                  title="Sign Out"
                  className="flex items-center space-x-1 px-2.5 py-1 rounded-md text-xs font-medium text-zinc-600 hover:text-red-700 hover:bg-red-50 border border-transparent hover:border-red-200 transition-all"
                >
                  <LogOut className="h-3 w-3" />
                  <span className="hidden sm:inline">Sign Out</span>
                </button>
              </div>
            ) : (
              <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-200 text-xs text-zinc-600">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span className="font-medium">Cloud Run Active</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
