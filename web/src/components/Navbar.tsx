"use client";

import React, { useEffect, useState } from "react";
import { LogOut, ArrowUpRight } from "lucide-react";

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

  const navItems: { id: NavTab; label: string; index: string }[] = [
    { id: "architecture", label: "Architecture", index: "01" },
    { id: "examples", label: "Interactive Lab", index: "02" },
    { id: "benchmarks", label: "Empirical Results", index: "03" },
    { id: "cloud", label: "Cloud System", index: "04" },
  ];

  return (
    <header className="fixed top-0 inset-x-0 z-50 h-14 bg-[#FAF9F5]/90 backdrop-blur-md border-b border-[#E8E5DF] transition-colors">
      <div className="max-w-6xl mx-auto h-full px-4 sm:px-6 flex items-center justify-between">
        {/* Masthead: Thinking Machines / Anthropic editorial branding */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2.5">
            {/* Minimalist Geometric Monogram */}
            <svg
              className="h-4 w-4 text-[#141413]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 7v10M7 12h10" />
            </svg>
            <span className="font-editorial text-lg tracking-tight font-medium text-[#141413]">
              Tiny Recursive Gemma
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-2 pl-2 border-l border-[#E8E5DF] text-xs text-[#8C887B]">
            <span className="font-mono text-[11px]">Gemma 4</span>
            <span>·</span>
            <span>Samsung SAIL Montréal</span>
          </div>
        </div>

        {/* Swift / Anthropic Flat Typographic Navigation (No box-in-box pills) */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`relative px-3 py-1.5 text-xs transition-all duration-150 rounded-full flex items-center space-x-1.5 ${
                  isActive
                    ? "text-[#141413] bg-[#EFECE4] font-medium"
                    : "text-[#57534E] hover:text-[#141413] hover:bg-[#F4F1EA]"
                }`}
              >
                <span className={`text-[10px] font-mono ${isActive ? "text-[#C96442]" : "text-[#8C887B]"}`}>
                  {item.index}
                </span>
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Status / Paper Citation Link */}
        <div className="flex items-center space-x-3">
          {user ? (
            <div className="flex items-center space-x-2">
              <span className="hidden lg:inline text-xs font-mono text-[#57534E] max-w-[130px] truncate">
                {user.email}
              </span>
              <button
                onClick={handleSignOut}
                title="Sign Out"
                className="text-xs text-[#8C887B] hover:text-[#C96442] transition-colors p-1"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <div className="hidden sm:flex items-center space-x-1.5 text-[11px] font-mono text-[#57534E]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2D6A4F]"></span>
              <span>Vertex AI L4</span>
            </div>
          )}

          <a
            href="https://arxiv.org/abs/2510.04871"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:flex items-center space-x-1 text-xs text-[#57534E] hover:text-[#141413] transition-colors pl-2 border-l border-[#E8E5DF]"
          >
            <span>Paper</span>
            <ArrowUpRight className="h-3 w-3 opacity-60" />
          </a>
        </div>
      </div>
    </header>
  );
};
