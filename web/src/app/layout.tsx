import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tiny Recursive Gemma | Cloud Showcase & Research Dashboard",
  description: "Comparing Zero-Shot Baseline, Discrete Recursive CoT, and Continuous Latent TRM on Google Gemma",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
