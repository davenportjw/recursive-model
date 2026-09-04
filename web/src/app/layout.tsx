import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tiny Recursive Gemma | Continuous Latent Reasoning",
  description: "Transferring Samsung SAIL Montréal's Tiny Recursive Model (TRM) continuous latent reasoning to Google Gemma 4 2B.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-[#FAF9F5] text-[#141413] antialiased min-h-screen selection:bg-[#FAF0EC] selection:text-[#C96442]">
        {children}
      </body>
    </html>
  );
}
