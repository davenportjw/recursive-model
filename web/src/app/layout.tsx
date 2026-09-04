import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tiny Recursive Gemma | Latent-Space Recurrence on Gemma 4",
  description: "Transferring Samsung SAIL Montréal's Tiny Recursive Model (TRM) continuous latent reasoning to Google Gemma 4 2B.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-[#fbfbfd] text-zinc-900 antialiased min-h-screen selection:bg-sky-100 selection:text-sky-900">
        {children}
      </body>
    </html>
  );
}
