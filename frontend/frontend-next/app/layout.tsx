// app/layout.tsx
import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Traffic Optimization Dashboard",
  description: "AI-powered traffic analysis and prediction for Ghana-Accra & Kumasi",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
