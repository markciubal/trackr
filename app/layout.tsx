import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { SiteHeader } from "./components/SiteHeader";
import { PWARegister } from "./components/PWARegister";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Trackr Shot Measurement",
  description: "Calibrate video frames and estimate object size with OpenCV + trig.",
  applicationName: "Trackr",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "Trackr",
    statusBarStyle: "black-translucent",
  },
  icons: {
    icon: "/icon.svg",
    apple: "/icon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#000000",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <SiteHeader />
        {children}
        <PWARegister />
      </body>
    </html>
  );
}
