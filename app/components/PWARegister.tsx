"use client";

import { useEffect } from "react";

// Registers the service worker after load. Production-only: a SW caching the
// shell during dev fights Turbopack HMR.
export function PWARegister() {
  useEffect(() => {
    if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;
    if (process.env.NODE_ENV !== "production") return;
    const register = () => {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        // Registration is best-effort; the app works without it.
      });
    };
    if (document.readyState === "complete") register();
    else {
      window.addEventListener("load", register);
      return () => window.removeEventListener("load", register);
    }
  }, []);

  return null;
}
