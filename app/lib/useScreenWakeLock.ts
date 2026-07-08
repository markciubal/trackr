"use client";

import { useEffect, useRef } from "react";

// Keeps the device screen awake (no dim/lock) while `active` is true — phones
// sleeping mid-activity kill camera streams and silence drill callouts.
// Best-effort: the Wake Lock API is optional and the request can be denied
// (battery saver, permissions); everything still works, the screen just isn't
// guaranteed to stay on. The lock auto-releases when the tab is hidden, so it
// re-acquires whenever the page becomes visible again.
export function useScreenWakeLock(active: boolean): void {
  const sentinelRef = useRef<{ release: () => Promise<void> } | null>(null);

  useEffect(() => {
    if (!active) return;
    const nav = navigator as Navigator & {
      wakeLock?: { request: (type: "screen") => Promise<{ release: () => Promise<void> }> };
    };
    if (!nav.wakeLock) return;

    let cancelled = false;
    const acquire = async () => {
      try {
        const sentinel = await nav.wakeLock!.request("screen");
        if (cancelled) {
          void sentinel.release().catch(() => undefined);
          return;
        }
        sentinelRef.current = sentinel;
      } catch {
        // Denied — proceed without the lock.
      }
    };
    void acquire();

    const onVisibility = () => {
      if (document.visibilityState === "visible") void acquire();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      const sentinel = sentinelRef.current;
      sentinelRef.current = null;
      if (sentinel) void sentinel.release().catch(() => undefined);
    };
  }, [active]);
}
