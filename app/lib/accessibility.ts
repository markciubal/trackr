import { useCallback, useSyncExternalStore } from "react";

// App-wide "color-blind friendly" preference, persisted to localStorage so it
// sticks across sessions and applies on every page (analysis, drill, designer).
export const COLOR_BLIND_STORAGE_KEY = "trackr:color-blind";

export function readColorBlindPreference(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(COLOR_BLIND_STORAGE_KEY) === "true";
}

// Same-tab subscribers (the native `storage` event only fires in *other* tabs,
// so we keep our own listener set and notify it on write).
const listeners = new Set<() => void>();

function subscribe(callback: () => void): () => void {
  listeners.add(callback);
  if (typeof window !== "undefined") window.addEventListener("storage", callback);
  return () => {
    listeners.delete(callback);
    if (typeof window !== "undefined") window.removeEventListener("storage", callback);
  };
}

// Hook: returns [enabled, setEnabled]. Reads through localStorage via
// useSyncExternalStore (SSR-safe, no setState-in-effect) and re-renders every
// subscriber when toggled — in this tab and others.
export function useColorBlindMode(): readonly [boolean, (next: boolean) => void] {
  const enabled = useSyncExternalStore(subscribe, readColorBlindPreference, () => false);
  const update = useCallback((next: boolean) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(COLOR_BLIND_STORAGE_KEY, String(next));
    }
    listeners.forEach((listener) => listener());
  }, []);
  return [enabled, update] as const;
}
