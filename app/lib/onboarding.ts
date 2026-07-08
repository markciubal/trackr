// Onboarding registry. This is the single place to grow onboarding as the
// project builds out: add a tip here and returning users automatically get
// shown any id they haven't dismissed yet. Bump ONBOARDING_VERSION to re-show
// everything to everyone (e.g. after a big redesign).

export const ONBOARDING_VERSION = 1;

export type OnboardingTip = {
  id: string;
  step?: number; // wizard step index this tip maps to (for highlighting); omit for general tips
  title: string;
  body: string;
};

export const ONBOARDING_TIPS: OnboardingTip[] = [
  {
    id: "welcome",
    title: "Welcome to Trackr",
    body: "Measure shot groups from a video. Work through the numbered steps — the next action lights up amber, and you can jump between steps anytime.",
  },
  {
    id: "source",
    step: 0,
    title: "1 · Source",
    body: "Upload a clip (or stream your phone camera). The first frame is used as your reference image automatically.",
  },
  {
    id: "capture",
    step: 1,
    title: "2 · Capture",
    body: "Drag a rectangle around the target on the reference image to set the area Trackr measures.",
  },
  {
    id: "calibrate",
    step: 2,
    title: "3 · Calibrate",
    body: "Pick your caliber and how you'll set the target's real size; enter the measurements on the next step so pixels convert to inches.",
  },
  {
    id: "scan",
    step: 4,
    title: "5 · Scan",
    body: "Press Start Scan to detect impacts. They appear on the target as they're found.",
  },
  {
    id: "review",
    step: 6,
    title: "7 · Map & Review",
    body: "Edit groups and scrub the reveal on the Map step; see group stats and the full shot table on Review.",
  },
];

const STORAGE_KEY = `trackr-onboarding-v${ONBOARDING_VERSION}`;

export function getSeenTips(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed)
      ? new Set(parsed.filter((value): value is string => typeof value === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

export function markTipsSeen(ids: string[]): void {
  if (typeof window === "undefined") return;
  const seen = getSeenTips();
  for (const id of ids) seen.add(id);
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify([...seen]));
  } catch {
    // Ignore storage quota / privacy-mode errors.
  }
}

export function hasUnseenTips(): boolean {
  const seen = getSeenTips();
  return ONBOARDING_TIPS.some((tip) => !seen.has(tip.id));
}

export function resetOnboarding(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore.
  }
}
