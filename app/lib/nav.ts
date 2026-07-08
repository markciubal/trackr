// Central registry of the app's sections, rendered as the global header nav.
//
// This array is the single source of truth: add an entry here and the new
// section appears in the header on every page automatically — no header markup
// to touch. Keep it ordered the way it should read left-to-right.

export type NavSection = {
  label: string;
  href: string;
  // Prefix that marks this section "active" for highlighting. Defaults to href.
  // Use it when a section owns child routes (e.g. /targets/new under /targets).
  activePrefix?: string;
  // When true, only render for signed-in users (the header passes auth state in).
  requiresAuth?: boolean;
  // External link: rendered as a plain <a> opening in a new tab, never "active".
  external?: boolean;
  // Optional small inline icon rendered before the label (see FeatureNav).
  icon?: "pistol";
};

export const NAV_SECTIONS: NavSection[] = [
  { label: "Scan", href: "/scan" },
  { label: "Library", href: "/shots", requiresAuth: true },
  { label: "Targets", href: "/targets/new", activePrefix: "/targets" },
  { label: "Drill", href: "/drill" },
  {
    label: "Donate",
    href: "https://www.paypal.com/donate/?hosted_button_id=Q9JCGNQF924WW",
    external: true,
    icon: "pistol",
  },
  // The brand ("Trackr") on the left links to the home/landing page. Account lives
  // in the auth chip on the right; add new app sections here and they appear in the
  // header everywhere. Set requiresAuth for signed-in-only ones.
];

// Whether a section should be highlighted for the current path. Root ("/") only
// matches exactly so it isn't active on every route; others match their subtree.
export function isSectionActive(section: NavSection, pathname: string): boolean {
  if (section.href === "/") return pathname === "/";
  const prefix = section.activePrefix ?? section.href;
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}
