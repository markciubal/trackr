"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV_SECTIONS, isSectionActive } from "@/app/lib/nav";

// Renders the section links from the central NAV_SECTIONS registry and
// highlights the one matching the current route. Auth-gated sections are only
// shown when `signedIn` is true (the server header knows the auth state).
export function FeatureNav({ signedIn }: { signedIn: boolean }) {
  const pathname = usePathname() ?? "/";
  const sections = NAV_SECTIONS.filter((section) => signedIn || !section.requiresAuth);

  return (
    // min-w-0 + overflow-x-auto lets the nav scroll within the header on small
    // screens instead of shoving the auth area (Sign in) off the edge. Scrollbar
    // hidden so it reads as a clean bar.
    <nav className="flex min-w-0 items-center gap-1 overflow-x-auto text-xs [-ms-overflow-style:none] [scrollbar-width:none] sm:gap-2 [&::-webkit-scrollbar]:hidden">
      {sections.map((section) => {
        const active = isSectionActive(section, pathname);
        return (
          <Link
            key={section.href}
            href={section.href}
            aria-current={active ? "page" : undefined}
            className={`shrink-0 whitespace-nowrap rounded-md px-2 py-1 font-medium transition ${
              active ? "bg-neutral-800 text-white" : "text-gray-400 hover:text-white"
            }`}
          >
            {section.label}
          </Link>
        );
      })}
    </nav>
  );
}
