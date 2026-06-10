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
    <nav className="flex items-center gap-1 text-xs sm:gap-2">
      {sections.map((section) => {
        const active = isSectionActive(section, pathname);
        return (
          <Link
            key={section.href}
            href={section.href}
            aria-current={active ? "page" : undefined}
            className={`rounded-md px-2 py-1 font-medium transition ${
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
