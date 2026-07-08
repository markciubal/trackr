"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV_SECTIONS, isSectionActive, type NavSection } from "@/app/lib/nav";

// Glock-style pistol silhouette, drawn inline so it renders identically on
// every platform (emoji pistols vary wildly per OS). Inherits the link's
// color, and the muzzle points right — toward the label text. Rear/front
// sight nubs on the flat slide, squared trigger guard with an open trigger
// window, and the backward-raked grip.
function PistolIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M1 3.2h.8v-.6h.8v.6h11v-.6h.8v.6h.6v3h-6.2c0 1.7-.2 3.3-.6 3.7H5.7c-.5 0-.7-.5-.75-1l-.05-.8c-.15 2.5-.55 4.6-1.2 5.9H.9c.35-2.6.3-5.2.15-7.8zm4.9 4.1h1.8c.05.9-.05 1.55-.25 1.75H6.15c-.13-.55-.21-1.15-.25-1.75z"
      />
    </svg>
  );
}

function NavIcon({ icon }: { icon: NavSection["icon"] }) {
  if (icon === "pistol") return <PistolIcon />;
  return null;
}

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
        if (section.external) {
          return (
            <a
              key={section.href}
              href={section.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex shrink-0 items-center gap-1 whitespace-nowrap rounded-md px-2 py-1 font-medium text-amber-300/90 transition hover:text-amber-200"
            >
              <NavIcon icon={section.icon} />
              {section.label}
            </a>
          );
        }
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
