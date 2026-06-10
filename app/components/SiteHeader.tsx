import Link from "next/link";
import { getEntitlement } from "@/app/lib/entitlements";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";

// Slim auth-aware bar. Renders nothing until Supabase is configured, so the
// tool's immersive layout is unchanged during local/offline use.
export async function SiteHeader() {
  if (!isSupabaseConfigured()) return null;

  const entitlement = await getEntitlement();

  return (
    <header className="flex items-center justify-between border-b border-gray-800 bg-black px-4 py-2 text-white">
      <Link href="/" className="text-sm font-semibold tracking-tight">
        Trackr
      </Link>
      <nav className="flex items-center gap-3 text-xs">
        {entitlement.userId ? (
          <>
            <span
              className={`rounded-full px-2 py-0.5 font-semibold ${
                entitlement.isPro ? "bg-emerald-500/20 text-emerald-200" : "bg-neutral-800 text-gray-300"
              }`}
            >
              {entitlement.isPro ? "Pro" : "Free"}
            </span>
            <Link href="/account" className="text-gray-300 transition hover:text-white">
              {entitlement.email ?? "Account"}
            </Link>
          </>
        ) : (
          <Link
            href="/login"
            className="rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-1 font-medium text-sky-100 transition hover:bg-sky-500/25"
          >
            Sign in
          </Link>
        )}
      </nav>
    </header>
  );
}
