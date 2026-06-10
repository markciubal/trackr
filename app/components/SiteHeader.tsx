import Link from "next/link";
import { FeatureNav } from "@/app/components/FeatureNav";
import { getEntitlement } from "@/app/lib/entitlements";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";

// Global header: brand + the section nav (driven by the NAV_SECTIONS registry,
// so it grows automatically as features are added) + an auth-aware chip. The
// nav always renders; only the auth area depends on Supabase being configured,
// so local/offline use still gets navigation without an auth prompt.
export async function SiteHeader() {
  const supabaseReady = isSupabaseConfigured();
  const entitlement = supabaseReady ? await getEntitlement() : null;
  const signedIn = Boolean(entitlement?.userId);

  return (
    <header className="flex items-center justify-between gap-2 border-b border-gray-800 bg-black px-3 py-2 text-white sm:gap-3 sm:px-4">
      <div className="flex min-w-0 items-center gap-2 sm:gap-3">
        <Link href="/" className="shrink-0 text-sm font-semibold tracking-tight">
          Trackr
        </Link>
        <FeatureNav signedIn={signedIn} />
      </div>
      {supabaseReady ? (
        <nav className="flex shrink-0 items-center gap-2 text-xs sm:gap-3">
          {entitlement?.userId ? (
            <>
              <span
                className={`rounded-full px-2 py-0.5 font-semibold ${
                  entitlement.isPro ? "bg-emerald-500/20 text-emerald-200" : "bg-neutral-800 text-gray-300"
                }`}
              >
                {entitlement.isPro ? "Pro" : "Free"}
              </span>
              <Link
                href="/account"
                className="max-w-[7rem] truncate text-gray-300 transition hover:text-white sm:max-w-none"
              >
                {entitlement.email ?? "Account"}
              </Link>
            </>
          ) : (
            <Link
              href="/login"
              className="shrink-0 rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-1 font-medium text-sky-100 transition hover:bg-sky-500/25"
            >
              Sign in
            </Link>
          )}
        </nav>
      ) : null}
    </header>
  );
}
