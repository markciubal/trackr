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
    <header className="flex items-center justify-between gap-3 border-b border-gray-800 bg-black px-4 py-2 text-white">
      <div className="flex items-center gap-3">
        <Link href="/" className="text-sm font-semibold tracking-tight">
          Trackr
        </Link>
        <FeatureNav signedIn={signedIn} />
      </div>
      {supabaseReady ? (
        <nav className="flex items-center gap-3 text-xs">
          {entitlement?.userId ? (
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
      ) : null}
    </header>
  );
}
