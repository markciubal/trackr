import Link from "next/link";
import { redirect } from "next/navigation";
import { signOut } from "@/app/auth/actions";
import { getEntitlement } from "@/app/lib/entitlements";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { isStripeConfigured } from "@/app/lib/stripe";
import { BillingButtons } from "./BillingButtons";

export const metadata = { title: "Account · Trackr" };

export default async function AccountPage() {
  if (!isSupabaseConfigured()) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-black px-4 text-white">
        <div className="w-full max-w-md rounded-xl border border-gray-700 bg-neutral-950 p-6">
          <h1 className="text-xl font-semibold">Account</h1>
          <p className="mt-2 text-sm text-amber-100">
            Accounts aren&apos;t configured yet. Add Supabase keys to <code>.env.local</code> to enable sign-in and
            billing.
          </p>
          <Link href="/" className="mt-4 inline-block text-sm text-sky-300 hover:underline">
            ← Back to Trackr
          </Link>
        </div>
      </main>
    );
  }

  const entitlement = await getEntitlement();
  if (!entitlement.userId) redirect("/login");

  return (
    <main className="flex min-h-screen items-start justify-center bg-black px-4 py-12 text-white">
      <div className="w-full max-w-md space-y-6">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-sm text-sky-300 hover:underline">
            ← Back to Trackr
          </Link>
          <form action={signOut}>
            <button type="submit" className="text-sm text-gray-400 transition hover:text-gray-200">
              Sign out
            </button>
          </form>
        </div>

        <div className="rounded-xl border border-gray-700 bg-neutral-950 p-6">
          <h1 className="text-xl font-semibold">Account</h1>
          <p className="mt-1 text-sm text-gray-400">{entitlement.email}</p>

          <div className="mt-4 flex items-center gap-2">
            <span className="text-sm text-gray-300">Plan</span>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                entitlement.isPro
                  ? "bg-emerald-500/20 text-emerald-200"
                  : "bg-neutral-800 text-gray-300"
              }`}
            >
              {entitlement.isPro ? "Pro" : "Free"}
            </span>
          </div>

          <div className="mt-5">
            <BillingButtons isPro={entitlement.isPro} stripeConfigured={isStripeConfigured()} />
          </div>
        </div>

        {!entitlement.isPro ? (
          <div className="rounded-xl border border-gray-800 bg-neutral-950/60 p-4 text-xs text-gray-400">
            <p className="font-semibold text-gray-200">Pro unlocks</p>
            <ul className="mt-2 list-inside list-disc space-y-1">
              <li>Unlimited scans and saved sessions</li>
              <li>Exporting results (CSV / shot data)</li>
              <li>Priority processing</li>
            </ul>
          </div>
        ) : null}
      </div>
    </main>
  );
}
