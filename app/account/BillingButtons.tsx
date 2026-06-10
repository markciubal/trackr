"use client";

import { useState } from "react";

// Calls a billing route, then redirects to the Stripe-hosted URL it returns.
export function BillingButtons({ isPro, stripeConfigured }: { isPro: boolean; stripeConfigured: boolean }) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const go = async (path: string, body?: Record<string, unknown>) => {
    setLoading(path + JSON.stringify(body ?? {}));
    setError(null);
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body ?? {}),
      });
      const data = (await res.json().catch(() => ({}))) as { url?: string; error?: string };
      if (data.url) {
        window.location.href = data.url;
        return;
      }
      setError(data.error ?? "Something went wrong.");
    } catch {
      setError("Network error — please try again.");
    }
    setLoading(null);
  };

  if (!stripeConfigured) {
    return <p className="text-xs text-gray-400">Billing isn&apos;t configured yet.</p>;
  }

  return (
    <div className="space-y-2">
      {isPro ? (
        <button
          type="button"
          onClick={() => go("/api/stripe/portal")}
          disabled={loading !== null}
          className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:opacity-50"
        >
          {loading ? "Opening…" : "Manage billing"}
        </button>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => go("/api/stripe/checkout", { interval: "monthly" })}
            disabled={loading !== null}
            className="rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25 disabled:opacity-50"
          >
            {loading ? "Redirecting…" : "Upgrade to Pro — Monthly"}
          </button>
          <button
            type="button"
            onClick={() => go("/api/stripe/checkout", { interval: "annual" })}
            disabled={loading !== null}
            className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:opacity-50"
          >
            Annual
          </button>
        </div>
      )}
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  );
}
