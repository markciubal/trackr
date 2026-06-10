import "server-only";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export type Plan = "anonymous" | "free" | "pro";

// Free-tier caps. Enforce these wherever the matching feature lives; they're
// centralized here so pricing/limits live in one place.
export const FREE_TIER = {
  maxSavedSessions: 3,
  maxScansPerDay: 10,
  allowExport: false,
} as const;

// Stripe statuses that grant Pro access.
const ACTIVE_STATUSES = new Set(["active", "trialing", "past_due"]);

export type Entitlement = {
  configured: boolean; // is Supabase wired up at all?
  plan: Plan;
  isPro: boolean;
  userId: string | null;
  email: string | null;
};

// Single source of truth for "what can this request do?". When Supabase isn't
// configured we grant full access so the tool works with zero setup; gating
// only switches on once you've wired auth + billing.
export async function getEntitlement(): Promise<Entitlement> {
  if (!isSupabaseConfigured()) {
    return { configured: false, plan: "pro", isPro: true, userId: null, email: null };
  }

  const supabase = await createSupabaseServerClient();
  if (!supabase) {
    return { configured: false, plan: "pro", isPro: true, userId: null, email: null };
  }

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    return { configured: true, plan: "anonymous", isPro: false, userId: null, email: null };
  }

  const { data: subs } = await supabase.from("subscriptions").select("status").eq("user_id", user.id);
  const isPro = (subs ?? []).some((row: { status: string }) => ACTIVE_STATUSES.has(row.status));

  return {
    configured: true,
    plan: isPro ? "pro" : "free",
    isPro,
    userId: user.id,
    email: user.email ?? null,
  };
}
