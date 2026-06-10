import "server-only";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Service-role client for trusted server code (e.g. the Stripe webhook). It
// bypasses RLS, so it must NEVER be imported into client/browser code. Returns
// null when the service-role key isn't configured.
export function createSupabaseAdminClient(): SupabaseClient | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  // New Supabase naming: "secret" key (sb_secret_...) replaces "service_role".
  // Accept either; both have elevated, RLS-bypassing privileges.
  const serviceKey = process.env.SUPABASE_SECRET_KEY ?? process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) return null;
  return createClient(url, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}
