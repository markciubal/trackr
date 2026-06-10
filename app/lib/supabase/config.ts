// Auth/billing activate only when Supabase is configured. Until then every
// helper degrades to a no-op so the tool keeps working with zero setup.

// Supabase renamed the browser-safe key from "anon" to "publishable"
// (sb_publishable_...). Accept either env name; publishable wins if both set.
export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_PUBLISHABLE_KEY =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export function isSupabaseConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_PUBLISHABLE_KEY);
}
