"use client";

import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import { SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL, isSupabaseConfigured } from "./config";

let cached: SupabaseClient | null = null;

// Browser Supabase client (singleton). Returns null when unconfigured.
export function createSupabaseBrowserClient(): SupabaseClient | null {
  if (!isSupabaseConfigured()) return null;
  if (!cached) cached = createBrowserClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
  return cached;
}
