import { cookies } from "next/headers";
import { createServerClient, type CookieOptions } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

type CookieToSet = { name: string; value: string; options: CookieOptions };
import { SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL, isSupabaseConfigured } from "./config";

// Request-scoped Supabase client (reads/writes the auth cookie). Returns null
// when Supabase isn't configured so callers can no-op gracefully.
export async function createSupabaseServerClient(): Promise<SupabaseClient | null> {
  if (!isSupabaseConfigured()) return null;
  const cookieStore = await cookies();
  return createServerClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet: CookieToSet[]) {
        try {
          cookiesToSet.forEach(({ name, value, options }) => cookieStore.set(name, value, options));
        } catch {
          // Called from a Server Component (read-only cookies). The middleware
          // refreshes the session, so this is safe to ignore.
        }
      },
    },
  });
}

// Convenience: the current signed-in user, or null.
export async function getCurrentUser() {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return null;
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user;
}
