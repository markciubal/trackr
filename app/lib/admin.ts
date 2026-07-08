import "server-only";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

// Is the current request an admin? Mirrors getEntitlement(): when Supabase isn't
// configured the app runs fully open (zero-setup mode), so the admin panel is
// reachable locally; once auth is wired, only profiles.is_admin = true passes.
// Set the flag in Supabase: update profiles set is_admin = true where email = '…'.
export async function isCurrentUserAdmin(): Promise<boolean> {
  if (!isSupabaseConfigured()) return true;

  const supabase = await createSupabaseServerClient();
  if (!supabase) return true;

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return false;

  const { data } = await supabase.from("profiles").select("is_admin").eq("id", user.id).maybeSingle();
  return Boolean(data?.is_admin);
}
