"use server";

import { redirect } from "next/navigation";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

// OAuth + magic-link sign-in is initiated from the browser (see LoginForm), so
// the only server action left here is sign-out.
export async function signOut() {
  const supabase = await createSupabaseServerClient();
  if (supabase) await supabase.auth.signOut();
  redirect("/login");
}
