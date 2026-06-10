import { NextResponse } from "next/server";
import type { EmailOtpType } from "@supabase/supabase-js";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

// Single redirect target for every sign-in method:
//   • OAuth (Google/GitHub) comes back with ?code=...        → exchangeCodeForSession
//   • Magic links come back with ?token_hash=...&type=...     → verifyOtp
//   • Provider/user errors come back with ?error_description= → bounce to /login
// Sets the session cookie on success, then forwards to `next`.
export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const tokenHash = url.searchParams.get("token_hash");
  const type = url.searchParams.get("type") as EmailOtpType | null;
  const providerError =
    url.searchParams.get("error_description") ?? url.searchParams.get("error");

  // Only allow same-origin relative redirects (no open-redirect via ?next=).
  const requestedNext = url.searchParams.get("next") ?? "/account";
  const next = requestedNext.startsWith("/") ? requestedNext : "/account";

  const fail = (msg: string) =>
    NextResponse.redirect(new URL(`/login?error=${encodeURIComponent(msg)}`, url.origin));

  if (providerError) return fail(providerError);

  const supabase = await createSupabaseServerClient();
  if (!supabase) return fail("Authentication is not configured yet.");

  if (code) {
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (error) return fail(error.message);
  } else if (tokenHash && type) {
    const { error } = await supabase.auth.verifyOtp({ type, token_hash: tokenHash });
    if (error) return fail(error.message);
  } else {
    return fail("This sign-in link is invalid or has expired.");
  }

  return NextResponse.redirect(new URL(next, url.origin));
}
