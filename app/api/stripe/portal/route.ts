import { NextResponse } from "next/server";
import { getStripe } from "@/app/lib/stripe";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export const runtime = "nodejs";

// Opens the Stripe Customer Portal so users can manage/cancel their subscription.
export async function POST(request: Request) {
  const stripe = getStripe();
  if (!stripe) return NextResponse.json({ error: "Billing isn't configured." }, { status: 503 });

  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Auth isn't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Please sign in first." }, { status: 401 });

  const { data: profile } = await supabase
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", user.id)
    .maybeSingle();

  if (!profile?.stripe_customer_id) {
    return NextResponse.json({ error: "No billing account yet — upgrade first." }, { status: 400 });
  }

  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? new URL(request.url).origin;
  const session = await stripe.billingPortal.sessions.create({
    customer: profile.stripe_customer_id,
    return_url: `${siteUrl}/account`,
  });

  return NextResponse.json({ url: session.url });
}
