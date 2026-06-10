import { NextResponse } from "next/server";
import { getStripe } from "@/app/lib/stripe";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export const runtime = "nodejs";

// Creates a Stripe Checkout Session for the Pro subscription and returns its URL.
export async function POST(request: Request) {
  const stripe = getStripe();
  if (!stripe) return NextResponse.json({ error: "Billing isn't configured." }, { status: 503 });

  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Auth isn't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Please sign in first." }, { status: 401 });

  const body = (await request.json().catch(() => ({}))) as { interval?: string };
  const interval = body.interval === "annual" ? "annual" : "monthly";
  const priceId =
    interval === "annual" ? process.env.STRIPE_PRICE_PRO_ANNUAL : process.env.STRIPE_PRICE_PRO_MONTHLY;
  if (!priceId) {
    return NextResponse.json({ error: `No ${interval} price configured.` }, { status: 503 });
  }

  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? new URL(request.url).origin;

  // Reuse the user's Stripe customer, or create one and store it on the profile.
  const { data: profile } = await supabase
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", user.id)
    .maybeSingle();

  let customerId: string | undefined = profile?.stripe_customer_id ?? undefined;
  if (!customerId) {
    const customer = await stripe.customers.create({
      email: user.email ?? undefined,
      metadata: { supabase_user_id: user.id },
    });
    customerId = customer.id;
    await supabase
      .from("profiles")
      .update({ stripe_customer_id: customerId, updated_at: new Date().toISOString() })
      .eq("id", user.id);
  }

  const session = await stripe.checkout.sessions.create({
    mode: "subscription",
    customer: customerId,
    line_items: [{ price: priceId, quantity: 1 }],
    allow_promotion_codes: true,
    success_url: `${siteUrl}/account?checkout=success`,
    cancel_url: `${siteUrl}/account?checkout=cancelled`,
    metadata: { supabase_user_id: user.id },
    subscription_data: { metadata: { supabase_user_id: user.id } },
  });

  return NextResponse.json({ url: session.url });
}
