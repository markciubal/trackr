import { NextResponse } from "next/server";
import type Stripe from "stripe";
import { getStripe } from "@/app/lib/stripe";
import { createSupabaseAdminClient } from "@/app/lib/supabase/admin";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Stripe -> our DB. Keeps the `subscriptions` table in sync so entitlements are
// authoritative. Configure the endpoint URL in Stripe to POST here, and set
// STRIPE_WEBHOOK_SECRET from the signing secret.
export async function POST(request: Request) {
  const stripe = getStripe();
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!stripe || !webhookSecret) {
    return NextResponse.json({ error: "Webhook isn't configured." }, { status: 503 });
  }

  const signature = request.headers.get("stripe-signature");
  if (!signature) return NextResponse.json({ error: "Missing signature." }, { status: 400 });

  const rawBody = await request.text();
  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, signature, webhookSecret);
  } catch {
    return NextResponse.json({ error: "Invalid signature." }, { status: 400 });
  }

  const admin = createSupabaseAdminClient();
  if (!admin) {
    return NextResponse.json({ error: "Service-role client unavailable." }, { status: 503 });
  }

  const upsertSubscription = async (sub: Stripe.Subscription) => {
    // Prefer the user id we stamped into metadata; otherwise resolve via customer.
    let userId = (sub.metadata?.supabase_user_id as string | undefined) ?? null;
    if (!userId) {
      const customerId = typeof sub.customer === "string" ? sub.customer : sub.customer.id;
      const { data } = await admin
        .from("profiles")
        .select("id")
        .eq("stripe_customer_id", customerId)
        .maybeSingle();
      userId = (data?.id as string | undefined) ?? null;
    }
    if (!userId) return;

    // `current_period_end` location varies across Stripe API versions; read defensively.
    const periodEnd = (sub as unknown as { current_period_end?: number }).current_period_end;
    await admin.from("subscriptions").upsert({
      id: sub.id,
      user_id: userId,
      status: sub.status,
      price_id: sub.items.data[0]?.price.id ?? null,
      current_period_end: typeof periodEnd === "number" ? new Date(periodEnd * 1000).toISOString() : null,
      cancel_at_period_end: sub.cancel_at_period_end ?? false,
      updated_at: new Date().toISOString(),
    });
  };

  try {
    switch (event.type) {
      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted":
        await upsertSubscription(event.data.object as Stripe.Subscription);
        break;
      case "checkout.session.completed": {
        const session = event.data.object as Stripe.Checkout.Session;
        if (session.subscription) {
          const subId =
            typeof session.subscription === "string" ? session.subscription : session.subscription.id;
          const sub = await stripe.subscriptions.retrieve(subId);
          await upsertSubscription(sub as Stripe.Subscription);
        }
        break;
      }
      default:
        break;
    }
  } catch {
    return NextResponse.json({ error: "Handler error." }, { status: 500 });
  }

  return NextResponse.json({ received: true });
}
