import "server-only";
import Stripe from "stripe";

export function isStripeConfigured(): boolean {
  return Boolean(process.env.STRIPE_SECRET_KEY);
}

let cached: Stripe | null = null;

// Server-side Stripe client. Returns null when not configured so routes can
// respond with a clean 503 instead of crashing.
export function getStripe(): Stripe | null {
  if (!isStripeConfigured()) return null;
  if (!cached) {
    cached = new Stripe(process.env.STRIPE_SECRET_KEY as string);
  }
  return cached;
}
