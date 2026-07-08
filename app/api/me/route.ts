import { NextResponse } from "next/server";
import { getEntitlement } from "@/app/lib/entitlements";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Lightweight membership probe for client components (the analysis page's Save
// step). Returns whether Supabase is configured, whether the request is signed
// in, the Pro flag, and the email.
export async function GET() {
  const entitlement = await getEntitlement();
  return NextResponse.json({
    configured: entitlement.configured,
    signedIn: Boolean(entitlement.userId),
    isPro: entitlement.isPro,
    email: entitlement.email,
  });
}
