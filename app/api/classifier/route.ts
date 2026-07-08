import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";
import { isModelJSON } from "@/app/lib/training/jsonModel";

export const runtime = "nodejs";
// The published model changes rarely; let clients/CDN cache it briefly so the
// scanner's startup fetch is cheap, while still picking up a new publish soon.
export const dynamic = "force-dynamic";

// Public: return the latest published shot-classifier model, or { model: null }.
// This is the "pushed out to users" endpoint — every scanner loads it on startup
// and feeds it into setHoleClassifier(). RLS already restricts SELECT to
// published rows, so an anon client can read it.
export async function GET() {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ model: null });

  const { data } = await supabase
    .from("classifier_models")
    .select("version,payload,created_at")
    .eq("is_published", true)
    .order("version", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (!data || !isModelJSON(data.payload)) return NextResponse.json({ model: null });

  return NextResponse.json(
    { version: data.version, model: data.payload, createdAt: data.created_at },
    { headers: { "Cache-Control": "public, max-age=60, stale-while-revalidate=300" } },
  );
}
