import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export const runtime = "nodejs";

// Save a designed target to the catalog so its QR id resolves to rich data.
export async function POST(request: Request) {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Auth isn't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in to save targets." }, { status: 401 });

  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const id = typeof body.id === "string" ? body.id : "";
  const widthValue = Number(body.widthValue);
  const heightValue = Number(body.heightValue);
  const qrSizeValue = Number(body.qrSizeValue);
  const unit = body.unit === "mm" || body.unit === "cm" || body.unit === "in" ? body.unit : "in";
  if (!id || !(widthValue > 0) || !(heightValue > 0) || !(qrSizeValue > 0)) {
    return NextResponse.json({ error: "Invalid target." }, { status: 400 });
  }

  const { error } = await supabase.from("targets").upsert({
    id,
    owner_id: user.id,
    name: typeof body.name === "string" ? body.name : null,
    width_value: widthValue,
    height_value: heightValue,
    unit,
    qr_size_value: qrSizeValue,
    scoring_id: typeof body.scoringId === "string" ? body.scoringId : null,
    // Drill layout, so a scanned id-only QR can rebuild the exact zones.
    drill_recipe: typeof body.drillRecipe === "string" ? body.drillRecipe : null,
    drill_palette_version: Number.isFinite(Number(body.drillPaletteVersion))
      ? Number(body.drillPaletteVersion)
      : null,
    is_published: true,
  });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}

// Resolve a published target's drill layout by id, for clients (e.g. the drill
// runner) that scanned an id-only QR and never saved the target locally. The QR
// no longer carries the recipe inline, so this is how a foreign device rebuilds
// the exact zones; it caches the result locally after the first online fetch.
export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id")?.trim();
  if (!id) return NextResponse.json({ error: "Missing id." }, { status: 400 });

  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Catalog isn't configured." }, { status: 503 });

  const { data } = await supabase
    .from("targets")
    .select("id,name,width_value,height_value,unit,qr_size_value,scoring_id,drill_recipe,drill_palette_version")
    .eq("id", id)
    .maybeSingle();
  if (!data) return NextResponse.json({ error: "Not found." }, { status: 404 });

  return NextResponse.json({
    id: data.id,
    name: data.name ?? null,
    widthValue: Number(data.width_value),
    heightValue: Number(data.height_value),
    unit: data.unit === "mm" || data.unit === "cm" || data.unit === "in" ? data.unit : "in",
    qrSizeValue: Number(data.qr_size_value),
    scoringId: data.scoring_id ?? null,
    drillRecipe: data.drill_recipe ?? null,
    drillPaletteVersion: data.drill_palette_version ?? null,
  });
}
