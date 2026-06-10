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

const TARGET_COLUMNS =
  "id,name,width_value,height_value,unit,qr_size_value,scoring_id,drill_recipe,drill_palette_version,created_at";

type TargetRow = {
  id: string;
  name: string | null;
  width_value: number | string;
  height_value: number | string;
  unit: string | null;
  qr_size_value: number | string;
  scoring_id: string | null;
  drill_recipe: string | null;
  drill_palette_version: number | null;
  created_at: string | null;
};

function serializeTarget(row: TargetRow) {
  return {
    id: row.id,
    name: row.name ?? null,
    widthValue: Number(row.width_value),
    heightValue: Number(row.height_value),
    unit: row.unit === "mm" || row.unit === "cm" || row.unit === "in" ? row.unit : "in",
    qrSizeValue: Number(row.qr_size_value),
    scoringId: row.scoring_id ?? null,
    drillRecipe: row.drill_recipe ?? null,
    drillPaletteVersion: row.drill_palette_version ?? null,
    createdAt: row.created_at ? new Date(row.created_at).getTime() : 0,
  };
}

// GET with `?id=` resolves a single published target (used by the drill runner
// and the in-app scanner to rebuild a scanned id-only QR). GET with no id lists
// the signed-in user's own saved targets, so their ids load back on any device.
export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id")?.trim();

  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Catalog isn't configured." }, { status: 503 });

  if (id) {
    const { data } = await supabase.from("targets").select(TARGET_COLUMNS).eq("id", id).maybeSingle();
    if (!data) return NextResponse.json({ error: "Not found." }, { status: 404 });
    return NextResponse.json(serializeTarget(data as TargetRow));
  }

  // List this account's targets (RLS already restricts to the owner's rows).
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in to list your targets." }, { status: 401 });

  const { data, error } = await supabase
    .from("targets")
    .select(TARGET_COLUMNS)
    .eq("owner_id", user.id)
    .order("created_at", { ascending: false });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ targets: (data ?? []).map((row) => serializeTarget(row as TargetRow)) });
}

// Remove one of the signed-in user's saved targets from the account. RLS plus
// the owner_id filter ensure a user can only delete their own.
export async function DELETE(request: Request) {
  const id = new URL(request.url).searchParams.get("id")?.trim();
  if (!id) return NextResponse.json({ error: "Missing id." }, { status: 400 });

  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Catalog isn't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in to delete targets." }, { status: 401 });

  const { error } = await supabase.from("targets").delete().eq("id", id).eq("owner_id", user.id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
