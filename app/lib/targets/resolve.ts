import "server-only";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";
import type { LinearUnit } from "./payload";

export type TargetRecord = {
  id: string;
  name: string | null;
  widthValue: number;
  heightValue: number;
  unit: LinearUnit;
  qrSizeValue: number;
  scoringId: string | null;
  drillRecipe: string | null; // encodeRecipe() output; rebuilds zones by id
  drillPaletteVersion: number | null;
  isPaid: boolean;
};

// Resolve the catalog row for a printed target's short id. Returns null when
// Supabase isn't configured or the id isn't found (callers fall back to the
// self-describing query params baked into the QR).
export async function resolveTarget(id: string): Promise<TargetRecord | null> {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return null;

  const { data } = await supabase
    .from("targets")
    .select("id,name,width_value,height_value,unit,qr_size_value,scoring_id,drill_recipe,drill_palette_version,is_paid")
    .eq("id", id)
    .maybeSingle();
  if (!data) return null;

  const unit: LinearUnit = data.unit === "mm" || data.unit === "cm" || data.unit === "in" ? data.unit : "in";
  return {
    id: data.id,
    name: data.name ?? null,
    widthValue: Number(data.width_value),
    heightValue: Number(data.height_value),
    unit,
    qrSizeValue: Number(data.qr_size_value),
    scoringId: data.scoring_id ?? null,
    drillRecipe: data.drill_recipe ?? null,
    drillPaletteVersion:
      data.drill_palette_version === null || data.drill_palette_version === undefined
        ? null
        : Number(data.drill_palette_version),
    isPaid: Boolean(data.is_paid),
  };
}
