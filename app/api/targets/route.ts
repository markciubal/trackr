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
    is_published: true,
  });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
