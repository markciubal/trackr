import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Fetch one saved session in full (target calibration + every shot). RLS limits
// this to the owner (or an admin), so we don't filter by user_id here.
export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Accounts aren't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in to view saved shots." }, { status: 401 });

  const { data, error } = await supabase
    .from("bullet_annotations")
    .select("id,name,target,shots,shot_count,created_at")
    .eq("id", id)
    .single();
  if (error) {
    const missing = error.code === "PGRST116";
    return NextResponse.json({ error: missing ? "Not found." : error.message }, { status: missing ? 404 : 500 });
  }

  return NextResponse.json({ session: data });
}

// Delete a saved session (RLS enforces ownership).
export async function DELETE(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Accounts aren't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in." }, { status: 401 });

  const { error } = await supabase.from("bullet_annotations").delete().eq("id", id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true });
}
