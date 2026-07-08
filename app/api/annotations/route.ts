import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Shot = Record<string, unknown>;

// Save a scanned session's bullet annotations to the signed-in user's account.
// RLS enforces ownership; we just attach the authenticated user id. The stored
// rows accumulate a labeled shots-on-target dataset for training later.
export async function POST(request: Request) {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ error: "Accounts aren't configured." }, { status: 503 });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Sign in to save to your account." }, { status: 401 });

  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const shots = Array.isArray(body.shots) ? (body.shots as Shot[]) : null;
  if (!shots || shots.length === 0) {
    return NextResponse.json({ error: "Nothing to save — no shots in this session." }, { status: 400 });
  }

  const name = typeof body.name === "string" && body.name.trim() ? body.name.trim().slice(0, 120) : null;
  const target = body.target && typeof body.target === "object" ? body.target : null;

  const { data, error } = await supabase
    .from("bullet_annotations")
    .insert({ user_id: user.id, name, target, shots, shot_count: shots.length })
    .select("id")
    .single();
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true, id: data?.id ?? null, shotCount: shots.length });
}

// List the signed-in user's saved annotation sets (most recent first).
export async function GET() {
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ sessions: [] });

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ sessions: [] });

  const { data, error } = await supabase
    .from("bullet_annotations")
    .select("id,name,shot_count,created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(50);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ sessions: data ?? [] });
}
