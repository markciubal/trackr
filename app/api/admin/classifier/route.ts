import { NextResponse } from "next/server";
import { isCurrentUserAdmin } from "@/app/lib/admin";
import { createSupabaseAdminClient } from "@/app/lib/supabase/admin";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";
import { isModelJSON } from "@/app/lib/training/jsonModel";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Admin: list published + draft model versions, newest first.
export async function GET() {
  if (!(await isCurrentUserAdmin())) {
    return NextResponse.json({ error: "Admins only." }, { status: 403 });
  }
  const supabase = await createSupabaseServerClient();
  if (!supabase) return NextResponse.json({ versions: [] });

  const { data, error } = await supabase
    .from("classifier_models")
    .select("id,version,kind,notes,is_published,created_at,payload")
    .order("version", { ascending: false })
    .limit(50);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const versions = (data ?? []).map((row) => ({
    id: row.id,
    version: row.version,
    kind: row.kind,
    notes: row.notes,
    isPublished: row.is_published,
    createdAt: row.created_at,
    // Surface quality metrics without shipping the full weight vector.
    meta: (row.payload as { meta?: unknown })?.meta ?? null,
  }));
  return NextResponse.json({ versions });
}

// Admin: publish a freshly trained model. Assigns the next version, marks it the
// single published row (unpublishing the prior), making it the model every
// scanner loads next. Uses the service-role client for the multi-row update, but
// only after verifying the caller is an admin.
export async function POST(request: Request) {
  if (!(await isCurrentUserAdmin())) {
    return NextResponse.json({ error: "Admins only." }, { status: 403 });
  }

  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const model = body.model;
  if (!isModelJSON(model)) {
    return NextResponse.json({ error: "Invalid model payload." }, { status: 400 });
  }
  const notes = typeof body.notes === "string" ? body.notes.slice(0, 500) : null;

  const admin = createSupabaseAdminClient();
  if (!admin) {
    return NextResponse.json({ error: "Server isn't configured for publishing." }, { status: 503 });
  }

  // Record who published, when available.
  const server = await createSupabaseServerClient();
  const userId = server ? (await server.auth.getUser()).data.user?.id ?? null : null;

  // Next version number.
  const { data: top } = await admin
    .from("classifier_models")
    .select("version")
    .order("version", { ascending: false })
    .limit(1)
    .maybeSingle();
  const nextVersion = (top?.version ?? 0) + 1;

  // Demote the currently-published model, then insert the new published one.
  await admin.from("classifier_models").update({ is_published: false }).eq("is_published", true);

  const { error } = await admin.from("classifier_models").insert({
    version: nextVersion,
    kind: model.kind,
    payload: model,
    notes,
    is_published: true,
    created_by: userId,
  });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ ok: true, version: nextVersion });
}
