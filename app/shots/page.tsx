import Link from "next/link";
import { redirect } from "next/navigation";
import { getEntitlement } from "@/app/lib/entitlements";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { createSupabaseServerClient } from "@/app/lib/supabase/server";
import { type SessionMeta } from "@/app/lib/shots/stats";
import { ShotsBrowser } from "./ShotsBrowser";

export const metadata = { title: "Shot library · Trackr" };
export const dynamic = "force-dynamic";

export default async function ShotsPage() {
  if (!isSupabaseConfigured()) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-black px-4 text-white">
        <div className="w-full max-w-md rounded-xl border border-gray-700 bg-neutral-950 p-6">
          <h1 className="text-xl font-semibold">Shot library</h1>
          <p className="mt-2 text-sm text-amber-100">
            Accounts aren&apos;t configured, so there are no saved shots to browse.
          </p>
          <Link href="/" className="mt-4 inline-block text-sm text-sky-300 hover:underline">
            ← Back to Trackr
          </Link>
        </div>
      </main>
    );
  }

  const entitlement = await getEntitlement();
  if (!entitlement.userId) redirect("/login");

  const supabase = await createSupabaseServerClient();
  let sessions: SessionMeta[] = [];
  if (supabase) {
    const { data } = await supabase
      .from("bullet_annotations")
      .select("id,name,shot_count,created_at")
      .order("created_at", { ascending: false })
      .limit(200);
    sessions = (data ?? []) as SessionMeta[];
  }

  return (
    <main className="min-h-screen bg-black px-4 py-8 text-white">
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h1 className="text-xl font-semibold">Shot library</h1>
            <p className="mt-1 text-sm text-gray-400">
              Browse saved sessions, combine them, and see group statistics.
            </p>
          </div>
          <Link href="/" className="shrink-0 text-sm text-sky-300 hover:underline">
            ← Back to Trackr
          </Link>
        </div>
        <ShotsBrowser initialSessions={sessions} />
      </div>
    </main>
  );
}
