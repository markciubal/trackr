import Link from "next/link";
import { redirect } from "next/navigation";
import { isCurrentUserAdmin } from "@/app/lib/admin";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { getCurrentUser } from "@/app/lib/supabase/server";
import { AdminClassifierPanel } from "./AdminClassifierPanel";

export const metadata = { title: "Shot classifier · Admin · Trackr" };
export const dynamic = "force-dynamic";

export default async function AdminPage() {
  // Gate server-side: signed-out users go to login; signed-in non-admins are
  // bounced home. When Supabase is unconfigured the app runs open, so the panel
  // is reachable locally (isCurrentUserAdmin returns true).
  if (isSupabaseConfigured()) {
    const user = await getCurrentUser();
    if (!user) redirect("/login");
    if (!(await isCurrentUserAdmin())) redirect("/");
  }

  return (
    <main className="min-h-screen bg-black px-4 py-10 text-white">
      <div className="mx-auto w-full max-w-5xl space-y-6">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-sm text-sky-300 hover:underline">
            ← Back to Trackr
          </Link>
          <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs font-semibold text-amber-200">
            Admin
          </span>
        </div>
        <header>
          <h1 className="text-2xl font-semibold">Shot classifier</h1>
          <p className="mt-1 text-sm text-gray-400">
            Load a target photo, click holes and non-holes to label patches, train a model in your browser, then
            publish it — every scanner picks up the latest published model on startup.
          </p>
        </header>
        <AdminClassifierPanel />
      </div>
    </main>
  );
}
