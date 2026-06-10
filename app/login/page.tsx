import Link from "next/link";
import { redirect } from "next/navigation";
import { LoginForm } from "./LoginForm";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { getCurrentUser } from "@/app/lib/supabase/server";

export const metadata = { title: "Sign in · Trackr" };

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const params = await searchParams;
  const configured = isSupabaseConfigured();

  // Already signed in? Go to the account page.
  if (configured) {
    const user = await getCurrentUser();
    if (user) redirect("/account");
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-black px-4 text-white">
      <div className="w-full max-w-sm rounded-xl border border-gray-700 bg-neutral-950 p-6">
        <Link href="/" className="text-sm text-sky-300 hover:underline">
          ← Back to Trackr
        </Link>
        <h1 className="mt-3 text-xl font-semibold">Sign in to Trackr</h1>
        <p className="mt-1 text-sm text-gray-400">Save your sessions and unlock Pro features.</p>

        {params.error ? (
          <p className="mt-4 rounded-md border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-100">
            {params.error}
          </p>
        ) : null}

        <LoginForm configured={configured} />
      </div>
    </main>
  );
}
