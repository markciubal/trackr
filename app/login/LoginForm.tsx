"use client";

import { useState } from "react";
import { createSupabaseBrowserClient } from "@/app/lib/supabase/client";

type Provider = "google" | "github";
type Busy = Provider | "magic" | null;

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4">
    <path
      fill="#EA4335"
      d="M12 10.2v3.9h5.5c-.24 1.4-1.66 4.1-5.5 4.1a6.2 6.2 0 1 1 0-12.4c1.93 0 3.23.82 3.97 1.53l2.7-2.6C16.96 2.6 14.7 1.6 12 1.6a10.4 10.4 0 1 0 0 20.8c6 0 9.98-4.2 9.98-10.13 0-.68-.07-1.2-.16-1.72H12z"
    />
  </svg>
);

const GitHubIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" className="h-4 w-4 fill-current">
    <path d="M12 1.8a10.2 10.2 0 0 0-3.22 19.88c.5.1.69-.22.69-.48v-1.7c-2.83.62-3.43-1.36-3.43-1.36-.46-1.18-1.13-1.5-1.13-1.5-.93-.63.07-.62.07-.62 1.02.07 1.56 1.05 1.56 1.05.91 1.56 2.38 1.11 2.96.85.09-.66.36-1.11.65-1.36-2.26-.26-4.64-1.13-4.64-5.02 0-1.11.4-2.02 1.04-2.73-.1-.26-.45-1.29.1-2.69 0 0 .85-.27 2.79 1.04a9.6 9.6 0 0 1 5.08 0c1.94-1.31 2.79-1.04 2.79-1.04.55 1.4.2 2.43.1 2.69.65.71 1.04 1.62 1.04 2.73 0 3.9-2.38 4.76-4.65 5.01.37.32.69.94.69 1.9v2.82c0 .27.18.59.7.48A10.2 10.2 0 0 0 12 1.8z" />
  </svg>
);

export function LoginForm({ configured }: { configured: boolean }) {
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [email, setEmail] = useState("");

  const callbackUrl = () =>
    typeof window === "undefined" ? "" : `${window.location.origin}/auth/callback?next=/account`;

  const oauth = async (provider: Provider) => {
    setError(null);
    setMessage(null);
    const supabase = createSupabaseBrowserClient();
    if (!supabase) {
      setError("Authentication is not configured yet.");
      return;
    }
    setBusy(provider);
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: { redirectTo: callbackUrl() },
    });
    // On success the browser navigates to the provider; only errors land here.
    if (error) {
      setError(error.message);
      setBusy(null);
    }
  };

  const magicLink = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setMessage(null);
    const trimmed = email.trim();
    if (!trimmed) return;
    const supabase = createSupabaseBrowserClient();
    if (!supabase) {
      setError("Authentication is not configured yet.");
      return;
    }
    setBusy("magic");
    const { error } = await supabase.auth.signInWithOtp({
      email: trimmed,
      options: { emailRedirectTo: callbackUrl() },
    });
    setBusy(null);
    if (error) setError(error.message);
    else setMessage(`We sent a sign-in link to ${trimmed}. Check your inbox.`);
  };

  const disabled = !configured || busy !== null;

  return (
    <div className="mt-5">
      {!configured ? (
        <p className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-100">
          Authentication isn&apos;t configured yet. Add your Supabase keys to <code>.env.local</code> (see{" "}
          <code>.env.example</code>) and enable the Google/GitHub providers in your Supabase dashboard.
        </p>
      ) : null}

      {error ? (
        <p className="mb-4 rounded-md border border-rose-500/40 bg-rose-500/10 p-3 text-xs text-rose-100">{error}</p>
      ) : null}
      {message ? (
        <p className="mb-4 rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3 text-xs text-emerald-100">
          {message}
        </p>
      ) : null}

      <div className="space-y-2">
        <button
          type="button"
          onClick={() => oauth("google")}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-700 bg-white px-3 py-2.5 text-sm font-medium text-gray-900 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <GoogleIcon />
          {busy === "google" ? "Redirecting…" : "Continue with Google"}
        </button>
        <button
          type="button"
          onClick={() => oauth("github")}
          disabled={disabled}
          className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-700 bg-neutral-900 px-3 py-2.5 text-sm font-medium text-gray-100 transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <GitHubIcon />
          {busy === "github" ? "Redirecting…" : "Continue with GitHub"}
        </button>
      </div>

      <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-wide text-gray-500">
        <span className="h-px flex-1 bg-gray-800" />
        or email me a link
        <span className="h-px flex-1 bg-gray-800" />
      </div>

      <form onSubmit={magicLink} className="space-y-2">
        <label className="block">
          <span className="sr-only">Email</span>
          <input
            name="email"
            type="email"
            required
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            disabled={disabled}
            className="w-full rounded-md border border-gray-700 bg-black px-3 py-2.5 text-sm outline-none ring-sky-400/40 focus:ring-2 disabled:opacity-50"
          />
        </label>
        <button
          type="submit"
          disabled={disabled}
          className="w-full rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-2.5 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy === "magic" ? "Sending…" : "Send magic link"}
        </button>
      </form>

      <p className="mt-4 text-[11px] leading-relaxed text-gray-500">
        No passwords. We&apos;ll create your account automatically the first time you sign in.
      </p>
    </div>
  );
}
