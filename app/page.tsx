import Link from "next/link";

export const metadata = {
  title: "Trackr — measure shot groups from video",
  description:
    "Upload a video of your target and Trackr finds every hit, groups them, and gives you real-world group statistics — extreme spread, CEP, and bias from your aim point.",
};

const STEPS: { n: number; title: string; body: string }[] = [
  { n: 1, title: "Add your video", body: "Upload a clip of your target, or record with your phone camera." },
  { n: 2, title: "Mark the target", body: "Drag a box around the target so Trackr knows where to look." },
  { n: 3, title: "Set caliber & method", body: "Pick your caliber and how you'll tell Trackr the target's real size." },
  { n: 4, title: "Enter measurements", body: "A known width, a line of known length, or a QR — so pixels become inches." },
  { n: 5, title: "Scan", body: "Press Start. Hits appear on the target as they're detected." },
  { n: 6, title: "Map", body: "Review the auto-grouped shots; tweak groups, strays, and aim points." },
  { n: 7, title: "Review", body: "Group stats, the full shot table, and detector detail views." },
  { n: 8, title: "Save", body: "Save the session to your account to build your shot library." },
];

const FEATURES: { title: string; body: string }[] = [
  {
    title: "Finds the hits for you",
    body: "Change + audio detection spots impacts, ignores camera shake and flicker, and even splits touching bullets.",
  },
  {
    title: "Real group statistics",
    body: "Extreme spread, group size, mean radius, CEP, and per-group spread — in inches, cm, or mm.",
  },
  {
    title: "Aim-point normalized",
    body: "Set where each group was aiming and the library aligns sessions on intent — separating bias (accuracy) from spread (precision).",
  },
  {
    title: "Your shot library",
    body: "Browse saved sessions, combine them, and compare dispersion across days on one chart.",
  },
];

export default function HomePage() {
  return (
    <main className="min-h-screen bg-black text-white">
      {/* Hero */}
      <section className="mx-auto max-w-4xl px-4 pb-10 pt-12 text-center sm:pt-20">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400">Shot-group measurement</p>
        <h1 className="mt-3 text-3xl font-bold leading-tight sm:text-5xl">
          Measure your shot groups <span className="text-sky-400">from a video.</span>
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-sm text-gray-300 sm:text-base">
          Upload a clip of your target and Trackr finds every hit, groups them automatically, and gives you real-world
          statistics — extreme spread, group size, CEP, and your bias from where you aimed.
        </p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/scan"
            className="rounded-lg border border-sky-400/50 bg-sky-500/20 px-5 py-2.5 text-sm font-semibold text-sky-100 transition hover:bg-sky-500/30"
          >
            Start scanning →
          </Link>
          <Link
            href="/shots"
            className="rounded-lg border border-gray-700 px-5 py-2.5 text-sm font-medium text-gray-200 transition hover:bg-neutral-900"
          >
            Shot library
          </Link>
        </div>
        <p className="mt-3 text-xs text-gray-500">No account needed to scan — sign in only to save and compare sessions.</p>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-4xl px-4 py-10">
        <h2 className="text-center text-lg font-semibold sm:text-xl">How it works</h2>
        <p className="mt-1 text-center text-sm text-gray-400">Eight quick steps — the app walks you through each one.</p>
        <ol className="mt-6 grid gap-3 sm:grid-cols-2">
          {STEPS.map((step) => (
            <li
              key={step.n}
              className="flex gap-3 rounded-xl border border-gray-800 bg-neutral-950 p-4"
            >
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-sky-500/20 text-sm font-semibold text-sky-200">
                {step.n}
              </span>
              <div>
                <h3 className="text-sm font-semibold text-white">{step.title}</h3>
                <p className="mt-0.5 text-xs text-gray-400">{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <div className="mt-6 text-center">
          <Link
            href="/scan"
            className="inline-block rounded-lg border border-sky-400/50 bg-sky-500/20 px-5 py-2.5 text-sm font-semibold text-sky-100 transition hover:bg-sky-500/30"
          >
            Try it now →
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-4xl px-4 py-10">
        <h2 className="text-center text-lg font-semibold sm:text-xl">What you get</h2>
        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-xl border border-gray-800 bg-neutral-950 p-4">
              <h3 className="text-sm font-semibold text-white">{f.title}</h3>
              <p className="mt-1 text-xs text-gray-400">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Other tools */}
      <section className="mx-auto max-w-4xl px-4 pb-16 pt-6">
        <div className="grid gap-3 sm:grid-cols-2">
          <Link
            href="/targets/new"
            className="rounded-xl border border-gray-800 bg-neutral-950 p-4 transition hover:border-sky-500/40 hover:bg-neutral-900"
          >
            <h3 className="text-sm font-semibold text-white">Design a printable target →</h3>
            <p className="mt-1 text-xs text-gray-400">
              Build a target with a QR code so Trackr auto-calibrates from the print size.
            </p>
          </Link>
          <Link
            href="/drill"
            className="rounded-xl border border-gray-800 bg-neutral-950 p-4 transition hover:border-sky-500/40 hover:bg-neutral-900"
          >
            <h3 className="text-sm font-semibold text-white">Run a drill →</h3>
            <p className="mt-1 text-xs text-gray-400">Practice scenarios with colored/numbered zones.</p>
          </Link>
        </div>
      </section>
    </main>
  );
}
