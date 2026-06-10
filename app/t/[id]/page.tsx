import Link from "next/link";
import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { resolveTarget } from "@/app/lib/targets/resolve";
import { type LinearUnit } from "@/app/lib/targets/payload";
import { zonesFromParam } from "@/app/lib/targets/scenario";

export const metadata = { title: "Smart target · Trackr" };

type SearchParams = Record<string, string | string[] | undefined>;

function readQuerySpec(sp: SearchParams) {
  const num = (key: string) => {
    const raw = Array.isArray(sp[key]) ? sp[key]?.[0] : sp[key];
    const value = Number(raw);
    return Number.isFinite(value) && value > 0 ? value : null;
  };
  const rawUnit = Array.isArray(sp.u) ? sp.u[0] : sp.u;
  const unit: LinearUnit = rawUnit === "mm" || rawUnit === "cm" || rawUnit === "in" ? rawUnit : "in";
  const widthValue = num("w");
  const heightValue = num("h");
  const qrSizeValue = num("q");
  if (widthValue && heightValue && qrSizeValue) {
    return { name: null as string | null, widthValue, heightValue, unit, qrSizeValue, isPaid: false };
  }
  return null;
}

export default async function TargetLandingPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<SearchParams>;
}) {
  const { id } = await params;
  const sp = await searchParams;

  const record = isSupabaseConfigured() ? await resolveTarget(id) : null;
  const spec = record ?? readQuerySpec(sp);
  const openInTrackrHref = `/?t=${encodeURIComponent(id)}`;

  // Drill layout: new id-only QRs resolve the recipe from the catalog by id;
  // legacy prints carried it inline as ?z=. Either way we hand the recipe to the
  // drill page so the device rebuilds the exact zones without another lookup.
  const legacyZ = Array.isArray(sp.z) ? sp.z[0] : sp.z;
  const drillRecipe = record?.drillRecipe ?? legacyZ ?? null;
  const drillPaletteVersion = record?.drillPaletteVersion ?? undefined;
  const drillZones = zonesFromParam(drillRecipe, drillPaletteVersion);
  const drillHref = drillRecipe
    ? `/drill?id=${encodeURIComponent(id)}&z=${encodeURIComponent(drillRecipe)}`
    : null;

  return (
    <main className="flex min-h-screen items-center justify-center bg-black px-4 py-12 text-white">
      <div className="w-full max-w-md space-y-5">
        <Link href="/" className="text-sm text-sky-300 hover:underline">
          ← Trackr
        </Link>

        <div className="rounded-xl border border-gray-700 bg-neutral-950 p-6">
          <p className="text-[11px] uppercase tracking-wide text-gray-400">Smart Target</p>
          <h1 className="mt-1 text-xl font-semibold">{spec?.name ?? "Trackr Target"}</h1>
          <p className="mt-1 font-mono text-xs text-gray-500">{id}</p>

          {spec ? (
            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <dt className="text-gray-400">Size</dt>
              <dd>
                {spec.widthValue} × {spec.heightValue} {spec.unit}
              </dd>
              <dt className="text-gray-400">QR reference</dt>
              <dd>
                {spec.qrSizeValue} {spec.unit}
              </dd>
            </dl>
          ) : (
            <p className="mt-4 text-sm text-amber-100">
              We couldn&apos;t find this target&apos;s details. Open it in Trackr and the QR&apos;s built-in size data
              will calibrate automatically.
            </p>
          )}

          <div className="mt-5 flex flex-wrap gap-2">
            <Link
              href={openInTrackrHref}
              className="inline-flex items-center justify-center rounded-md border border-sky-400/40 bg-sky-500/15 px-4 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
            >
              Open in Trackr →
            </Link>
            {drillZones && drillHref ? (
              <Link
                href={drillHref}
                className="inline-flex items-center justify-center rounded-md border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-100 transition hover:bg-emerald-500/20"
              >
                Run drill ({drillZones.length} zones) →
              </Link>
            ) : null}
          </div>

          {record?.isPaid ? (
            <p className="mt-3 text-xs text-gray-400">
              This is a premium target. Pro members get full scoring and analysis.
            </p>
          ) : null}
        </div>

        <p className="text-center text-xs text-gray-500">
          Point Trackr at this target — the QR sets the scale automatically, so there&apos;s nothing to calibrate.
        </p>
      </div>
    </main>
  );
}
