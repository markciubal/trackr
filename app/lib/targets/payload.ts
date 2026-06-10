// Hybrid "smart target" QR payload.
//
// The QR encodes a short URL: {base}/t/{id}?w=&h=&u=&q=&s=&z=&v=
//   id  short catalog code (resolves to a Supabase row: rings, SKU, owner, ...)
//   w,h target width/height in `u`
//   u   units: mm | cm | in
//   q   printed side length of the QR square, in `u` (the calibration reference)
//   s   optional scoring/ring-layout id
//   z   optional drill layout — preferably a tiny {count, attrs, seed} recipe
//       that regenerates the zones on scan; legacy prints carry the full
//       per-zone encoding (see scenario.ts encodeRecipe / encodeZones)
//   pv  palette version for `z` (which frozen shape/color/pattern set to
//       regenerate from); absent ⇒ 1. Lets the palette evolve without breaking
//       targets already printed.
//   v   payload schema version
//
// The inline w/h/u/q let the app auto-calibrate with no network; the id unlocks
// the catalog/sales/anti-copy side when online. Scanning with a phone camera
// just opens the /t/{id} product page.

import {
  SCENARIO_PALETTE_VERSION,
  decodeRecipe,
  encodeRecipe,
  encodeZones,
  zonesFromParam,
  type ScenarioRecipe,
  type ScenarioZone,
} from "./scenario";

export type LinearUnit = "mm" | "cm" | "in";

export const TARGET_PAYLOAD_VERSION = 1;
export const TARGET_PATH_PREFIX = "/t/";

export type TargetPayload = {
  id: string;
  unit: LinearUnit;
  widthValue?: number;
  heightValue?: number;
  qrSizeValue?: number;
  scoringId?: string;
  drill?: ScenarioRecipe; // preferred: tiny recipe the zones regenerate from
  zones?: ScenarioZone[]; // resolved zones (from the recipe, or a legacy full encoding)
  paletteVersion?: number; // which frozen palette the drill `z` decodes against
  version: number;
};

const PER_INCH: Record<LinearUnit, number> = { in: 1, cm: 2.54, mm: 25.4 };

export function toInches(value: number, unit: LinearUnit): number {
  return value / PER_INCH[unit];
}

export function fromInches(valueInches: number, unit: LinearUnit): number {
  return valueInches * PER_INCH[unit];
}

function normalizeUnit(unit: string | null | undefined): LinearUnit {
  return unit === "mm" || unit === "cm" || unit === "in" ? unit : "in";
}

// True when the payload carries everything needed to calibrate offline.
export function hasInlineCalibration(payload: TargetPayload): boolean {
  return (
    typeof payload.widthValue === "number" &&
    payload.widthValue > 0 &&
    typeof payload.heightValue === "number" &&
    payload.heightValue > 0 &&
    typeof payload.qrSizeValue === "number" &&
    payload.qrSizeValue > 0
  );
}

export function encodeTargetPayload(payload: TargetPayload, baseUrl: string): string {
  const base = (baseUrl || "").replace(/\/+$/, "");
  const params = new URLSearchParams();
  if (typeof payload.widthValue === "number") params.set("w", trimNumber(payload.widthValue));
  if (typeof payload.heightValue === "number") params.set("h", trimNumber(payload.heightValue));
  params.set("u", payload.unit);
  if (typeof payload.qrSizeValue === "number") params.set("q", trimNumber(payload.qrSizeValue));
  if (payload.scoringId) params.set("s", payload.scoringId);
  const paletteVersion = payload.paletteVersion ?? SCENARIO_PALETTE_VERSION;
  if (payload.drill) {
    params.set("z", encodeRecipe(payload.drill));
    params.set("pv", String(paletteVersion));
  } else if (payload.zones?.length) {
    params.set("z", encodeZones(payload.zones, paletteVersion));
    params.set("pv", String(paletteVersion));
  }
  params.set("v", String(payload.version || TARGET_PAYLOAD_VERSION));
  return `${base}${TARGET_PATH_PREFIX}${encodeURIComponent(payload.id)}?${params.toString()}`;
}

// Parse a scanned string. Accepts the full URL form; returns null if it isn't a
// recognizable target payload. id-only payloads (no inline calibration) are
// still returned so the app can resolve them from the catalog.
export function decodeTargetPayload(text: string): TargetPayload | null {
  if (!text) return null;
  let url: URL;
  try {
    url = new URL(text.trim());
  } catch {
    return null;
  }
  const match = url.pathname.match(/\/t\/([^/]+)\/?$/);
  if (!match) return null;
  const id = safeDecode(match[1]);
  if (!id) return null;

  const p = url.searchParams;
  const width = numberOrUndefined(p.get("w"));
  const height = numberOrUndefined(p.get("h"));
  const qr = numberOrUndefined(p.get("q"));
  const rawZ = p.get("z");
  // Absent pv ⇒ palette v1 (the first, locked palette).
  const paletteVersion = numberOrUndefined(p.get("pv")) ?? 1;

  return {
    id,
    unit: normalizeUnit(p.get("u")),
    widthValue: width,
    heightValue: height,
    qrSizeValue: qr,
    scoringId: p.get("s") ?? undefined,
    drill: decodeRecipe(rawZ) ?? undefined,
    zones: zonesFromParam(rawZ, paletteVersion) ?? undefined,
    paletteVersion,
    version: numberOrUndefined(p.get("v")) ?? TARGET_PAYLOAD_VERSION,
  };
}

// px-per-inch from a detected QR side length (pixels) + its known printed size.
export function pixelsPerInchFromQr(qrSidePx: number, qrSizeValue: number, unit: LinearUnit): number | null {
  const qrInches = toInches(qrSizeValue, unit);
  if (!Number.isFinite(qrSidePx) || qrSidePx <= 0 || !Number.isFinite(qrInches) || qrInches <= 0) return null;
  return qrSidePx / qrInches;
}

// Crockford base32 (no I/L/O/U) for human-readable, unambiguous short codes.
const ID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

export function generateTargetId(length = 7): string {
  const bytes = new Uint8Array(length);
  const cryptoObj = typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
  if (cryptoObj?.getRandomValues) {
    cryptoObj.getRandomValues(bytes);
  } else {
    for (let i = 0; i < length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  }
  let out = "";
  for (let i = 0; i < length; i += 1) out += ID_ALPHABET[bytes[i] % ID_ALPHABET.length];
  return out;
}

function trimNumber(value: number): string {
  return String(Number(value.toFixed(4)));
}

function numberOrUndefined(raw: string | null): number | undefined {
  if (raw === null || raw.trim() === "") return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
