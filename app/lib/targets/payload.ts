// "Smart target" QR payload.
//
// QRs are SELF-CONTAINED: everything a scan needs rides inside the URL itself —
// no account, no catalog lookup, no saved id resolution.
//
//   • Drill targets:    {base}/drill?z=&pv=&w=&h=&u=&q=&v=
//     Scanning goes STRAIGHT to the scenario drill; the tiny {count, attrs,
//     seed} recipe in `z` regenerates the exact printed zones on any device,
//     and w/h/u/q carry the physical size so the QR still calibrates the
//     scanner. No id at all.
//   • Other targets:    {base}/t/{id}?w=&h=&u=&q=&s=&v=
//     Calibration is inline; the id is only a human-readable label for the
//     local "My targets" list, never something that must resolve anywhere.
//
// The query string forces the QR into byte mode (bigger than the old id-only
// alphanumeric form), which is a deliberate trade: a denser code that works
// with zero infrastructure beats a tiny one that needs a catalog.
//
// LEGACY: earlier prints encoded an id-only uppercase URL (HTTPS://…/T/{ID},
// resolved from the catalog/local store) or the same /t/{id}?w=… inline form.
// decodeTargetPayload still parses all of them so prints in the wild keep
// working forever.

import {
  decodeRecipe,
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

// Build the printed QR string for a DRILL target: a direct /drill URL carrying
// the zone recipe + physical size inline. Fully self-contained — scanning it
// opens the exact drill on any device with no account or lookup.
export function encodeDrillPayload(
  args: {
    recipe: string; // encodeRecipe() output
    paletteVersion: number;
    unit: LinearUnit;
    widthValue: number;
    heightValue: number;
    qrSizeValue: number;
  },
  baseUrl: string,
): string {
  const base = (baseUrl || "").replace(/\/+$/, "");
  const params = new URLSearchParams({
    z: args.recipe,
    pv: String(args.paletteVersion),
    w: String(args.widthValue),
    h: String(args.heightValue),
    u: args.unit,
    q: String(args.qrSizeValue),
    v: String(TARGET_PAYLOAD_VERSION),
  });
  return `${base}/drill?${params.toString()}`;
}

// Build the printed QR string for a non-drill target: /t/{id} with calibration
// inline, so the info page and the scanner both work without any stored record.
export function encodeTargetPayload(
  payload: TargetPayload & { widthValue: number; heightValue: number; qrSizeValue: number },
  baseUrl: string,
): string {
  const base = (baseUrl || "").replace(/\/+$/, "");
  const params = new URLSearchParams({
    w: String(payload.widthValue),
    h: String(payload.heightValue),
    u: payload.unit,
    q: String(payload.qrSizeValue),
    v: String(payload.version),
  });
  if (payload.scoringId) params.set("s", payload.scoringId);
  return `${base}${TARGET_PATH_PREFIX}${payload.id}?${params.toString()}`;
}

// Parse a scanned string. Handles every form we've ever printed:
//   • drill direct:  https://trkr.gg/drill?z=&pv=&w=&h=&u=&q=&v=  (no id at all)
//   • inline query:  https://trkr.gg/t/ABC1234?w=&h=&u=&q=&s=&v=
//   • legacy id-only: HTTPS://TRKR.GG/T/ABC1234  (resolved from catalog/local store)
//   • bare no scheme: TRKR.GG/T/ABC1234          (some camera apps strip it)
// Returns null if it isn't a recognizable target payload.
export function decodeTargetPayload(text: string): TargetPayload | null {
  if (!text) return null;
  const trimmed = text.trim();
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    // No scheme (e.g. "TRKR.GG/T/ABC1234") — retry assuming https.
    try {
      url = new URL(`https://${trimmed}`);
    } catch {
      return null;
    }
  }

  // Direct drill form: no id — the recipe IS the identity. Synthesize a stable
  // local label from it so target lists/stores still have a key.
  if (/\/drill\/?$/i.test(url.pathname)) {
    const p = url.searchParams;
    const rawZ = p.get("z");
    const paletteVersion = numberOrUndefined(p.get("pv")) ?? 1;
    const zones = zonesFromParam(rawZ, paletteVersion);
    if (!rawZ || !zones?.length) return null;
    return {
      id: `DRILL-${rawZ.toUpperCase()}`,
      unit: normalizeUnit(p.get("u")),
      widthValue: numberOrUndefined(p.get("w")),
      heightValue: numberOrUndefined(p.get("h")),
      qrSizeValue: numberOrUndefined(p.get("q")),
      scoringId: "drill",
      drill: decodeRecipe(rawZ) ?? undefined,
      zones,
      paletteVersion,
      version: numberOrUndefined(p.get("v")) ?? TARGET_PAYLOAD_VERSION,
    };
  }

  // Case-insensitive, un-anchored: matches both /t/ and /T/, stops at ?/#.
  const match = url.pathname.match(/\/t\/([^/?#]+)/i);
  if (!match) return null;
  // The id is uppercase Crockford base32 in every form we emit — capture it
  // verbatim (lookups are case-sensitive). safeDecode is harmless on un-encoded
  // ids and still unwraps any percent-encoding in legacy prints.
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
