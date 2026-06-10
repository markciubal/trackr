// "Smart target" QR payload.
//
// The QR encodes the shortest possible URL: {base}/t/{id} — UPPERCASED, e.g.
// HTTPS://TRKR.GG/T/ABC1234. Uppercase + only the symbols `:/.` keeps the string
// inside the QR "alphanumeric" charset (5.5 bits/char vs 8 in byte mode), so a
// 25-char URL fits a version-1 (21×21) QR. Fewer, bigger modules ⇒ it scans from
// much farther away. The id is all the QR carries; everything else (calibration
// w/h/u/q, drill layout) is resolved by id from the Supabase catalog or local
// store. Scanning with a phone camera just opens the /t/{id} product page (a
// middleware redirect maps the uppercase /T/ path to the lowercase route).
//
// LEGACY: older prints encode a lowercase URL with an inline query string
//   {base}/t/{id}?w=&h=&u=&q=&s=&z=&pv=&v=
// carrying self-describing calibration + drill recipe. decodeTargetPayload still
// parses that form (detected by the presence of query params) so prints already
// in the wild keep working forever.

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

// Build the printed QR string: an id-only URL, uppercased so the whole thing
// stays in the QR alphanumeric charset (version-1 21×21 QR ⇒ scans from a
// distance). Calibration + drill layout are resolved by id, not baked in. The id
// is already uppercase Crockford base32, so toUpperCase only normalizes the
// scheme/host/path and never corrupts it.
export function encodeTargetPayload(payload: TargetPayload, baseUrl: string): string {
  const base = (baseUrl || "").replace(/\/+$/, "");
  return `${base}${TARGET_PATH_PREFIX}${payload.id}`.toUpperCase();
}

// Parse a scanned string. Handles all three forms:
//   • new id-only:   HTTPS://TRKR.GG/T/ABC1234       (no query string)
//   • legacy query:  https://trkr.gg/t/ABC1234?w=&h=&u=&q=&s=&z=&pv=&v=
//   • bare no scheme: TRKR.GG/T/ABC1234              (some camera apps strip it)
// Returns null if it isn't a recognizable target payload. id-only payloads carry
// no inline calibration — the app resolves those from the catalog / local store.
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
