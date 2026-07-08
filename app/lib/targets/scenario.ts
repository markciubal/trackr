// Shooting scenarios for smart targets. A target carries a set of zones, each
// identified by some mix of shape / color / number / fill pattern. The drill
// announces a sequence that refers to zones by any one of those attributes
// ("Red", "12", "Circle", "Striped") and the shooter must hit them in the
// called order. The same zones are detected in the frame, so a hit maps back
// to a zone via resolveZoneAtPoint.
//
// Which attributes are in play is a per-target choice (e.g. a black-and-white
// printer can't reproduce colors, so those users turn colors off and use
// patterns instead). Callouts only ever use attributes whose values are unique
// across the zone set, so loading an older target degrades gracefully.

export type ZoneShape =
  | "circle"
  | "square"
  | "triangle"
  | "diamond"
  | "star"
  | "pentagon"
  | "hexagon"
  | "cross"
  | "octagon";

export type ZoneColorName =
  | "red"
  | "orange"
  | "yellow"
  | "green"
  | "blue"
  | "purple"
  | "pink"
  | "cyan"
  | "brown"
  | "black"; // neutral fill used when the color attribute is disabled (B&W printers)

export type ZoneColor = { name: ZoneColorName; hex: string; spoken: string };

// Fill patterns: printable on any black-and-white printer, so they stand in
// for colors when that attribute is off. "solid" doubles as the no-pattern
// fill when the pattern attribute is disabled.
export type ZonePattern =
  | "solid"
  | "striped"
  | "banded"
  | "dotted"
  | "ringed"
  | "checkered"
  | "hatched"
  | "zigzag"
  | "grid";

// ---- Palette versions (FROZEN WIRE FORMAT) ----
//
// A printed drill QR carries only a recipe ({count, attributes, seed}) or a
// per-zone index encoding, and regenerates its zones on scan from the palette
// matching the QR's palette version. The arrays below are therefore a wire
// format: their CONTENTS and ORDER are load-bearing. Never edit an existing
// version's arrays — doing so silently rewrites every target already printed.
// To evolve the palette: add a PALETTE_V2 literal, register it in
// SCENARIO_PALETTES, bump SCENARIO_PALETTE_VERSION, and leave v1 untouched.
export const SCENARIO_PALETTE_VERSION = 1;

export type ScenarioPalette = {
  shapes: ZoneShape[];
  colorNames: Exclude<ZoneColorName, "black">[];
  patterns: ZonePattern[];
};

// v1 — locked. Do not modify.
const PALETTE_V1: ScenarioPalette = {
  shapes: ["circle", "square", "triangle", "diamond", "star", "pentagon", "hexagon", "cross", "octagon"],
  colorNames: ["red", "orange", "yellow", "green", "blue", "purple", "pink", "cyan", "brown"],
  patterns: ["solid", "striped", "banded", "dotted", "ringed", "checkered", "hatched", "zigzag", "grid"],
};

const SCENARIO_PALETTES: Record<number, ScenarioPalette> = { 1: PALETTE_V1 };

// The palette a given version's zones were generated from. Unknown/newer
// versions fall back to the latest this build knows (best effort).
export function paletteForVersion(version: number): ScenarioPalette {
  return SCENARIO_PALETTES[version] ?? SCENARIO_PALETTES[SCENARIO_PALETTE_VERSION];
}

const CURRENT_PALETTE = paletteForVersion(SCENARIO_PALETTE_VERSION);

function maxZonesForPalette(palette: ScenarioPalette): number {
  return Math.min(palette.shapes.length, palette.colorNames.length, palette.patterns.length);
}

// Appearance for every color the registry has ever defined, so zones from an
// older palette still render even if a future version drops a color.
const COLOR_REGISTRY: Record<Exclude<ZoneColorName, "black">, ZoneColor> = {
  red: { name: "red", hex: "#ef4444", spoken: "Red" },
  orange: { name: "orange", hex: "#f97316", spoken: "Orange" },
  yellow: { name: "yellow", hex: "#eab308", spoken: "Yellow" },
  green: { name: "green", hex: "#22c55e", spoken: "Green" },
  blue: { name: "blue", hex: "#3b82f6", spoken: "Blue" },
  purple: { name: "purple", hex: "#a855f7", spoken: "Purple" },
  pink: { name: "pink", hex: "#ec4899", spoken: "Pink" },
  cyan: { name: "cyan", hex: "#06b6d4", spoken: "Cyan" },
  brown: { name: "brown", hex: "#92400e", spoken: "Brown" },
};

const NEUTRAL_ZONE_COLOR: ZoneColor = { name: "black", hex: "#262626", spoken: "Black" };

// Color-blind-friendly hues for the same names. Tuned from the Okabe–Ito / IBM
// color-safe palettes for maximum separation under common color-vision
// deficiency, while staying close enough to each name to remain recognizable.
// The names (and therefore the spoken callouts + the QR wire format) are
// unchanged — only the rendered hex differs when the option is on.
const COLOR_REGISTRY_CB: Record<Exclude<ZoneColorName, "black">, ZoneColor> = {
  red: { name: "red", hex: "#d55e00", spoken: "Red" }, // vermillion
  orange: { name: "orange", hex: "#e69f00", spoken: "Orange" },
  yellow: { name: "yellow", hex: "#f0e442", spoken: "Yellow" },
  green: { name: "green", hex: "#009e73", spoken: "Green" }, // bluish green
  blue: { name: "blue", hex: "#0072b2", spoken: "Blue" },
  purple: { name: "purple", hex: "#785ef0", spoken: "Purple" },
  pink: { name: "pink", hex: "#cc79a7", spoken: "Pink" }, // reddish purple
  cyan: { name: "cyan", hex: "#56b4e9", spoken: "Cyan" }, // sky blue
  brown: { name: "brown", hex: "#7f4d1c", spoken: "Brown" },
};

// Current-version palettes, exported for the designer UI + ScenarioBoard. These
// follow whatever SCENARIO_PALETTE_VERSION points at; the frozen per-version
// copies above are what the codecs actually use.
export const ZONE_SHAPES: ZoneShape[] = CURRENT_PALETTE.shapes;
export const ZONE_PATTERNS: ZonePattern[] = CURRENT_PALETTE.patterns;
export const ZONE_COLORS: ZoneColor[] = CURRENT_PALETTE.colorNames.map((name) => COLOR_REGISTRY[name]);

// All-unique attributes means a zone count can't exceed the smaller palette.
export const MAX_SCENARIO_ZONES = maxZonesForPalette(CURRENT_PALETTE);

export type ScenarioZone = {
  id: string;
  shape: ZoneShape;
  color: ZoneColorName;
  number: number; // 0 = numbers disabled for this target
  pattern?: ZonePattern; // absent on older saved zones → "solid"
  // Normalized position/size within the target (0..1), origin top-left.
  cx: number;
  cy: number;
  radius: number;
};

// Which zone attributes a target uses, both for rendering and callouts.
export type ScenarioAttribute = "shape" | "color" | "number" | "pattern";

export const DEFAULT_SCENARIO_ATTRIBUTES: ScenarioAttribute[] = ["shape", "color", "number"];

// For UI toggle rows (ordered: visual attributes first, B&W-friendly ones together).
export const SCENARIO_ATTRIBUTE_OPTIONS: { id: ScenarioAttribute; label: string }[] = [
  { id: "shape", label: "Shapes" },
  { id: "color", label: "Colors" },
  { id: "pattern", label: "Patterns" },
  { id: "number", label: "Numbers" },
];

export type PromptKind = ScenarioAttribute;

export type DrillStep = {
  zoneId: string;
  kind: PromptKind;
  label: string; // shown on screen
  spoken: string; // read aloud
};

export type CalloutMode = "sequence" | "reactive" | "timed";

// ---- Timed callouts ----
//
// "Timed" mode calls zones at unpredictable moments spread across a drill
// window that itself varies run to run. Given an approximate length, the
// actual span is jittered ±25%, then each gap between callouts gets a random
// weight — so neither the total time nor the rhythm is predictable, but a
// minimum gap keeps speech from overlapping and gives each call a fair
// reaction window.
export const TIMED_MIN_GAP_MS = 1800;
export const TIMED_SPAN_JITTER = 0.25;
export const TIMED_LEAD_IN_MIN_MS = 800;
export const TIMED_LEAD_IN_MAX_MS = 2500;
export const TIMED_END_GRACE_MS = 4000;

export type TimedSchedule = {
  calloutAtMs: number[]; // one entry per step, ms from drill start
  endAtMs: number; // when the final step times out and the drill ends
  spanMs: number; // the actual (jittered) span this run
};

export function generateTimedSchedule(
  stepCount: number,
  approxSpanSec: number,
  rng: () => number = Math.random,
): TimedSchedule {
  const count = Math.max(1, Math.floor(stepCount));
  const jitter = 1 - TIMED_SPAN_JITTER + rng() * TIMED_SPAN_JITTER * 2;
  const spanMs = Math.max(count * TIMED_MIN_GAP_MS, approxSpanSec * 1000 * jitter);
  const leadInMs = TIMED_LEAD_IN_MIN_MS + rng() * (TIMED_LEAD_IN_MAX_MS - TIMED_LEAD_IN_MIN_MS);
  // Random gap weights (floored away from zero so no two calls collide), then
  // normalized over whatever span remains after the guaranteed minimum gaps.
  const weights = Array.from({ length: count }, () => 0.35 + rng());
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const spreadMs = Math.max(0, spanMs - leadInMs - TIMED_MIN_GAP_MS * (count - 1));
  const calloutAtMs: number[] = [];
  let t = leadInMs;
  weights.forEach((weight, i) => {
    calloutAtMs.push(Math.round(t));
    if (i < count - 1) t += TIMED_MIN_GAP_MS + (spreadMs * weight) / totalWeight;
  });
  return { calloutAtMs, endAtMs: Math.round(t + TIMED_END_GRACE_MS), spanMs: Math.round(spanMs) };
}

export function colorMeta(name: ZoneColorName, colorBlind = false): ZoneColor {
  if (name === "black") return NEUTRAL_ZONE_COLOR;
  const registry = colorBlind ? COLOR_REGISTRY_CB : COLOR_REGISTRY;
  return registry[name] ?? registry.red;
}

export function shapeSpoken(shape: ZoneShape): string {
  return shape.charAt(0).toUpperCase() + shape.slice(1);
}

export function patternSpoken(pattern: ZonePattern): string {
  return pattern.charAt(0).toUpperCase() + pattern.slice(1);
}

export function zonePattern(zone: ScenarioZone): ZonePattern {
  return zone.pattern ?? "solid";
}

// Which attributes can identify a zone in this set — i.e. their values are
// unique across all zones. This is derived (not stored) so older saved targets
// and hand-edited zone sets degrade gracefully.
export function attributesFromZones(zones: ScenarioZone[]): ScenarioAttribute[] {
  if (zones.length === 0) return [...DEFAULT_SCENARIO_ATTRIBUTES];
  const unique = (values: unknown[]) => new Set(values).size === values.length;
  const attrs: ScenarioAttribute[] = [];
  if (unique(zones.map((zone) => zone.shape))) attrs.push("shape");
  if (zones.every((zone) => zone.color !== "black") && unique(zones.map((zone) => zone.color))) attrs.push("color");
  if (zones.every((zone) => zone.number > 0) && unique(zones.map((zone) => zone.number))) attrs.push("number");
  if (unique(zones.map(zonePattern)) && zones.some((zone) => zonePattern(zone) !== "solid")) attrs.push("pattern");
  return attrs.length > 0 ? attrs : ["number"];
}

// Small deterministic PRNG (mulberry32). Seeded generation lets a QR carry
// just the seed instead of the full zone list.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Random 32-bit seed for a fresh drill layout.
export function generateSeed(): number {
  const cryptoObj = typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
  if (cryptoObj?.getRandomValues) {
    const buf = new Uint32Array(1);
    cryptoObj.getRandomValues(buf);
    return buf[0];
  }
  return Math.floor(Math.random() * 4294967296);
}

export function shuffle<T>(items: T[], rng: () => number = Math.random): T[] {
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function distinctNumbers(count: number, rng: () => number, min = 1, max = 99): number[] {
  const pool = new Set<number>();
  let guard = 0;
  while (pool.size < count && guard < 2000) {
    pool.add(min + Math.floor(rng() * (max - min + 1)));
    guard += 1;
  }
  return [...pool];
}

// The deterministic grid layout for n zones. Kept separate from generation so
// a zone set decoded from a QR (which only carries attributes, not positions)
// reconstructs the exact same geometry.
//
// Rows fill top→bottom; a partial last row is RIGHT-aligned so any empty cells
// fall on the bottom-left — i.e. the bottom-left space is the last to be filled.
export function scenarioZoneLayout(n: number): { cx: number; cy: number; radius: number }[] {
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  const cellW = 1 / cols;
  const cellH = 1 / rows;
  const radius = Math.min(cellW, cellH) * 0.36;
  // Shift the final row right by the number of cells it's missing, so gaps sit
  // on the bottom-left rather than the bottom-right.
  const lastRowCount = n - (rows - 1) * cols;
  const lastRowOffset = cols - lastRowCount;
  return Array.from({ length: n }, (_, i) => {
    const row = Math.floor(i / cols);
    const col = (i % cols) + (row === rows - 1 ? lastRowOffset : 0);
    return {
      cx: (col + 0.5) * cellW,
      cy: (row + 0.5) * cellH,
      radius,
    };
  });
}

// Build a fresh set of zones laid out on a grid. Each *enabled* attribute gets
// unique values so it identifies a zone unambiguously; disabled attributes
// collapse to a neutral value (circle / black / 0 / solid) so e.g. a
// black-and-white printout can rely on patterns + numbers alone.
//
// CONTRACT: when `seed` is given the output is deterministic and is part of
// the printed-QR format — a code in the wild carries only {count, attrs, seed}
// and regenerates the zones on scan. Changing this function's draw order, the
// palette arrays, the layout, or mulberry32 breaks every printed drill target;
// if that's ever needed, bump TARGET_PAYLOAD_VERSION and branch on it.
export function generateScenarioZones(
  count: number,
  attributes: ScenarioAttribute[] = DEFAULT_SCENARIO_ATTRIBUTES,
  seed?: number,
  version: number = SCENARIO_PALETTE_VERSION,
): ScenarioZone[] {
  const palette = paletteForVersion(version);
  const rng = seed === undefined ? Math.random : mulberry32(seed);
  const attrs = attributes.length > 0 ? attributes : ["number" as ScenarioAttribute];
  const n = Math.max(2, Math.min(maxZonesForPalette(palette), Math.floor(count)));
  const shapes = attrs.includes("shape")
    ? shuffle(palette.shapes, rng).slice(0, n)
    : Array<ZoneShape>(n).fill("circle");
  const colors = attrs.includes("color")
    ? shuffle(palette.colorNames, rng).slice(0, n)
    : Array<ZoneColorName>(n).fill("black");
  const numbers = attrs.includes("number") ? distinctNumbers(n, rng) : Array<number>(n).fill(0);
  const patterns = attrs.includes("pattern")
    ? shuffle(palette.patterns, rng).slice(0, n)
    : Array<ZonePattern>(n).fill("solid");

  return scenarioZoneLayout(n).map((cell, i) => ({
    id: `z${i + 1}`,
    shape: shapes[i],
    color: colors[i],
    number: numbers[i],
    pattern: patterns[i],
    ...cell,
  }));
}

// ---- Recipe codec (the preferred, tiny QR form) ----
//
// A drill is fully described by {count, attributes, seed}: ~4-10 characters
// (count digit + attribute bitmask hex digit + seed in base36) instead of
// 5 chars per zone. Example: "6b1x7k2p" = 6 zones, shapes+colors+numbers,
// seed 0x... — scanning regenerates the exact zones via the seeded PRNG.

export type ScenarioRecipe = {
  count: number;
  attributes: ScenarioAttribute[];
  seed: number;
};

const ATTR_BITS: Record<ScenarioAttribute, number> = { shape: 1, color: 2, number: 4, pattern: 8 };

export function zonesFromRecipe(
  recipe: ScenarioRecipe,
  version: number = SCENARIO_PALETTE_VERSION,
): ScenarioZone[] {
  return generateScenarioZones(recipe.count, recipe.attributes, recipe.seed, version);
}

export function encodeRecipe(recipe: ScenarioRecipe): string {
  const count = Math.max(2, Math.min(MAX_SCENARIO_ZONES, Math.floor(recipe.count)));
  const mask = recipe.attributes.reduce((m, attr) => m | ATTR_BITS[attr], 0) || ATTR_BITS.number;
  return `${count}${mask.toString(16)}${(recipe.seed >>> 0).toString(36)}`;
}

export function decodeRecipe(raw: string | null | undefined): ScenarioRecipe | null {
  // Count digit (2–9), non-zero attribute mask, then a ≤7-char base36 seed.
  // Legacy per-zone strings are always ≥10 chars (5 per zone), so the forms are
  // disjoint by length.
  if (!raw || !/^[2-9][1-9a-f][0-9a-z]{1,7}$/.test(raw)) return null;
  const seed = parseInt(raw.slice(2), 36);
  if (!Number.isFinite(seed) || seed > 0xffffffff) return null;
  const mask = parseInt(raw[1], 16);
  const attributes = (Object.keys(ATTR_BITS) as ScenarioAttribute[]).filter((attr) => mask & ATTR_BITS[attr]);
  return { count: Number(raw[0]), attributes, seed };
}

// Decode a QR z param of either form (recipe or legacy per-zone) into zones,
// using the palette for the QR's `version` so old prints stay reproducible.
export function zonesFromParam(
  raw: string | null | undefined,
  version: number = SCENARIO_PALETTE_VERSION,
): ScenarioZone[] | null {
  const recipe = decodeRecipe(raw);
  if (recipe) return zonesFromRecipe(recipe, version);
  return decodeZones(raw, version);
}

// ---- Compact zone codec (for QR URLs) ----
//
// Five characters per zone: shape index, color index ("x" = the neutral black
// fill), pattern index, then the number in 2-digit base36 ("00" = numbers
// off). Positions aren't encoded — they're recomputed from the zone count via
// scenarioZoneLayout. 8 zones = 40 chars, small enough to keep the printed QR
// easily scannable.

export function encodeZones(zones: ScenarioZone[], version: number = SCENARIO_PALETTE_VERSION): string {
  const palette = paletteForVersion(version);
  return zones
    .map((zone) => {
      const shape = Math.max(0, palette.shapes.indexOf(zone.shape));
      const colorIdx =
        zone.color === "black"
          ? -1
          : palette.colorNames.indexOf(zone.color as Exclude<ZoneColorName, "black">);
      const color = colorIdx < 0 ? "x" : String(colorIdx);
      const pattern = Math.max(0, palette.patterns.indexOf(zonePattern(zone)));
      const number = Math.max(0, Math.min(zone.number, 1295)) // 1295 = "zz"
        .toString(36)
        .padStart(2, "0");
      return `${shape}${color}${pattern}${number}`;
    })
    .join("");
}

export function decodeZones(
  raw: string | null | undefined,
  version: number = SCENARIO_PALETTE_VERSION,
): ScenarioZone[] | null {
  if (!raw || raw.length % 5 !== 0) return null;
  const palette = paletteForVersion(version);
  const n = raw.length / 5;
  if (n < 2 || n > maxZonesForPalette(palette)) return null;
  const layout = scenarioZoneLayout(n);
  const zones: ScenarioZone[] = [];
  for (let i = 0; i < n; i += 1) {
    const chunk = raw.slice(i * 5, i * 5 + 5);
    const shape = palette.shapes[Number(chunk[0])];
    const color = chunk[1] === "x" ? ("black" as ZoneColorName) : palette.colorNames[Number(chunk[1])];
    const pattern = palette.patterns[Number(chunk[2])];
    const number = parseInt(chunk.slice(3), 36);
    if (!shape || !color || !pattern || Number.isNaN(number)) return null;
    zones.push({ id: `z${i + 1}`, shape, color, number, pattern, ...layout[i] });
  }
  return zones;
}

// One ad-hoc callout for a single zone: a randomly chosen attribute (among the
// enabled ones that actually apply to this zone), phrased exactly like a drill
// step would be. Used when a zone is pressed directly — the app announces a
// random property of it on the spot.
export function randomZoneStep(
  zone: ScenarioZone,
  attributes: ScenarioAttribute[],
  rng: () => number = Math.random,
): DrillStep {
  const applicable = (attributes.length > 0 ? attributes : DEFAULT_SCENARIO_ATTRIBUTES).filter((attr) => {
    if (attr === "color") return zone.color !== "black";
    if (attr === "number") return zone.number > 0;
    return true; // shape always applies; pattern includes "Solid" as a real value
  });
  const kinds = applicable.length > 0 ? applicable : (["shape"] as ScenarioAttribute[]);
  const kind = kinds[Math.floor(rng() * kinds.length)];
  return makeStep(zone, kind);
}

function makeStep(zone: ScenarioZone, kind: PromptKind): DrillStep {
  if (kind === "color") {
    const meta = colorMeta(zone.color);
    return { zoneId: zone.id, kind, label: meta.spoken, spoken: meta.spoken };
  }
  if (kind === "number") {
    const value = String(zone.number);
    return { zoneId: zone.id, kind, label: value, spoken: value };
  }
  if (kind === "pattern") {
    const label = patternSpoken(zonePattern(zone));
    return { zoneId: zone.id, kind, label, spoken: label };
  }
  return { zoneId: zone.id, kind, label: shapeSpoken(zone.shape), spoken: shapeSpoken(zone.shape) };
}

// The ordered callout, e.g. ["Red", "12", "Striped"]. Only attributes whose
// values uniquely identify a zone are ever called. Without repeats it's a
// permutation capped at the zone count; with repeats it can run any length.
export function generateCallSequence(
  zones: ScenarioZone[],
  length: number,
  allowRepeats = false,
): DrillStep[] {
  if (zones.length === 0) return [];
  const kinds = attributesFromZones(zones);
  const pickKind = () => kinds[Math.floor(Math.random() * kinds.length)];

  let order: ScenarioZone[];
  if (allowRepeats) {
    order = Array.from({ length: Math.max(1, length) }, () => zones[Math.floor(Math.random() * zones.length)]);
  } else {
    order = shuffle(zones).slice(0, Math.max(1, Math.min(length, zones.length)));
  }
  return order.map((zone) => makeStep(zone, pickKind()));
}

// ---- Drill engine (pure) ----

export type HitResult = {
  expectedZoneId: string;
  hitZoneId: string | null; // null = a hit that landed in no zone
  correct: boolean;
  reactionMs: number;
};

export type DrillStatus = "idle" | "running" | "done";

export type DrillState = {
  steps: DrillStep[];
  index: number;
  results: HitResult[];
  status: DrillStatus;
};

export function startDrill(steps: DrillStep[]): DrillState {
  return { steps, index: 0, results: [], status: steps.length > 0 ? "running" : "idle" };
}

export function currentStep(state: DrillState): DrillStep | null {
  return state.status === "running" ? state.steps[state.index] ?? null : null;
}

// Record a hit (a zone id, or null for a miss) against the current expected
// step and advance. Returns a new state.
export function registerHit(state: DrillState, hitZoneId: string | null, reactionMs: number): DrillState {
  if (state.status !== "running") return state;
  const step = state.steps[state.index];
  if (!step) return state;

  const result: HitResult = {
    expectedZoneId: step.zoneId,
    hitZoneId,
    correct: hitZoneId === step.zoneId,
    reactionMs,
  };
  const index = state.index + 1;
  return {
    ...state,
    index,
    results: [...state.results, result],
    status: index >= state.steps.length ? "done" : "running",
  };
}

export type DrillScore = {
  correct: number;
  total: number;
  accuracyPct: number;
  totalReactionMs: number;
  avgReactionMs: number;
};

export function scoreDrill(state: DrillState): DrillScore {
  const total = state.results.length;
  const correct = state.results.filter((result) => result.correct).length;
  const totalReactionMs = state.results.reduce((sum, result) => sum + result.reactionMs, 0);
  return {
    correct,
    total,
    accuracyPct: total > 0 ? (correct / total) * 100 : 0,
    totalReactionMs,
    avgReactionMs: total > 0 ? totalReactionMs / total : 0,
  };
}

// Map a hit position (normalized 0..1 within the target) to a zone id, or null
// if it landed outside every zone. This is the bridge to live shot detection:
// feed a detected impact's normalized coordinates here.
export function resolveZoneAtPoint(zones: ScenarioZone[], x: number, y: number): string | null {
  let bestId: string | null = null;
  let bestDist = Infinity;
  for (const zone of zones) {
    const dist = Math.hypot(x - zone.cx, y - zone.cy);
    if (dist <= zone.radius && dist < bestDist) {
      bestId = zone.id;
      bestDist = dist;
    }
  }
  return bestId;
}
