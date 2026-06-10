import type { LinearUnit } from "./payload";

export type ScoringId =
  | "none"
  | "bullseye-5"
  | "grid-1in"
  | "silhouette-b27"
  | "uspsa"
  | "usmc-able"
  | "usmc-a"
  | "usmc-d"
  | "usmc-b-mod"
  | "dots-2in"
  | "drill"; // scenario zones (shapes/colors/numbers/patterns), carried in the QR

// Single source of truth for the scoring/artwork dropdown in the designer.
export const SCORING_OPTIONS: { id: ScoringId; label: string }[] = [
  { id: "none", label: "Plain + crosshair" },
  { id: "bullseye-5", label: "Bullseye (rings)" },
  { id: "grid-1in", label: "1-inch grid" },
  { id: "silhouette-b27", label: "Person silhouette (B-27 style)" },
  { id: "uspsa", label: "USPSA/IPSC silhouette" },
  { id: "usmc-able", label: "USMC head & shoulders" },
  { id: "usmc-a", label: "USMC \"A\" target (36/24/12 rings)" },
  { id: "usmc-d", label: "USMC \"D\" target (threat silhouette)" },
  { id: "usmc-b-mod", label: "USMC \"B\" modified (60\" ring + bottle)" },
  { id: "dots-2in", label: "2-inch dot drill" },
  { id: "drill", label: "Drill (zones: shapes · colors · patterns)" },
];

export type TargetSpec = {
  id: string; // preset id (distinct from the per-print QR id)
  name: string;
  widthValue: number;
  heightValue: number;
  unit: LinearUnit;
  qrSizeValue: number; // default printed QR side length, in `unit`
  scoringId: ScoringId;
  description?: string;
};

// Starter catalog. Paper sizes are exact; the bullseye/grid are generic
// (expand or replace with official specs as you build the product catalog).
export const TARGET_PRESETS: TargetSpec[] = [
  {
    id: "us-letter",
    name: "US Letter — 8.5 × 11 in",
    widthValue: 8.5,
    heightValue: 11,
    unit: "in",
    qrSizeValue: 1.5,
    scoringId: "none",
    description: "Fits standard printer paper.",
  },
  {
    id: "a4",
    name: "A4 — 210 × 297 mm",
    widthValue: 210,
    heightValue: 297,
    unit: "mm",
    qrSizeValue: 40,
    scoringId: "none",
    description: "Standard A4 paper.",
  },
  {
    id: "bullseye-10",
    name: "Bullseye — 10.5 in",
    widthValue: 10.5,
    heightValue: 10.5,
    unit: "in",
    qrSizeValue: 1.5,
    scoringId: "bullseye-5",
    description: "Round bullseye with 5 scoring rings.",
  },
  {
    id: "sight-in-grid",
    name: "Sight-in grid — 12 × 18 in",
    widthValue: 12,
    heightValue: 18,
    unit: "in",
    qrSizeValue: 2,
    scoringId: "grid-1in",
    description: "1-inch grid for zeroing.",
  },
  {
    id: "silhouette-b27",
    name: "Person silhouette — 12 × 18 in",
    widthValue: 12,
    heightValue: 18,
    unit: "in",
    qrSizeValue: 2,
    scoringId: "silhouette-b27",
    description: "B-27-style half silhouette with center-mass scoring rings.",
  },
  {
    id: "uspsa-practice",
    name: "USPSA practice — 18 × 30 in",
    widthValue: 18,
    heightValue: 30,
    unit: "in",
    qrSizeValue: 2.5,
    scoringId: "uspsa",
    description: "USPSA/IPSC-style silhouette with A/C/D zone lines.",
  },
  {
    id: "usmc-able",
    name: "USMC head & shoulders — 12 × 18 in",
    widthValue: 12,
    heightValue: 18,
    unit: "in",
    qrSizeValue: 2,
    scoringId: "usmc-able",
    description: "Marine-Corps-style 'Able' head-and-shoulders silhouette.",
  },
  {
    id: "usmc-a",
    name: 'USMC "A" target — 4 × 6 ft',
    widthValue: 48,
    heightValue: 72,
    unit: "in",
    qrSizeValue: 4,
    scoringId: "usmc-a",
    description: "Table 1 course of fire. Concentric 36/24/12-inch rings, 12-inch filled center.",
  },
  {
    id: "usmc-d",
    name: 'USMC "D" target — 6 ft wide',
    widthValue: 72,
    heightValue: 54,
    unit: "in",
    qrSizeValue: 4,
    scoringId: "usmc-d",
    description: "Table 1 threat silhouette: 34-inch shoulders, 26-inch waist (8/19/14/6-inch bands).",
  },
  {
    id: "usmc-b-mod",
    name: 'USMC "B" modified — 6 ft, 60-inch ring',
    widthValue: 72,
    heightValue: 72,
    unit: "in",
    qrSizeValue: 4,
    scoringId: "usmc-b-mod",
    description: "Table 1: 40-inch bottle threat (20-inch neck) inside a 60-inch scoring ring.",
  },
  {
    id: "dot-drill-letter",
    name: "Dot drill — 8.5 × 11 in",
    widthValue: 8.5,
    heightValue: 11,
    unit: "in",
    qrSizeValue: 1.5,
    scoringId: "dots-2in",
    description: "Numbered 2-inch dots for dot-torture-style drills.",
  },
  {
    id: "drill-18",
    name: "Drill target — 18 × 18 in",
    widthValue: 18,
    heightValue: 18,
    unit: "in",
    qrSizeValue: 2,
    scoringId: "drill",
    description: "Scenario zones (shapes/colors/numbers/patterns); the QR carries the layout.",
  },
];

export function findPreset(id: string): TargetSpec | undefined {
  return TARGET_PRESETS.find((preset) => preset.id === id);
}
