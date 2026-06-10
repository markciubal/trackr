// Local registry of targets you've made. Each target is keyed by a random id
// (the same id the QR encodes and the OpenCV detector reads back). This is the
// browser-side source of truth so you can make + reuse targets with zero setup;
// the Supabase catalog (/api/targets) is the optional shared layer on top.

import { generateTargetId, type LinearUnit } from "./payload";
import type { ScoringId } from "./specs";
import type { ScenarioZone } from "./scenario";

export type DistanceUnit = "yd" | "m" | "ft";

export type TargetInfo = {
  id: string; // random short code, assigned on creation
  name: string;
  unit: LinearUnit;
  widthValue: number;
  heightValue: number;
  qrSizeValue: number;
  scoringId: ScoringId;
  distanceValue?: number; // optional shooting distance, for trig/spread math later
  distanceUnit?: DistanceUnit;
  zones?: ScenarioZone[]; // present on scenario ("smart drill") targets
  drillSeed?: number; // seed the zones were generated from (lets the QR carry a tiny recipe)
  createdAt: number; // epoch ms
};

export type NewTargetInput = Omit<TargetInfo, "id" | "createdAt"> & {
  id?: string;
  createdAt?: number;
};

const STORAGE_KEY = "trackr-targets-v1";

// Mint a target: assign a random id to the supplied information. This is the
// first step of making a target — everything downstream (QR, detection,
// calibration) is keyed off this id.
export function createTargetInfo(input: NewTargetInput): TargetInfo {
  return {
    id: input.id && input.id.length > 0 ? input.id : generateTargetId(),
    name: input.name,
    unit: input.unit,
    widthValue: input.widthValue,
    heightValue: input.heightValue,
    qrSizeValue: input.qrSizeValue,
    scoringId: input.scoringId,
    distanceValue: input.distanceValue,
    distanceUnit: input.distanceUnit,
    zones: input.zones,
    drillSeed: input.drillSeed,
    createdAt: typeof input.createdAt === "number" ? input.createdAt : Date.now(),
  };
}

function readAll(): TargetInfo[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isTargetInfo);
  } catch {
    return [];
  }
}

function writeAll(targets: TargetInfo[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(targets));
  } catch {
    // Ignore quota / privacy-mode errors.
  }
}

// Newest first.
export function listTargets(): TargetInfo[] {
  return readAll().sort((a, b) => b.createdAt - a.createdAt);
}

export function getTarget(id: string): TargetInfo | null {
  return readAll().find((target) => target.id === id) ?? null;
}

// Upsert by id (re-saving an existing id updates it in place).
export function saveTarget(target: TargetInfo): void {
  const all = readAll().filter((existing) => existing.id !== target.id);
  all.push(target);
  writeAll(all);
}

export function removeTarget(id: string): void {
  writeAll(readAll().filter((target) => target.id !== id));
}

function isTargetInfo(value: unknown): value is TargetInfo {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.name === "string" &&
    typeof v.widthValue === "number" &&
    typeof v.heightValue === "number" &&
    typeof v.qrSizeValue === "number" &&
    typeof v.createdAt === "number"
  );
}
