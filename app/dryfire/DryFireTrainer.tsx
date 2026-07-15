"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ScenarioBoard, type ZoneFeedback } from "@/app/components/scenario/ScenarioBoard";
import {
  DEFAULT_SCENARIO_ATTRIBUTES,
  SCENARIO_SHEET_ASPECT,
  type CalloutMode,
  type DrillState,
  type DrillStep,
  type ScenarioZone,
  currentStep,
  generateCallSequence,
  generateScenarioZones,
  generateSeed,
  generateTimedSchedule,
  registerHit,
  scoreDrill,
  startDrill,
} from "@/app/lib/targets/scenario";
import { cancelSpeech, isSpeaking, speak, speakSequence } from "@/app/lib/targets/speech";
import { useScreenWakeLock } from "@/app/lib/useScreenWakeLock";
import {
  analyzeColorFlag,
  analyzeFlag,
  analyzeShapeFlag,
  classifyRGB,
  type ColorGateTuning,
  type ColorPalette,
  DEFAULT_SHAPE_REQUIREMENTS,
  isShapeMark,
  type RedSensitivity,
  type ShapeRequirements,
  type ColorSensitivity,
  tuningFromPreset,
  type FlagDebug,
  type FlagFailStage,
  type FlagObservation,
  type ForegroundGrid,
  type FlagPatternMode,
  type TrackedColors,
  analyzeColorFlagLoose,
  buildChromaLocus,
  cymBeacon,
  locusDist2,
  prescanCMY,
  trackColorFlag,
  trackShapeFlag,
} from "@/app/lib/dryfire/flagTracker";
import { fitAimModel, meanFeatures, predictAim, type AimModel, type AimSample } from "@/app/lib/dryfire/aimModel";
import {
  createClickTrigger,
  fingerprintSimilarity,
  meanFingerprint,
  type ClickTrigger,
  type TriggerSensitivity,
} from "@/app/lib/dryfire/audioTrigger";
import { computeTargetStats, type TargetShot } from "@/app/lib/dryfire/targetStats";
import { appendShot } from "@/app/lib/dryfire/shotLog";
import { createBackgroundModel } from "@/app/lib/dryfire/backgroundModel";
import { createFeatureFilter } from "@/app/lib/dryfire/featureFilter";

// 3×3 calibration dots in board-normalized coordinates.
const CAL_POINTS: readonly (readonly [number, number])[] = [
  [0.15, 0.12],
  [0.5, 0.12],
  [0.85, 0.12],
  [0.15, 0.5],
  [0.5, 0.5],
  [0.85, 0.5],
  [0.15, 0.88],
  [0.5, 0.88],
  [0.85, 0.88],
];

const TRACE_WINDOW_MS = 2500;

// Dual-handle slider for one color channel's pickup gate. The two handles
// sit on a 0..240 channel-value track: the LOWER handle is the brightness
// floor the channel must clear, the UPPER handle is floor + dominance
// margin — how far the channel must beat the strongest rival. The band
// between them shows the demanded dominance at a glance.
const TUNE_MAX = 240;
function DualSlider({
  label,
  accent,
  value,
  onChange,
  boundaryColor,
}: {
  label: string;
  accent: string;
  value: [number, number]; // [floor, margin]
  onChange: (next: [number, number]) => void;
  // The WEAKEST color still accepted at dominant-channel value v with the
  // current margin — drives the live gradient on the track.
  boundaryColor: (v: number, margin: number) => [number, number, number];
}) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const activeRef = useRef<0 | 1 | null>(null);
  const lo = Math.min(value[0], TUNE_MAX);
  const hi = Math.min(value[0] + value[1], TUNE_MAX);
  const margin = hi - lo;
  // Track gradient: neutral below the floor (rejected), then the actual
  // borderline-accepted colors from the floor to full brightness. Updates
  // live as either handle moves.
  const loPct = (lo / TUNE_MAX) * 100;
  const stops: string[] = [`#1f1f23 0%`, `#1f1f23 ${loPct.toFixed(1)}%`];
  const steps = 10;
  for (let i = 0; i <= steps; i += 1) {
    const v = lo + ((TUNE_MAX - lo) * i) / steps;
    const [r, g, b] = boundaryColor(v, margin);
    const pct = loPct + ((100 - loPct) * i) / steps;
    stops.push(`rgb(${r},${g},${b}) ${pct.toFixed(1)}%`);
  }
  const trackGradient = `linear-gradient(90deg, ${stops.join(", ")})`;
  const weakest = boundaryColor(lo, margin);
  const toValue = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    return Math.round(Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) * TUNE_MAX);
  };
  const apply = (thumb: 0 | 1, v: number) => {
    if (thumb === 0) {
      const nlo = Math.min(v, hi);
      onChange([nlo, hi - nlo]);
    } else {
      const nhi = Math.max(v, lo);
      onChange([lo, nhi - lo]);
    }
  };
  return (
    <div className="flex items-center gap-2">
      <span className="w-3 text-[11px] font-semibold" style={{ color: accent }}>
        {label}
      </span>
      <div
        ref={trackRef}
        className="relative h-5 w-44 cursor-pointer touch-none"
        onPointerDown={(e) => {
          const v = toValue(e.clientX);
          const thumb: 0 | 1 = Math.abs(v - lo) <= Math.abs(v - hi) ? 0 : 1;
          activeRef.current = thumb;
          e.currentTarget.setPointerCapture(e.pointerId);
          apply(thumb, v);
        }}
        onPointerMove={(e) => {
          if (activeRef.current === null) return;
          apply(activeRef.current, toValue(e.clientX));
        }}
        onPointerUp={() => {
          activeRef.current = null;
        }}
        onPointerCancel={() => {
          activeRef.current = null;
        }}
      >
        {/* The track IS the acceptance visualization: gray = rejected,
            colored = the actual borderline colors that pass the gate. */}
        <div
          className="absolute left-0 right-0 top-1/2 h-2 -translate-y-1/2 rounded"
          style={{ background: trackGradient }}
        />
        {([lo, hi] as const).map((v, i) => (
          <div
            key={i}
            className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-neutral-950 shadow"
            style={{ left: `${(v / TUNE_MAX) * 100}%`, background: accent }}
          />
        ))}
      </div>
      {/* Live swatch: the dimmest, least-dominant color that still counts. */}
      <span
        className="h-4 w-4 shrink-0 rounded border border-neutral-700"
        title="Weakest color the gate still accepts"
        style={{ background: `rgb(${weakest[0]},${weakest[1]},${weakest[2]})` }}
      />
      <span className="w-24 text-[10px] tabular-nums text-gray-500">
        floor {lo} · dom +{margin}
      </span>
    </div>
  );
}

type Phase = "setup" | "calibrate" | "train";

// ---- Scenario profiles ------------------------------------------------------
// Everything scene-specific — the CMY reference colors captured in THIS room
// under THIS lighting, the tuning, the trigger calibration, the drill setup —
// saved to localStorage per named scenario. Returning to the same shooting
// scene, the last profile auto-loads so the saved color ranges apply
// immediately (assumes the scene hasn't changed; recapture if it has).
type SavedRefColors = {
  red: [number, number, number] | null;
  green: [number, number, number] | null;
  blue: [number, number, number] | null;
};
type ScenarioProfile = {
  name: string;
  savedAt: number;
  patternMode: FlagPatternMode;
  trainMode: "drill" | "target";
  colorSensitivity: ColorSensitivity | "custom";
  colorTuning: ColorGateTuning;
  refColors: SavedRefColors;
  refBrightTol: number;
  redSensitivity: RedSensitivity;
  shapeReq: ShapeRequirements;
  trigSensitivity: TriggerSensitivity;
  trigCal: {
    clickPeak: number;
    floor: number;
    rackOk: boolean | null;
    clickFp?: number[] | null;
    rackFp?: number[] | null;
  } | null;
  drillMode: CalloutMode;
  drillLen: number;
  timespanSec: number;
  protocol: RangeProtocol;
};
const PROFILES_KEY = "trackr.dryfire.profiles";
const LAST_PROFILE_KEY = "trackr.dryfire.lastProfile";
function loadProfilesFromStorage(): ScenarioProfile[] {
  try {
    const raw = localStorage.getItem(PROFILES_KEY);
    return raw ? (JSON.parse(raw) as ScenarioProfile[]) : [];
  } catch {
    return [];
  }
}
function persistProfiles(list: ScenarioProfile[]) {
  try {
    localStorage.setItem(PROFILES_KEY, JSON.stringify(list));
  } catch {
    // Storage full/unavailable — profiles just won't persist.
  }
}

// User-mode guided setup: the shippable walkthrough. Pro mode is the full
// diagnostic surface underneath.
type WizardStep = "choose" | "safety" | "prep";
const WIZARD_STEPS: WizardStep[] = ["choose", "safety", "prep"];
const WIZARD_TITLES: Record<WizardStep, string> = {
  choose: "Choose & train",
  safety: "Safety first",
  prep: "Point & hold",
};

// Start routines, spoken before the timer beep. "standard" is the house
// default — a dry-fire-appropriate make-ready; the others match the
// competition rule sets so the cadence matches match day.
type RangeProtocol = "standard" | "uspsa" | "ipsc" | "idpa" | "none";
const RANGE_COMMANDS: Record<Exclude<RangeProtocol, "none">, string[]> = {
  standard: ["Check chamber.", "Survey target.", "Engage."],
  uspsa: ["Make ready.", "Are you ready?", "Standby."],
  ipsc: ["Load and make ready.", "Are you ready?", "Standby."],
  idpa: ["Range is hot. Eyes and ears.", "Load and make ready.", "Are you ready?", "Standby."],
};

// Shot-timer start beep: ~2.1 kHz square-ish tone, 300 ms — the sound every
// competition shooter's lizard brain is trained on.
function playStartBeep(onDone?: () => void) {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 2100;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.5, ctx.currentTime + 0.008);
    gain.gain.setValueAtTime(0.5, ctx.currentTime + 0.28);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.32);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.33);
    osc.onended = () => {
      void ctx.close().catch(() => undefined);
      onDone?.();
    };
  } catch {
    onDone?.();
  }
}

// Why-it-failed hints for the diagnostics panel, per detector stage.
const DIAG_HINTS: Record<FlagFailStage, string> = {
  ok: "Flag locked — all stages passing.",
  "too-few-blobs":
    "Almost no dark shapes seen. Is the flag facing the camera? Even out the lighting (webcams overexpose against windows) and avoid glare on the card.",
  "too-few-tiles":
    "No tight group of 8 similar-size dark tiles anywhere in view (at any scale). Usual causes: tiles merging under glare/blur (check the red tint — each tile should be a separate red square), or the pattern too small. Move to 1.5–2 m, steady up, kill glare.",
  "no-collinear-quad":
    "8 tile-sized shapes found but no straight diagonal among them — probably background clutter being picked up instead of the card. Clear dark objects behind the gun.",
  "bad-side-split":
    "Diagonal found, but the remaining tiles don't split 2-and-2 around it — a stray dark shape is stealing a tile slot. Clean up the background.",
  "no-dot-candidates":
    "Tiles look right but the small orientation dot isn't seen — washed out by glare, or below resolution. Move closer.",
  "affine-fail": "Degenerate tile geometry — extreme viewing angle. Face the flag more toward the camera.",
  "tiny-cell": "Pattern resolved but under ~3 px per cell — too far for this camera. Move closer.",
  "dot-mismatch":
    "A checker matched but the dot isn't where it should be — either a false match on background clutter or the dot is obscured.",
  "high-residual": "Tiles found but they don't sit on a flat grid — motion blur, or a curled/bent card face.",
  "missing-color":
    "At least one of red/green/blue wasn't seen at all. Check the tint overlay — each quadrant should light up in its color. Usual causes: card facing away, low light desaturating the colors, or glossy filament glare.",
  "bad-color-layout":
    "Red, green, and blue blobs exist but no combination forms the card's quadrant layout with the blue dot confirming — usually colored objects elsewhere in the room, or the dot washed out. Check the tint overlay for stray colored patches.",
  "missing-shape":
    "Didn't see all three of disk / ring / two-hole among the RED mark blobs (only strongly red pixels count — the tint overlay should light up the card's patches and almost nothing else). Rings need their holes to resolve (~4 px) — move closer, kill glare; if the patches don't tint at all, the red is washed out by overexposure.",
  "bad-shape-layout":
    "Disk, ring, and two-hole blobs exist but no combination forms the card's layout with the dot confirming — another red object may be supplying a fake patch, or a hole has blurred shut. Check the tint overlay.",
};

type DiagInfo = {
  stage: FlagFailStage | "none";
  threshold: number;
  blobCount: number;
  clusterCount: number;
  tileCount: number;
  quadScore: number | null;
  dotCount: number;
  cellPx: number | null;
  residual: number | null;
  procW: number;
  procH: number;
  roiActive: boolean;
  colorCounts: { red: number; green: number; blue: number } | null;
  colorGates: FlagDebug["colorGates"];
};

type TrackerStatus = {
  locked: boolean;
  residual: number;
  cellPx: number;
  fps: number;
  tracking: boolean; // color mode: the learned-appearance tracker is engaged
  misses: number; // consecutive track misses (re-seeking when > 0)
  regional: boolean; // lock has matured: gates relaxed inside the trusted region
};

// Track mode state: the locked card's learned appearance + motion.
type TrackState = {
  colors: TrackedColors;
  center: { x: number; y: number };
  velocity: { x: number; y: number }; // px per ms
  cellPx: number;
  affine: [number, number, number, number];
  atMs: number;
  misses: number;
  seedAtMs: number; // when acquisition handed the card to the tracker
  hits: number; // successful track frames since seed (lock maturity)
  seedCellPx: number; // scale anchor — the track can't drift far from what acquisition proved
};

// A lock this old with this many confirmations graduates to REGIONAL mode:
// the ROI is trusted, so the color/geometry gates open up inside it.
const REGIONAL_AFTER_MS = 450;
const REGIONAL_AFTER_HITS = 4;
// Color-photo capture zone: the card must sit inside this ring (center of
// frame, radius × min processing dimension) marked by marching black/white
// ants on the preview. Capture-time detection only looks here.
const PHOTO_ZONE_R = 0.234;
// Simulated range for target practice: the bullseye scales as if it were
// this many meters away, with BASE_RANGE_M filling the screen. Same angular
// math as real distance — a 25 m target subtends 3/25 of the 3 m one.
const RANGE_PRESETS = [3, 5, 7, 10, 15, 25];
const BASE_RANGE_M = 3;

const lerp3 = (a: [number, number, number], b: [number, number, number], t: number): [number, number, number] => [
  a[0] + t * (b[0] - a[0]),
  a[1] + t * (b[1] - a[1]),
  a[2] + t * (b[2] - a[2]),
];
const blendColors = (prev: TrackedColors, next: TrackedColors, t: number): TrackedColors => ({
  red: lerp3(prev.red, next.red, t),
  green: lerp3(prev.green, next.green, t),
  blue: lerp3(prev.blue, next.blue, t),
});

type ShotRecord = {
  n: number;
  zoneId: string | null;
  expectedZoneId: string | null;
  correct: boolean | null; // null = free practice (no active call)
  atMs: number;
};

// Running tally for the whole session (across drills and recalibrations).
type SessionStats = {
  shots: number; // every trigger break
  called: number; // shots (or timeouts) answering a call
  hits: number;
  misses: number; // wrong zone, off-board, or timed out
  drills: number; // drills finished
  streak: number; // current consecutive hits
  bestStreak: number;
  reactionSumMs: number;
  reactionCount: number;
};
const EMPTY_SESSION: SessionStats = {
  shots: 0,
  called: 0,
  hits: 0,
  misses: 0,
  drills: 0,
  streak: 0,
  bestStreak: 0,
  reactionSumMs: 0,
  reactionCount: 0,
};

// Shooter-facing coaching: turn the detector's failure stage into ONE plain
// sentence about what to physically change. Shown as a chip on the board
// whenever the flag isn't locked.
function coachHintFor(debug: FlagDebug | null, mode: FlagPatternMode, palette: ColorPalette): string | null {
  if (!debug) return "Point the muzzle flag toward the camera.";
  const c = debug.colorCounts;
  const g = debug.colorGates;
  const slotNames = palette === "cmy" ? (["yellow", "cyan", "magenta"] as const) : (["red", "green", "blue"] as const);
  switch (debug.failStage) {
    case "ok":
      return null;
    case "missing-shape": {
      if (c && c.red === 0 && c.green === 0 && c.blue === 0)
        return "I can't see the card's red at all — add light on the card, avoid backlight, or step closer.";
      if (c && c.green === 0)
        return "I see red patches but the RING's hole isn't resolving — move closer or kill the glare on the card.";
      if (c && c.blue === 0)
        return "I see the card but not the two-hole patch — move a little closer so its holes resolve.";
      if (c && c.red === 0)
        return "Ring and two-hole are visible but no solid disk — glare may be washing one patch out; tilt the card slightly.";
      return "Not all of the card's shapes are visible — face it toward the camera and even out the lighting.";
    }
    case "missing-color": {
      if (c) {
        // List missing colors in card-reading order (CMY: cyan, yellow,
        // magenta; RGB: red, green, blue).
        const counts = [c.red, c.green, c.blue];
        const order = palette === "cmy" ? [1, 0, 2] : [0, 1, 2];
        const missing = order
          .map((i) => (counts[i] === 0 ? slotNames[i] : null))
          .filter(Boolean)
          .join(" and ");
        if (missing) return `The ${missing} quadrant isn't showing — low light desaturates colors; brighten the room or move closer.`;
      }
      return "A color quadrant is missing — better lighting on the card will bring it back.";
    }
    case "bad-shape-layout":
    case "bad-color-layout": {
      if (g && g.chiralityOk === false)
        return "The pattern looks MIRRORED — the card may be facing away from the camera, or a reflection is being picked up.";
      if (g && g.nearestDotErrCells === null)
        return "I see the right pattern but I don't see the dot — try moving to better lighting or a touch closer.";
      if (g && g.nearestDotErrCells !== null && g.nearestDotErrCells > 0.6)
        return "Right pattern, but the small dot isn't where it should be — angle the card more toward the camera.";
      return "The pattern's pieces don't quite line up — hold the gun steady for a second so I can lock on.";
    }
    case "too-few-blobs":
      return "Almost nothing visible — is the flag facing the camera? Watch out for a bright window behind you.";
    case "too-few-tiles":
      return "The checkerboard's tiles are merging — move to 1.5–2 m, steady up, and kill glare.";
    case "tiny-cell":
      return "Too far from the camera — the pattern is under 3 pixels per cell; move closer.";
    case "no-dot-candidates":
    case "dot-mismatch":
      return "I see the right pattern but I don't see the dot — try moving to better lighting.";
    case "high-residual":
      return "Pattern found but unstable — motion blur or a bent card; slow the muzzle and check the card sits flat.";
    default:
      return mode === "checker"
        ? "Searching for the checkerboard — face it to the camera and hold still."
        : "Searching for the card — face it to the camera and hold still.";
  }
}

// Animated mini-previews for the "Choose & train" path cards — a living
// thumbnail of what each screen actually looks like.
function PathPreviewStyles() {
  return (
    <style>{`
      @keyframes tpShot { 0% { opacity: 0; r: 1; } 4% { opacity: 1; r: 4.5; } 7% { r: 2.6; } 82% { opacity: 1; } 92%, 100% { opacity: 0; } }
      @keyframes dpCall { 0%, 2% { opacity: 0; } 6%, 22% { opacity: 1; } 30%, 100% { opacity: 0; } }
      @keyframes dpHit { 0%, 12% { opacity: 0; r: 0.5; } 16% { opacity: 1; r: 4.5; } 19% { r: 2.6; } 28% { opacity: 1; } 34%, 100% { opacity: 0; } }
      @keyframes dpBanner { 0%, 33% { opacity: 1; } 33.5%, 100% { opacity: 0; } }
    `}</style>
  );
}

function MiniTargetPreview() {
  return (
    <svg viewBox="0 0 120 84" className="h-20 w-28 shrink-0 rounded-md border border-gray-700 bg-white" aria-hidden>
      {[36, 27, 18, 9].map((r, i) => (
        <circle
          key={r}
          cx="60"
          cy="42"
          r={r}
          fill={i === 3 ? "#111" : "none"}
          stroke={i >= 2 ? "#555" : "#222"}
          strokeWidth="1"
        />
      ))}
      <path d="M 57.5 42 H 62.5 M 60 39.5 V 44.5" stroke="#fff" strokeWidth="0.8" />
      {[
        [65, 38],
        [55, 46],
        [63, 48],
        [57, 37],
      ].map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r="2.6"
          fill="#0284c7"
          stroke="#fff"
          strokeWidth="0.9"
          style={{ animation: "tpShot 6s linear infinite", animationDelay: `${0.6 + i * 1.3}s`, opacity: 0 }}
        />
      ))}
    </svg>
  );
}

function MiniDrillPreview() {
  // Zone hues mirror the board's real callable palette (scenario.ts registry).
  const zoneColors = ["#3b82f6", "#eab308", "#ec4899", "#eab308", "#ec4899", "#3b82f6", "#ec4899", "#3b82f6", "#eab308"];
  const colorNames: Record<string, string> = { "#3b82f6": "BLUE", "#eab308": "YELLOW", "#ec4899": "PINK" };
  const xs = [28, 60, 92];
  const ys = [30, 51, 72];
  // Which zones get "called" (index into the 3×3) and when. The banner swaps
  // to the called zone's color name, tinted to match, in sync with the ring.
  const calls: [number, number][] = [
    [4, 0],
    [0, 2],
    [8, 4],
  ];
  return (
    <svg viewBox="0 0 120 84" className="h-20 w-28 shrink-0 rounded-md border border-gray-700 bg-white" aria-hidden>
      <rect x="30" y="3" width="60" height="11" rx="3" fill="#111" />
      {calls.map(([zone, delay]) => (
        <text
          key={zone}
          x="60"
          y="11"
          textAnchor="middle"
          fontSize="7"
          fontWeight="700"
          fill="#fff"
          style={{ animation: "dpBanner 6s linear infinite", animationDelay: `${delay}s`, opacity: 0 }}
        >
          SHOOT · <tspan fill={zoneColors[zone]}>{colorNames[zoneColors[zone]]}</tspan>
        </text>
      ))}
      {zoneColors.map((c, i) => (
        <circle key={i} cx={xs[i % 3]} cy={ys[Math.floor(i / 3)]} r="6.5" fill={c} opacity="0.9" />
      ))}
      {calls.map(([zone, delay]) => (
        <g key={zone}>
          <circle
            cx={xs[zone % 3]}
            cy={ys[Math.floor(zone / 3)]}
            r="10"
            fill="none"
            stroke="#f59e0b"
            strokeWidth="2"
            style={{ animation: "dpCall 6s linear infinite", animationDelay: `${delay}s`, opacity: 0 }}
          />
          <circle
            cx={xs[zone % 3]}
            cy={ys[Math.floor(zone / 3)]}
            r="2.6"
            fill="#16a34a"
            stroke="#fff"
            strokeWidth="0.9"
            style={{ animation: "dpHit 6s linear infinite", animationDelay: `${delay}s`, opacity: 0 }}
          />
        </g>
      ))}
    </svg>
  );
}

// What each target statistic means and how it's computed — shown as hover
// tooltips on the labels and in the ⓘ panel.
const STAT_HELP: Record<string, string> = {
  "shots / score":
    "Ring value per shot: 10 for the innermost tenth of the target radius, down to 1 at the outer ring, 0 outside. Score = sum of ring values; avg = score ÷ shots.",
  "MPI offset":
    "Mean Point of Impact = (mean x, mean y) of all shots — the group's center. Offset = its distance from the point of aim (target center), with the direction as a clock bearing. This measures ZERO ERROR (sights/calibration), not wobble.",
  "mean radius":
    "Average distance of each shot from the MPI: MR = Σ √((xᵢ−x̄)² + (yᵢ−ȳ)²) ÷ n. The precision measure ballisticians prefer — every shot contributes, so it stabilizes quickly.",
  "radial σ":
    "Standard deviation of the shot-to-MPI distances: √(Σ (rᵢ − MR)² ÷ n). How consistent your dispersion is around the group center.",
  "σx / σy":
    "Per-axis standard deviation about the MPI. σy ≫ σx = vertical stringing (breathing, inconsistent trigger press); σx ≫ σy = horizontal stringing (trigger jerk, grip). Flagged at a 1.5× ratio.",
  "extreme spread":
    "Largest center-to-center distance between any two shots — the classic “group size” quoted at the range. Note it's driven entirely by your two worst shots.",
  CEP50:
    "Circular Error Probable: the median shot-to-MPI distance — the radius of a circle centered on the group that contains half the shots.",
  "avg split": "Average time between consecutive shots. Gaps over 20 s count as pauses, not splits, and are excluded.",
  "best split": "Fastest time between two consecutive shots this string.",
  "string / HF":
    "String = time from first shot to last. Hit Factor = score ÷ string seconds (points per second) — the USPSA-style speed-vs-accuracy economy.",
  groups:
    "2-means clustering, deterministically seeded with the two farthest-apart shots, iterated to convergence. TWO groups are reported only when the cluster centers sit farther apart than 2× the average within-group radius — a satellite group (often low-left) is the classic flinch signature.",
};

// Bullseye face for open-target mode: ten scoring rings, black aiming
// center. The scoring radius is 45% of the shorter side — the same factor
// the shot→target-frame conversion uses.
function TargetFace({
  aspectRatio,
  className,
  scale = 1,
}: {
  aspectRatio: number;
  className?: string;
  scale?: number; // simulated range: 1 = point blank, smaller = farther
}) {
  const safeAspect = aspectRatio > 0 ? aspectRatio : 1;
  const vh = 100 / safeAspect;
  const cx = 50;
  const cy = vh / 2;
  const R = 0.45 * Math.min(100, vh) * scale;
  return (
    <svg
      viewBox={`0 0 100 ${vh}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ aspectRatio: String(safeAspect) }}
      className={`w-full rounded-lg border border-gray-700 bg-white ${className ?? ""}`}
    >
      {Array.from({ length: 10 }, (_, i) => {
        const r = (R * (10 - i)) / 10;
        const black = r <= R * 0.4; // 8-ring in: the black aiming area
        return (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={r}
            fill={black ? (r === R * 0.4 ? "#111" : "none") : "none"}
            stroke={black ? "#666" : "#222"}
            strokeWidth={0.25}
          />
        );
      })}
      {/* Ring numbers along the horizontal — sized to the ring band (R/10)
          so they always fit inside their ring at any aspect. */}
      {Array.from({ length: 5 }, (_, i) => {
        const ring = 6 + i; // label 6..10 outward-in
        const r = (R * (10 - (ring - 1) - 0.5)) / 10;
        return (
          <text
            key={ring}
            x={cx + r}
            y={cy}
            fontSize={R * 0.05}
            textAnchor="middle"
            dominantBaseline="central"
            fill={ring >= 8 ? "#bbb" : "#333"}
          >
            {ring}
          </text>
        );
      })}
      {/* Center cross. */}
      <path
        d={`M ${cx - R * 0.035} ${cy} H ${cx + R * 0.035} M ${cx} ${cy - R * 0.035} V ${cy + R * 0.035}`}
        stroke="#fff"
        strokeWidth={R * 0.008}
      />
    </svg>
  );
}

export function DryFireTrainer() {
  // "user" = guided walkthrough (default, shippable); "pro" = full
  // diagnostic surface. All machinery is shared — user mode is a layer.
  const [uiMode, setUiMode] = useState<"user" | "pro">("user");
  const uiModeRef = useRef<"user" | "pro">("user");
  useEffect(() => {
    uiModeRef.current = uiMode;
  }, [uiMode]);
  const [wizardStep, setWizardStep] = useState<WizardStep>("choose");
  const [phase, setPhase] = useState<Phase>("setup");
  const phaseRef = useRef<Phase>("setup");
  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [tracker, setTracker] = useState<TrackerStatus>({
    locked: false,
    residual: 0,
    cellPx: 0,
    fps: 0,
    tracking: false,
    misses: 0,
    regional: false,
  });
  const [calIndex, setCalIndex] = useState(0);
  const [calHold, setCalHold] = useState(0); // 0..1 dwell progress
  const [model, setModel] = useState<AimModel | null>(null);
  const [micActive, setMicActive] = useState(false);
  const [micStats, setMicStats] = useState({
    pct: 0,
    peakPct: 0,
    peak: 0,
    ambient: 0,
    threshold: 1,
    suppressed: false,
    rejectedAgoMs: Infinity,
    sim: null as number | null, // last candidate's fingerprint match (0..1)
  });
  // Rolling audio waveform for the trigger-calibration steps: peak samples,
  // gate/ambient traces, and click/reject event markers on a small canvas.
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const micWaveRef = useRef<{ t: number; peak: number; thr: number; ambient: number; suppressed: boolean }[]>([]);
  const trigWaveEventsRef = useRef<{ t: number; kind: "click" | "reject" }[]>([]);
  const lastRejectSeenRef = useRef(0);
  // Shot-trigger sensitivity: "high" arms on the faintest clean click.
  const [trigSensitivity, setTrigSensitivity] = useState<TriggerSensitivity>("high");
  // Trigger-click calibration: collect the peaks of 3 real dry-fire clicks
  // and derive a custom threshold floor from YOUR striker on YOUR mic.
  const TRIG_CAL_CLICKS = 3;
  const trigCalRef = useRef<{
    // SLIDES first: three prompted 2-second listen windows — whatever lands
    // inside a window IS the slide (both rack impacts); the gaps are deaf.
    // Then CLICKS: three dry-fires, screened against the slide template.
    stage: "slides" | "clicks";
    slidesDone: number; // windows that captured at least one slide sound
    windowCloseAt: number | null; // open listen window; null = waiting
    windowStartCount: number; // rackFps length when the window opened
    peaks: number[]; // click peaks (floor derivation)
    clickFps: number[][]; // one fingerprint per dry-fire
    rackFps: number[][]; // slide fingerprints — the reject template
    armAt: number; // clicks: ignore spikes before this (TTS guard)
    remindAt: number; // gentle re-prompt when a step sits idle
    rackOk?: boolean | null;
  } | null>(null);
  const [slideWindowOpen, setSlideWindowOpen] = useState(false); // "RACK NOW" vs "wait"
  const [trigCalCount, setTrigCalCount] = useState<number | null>(null); // clicks so far, null = not calibrating
  const [trigCalRacking, setTrigCalRacking] = useState(false); // rack-verification phase
  const [trigCalResult, setTrigCalResult] = useState<{
    clickPeak: number;
    floor: number;
    rackOk: boolean | null; // true = rack correctly ignored; null = not verified
    clickFp?: number[] | null; // your striker's spectral fingerprint
    rackFp?: number[] | null; // your rack's spectral fingerprint
  } | null>(null);
  const [statusLine, setStatusLine] = useState("Print the flag, chamber the stem, and start the camera.");
  const [feedback, setFeedback] = useState<ZoneFeedback>({});
  const [shots, setShots] = useState<ShotRecord[]>([]);
  const [drill, setDrill] = useState<DrillState | null>(null);
  // Drill configuration — mirrors the drill page: random timing (default),
  // full sequence, or one call at a time; plus call count and window.
  const [drillMode, setDrillMode] = useState<CalloutMode>("timed");
  const drillModeRef = useRef<CalloutMode>("timed");
  useEffect(() => {
    drillModeRef.current = drillMode;
  }, [drillMode]);
  const [drillLen, setDrillLen] = useState(10);
  const [timespanSec, setTimespanSec] = useState(60);
  const [protocol, setProtocol] = useState<RangeProtocol>("standard");
  const [timedRun, setTimedRun] = useState<{ steps: DrillStep[]; calloutAtMs: number[]; endAtMs: number } | null>(
    null,
  );
  const announcedIndexRef = useRef(-1);
  // The call currently in play — mirrored as a big readable banner on the
  // fullscreen board so the shooter can READ the call, not just hear it.
  const [currentCall, setCurrentCall] = useState<{ label: string; kind: string } | null>(null);
  const [session, setSession] = useState<SessionStats>(EMPTY_SESSION);
  const lastCallAtRef = useRef(0); // when the current call finished speaking
  // Training style: called drills against the zone board, or an open
  // bullseye you just shoot at, with full group statistics.
  const [trainMode, setTrainMode] = useState<"drill" | "target">("target");
  const trainModeRef = useRef<"drill" | "target">("target");
  useEffect(() => {
    trainModeRef.current = trainMode;
  }, [trainMode]);
  const [targetShots, setTargetShots] = useState<TargetShot[]>([]);
  const targetStats = useMemo(() => computeTargetStats(targetShots), [targetShots]);
  // Simulated range (target mode): scales the bullseye like real distance.
  const [simRangeM, setSimRangeM] = useState(5);
  const targetScale = Math.min(1, BASE_RANGE_M / simRangeM);
  const stepRange = (dir: number) => {
    const i = RANGE_PRESETS.indexOf(simRangeM);
    const next = RANGE_PRESETS[Math.min(RANGE_PRESETS.length - 1, Math.max(0, (i < 0 ? 1 : i) + dir))];
    if (next !== simRangeM) {
      setSimRangeM(next);
      speak(`Range ${next} meters.`);
    }
  };
  const [showStatHelp, setShowStatHelp] = useState(false);
  const [resizedSinceCal, setResizedSinceCal] = useState(false);
  const [briefActive, setBriefActive] = useState(false); // safety brief speaking
  // Which printed card is on the gun: the RGB quadrant card (recommended,
  // near-unbreakable detection) or the black checkerboard.
  const [patternMode, setPatternMode] = useState<FlagPatternMode>("color");
  const patternModeRef = useRef<FlagPatternMode>("color");
  useEffect(() => {
    patternModeRef.current = patternMode;
  }, [patternMode]);
  // How aggressive the color classifier is. The card sits at 45° in ambient
  // light, so "forgiving" is the practical default. The presets seed the
  // per-channel dual-handle sliders (colorTuning — the actual gates the
  // classifier runs); dragging any slider switches the preset to "custom".
  const [colorSensitivity, setColorSensitivity] = useState<ColorSensitivity | "custom">("forgiving");
  // Which filament set is on the color card: classic RGB, or CMY (cyan in
  // red's slot, yellow in green's, magenta in blue's + the dot).
  // ARCHIVED: the RGB palette and the checkerboard pattern mode remain in
  // the detection library but are retired from the UI — the product path is
  // the CMY card (black dot) and the red shapes card.
  const colorPalette: ColorPalette = "cmy";
  const colorPaletteRef = useRef<ColorPalette>("cmy");
  // Per-channel [floor, margin] gates — what actually classifies pixels.
  const [colorTuning, setColorTuning] = useState<ColorGateTuning>(() => tuningFromPreset("forgiving", "cmy"));
  const colorTuningRef = useRef<ColorGateTuning>(colorTuning);
  useEffect(() => {
    colorTuningRef.current = colorTuning;
  }, [colorTuning]);
  // REFERENCE colors: the card's actual patch colors, sampled by eyedropper
  // or captured automatically at the first REGIONAL lock. While set (all 3
  // slots), acquisition classifies by distance to THESE instead of the
  // gates — valid as long as the shooter and lighting stay put. ☀/🌙 scale
  // them; the tolerance slider sets how much brightening/darkening of the
  // scene is accepted around them.
  type RefDraft = { red: [number, number, number] | null; green: [number, number, number] | null; blue: [number, number, number] | null };
  const [refDraft, setRefDraft] = useState<RefDraft>({ red: null, green: null, blue: null });
  // Color-photo countdown: the reference snapshot is NEVER taken by surprise —
  // a spoken, visible 3-2-1 runs first so the shooter knows to freeze.
  const [photoCount, setPhotoCount] = useState<number | null>(null);
  const photoBusyRef = useRef(false);
  const photoTimersRef = useRef<number[]>([]);
  const startPhotoCountdownRef = useRef<() => void>(() => {});
  // The photo is a STAGED sequence, and the active stage is explicit:
  //   brief     — instructions are being spoken; matches are IGNORED, the
  //               shooter is still seating the card / reading
  //   position  — waiting for the shooter to raise the gun and hold a
  //               steady lock; nothing fires on a momentary match
  //   countdown — the spoken 3-2-1 is running; freeze
  //   done      — reference captured, calibration continues
  type PhotoStage = "idle" | "brief" | "position" | "countdown" | "done";
  const [photoStage, setPhotoStage] = useState<PhotoStage>("idle");
  const photoStageRef = useRef<PhotoStage>("idle");
  useEffect(() => {
    photoStageRef.current = photoStage;
  }, [photoStage]);
  // "Reasonable lock" evidence while the marching ring is up: every strict or
  // loose observation landing INSIDE the ring refreshes this streak. The
  // human is holding the card exactly where they were told to — if a
  // detector keeps agreeing there, that area IS the card, even when the
  // formal track can't mature (dim room, marginal gates). Feeds the relaxed
  // photo arm and the countdown-zero capture fallback.
  const zoneSeenRef = useRef<{ streakStartMs: number; lastMs: number; colors: TrackedColors | null }>({
    streakStartMs: 0,
    lastMs: 0,
    colors: null,
  });
  // The center ring on the preview is GUIDANCE ONLY — no detection or
  // capture boundary. This ref just colors the ring's label green when the
  // card happens to sit inside it.
  const photoZoneOkRef = useRef(false);
  // COLOR COMMIT: a lock held ≥15 s has proven its colors beyond doubt —
  // they are frozen as THE card colors for the rest of this training
  // instance (no more per-frame adaptation, reseeds reuse them), cleared
  // when the instance ends or the user explicitly restarts/retakes.
  const colorsCommittedRef = useRef<TrackedColors | null>(null);
  const refColorsRef = useRef<TrackedColors | null>(null);
  useEffect(() => {
    refColorsRef.current = refDraft.red && refDraft.green && refDraft.blue ? (refDraft as TrackedColors) : null;
  }, [refDraft]);
  const [refBrightTol, setRefBrightTol] = useState(0.7);
  const refBrightTolRef = useRef(0.7);
  useEffect(() => {
    refBrightTolRef.current = refBrightTol;
  }, [refBrightTol]);
  const [pickSlot, setPickSlot] = useState<"red" | "green" | "blue" | null>(null);
  const pickSlotRef = useRef<"red" | "green" | "blue" | null>(null);
  useEffect(() => {
    pickSlotRef.current = pickSlot;
  }, [pickSlot]);
  // Eyedropper: sample the (untinted) processing frame at normalized coords.
  const sampleProcRef = useRef<((nx: number, ny: number) => [number, number, number] | null) | null>(null);
  // "Help computer." — can't lock? Freeze a still of the shooting position
  // and walk the user through tapping the patches in order.
  const [helpPick, setHelpPick] = useState<{ stage: number } | null>(null);
  const helpPickRef = useRef<{ stage: number } | null>(null);
  useEffect(() => {
    helpPickRef.current = helpPick;
  }, [helpPick]);
  const freezeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureFrameRef = useRef<(() => boolean) | null>(null);
  // Background subtraction (Help computer): raw proc-frame grabs + the
  // stored no-gun background frame.
  const captureProcImageRef = useRef<(() => ImageData | null) | null>(null);
  const bgFrameRef = useRef<ImageData | null>(null);
  // Tap order + slot mapping per palette.
  const helpOrder: { slot: "red" | "green" | "blue"; name: string }[] =
    colorPalette === "cmy"
      ? [
          { slot: "green", name: "CYAN" },
          { slot: "red", name: "YELLOW" },
          { slot: "blue", name: "MAGENTA" },
        ]
      : [
          { slot: "red", name: "RED" },
          { slot: "green", name: "GREEN" },
          { slot: "blue", name: "BLUE" },
        ];
  // Selecting a preset (or flipping the palette while one is active) reseeds
  // the sliders from that preset's table; "custom" sticks to the sliders.
  // Reseeding happens in the button handlers (not an effect) so it can't
  // cascade renders.
  // Shape mode: how strict the red gate is. Pick the strictest level whose
  // tint overlay still fills the card's patches solidly.
  const [redSensitivity, setRedSensitivity] = useState<RedSensitivity>("strict");
  const redSensitivityRef = useRef<RedSensitivity>("strict");
  useEffect(() => {
    redSensitivityRef.current = redSensitivity;
  }, [redSensitivity]);
  // Which shapes a lock must positively identify. Un-required slots accept
  // any red blob at the right layout position.
  const [shapeReq, setShapeReq] = useState<ShapeRequirements>(DEFAULT_SHAPE_REQUIREMENTS);
  const shapeReqRef = useRef<ShapeRequirements>(DEFAULT_SHAPE_REQUIREMENTS);
  useEffect(() => {
    shapeReqRef.current = shapeReq;
  }, [shapeReq]);
  // Diagnostics mode: bigger annotated preview + per-stage detector readout.
  const [showDiag, setShowDiag] = useState(true);
  const showDiagRef = useRef(true);
  useEffect(() => {
    showDiagRef.current = showDiag;
  }, [showDiag]);
  // Ideal-position guide on the camera preview (toggleable): silhouette +
  // sweet-spot for where the shooter and muzzle flag should sit in frame.
  const [showGuide, setShowGuide] = useState(true);
  const showGuideRef = useRef(true);
  useEffect(() => {
    showGuideRef.current = showGuide;
  }, [showGuide]);
  const [diag, setDiag] = useState<DiagInfo | null>(null);
  const [coach, setCoach] = useState<string | null>(null);
  const diagRef = useRef<{ debug: FlagDebug; roi: { x: number; y: number; w: number; h: number } | null } | null>(null);

  // Fixed zone set per mount (dry-fire is about the shooter, not the layout):
  // NINE equal targets on a 3×3 grid, small enough that aim error between
  // neighbors is unambiguous, spaced to fit any screen aspect.
  const [zones] = useState<ScenarioZone[]>(() => {
    const raw = generateScenarioZones(9, DEFAULT_SCENARIO_ATTRIBUTES, generateSeed());
    return raw.map((zone, i) => ({
      ...zone,
      cx: (1 + 2 * (i % 3)) / 6, // 1/6, 1/2, 5/6
      cy: [0.18, 0.5, 0.82][Math.floor(i / 3)],
      radius: 0.075,
    }));
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const boardWrapRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const obsRef = useRef<{ obs: FlagObservation; atMs: number } | null>(null);
  const trackRef = useRef<TrackState | null>(null);
  // Switching cards (or the filament palette) invalidates the learned
  // appearance.
  useEffect(() => {
    trackRef.current = null;
  }, [patternMode, colorPalette]);
  const smoothFeaturesRef = useRef<number[] | null>(null);
  // Constant-velocity filter with outlier rejection (replaces the plain EMA).
  const featureFilterRef = useRef(createFeatureFilter());
  // Raw per-frame observations from the last second — shots sample the
  // window just BEFORE the click, when the sights were still honest.
  const aimHistoryRef = useRef<{ atMs: number; features: number[] }[]>([]);
  // Which feature regime the aim model was fitted on: "h" = homography
  // frames (dot found), "a" = affine-only. Click sampling sticks to the
  // model's regime — mixing them biases the prediction.
  const modelRegimeRef = useRef<"h" | "a">("a");
  // Perspective-frame log across distinct poses — the raw material for
  // Zhang-style camera self-calibration (future metric pose work).
  const homographyLogRef = useRef<number[][]>([]);
  const hLogWriteRef = useRef(0);
  const [hLogCount, setHLogCount] = useState(0);
  // Hold test: collect aim for 10 s, report RMS jitter + drift.
  const holdTestRef = useRef<{ startMs: number; until: number; pts: { x: number; y: number; atMs: number }[] } | null>(
    null,
  );
  const [holdActive, setHoldActive] = useState(false);
  const [holdResult, setHoldResult] = useState<{ rmsPx: number; driftPxPerS: number; n: number } | null>(null);
  const dwellRef = useRef<{ atMs: number; features: number[]; cx: number; cy: number }[]>([]);
  const samplesRef = useRef<AimSample[]>([]);
  const calIndexRef = useRef(0);
  const calCooldownUntilRef = useRef(0);
  const modelRef = useRef<AimModel | null>(null);
  // Provisional model refit after every captured dot (from the 3rd on) — it
  // drives the live "estimated aim" marker during calibration so you can
  // watch the solution converge.
  const provisionalModelRef = useRef<AimModel | null>(null);
  const traceRef = useRef<{ x: number; y: number; atMs: number }[]>([]); // viewport px
  const shotMarkersRef = useRef<{ nx: number; ny: number; correct: boolean | null; atMs: number }[]>([]);
  const drillRef = useRef<DrillState | null>(null);
  const beginDrillRef = useRef<() => void>(() => {});
  const shotCountRef = useRef(0);
  const feedbackTimerRef = useRef<number | null>(null);
  const triggerRef = useRef<ClickTrigger | null>(null);

  useScreenWakeLock(phase !== "setup");

  // Aspect-corrected zone hit test (radii are width-normalized). The aspect
  // is the board's RENDERED width/height — the sheet ratio in windowed mode,
  // the viewport's in fullscreen.
  const zoneAt = (nx: number, ny: number, aspect: number): string | null => {
    let bestId: string | null = null;
    let bestDist = Infinity;
    for (const zone of zones) {
      const dist = Math.hypot(nx - zone.cx, (ny - zone.cy) / Math.max(0.1, aspect));
      if (dist <= zone.radius && dist < bestDist) {
        bestDist = dist;
        bestId = zone.id;
      }
    }
    return bestId;
  };

  // Viewport aspect for the fullscreen board (the target fills 100vw×100vh).
  const [winAspect, setWinAspect] = useState(16 / 9);
  useEffect(() => {
    const update = () => setWinAspect(window.innerWidth / Math.max(1, window.innerHeight));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        typeof window !== "undefined" && !window.isSecureContext
          ? "Camera needs HTTPS (or localhost)."
          : "Camera API unavailable in this browser.",
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "user" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play().catch(() => undefined);
      setCameraActive(true);
      setStatusLine("Camera running. Hold the gun naturally — waiting for the flag to lock.");
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Unable to access the camera.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
      cancelSpeech();
      triggerRef.current?.stop();
      if (feedbackTimerRef.current) window.clearTimeout(feedbackTimerRef.current);
    };
     
  }, []);

  // Invalidate calibration on resize/scroll (targets were captured in
  // viewport coordinates).
  useEffect(() => {
    const onChange = () => {
      if (modelRef.current || samplesRef.current.length > 0) setResizedSinceCal(true);
    };
    window.addEventListener("resize", onChange);
    return () => window.removeEventListener("resize", onChange);
  }, []);

  // ---- The processing loop: video → gray → flag → (calibrate | aim) --------
  useEffect(() => {
    if (!cameraActive) return;
    const video = videoRef.current;
    if (!video) return;

    const procCanvas = document.createElement("canvas");
    const procCtx = procCanvas.getContext("2d", { willReadFrequently: true });
    if (!procCtx) return;
    // Dedicated canvas for FULL-RESOLUTION ROI processing in track mode —
    // the ROI is small, so it can afford native-camera detail even though
    // full frames are downscaled.
    const roiCanvas = document.createElement("canvas");
    const roiCtx = roiCanvas.getContext("2d", { willReadFrequently: true });

    // "Help computer.": snapshot the current frame for guided picking.
    captureFrameRef.current = () => {
      if (procCanvas.width < 5) return false;
      const c = freezeCanvasRef.current ?? document.createElement("canvas");
      freezeCanvasRef.current = c;
      c.width = procCanvas.width;
      c.height = procCanvas.height;
      const fctx = c.getContext("2d");
      if (!fctx) return false;
      fctx.drawImage(procCanvas, 0, 0);
      return true;
    };
    // Background-subtraction source: the raw processing frame as ImageData.
    captureProcImageRef.current = () => {
      if (procCanvas.width < 5) return null;
      return procCtx.getImageData(0, 0, procCanvas.width, procCanvas.height);
    };
    // Eyedropper source: 5×5 mean from the raw processing frame.
    sampleProcRef.current = (nx, ny) => {
      const w = procCanvas.width;
      const h = procCanvas.height;
      if (w < 5 || h < 5) return null;
      const x = Math.max(2, Math.min(w - 3, Math.round(nx * w)));
      const y = Math.max(2, Math.min(h - 3, Math.round(ny * h)));
      const d = procCtx.getImageData(x - 2, y - 2, 5, 5).data;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let i = 0; i < 25; i += 1) {
        r += d[i * 4];
        g += d[i * 4 + 1];
        b += d[i * 4 + 2];
      }
      return [r / 25, g / 25, b / 25];
    };

    // Rescale an observation / debug produced at a higher-resolution pixel
    // scale back into processing-frame coordinates (features are already
    // scale-invariant — coordinates and normPx scale together).
    const scaleObs = (o: FlagObservation, f: number): FlagObservation => ({
      ...o,
      tiles: o.tiles.map((p) => ({ x: p.x * f, y: p.y * f })),
      dot: { x: o.dot.x * f, y: o.dot.y * f },
      center: { x: o.center.x * f, y: o.center.y * f },
      cellPx: o.cellPx * f,
    });
    const scaleDebug = (d: FlagDebug, f: number) => {
      d.blobs = d.blobs.map((b) => ({ ...b, x: b.x * f, y: b.y * f }));
      d.tileCandidates = d.tileCandidates.map((p) => ({ x: p.x * f, y: p.y * f }));
      d.dotCandidates = d.dotCandidates.map((p) => ({ x: p.x * f, y: p.y * f }));
      if (d.quad) d.quad = d.quad.map((p) => ({ x: p.x * f, y: p.y * f }));
      if (d.cellPx !== null) d.cellPx *= f;
      if (d.colorGates?.predictedDot)
        d.colorGates.predictedDot = { x: d.colorGates.predictedDot.x * f, y: d.colorGates.predictedDot.y * f };
      if (d.colorGates?.dotBox)
        d.colorGates.dotBox = {
          x: d.colorGates.dotBox.x * f,
          y: d.colorGates.dotBox.y * f,
          w: d.colorGates.dotBox.w * f,
          h: d.colorGates.dotBox.h * f,
        };
    };

    let raf = 0;
    let lastFullScanMs = 0;
    // Background subtraction: per-camera-session model, stepped on the same
    // throttled cadence as full scans (that's when a full frame is in hand).
    const bgModel = createBackgroundModel();
    let dotWaitIdx = -1; // per-calibration-dot stall timer
    let dotWaitStartMs = 0;
    let frameCount = 0;
    let fpsWindowStart = performance.now();
    let fps = 0;
    let lastStatusPushMs = 0;
    let stopped = false;

    const toGray = (data: Uint8ClampedArray, n: number): Uint8Array => {
      const gray = new Uint8Array(n);
      for (let i = 0; i < n; i += 1) {
        const o = i * 4;
        gray[i] = (data[o] * 77 + data[o + 1] * 150 + data[o + 2] * 29) >> 8;
      }
      return gray;
    };

    const captureCalibrationSample = (now: number) => {
      const idx = calIndexRef.current;
      const wrap = boardWrapRef.current;
      if (!wrap || idx >= CAL_POINTS.length) return;
      const rect = wrap.getBoundingClientRect();
      const [nx, ny] = CAL_POINTS[idx];
      // Average within ONE feature regime — homography frames (dot found)
      // and affine frames have systematically different linear terms, and a
      // mixed mean lands between them (an aim bias that wanders with the
      // dot's visibility). Use whichever regime dominates this dwell.
      const dwellFeats = dwellRef.current.map((d) => d.features);
      const hFeats = dwellFeats.filter((f) => f[7] !== 0 || f[8] !== 0);
      const aFeats = dwellFeats.filter((f) => f[7] === 0 && f[8] === 0);
      const majority = hFeats.length >= aFeats.length ? hFeats : aFeats;
      samplesRef.current.push({
        features: meanFeatures(majority.length >= 3 ? majority : dwellFeats),
        targetX: rect.left + nx * rect.width,
        targetY: rect.top + ny * rect.height,
      });
      dwellRef.current = [];
      calCooldownUntilRef.current = now + 900;
      const next = idx + 1;
      calIndexRef.current = next;
      setCalIndex(next);
      setCalHold(0);
      // Refit the provisional model from whatever dots exist so far (heavier
      // ridge — few points want more regularization).
      provisionalModelRef.current = fitAimModel(samplesRef.current, 1e-3, 3);
      if (next < CAL_POINTS.length) {
        speak("Mark. Next dot.");
      } else {
        // NOTE: the per-dot COLOR REFIT was removed by design — the 9-dot
        // calibration touches NOTHING about color. Point of aim comes from
        // placement and card geometry (the pose features) alone; the color
        // reference only ever changes when the user explicitly asks.
        // Remember which regime the model speaks — click sampling must match.
        const hDots = samplesRef.current.filter((sm) => sm.features[7] !== 0 || sm.features[8] !== 0).length;
        modelRegimeRef.current = hDots >= samplesRef.current.length / 2 ? "h" : "a";
        const fitted = fitAimModel(samplesRef.current);
        if (fitted) {
          modelRef.current = fitted;
          setModel(fitted);
          setResizedSinceCal(false);
          setPhase("train");
          setStatusLine(
            `Calibrated — fit residual ${fitted.rmsErrorPx.toFixed(0)}px over ${fitted.sampleCount} dots. Drill starting…`,
          );
          speak("Calibration complete.");
          // Straight into a called drill — no extra tap. Small delay so the
          // completion callout isn't cancelled by the first drill call.
          window.setTimeout(() => {
            if (phaseRef.current !== "train") return;
            if (trainModeRef.current === "target") {
              speak("Target is hot. Fire when ready.");
              return;
            }
            if (!drillRef.current || drillRef.current.status !== "running") {
              beginDrillRef.current();
            }
          }, 1800);
        } else {
          setStatusLine("Calibration failed to fit — restart calibration.");
          speak("Calibration failed.");
        }
      }
    };

    const processFrame = () => {
      if (stopped) return;
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw <= 0 || vh <= 0) {
        raf = requestAnimationFrame(processFrame);
        return;
      }
      const procW = Math.min(vw, 1600);
      const procH = Math.round((vh / vw) * procW);
      if (procCanvas.width !== procW || procCanvas.height !== procH) {
        procCanvas.width = procW;
        procCanvas.height = procH;
      }
      procCtx.drawImage(video, 0, 0, procW, procH);

      const now = performance.now();
      const last = obsRef.current;
      let obs: FlagObservation | null = null;
      let frameDebug: FlagDebug | null = null;
      let frameRoi: { x: number; y: number; w: number; h: number } | null = null;

      const runDetect = (data: Uint8ClampedArray, w: number, h: number, ox: number, oy: number, fg: ForegroundGrid | null = null) =>
        patternModeRef.current === "color"
          ? analyzeColorFlag(
              data,
              w,
              h,
              procW,
              ox,
              oy,
              colorTuningRef.current,
              colorPaletteRef.current,
              refColorsRef.current,
              refBrightTolRef.current,
              fg,
            )
          : patternModeRef.current === "shape"
            ? analyzeShapeFlag(data, w, h, procW, ox, oy, redSensitivityRef.current, shapeReqRef.current)
            : analyzeFlag(toGray(data, w * h), w, h, procW, ox, oy);

      const seedTrack = (fromObs: FlagObservation) => {
        if (patternModeRef.current === "checker" || !fromObs.quadColors) return;
        trackRef.current = {
          // Committed colors (a ≥15 s proven lock) override whatever this
          // seed sampled — the card's identity is settled for the instance.
          colors: colorsCommittedRef.current ?? fromObs.quadColors,
          center: { ...fromObs.center },
          velocity: { x: 0, y: 0 },
          cellPx: fromObs.cellPx,
          affine: [
            fromObs.features[3] * procW,
            fromObs.features[4] * procW,
            fromObs.features[5] * procW,
            fromObs.features[6] * procW,
          ],
          atMs: now,
          misses: 0,
          seedAtMs: now,
          hits: 0,
          seedCellPx: fromObs.cellPx,
        };
      };

      // ---- STAND DOWN until it matters: no locking while the user is still
      // reading instructions or racking the slide. Detection arms as soon as
      // the target board is up — INCLUDING during the safety brief, so the
      // shooter can settle on dot 1 and the aim calibration can already be
      // capturing while the announcement plays. Also armed in pro mode and
      // during a manual photo/help action.
      const detectionArmed =
        uiModeRef.current === "pro" ||
        phaseRef.current !== "setup" ||
        photoStageRef.current === "position" ||
        photoStageRef.current === "countdown" ||
        helpPickRef.current !== null;
      if (!detectionArmed && trackRef.current) trackRef.current = null;

      // ---- TRACK MODE (color card): the locked card is the prime candidate.
      // Every frame: predict where it moved, search a tight ROI, classify by
      // its LEARNED colors. Misses widen the search progressively; after ~16
      // the track is surrendered back to acquisition.
      const track = !detectionArmed || patternModeRef.current === "checker" ? null : trackRef.current;
      if (track) {
        const dtMs = Math.max(1, now - track.atMs);
        const predX = track.center.x + track.velocity.x * dtMs;
        const predY = track.center.y + track.velocity.y * dtMs;
        // Lock maturity: after a brief second of confirmed tracking the ROI
        // is trusted, and everything relaxes INSIDE it — gates open, the
        // region stays isolated (widens slower on misses), and lighting
        // adaptation speeds up. Fresh locks stay strict so a wrong seed
        // can't dig in.
        const regional = track.hits >= REGIONAL_AFTER_HITS && now - track.seedAtMs >= REGIONAL_AFTER_MS;
        // NO area boundary while re-seeking: a single missed frame widens
        // the search to the ENTIRE frame — the card is tracked wherever it
        // went, not just near where it was. (While hitting, the tight ROI is
        // pure CPU savings, not a constraint — the card is in it.)
        const side = Math.round(
          track.misses > 0 ? Math.max(procW, procH) : Math.min(Math.max(track.cellPx * 14, 220), Math.max(procW, procH)),
        );
        const rx = Math.max(0, Math.min(procW - Math.min(side, procW), Math.round(predX - side / 2)));
        const ry = Math.max(0, Math.min(procH - Math.min(side, procH), Math.round(predY - side / 2)));
        const rw = Math.min(side, procW - rx);
        const rh = Math.min(side, procH - ry);
        const trackFn = patternModeRef.current === "shape" ? trackShapeFlag : trackColorFlag;
        // Full-resolution ROI: the camera usually has more detail than the
        // downscaled processing frame — spend it where the card actually is.
        const kNative = vw / procW;
        const s2 = Math.min(2, kNative);
        let result: ReturnType<typeof trackFn>;
        if (s2 > 1.2 && roiCtx) {
          const dw = Math.round(rw * s2);
          const dh = Math.round(rh * s2);
          if (roiCanvas.width !== dw || roiCanvas.height !== dh) {
            roiCanvas.width = dw;
            roiCanvas.height = dh;
          }
          roiCtx.drawImage(video, rx * kNative, ry * kNative, rw * kNative, rh * kNative, 0, 0, dw, dh);
          const img = roiCtx.getImageData(0, 0, dw, dh);
          result = trackFn(img.data, dw, dh, procW * s2, rx * s2, ry * s2, {
            colors: track.colors,
            predictedCenter: { x: predX * s2, y: predY * s2 },
            cellPx: track.cellPx * s2,
            affine: track.affine.map((v) => v * s2) as [number, number, number, number],
            relax: regional ? 1 : 0,
            misses: track.misses,
            palette: colorPaletteRef.current,
          });
          if (result.observation) result.observation = scaleObs(result.observation, 1 / s2);
          scaleDebug(result.debug, 1 / s2);
        } else {
          const img = procCtx.getImageData(rx, ry, rw, rh);
          result = trackFn(img.data, rw, rh, procW, rx, ry, {
            colors: track.colors,
            predictedCenter: { x: predX, y: predY },
            cellPx: track.cellPx,
            affine: track.affine,
            relax: regional ? 1 : 0,
            misses: track.misses,
            palette: colorPaletteRef.current,
          });
        }
        frameDebug = result.debug;
        frameRoi = { x: rx, y: ry, w: rw, h: rh };
        if (result.observation) {
          obs = result.observation;
          const instVx = (obs.center.x - track.center.x) / dtMs;
          const instVy = (obs.center.y - track.center.y) / dtMs;
          track.velocity = {
            x: track.velocity.x + 0.5 * (instVx - track.velocity.x),
            y: track.velocity.y + 0.5 * (instVy - track.velocity.y),
          };
          track.center = { ...obs.center };
          // Scale rides the observation but stays anchored to the seed:
          // a runaway prior (which balloons minArea until nothing passes)
          // can't compound past 2.5× what acquisition actually verified.
          track.cellPx = Math.min(Math.max(obs.cellPx, track.seedCellPx * 0.4), track.seedCellPx * 2.5);
          track.affine = [
            obs.features[3] * procW,
            obs.features[4] * procW,
            obs.features[5] * procW,
            obs.features[6] * procW,
          ];
          track.atMs = now;
          track.misses = 0;
          track.hits += 1;
          // Follow lighting drift — faster once the region is trusted, so
          // the learned colors ride through exposure swings.
          if (obs.quadColors && !colorsCommittedRef.current) {
            track.colors = blendColors(track.colors, obs.quadColors, regional ? 0.25 : 0.15);
          }
          // COLOR COMMIT: 15 s of sustained lock settles the card's colors
          // for the remainder of this training instance — adaptation stops,
          // the acquisition reference adopts them, reseeds reuse them.
          if (
            !colorsCommittedRef.current &&
            patternModeRef.current === "color" &&
            now - track.seedAtMs >= 15000
          ) {
            const committed: TrackedColors = {
              red: [...track.colors.red] as [number, number, number],
              green: [...track.colors.green] as [number, number, number],
              blue: [...track.colors.blue] as [number, number, number],
            };
            colorsCommittedRef.current = committed;
            setRefDraft({ red: committed.red, green: committed.green, blue: committed.blue });
          }
        } else {
          // Dead-reckon forward, damp the velocity, widen next frame's search.
          track.center = { x: predX, y: predY };
          track.velocity = { x: track.velocity.x * 0.9, y: track.velocity.y * 0.9 };
          track.atMs = now;
          track.misses += 1;
          // BEACON STEER: the sub-millisecond rank pass finds where the CYM
          // ink is RIGHT NOW; recenter the next search there instead of
          // dead-reckoning a stale prediction into empty space. (The track
          // gates still verify whatever the steered search finds.)
          if (patternModeRef.current === "color" && track.misses >= 3 && track.misses % 2 === 1) {
            const full = procCtx.getImageData(0, 0, procW, procH);
            const beacon = cymBeacon(full.data, procW, procH);
            if (beacon) {
              track.center = { x: beacon.x, y: beacon.y };
              track.velocity = { x: 0, y: 0 };
            }
          }
          // SELF-RESCUE before surrender: after a run of misses, run the full
          // strict analyzer on the track's own ROI. If the card is still
          // there (it usually is — a marginal gate lost it, not the scene),
          // re-seed IN PLACE and carry the lock's maturity over, instead of
          // dropping to acquisition and visibly flip-flopping the lock.
          if (track.misses >= 6 && track.misses % 3 === 0) {
            const rescueImg = procCtx.getImageData(rx, ry, rw, rh);
            let rescue = runDetect(rescueImg.data, rw, rh, rx, ry);
            if (!rescue.observation && patternModeRef.current === "color") {
              // Strict rescue failed — the rank-based path gets a shot at
              // the same ROI before the misses keep climbing.
              rescue = analyzeColorFlagLoose(rescueImg.data, rw, rh, procW, rx, ry);
            }
            if (rescue.observation) {
              obs = rescue.observation;
              frameDebug = rescue.debug;
              const prevSeedAtMs = track.seedAtMs;
              const prevHits = track.hits;
              seedTrack(obs);
              const reseeded = trackRef.current;
              if (reseeded) {
                // Same card, same place, verified by the strict path — the
                // lock keeps its maturity rather than restarting cold.
                reseeded.seedAtMs = prevSeedAtMs;
                reseeded.hits = prevHits;
              }
            }
          }
          // A matured lock gets more patience before surrendering the region
          // back to full-frame acquisition — with the seek-widened gates and
          // the self-rescue it rarely comes to this.
          if (!obs && track.misses > (regional ? 40 : 16)) {
            trackRef.current = null;
            lastFullScanMs = 0; // reacquire NOW, not after the scan throttle
          }
        }
      }

      // ---- ACQUISITION (checker mode always; color mode when untracked) ----
      if (detectionArmed && !obs && !trackRef.current) {
        if (patternModeRef.current !== "color" && last && now - last.atMs < 700) {
          // Checker mode keeps the simple recent-position ROI.
          const side = Math.max(180, Math.min(680, Math.round(last.obs.cellPx * 16)));
          const rx = Math.max(0, Math.min(procW - side, Math.round(last.obs.center.x - side / 2)));
          const ry = Math.max(0, Math.min(procH - side, Math.round(last.obs.center.y - side / 2)));
          const rw = Math.min(side, procW - rx);
          const rh = Math.min(side, procH - ry);
          const img = procCtx.getImageData(rx, ry, rw, rh);
          const result = runDetect(img.data, rw, rh, rx, ry);
          obs = result.observation;
          frameDebug = result.debug;
          frameRoi = { x: rx, y: ry, w: rw, h: rh };
        }
        if (!obs && now - lastFullScanMs > (showDiagRef.current ? 250 : 400)) {
          lastFullScanMs = now;
          const img = procCtx.getImageData(0, 0, procW, procH);
          // Background subtraction: learn the static scene, get the cells
          // that changed. The last-seen card region is excluded from
          // adaptation so a steady hold can't be absorbed. Null when the
          // veto shouldn't be trusted (warmup, nothing moving, camera bump).
          const exclObs = last && now - last.atMs < 1500 ? last.obs : null;
          const fgGrid =
            patternModeRef.current === "color"
              ? bgModel.step(
                  img.data,
                  procW,
                  procH,
                  exclObs
                    ? {
                        x: exclObs.center.x - exclObs.cellPx * 3,
                        y: exclObs.center.y - exclObs.cellPx * 3,
                        w: exclObs.cellPx * 6,
                        h: exclObs.cellPx * 6,
                      }
                    : null,
                )
              : null;
          // Color mode: preliminary tri-color scan first. Rank the frame for
          // "most cyan + most yellow + most magenta TOGETHER" and focus
          // acquisition on that region — a wall or shirt flooding one channel
          // can no longer offer the analyzer a plausible-but-wrong layout
          // elsewhere in frame. Full-frame scan only when no tri-color
          // region stands out.
          // NO area boundaries: detection runs over the WHOLE frame in every
          // state — the photo ring is guidance, never a detection window.
          let pre: ReturnType<typeof prescanCMY> = null;
          if (patternModeRef.current === "color") {
            pre = prescanCMY(img.data, procW, procH);
            if (pre) {
              const sub = procCtx.getImageData(pre.roi.x, pre.roi.y, pre.roi.w, pre.roi.h);
              let result = fgGrid ? runDetect(sub.data, pre.roi.w, pre.roi.h, pre.roi.x, pre.roi.y, fgGrid) : null;
              if (!result?.observation) result = runDetect(sub.data, pre.roi.w, pre.roi.h, pre.roi.x, pre.roi.y);
              obs = result.observation;
              frameDebug = result.debug;
              frameRoi = pre.roi;
            }
            if (!obs) {
              // MOST-FORGIVING fallback: rank-based "the three most distinct
              // colors in frame" acquisition — no absolute gates, full frame.
              // With the motion veto the ranking sees only MOVING ink, so it
              // runs first; the un-vetoed pass remains the last resort.
              let loose = fgGrid ? analyzeColorFlagLoose(img.data, procW, procH, procW, 0, 0, fgGrid) : null;
              if (!loose?.observation) loose = analyzeColorFlagLoose(img.data, procW, procH, procW, 0, 0);
              if (loose.observation) {
                obs = loose.observation;
                frameDebug = loose.debug;
                frameRoi = null;
              }
            }
            if (!obs && pre) {
              // Last resort: the pre-scan's focus region is a fixed size and
              // can CROP a close-up card, so the focused strict pass fails on
              // a perfectly good frame. Whole-frame strict before giving up.
              const result = runDetect(img.data, procW, procH, 0, 0);
              obs = result.observation;
              frameDebug = result.debug;
              frameRoi = null;
            }
          }
          if (!obs && !pre) {
            let result = fgGrid ? runDetect(img.data, procW, procH, 0, 0, fgGrid) : null;
            if (!result?.observation) result = runDetect(img.data, procW, procH, 0, 0);
            obs = result.observation;
            frameDebug = result.debug;
            frameRoi = null;
          }
        }
        // A fresh color acquisition seeds the tracker with the card's
        // learned appearance.
        if (obs) seedTrack(obs);
      }
      if (frameDebug) diagRef.current = { debug: frameDebug, roi: frameRoi };

      if (obs) {
        obsRef.current = { obs, atMs: now };
        // Photo-stage evidence: an observation inside the marching ring is
        // "the card is where the human says it is". Gaps over 900 ms break
        // the streak; the freshest quadrant colors ride along so the capture
        // can use them even if the formal track slips at the last moment.
        if (
          patternModeRef.current === "color" &&
          (photoStageRef.current === "position" || photoStageRef.current === "countdown") &&
          Math.hypot(obs.center.x - procW / 2, obs.center.y - procH / 2) <= Math.min(procW, procH) * PHOTO_ZONE_R
        ) {
          const z = zoneSeenRef.current;
          if (now - z.lastMs > 900) z.streakStartMs = now;
          z.lastMs = now;
          if (obs.quadColors) z.colors = obs.quadColors;
        }
        // Fit quality (from the pose's reprojection residual) scales how
        // much this frame is allowed to move the filtered state.
        const quality = Math.min(1, Math.max(0.3, 1 - obs.residual * 2));
        smoothFeaturesRef.current = featureFilterRef.current.update(obs.features, now, quality);
        // Raw history for click-time aim sampling.
        aimHistoryRef.current.push({ atMs: now, features: obs.features });
        while (aimHistoryRef.current.length > 0 && now - aimHistoryRef.current[0].atMs > 1000) {
          aimHistoryRef.current.shift();
        }
        // Background HOMOGRAPHY LOG for Zhang self-calibration: perspective
        // frames at sufficiently distinct poses, capped ring buffer. Enough
        // of these over-determine the camera intrinsics later.
        if (obs.features[7] !== 0 || obs.features[8] !== 0) {
          const log = homographyLogRef.current;
          const lastH = log.length > 0 ? log[log.length - 1] : null;
          const distinct =
            !lastH ||
            obs.features.reduce((s, v, i) => (i >= 3 ? s + Math.abs(v - lastH[i]) : s), 0) > 0.02;
          if (distinct) {
            if (log.length < 400) log.push([...obs.features]);
            else log[hLogWriteRef.current++ % 400] = [...obs.features];
          }
        }
      } else if (last && now - last.atMs > 900) {
        smoothFeaturesRef.current = null;
        featureFilterRef.current.reset();
      }

      // --- Calibration dwell logic ---
      if (phaseRef.current === "calibrate" && obs && now > calCooldownUntilRef.current) {
        dwellRef.current.push({ atMs: now, features: obs.features, cx: obs.center.x, cy: obs.center.y });
        dwellRef.current = dwellRef.current.filter((d) => now - d.atMs <= 1300);
        const window_ = dwellRef.current;
        if (window_.length >= 8) {
          let minX = Infinity;
          let maxX = -Infinity;
          let minY = Infinity;
          let maxY = -Infinity;
          for (const d of window_) {
            if (d.cx < minX) minX = d.cx;
            if (d.cx > maxX) maxX = d.cx;
            if (d.cy < minY) minY = d.cy;
            if (d.cy > maxY) maxY = d.cy;
          }
          const drift = Math.max(maxX - minX, maxY - minY);
          // Hand tremor + centroid jitter easily wander ~10 px of card
          // center at processing resolution — the old 0.15-cell limit made
          // a perfectly good lock look "unstable" and the dot never fired.
          // The dwell MEAN is what's captured, so jitter averages out.
          const driftLimit = Math.max(10, obs.cellPx * 0.5);
          const span = window_[window_.length - 1].atMs - window_[0].atMs;
          const stable = drift <= driftLimit;
          if (calIndexRef.current !== dotWaitIdx) {
            dotWaitIdx = calIndexRef.current;
            dotWaitStartMs = now;
          }
          setCalHold(stable ? Math.min(1, span / 900) : 0);
          // Fire on a steady hold — or, if the shooter has clearly been ON
          // the dot for 7 s without ever reading "stable", take the mean
          // anyway rather than stalling the whole calibration. The fallback
          // still demands a ROUGHLY held aim (3× the drift limit) — it must
          // never capture a gun that's being racked or waved around.
          if ((stable && span >= 900) || (now - dotWaitStartMs > 7000 && span >= 900 && drift <= driftLimit * 3)) {
            captureCalibrationSample(now);
          }
        } else {
          setCalHold(0);
        }
      }

      // --- Aim trace (training) ---
      if (phaseRef.current === "train" && modelRef.current && smoothFeaturesRef.current && obs) {
        const aim = predictAim(modelRef.current, smoothFeaturesRef.current);
        traceRef.current.push({ x: aim.x, y: aim.y, atMs: now });
        while (traceRef.current.length > 0 && now - traceRef.current[0].atMs > TRACE_WINDOW_MS) traceRef.current.shift();
        // Hold test: collect for the window, then report RMS jitter + drift.
        const ht = holdTestRef.current;
        if (ht && now >= ht.startMs) {
          if (now <= ht.until) {
            ht.pts.push({ x: aim.x, y: aim.y, atMs: now });
          } else {
            holdTestRef.current = null;
            const pts = ht.pts;
            if (pts.length >= 10) {
              const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
              const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
              const rmsPx = Math.sqrt(pts.reduce((s, p) => s + (p.x - mx) ** 2 + (p.y - my) ** 2, 0) / pts.length);
              // Drift: least-squares slope of x(t) and y(t), in px/s.
              const t0 = pts[0].atMs;
              const ts = pts.map((p) => (p.atMs - t0) / 1000);
              const tm = ts.reduce((s, t) => s + t, 0) / ts.length;
              const tt = ts.reduce((s, t) => s + (t - tm) ** 2, 0) || 1e-6;
              const slopeX = pts.reduce((s, p, i) => s + (ts[i] - tm) * (p.x - mx), 0) / tt;
              const slopeY = pts.reduce((s, p, i) => s + (ts[i] - tm) * (p.y - my), 0) / tt;
              const driftPxPerS = Math.hypot(slopeX, slopeY);
              setHoldResult({ rmsPx, driftPxPerS, n: pts.length });
              speak(`Hold test complete. R M S ${Math.round(rmsPx)} pixels, drift ${Math.round(driftPxPerS)} per second.`);
            }
            setHoldActive(false);
          }
        }
      }

      // --- Overlay drawing (trace + shot markers + calibration dot ring) ---
      const wrap = boardWrapRef.current;
      const overlay = overlayRef.current;
      if (wrap && overlay) {
        const rect = wrap.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(rect.height));
        if (overlay.width !== w || overlay.height !== h) {
          overlay.width = w;
          overlay.height = h;
        }
        const ctx = overlay.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, w, h);
          // Wobble trace, fading with age — dark underlay keeps it readable
          // on the white board.
          const trace = traceRef.current;
          for (let i = 1; i < trace.length; i += 1) {
            const age = (now - trace[i].atMs) / TRACE_WINDOW_MS;
            const alpha = Math.max(0, 0.95 - age);
            ctx.beginPath();
            ctx.moveTo(trace[i - 1].x - rect.left, trace[i - 1].y - rect.top);
            ctx.lineTo(trace[i].x - rect.left, trace[i].y - rect.top);
            ctx.strokeStyle = `rgba(2, 44, 66, ${alpha * 0.5})`;
            ctx.lineWidth = 5;
            ctx.stroke();
            ctx.strokeStyle = `rgba(14, 165, 233, ${alpha})`;
            ctx.lineWidth = 3;
            ctx.stroke();
          }
          // A bold reticle at a point: halo ring + colored ring + center dot
          // + crosshair ticks. Used for the live aim and the cal estimate.
          const drawReticle = (x: number, y: number, color: string, r: number) => {
            ctx.strokeStyle = "rgba(255,255,255,0.9)";
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            for (const [dx, dy] of [
              [-1, 0],
              [1, 0],
              [0, -1],
              [0, 1],
            ]) {
              ctx.moveTo(x + dx * (r + 2), y + dy * (r + 2));
              ctx.lineTo(x + dx * (r + 8), y + dy * (r + 8));
            }
            ctx.stroke();
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();
          };
          // Live reticle (training).
          const lastPt = trace[trace.length - 1];
          if (lastPt && now - lastPt.atMs < 250) {
            drawReticle(lastPt.x - rect.left, lastPt.y - rect.top, "#0284c7", 12);
          }
          // Calibrating: the CURRENT aim estimate from the dots captured so
          // far (fuchsia) — watch it converge onto where you're really aiming
          // as more dots land.
          if (phaseRef.current === "calibrate" && provisionalModelRef.current && smoothFeaturesRef.current && obs) {
            const est = predictAim(provisionalModelRef.current, smoothFeaturesRef.current);
            const ex = est.x - rect.left;
            const ey = est.y - rect.top;
            if (ex > -40 && ex < w + 40 && ey > -40 && ey < h + 40) {
              drawReticle(ex, ey, "#c026d3", 12);
              ctx.fillStyle = "#c026d3";
              ctx.font = "bold 11px ui-monospace, monospace";
              ctx.fillText(`est. aim (${samplesRef.current.length} dots)`, ex + 18, ey - 12);
            }
          }
          // Shot markers (board-normalized → element px) — halo + bold ring.
          for (const marker of shotMarkersRef.current) {
            const mx = marker.nx * w;
            const my = marker.ny * h;
            const color = marker.correct === false ? "#e11d48" : marker.correct ? "#16a34a" : "#525252";
            ctx.strokeStyle = "rgba(255,255,255,0.9)";
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.arc(mx, my, 9, 0, Math.PI * 2);
            ctx.stroke();
            ctx.strokeStyle = color;
            ctx.lineWidth = 3.5;
            ctx.beginPath();
            ctx.arc(mx, my, 9, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(mx - 4, my);
            ctx.lineTo(mx + 4, my);
            ctx.moveTo(mx, my - 4);
            ctx.lineTo(mx, my + 4);
            ctx.stroke();
          }
        }
      }

      // Camera preview — small when training, large + annotated in diagnostics.
      const preview = previewCanvasRef.current;
      if (preview) {
        const diagOn = showDiagRef.current;
        const pw = diagOn ? 460 : 240;
        const ph = Math.round((procH / procW) * pw);
        if (preview.width !== pw || preview.height !== ph) {
          preview.width = pw;
          preview.height = ph;
        }
        const pctx = preview.getContext("2d", { willReadFrequently: diagOn });
        if (pctx) {
          // While "Help computer." is picking, show the FROZEN frame, clean
          // (no tint/markers) — the user is tapping patches on a still.
          const frozen = helpPickRef.current && freezeCanvasRef.current ? freezeCanvasRef.current : null;
          pctx.drawImage(frozen ?? procCanvas, 0, 0, pw, ph);
          const s = pw / procW;
          const diagData = diagRef.current;
          if (diagOn && diagData && !frozen) {
            const { debug, roi } = diagData;
            // Tint what the detector considers signal: in checker mode,
            // everything under the dark threshold; in color mode, each pixel
            // classified as red/green/blue lights up saturated in its class.
            const img = pctx.getImageData(0, 0, pw, ph);
            const px = img.data;
            const colorMode = patternModeRef.current === "color";
            // Reference mode tints by chromaticity distance to the sampled
            // colors — the same rule the detector runs.
            const refC = refColorsRef.current;
            const refTol = refBrightTolRef.current;
            const refData = refC
              ? ([refC.red, refC.green, refC.blue] as const).map((c) => {
                  const s = Math.max(1, c[0] + c[1] + c[2]);
                  return { cx: c[0] / s, cy: c[1] / s, s };
                })
              : null;
            // Illuminant loci — the same 2700–6500 K synthetic sweep the
            // detector uses, so the tint shows exactly what it accepts.
            const refLoci = refC
              ? [buildChromaLocus(refC.red), buildChromaLocus(refC.green), buildChromaLocus(refC.blue)]
              : null;
            const classifyRef = (r: number, g: number, b: number): 0 | 1 | 2 | 3 => {
              const sum = r + g + b;
              if (sum < 30 || !refData || !refLoci) return 0;
              // Neutral competes — mirrors the detector exactly.
              const cr = r / sum;
              const cg = g / sum;
              const drN = cr - 1 / 3;
              const dgN = cg - 1 / 3;
              const distNeutral = drN * drN + dgN * dgN;
              let best: 0 | 1 | 2 | 3 = 0;
              let bestD = 0.13 * 0.13; // matches the detector's REF_THR
              for (let k = 0; k < 3; k += 1) {
                if (sum < refData[k].s * (1 - refTol) || sum > refData[k].s * (1 + refTol)) continue;
                const d = locusDist2(cr, cg, refLoci[k]);
                if (d < bestD && d < distNeutral) {
                  bestD = d;
                  best = (k + 1) as 1 | 2 | 3;
                }
              }
              return best;
            };
            for (let i = 0; i < pw * ph; i += 1) {
              const o = i * 4;
              if (colorMode) {
                const cmy = colorPaletteRef.current === "cmy";
                const cls =
                  colorMode && refData
                    ? classifyRef(px[o], px[o + 1], px[o + 2])
                    : classifyRGB(px[o], px[o + 1], px[o + 2], colorTuningRef.current, colorPaletteRef.current);
                if (cls === 1) {
                  // red slot: red (rgb) / yellow (cmy)
                  px[o] = 255;
                  px[o + 1] = cmy ? 235 : 40;
                  px[o + 2] = 40;
                } else if (cls === 2) {
                  // green slot: green (rgb) / cyan (cmy)
                  px[o] = 40;
                  px[o + 1] = cmy ? 235 : 255;
                  px[o + 2] = cmy ? 255 : 40;
                } else if (cls === 3) {
                  // blue slot: blue (rgb) / magenta (cmy)
                  px[o] = cmy ? 255 : 60;
                  px[o + 1] = cmy ? 40 : 60;
                  px[o + 2] = 255;
                }
              } else {
                const isMark =
                  patternModeRef.current === "shape"
                    ? isShapeMark(px[o], px[o + 1], px[o + 2], redSensitivityRef.current)
                    : ((px[o] * 77 + px[o + 1] * 150 + px[o + 2] * 29) >> 8) < debug.threshold;
                if (isMark) {
                  px[o] = Math.min(255, px[o] * 0.4 + 130);
                  px[o + 1] = px[o + 1] * 0.35;
                  px[o + 2] = px[o + 2] * 0.35;
                }
              }
            }
            pctx.putImageData(img, 0, 0);
            // All dark blobs (gray), tile candidates (green), diagonal (cyan),
            // dot candidates (amber), search ROI (blue).
            pctx.fillStyle = "rgba(229,231,235,0.85)";
            for (const blob of debug.blobs) pctx.fillRect(blob.x * s - 1, blob.y * s - 1, 2, 2);
            pctx.strokeStyle = "#22c55e";
            pctx.lineWidth = 1.5;
            for (const tile of debug.tileCandidates) {
              pctx.beginPath();
              pctx.arc(tile.x * s, tile.y * s, 5, 0, Math.PI * 2);
              pctx.stroke();
            }
            if (debug.quad && debug.quad.length === 4) {
              pctx.strokeStyle = "#22d3ee";
              pctx.beginPath();
              pctx.moveTo(debug.quad[0].x * s, debug.quad[0].y * s);
              for (let i = 1; i < 4; i += 1) pctx.lineTo(debug.quad[i].x * s, debug.quad[i].y * s);
              pctx.stroke();
            }
            pctx.strokeStyle = "#f59e0b";
            for (const dot of debug.dotCandidates) {
              pctx.beginPath();
              pctx.arc(dot.x * s, dot.y * s, 3.5, 0, Math.PI * 2);
              pctx.stroke();
            }
            // Where the dot SHOULD be for the best color triple (magenta ✕).
            if (debug.colorGates?.predictedDot) {
              const dx = debug.colorGates.predictedDot.x * s;
              const dy = debug.colorGates.predictedDot.y * s;
              pctx.strokeStyle = "#e879f9";
              pctx.lineWidth = 2;
              pctx.beginPath();
              pctx.moveTo(dx - 5, dy - 5);
              pctx.lineTo(dx + 5, dy + 5);
              pctx.moveTo(dx + 5, dy - 5);
              pctx.lineTo(dx - 5, dy + 5);
              pctx.stroke();
            }
            // The dot blob's bounding box — solid amber when it contains the
            // predicted dot (box-hit confirmation), dashed when it only
            // passed the centroid-distance test.
            if (debug.colorGates?.dotBox) {
              const box = debug.colorGates.dotBox;
              pctx.strokeStyle = "#f59e0b";
              pctx.lineWidth = 1.5;
              if (!debug.colorGates.dotBoxHit) pctx.setLineDash([3, 2]);
              pctx.strokeRect(box.x * s, box.y * s, box.w * s, box.h * s);
              pctx.setLineDash([]);
            }
            if (roi) {
              pctx.strokeStyle = "#3b82f6";
              pctx.setLineDash([4, 3]);
              pctx.strokeRect(roi.x * s, roi.y * s, roi.w * s, roi.h * s);
              pctx.setLineDash([]);
            }
          }
          const current = obsRef.current;
          if (current && now - current.atMs < 400) {
            pctx.fillStyle = "#22c55e";
            for (const tile of current.obs.tiles) pctx.fillRect(tile.x * s - 1.5, tile.y * s - 1.5, 3, 3);
            pctx.fillStyle = "#f59e0b";
            pctx.beginPath();
            pctx.arc(current.obs.dot.x * s, current.obs.dot.y * s, 2.5, 0, Math.PI * 2);
            pctx.fill();
          }
          // PHOTO CAPTURE ZONE: while waiting for / counting down the color
          // photo, marching black/white ants ring the center zone — the
          // only place the card will be accepted. Unmissable on any
          // background, and drawn regardless of the guide toggle.
          const pStage = photoStageRef.current;
          if ((pStage === "position" || pStage === "countdown") && !frozen) {
            const zoneR = Math.min(pw, ph) * PHOTO_ZONE_R;
            const zcx = pw / 2;
            const zcy = ph / 2;
            const dash = 9;
            const phase = (now / 25) % (dash * 2);
            pctx.lineWidth = 3.5;
            pctx.setLineDash([dash, dash]);
            pctx.strokeStyle = "#fff";
            pctx.lineDashOffset = -phase;
            pctx.beginPath();
            pctx.arc(zcx, zcy, zoneR, 0, Math.PI * 2);
            pctx.stroke();
            pctx.strokeStyle = "#000";
            pctx.lineDashOffset = -phase + dash;
            pctx.beginPath();
            pctx.arc(zcx, zcy, zoneR, 0, Math.PI * 2);
            pctx.stroke();
            pctx.setLineDash([]);
            pctx.lineDashOffset = 0;
            pctx.font = "700 11px ui-sans-serif, system-ui, sans-serif";
            pctx.textAlign = "center";
            pctx.fillStyle = photoZoneOkRef.current ? "rgba(34,197,94,0.95)" : "#fff";
            pctx.strokeStyle = "rgba(0,0,0,0.8)";
            pctx.lineWidth = 3;
            const label = photoZoneOkRef.current ? "card in the zone ✓" : "card HERE";
            pctx.strokeText(label, zcx, zcy - zoneR - 7);
            pctx.fillText(label, zcx, zcy - zoneR - 7);
            pctx.textAlign = "start";
          }
          // Ideal-position guide: a faint silhouette (head + shoulders) and a
          // sweet-spot ring where the muzzle flag should sit — center frame,
          // chest height, i.e. a raised pistol with the camera at screen
          // height. Turns green while the lock is inside the ring.
          if (showGuideRef.current && !frozen) {
            const minDim = Math.min(pw, ph);
            const cx = pw / 2;
            const zoneY = ph * 0.5;
            const zoneR = minDim * 0.16;
            const lockIn = Boolean(
              current &&
                now - current.atMs < 700 &&
                Math.hypot(current.obs.center.x * s - cx, current.obs.center.y * s - zoneY) < zoneR,
            );
            pctx.lineWidth = 1.5;
            pctx.setLineDash([5, 4]);
            // Silhouette: head + shoulder curve, always faint white.
            pctx.strokeStyle = "rgba(255,255,255,0.45)";
            pctx.beginPath();
            pctx.arc(cx, ph * 0.2, minDim * 0.1, 0, Math.PI * 2);
            pctx.stroke();
            pctx.beginPath();
            pctx.moveTo(cx - pw * 0.3, ph * 0.62);
            pctx.quadraticCurveTo(cx, ph * 0.34, cx + pw * 0.3, ph * 0.62);
            pctx.stroke();
            // Sweet-spot ring for the flag.
            pctx.strokeStyle = lockIn ? "rgba(34,197,94,0.95)" : "rgba(56,189,248,0.85)";
            pctx.lineWidth = 2;
            pctx.beginPath();
            pctx.arc(cx, zoneY, zoneR, 0, Math.PI * 2);
            pctx.stroke();
            pctx.setLineDash([]);
            pctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
            pctx.textAlign = "center";
            pctx.fillStyle = lockIn ? "rgba(34,197,94,0.95)" : "rgba(56,189,248,0.9)";
            pctx.fillText(lockIn ? "position ✓" : "flag here", cx, zoneY + zoneR + 12);
            pctx.textAlign = "start";
          }
          // ---- Detection-status layer -----------------------------------
          // Everything here is drawn ON the canvas — an overlapping layer —
          // so per-frame changes repaint pixels and never reflow the DOM
          // (no jitter from text growing/shrinking).
          if (!frozen) {
            // Marching-ants box around the DETECTED CARD.
            if (current && now - current.atMs < 700) {
              const pts = [...current.obs.tiles, current.obs.dot];
              let bx0 = Infinity;
              let by0 = Infinity;
              let bx1 = -Infinity;
              let by1 = -Infinity;
              for (const p of pts) {
                bx0 = Math.min(bx0, p.x);
                by0 = Math.min(by0, p.y);
                bx1 = Math.max(bx1, p.x);
                by1 = Math.max(by1, p.y);
              }
              const pad = current.obs.cellPx * 0.7;
              const rx0 = (bx0 - pad) * s;
              const ry0 = (by0 - pad) * s;
              const rw2 = (bx1 - bx0 + pad * 2) * s;
              const rh2 = (by1 - by0 + pad * 2) * s;
              const dash = 6;
              const antPhase = (now / 30) % (dash * 2);
              pctx.lineWidth = 2;
              pctx.setLineDash([dash, dash]);
              pctx.strokeStyle = "rgba(34,197,94,0.95)";
              pctx.lineDashOffset = -antPhase;
              pctx.strokeRect(rx0, ry0, rw2, rh2);
              pctx.strokeStyle = "rgba(0,0,0,0.85)";
              pctx.lineDashOffset = -antPhase + dash;
              pctx.strokeRect(rx0, ry0, rw2, rh2);
              pctx.setLineDash([]);
              pctx.lineDashOffset = 0;
            }
            // Verbose stage strip along the bottom: exactly what the
            // detector is doing right now, and why it isn't further along.
            const trk = trackRef.current;
            const fail = diagRef.current?.debug.failStage;
            const counts = diagRef.current?.debug.colorCounts;
            let stageText: string;
            let stageColor: string;
            if (!detectionArmed) {
              stageText = "⏸ standing by — arms when the target appears";
              stageColor = "#9ca3af";
            } else if (trk && trk.misses === 0) {
              const regional = trk.hits >= REGIONAL_AFTER_HITS && now - trk.seedAtMs >= REGIONAL_AFTER_MS;
              stageText = `● locked — ${regional ? "REGIONAL (gates open)" : "fresh (maturing)"} · ${trk.hits} frames${
                colorsCommittedRef.current ? " · colors settled ✓" : ""
              }`;
              stageColor = "#22c55e";
            } else if (trk && trk.misses < 6) {
              stageText = `◐ lost it — re-seeking ${trk.misses} · beacon steering to the ink`;
              stageColor = "#f59e0b";
            } else if (trk) {
              stageText = `◑ self-rescue ${trk.misses} — full re-verify inside the region`;
              stageColor = "#f59e0b";
            } else {
              stageText = `○ acquiring — ${fail === "ok" ? "verifying layout" : (fail ?? "scanning")}${
                counts ? ` · Y${counts.red} C${counts.green} M${counts.blue}` : ""
              }`;
              stageColor = "#38bdf8";
            }
            const strip = 15;
            pctx.fillStyle = "rgba(0,0,0,0.68)";
            pctx.fillRect(0, ph - strip, pw, strip);
            pctx.font = "600 9px ui-monospace, SFMono-Regular, monospace";
            pctx.textAlign = "left";
            pctx.fillStyle = stageColor;
            pctx.fillText(stageText, 4, ph - 4.5);
          }
        }
      }

      // Throttled status.
      frameCount += 1;
      if (now - fpsWindowStart >= 1000) {
        fps = frameCount;
        frameCount = 0;
        fpsWindowStart = now;
      }
      if (now - lastStatusPushMs > 300) {
        lastStatusPushMs = now;
        const current = obsRef.current;
        // 700 ms of hysteresis: a few marginal frames (self-rescue window)
        // shouldn't flip the LOCK badge and everything keyed off it.
        const locked = Boolean(current && now - current.atMs < 700);
        // Shooter coaching: what to physically change to get (or improve)
        // the lock. Cleared while the lock is healthy.
        setCoach(
          locked
            ? current && current.obs.cellPx < 6
              ? "Lock is marginal — a step closer to the camera will steady it a lot."
              : null
            : coachHintFor(diagRef.current?.debug ?? null, patternModeRef.current, colorPaletteRef.current),
        );
        const t = trackRef.current;
        setHLogCount(Math.min(400, homographyLogRef.current.length));
        const regionalNow = t !== null && t.hits >= REGIONAL_AFTER_HITS && now - t.seedAtMs >= REGIONAL_AFTER_MS;
        // Photo staging, part 2 — the POSITION wait. The reference snapshot
        // only arms while the staged flow is explicitly waiting for the
        // shooter ("position" stage: the brief has finished playing). A
        // momentary match while the card is still on the table or being
        // seated can never fire the photo. The lock must then be HELD:
        // ≥2 s steady AND inside a fresh tri-color prescan region, or ≥3.5 s
        // outright in a prescan-hostile scene. Only then does the spoken
        // 3-2-1 start, and the photo is taken at zero.
        // Is the card inside the marching center ring right now?
        photoZoneOkRef.current = Boolean(
          t && Math.hypot(t.center.x - procW / 2, t.center.y - procH / 2) <= Math.min(procW, procH) * PHOTO_ZONE_R,
        );
        if (
          regionalNow &&
          t &&
          !refColorsRef.current &&
          patternModeRef.current === "color" &&
          photoStageRef.current === "position"
        ) {
          // NO position requirement — the card is captured wherever it is
          // held; the ring on the preview is guidance only.
          {
            const lockAge = now - t.seedAtMs;
            let confirmed = lockAge >= 3500;
            if (!confirmed && lockAge >= 2000) {
              const img = procCtx.getImageData(0, 0, procW, procH);
              const pre = prescanCMY(img.data, procW, procH);
              const near = pre ? Math.max(pre.roi.w, pre.roi.h) * 0.75 : 0;
              confirmed = Boolean(
                pre && Math.abs(t.center.x - pre.center.x) < near && Math.abs(t.center.y - pre.center.y) < near,
              );
            }
            if (confirmed && !photoBusyRef.current) {
              startPhotoCountdownRef.current();
            }
          }
        } else if (
          !refColorsRef.current &&
          patternModeRef.current === "color" &&
          photoStageRef.current === "position" &&
          !photoBusyRef.current
        ) {
          // RELAXED arm — the lock hasn't matured to regional (or the track
          // keeps dying before it can), but detectors have kept agreeing the
          // card is inside the marching ring: ≥1.6 s of unbroken in-ring
          // evidence WITH sampled colors is a reasonable lock. Use that area
          // as the card and start the photo — the capture itself can finish
          // from the evidence colors even if the track slips again.
          const z = zoneSeenRef.current;
          if (z.colors && now - z.lastMs < 900 && z.lastMs - z.streakStartMs >= 1600) {
            startPhotoCountdownRef.current();
          }
        }
        setTracker({
          locked,
          residual: locked && current ? current.obs.residual : 0,
          cellPx: locked && current ? current.obs.cellPx : 0,
          fps,
          tracking: t !== null,
          misses: t?.misses ?? 0,
          regional: regionalNow,
        });
        if (showDiagRef.current) {
          const diagData = diagRef.current;
          setDiag(
            diagData
              ? {
                  stage: diagData.debug.failStage,
                  threshold: diagData.debug.threshold,
                  blobCount: diagData.debug.blobCount,
                  clusterCount: diagData.debug.clusterCount,
                  tileCount: diagData.debug.tileCandidates.length,
                  quadScore: diagData.debug.quadScore,
                  dotCount: diagData.debug.dotCandidates.length,
                  cellPx: diagData.debug.cellPx,
                  residual: diagData.debug.residual,
                  procW,
                  procH,
                  roiActive: diagData.roi !== null,
                  colorCounts: diagData.debug.colorCounts,
                  colorGates: diagData.debug.colorGates,
                }
              : { stage: "none", threshold: 0, blobCount: 0, clusterCount: 0, tileCount: 0, quadScore: null, dotCount: 0, cellPx: null, residual: null, procW, procH, roiActive: false, colorCounts: null, colorGates: null },
          );
        }
      }

      raf = requestAnimationFrame(processFrame);
    };

    raf = requestAnimationFrame(processFrame);
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
     
  }, [cameraActive]);

  // ---- Shot handling (mic click) --------------------------------------------
  const handleShot = (clickAtMs = performance.now()) => {
    if (phaseRef.current !== "train" || !modelRef.current) return;
    const wrap = boardWrapRef.current;
    if (!wrap) return;
    // Sample aim from the raw observations JUST BEFORE the click: the
    // striker fall disturbs the muzzle and the filter lags, so the frames
    // in [-180, -20] ms are where the sights were honest.
    const rawWindow = aimHistoryRef.current.filter(
      (hArr) => hArr.atMs >= clickAtMs - 180 && hArr.atMs <= clickAtMs - 20,
    );
    // Sample only frames in the SAME feature regime the model was fitted on
    // (homography vs affine) — mixing regimes biases the prediction.
    const wantH = modelRegimeRef.current === "h";
    const regimeWindow = rawWindow.filter((hArr) => (hArr.features[7] !== 0 || hArr.features[8] !== 0) === wantH);
    const windowFeats = regimeWindow.length >= 2 ? regimeWindow : rawWindow;
    let feats = smoothFeaturesRef.current;
    if (windowFeats.length >= 2) {
      feats = windowFeats[0].features.map(
        (_, i) => windowFeats.reduce((s, hArr) => s + hArr.features[i], 0) / windowFeats.length,
      );
    }
    if (!feats) return;
    const aim = predictAim(modelRef.current, feats);
    const rect = wrap.getBoundingClientRect();
    const nx = (aim.x - rect.left) / Math.max(1, rect.width);
    const ny = (aim.y - rect.top) / Math.max(1, rect.height);
    const onBoard = nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1;

    // ---- Open-target mode: record the shot in target-frame units and let
    // the group statistics do the talking.
    if (trainModeRef.current === "target") {
      // Scoring radius follows the simulated range — shots are recorded in
      // target radii of the SCALED face, so all stats sharpen with distance.
      const rPx = 0.45 * Math.min(rect.width, rect.height) * targetScale;
      const tx = ((nx - 0.5) * rect.width) / rPx;
      const ty = ((ny - 0.5) * rect.height) / rPx;
      const atMs = performance.now();
      shotCountRef.current += 1;
      shotMarkersRef.current.push({ nx, ny, correct: Math.hypot(tx, ty) <= 1, atMs });
      if (shotMarkersRef.current.length > 40) shotMarkersRef.current.shift();
      setTargetShots((prev) => [...prev, { x: tx, y: ty, atMs }]);
      setSession((s) => ({ ...s, shots: s.shots + 1 }));
      // Lifetime log: the bullseye center IS the intended point, and tx/ty
      // are already the deviation in target radii.
      appendShot({ t: Date.now(), src: "target", label: "bullseye", dx: tx, dy: ty, hit: Math.hypot(tx, ty) <= 1 });
      return;
    }

    const zoneId = onBoard ? zoneAt(nx, ny, rect.width / Math.max(1, rect.height)) : null;

    const state = drillRef.current;
    // In timed mode a shot before the call has been announced is free
    // practice, not an answer to a question the shooter hasn't heard.
    const awaitingCall =
      drillModeRef.current === "timed" && state !== null && state.index > announcedIndexRef.current;
    const expected = state && !awaitingCall ? currentStep(state) : null;
    const correct = expected ? zoneId === expected.zoneId : null;

    shotCountRef.current += 1;
    const record: ShotRecord = {
      n: shotCountRef.current,
      zoneId,
      expectedZoneId: expected?.zoneId ?? null,
      correct,
      atMs: performance.now(),
    };
    shotMarkersRef.current.push({ nx, ny, correct, atMs: record.atMs });
    if (shotMarkersRef.current.length > 40) shotMarkersRef.current.shift();
    setShots((prev) => [record, ...prev].slice(0, 12));

    // Lifetime log: the called zone's center is where the bullet SHOULD be.
    // Deviation in zone radii, isotropic (y corrected by the board aspect,
    // same convention as zoneAt).
    if (expected) {
      const expectedZone = zones.find((z) => z.id === expected.zoneId);
      if (expectedZone && expectedZone.radius > 0) {
        const aspect = rect.width / Math.max(1, rect.height);
        appendShot({
          t: Date.now(),
          src: "drill",
          label: expectedZone.color,
          dx: (nx - expectedZone.cx) / expectedZone.radius,
          dy: (ny - expectedZone.cy) / aspect / expectedZone.radius,
          hit: correct === true,
        });
      }
    }

    if (zoneId) {
      setFeedback({ [zoneId]: correct === false ? "wrong" : "correct" });
      if (feedbackTimerRef.current) window.clearTimeout(feedbackTimerRef.current);
      feedbackTimerRef.current = window.setTimeout(() => setFeedback({}), 450);
    }

    // Session tally: every trigger break counts; called shots also score.
    const callDriven = drillModeRef.current !== "sequence";
    const reactionMs =
      expected && callDriven && lastCallAtRef.current > 0
        ? Math.max(0, performance.now() - lastCallAtRef.current)
        : 0;
    setSession((s) => {
      if (!expected) return { ...s, shots: s.shots + 1 };
      const hit = correct === true;
      const streak = hit ? s.streak + 1 : 0;
      const goodReaction = reactionMs > 0 && reactionMs < 30000;
      return {
        ...s,
        shots: s.shots + 1,
        called: s.called + 1,
        hits: s.hits + (hit ? 1 : 0),
        misses: s.misses + (hit ? 0 : 1),
        streak,
        bestStreak: Math.max(s.bestStreak, streak),
        reactionSumMs: s.reactionSumMs + (goodReaction ? reactionMs : 0),
        reactionCount: s.reactionCount + (goodReaction ? 1 : 0),
      };
    });

    if (state && expected) {
      // NOTE: online recalibration from drill hits was REMOVED. A "confirmed
      // hit" only proves the shot landed somewhere in a zone a third of the
      // screen wide — feeding it back as "aimed at the zone CENTER" let up
      // to 18 fuzzy samples outvote the 9 precise dots and visibly drift
      // the aim over a session. The dots are the only calibration truth.
      const next = registerHit(state, zoneId, reactionMs);
      drillRef.current = next;
      setDrill(next);
      if (next.status === "done") {
        setTimedRun(null);
        setCurrentCall(null);
        setSession((s) => ({ ...s, drills: s.drills + 1 }));
        const score = scoreDrill(next);
        speak(`Done. ${score.correct} of ${score.total}.`);
      } else if (drillModeRef.current === "reactive") {
        // One-at-a-time: the hit earns the next call. Timed and sequence
        // modes don't announce here — timers or memory drive them.
        announcedIndexRef.current = next.index;
        const step = currentStep(next);
        if (step) {
          lastCallAtRef.current = performance.now();
          setCurrentCall({ label: step.label, kind: step.kind });
          speak(step.spoken, { onEnd: () => (lastCallAtRef.current = performance.now()) });
        }
      }
    }
  };
  const handleShotRef = useRef(handleShot);
  useEffect(() => {
    handleShotRef.current = handleShot;
  });

  // Route mic clicks: calibration samples while calibrating, shots otherwise.
  // SLIDE calibration is WINDOWED, not free-running: each prompt opens a
  // 2-second listen window — whatever lands inside IS the slide (both rack
  // impacts), and the "wait" gaps between windows are deaf. Three windows,
  // then the averaged slide fingerprint screens the dry-fires and all
  // future detection.
  const onMicClick = (atMs: number, peak: number, fingerprint: number[] | null = null) => {
    const cal = trigCalRef.current;
    if (cal) {
      // NEVER count our own voice prompts: the mic hears the speakers, and
      // a TTS syllable is a spike like any other.
      if (isSpeaking()) return;
      const nowMs = performance.now();
      if (cal.stage === "slides") {
        // Deaf outside the open window — by design.
        if (cal.windowCloseAt === null || nowMs > cal.windowCloseAt) return;
        if (fingerprint) cal.rackFps.push(fingerprint);
        cal.rackOk = true;
        trigWaveEventsRef.current.push({ t: nowMs, kind: "reject" }); // amber = slide
        return;
      }
      // ---- Dry-fire stage ----
      if (nowMs < cal.armAt) return; // TTS prompt guard
      const rackTpl = meanFingerprint(cal.rackFps);
      const rackSim = fingerprint && rackTpl ? fingerprintSimilarity(fingerprint, rackTpl) : null;
      if (rackSim !== null && rackSim > 0.92) {
        // Sounds like the STORED SLIDE — ignore it, it isn't a trigger pull.
        trigWaveEventsRef.current.push({ t: nowMs, kind: "reject" });
        return;
      }
      trigWaveEventsRef.current.push({ t: nowMs, kind: "click" });
      cal.peaks.push(peak);
      if (fingerprint) cal.clickFps.push(fingerprint);
      setTrigCalCount(cal.peaks.length);
      if (cal.peaks.length < TRIG_CAL_CLICKS) {
        cal.armAt = nowMs + 8000; // fallback — onEnd arms sooner
        cal.remindAt = nowMs + 15000;
        speak(`${cal.peaks.length}.`, {
          onEnd: () => {
            const c = trigCalRef.current;
            if (c) {
              c.armAt = performance.now() + 250;
              c.remindAt = performance.now() + 15000;
            }
          },
        });
      } else {
          const sorted = [...cal.peaks].sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          // Fire comfortably below the measured click, safely above silence.
          const floor = Math.min(0.4, Math.max(0.005, median * 0.4));
          triggerRef.current?.setCustomFloor(floor);
          // Average the three clicks and the three racks into the accept /
          // reject templates, and arm spectral matching from here on.
          const clickFp = meanFingerprint(cal.clickFps);
          const rackFp = meanFingerprint(cal.rackFps);
          triggerRef.current?.setFingerprints(clickFp, rackFp);
          triggerRef.current?.setCaptureMode(false); // back to live discrimination
          const rackOk = cal.rackOk ?? null;
          trigCalRef.current = null;
          setTrigCalCount(null);
          setTrigCalResult({ clickPeak: median, floor, rackOk, clickFp, rackFp });
          speak("Trigger calibrated.");
          // Calibration chain: flow straight into the 9-dot aim calibration.
          if (pendingAimCalRef.current) {
            pendingAimCalRef.current = false;
            window.setTimeout(() => beginCalibrationRef.current(), 1600);
          }
        }
      return;
    }
    handleShotRef.current(atMs);
  };
  const onMicClickRef = useRef(onMicClick);
  useEffect(() => {
    onMicClickRef.current = onMicClick;
  });

  const startMic = async () => {
    const trigger = createClickTrigger((atMs, peak, fp) => onMicClickRef.current(atMs, peak, fp), trigSensitivity);
    const ok = await trigger.start();
    if (!ok) {
      setStatusLine("Microphone unavailable — allow mic access, or use the manual shot button.");
      return;
    }
    triggerRef.current?.stop();
    triggerRef.current = trigger;
    // Re-apply a previously calibrated click floor + fingerprints.
    if (trigCalResult) {
      trigger.setCustomFloor(trigCalResult.floor);
      trigger.setFingerprints(trigCalResult.clickFp ?? null, trigCalResult.rackFp ?? null);
    }
    setMicActive(true);
  };
  // Manual fallback: freeze the (still-aimed) frame and have the human tap
  // the patches in order. Sets the reference colors directly.
  const beginHelpTap = () => {
    if (captureFrameRef.current?.()) {
      setHelpPick({ stage: 0 });
      speak(
        `Frame captured. Set the weapon down safely, then tap the ${helpOrder[0].name.toLowerCase()} patch on the picture.`,
      );
    } else {
      speak("Camera not ready.");
    }
  };
  // "Help computer." — BACKGROUND SUBTRACTION first: photograph the scene
  // WITHOUT the gun, then WITH it raised. The per-pixel difference isolates
  // exactly what appeared (the gun and its card); unchanged pixels are
  // blanked and the rank-based detector runs on the changed region alone —
  // the background physically cannot offer it a false candidate. Falls back
  // to guided patch-tapping when the diff or the detector comes up empty.
  const startHelpComputer = () => {
    if (helpPick) {
      setHelpPick(null);
      return;
    }
    speak("Help mode. Lower the pistol out of the camera's view. Background photo in three. Two. One.", {
      onEnd: () => {
        const bg = captureProcImageRef.current?.();
        if (!bg) {
          speak("Camera not ready.");
          return;
        }
        bgFrameRef.current = bg;
        speak("Got it. Now raise your pistol and hold your normal aim. Three. Two. One.", {
          onEnd: () => {
            const fg = captureProcImageRef.current?.();
            const bg2 = bgFrameRef.current;
            bgFrameRef.current = null;
            if (!fg || !bg2 || fg.data.length !== bg2.data.length) {
              beginHelpTap();
              return;
            }
            // Diff → bounding box of what CHANGED between the shots.
            const w = fg.width;
            const h = fg.height;
            let minX = w;
            let minY = h;
            let maxX = -1;
            let maxY = -1;
            for (let y = 0; y < h; y += 2) {
              for (let x = 0; x < w; x += 2) {
                const o = (y * w + x) * 4;
                const d =
                  Math.abs(fg.data[o] - bg2.data[o]) +
                  Math.abs(fg.data[o + 1] - bg2.data[o + 1]) +
                  Math.abs(fg.data[o + 2] - bg2.data[o + 2]);
                if (d > 60) {
                  if (x < minX) minX = x;
                  if (x > maxX) maxX = x;
                  if (y < minY) minY = y;
                  if (y > maxY) maxY = y;
                }
              }
            }
            if (maxX < 0 || maxX - minX < 8 || maxY - minY < 8) {
              speak("I couldn't see a difference between the two photos. Let's do it by touch.");
              beginHelpTap();
              return;
            }
            // Crop with margin, BLANKING unchanged pixels — only what
            // appeared with the gun is visible to the detector.
            const pad = 12;
            const cx0 = Math.max(0, minX - pad);
            const cy0 = Math.max(0, minY - pad);
            const cw = Math.min(w - cx0, maxX - minX + 1 + pad * 2);
            const ch = Math.min(h - cy0, maxY - minY + 1 + pad * 2);
            const sub = new Uint8ClampedArray(cw * ch * 4);
            for (let y = 0; y < ch; y += 1) {
              for (let x = 0; x < cw; x += 1) {
                const so = ((cy0 + y) * w + (cx0 + x)) * 4;
                const d =
                  Math.abs(fg.data[so] - bg2.data[so]) +
                  Math.abs(fg.data[so + 1] - bg2.data[so + 1]) +
                  Math.abs(fg.data[so + 2] - bg2.data[so + 2]);
                const to = (y * cw + x) * 4;
                if (d > 45) {
                  sub[to] = fg.data[so];
                  sub[to + 1] = fg.data[so + 1];
                  sub[to + 2] = fg.data[so + 2];
                }
                sub[to + 3] = 255;
              }
            }
            const res = analyzeColorFlagLoose(sub, cw, ch, w, cx0, cy0);
            if (res.observation?.quadColors) {
              const qc = res.observation.quadColors;
              setRefDraft({ red: [...qc.red], green: [...qc.green], blue: [...qc.blue] });
              speak("Found the card. Colors captured — carry on.");
              return;
            }
            speak("I couldn't isolate the card automatically. Let's do it by touch.");
            beginHelpTap();
          },
        });
      },
    });
  };

  // Opens one 2-second slide listen window (called from prompt onEnd, so
  // the window can never overlap our own narration).
  const openSlideWindow = () => {
    const c = trigCalRef.current;
    if (!c || c.stage !== "slides") return;
    c.windowStartCount = c.rackFps.length;
    c.windowCloseAt = performance.now() + 2000;
    c.remindAt = performance.now() + 12000;
    setSlideWindowOpen(true);
  };
  const openSlideWindowRef = useRef(openSlideWindow);
  useEffect(() => {
    openSlideWindowRef.current = openSlideWindow;
  });

  // Guided trigger calibration: the SLIDE comes first — three prompted
  // 2-second windows (rack, wait, rack, wait, rack) build the slide
  // fingerprint; then three dry-fires build the click fingerprint, screened
  // against the slide's.
  const beginTrigCalibration = async () => {
    if (!micActive) await startMic();
    trigCalRef.current = {
      stage: "slides",
      slidesDone: 0,
      windowCloseAt: null,
      windowStartCount: 0,
      peaks: [],
      clickFps: [],
      rackFps: [],
      armAt: performance.now() + 8000,
      remindAt: performance.now() + 12000,
    };
    setTrigCalCount(0);
    setTrigCalRacking(true);
    setSlideWindowOpen(false);
    setTrigCalResult(null);
    triggerRef.current?.setCustomFloor(null); // measure on preset gates
    triggerRef.current?.setFingerprints(null, null); // and without matching
    triggerRef.current?.setCaptureMode(true); // raw spikes, no discrimination
    speak("Calibrating the slide. Rack the slide now.", { onEnd: openSlideWindow });
  };
  const cancelTrigCalibration = () => {
    trigCalRef.current = null;
    // eslint-disable-next-line react-hooks/immutability -- latest-value ref, mutable by design
    pendingAimCalRef.current = false;
    setTrigCalCount(null);
    setTrigCalRacking(false);
    setSlideWindowOpen(false);
    triggerRef.current?.setCaptureMode(false);
    if (trigCalResult) triggerRef.current?.setCustomFloor(trigCalResult.floor);
  };
  // Restart the slide & trigger calibration from the top WITHOUT losing the
  // chain (a pending 9-dot aim calibration stays pending).
  const restartTrigCalibration = () => {
    const pending = pendingAimCalRef.current;
    cancelTrigCalibration();
    // eslint-disable-next-line react-hooks/immutability -- latest-value ref, mutable by design
    pendingAimCalRef.current = pending;
    void beginTrigCalibration();
  };

  // ---- Color-photo countdown -------------------------------------------------
  // Both the automatic capture (confirmed lock) and the manual button run
  // through here: announce, count 3-2-1 out loud and on screen, THEN take
  // the reference photo from the live track.
  const cancelPhotoCountdown = () => {
    for (const id of photoTimersRef.current) window.clearTimeout(id);
    photoTimersRef.current = [];
    photoBusyRef.current = false;
    setPhotoCount(null);
  };
  const startPhotoCountdown = () => {
    if (photoBusyRef.current) return;
    photoBusyRef.current = true;
    setPhotoStage("countdown");
    setPhotoCount(3);
    speak("Hold still for the photo. Three.");
    photoTimersRef.current = [
      window.setTimeout(() => {
        setPhotoCount(2);
        speak("Two.");
      }, 1200),
      window.setTimeout(() => {
        setPhotoCount(1);
        speak("One.");
      }, 2400),
      window.setTimeout(() => {
        setPhotoCount(null);
        photoBusyRef.current = false;
        photoTimersRef.current = [];
        const t = trackRef.current;
        const z = zoneSeenRef.current;
        // Fallback: the formal track slipped at zero, but a detector saw the
        // card inside the ring within the last second and brought its
        // quadrant colors along — that area IS the card; capture from it.
        const zoneFresh = z.colors && performance.now() - z.lastMs < 1200;
        // NO position requirement: capture from the live track wherever the
        // card is held. A manual photo OVERRIDES any 15 s color commit —
        // the human's explicit capture outranks the automatic one.
        if (t) {
          const cap: TrackedColors = {
            red: [...t.colors.red] as [number, number, number],
            green: [...t.colors.green] as [number, number, number],
            blue: [...t.colors.blue] as [number, number, number],
          };
          colorsCommittedRef.current = cap;
          setRefDraft({ red: cap.red, green: cap.green, blue: cap.blue });
          setPhotoStage("done");
          speak("Colors captured.");
        } else if (zoneFresh && z.colors) {
          const cap: TrackedColors = {
            red: [...z.colors.red] as [number, number, number],
            green: [...z.colors.green] as [number, number, number],
            blue: [...z.colors.blue] as [number, number, number],
          };
          colorsCommittedRef.current = cap;
          setRefDraft({ red: cap.red, green: cap.green, blue: cap.blue });
          setPhotoStage("done");
          speak("Colors captured.");
        } else {
          setPhotoStage("idle"); // manual-only now — no automatic retry
          speak("Lock slipped. Tap the photo button to retry.");
        }
      }, 3600),
    ];
  };
  useEffect(() => {
    startPhotoCountdownRef.current = startPhotoCountdown;
  });

  // Restart the flag lock from scratch: drop the track, clear the color
  // reference, and re-acquire — the step's do-over button.
  const restartFlagLock = () => {
    cancelPhotoCountdown();
    trackRef.current = null;
    colorsCommittedRef.current = null; // explicit do-over — colors unsettled
    setRefDraft({ red: null, green: null, blue: null });
    setPhotoStage("idle");
    speak("Restarting. Raise and hold your aim.");
  };

  // ---- Default-session automation (user mode) -------------------------------
  // NOTE: "Help computer." no longer auto-engages — the guided flow sticks
  // with default detection (no reference colors). The manual button on the
  // camera window still works.
  // ONE-SHOT CALIBRATION CHAIN: once the training path is chosen, the
  //    green button runs slide → trigger → 9-dot aim as a single flow.
  //    pendingAimCalRef bridges the trigger calibration's completion into
  //    beginCalibration.
  const pendingAimCalRef = useRef(false);
  const beginCalibrationRef = useRef<() => void>(() => {});
  const startFullCalibration = () => {
    if (trigCalResult) {
      // Slide & trigger already calibrated (profile / earlier run) — go
      // straight to the aim dots.
      beginCalibrationRef.current();
      return;
    }
    // eslint-disable-next-line react-hooks/immutability -- latest-value ref, mutable by design
    pendingAimCalRef.current = true;
    void beginTrigCalibration();
  };
  const startFullCalibrationRef = useRef(startFullCalibration);
  useEffect(() => {
    startFullCalibrationRef.current = startFullCalibration;
  });

  const stopMic = () => {
    triggerRef.current?.stop();
    triggerRef.current = null;
    setMicActive(false);
  };

  // Mic level meter (train phase only).
  useEffect(() => {
    if (!micActive) return;
    const timer = window.setInterval(() => {
      const level = triggerRef.current?.getLevel();
      if (level) {
        // Rack phase (runs FIRST): success when the discriminator REJECTS
        // the rack — which also proves the chamber-clearing action happened.
        // Then the click phase begins.
        const cal = trigCalRef.current;
        const nowP = performance.now();
        if (cal && cal.stage === "slides") {
          if (cal.windowCloseAt !== null && nowP > cal.windowCloseAt) {
            // Window closed — tally it: did the slide land inside?
            cal.windowCloseAt = null;
            setSlideWindowOpen(false);
            const got = cal.rackFps.length > cal.windowStartCount;
            if (got) {
              cal.slidesDone += 1;
              setTrigCalCount(cal.slidesDone); // slide progress for the UI
            }
            if (cal.slidesDone >= TRIG_CAL_CLICKS) {
              // Slide fingerprint complete → the dry-fire stage.
              cal.stage = "clicks";
              cal.armAt = nowP + 8000; // fallback — onEnd arms sooner
              cal.remindAt = nowP + 15000;
              setTrigCalRacking(false);
              setTrigCalCount(0);
              speak("Slide captured. Now dry fire three times, pausing between clicks.", {
                onEnd: () => {
                  const c = trigCalRef.current;
                  if (c) {
                    c.armAt = performance.now() + 250;
                    c.remindAt = performance.now() + 15000;
                  }
                },
              });
            } else {
              speak(got ? "Good. And again — rack the slide." : "I didn't hear it. Rack the slide.", {
                onEnd: () => openSlideWindowRef.current(),
              });
            }
          } else if (cal.windowCloseAt === null && nowP > cal.remindAt && !isSpeaking()) {
            // The prompt's onEnd never opened a window (TTS hiccup) — recover.
            cal.remindAt = nowP + 12000;
            speak("Rack the slide.", { onEnd: () => openSlideWindowRef.current() });
          }
        } else if (cal && nowP > cal.remindAt && !isSpeaking()) {
          cal.remindAt = nowP + 15000;
          speak("Dry fire once.");
        }
        const thr = Math.max(1e-4, level.threshold);
        setMicStats({
          pct: Math.min(100, Math.round((level.rms / thr) * 100)),
          peakPct: Math.min(100, Math.round((level.peakHold / thr) * 100)),
          peak: level.peakHold,
          ambient: level.ambient,
          threshold: level.threshold,
          suppressed: level.suppressed,
          rejectedAgoMs: level.rejectedAtMs > 0 ? performance.now() - level.rejectedAtMs : Infinity,
          sim: level.clickSim,
        });
      }
    }, 150);
    return () => window.clearInterval(timer);
  }, [micActive]);

  // Trigger-calibration waveform: fast (45 ms) sampling of the mic level
  // into a rolling 8 s window, rendered live — sound bars, the trigger gate
  // (dashed red), room ambient (gray), suppressed spans (amber wash), and
  // event markers (green = accepted click, amber = rejected as rack/noise).
  const trigCalActive = trigCalRacking || trigCalCount !== null;
  useEffect(() => {
    if (!micActive || !trigCalActive) return;
    micWaveRef.current = [];
    trigWaveEventsRef.current = [];
    const WINDOW_MS = 8000;
    const timer = window.setInterval(() => {
      const level = triggerRef.current?.getLevel();
      const canvas = waveCanvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!level || !canvas || !ctx) return;
      const now = performance.now();
      const buf = micWaveRef.current;
      buf.push({ t: now, peak: level.rms, thr: level.threshold, ambient: level.ambient, suppressed: level.suppressed });
      while (buf.length > 0 && now - buf[0].t > WINDOW_MS) buf.shift();
      if (level.rejectedAtMs > lastRejectSeenRef.current) {
        lastRejectSeenRef.current = level.rejectedAtMs;
        trigWaveEventsRef.current.push({ t: now, kind: "reject" });
      }
      const evs = trigWaveEventsRef.current;
      while (evs.length > 0 && now - evs[0].t > WINDOW_MS) evs.shift();

      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, w, h);
      const xOf = (t: number) => w - ((now - t) / WINDOW_MS) * w;
      // sqrt scale so faint clicks are visible next to loud racks.
      let maxV = level.threshold * 2.5;
      for (const s of buf) maxV = Math.max(maxV, s.peak);
      const yOf = (v: number) => h - 2 - Math.sqrt(Math.min(1, v / maxV)) * (h - 6);
      // Suppressed spans (rack/noise hold-off) as an amber wash.
      ctx.fillStyle = "rgba(245,158,11,0.13)";
      for (let i = 0; i < buf.length; i += 1) {
        if (!buf[i].suppressed) continue;
        const x0 = xOf(buf[i].t);
        const x1 = i + 1 < buf.length ? xOf(buf[i + 1].t) : w;
        ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
      }
      // Sound bars.
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const s of buf) {
        const x = xOf(s.t);
        ctx.moveTo(x, h - 2);
        ctx.lineTo(x, yOf(s.peak));
      }
      ctx.stroke();
      // Ambient (rolling room noise).
      ctx.strokeStyle = "rgba(148,163,184,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      buf.forEach((s, i) => {
        const x = xOf(s.t);
        const y = yOf(s.ambient);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // Trigger gate.
      ctx.strokeStyle = "rgba(244,63,94,0.9)";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      buf.forEach((s, i) => {
        const x = xOf(s.t);
        const y = yOf(s.thr);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      // Event markers.
      for (const e of evs) {
        const x = xOf(e.t);
        ctx.strokeStyle = e.kind === "click" ? "#22c55e" : "#f59e0b";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
    }, 45);
    return () => window.clearInterval(timer);
  }, [micActive, trigCalActive]);

  // ---- Phase controls ---------------------------------------------------------
  const beginCalibration = () => {
    cancelSpeech();
    samplesRef.current = [];
    dwellRef.current = [];
    calIndexRef.current = 0;
    calCooldownUntilRef.current = 0;
    modelRef.current = null;
    provisionalModelRef.current = null;
    holdTestRef.current = null;
    setHoldActive(false);
    setHoldResult(null);
    featureFilterRef.current.reset();
    setModel(null);
    setCalIndex(0);
    setCalHold(0);
    setResizedSinceCal(false);
    setPhase("calibrate");
    setStatusLine("Safety brief… then aim at the amber dot and hold steady — it captures automatically.");
    // Fullscreen landscape: the board becomes the entire screen, so the
    // targets are the only thing showing — and calibration happens in the
    // exact geometry training will use.
    if (typeof document !== "undefined" && !document.fullscreenElement) {
      void document.documentElement
        .requestFullscreen?.()
        .then(() => {
          const orientation = screen.orientation as ScreenOrientation & {
            lock?: (mode: string) => Promise<void>;
          };
          orientation.lock?.("landscape").catch(() => undefined);
        })
        .catch(() => undefined);
    }
    // Arm the shot trigger automatically so calibration flows into a fully
    // hands-free drill (the mic permission prompt only appears once).
    if (!micActive) void startMic();
    // Safety announcement first — the weapons-handling rules, every time.
    setBriefActive(true);
    speak(
      "Please ensure your weapon is unloaded and that there is no ammunition in the room. " +
        "Treat every weapon as if it were loaded. " +
        "Never point a weapon at anything you do not intend to shoot. " +
        "Keep your finger straight and off the trigger until you are ready to fire. " +
        "Keep your weapon on safe until you intend to fire. " +
        "Know your target and what lies beyond. " +
        "Hold your sights on each flashing dot until its ring fills. " +
        "Three. Two. One. First dot.",
      { onEnd: () => setBriefActive(false) },
    );
  };

  // Prep-step auto-chain: the shooter raises the gun ONCE. When the flag
  // locks and (color mode) the reference colors have been auto-captured at
  // REGIONAL lock, the full calibration chain starts by itself — slide →
  // trigger → 9-dot aim — with no further screen taps.
  // NOTE: depend on derived BOOLEANS, not the tracker object — the object is
  // recreated every status push, which would endlessly reset the timer. The
  // started guard is set when the timer FIRES (not when armed), so a lock
  // flicker that cancels the pending timer re-arms cleanly.
  const camLocked = (tracker.tracking && tracker.misses === 0) || tracker.locked;
  const refComplete = Boolean(refDraft.red && refDraft.green && refDraft.blue);
  // No color-photo calibration in the guided flow — detection runs on the
  // default gates (with the rank-based loose path behind them). Leaving the
  // prep step still resets the manual-photo staging.
  useEffect(() => {
    if (wizardStep === "prep") return;
    const t = window.setTimeout(() => setPhotoStage("idle"), 0);
    return () => window.clearTimeout(t);
  }, [wizardStep]);
  // NO auto-advance on the prep step: the chain (alternating slide/trigger
  // calibration → 9-dot aim calibration) starts only when the user taps
  // Start — they control when the gun comes up.

  // ---- Scenario profiles ----------------------------------------------------
  const [profiles, setProfiles] = useState<ScenarioProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState("");
  const [profileName, setProfileName] = useState("");

  const applyProfile = (p: ScenarioProfile) => {
    setPatternMode(p.patternMode === "checker" ? "color" : p.patternMode); // checker archived
    setTrainMode(p.trainMode);
    setColorSensitivity(p.colorSensitivity);
    setColorTuning(p.colorTuning);
    setRefDraft(p.refColors);
    setRefBrightTol(p.refBrightTol);
    setRedSensitivity(p.redSensitivity);
    setShapeReq(p.shapeReq);
    setTrigSensitivity(p.trigSensitivity);
    setTrigCalResult(p.trigCal);
    setDrillMode(p.drillMode);
    setDrillLen(p.drillLen);
    setTimespanSec(p.timespanSec);
    setProtocol(p.protocol);
    triggerRef.current?.setSensitivity(p.trigSensitivity);
    triggerRef.current?.setCustomFloor(p.trigCal?.floor ?? null);
    triggerRef.current?.setFingerprints(p.trigCal?.clickFp ?? null, p.trigCal?.rackFp ?? null);
  };

  const saveProfile = () => {
    const name = (profileName.trim() || activeProfile || "Default").slice(0, 40);
    const profile: ScenarioProfile = {
      name,
      savedAt: Date.now(),
      patternMode,
      trainMode,
      colorSensitivity,
      colorTuning,
      refColors: refDraft,
      refBrightTol,
      redSensitivity,
      shapeReq,
      trigSensitivity,
      trigCal: trigCalResult,
      drillMode,
      drillLen,
      timespanSec,
      protocol,
    };
    setProfiles((list) => {
      const next = [...list.filter((x) => x.name !== name), profile].sort((a, b) => a.name.localeCompare(b.name));
      persistProfiles(next);
      return next;
    });
    setActiveProfile(name);
    setProfileName("");
    try {
      localStorage.setItem(LAST_PROFILE_KEY, name);
    } catch {
      /* best effort */
    }
  };

  const deleteProfile = () => {
    if (!activeProfile) return;
    setProfiles((list) => {
      const next = list.filter((x) => x.name !== activeProfile);
      persistProfiles(next);
      return next;
    });
    setActiveProfile("");
    try {
      localStorage.removeItem(LAST_PROFILE_KEY);
    } catch {
      /* best effort */
    }
  };

  // Load the saved profile LIST for the picker — but NEVER auto-apply one.
  // Every session starts on the defaults; a profile's settings only take
  // effect when the user explicitly selects it.
  useEffect(() => {
    const t = window.setTimeout(() => {
      setProfiles(loadProfilesFromStorage());
    }, 0);
    return () => window.clearTimeout(t);
  }, []);

  // Nudge the reference colors for a room that got brighter (☀) or darker
  // (🌙) since they were captured.
  const scaleRefColors = (f: number) =>
    setRefDraft((d) => ({
      red: d.red ? (d.red.map((v) => Math.max(0, Math.min(255, v * f))) as [number, number, number]) : null,
      green: d.green ? (d.green.map((v) => Math.max(0, Math.min(255, v * f))) as [number, number, number]) : null,
      blue: d.blue ? (d.blue.map((v) => Math.max(0, Math.min(255, v * f))) as [number, number, number]) : null,
    }));

  const skipBrief = () => {
    cancelSpeech();
    setBriefActive(false);
    setStatusLine("Aim at the amber dot and hold steady — it captures automatically.");
    speak("Aim at the first dot and hold.");
  };

  // "Wait — I messed up.": undo the PREVIOUS calibration dot (jerked while
  // it captured, wrong dot, sneeze) and redo it.
  const undoLastCalDot = () => {
    if (samplesRef.current.length === 0 || calIndexRef.current === 0) return;
    samplesRef.current.pop();
    const prev = Math.max(0, calIndexRef.current - 1);
    calIndexRef.current = prev;
    setCalIndex(prev);
    dwellRef.current = [];
    setCalHold(0);
    calCooldownUntilRef.current = performance.now() + 800;
    provisionalModelRef.current = fitAimModel(samplesRef.current, 1e-3, 3);
    speak(`Redoing dot ${prev + 1}. Aim and hold.`);
  };

  // "Wait — I messed up." during target practice: strike the last shot (a
  // false trigger from noise, or a shot you called before the gun settled).
  const undoLastShot = () => {
    if (targetShots.length === 0) return;
    setTargetShots((prev) => prev.slice(0, -1));
    shotMarkersRef.current.pop();
    setSession((s) => ({ ...s, shots: Math.max(0, s.shots - 1) }));
    speak("Last shot removed.");
  };
  useEffect(() => {
    // eslint-disable-next-line react-hooks/immutability -- latest-value ref, mutable by design
    beginCalibrationRef.current = beginCalibration;
  });

  const beginDrill = () => {
    cancelSpeech();
    setTimedRun(null);
    setCurrentCall(null);
    announcedIndexRef.current = -1;
    const steps = generateCallSequence(zones, drillLen, drillMode === "timed");
    const state = startDrill(steps);
    drillRef.current = state;
    setDrill(state);

    // After the beep: arm the chosen callout engine.
    const go = () => {
      if (drillMode === "timed") {
        const schedule = generateTimedSchedule(steps.length, timespanSec);
        setTimedRun({ steps, calloutAtMs: schedule.calloutAtMs, endAtMs: schedule.endAtMs });
      } else if (drillMode === "reactive") {
        announcedIndexRef.current = 0;
        const step = currentStep(state);
        if (step) {
          lastCallAtRef.current = performance.now();
          setCurrentCall({ label: step.label, kind: step.kind });
          speak(step.spoken, { onEnd: () => (lastCallAtRef.current = performance.now()) });
        }
      } else {
        // Sequence: the course of fire was announced before the commands —
        // the beep IS "go", shoot them in order from memory.
        announcedIndexRef.current = steps.length;
      }
    };
    const commands = () => {
      if (protocol === "none") {
        playStartBeep(go);
      } else {
        speakSequence(RANGE_COMMANDS[protocol], { onEnd: () => window.setTimeout(() => playStartBeep(go), 600) });
      }
    };
    if (drillMode === "sequence") {
      // Announce the full course of fire first, then run the range commands.
      speakSequence(
        steps.map((s) => s.spoken),
        { onEnd: commands },
      );
    } else {
      commands();
    }
  };
  useEffect(() => {
    beginDrillRef.current = beginDrill;
  });

  // Timed-callout engine (mirrors the drill page): one timer per callout plus
  // a final timeout; an unanswered call scores as a miss when the next fires.
  useEffect(() => {
    if (!timedRun) return;
    const advanceMisses = (uptoIndex: number): DrillState | null => {
      let state = drillRef.current;
      if (!state || state.status !== "running") return null;
      let timedOut = 0;
      while (state.index < uptoIndex && state.status === "running") {
        state = registerHit(state, null, 0);
        timedOut += 1;
      }
      if (timedOut > 0) {
        // Unanswered calls count against the session too.
        setSession((s) => ({ ...s, called: s.called + timedOut, misses: s.misses + timedOut, streak: 0 }));
      }
      drillRef.current = state;
      setDrill(state);
      return state;
    };
    const finishWith = (state: DrillState) => {
      setTimedRun(null);
      setCurrentCall(null);
      setSession((s) => ({ ...s, drills: s.drills + 1 }));
      const score = scoreDrill(state);
      speak(`Done. ${score.correct} of ${score.total}.`);
    };
    const timers: number[] = timedRun.calloutAtMs.map((atMs, index) =>
      window.setTimeout(() => {
        const state = advanceMisses(index);
        if (!state) return;
        if (state.status !== "running") {
          finishWith(state);
          return;
        }
        announcedIndexRef.current = index;
        const step = timedRun.steps[index];
        if (step) {
          lastCallAtRef.current = performance.now();
          setCurrentCall({ label: step.label, kind: step.kind });
          speak(step.spoken, { onEnd: () => (lastCallAtRef.current = performance.now()) });
        }
      }, atMs),
    );
    timers.push(
      window.setTimeout(() => {
        const state = advanceMisses(Number.MAX_SAFE_INTEGER);
        if (state) finishWith(state);
      }, timedRun.endAtMs),
    );
    return () => {
      for (const timer of timers) window.clearTimeout(timer);
    };
  }, [timedRun]);

  const stopDrill = () => {
    cancelSpeech();
    setTimedRun(null);
    setCurrentCall(null);
    announcedIndexRef.current = -1;
    drillRef.current = null;
    setDrill(null);
  };

  // Leave the fullscreen target board and return to the setup panel.
  const exitFullBoard = () => {
    if (typeof document !== "undefined" && document.fullscreenElement) {
      void document.exitFullscreen().catch(() => undefined);
    }
    stopDrill();
    setBriefActive(false);
    colorsCommittedRef.current = null; // the training instance is over
    setPhase("setup");
    setWizardStep("choose"); // back to the landing chooser
    setStatusLine("Ready — calibrate to train again.");
  };

  const activeCalPoint = phase === "calibrate" && calIndex < CAL_POINTS.length ? CAL_POINTS[calIndex] : null;
  const drillStep = drill ? currentStep(drill) : null;
  const drillScore = drill && drill.status === "done" ? scoreDrill(drill) : null;

  return (
    <main className="min-h-screen bg-black px-4 py-6 text-white">
      {/* ---- USER MODE: guided walkthrough (the shippable experience). The
           full pro UI stays mounted underneath so every ref keeps living. */}
      {uiMode === "user" && phase === "setup" ? (
        <div className="fixed inset-0 z-[70] overflow-y-auto bg-neutral-950/[0.97] px-4 py-8">
          <div className="mx-auto max-w-lg space-y-5">
            <div className="flex items-center justify-between">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-sky-400">Trackr · dry-fire trainer</p>
              <button
                type="button"
                onClick={() => setUiMode("pro")}
                className="text-[11px] text-gray-500 transition hover:text-gray-300"
              >
                Pro / diagnostic mode →
              </button>
            </div>
            {/* Step rail */}
            <div className="flex items-center gap-1.5">
              {WIZARD_STEPS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setWizardStep(s)}
                  title={WIZARD_TITLES[s]}
                  className={`h-1.5 flex-1 rounded-full transition ${
                    s === wizardStep
                      ? "bg-sky-400"
                      : WIZARD_STEPS.indexOf(s) < WIZARD_STEPS.indexOf(wizardStep)
                        ? "bg-sky-800"
                        : "bg-neutral-800"
                  }`}
                />
              ))}
            </div>
            <h1 className="text-2xl font-semibold">{WIZARD_TITLES[wizardStep]}</h1>

            {wizardStep === "choose" ? (
              <div className="space-y-4">
                <p className="text-sm leading-relaxed text-gray-300">
                  A printed flag in your (verified empty) pistol&apos;s muzzle lets the webcam compute exactly where
                  you&apos;re aiming — this screen is the target. Pick how you want to train; you can switch anytime.
                </p>
                <PathPreviewStyles />
                <button
                  type="button"
                  onClick={() => setTrainMode("target")}
                  aria-pressed={trainMode === "target"}
                  className={`w-full rounded-lg border p-3.5 text-left transition ${
                    trainMode === "target"
                      ? "border-sky-400 bg-sky-500/15 ring-1 ring-sky-400"
                      : "border-gray-700 hover:bg-neutral-900"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-white">🎯 Fixed target</p>
                      <p className="mt-1 text-xs leading-relaxed text-gray-400">
                        A ten-ring bullseye. Shoot at your own pace — every shot is plotted and scored, with full
                        group analysis: group size, zero offset, splits, and flinch detection. The right place to
                        build a clean trigger press.
                      </p>
                    </div>
                    <MiniTargetPreview />
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setTrainMode("drill")}
                  aria-pressed={trainMode === "drill"}
                  className={`w-full rounded-lg border p-3.5 text-left transition ${
                    trainMode === "drill"
                      ? "border-sky-400 bg-sky-500/15 ring-1 ring-sky-400"
                      : "border-gray-700 hover:bg-neutral-900"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-white">📢 Called drills</p>
                      <p className="mt-1 text-xs leading-relaxed text-gray-400">
                        Nine zones on screen. Range commands and the timer beep, then targets are called out at
                        surprise moments — read the call, find the zone, break the shot. Hits, misses, and reaction
                        times are scored. Trains target transitions and decision speed.
                      </p>
                    </div>
                    <MiniDrillPreview />
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setWizardStep("safety")}
                  className="w-full rounded-lg bg-emerald-500 px-4 py-4 text-base font-semibold text-black shadow-lg shadow-emerald-500/20 transition hover:bg-emerald-400"
                >
                  🎯 Calibrate and shoot!
                </button>
                <details className="text-[12px] text-gray-500">
                  <summary className="cursor-pointer text-gray-400 transition hover:text-gray-300">
                    What you&apos;ll need
                  </summary>
                  <ul className="mt-1.5 list-inside list-disc space-y-1">
                    <li>The printed muzzle flag, clicked together and seated in the bore</li>
                    <li>A webcam near this screen, facing you (1080p helps)</li>
                    <li>A quiet-ish room and 2–4 m of space</li>
                  </ul>
                </details>
                {session.shots > 0 ? (
                  <p className="font-mono text-[12px] text-gray-500">
                    This session: {session.shots} shots · {session.hits} hits
                    {session.called > 0 ? ` · ${Math.round((session.hits / session.called) * 100)}%` : ""}
                  </p>
                ) : null}
              </div>
            ) : null}

            {wizardStep === "safety" ? (
              <div className="space-y-4">
                <div className="rounded-lg border border-rose-500/50 bg-rose-500/10 p-4 text-sm text-rose-100">
                  <p className="font-semibold">Before anything else:</p>
                  <ul className="mt-2 list-inside list-disc space-y-1 text-[13px] text-rose-200/90">
                    <li>Remove the magazine. Lock the slide back.</li>
                    <li>Visually AND physically confirm the chamber is empty.</li>
                    <li>No ammunition in the room. None.</li>
                    <li>Treat every weapon as if it were loaded.</li>
                    <li>Never point a weapon at anything you do not intend to shoot.</li>
                    <li>Keep your finger straight and off the trigger until ready to fire.</li>
                    <li>Know your target and what lies beyond.</li>
                  </ul>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setWizardStep("prep");
                    if (!cameraActive) void startCamera();
                  }}
                  className="w-full rounded-md border border-rose-400/50 bg-rose-500/15 px-4 py-3 text-sm font-medium text-rose-100 transition hover:bg-rose-500/25"
                >
                  I verified the chamber is EMPTY and no ammunition is in the room →
                </button>
                <button
                  type="button"
                  onClick={() => setWizardStep("choose")}
                  className="w-full rounded-md border border-gray-700 px-4 py-2 text-xs text-gray-400 transition hover:bg-neutral-900"
                >
                  ← Back to choose &amp; train
                </button>
              </div>
            ) : null}

            {wizardStep === "prep" ? (
              <div className="space-y-4">
                <p className="text-sm leading-relaxed text-gray-300">
                  When you&apos;re ready, tap <em>Start calibration</em> — from there the voice guides everything, and
                  there are no more taps until you&apos;re shooting.
                </p>
                <ol className="list-inside list-decimal space-y-1.5 text-[13px] leading-relaxed text-gray-400">
                  {trigCalResult ? (
                    <li>Slide &amp; trigger: already calibrated — skipped.</li>
                  ) : (
                    <li>
                      Rack the slide when told — three short listen windows — then dry fire three times.
                    </li>
                  )}
                  <li>Seat the flag in the muzzle (colors to camera), raise, and hold your normal aim.</li>
                  <li>
                    Hold your sights on each of the 9 flashing dots until its ring fills.{" "}
                    {trainMode === "target" ? "Then the target is hot." : "Then the drill starts — wait for the beep."}
                  </li>
                </ol>
                {!cameraActive ? (
                  <button
                    type="button"
                    onClick={() => void startCamera()}
                    className="w-full rounded-md border border-sky-400/40 bg-sky-500/15 px-4 py-3 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
                  >
                    📷 Turn on the camera
                  </button>
                ) : (
                  <div className="rounded-lg border border-gray-800 p-3 text-sm">
                    <p className="font-mono text-[13px]">
                      {trigCalRacking
                        ? slideWindowOpen
                          ? `🎚 RACK NOW — listening (${Math.min(TRIG_CAL_CLICKS, (trigCalCount ?? 0) + 1)}/${TRIG_CAL_CLICKS})`
                          : `⏳ Wait… (slide ${Math.min(TRIG_CAL_CLICKS, (trigCalCount ?? 0) + 1)}/${TRIG_CAL_CLICKS})`
                        : trigCalCount !== null
                          ? `🎚 Dry fire (click ${Math.min(TRIG_CAL_CLICKS, trigCalCount + 1)}/${TRIG_CAL_CLICKS})…`
                          : photoCount !== null
                            ? `📸 FREEZE — photo in ${photoCount}…`
                            : photoStage === "position"
                              ? "🔍 Searching for the flag — hold the card in the flashing ring…"
                              : camLocked
                                ? "✅ Locked — follow the voice…"
                                : "🕓 Standing by — locking starts when the target and dots appear…"}
                    </p>
                    {coach && !trigCalRacking && trigCalCount === null ? (
                      <p className="mt-1.5 text-[12px] leading-snug text-amber-300/90">💡 {coach}</p>
                    ) : null}
                    {patternMode === "color" ? (
                      <div className="mt-2 flex gap-2">
                        {(colorPalette === "cmy"
                          ? (["green", "red", "blue"] as const)
                          : (["red", "green", "blue"] as const)
                        ).map((slot) => {
                          const c = refDraft[slot];
                          return (
                            <span
                              key={slot}
                              className="h-6 w-10 rounded border border-neutral-700"
                              style={{ background: c ? `rgb(${c[0]},${c[1]},${c[2]})` : "#171717" }}
                            />
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                )}
                {/* Live audio waveform + detection info during the slide &
                    trigger calibration steps. */}
                {micActive && (trigCalRacking || trigCalCount !== null) ? (
                  <div className="rounded-lg border border-gray-800 p-2">
                    <canvas ref={waveCanvasRef} width={640} height={96} className="h-20 w-full rounded bg-neutral-950" />
                    <p className="mt-1 flex flex-wrap justify-between gap-x-3 font-mono text-[10px] text-gray-400">
                      <span>peak {(micStats.peak * 100).toFixed(1)}</span>
                      <span>room {(micStats.ambient * 100).toFixed(2)}</span>
                      <span>gate {(micStats.threshold * 100).toFixed(1)}</span>
                      {micStats.sim !== null ? (
                        <span className={micStats.sim >= 0.7 ? "text-emerald-300" : "text-amber-300"}>
                          match {Math.round(micStats.sim * 100)}%
                        </span>
                      ) : null}
                      <span className={micStats.suppressed ? "text-amber-300" : "text-emerald-300"}>
                        {micStats.suppressed ? "HOLDING (noisy)" : "ARMED"}
                      </span>
                    </p>
                    <p className="mt-0.5 text-[10px] leading-snug text-gray-500">
                      <span className="text-sky-400">▮</span> sound · <span className="text-rose-400">┄</span> trigger
                      gate · <span className="text-gray-400">—</span> room noise ·{" "}
                      <span className="text-emerald-400">│</span> trigger pull ·{" "}
                      <span className="text-amber-400">│</span> slide / noise
                    </p>
                  </div>
                ) : null}
                {cameraError ? <p className="text-xs text-rose-300">{cameraError}</p> : null}
                {cameraActive && !(trigCalRacking || trigCalCount !== null) ? (
                  <button
                    type="button"
                    onClick={startFullCalibration}
                    className="w-full rounded-lg bg-emerald-500 px-4 py-4 text-base font-semibold text-black shadow-lg shadow-emerald-500/20 transition hover:bg-emerald-400"
                  >
                    {trigCalResult
                      ? "🎯 Start calibration (straight to the aim dots)"
                      : "🎯 Start calibration — slide & trigger, then the dots"}
                  </button>
                ) : null}
                {/* Per-step controls: every stage can be cancelled, restarted,
                    or redone without leaving the flow. */}
                {trigCalRacking || trigCalCount !== null ? (
                  <div className="flex gap-1.5">
                    <button
                      type="button"
                      onClick={cancelTrigCalibration}
                      className="flex-1 rounded-md border border-amber-400/50 bg-amber-500/10 px-4 py-2.5 text-sm font-medium text-amber-200 transition hover:bg-amber-500/20"
                    >
                      ✕ Cancel
                    </button>
                    <button
                      type="button"
                      onClick={restartTrigCalibration}
                      title="Start the slide & trigger calibration over from the rack"
                      className="flex-1 rounded-md border border-sky-400/40 bg-sky-500/10 px-4 py-2.5 text-sm font-medium text-sky-200 transition hover:bg-sky-500/20"
                    >
                      🙋 Wait — I messed up. Redo.
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {cameraActive && patternMode === "color" ? (
                      <>
                        <button
                          type="button"
                          onClick={startPhotoCountdown}
                          disabled={!tracker.tracking || photoCount !== null}
                          title="Runs the spoken 3-2-1, then captures your card's colors from the live lock"
                          className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition hover:bg-neutral-900 disabled:opacity-40"
                        >
                          {refComplete
                            ? "📸 Retake the color photo (3·2·1)"
                            : "📸 Optional: photo the card's colors (3·2·1)"}
                        </button>
                      </>
                    ) : null}
                    {cameraActive ? (
                      <button
                        type="button"
                        onClick={restartFlagLock}
                        title="Drop the current lock and color reference and re-acquire from scratch"
                        className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-400 transition hover:bg-neutral-900"
                      >
                        ↺ Restart flag lock
                      </button>
                    ) : null}
                    {trigCalResult ? (
                      <button
                        type="button"
                        onClick={() => {
                          setTrigCalResult(null);
                          triggerRef.current?.setCustomFloor(null);
                          triggerRef.current?.setFingerprints(null, null);
                        }}
                        className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-400 transition hover:bg-neutral-900"
                        title="A saved slide & trigger calibration will be reused — tap to redo it as part of the chain"
                      >
                        🎚 Recalibrate slide &amp; trigger too
                      </button>
                    ) : null}
                    <button
                      type="button"
                      onClick={() => {
                        cancelPhotoCountdown();
                        setWizardStep("choose");
                      }}
                      title="Back to the first page — nothing is lost, the camera stays on"
                      className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-400 transition hover:bg-neutral-900"
                    >
                      ← Start over
                    </button>
                  </div>
                )}
                {trigCalResult && !(trigCalRacking || trigCalCount !== null) ? (
                  <p className="text-[12px] text-gray-500">
                    Slide &amp; trigger are already calibrated from your saved scene — the chain will jump straight to
                    the aim dots. Use the recalibrate button above if the mic or the gun changed.
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-[1fr_300px]">
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <Link href="/" className="text-sm text-sky-300 hover:underline">
              ← Trackr
            </Link>
            <Link href="/drill" className="text-sm text-sky-300 hover:underline">
              Live-fire drill →
            </Link>
          </div>

          {/* Prompt banner */}
          <div className="flex min-h-[56px] items-center justify-between rounded-lg border border-gray-700 bg-neutral-950 px-4 py-3">
            {phase === "calibrate" ? (
              <p className="text-sm text-amber-200">
                Calibration dot {Math.min(calIndex + 1, CAL_POINTS.length)}/{CAL_POINTS.length} — hold your aim on it.
              </p>
            ) : drillStep ? (
              <div>
                <p className="text-[11px] uppercase tracking-wide text-gray-500">
                  Shoot {drillStep.kind} · {Math.min((drill?.index ?? 0) + 1, drill?.steps.length ?? 0)}/
                  {drill?.steps.length}
                </p>
                <p className="text-3xl font-bold">{drillStep.label}</p>
              </div>
            ) : drillScore ? (
              <p className="text-lg font-semibold">
                {drillScore.correct}/{drillScore.total} correct · {drillScore.accuracyPct.toFixed(0)}%
              </p>
            ) : (
              <p className="text-sm text-gray-400">{statusLine}</p>
            )}
          </div>

          {/* The board is the target. Trace + markers draw on the overlay.
              Out of setup it takes over the ENTIRE viewport (100vw × 100vh,
              landscape-locked fullscreen) — the targets are all that shows. */}
          <div ref={boardWrapRef} className={phase !== "setup" ? "fixed inset-0 z-50 bg-white" : "relative"}>
            {trainMode === "target" ? (
              <TargetFace
                aspectRatio={phase !== "setup" ? winAspect : SCENARIO_SHEET_ASPECT}
                className={phase !== "setup" ? "!h-full !rounded-none !border-0" : ""}
                scale={targetScale}
              />
            ) : (
              <ScenarioBoard
                zones={zones}
                feedback={feedback}
                interactive={false}
                paper
                aspectRatio={phase !== "setup" ? winAspect : SCENARIO_SHEET_ASPECT}
                className={phase !== "setup" ? "!h-full !rounded-none !border-0" : ""}
              />
            )}
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            {activeCalPoint ? (
              <div
                className="pointer-events-none absolute z-[95]"
                style={{ left: `${activeCalPoint[0] * 100}%`, top: `${activeCalPoint[1] * 100}%` }}
              >
                {/* Flashing black/white calibration marker — unmissable on
                    any background, and topmost while calibrating. */}
                <style>{`@keyframes calFlash { 0%, 49% { background:#000; border-color:#fff; } 50%, 100% { background:#fff; border-color:#000; } }`}</style>
                <div className="relative -translate-x-1/2 -translate-y-1/2">
                  <div
                    className="h-11 w-11 rounded-full border-4 shadow-lg"
                    style={{ animation: "calFlash 0.45s steps(1) infinite" }}
                  />
                  {calHold > 0 ? (
                    <div
                      className="absolute inset-0 rounded-full border-4 border-emerald-400"
                      style={{ clipPath: `inset(${(1 - calHold) * 100}% 0 0 0)` }}
                    />
                  ) : null}
                </div>
              </div>
            ) : null}
            {phase !== "setup" ? (
              <>
                {/* Big readable banner: the current CALL during drills, the
                    dot progress during calibration. */}
                {phase === "train" && currentCall ? (
                  <div className="pointer-events-none absolute left-1/2 top-3 z-[85] -translate-x-1/2 rounded-2xl bg-neutral-950/90 px-10 py-3 text-center shadow-2xl">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.35em] text-amber-400">
                      shoot · {currentCall.kind}
                    </p>
                    <p className="text-5xl font-black uppercase tracking-wide text-white">{currentCall.label}</p>
                  </div>
                ) : phase === "calibrate" ? (
                  <div className="pointer-events-none absolute left-1/2 top-3 z-[85] -translate-x-1/2 rounded-2xl bg-neutral-950/90 px-8 py-2.5 text-center shadow-2xl">
                    <p className="text-sm font-semibold text-white">
                      Dot {Math.min(calIndex + 1, CAL_POINTS.length)}/{CAL_POINTS.length} — hold your aim on the
                      flashing marker until its green ring fills
                    </p>
                  </div>
                ) : null}
                {/* Minimal floating controls — everything else is audio. */}
                <button
                  type="button"
                  onClick={exitFullBoard}
                  title="Exit to setup"
                  className="absolute right-2 top-2 rounded-full bg-black/25 px-2.5 py-1 text-sm text-black/60 transition hover:bg-black/40 hover:text-white"
                >
                  ✕
                </button>
                <button
                  type="button"
                  onClick={() => beginCalibration()}
                  title={
                    phase === "calibrate"
                      ? "Start the 9-dot aim calibration over from dot 1"
                      : "Redo the 9-dot aim calibration (your session stats are kept)"
                  }
                  className="absolute left-2 top-2 rounded-full bg-black/25 px-3 py-1 text-xs text-black/60 transition hover:bg-black/40 hover:text-white"
                >
                  {phase === "calibrate" ? "↺ Restart calibration" : "↻ Recalibrate aim"}
                </button>
                {phase === "train" && trainMode === "target" ? (
                  <div className="absolute bottom-3 left-1/2 z-[80] flex -translate-x-1/2 items-center gap-2">
                    <div className="flex items-center gap-1 rounded-full bg-black/30 px-2 py-1 text-xs text-white/90">
                      <button
                        type="button"
                        onClick={() => stepRange(-1)}
                        title="Bring the target closer"
                        className="rounded-full px-2 py-0.5 transition hover:bg-black/40"
                      >
                        −
                      </button>
                      <span className="min-w-14 text-center font-mono">🎯 {simRangeM} m</span>
                      <button
                        type="button"
                        onClick={() => stepRange(1)}
                        title="Push the target farther out"
                        className="rounded-full px-2 py-0.5 transition hover:bg-black/40"
                      >
                        +
                      </button>
                    </div>
                    {targetShots.length > 0 ? (
                      <button
                        type="button"
                        onClick={undoLastShot}
                        title="Strike the last recorded shot from the target and the stats"
                        className="rounded-full bg-black/30 px-3 py-1 text-xs text-white/90 transition hover:bg-black/50"
                      >
                        🙋 Wait — I messed up. Undo shot
                      </button>
                    ) : null}
                  </div>
                ) : null}
                {phase === "calibrate" && !briefActive && calIndex > 0 && calIndex < CAL_POINTS.length ? (
                  <button
                    type="button"
                    onClick={undoLastCalDot}
                    title="Throw away the previous dot's capture and hold on it again"
                    className="absolute bottom-3 left-1/2 z-[80] -translate-x-1/2 rounded-full bg-black/30 px-3 py-1.5 text-xs text-white/80 transition hover:bg-black/50"
                  >
                    🙋 Wait — I messed up. Redo dot {calIndex}
                  </button>
                ) : null}
                {phase === "calibrate" && briefActive ? (
                  <button
                    type="button"
                    onClick={skipBrief}
                    className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-black/30 px-3 py-1.5 text-xs text-white/80 transition hover:bg-black/50"
                  >
                    ⏭ Skip safety brief (weapon verified clear)
                  </button>
                ) : null}
                {coach ? (
                  <div className="pointer-events-none absolute bottom-3 left-3 max-w-xs rounded-lg bg-black/60 px-3 py-2 text-xs leading-snug text-amber-200">
                    💡 {coach}
                  </div>
                ) : null}
              </>
            ) : null}
          </div>
          <p className="text-[11px] text-gray-500">
            Dry-fire only — verified-empty firearm, no ammunition in the room. The screen is the target; the webcam
            watches the flag on the muzzle.
          </p>
        </section>

        {/* Controls */}
        <aside className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Dry-fire trainer</h1>
            <button
              type="button"
              onClick={() => {
                setUiMode("user");
                setWizardStep("choose");
              }}
              className="rounded px-2 py-1 text-[11px] text-gray-500 transition hover:bg-neutral-900 hover:text-gray-300"
            >
              🧭 Guided setup
            </button>
          </div>

          <div className="rounded-lg border border-rose-500/50 bg-rose-500/10 p-3 text-xs text-rose-100">
            <p className="font-semibold">⚠ Check weapon condition first</p>
            <p className="mt-1 text-rose-200/90">
              Remove the magazine, lock the slide back, and visually AND physically confirm the chamber is empty. No
              ammunition in the room. Re-check every single time you pick the gun up — the flag in the muzzle is a
              reminder, not a guarantee.
            </p>
          </div>

          <div className="space-y-2 rounded-lg border border-gray-800 p-3">
            <p className="text-xs font-semibold text-gray-300">Scenario profiles</p>
            <div className="flex gap-1.5">
              <select
                value={activeProfile}
                onChange={(e) => {
                  const p = profiles.find((x) => x.name === e.target.value);
                  setActiveProfile(e.target.value);
                  if (p) {
                    applyProfile(p);
                    try {
                      localStorage.setItem(LAST_PROFILE_KEY, p.name);
                    } catch {
                      /* best effort */
                    }
                  }
                }}
                className="flex-1 rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-xs text-gray-200"
              >
                <option value="">— pick a saved scene —</option>
                {profiles.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={deleteProfile}
                disabled={!activeProfile}
                title="Delete the selected profile"
                className="rounded-md border border-gray-700 px-2 text-xs text-gray-400 transition hover:bg-neutral-900 disabled:opacity-40"
              >
                🗑
              </button>
            </div>
            <div className="flex gap-1.5">
              <input
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder={activeProfile ? `Save as… (blank = "${activeProfile}")` : "Name this scene (e.g. Garage evening)"}
                className="min-w-0 flex-1 rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-xs text-gray-200 placeholder:text-gray-600"
              />
              <button
                type="button"
                onClick={saveProfile}
                className="rounded-md bg-sky-500 px-3 py-1.5 text-xs font-semibold text-black transition hover:bg-sky-400"
              >
                Save
              </button>
            </div>
            <p className="text-[10px] text-gray-600">
              Saves this scene&apos;s CMY color reference, tuning, slide/trigger calibration, and drill setup. The
              last-used profile auto-loads next visit — same spot, zero re-setup.
            </p>
          </div>

          <div className="space-y-3 rounded-lg border border-gray-800 p-3">
            <p className="text-xs font-semibold text-gray-300">1 · Camera & flag</p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => (cameraActive ? stopCamera() : void startCamera())}
                className="flex-1 rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
              >
                {cameraActive ? "Stop camera" : "Start camera"}
              </button>
              <button
                type="button"
                onClick={() => setShowDiag((prev) => !prev)}
                aria-pressed={showDiag}
                title="Show what the flag detector sees, stage by stage"
                className={`rounded-md border px-3 py-2 text-sm transition ${
                  showDiag ? "border-amber-300 bg-amber-500/20 text-amber-100" : "border-gray-600 text-gray-200 hover:bg-neutral-800"
                }`}
              >
                🔬
              </button>
            </div>
            {cameraError ? <p className="text-xs text-rose-300">{cameraError}</p> : null}
            {/* Checker mode + the RGB palette are ARCHIVED — CMY card and
                red shapes card are the supported paths. */}
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={() => setPatternMode("color")}
                aria-pressed={patternMode === "color"}
                className={`rounded-md border px-2 py-1.5 text-xs transition ${
                  patternMode === "color"
                    ? "border-sky-400/50 bg-sky-500/15 text-sky-100"
                    : "border-gray-700 text-gray-400 hover:bg-neutral-900"
                }`}
              >
                🩵💛🩷 Color card
              </button>
              <button
                type="button"
                onClick={() => setPatternMode("shape")}
                aria-pressed={patternMode === "shape"}
                className={`rounded-md border px-2 py-1.5 text-xs transition ${
                  patternMode === "shape"
                    ? "border-sky-400/50 bg-sky-500/15 text-sky-100"
                    : "border-gray-700 text-gray-400 hover:bg-neutral-900"
                }`}
              >
                ●◍◎ Shapes
              </button>
            </div>
            {patternMode === "color" ? (
              <span className="text-[10px] text-gray-600">
                Facing the card: CYAN top-left, YELLOW top-right, MAGENTA bottom-left, WHITE with a BLACK dot
                bottom-right. CMY patches are ~2× brighter in dim rooms.
              </span>
            ) : null}
            {patternMode === "color" ? (
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-gray-500">Color pickup:</span>
                  {(["forgiving", "normal", "strict"] as const).map((level) => (
                    <button
                      key={level}
                      type="button"
                      onClick={() => {
                        setColorSensitivity(level);
                        setColorTuning(tuningFromPreset(level, colorPalette));
                      }}
                      aria-pressed={colorSensitivity === level}
                      className={`rounded px-2 py-1 text-[11px] transition ${
                        colorSensitivity === level
                          ? "bg-sky-500/20 text-sky-100"
                          : "text-gray-400 hover:bg-neutral-900"
                      }`}
                    >
                      {level}
                    </button>
                  ))}
                  {colorSensitivity === "custom" ? (
                    <span className="rounded bg-sky-500/20 px-2 py-1 text-[11px] text-sky-100">custom</span>
                  ) : null}
                </div>
                {(colorPalette === "cmy"
                  ? ([
                      ["C", "#22d3ee", "green"],
                      ["Y", "#facc15", "red"],
                      ["M", "#e879f9", "blue"],
                    ] as const)
                  : ([
                      ["R", "#f87171", "red"],
                      ["G", "#4ade80", "green"],
                      ["B", "#60a5fa", "blue"],
                    ] as const)
                ).map(([label, accent, slot]) => (
                  <DualSlider
                    key={slot}
                    label={label}
                    accent={accent}
                    value={colorTuning[slot]}
                    boundaryColor={(v, m) => {
                      const dom = Math.min(255, Math.round(v));
                      const rival = Math.max(0, Math.round(v - m));
                      if (colorPalette === "cmy") {
                        // yellow / cyan / magenta: two channels at dom, the
                        // excluded one at the rival ceiling.
                        if (slot === "red") return [dom, dom, rival];
                        if (slot === "green") return [rival, dom, dom];
                        return [dom, rival, dom];
                      }
                      if (slot === "red") return [dom, rival, rival];
                      if (slot === "green") return [rival, dom, rival];
                      return [rival, rival, dom];
                    }}
                    onChange={(next) => {
                      setColorSensitivity("custom");
                      setColorTuning((t) => ({ ...t, [slot]: next }));
                    }}
                  />
                ))}
                <span className="text-[10px] text-gray-600">
                  Lower handle: brightness floor. Upper handle: how far the color must beat the other channels.
                  Use 🔬 — tighten until background strays stop lighting up but the card&apos;s patches stay solid.
                </span>
                <div className="flex flex-wrap items-center gap-1.5 border-t border-gray-800 pt-1.5">
                  <span className="text-[11px] text-gray-500">Reference:</span>
                  {(colorPalette === "cmy" ? (["green", "red", "blue"] as const) : (["red", "green", "blue"] as const)).map((slot) => {
                    const c = refDraft[slot];
                    const names =
                      colorPalette === "cmy"
                        ? ({ green: "C", red: "Y", blue: "M" } as const)
                        : ({ red: "R", green: "G", blue: "B" } as const);
                    return (
                      <button
                        key={slot}
                        type="button"
                        onClick={() => setPickSlot(pickSlot === slot ? null : slot)}
                        aria-pressed={pickSlot === slot}
                        title={
                          c
                            ? `rgb(${c.map((v) => Math.round(v)).join(",")}) — click, then tap the patch in the preview to resample`
                            : "Click, then tap this patch in the camera preview"
                        }
                        className={`h-6 w-8 rounded border text-[10px] font-semibold ${
                          pickSlot === slot ? "border-sky-400 ring-1 ring-sky-400" : "border-neutral-700"
                        }`}
                        style={{
                          background: c ? `rgb(${c[0]},${c[1]},${c[2]})` : "transparent",
                          color: c ? "#000" : "#9ca3af",
                        }}
                      >
                        {names[slot]}
                      </button>
                    );
                  })}
                  <button
                    type="button"
                    onClick={() => scaleRefColors(1.15)}
                    title="Room got brighter — brighten the references"
                    className="rounded px-1.5 py-1 text-[12px] text-gray-400 transition hover:bg-neutral-900"
                  >
                    ☀
                  </button>
                  <button
                    type="button"
                    onClick={() => scaleRefColors(0.87)}
                    title="Room got darker — darken the references"
                    className="rounded px-1.5 py-1 text-[12px] text-gray-400 transition hover:bg-neutral-900"
                  >
                    🌙
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setRefDraft({ red: null, green: null, blue: null });
                      setPickSlot(null);
                    }}
                    title="Clear the reference — back to the gate sliders"
                    className="rounded px-1.5 py-1 text-[12px] text-gray-400 transition hover:bg-neutral-900"
                  >
                    ✕
                  </button>
                  <label className="flex w-full items-center gap-2 text-[10px] text-gray-500">
                    <span className="whitespace-nowrap">±{Math.round(refBrightTol * 100)}% brightness</span>
                    <input
                      type="range"
                      min={15}
                      max={90}
                      step={5}
                      value={Math.round(refBrightTol * 100)}
                      onChange={(event) => setRefBrightTol(Number(event.target.value) / 100)}
                      className="flex-1"
                    />
                  </label>
                  <span className="w-full text-[10px] text-gray-600">
                    {refDraft.red && refDraft.green && refDraft.blue
                      ? "Reference ACTIVE — pickup matches these exact colors (overrides the sliders). ☀/🌙 nudge for lighting changes; ✕ reverts to the gates."
                      : "Tap a swatch then tap that patch in the camera preview — or just hold a lock for a second and the reference auto-fills."}
                  </span>
                </div>
              </div>
            ) : null}
            {patternMode === "shape" ? (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-gray-500">Red pickup:</span>
                {(["strict", "normal", "forgiving", "wide"] as const).map((level) => (
                  <button
                    key={level}
                    type="button"
                    onClick={() => setRedSensitivity(level)}
                    aria-pressed={redSensitivity === level}
                    className={`rounded px-2 py-1 text-[11px] transition ${
                      redSensitivity === level
                        ? "bg-sky-500/20 text-sky-100"
                        : "text-gray-400 hover:bg-neutral-900"
                    }`}
                  >
                    {level}
                  </button>
                ))}
                <span className="w-full text-[10px] text-gray-600">
                  Use 🔬: pick the strictest level that still fills the card&apos;s patches solidly.
                </span>
              </div>
            ) : null}
            {patternMode === "shape" ? (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-gray-500">Require:</span>
                {(
                  [
                    ["disk", "● disk"],
                    ["ring", "◍ ring"],
                    ["two", "◎ 2-hole"],
                    ["dot", "• dot"],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setShapeReq((s) => ({ ...s, [key]: !s[key] }))}
                    aria-pressed={shapeReq[key]}
                    className={`rounded px-2 py-1 text-[11px] transition ${
                      shapeReq[key] ? "bg-sky-500/20 text-sky-100" : "text-gray-400 hover:bg-neutral-900"
                    }`}
                  >
                    {label}
                  </button>
                ))}
                <span className="w-full text-[10px] text-gray-600">
                  Off = that slot accepts any red blob (layout + chirality still gate). Dot on = no dotless locks.
                </span>
              </div>
            ) : null}
            {cameraActive && coach ? <p className="text-[11px] leading-snug text-amber-300/90">💡 {coach}</p> : null}
            {/* Camera + detection preview. Out of setup it becomes a small
                picture-in-picture pinned to the bottom-right of the
                fullscreen board, with a compact detection readout. During
                the wizard's point-and-hold step it floats larger above
                the walkthrough overlay. */}
            <div
              className={
                phase === "calibrate"
                  ? // Parked on the right edge BETWEEN the dot rows (12% / 50%
                    // / 88%) and shrunk — a calibration dot must never hide
                    // behind the camera window.
                    "fixed right-3 top-[22%] z-[60] w-40"
                  : phase === "train"
                    ? "fixed bottom-3 right-3 z-[60] w-64"
                    : uiMode === "user" && wizardStep === "prep"
                      ? // Top-right during the wizard so it can NEVER cover the
                        // centered action buttons.
                        "fixed right-3 top-3 z-[80] w-56 sm:w-80"
                      : "relative"
              }
            >
              <video ref={videoRef} className="hidden" />
              <canvas
                ref={previewCanvasRef}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const nx = (e.clientX - rect.left) / Math.max(1, rect.width);
                  const ny = (e.clientY - rect.top) / Math.max(1, rect.height);
                  // "Help computer." guided picking on the frozen frame.
                  if (helpPick) {
                    const fc = freezeCanvasRef.current;
                    const fctx = fc?.getContext("2d");
                    if (!fc || !fctx) return;
                    const x = Math.max(2, Math.min(fc.width - 3, Math.round(nx * fc.width)));
                    const y = Math.max(2, Math.min(fc.height - 3, Math.round(ny * fc.height)));
                    const d = fctx.getImageData(x - 2, y - 2, 5, 5).data;
                    let r = 0;
                    let g = 0;
                    let b = 0;
                    for (let i = 0; i < 25; i += 1) {
                      r += d[i * 4];
                      g += d[i * 4 + 1];
                      b += d[i * 4 + 2];
                    }
                    const rgb: [number, number, number] = [r / 25, g / 25, b / 25];
                    const step = helpOrder[helpPick.stage];
                    setRefDraft((prev) => ({ ...prev, [step.slot]: rgb }));
                    if (helpPick.stage + 1 < helpOrder.length) {
                      setHelpPick({ stage: helpPick.stage + 1 });
                      speak(`Now tap the ${helpOrder[helpPick.stage + 1].name.toLowerCase()} patch.`);
                    } else {
                      setHelpPick(null);
                      speak("Reference colors set. Resume your shooting position.");
                    }
                    return;
                  }
                  if (!pickSlot || !sampleProcRef.current) return;
                  const rgb = sampleProcRef.current(nx, ny);
                  if (!rgb) return;
                  const nd = { ...refDraft, [pickSlot]: rgb };
                  setRefDraft(nd);
                  setPickSlot((["red", "green", "blue"] as const).find((s) => !nd[s]) ?? null);
                }}
                className={`w-full rounded border bg-black ${
                  pickSlot || helpPick ? "cursor-crosshair border-sky-400" : "border-gray-800"
                }`}
              />
              <button
                type="button"
                onClick={() => setShowGuide((v) => !v)}
                title={showGuide ? "Hide the ideal-position guide" : "Show the ideal-position guide"}
                className={`absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold transition ${
                  showGuide ? "bg-sky-500/80 text-black" : "bg-neutral-800/80 text-gray-400 hover:text-gray-200"
                }`}
              >
                👤
              </button>
              <span
                className={`absolute right-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                  tracker.tracking && tracker.misses === 0
                    ? "bg-emerald-500/80 text-black"
                    : tracker.tracking
                      ? "bg-amber-500/80 text-black"
                      : tracker.locked
                        ? "bg-emerald-500/80 text-black"
                        : "bg-neutral-800 text-gray-300"
                }`}
              >
                {tracker.tracking && tracker.misses === 0
                  ? `${tracker.regional ? "REGIONAL LOCK" : "TRACK LOCK"} · ${tracker.fps}fps`
                  : tracker.tracking
                    ? `re-seeking ${tracker.misses}/${tracker.regional ? 40 : 16}…`
                    : tracker.locked
                      ? `FLAG LOCK · ${tracker.fps}fps`
                      : cameraActive
                        ? uiMode === "pro" ||
                          phase !== "setup" ||
                          photoStage === "position" ||
                          photoStage === "countdown" ||
                          helpPick !== null
                          ? "searching…"
                          : "standing by"
                        : "camera off"}
              </span>
              {phase !== "setup" && diag ? (
                <p className="mt-0.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[9px] leading-tight text-amber-200/90">
                  {diag.stage}
                  {diag.cellPx !== null ? ` · quad ${diag.cellPx.toFixed(0)}px` : ""}
                  {diag.residual !== null ? ` · fit ${(diag.residual * 100).toFixed(0)}%` : ""}
                  {diag.colorCounts
                    ? patternMode === "shape"
                      ? ` · ●${diag.colorCounts.red} ◍${diag.colorCounts.green} ◎${diag.colorCounts.blue}`
                      : ` · R${diag.colorCounts.red} G${diag.colorCounts.green} B${diag.colorCounts.blue}`
                    : ""}
                  {diag.colorGates?.anchors !== null && diag.colorGates?.anchors !== undefined
                    ? ` · anc ${diag.colorGates.anchors}/4`
                    : ""}
                  {hLogCount > 0 ? ` · H${hLogCount}` : ""}
                </p>
              ) : null}
              {cameraActive &&
              patternMode === "color" &&
              (helpPick !== null || !(tracker.locked || (tracker.tracking && tracker.misses === 0))) ? (
                <button
                  type="button"
                  onClick={startHelpComputer}
                  className={`mt-1 w-full rounded px-2 py-1.5 text-[11px] transition ${
                    helpPick
                      ? "bg-sky-500/25 text-sky-100"
                      : "bg-black/70 text-sky-200 hover:bg-black/90 hover:text-sky-100"
                  }`}
                >
                  {helpPick
                    ? `👉 Tap the ${helpOrder[helpPick.stage].name} patch on the still — ${helpPick.stage + 1}/${helpOrder.length} (tap here to cancel)`
                    : "🖖 Help computer."}
                </button>
              ) : null}
            </div>
            {tracker.locked ? (
              <p className="text-[11px] text-gray-500">
                cell {tracker.cellPx.toFixed(1)}px · fit {(tracker.residual * 100).toFixed(0)}% — bigger cell = better;
                move closer or raise camera resolution if under ~6px.
              </p>
            ) : null}
            {showDiag ? (
              <div className="space-y-1.5 rounded-md border border-amber-400/30 bg-amber-500/5 p-2 text-[11px] leading-relaxed">
                {diag && diag.stage !== "none" ? (
                  <>
                    <p className="font-semibold text-amber-100">
                      Stage: {diag.stage === "ok" ? "✓ all passing" : `✗ failing at "${diag.stage}"`}
                    </p>
                    <p className="text-amber-100/90">{DIAG_HINTS[diag.stage as FlagFailStage]}</p>
                    <p className="font-mono text-amber-200/80">
                      cam {diag.procW}×{diag.procH}
                      {diag.procW < 1280 ? " (LOW — hurts range)" : ""}
                      {patternMode === "checker" ? ` · dark<${diag.threshold}` : ""} · {diag.roiActive ? "ROI" : "full scan"}
                    </p>
                    {diag.colorCounts ? (
                      <p className="font-mono text-amber-200/80">
                        {patternMode === "shape"
                          ? `disk ${diag.colorCounts.red} · ring ${diag.colorCounts.green} · 2-hole ${diag.colorCounts.blue}`
                          : colorPalette === "cmy" && patternMode === "color"
                            ? `cyan ${diag.colorCounts.green} · yellow ${diag.colorCounts.red} · magenta ${diag.colorCounts.blue}`
                            : `red ${diag.colorCounts.red} · green ${diag.colorCounts.green} · blue ${diag.colorCounts.blue}`}
                        {diag.cellPx !== null ? ` · quadrant ${diag.cellPx.toFixed(1)}px` : ""}
                        {diag.colorGates
                          ? ` · combos ${diag.colorGates.triples}→size ${diag.colorGates.sizeOk}→area ${diag.colorGates.areaOk}` +
                            (diag.colorGates.nearestDotErrCells !== null
                              ? ` · dot miss ${(diag.colorGates.nearestDotErrCells * 100).toFixed(0)}%≤60%`
                              : patternMode === "shape"
                                ? " · no dot-sized blob anywhere"
                                : " · no dot-sized blue anywhere") +
                            (diag.colorGates.dotBoxHit ? " · dot-in-box ✓" : "") +
                            (diag.colorGates.strengths
                              ? ` · purity ${diag.colorGates.strengths.map((s) => (s * 100).toFixed(0)).join("/")}`
                              : "") +
                            (diag.colorGates.chiralityOk !== null
                              ? ` · sweep ${diag.colorGates.chiralityOk ? "✓" : "✗ MIRRORED"}`
                              : "") +
                            (diag.colorGates.anchors !== null ? ` · anchors ${diag.colorGates.anchors}/4` : "") +
                            (diag.colorGates.dotless ? " · DOTLESS LOCK" : "")
                          : ""}
                      </p>
                    ) : (
                      <p className="font-mono text-amber-200/80">
                        blobs {diag.blobCount} → clusters {diag.clusterCount} → tiles {diag.tileCount}/8
                        {diag.quadScore !== null ? ` · diag ${diag.quadScore.toFixed(3)}≤0.12` : ""} · dots {diag.dotCount}
                        {diag.cellPx !== null ? ` · cell ${diag.cellPx.toFixed(1)}px` : ""}
                        {diag.residual !== null ? ` · fit ${(diag.residual * 100).toFixed(0)}%≤35%` : ""}
                      </p>
                    )}
                    <p className="text-amber-200/70">
                      {patternMode === "color"
                        ? "Overlay: saturated tint = pixel classified as that color · green ○ = chosen quadrants · amber ○ = dot · amber box = dot blob bounds (solid = contains prediction) · magenta ✕ = where the dot SHOULD be · blue box = search area."
                        : patternMode === "shape"
                          ? "Overlay: red tint = counted as dark · green ○ = chosen disk/ring/2-hole · amber ○ = dot · amber box = dot blob bounds (solid = contains prediction) · magenta ✕ = where the dot SHOULD be · blue box = search area."
                          : "Overlay: red tint = counted as dark · gray dots = all blobs · green ○ = tile picks · cyan line = diagonal · amber ○ = dot candidates · blue box = search area."}
                    </p>
                  </>
                ) : (
                  <p className="text-amber-100/80">Waiting for the first analysis…</p>
                )}
              </div>
            ) : null}
          </div>

          <div className="space-y-2 rounded-lg border border-gray-800 p-3">
            <p className="text-xs font-semibold text-gray-300">2 · Calibrate aim</p>
            <button
              type="button"
              onClick={beginCalibration}
              disabled={!cameraActive}
              className="w-full rounded-md bg-amber-400 px-3 py-2.5 text-sm font-semibold text-black transition hover:bg-amber-300 disabled:opacity-40"
            >
              {model ? "Recalibrate (9 dots)" : "Calibrate (9 dots)"}
            </button>
            {phase === "calibrate" && briefActive ? (
              <button
                type="button"
                onClick={skipBrief}
                className="w-full rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition hover:bg-neutral-800"
              >
                ⏭ Skip safety brief (weapon already verified clear)
              </button>
            ) : null}
            {model ? (
              <p className="text-[11px] text-gray-500">
                Fit residual {model.rmsErrorPx.toFixed(0)}px over {model.sampleCount} samples.
              </p>
            ) : null}
            <button
              type="button"
              onClick={() => {
                if (!modelRef.current || holdActive) return;
                speak("Hold test. Aim at one point and hold for ten seconds.");
                holdTestRef.current = { startMs: performance.now() + 2500, until: performance.now() + 12500, pts: [] };
                setHoldActive(true);
                setHoldResult(null);
              }}
              disabled={!model || phase !== "train" || holdActive}
              className="w-full rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800 disabled:opacity-50"
            >
              {holdActive ? "⏱ Hold test running…" : "⏱ 10 s hold test (measure jitter)"}
            </button>
            {holdResult ? (
              <p className="font-mono text-[11px] text-gray-400">
                hold: RMS {holdResult.rmsPx.toFixed(0)}px · drift {holdResult.driftPxPerS.toFixed(0)}px/s ·{" "}
                {holdResult.n} samples
              </p>
            ) : null}
            {resizedSinceCal ? (
              <p className="text-[11px] text-amber-300">Window changed since calibration — recalibrate.</p>
            ) : null}
          </div>

          <div className="space-y-2 rounded-lg border border-gray-800 p-3">
            <p className="text-xs font-semibold text-gray-300">3 · Train</p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => (micActive ? stopMic() : void startMic())}
                disabled={phase !== "train"}
                className="flex-1 rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:opacity-50"
              >
                {micActive ? "🎙 Mic trigger on" : "🎙 Enable mic trigger"}
              </button>
              <button
                type="button"
                onClick={() => handleShotRef.current()}
                disabled={phase !== "train"}
                title="Manual shot (or use the mic to catch the striker click)"
                className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:opacity-50"
              >
                ⬤
              </button>
            </div>
            {micActive ? (
              <div className="space-y-1">
                <div className="relative h-2 w-full overflow-hidden rounded bg-neutral-800">
                  <div className="h-full bg-emerald-500 transition-all" style={{ width: `${micStats.pct}%` }} />
                  {/* Decaying click-peak marker: the bar to beat is the right edge. */}
                  <div
                    className={`absolute inset-y-0 w-1 rounded ${micStats.peakPct >= 100 ? "bg-emerald-300" : "bg-amber-400"}`}
                    style={{ left: `calc(${Math.min(99, micStats.peakPct)}% )` }}
                  />
                </div>
                <p className="font-mono text-[10px] text-gray-500">
                  click peak {micStats.peak.toFixed(3)} · fires at {micStats.threshold.toFixed(3)} · room{" "}
                  {micStats.ambient.toFixed(3)}
                  {micStats.suppressed
                    ? " — 🔇 noisy action (rack?) — trigger held"
                    : micStats.rejectedAgoMs < 2000
                      ? " — 🔇 rack/noise ignored ✓"
                      : micStats.peakPct >= 100
                        ? " — ✓ clearing the bar"
                        : " — dry-fire once: marker must reach the end"}
                </p>
              </div>
            ) : null}
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-gray-500">Trigger:</span>
              {(["high", "normal", "low"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => {
                    setTrigSensitivity(s);
                    triggerRef.current?.setSensitivity(s);
                  }}
                  aria-pressed={trigSensitivity === s}
                  className={`rounded px-2 py-1 text-[11px] transition ${
                    trigSensitivity === s ? "bg-sky-500/20 text-sky-100" : "text-gray-400 hover:bg-neutral-900"
                  }`}
                >
                  {s === "high" ? "hair" : s === "normal" ? "normal" : "firm"}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() =>
                  trigCalCount !== null || trigCalRacking ? cancelTrigCalibration() : void beginTrigCalibration()
                }
                className={`rounded-md border px-2 py-1.5 text-xs transition ${
                  trigCalCount !== null || trigCalRacking
                    ? "border-amber-400/50 bg-amber-500/15 text-amber-100"
                    : "border-gray-700 text-gray-300 hover:bg-neutral-900"
                }`}
              >
                {trigCalRacking
                  ? "🎚 Rack the slide… (tap to cancel)"
                  : trigCalCount !== null
                    ? `🎚 Listening… click ${trigCalCount}/${TRIG_CAL_CLICKS} (tap to cancel)`
                    : "🎚 Calibrate slide & trigger"}
              </button>
              {trigCalResult ? (
                <span className="font-mono text-[10px] text-gray-500">
                  click ~{trigCalResult.clickPeak.toFixed(3)} → fires at {trigCalResult.floor.toFixed(3)}
                  {trigCalResult.rackOk === true
                    ? " · rack ✓ ignored"
                    : trigCalResult.rackOk === false
                      ? " · ⚠ rack registers as shot"
                      : ""}
                  <button
                    type="button"
                    onClick={() => {
                      setTrigCalResult(null);
                      triggerRef.current?.setCustomFloor(null);
                      triggerRef.current?.setFingerprints(null, null);
                    }}
                    title="Clear the calibrated threshold — back to presets"
                    className="ml-1 rounded px-1 text-gray-500 hover:bg-neutral-900 hover:text-gray-300"
                  >
                    ✕
                  </button>
                </span>
              ) : null}
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={() => setTrainMode("drill")}
                aria-pressed={trainMode === "drill"}
                className={`rounded-md border px-2 py-1.5 text-xs ${
                  trainMode === "drill" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                }`}
              >
                📢 Called drill
              </button>
              <button
                type="button"
                onClick={() => {
                  if (drill && drill.status === "running") stopDrill();
                  setTrainMode("target");
                }}
                aria-pressed={trainMode === "target"}
                className={`rounded-md border px-2 py-1.5 text-xs ${
                  trainMode === "target" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                }`}
              >
                🎯 Open target
              </button>
            </div>
            {trainMode === "target" ? (
              <button
                type="button"
                onClick={() => {
                  setTargetShots([]);
                  shotMarkersRef.current = [];
                }}
                className="w-full rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800"
              >
                Clear target ({targetShots.length} shots)
              </button>
            ) : null}
            {trainMode === "target" ? (
              <div className="flex items-center justify-between rounded-md border border-gray-800 px-3 py-2">
                <span className="text-xs text-gray-400">Simulated range</span>
                <span className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => stepRange(-1)}
                    className="rounded px-2 py-0.5 text-sm text-gray-300 transition hover:bg-neutral-800"
                  >
                    −
                  </button>
                  <span className="min-w-12 text-center font-mono text-sm text-gray-200">{simRangeM} m</span>
                  <button
                    type="button"
                    onClick={() => stepRange(1)}
                    className="rounded px-2 py-0.5 text-sm text-gray-300 transition hover:bg-neutral-800"
                  >
                    +
                  </button>
                </span>
              </div>
            ) : null}
            <div className={trainMode === "drill" ? "" : "hidden"}>
              <span className="text-xs text-gray-400">Range commands</span>
              <div className="mt-1 grid grid-cols-5 gap-1.5">
                {(
                  [
                    ["standard", "Std"],
                    ["uspsa", "USPSA"],
                    ["ipsc", "IPSC"],
                    ["idpa", "IDPA"],
                    ["none", "Off"],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setProtocol(key)}
                    aria-pressed={protocol === key}
                    className={`rounded-md border px-2 py-1.5 text-xs ${
                      protocol === key ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <p className="mt-1 text-[11px] text-gray-500">
                {protocol === "none"
                  ? "Straight to the beep."
                  : `"${RANGE_COMMANDS[protocol].join(" … ")}" … BEEP`}
              </p>
            </div>
            <div className={trainMode === "drill" ? "" : "hidden"}>
              <span className="text-xs text-gray-400">Callout mode</span>
              <div className="mt-1 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setDrillMode("timed")}
                  className={`col-span-2 rounded-md border px-2 py-1.5 text-xs ${
                    drillMode === "timed" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  Random timing — calls at surprise moments
                </button>
                <button
                  type="button"
                  onClick={() => setDrillMode("sequence")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    drillMode === "sequence"
                      ? "border-sky-400/50 bg-sky-500/15 text-sky-100"
                      : "border-gray-700 text-gray-300"
                  }`}
                >
                  Full sequence
                </button>
                <button
                  type="button"
                  onClick={() => setDrillMode("reactive")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    drillMode === "reactive"
                      ? "border-sky-400/50 bg-sky-500/15 text-sky-100"
                      : "border-gray-700 text-gray-300"
                  }`}
                >
                  One at a time
                </button>
              </div>
              <label className="mt-2 block">
                <span className="text-xs text-gray-400">Calls: {drillLen}</span>
                <input
                  type="range"
                  min={3}
                  max={20}
                  step={1}
                  value={drillLen}
                  onChange={(event) => setDrillLen(Number(event.target.value))}
                  className="mt-1 w-full"
                />
              </label>
              {drillMode === "timed" ? (
                <label className="mt-2 block">
                  <span className="text-xs text-gray-400">
                    Drill window: ~{timespanSec}s (the actual length and call moments vary every run)
                  </span>
                  <input
                    type="range"
                    min={10}
                    max={120}
                    step={5}
                    value={timespanSec}
                    onChange={(event) => setTimespanSec(Number(event.target.value))}
                    className="mt-1 w-full"
                  />
                </label>
              ) : null}
            </div>
            <div className={trainMode === "drill" ? "flex gap-2" : "hidden"}>
              <button
                type="button"
                onClick={drill && drill.status === "running" ? stopDrill : beginDrill}
                disabled={phase !== "train"}
                className="flex-1 rounded-md border border-emerald-400/40 bg-emerald-500/15 px-3 py-2 text-sm font-medium text-emerald-100 transition hover:bg-emerald-500/25 disabled:opacity-50"
              >
                {drill && drill.status === "running" ? "Stop drill" : "Start called drill"}
              </button>
            </div>
          </div>

          {trainMode === "target" && targetStats ? (
            <div className="space-y-2 rounded-lg border border-gray-800 p-3">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-gray-300">Target group · units = target radius</p>
                <button
                  type="button"
                  onClick={() => setShowStatHelp((v) => !v)}
                  aria-pressed={showStatHelp}
                  title="What these statistics mean and how they're derived"
                  className={`rounded-full px-2 py-0.5 text-[11px] transition ${
                    showStatHelp ? "bg-sky-500/20 text-sky-100" : "text-gray-500 hover:bg-neutral-900 hover:text-gray-300"
                  }`}
                >
                  ⓘ
                </button>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px] text-gray-300">
                {(
                  [
                    ["shots / score", `${targetStats.count} / ${targetStats.totalScore} (${targetStats.avgScore.toFixed(1)} avg)`],
                    ["MPI offset", `${(targetStats.offset * 100).toFixed(1)}% @ ${targetStats.offsetClock}`],
                    ["mean radius", `${(targetStats.meanRadius * 100).toFixed(1)}%`],
                    ["radial σ", `${(targetStats.radialSd * 100).toFixed(1)}%`],
                    [
                      "σx / σy",
                      `${(targetStats.sdX * 100).toFixed(1)}% / ${(targetStats.sdY * 100).toFixed(1)}%${
                        targetStats.sdY > targetStats.sdX * 1.5
                          ? " (vertical stringing)"
                          : targetStats.sdX > targetStats.sdY * 1.5
                            ? " (horizontal stringing)"
                            : ""
                      }`,
                    ],
                    ["extreme spread", `${(targetStats.extremeSpread * 100).toFixed(1)}%`],
                    ["CEP50", `${(targetStats.cep50 * 100).toFixed(1)}%`],
                    ["avg split", targetStats.avgSplitMs !== null ? `${(targetStats.avgSplitMs / 1000).toFixed(2)}s` : "—"],
                    ["best split", targetStats.bestSplitMs !== null ? `${(targetStats.bestSplitMs / 1000).toFixed(2)}s` : "—"],
                    [
                      "string / HF",
                      `${targetStats.stringSec !== null ? `${targetStats.stringSec.toFixed(1)}s` : "—"}${
                        targetStats.hitFactor !== null ? ` / ${targetStats.hitFactor.toFixed(2)}` : ""
                      }`,
                    ],
                    [
                      "groups",
                      targetStats.clusters.k === 1
                        ? "one group"
                        : `TWO groups (${targetStats.clusters.centers.map((c) => c.count).join("+")} shots, sep ${targetStats.clusters.separationRatio.toFixed(1)}×)`,
                    ],
                  ] as const
                ).map(([label, value]) => (
                  <div key={label} className="contents">
                    <span
                      className="cursor-help text-gray-500 underline decoration-dotted decoration-gray-700 underline-offset-2"
                      title={STAT_HELP[label]}
                    >
                      {label}
                    </span>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
              {showStatHelp ? (
                <dl className="space-y-1.5 border-t border-gray-800 pt-2 text-[10px] leading-relaxed text-gray-400">
                  {Object.entries(STAT_HELP).map(([term, definition]) => (
                    <div key={term}>
                      <dt className="font-semibold text-gray-300">{term}</dt>
                      <dd>{definition}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="text-[10px] text-gray-600">Hover a label (or tap ⓘ) for definitions and derivations.</p>
              )}
            </div>
          ) : null}

          <div className="space-y-2 rounded-lg border border-gray-800 p-3">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-300">Session stats</p>
              <button
                type="button"
                onClick={() => setSession(EMPTY_SESSION)}
                className="rounded px-2 py-0.5 text-[11px] text-gray-500 transition hover:bg-neutral-900 hover:text-gray-300"
              >
                reset
              </button>
            </div>
            <div className="grid grid-cols-3 gap-x-2 gap-y-1 font-mono text-[11px] text-gray-300">
              <span className="text-gray-500">shots</span>
              <span className="text-gray-500">hits</span>
              <span className="text-gray-500">misses</span>
              <span>{session.shots}</span>
              <span className="text-emerald-300">{session.hits}</span>
              <span className="text-rose-300">{session.misses}</span>
              <span className="text-gray-500">accuracy</span>
              <span className="text-gray-500">avg react</span>
              <span className="text-gray-500">streak</span>
              <span>{session.called > 0 ? `${Math.round((session.hits / session.called) * 100)}%` : "—"}</span>
              <span>
                {session.reactionCount > 0 ? `${(session.reactionSumMs / session.reactionCount / 1000).toFixed(2)}s` : "—"}
              </span>
              <span>
                {session.streak}
                <span className="text-gray-500"> / best {session.bestStreak}</span>
              </span>
            </div>
            <p className="text-[10px] text-gray-600">
              {session.drills} drill{session.drills === 1 ? "" : "s"} completed · accuracy counts called shots and
              timeouts; free practice counts shots only.
            </p>
            <p className="text-[11px] leading-relaxed text-gray-500">
              Free practice: every trigger click logs a shot at your aim point. The called drill announces zones and
              scores you.
            </p>
          </div>

          {shots.length > 0 ? (
            <div className="rounded-lg border border-gray-800 p-3">
              <p className="text-xs font-semibold text-gray-300">Shots</p>
              <ul className="mt-2 space-y-1 text-xs">
                {shots.map((shot) => (
                  <li key={shot.n} className="flex items-center gap-2">
                    <span className="font-mono text-gray-500">#{shot.n}</span>
                    <span className={shot.correct === false ? "text-rose-300" : shot.correct ? "text-emerald-300" : "text-gray-300"}>
                      {shot.zoneId ? `zone ${shot.zoneId}` : "off target"}
                      {shot.correct === false && shot.expectedZoneId ? ` (called ${shot.expectedZoneId})` : ""}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </aside>
      </div>
    </main>
  );
}
