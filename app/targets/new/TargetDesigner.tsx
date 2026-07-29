"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import QRCode from "qrcode";
import { useEffect, useMemo, useState } from "react";
import {
  type LinearUnit,
  TARGET_PAYLOAD_VERSION,
  encodeDrillPayload,
  encodeTargetPayload,
  fromInches,
  generateTargetId,
  toInches,
} from "@/app/lib/targets/payload";
import { SCORING_OPTIONS, TARGET_PRESETS, findPreset, type ScoringId } from "@/app/lib/targets/specs";
import {
  createTargetInfo,
  listTargets,
  removeTarget,
  saveTarget,
  type TargetInfo,
} from "@/app/lib/targets/store";
import {
  DEFAULT_SCENARIO_ATTRIBUTES,
  MAX_SCENARIO_ZONES,
  SCENARIO_ATTRIBUTE_OPTIONS,
  SCENARIO_PALETTE_VERSION,
  attributesFromZones,
  encodeRecipe,
  generateScenarioZones,
  generateSeed,
  type ScenarioAttribute,
  type ScenarioZone,
} from "@/app/lib/targets/scenario";
import { ScenarioBoard } from "@/app/components/scenario/ScenarioBoard";

const UNITS: LinearUnit[] = ["in", "cm", "mm"];

export function TargetDesigner({ baseUrl }: { baseUrl: string }) {
  const [presetId, setPresetId] = useState(TARGET_PRESETS[0].id);
  const [name, setName] = useState(TARGET_PRESETS[0].name);
  const [unit, setUnit] = useState<LinearUnit>(TARGET_PRESETS[0].unit);
  const [widthValue, setWidthValue] = useState(TARGET_PRESETS[0].widthValue);
  const [heightValue, setHeightValue] = useState(TARGET_PRESETS[0].heightValue);
  const [qrSizeValue, setQrSizeValue] = useState(TARGET_PRESETS[0].qrSizeValue);
  const [scoringId, setScoringId] = useState<ScoringId>(TARGET_PRESETS[0].scoringId);
  // Random client-only short code. Lazy-init (not an effect) to avoid cascading
  // renders; the few spots that render it during SSR get suppressHydrationWarning.
  const [targetId, setTargetId] = useState<string>(() => generateTargetId());
  const [qrSvg, setQrSvg] = useState("");
  // Locally-saved targets (keyed by their random id), loaded client-side.
  // Saving is purely a device-local convenience — the QR itself carries
  // everything a scan needs, so no account is ever required.
  const [savedTargets, setSavedTargets] = useState<TargetInfo[]>([]);
  const [localSaved, setLocalSaved] = useState(false);
  // Scenario ("drill") mode: scoring "drill" carries shape/color/number/pattern
  // zones, and they ride along inside the QR payload (z param).
  const [scenarioCount, setScenarioCount] = useState(6);
  // Seed the zones were generated from. When set, the QR carries just the tiny
  // {count, attrs, seed} recipe; null (e.g. zones loaded from an old save)
  // falls back to encoding the full zone list. Drill is the default scoring, so
  // a zone set is seeded up front (the live QR needs one immediately).
  const [scenarioSeed, setScenarioSeed] = useState<number | null>(() => generateSeed());
  const [scenarioZones, setScenarioZones] = useState<ScenarioZone[]>(() =>
    generateScenarioZones(6, DEFAULT_SCENARIO_ATTRIBUTES, scenarioSeed ?? undefined),
  );
  // Which attributes identify zones — e.g. B&W printers turn colors off and
  // lean on patterns/numbers instead.
  const [scenarioAttrs, setScenarioAttrs] = useState<ScenarioAttribute[]>(DEFAULT_SCENARIO_ATTRIBUTES);
  // Reduced-scale practice: shrink the target so its sight picture at a short
  // practice range matches the full target at the real (simulated) range.
  const [simDistanceYd, setSimDistanceYd] = useState(25);
  const [practiceDistanceYd, setPracticeDistanceYd] = useState(7);
  const router = useRouter();

  const isScenario = scoringId === "drill";

  const round2 = (value: number) => Math.round(value * 100) / 100;
  const scaleFactor =
    simDistanceYd > 0 && practiceDistanceYd > 0 ? practiceDistanceYd / simDistanceYd : 1;

  const applyDistanceScale = () => {
    if (!(scaleFactor > 0) || scaleFactor === 1) return;
    setWidthValue((value) => round2(value * scaleFactor));
    setHeightValue((value) => round2(value * scaleFactor));
    // Keep the QR scannable even on tiny targets.
    setQrSizeValue((value) => Math.max(round2(value * scaleFactor), 0.5));
    setLocalSaved(false);
  };

  useEffect(() => {
    // Client-only read of localStorage; intentional one-time load in an effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSavedTargets(listTargets());
  }, []);

  // Local targets, newest first.
  const allTargets = useMemo(
    () => [...savedTargets].sort((a, b) => b.createdAt - a.createdAt),
    [savedTargets],
  );

  // The QR is fully self-contained and LIVE: it always encodes the current
  // design. Drill targets encode a direct /drill URL carrying the zone recipe +
  // physical size; other targets encode /t/{id} with the calibration inline.
  // Nothing needs to be saved anywhere for a print to work.
  const payloadUrl = useMemo(() => {
    if (isScenario) {
      if (scenarioSeed === null || scenarioZones.length === 0) return "";
      const recipe = encodeRecipe({
        count: scenarioZones.length,
        attributes: scenarioAttrs,
        seed: scenarioSeed,
      });
      return encodeDrillPayload(
        { recipe, paletteVersion: SCENARIO_PALETTE_VERSION, unit, widthValue, heightValue, qrSizeValue },
        baseUrl,
      );
    }
    if (!targetId) return "";
    return encodeTargetPayload(
      { id: targetId, unit, widthValue, heightValue, qrSizeValue, scoringId, version: TARGET_PAYLOAD_VERSION },
      baseUrl,
    );
  }, [
    isScenario,
    scenarioSeed,
    scenarioZones,
    scenarioAttrs,
    unit,
    widthValue,
    heightValue,
    qrSizeValue,
    scoringId,
    targetId,
    baseUrl,
  ]);

  useEffect(() => {
    if (!payloadUrl) return;
    let cancelled = false;
    // ECC "L": the URL now carries the full payload (byte mode), so the lowest
    // error-correction level keeps the module count down — bigger modules scan
    // from farther away, and clean prints don't need M's damage tolerance.
    QRCode.toString(payloadUrl, { type: "svg", margin: 0, errorCorrectionLevel: "L" })
      .then((svg) => {
        if (!cancelled) setQrSvg(svg);
      })
      .catch(() => {
        if (!cancelled) setQrSvg("");
      });
    return () => {
      cancelled = true;
    };
  }, [payloadUrl]);

  // Generate zones from a fresh seed so the QR can carry the recipe.
  const rollZones = (count: number, attrs: ScenarioAttribute[]) => {
    const seed = generateSeed();
    setScenarioSeed(seed);
    setScenarioZones(generateScenarioZones(count, attrs, seed));
  };

  // Switch scoring; entering drill mode seeds a zone set if there isn't one.
  const changeScoring = (id: ScoringId) => {
    setScoringId(id);
    setLocalSaved(false);
    if (id === "drill" && scenarioZones.length === 0) {
      rollZones(scenarioCount, scenarioAttrs);
    }
  };

  const applyPreset = (id: string) => {
    const preset = findPreset(id);
    if (!preset) return;
    setPresetId(id);
    setName(preset.name);
    setUnit(preset.unit);
    setWidthValue(preset.widthValue);
    setHeightValue(preset.heightValue);
    setQrSizeValue(preset.qrSizeValue);
    changeScoring(preset.scoringId);
  };

  const currentTargetInfo = () =>
    createTargetInfo({
      id: targetId,
      name,
      unit,
      widthValue,
      heightValue,
      qrSizeValue,
      scoringId,
      zones: isScenario ? scenarioZones : undefined,
      drillSeed: isScenario ? (scenarioSeed ?? undefined) : undefined,
    });

  const loadTarget = (target: TargetInfo) => {
    setTargetId(target.id);
    setName(target.name);
    setUnit(target.unit);
    setWidthValue(target.widthValue);
    setHeightValue(target.heightValue);
    setQrSizeValue(target.qrSizeValue);
    if (target.zones?.length) {
      // Older saves kept zones alongside scoring "none"; treat both as drill.
      setScoringId("drill");
      setScenarioZones(target.zones);
      setScenarioSeed(target.drillSeed ?? null);
      setScenarioCount(target.zones.length);
      setScenarioAttrs(attributesFromZones(target.zones));
    } else {
      changeScoring(target.scoringId);
    }
    setLocalSaved(false);
  };

  const regenerateZones = (count = scenarioCount, attrs = scenarioAttrs) => {
    rollZones(count, attrs);
    setLocalSaved(false);
  };

  // Toggle a zone attribute and rebuild the zones with the new mix. At least
  // one attribute must stay on, otherwise zones can't be called out.
  const toggleScenarioAttr = (attr: ScenarioAttribute) => {
    const has = scenarioAttrs.includes(attr);
    if (has && scenarioAttrs.length === 1) return;
    const next = has ? scenarioAttrs.filter((a) => a !== attr) : [...scenarioAttrs, attr];
    setScenarioAttrs(next);
    regenerateZones(scenarioCount, next);
  };

  // Jump straight into the live drill, carrying the recipe inline — the same
  // self-contained form the printed QR uses (also saved locally for convenience).
  const testDrill = () => {
    saveTarget(currentTargetInfo());
    setSavedTargets(listTargets());
    if (scenarioSeed !== null && scenarioZones.length > 0) {
      const recipe = encodeRecipe({
        count: scenarioZones.length,
        attributes: scenarioAttrs,
        seed: scenarioSeed,
      });
      router.push(`/drill?z=${encodeURIComponent(recipe)}&pv=${SCENARIO_PALETTE_VERSION}`);
    } else {
      router.push(`/drill?id=${encodeURIComponent(targetId)}`);
    }
  };

  const deleteTarget = (id: string) => {
    removeTarget(id);
    setSavedTargets(listTargets());
  };

  // Save the target to this device (localStorage) — a convenience for reloading
  // designs later. Prints never depend on it: the QR is self-contained.
  const save = () => {
    saveTarget(currentTargetInfo());
    setSavedTargets(listTargets());
    setLocalSaved(true);
  };

  // Physical → on-screen preview scale (true size is preserved for printing).
  const widthInches = toInches(widthValue || 1, unit);
  const heightInches = toInches(heightValue || 1, unit);
  const sheetPxWidth = Math.max(1, widthInches * 96);
  const sheetPxHeight = Math.max(1, heightInches * 96);
  const previewScale = Math.min(1, 460 / sheetPxWidth);

  const numberInput =
    "w-full rounded-md border border-gray-700 bg-black px-2 py-1.5 text-sm outline-none ring-sky-400/40 focus:ring-1";

  return (
    <main className="min-h-screen bg-black px-4 py-8 text-white">
      <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[320px_1fr]">
        {/* Controls (hidden when printing) */}
        <section className="ts-controls space-y-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="text-sm text-sky-300 hover:underline">
              ← Trackr
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/drill" className="text-sm text-sky-300 hover:underline">
                Scenario drill →
              </Link>
              <span className="font-mono text-xs text-gray-500" suppressHydrationWarning>
                id {targetId}
              </span>
            </div>
          </div>
          <h1 className="text-xl font-semibold">Design a smart target</h1>

          <label className="block">
            <span className="text-xs text-gray-400">Preset</span>
            <select
              value={presetId}
              onChange={(event) => applyPreset(event.target.value)}
              className="mt-1 w-full rounded-md border border-gray-700 bg-black px-2 py-1.5 text-sm"
            >
              {TARGET_PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-xs text-gray-400">Name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} className={`mt-1 ${numberInput}`} />
          </label>

          <div className="grid grid-cols-3 gap-2">
            <label className="block">
              <span className="text-xs text-gray-400">Width</span>
              <input
                type="number"
                min={0}
                step="any"
                value={widthValue}
                onChange={(event) => {
                  setWidthValue(Number(event.target.value));
                  setLocalSaved(false);
                }}
                className={`mt-1 ${numberInput}`}
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400">Height</span>
              <input
                type="number"
                min={0}
                step="any"
                value={heightValue}
                onChange={(event) => {
                  setHeightValue(Number(event.target.value));
                  setLocalSaved(false);
                }}
                className={`mt-1 ${numberInput}`}
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400">Units</span>
              <select
                value={unit}
                onChange={(event) => {
                  setUnit(event.target.value as LinearUnit);
                  setLocalSaved(false);
                }}
                className="mt-1 w-full rounded-md border border-gray-700 bg-black px-2 py-1.5 text-sm"
              >
                {UNITS.map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="text-xs text-gray-400">QR size ({unit})</span>
              <input
                type="number"
                min={0}
                step="any"
                value={qrSizeValue}
                onChange={(event) => {
                  setQrSizeValue(Number(event.target.value));
                  setLocalSaved(false);
                }}
                className={`mt-1 ${numberInput}`}
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400">Scoring</span>
              <select
                value={scoringId}
                onChange={(event) => changeScoring(event.target.value as ScoringId)}
                className="mt-1 w-full rounded-md border border-gray-700 bg-black px-2 py-1.5 text-sm"
              >
                {SCORING_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="rounded-md border border-gray-800 p-2.5">
            <p className="text-xs font-semibold text-gray-200">Scale for practice distance</p>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <label className="block">
                <span className="text-xs text-gray-400">Simulated range (yd)</span>
                <input
                  type="number"
                  min={1}
                  step="any"
                  value={simDistanceYd}
                  onChange={(event) => setSimDistanceYd(Number(event.target.value))}
                  className={`mt-1 ${numberInput}`}
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-400">Practice range (yd)</span>
                <input
                  type="number"
                  min={1}
                  step="any"
                  value={practiceDistanceYd}
                  onChange={(event) => setPracticeDistanceYd(Number(event.target.value))}
                  className={`mt-1 ${numberInput}`}
                />
              </label>
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2">
              <p className="text-[11px] text-gray-400">
                ×{scaleFactor.toFixed(2)} → {round2(widthValue * scaleFactor)} × {round2(heightValue * scaleFactor)}{" "}
                {unit}
              </p>
              <button
                type="button"
                onClick={applyDistanceScale}
                className="rounded-md border border-sky-400/40 bg-sky-500/15 px-2.5 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-500/25"
              >
                Apply scale
              </button>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-gray-500">
              Shrinks the printed target so its sight picture at {practiceDistanceYd} yd matches the full target at{" "}
              {simDistanceYd} yd. The QR still calibrates real size.
            </p>
          </div>

          {isScenario ? (
            <div className="rounded-md border border-gray-800 p-2.5">
              <p className="text-xs font-semibold text-gray-200">Drill settings</p>
              <div className="mt-2.5 space-y-2">
                <label className="block">
                  <span className="text-xs text-gray-400">Zones: {scenarioCount}</span>
                  <input
                    type="range"
                    min={2}
                    max={MAX_SCENARIO_ZONES}
                    value={scenarioCount}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      setScenarioCount(value);
                      regenerateZones(value);
                    }}
                    className="mt-1 w-full"
                  />
                </label>
                <div>
                  <span className="text-xs text-gray-400">Identify zones by</span>
                  <div className="mt-1 grid grid-cols-2 gap-1.5">
                    {SCENARIO_ATTRIBUTE_OPTIONS.map((option) => {
                      const on = scenarioAttrs.includes(option.id);
                      return (
                        <button
                          key={option.id}
                          type="button"
                          onClick={() => toggleScenarioAttr(option.id)}
                          aria-pressed={on}
                          className={`rounded-md border px-2 py-1.5 text-xs transition ${
                            on
                              ? "border-sky-400/50 bg-sky-500/15 text-sky-100"
                              : "border-gray-700 text-gray-400 hover:bg-neutral-900"
                          }`}
                        >
                          {option.label}
                        </button>
                      );
                    })}
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-gray-500">
                    Black-and-white printer? Turn off <b>Colors</b> and use <b>Patterns</b> (striped, dotted,
                    checkered…) instead.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => regenerateZones()}
                    className="rounded-md border border-gray-600 px-2.5 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800"
                  >
                    Regenerate
                  </button>
                  <button
                    type="button"
                    onClick={testDrill}
                    className="rounded-md border border-sky-400/40 bg-sky-500/15 px-2.5 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-500/25"
                  >
                    Test drill →
                  </button>
                </div>
                <p className="text-[11px] leading-relaxed text-gray-500">
                  The zone layout is encoded in the QR itself — scanning the printed target rebuilds this exact
                  drill on any device, no account needed.
                </p>
              </div>
            </div>
          ) : null}

          <p className="rounded-md border border-amber-500/30 bg-amber-500/5 p-2 text-[11px] text-amber-100/90">
            When printing, choose <b>Actual size / 100%</b> (turn off &quot;Fit to page&quot;) so the QR prints at
            exactly {qrSizeValue} {unit} — that&apos;s what makes Trackr calibrate itself.
          </p>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => window.print()}
              className="rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
            >
              Print
            </button>
            <button
              type="button"
              onClick={() => {
                setTargetId(generateTargetId());
                setLocalSaved(false);
              }}
              className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800"
            >
              New id
            </button>
            <button
              type="button"
              onClick={save}
              className="rounded-md border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-100 transition hover:bg-emerald-500/20"
            >
              {localSaved ? "Saved ✓" : "Save to this device"}
            </button>
          </div>
          <p className="rounded-md border border-gray-800 bg-neutral-950 px-2 py-1.5 text-[11px] text-gray-400">
            The QR is self-contained and updates live as you edit — scanning a print opens the exact drill or
            calibration on any device. No account, no upload; saving only keeps a copy in this browser.
          </p>
          <p className="break-all font-mono text-[10px] text-gray-600" suppressHydrationWarning>
            {payloadUrl}
          </p>

          {allTargets.length > 0 ? (
            <div className="border-t border-gray-800 pt-3">
              <p className="text-xs font-semibold text-gray-300">My targets ({allTargets.length})</p>
              <ul className="mt-2 space-y-1">
                {allTargets.map((target) => (
                  <li
                    key={target.id}
                    className={`flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs ${
                      target.id === targetId ? "border-sky-400/50 bg-sky-500/10" : "border-gray-800"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => loadTarget(target)}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <span className="font-mono text-[10px] text-amber-200">{target.id}</span>
                      <span className="truncate text-gray-200">{target.name}</span>
                      <span className="ml-auto whitespace-nowrap text-gray-500">
                        {target.widthValue}×{target.heightValue}
                        {target.unit}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteTarget(target.id)}
                      aria-label={`Delete ${target.name}`}
                      className="text-gray-500 transition hover:text-rose-300"
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>

        {/* Printable preview (true size on print). The dynamic @page rule
            makes the PAPER exactly the sheet's physical size — the sheet
            fills page 1 completely and nothing can exist beyond it, which
            is what kills the trailing blank page. */}
        <style>{`@media print { @page { size: ${widthValue}${unit} ${heightValue}${unit}; margin: 0; } }`}</style>
        <section className="ts-print-area overflow-auto">
          {/* Repeats at the top of EVERY printed page (print-fixed elements
              render on each page) — full identification even on tiled
              multi-page prints. Hidden on screen. */}
          <div className="ts-print-header hidden">
            {name} · {widthValue}×{heightValue}
            {unit} · scoring: {scoringId} · QR {qrSizeValue}
            {unit} · Trackr target — print at 100% scale
          </div>
          <div
            style={{ width: sheetPxWidth * previewScale, height: sheetPxHeight * previewScale }}
            className="rounded-md border border-gray-800"
          >
            <div
              className="ts-sheet relative bg-white text-black"
              style={{
                width: `${widthValue}${unit}`,
                height: `${heightValue}${unit}`,
                transform: `scale(${previewScale})`,
                transformOrigin: "top left",
              }}
            >
              {isScenario ? (
                <>
                  <svg
                    viewBox={`0 0 ${widthValue} ${heightValue}`}
                    preserveAspectRatio="none"
                    className="absolute inset-0 h-full w-full"
                    aria-hidden="true"
                  >
                    <rect
                      x={0}
                      y={0}
                      width={widthValue}
                      height={heightValue}
                      fill="none"
                      stroke="#000"
                      strokeWidth={fromInches(0.02, unit)}
                    />
                  </svg>
                  <ScenarioBoard
                    zones={scenarioZones}
                    interactive={false}
                    bare
                    aspectRatio={widthValue > 0 && heightValue > 0 ? widthValue / heightValue : 1}
                    className="absolute inset-0 h-full w-full"
                  />
                </>
              ) : (
                <TargetArtwork
                  widthValue={widthValue}
                  heightValue={heightValue}
                  unit={unit}
                  scoringId={scoringId}
                  qrSizeValue={qrSizeValue}
                />
              )}
              <QrBlock
                qrSvg={qrSvg}
                qrSizeValue={qrSizeValue}
                unit={unit}
                label={`${name} · QR ${qrSizeValue}${unit}`}
              />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

// Dark gray (not pure black) so fresh bullet holes still change enough pixels
// for the frame-diff hit detector inside the filled areas.
const SILHOUETTE_FILL = "#3f3f3f";

// Uniform fit of an abstract shape box into the sheet, centered with a margin.
function fitShape(w: number, h: number, unit: LinearUnit, shapeW: number, shapeH: number) {
  const margin = fromInches(0.5, unit);
  const availW = Math.max(0.01, w - margin * 2);
  const availH = Math.max(0.01, h - margin * 2);
  const s = Math.min(availW / shapeW, availH / shapeH);
  return { s, x: (w - shapeW * s) / 2, y: (h - shapeH * s) / 2 };
}

// B-27-style half silhouette in a 100 × 140 box, rings centered on the chest.
function SilhouetteB27({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 100, 140);
  const rings = [
    { rx: 27, ry: 34, label: "8", labelY: 113 },
    { rx: 19, ry: 25, label: "9", labelY: 104 },
    { rx: 11.5, ry: 15, label: "10", labelY: 94.5 },
    { rx: 5, ry: 6.5, label: "X", labelY: 83.5 },
  ];
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path
        d="M50 4 C61 4 69 12 69 22 C69 28 66 33 61 36 C77 41 87 53 89 70 L92 140 L8 140 L11 70 C13 53 23 41 39 36 C34 33 31 28 31 22 C31 12 39 4 50 4 Z"
        fill={SILHOUETTE_FILL}
      />
      {rings.map((ring) => (
        <g key={ring.label}>
          <ellipse cx={50} cy={82} rx={ring.rx} ry={ring.ry} fill="none" stroke="#fff" strokeWidth={0.7} />
          <text x={50} y={ring.labelY} textAnchor="middle" fontSize={4} fill="#fff">
            {ring.label}
          </text>
        </g>
      ))}
    </g>
  );
}

// USPSA/IPSC-style practice silhouette in its real 18 × 30 in proportions,
// scaled to fit the sheet. Zone lines are approximate (practice, not match).
function SilhouetteUspsa({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 18, 30);
  const zoneLabel = { fontSize: 1, fill: "#777" } as const;
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path d="M6 0 H12 V6 L18 10 V30 H0 V10 L6 6 Z" fill="none" stroke="#000" strokeWidth={0.08} />
      {/* Perforation-style zone lines */}
      <line x1={6} y1={4} x2={12} y2={4} stroke="#555" strokeWidth={0.04} strokeDasharray="0.35 0.25" />
      <rect x={6} y={13} width={6} height={11} fill="none" stroke="#555" strokeWidth={0.04} strokeDasharray="0.35 0.25" />
      <rect x={3} y={9} width={12} height={18} fill="none" stroke="#555" strokeWidth={0.04} strokeDasharray="0.35 0.25" />
      <text x={9} y={19} textAnchor="middle" {...zoneLabel}>
        A
      </text>
      <text x={9} y={26.4} textAnchor="middle" {...zoneLabel}>
        C
      </text>
      <text x={9} y={29.2} textAnchor="middle" {...zoneLabel}>
        D
      </text>
      <text x={9} y={3} textAnchor="middle" {...zoneLabel}>
        A
      </text>
    </g>
  );
}

// USMC "Able"-style head-and-shoulders silhouette in a 60 × 60 box.
function SilhouetteUsmc({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 60, 60);
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path
        d="M0 60 L0 44 C0 39 4 36 9 36 L21 36 C16.5 31.5 14 25.6 14 19 C14 9.6 21.2 2 30 2 C38.8 2 46 9.6 46 19 C46 25.6 43.5 31.5 39 36 L51 36 C56 36 60 39 60 44 L60 60 Z"
        fill={SILHOUETTE_FILL}
      />
      {/* White aiming cross at center mass */}
      <line x1={26} y1={46} x2={34} y2={46} stroke="#fff" strokeWidth={0.7} />
      <line x1={30} y1={42} x2={30} y2={50} stroke="#fff" strokeWidth={0.7} />
    </g>
  );
}

// USMC Table-1 "A" target: concentric 36 / 24 / 12-inch rings with a 12-inch
// filled center, on a 48 × 72 in (4 × 6 ft) backer.
function SilhouetteUsmcA({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 48, 72);
  const cx = 24;
  const cy = 36;
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <circle cx={cx} cy={cy} r={18} fill="none" stroke="#000" strokeWidth={0.45} />
      <circle cx={cx} cy={cy} r={12} fill="none" stroke="#000" strokeWidth={0.45} />
      <circle cx={cx} cy={cy} r={6} fill={SILHOUETTE_FILL} />
    </g>
  );
}

// USMC Table-1 "D" target: tombstone threat silhouette — 34-inch shoulders,
// 26-inch waist, vertical bands of 8/19/14 in with a 6-inch bottom margin, on a
// 72-in-wide backer (height not dimensioned in the spec; 54 in assumed).
function SilhouetteUsmcD({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 72, 54);
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path
        d="M36 7 C41 7 44 10 44 14 C44 15 43 15.5 42 15.8 C49 16.5 53 18 53 23 L49 48 L23 48 L19 23 C19 18 23 16.5 30 15.8 C29 15.5 28 15 28 14 C28 10 31 7 36 7 Z"
        fill={SILHOUETTE_FILL}
      />
    </g>
  );
}

// USMC Table-1 "B" modified target: a 40-inch bottle threat (20-inch neck,
// 40-inch base) inside a 60-inch scoring ring, on a 72-in backer.
function SilhouetteUsmcBMod({ w, h, unit }: { w: number; h: number; unit: LinearUnit }) {
  const { s, x, y } = fitShape(w, h, unit, 72, 72);
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <circle cx={36} cy={36} r={30} fill="none" stroke="#000" strokeWidth={0.5} />
      <path
        d="M26 16 L46 16 C50 16 56 20 56 28 L56 56 L16 56 L16 28 C16 20 22 16 26 16 Z"
        fill={SILHOUETTE_FILL}
      />
    </g>
  );
}

// Numbered 2-inch dots laid out on the sheet, skipping the printed QR corner.
function computeDots(w: number, h: number, unit: LinearUnit, qrSizeValue: number) {
  const r = fromInches(1, unit);
  const gap = fromInches(0.5, unit);
  const margin = fromInches(0.5, unit);
  const pitch = r * 2 + gap;
  const cols = Math.max(1, Math.floor((w - margin * 2 + gap) / pitch));
  const rows = Math.max(1, Math.floor((h - margin * 2 + gap) / pitch));
  const x0 = (w - (cols * r * 2 + (cols - 1) * gap)) / 2 + r;
  const y0 = (h - (rows * r * 2 + (rows - 1) * gap)) / 2 + r;

  // Footprint of the QR block (code + quiet zone + label) in the bottom-left.
  const qrMargin = fromInches(0.4, unit);
  const qrSide = qrSizeValue * 1.3;
  const qrRect = {
    x1: qrMargin,
    y1: h - qrMargin - qrSide - fromInches(0.3, unit),
    x2: qrMargin + qrSide,
    y2: h - qrMargin,
  };

  const dots: { x: number; y: number }[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x = x0 + col * pitch;
      const y = y0 + row * pitch;
      const nearestX = Math.min(Math.max(x, qrRect.x1), qrRect.x2);
      const nearestY = Math.min(Math.max(y, qrRect.y1), qrRect.y2);
      if (Math.hypot(x - nearestX, y - nearestY) < r) continue; // under the QR block
      dots.push({ x, y });
    }
  }
  return { dots, r };
}

function TargetArtwork({
  widthValue,
  heightValue,
  unit,
  scoringId,
  qrSizeValue,
}: {
  widthValue: number;
  heightValue: number;
  unit: LinearUnit;
  scoringId: ScoringId;
  qrSizeValue: number;
}) {
  const w = widthValue;
  const h = heightValue;
  const border = fromInches(0.02, unit);
  const thin = fromInches(0.012, unit);
  const cx = w / 2;
  const cy = h / 2;

  const rings: number[] = [];
  if (scoringId === "bullseye-5") {
    const outer = (Math.min(w, h) / 2) * 0.92;
    for (let i = 0; i < 5; i += 1) rings.push(outer * (1 - i * 0.2));
  }

  const gridLines: { x1: number; y1: number; x2: number; y2: number; strong: boolean }[] = [];
  if (scoringId === "grid-1in") {
    const step = fromInches(1, unit);
    for (let x = 0; x <= w + 1e-6; x += step) {
      gridLines.push({ x1: x, y1: 0, x2: x, y2: h, strong: Math.abs(x - cx) < step / 2 });
    }
    for (let y = 0; y <= h + 1e-6; y += step) {
      gridLines.push({ x1: 0, y1: y, x2: w, y2: y, strong: Math.abs(y - cy) < step / 2 });
    }
  }

  const dotLayout = scoringId === "dots-2in" ? computeDots(w, h, unit, qrSizeValue) : null;

  // Crosshair + center dot only make sense on centered targets.
  const showCrosshair = scoringId === "none" || scoringId === "bullseye-5";
  const crossArm = Math.min(w, h) * 0.06;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="absolute inset-0 h-full w-full"
      aria-hidden="true"
    >
      <rect x={0} y={0} width={w} height={h} fill="none" stroke="#000" strokeWidth={border} />
      {gridLines.map((line, index) => (
        <line
          key={`g${index}`}
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          stroke={line.strong ? "#000" : "#9ca3af"}
          strokeWidth={line.strong ? thin * 1.6 : thin}
        />
      ))}
      {rings.map((r, index) => (
        <circle key={`r${index}`} cx={cx} cy={cy} r={r} fill="none" stroke="#000" strokeWidth={thin * 1.4} />
      ))}
      {scoringId === "silhouette-b27" ? <SilhouetteB27 w={w} h={h} unit={unit} /> : null}
      {scoringId === "uspsa" ? <SilhouetteUspsa w={w} h={h} unit={unit} /> : null}
      {scoringId === "usmc-able" ? <SilhouetteUsmc w={w} h={h} unit={unit} /> : null}
      {scoringId === "usmc-a" ? <SilhouetteUsmcA w={w} h={h} unit={unit} /> : null}
      {scoringId === "usmc-d" ? <SilhouetteUsmcD w={w} h={h} unit={unit} /> : null}
      {scoringId === "usmc-b-mod" ? <SilhouetteUsmcBMod w={w} h={h} unit={unit} /> : null}
      {dotLayout
        ? dotLayout.dots.map((dot, index) => (
            <g key={`d${index}`}>
              <circle cx={dot.x} cy={dot.y} r={dotLayout.r} fill={SILHOUETTE_FILL} />
              <text
                x={dot.x}
                y={dot.y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={dotLayout.r * 0.8}
                fill="#fff"
              >
                {index + 1}
              </text>
            </g>
          ))
        : null}
      {showCrosshair ? (
        <>
          <line x1={cx - crossArm} y1={cy} x2={cx + crossArm} y2={cy} stroke="#000" strokeWidth={thin} />
          <line x1={cx} y1={cy - crossArm} x2={cx} y2={cy + crossArm} stroke="#000" strokeWidth={thin} />
          <circle cx={cx} cy={cy} r={fromInches(0.05, unit)} fill="#000" />
        </>
      ) : null}
    </svg>
  );
}

function QrBlock({
  qrSvg,
  qrSizeValue,
  unit,
  label,
}: {
  qrSvg: string;
  qrSizeValue: number;
  unit: LinearUnit;
  label: string;
}) {
  // Quiet zone (white padding) around the code; the code area itself prints at
  // exactly qrSizeValue, which is the calibration reference the reader measures.
  const pad = qrSizeValue * 0.15;
  const margin = fromInches(0.4, unit);
  return (
    <div
      className="absolute"
      style={{ left: `${margin}${unit}`, bottom: `${margin}${unit}` }}
    >
      <div style={{ background: "#fff", padding: `${pad}${unit}` }}>
        <div
          style={{ width: `${qrSizeValue}${unit}`, height: `${qrSizeValue}${unit}` }}
          dangerouslySetInnerHTML={{ __html: qrSvg }}
        />
      </div>
      <div
        style={{ fontSize: `${fromInches(0.1, unit)}${unit}`, maxWidth: `${qrSizeValue + pad * 2}${unit}` }}
        className="mt-1 font-mono leading-tight text-black"
      >
        {label}
      </div>
    </div>
  );
}
