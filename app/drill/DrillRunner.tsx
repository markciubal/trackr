"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ScenarioBoard, type ZoneFeedback } from "@/app/components/scenario/ScenarioBoard";
import {
  type CalloutMode,
  type DrillState,
  type DrillStep,
  type ScenarioAttribute,
  type ScenarioZone,
  DEFAULT_SCENARIO_ATTRIBUTES,
  MAX_SCENARIO_ZONES,
  SCENARIO_ATTRIBUTE_OPTIONS,
  attributesFromZones,
  currentStep,
  decodeRecipe,
  decodeZones,
  generateCallSequence,
  generateScenarioZones,
  generateSeed,
  registerHit,
  scoreDrill,
  startDrill,
  zonesFromRecipe,
} from "@/app/lib/targets/scenario";
import { cancelSpeech, speak, speakSequence } from "@/app/lib/targets/speech";
import {
  createTargetInfo,
  getTarget,
  listTargets,
  saveTarget,
  type TargetInfo,
} from "@/app/lib/targets/store";

type Phase = "setup" | "announcing" | "playing" | "done";

export function DrillRunner() {
  // Seed the current zones came from (null = unknown, e.g. an old save); when
  // known it's stored with the scenario so printed QRs can carry just a recipe.
  const [seed, setSeed] = useState<number | null>(() => generateSeed());
  const [zones, setZones] = useState<ScenarioZone[]>(() =>
    generateScenarioZones(6, DEFAULT_SCENARIO_ATTRIBUTES, seed ?? undefined),
  );
  const [zoneCount, setZoneCount] = useState(6);
  const [attrs, setAttrs] = useState<ScenarioAttribute[]>(DEFAULT_SCENARIO_ATTRIBUTES);
  const [mode, setMode] = useState<CalloutMode>("sequence");
  const [length, setLength] = useState(6);
  const [drill, setDrill] = useState<DrillState | null>(null);
  const [phase, setPhase] = useState<Phase>("setup");
  const [feedback, setFeedback] = useState<ZoneFeedback>({});
  const [scenarioTargets, setScenarioTargets] = useState<TargetInfo[]>([]);
  const stepStartRef = useRef<number>(0);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setScenarioTargets(listTargets().filter((target) => (target.zones?.length ?? 0) > 0));
    // Deep-link: /drill?id=XXXX loads that target's zones (used by the designer's
    // "Test drill" and by scanned QRs). A scanned QR also carries the zones
    // inline (?z=...), so the drill works even if this device never saved the id.
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    const rawZ = params.get("z");
    const stored = id ? getTarget(id) : null;
    const recipe = stored?.zones?.length ? null : decodeRecipe(rawZ);
    const linked = stored?.zones?.length ? stored.zones : recipe ? zonesFromRecipe(recipe) : decodeZones(rawZ);
    if (linked?.length) {
      setZones(linked);
      setZoneCount(linked.length);
      setLength(linked.length);
      setAttrs(attributesFromZones(linked));
      setSeed(stored?.zones?.length ? (stored.drillSeed ?? null) : (recipe?.seed ?? null));
    }
  }, []);

  useEffect(() => {
    return () => {
      cancelSpeech();
      if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    };
  }, []);

  const step = drill ? currentStep(drill) : null;
  const score = useMemo(() => (drill ? scoreDrill(drill) : null), [drill]);

  const regenerate = (count = zoneCount, attributes = attrs) => {
    const nextSeed = generateSeed();
    setSeed(nextSeed);
    const next = generateScenarioZones(count, attributes, nextSeed);
    setZones(next);
    setLength((current) => Math.min(current, next.length));
    setDrill(null);
    setPhase("setup");
    setFeedback({});
  };

  // Toggle a zone attribute (shapes / colors / numbers / patterns) and rebuild.
  // At least one must stay on or zones can't be called out.
  const toggleAttr = (attr: ScenarioAttribute) => {
    const has = attrs.includes(attr);
    if (has && attrs.length === 1) return;
    const next = has ? attrs.filter((a) => a !== attr) : [...attrs, attr];
    setAttrs(next);
    regenerate(zoneCount, next);
  };

  const start = () => {
    cancelSpeech();
    setFeedback({});
    const steps = generateCallSequence(zones, length, false);
    const state = startDrill(steps);
    setDrill(state);

    if (mode === "sequence") {
      // Announce the whole list, then "Go", then start the clock.
      setPhase("announcing");
      speakSequence(
        steps.map((s) => s.spoken),
        {
          onEnd: () =>
            speak("Go", {
              onEnd: () => {
                stepStartRef.current = Date.now();
                setPhase("playing");
              },
            }),
        },
      );
    } else {
      // Reactive: call them out one at a time as the shooter advances.
      setPhase("playing");
      announceStep(steps[0]);
    }
  };

  const announceStep = (next: DrillStep | undefined) => {
    if (!next) return;
    speak(next.spoken, {
      onEnd: () => {
        stepStartRef.current = Date.now();
      },
    });
    // Fallback in case speech is unavailable (onEnd still fires synchronously).
    stepStartRef.current = Date.now();
  };

  const flash = (zoneId: string, kind: "correct" | "wrong") => {
    setFeedback({ [zoneId]: kind });
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    feedbackTimer.current = setTimeout(() => setFeedback({}), 450);
  };

  const handleHit = (zoneId: string) => {
    if (phase !== "playing" || !drill) return;
    const expected = currentStep(drill);
    if (!expected) return;
    const reactionMs = Math.max(0, Date.now() - stepStartRef.current);
    flash(zoneId, zoneId === expected.zoneId ? "correct" : "wrong");

    const nextState = registerHit(drill, zoneId, reactionMs);
    setDrill(nextState);

    if (nextState.status === "done") {
      setPhase("done");
      speak("Done");
      return;
    }
    if (mode === "reactive") announceStep(nextState.steps[nextState.index]);
  };

  const loadScenario = (target: TargetInfo) => {
    if (!target.zones?.length) return;
    cancelSpeech();
    setZones(target.zones);
    setZoneCount(target.zones.length);
    setLength((current) => Math.min(current, target.zones?.length ?? current));
    setAttrs(attributesFromZones(target.zones));
    setSeed(target.drillSeed ?? null);
    setDrill(null);
    setPhase("setup");
    setFeedback({});
  };

  const saveScenario = () => {
    const info = createTargetInfo({
      name: `Scenario ${zones.length}-zone`,
      unit: "in",
      widthValue: 18,
      heightValue: 18,
      qrSizeValue: 1.5,
      scoringId: "drill",
      zones,
      drillSeed: seed ?? undefined,
    });
    saveTarget(info);
    setScenarioTargets(listTargets().filter((target) => (target.zones?.length ?? 0) > 0));
  };

  const stepNumber = drill ? Math.min(drill.index + 1, drill.steps.length) : 0;

  return (
    <main className="min-h-screen bg-black px-4 py-8 text-white">
      <div className="mx-auto grid max-w-5xl gap-6 lg:grid-cols-[1fr_320px]">
        {/* Board + live prompt */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <Link href="/" className="text-sm text-sky-300 hover:underline">
              ← Trackr
            </Link>
            <Link href="/targets/new" className="text-sm text-sky-300 hover:underline">
              Design targets →
            </Link>
          </div>

          {/* Prompt banner */}
          <div className="flex min-h-[64px] items-center justify-between rounded-lg border border-gray-700 bg-neutral-950 px-4 py-3">
            {phase === "setup" ? (
              <p className="text-sm text-gray-400">Press Start. The app calls a sequence — shoot the zones in that order.</p>
            ) : phase === "announcing" ? (
              <p className="text-lg font-semibold text-amber-200">Listen…</p>
            ) : phase === "playing" && step ? (
              <div className="flex w-full items-center justify-between">
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">
                    Shoot {step.kind} · {stepNumber}/{drill?.steps.length}
                  </p>
                  <p className="text-3xl font-bold text-white">{step.label}</p>
                </div>
                {mode === "sequence" ? (
                  <ol className="hidden gap-1 text-xs text-gray-400 sm:flex">
                    {drill?.steps.map((s, i) => (
                      <li
                        key={i}
                        className={`rounded px-1.5 py-0.5 ${
                          i < (drill?.index ?? 0)
                            ? "bg-emerald-500/15 text-emerald-200"
                            : i === drill?.index
                              ? "bg-amber-500/20 text-amber-100"
                              : "bg-neutral-800"
                        }`}
                      >
                        {s.label}
                      </li>
                    ))}
                  </ol>
                ) : null}
              </div>
            ) : phase === "done" && score ? (
              <div className="flex w-full items-center justify-between">
                <p className="text-lg font-semibold text-white">
                  {score.correct}/{score.total} correct · {score.accuracyPct.toFixed(0)}%
                </p>
                <p className="text-sm text-gray-300">
                  {(score.totalReactionMs / 1000).toFixed(2)}s total · {Math.round(score.avgReactionMs)}ms avg
                </p>
              </div>
            ) : (
              <p className="text-sm text-gray-400">Ready.</p>
            )}
          </div>

          <ScenarioBoard
            zones={zones}
            feedback={feedback}
            interactive={phase === "playing"}
            onHitZone={handleHit}
          />
          <p className="text-[11px] text-gray-500">
            Tap/click a zone to register a hit. (This same engine accepts live shot-detection impacts — that wiring is
            the next step.)
          </p>
        </section>

        {/* Controls */}
        <aside className="space-y-4">
          <h1 className="text-xl font-semibold">Scenario drill</h1>

          <div className="space-y-3 rounded-lg border border-gray-800 p-3">
            <label className="block">
              <span className="text-xs text-gray-400">Zones: {zoneCount}</span>
              <input
                type="range"
                min={2}
                max={MAX_SCENARIO_ZONES}
                value={zoneCount}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setZoneCount(value);
                  regenerate(value);
                }}
                className="mt-1 w-full"
              />
            </label>

            <label className="block">
              <span className="text-xs text-gray-400">Sequence length: {length}</span>
              <input
                type="range"
                min={1}
                max={zones.length}
                value={length}
                onChange={(event) => setLength(Number(event.target.value))}
                className="mt-1 w-full"
              />
            </label>

            <div>
              <span className="text-xs text-gray-400">Identify zones by</span>
              <div className="mt-1 grid grid-cols-2 gap-1.5">
                {SCENARIO_ATTRIBUTE_OPTIONS.map((option) => {
                  const on = attrs.includes(option.id);
                  return (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => toggleAttr(option.id)}
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
                Printing on a black-and-white printer? Turn off Colors and use Patterns.
              </p>
            </div>

            <div>
              <span className="text-xs text-gray-400">Callout mode</span>
              <div className="mt-1 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setMode("sequence")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    mode === "sequence" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  Full sequence
                </button>
                <button
                  type="button"
                  onClick={() => setMode("reactive")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    mode === "reactive" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  One at a time
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-1">
              <button
                type="button"
                onClick={start}
                disabled={phase === "announcing"}
                className="flex-1 rounded-md border border-emerald-400/40 bg-emerald-500/15 px-3 py-2 text-sm font-medium text-emerald-100 transition hover:bg-emerald-500/25 disabled:opacity-50"
              >
                {phase === "playing" || phase === "done" ? "Restart" : "Start"}
              </button>
              <button
                type="button"
                onClick={() => regenerate()}
                className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800"
              >
                Shuffle
              </button>
              <button
                type="button"
                onClick={saveScenario}
                className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800"
              >
                Save
              </button>
            </div>
          </div>

          {scenarioTargets.length > 0 ? (
            <div className="rounded-lg border border-gray-800 p-3">
              <p className="text-xs font-semibold text-gray-300">Saved scenarios</p>
              <ul className="mt-2 space-y-1">
                {scenarioTargets.map((target) => (
                  <li key={target.id}>
                    <button
                      type="button"
                      onClick={() => loadScenario(target)}
                      className="flex w-full items-center gap-2 rounded-md border border-gray-800 px-2 py-1.5 text-left text-xs transition hover:bg-neutral-900"
                    >
                      <span className="font-mono text-[10px] text-amber-200">{target.id}</span>
                      <span className="truncate text-gray-200">{target.name}</span>
                      <span className="ml-auto text-gray-500">{target.zones?.length} zones</span>
                    </button>
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
