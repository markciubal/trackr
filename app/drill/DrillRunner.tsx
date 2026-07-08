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
  generateTimedSchedule,
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

// Shape returned by GET /api/targets?id= — enough to rebuild the drill and cache
// a usable local target.
type CatalogTarget = {
  name: string | null;
  unit: "mm" | "cm" | "in";
  widthValue: number;
  heightValue: number;
  qrSizeValue: number;
  drillRecipe: string | null;
  drillPaletteVersion: number | null;
};

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
  // Timed mode: approximate drill length in seconds. The real span is jittered
  // ±25% per run and the callout moments inside it are random.
  const [timespanSec, setTimespanSec] = useState(30);
  const [drill, setDrill] = useState<DrillState | null>(null);
  const [phase, setPhase] = useState<Phase>("setup");
  const [feedback, setFeedback] = useState<ZoneFeedback>({});
  const [scenarioTargets, setScenarioTargets] = useState<TargetInfo[]>([]);
  // How far the announcer has gotten (timed mode): hits past this index wait
  // for their callout, and the banner hides not-yet-called steps.
  const [announcedIndex, setAnnouncedIndex] = useState(-1);
  const announcedIndexRef = useRef(-1);
  const drillRef = useRef<DrillState | null>(null);
  // The active timed run. Setting this arms the callout timers (see the effect
  // below); clearing it cancels them.
  const [timedRun, setTimedRun] = useState<{
    steps: DrillStep[];
    calloutAtMs: number[];
    endAtMs: number;
  } | null>(null);
  const stepStartRef = useRef<number>(0);
  const feedbackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setScenarioTargets(listTargets().filter((target) => (target.zones?.length ?? 0) > 0));

    // Deep-link: /drill?id=XXXX loads that target's zones (the designer's "Test
    // drill" and scanned QRs). Resolution order: (1) this device saved it, (2) a
    // recipe rode along inline as ?z=, (3) foreign device with only the id —
    // fetch the recipe from the catalog once and cache it for offline reuse.
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    const rawZ = params.get("z");
    // Palette version for inline recipes (self-contained QRs carry it as ?pv=).
    const rawPv = Number(params.get("pv"));
    const paletteVersion = Number.isFinite(rawPv) && rawPv > 0 ? rawPv : undefined;

    const apply = (linked: ScenarioZone[], seedValue: number | null) => {
      if (!linked.length) return;
      setZones(linked);
      setZoneCount(linked.length);
      setLength(linked.length);
      setAttrs(attributesFromZones(linked));
      setSeed(seedValue);
    };

    // 1) Saved locally → exact zones (works fully offline).
    const stored = id ? getTarget(id) : null;
    if (stored?.zones?.length) {
      apply(stored.zones, stored.drillSeed ?? null);
      return;
    }
    // 2) Recipe carried inline (self-contained QR or the landing-page link).
    const recipe = decodeRecipe(rawZ);
    const inline = recipe ? zonesFromRecipe(recipe, paletteVersion) : decodeZones(rawZ, paletteVersion);
    if (inline?.length) {
      apply(inline, recipe?.seed ?? null);
      return;
    }
    // 3) Id-only → resolve from the catalog, then cache the whole target.
    if (!id) return;
    let cancelled = false;
    fetch(`/api/targets?id=${encodeURIComponent(id)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: CatalogTarget | null) => {
        if (cancelled || !data?.drillRecipe) return;
        const fetched = decodeRecipe(data.drillRecipe);
        const zones = fetched
          ? zonesFromRecipe(fetched, data.drillPaletteVersion ?? undefined)
          : null;
        if (!zones?.length) return;
        apply(zones, fetched?.seed ?? null);
        saveTarget(
          createTargetInfo({
            id,
            name: data.name ?? "Scanned target",
            unit: data.unit,
            widthValue: data.widthValue,
            heightValue: data.heightValue,
            qrSizeValue: data.qrSizeValue,
            scoringId: "drill",
            zones,
            drillSeed: fetched?.seed ?? undefined,
          }),
        );
        setScenarioTargets(listTargets().filter((target) => (target.zones?.length ?? 0) > 0));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
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
    setTimedRun(null);
    drillRef.current = null;
    setDrill(null);
    setPhase("setup");
    setFeedback({});
  };

  // Timed mode allows repeats, so its sequence can outrun the zone count; the
  // other modes cap the length at one call per zone.
  const chooseMode = (next: CalloutMode) => {
    setMode(next);
    if (next !== "timed") setLength((current) => Math.min(current, zones.length));
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

  // Timed-callout engine. Arms one timer per callout plus a final timeout.
  // Living in an effect keeps every timer cancelable (restart, load, unmount →
  // cleanup) and is where timer-driven work belongs. Each callout first scores
  // any still-unanswered earlier call as a miss, then announces its step.
  useEffect(() => {
    if (!timedRun) return;
    const { steps, calloutAtMs, endAtMs } = timedRun;

    const advanceMisses = (uptoIndex: number): DrillState | null => {
      let state = drillRef.current;
      if (!state || state.status !== "running") return null;
      while (state.index < uptoIndex && state.status === "running") {
        state = registerHit(state, null, Math.max(0, Date.now() - stepStartRef.current));
      }
      drillRef.current = state;
      setDrill(state);
      return state;
    };

    const fireCallout = (index: number) => {
      const state = advanceMisses(index);
      if (!state) return;
      if (state.status !== "running") {
        setPhase("done");
        setTimedRun(null);
        speak("Done");
        return;
      }
      announcedIndexRef.current = index;
      setAnnouncedIndex(index);
      const step = steps[index];
      if (step) {
        speak(step.spoken, {
          onEnd: () => {
            stepStartRef.current = Date.now();
          },
        });
        // Fallback in case speech is unavailable (onEnd still fires synchronously).
        stepStartRef.current = Date.now();
      }
    };

    const finish = () => {
      const state = advanceMisses(Number.MAX_SAFE_INTEGER);
      if (!state) return;
      setPhase("done");
      setTimedRun(null);
      speak("Done");
    };

    const timers: number[] = [];
    for (let index = 0; index < calloutAtMs.length; index += 1) {
      timers.push(window.setTimeout(() => fireCallout(index), calloutAtMs[index]));
    }
    timers.push(window.setTimeout(finish, endAtMs));
    return () => {
      for (const timer of timers) window.clearTimeout(timer);
    };
  }, [timedRun]);

  const start = () => {
    cancelSpeech();
    setTimedRun(null);
    setFeedback({});
    announcedIndexRef.current = -1;
    setAnnouncedIndex(-1);
    const steps = generateCallSequence(zones, length, mode === "timed");
    const state = startDrill(steps);
    drillRef.current = state;
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
    } else if (mode === "timed") {
      // Timed: callouts land at random moments across a randomized window —
      // neither the rhythm nor the total length repeats run to run. A call the
      // shooter hasn't answered scores as a miss when the next one fires.
      // Arming timedRun starts the timers (see the timed-callout effect).
      setPhase("playing");
      const schedule = generateTimedSchedule(steps.length, timespanSec);
      speak("Standby", {
        onEnd: () => {
          stepStartRef.current = Date.now();
        },
      });
      setTimedRun({ steps, calloutAtMs: schedule.calloutAtMs, endAtMs: schedule.endAtMs });
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
    // Timed mode: the shooter can't answer a call that hasn't happened yet —
    // after answering the current call, further hits wait for the next one.
    if (mode === "timed" && drill.index > announcedIndexRef.current) return;
    const expected = currentStep(drill);
    if (!expected) return;
    const reactionMs = Math.max(0, Date.now() - stepStartRef.current);
    flash(zoneId, zoneId === expected.zoneId ? "correct" : "wrong");

    const nextState = registerHit(drill, zoneId, reactionMs);
    drillRef.current = nextState;
    setDrill(nextState);

    if (nextState.status === "done") {
      setTimedRun(null);
      setPhase("done");
      speak("Done");
      return;
    }
    if (mode === "reactive") announceStep(nextState.steps[nextState.index]);
  };

  const loadScenario = (target: TargetInfo) => {
    if (!target.zones?.length) return;
    cancelSpeech();
    setTimedRun(null);
    drillRef.current = null;
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
            ) : phase === "playing" && mode === "timed" && drill && drill.index > announcedIndex ? (
              // Timed mode between calls: the next step exists but hasn't been
              // announced yet — don't reveal it.
              <p className="text-lg font-semibold text-amber-200">Stand by…</p>
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
              <span className="text-xs text-gray-400">
                Sequence length: {length}
                {mode === "timed" && length > zones.length ? " (zones repeat)" : ""}
              </span>
              <input
                type="range"
                min={1}
                max={mode === "timed" ? Math.max(zones.length, 20) : zones.length}
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
                  onClick={() => chooseMode("sequence")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    mode === "sequence" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  Full sequence
                </button>
                <button
                  type="button"
                  onClick={() => chooseMode("reactive")}
                  className={`rounded-md border px-2 py-1.5 text-xs ${
                    mode === "reactive" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  One at a time
                </button>
                <button
                  type="button"
                  onClick={() => chooseMode("timed")}
                  className={`col-span-2 rounded-md border px-2 py-1.5 text-xs ${
                    mode === "timed" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  Random timing — calls at surprise moments
                </button>
              </div>
              {mode === "timed" ? (
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
