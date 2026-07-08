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
  randomZoneStep,
  registerHit,
  scoreDrill,
  startDrill,
  zonesFromRecipe,
} from "@/app/lib/targets/scenario";
import {
  DEFAULT_VOICE_SETTINGS,
  type VoiceSettings,
  canSpeak,
  cancelSpeech,
  listVoices,
  loadVoiceSettings,
  setVoiceSettings,
  speak,
  speakSequence,
} from "@/app/lib/targets/speech";
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
  // Random timing is the default — the most realistic training mode.
  const [mode, setMode] = useState<CalloutMode>("timed");
  const [length, setLength] = useState(6);
  // Timed mode: approximate drill length in seconds. The real span is jittered
  // ±25% per run and the callout moments inside it are random.
  const [timespanSec, setTimespanSec] = useState(30);
  const [drill, setDrill] = useState<DrillState | null>(null);
  const [phase, setPhase] = useState<Phase>("setup");
  const [feedback, setFeedback] = useState<ZoneFeedback>({});
  const [scenarioTargets, setScenarioTargets] = useState<TargetInfo[]>([]);
  // When zones came from a scanned/saved target, the layout is LOCKED: no
  // shuffling, zone-count, or attribute changes — the on-screen drill must keep
  // matching the physical print. Unlocking detaches from the target.
  const [lockedSource, setLockedSource] = useState<string | null>(null);
  // Voice settings popover: which system voice reads the callouts + rate/pitch/
  // volume. Persisted (localStorage) and applied to every utterance.
  const [showVoiceSettings, setShowVoiceSettings] = useState(false);
  const [voiceSettings, setVoiceSettingsState] = useState<VoiceSettings>(DEFAULT_VOICE_SETTINGS);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    // Load the persisted settings and the device's installed voices (the list
    // often arrives asynchronously via "voiceschanged").
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setVoiceSettingsState(loadVoiceSettings());
    const refreshVoices = () => setVoices(listVoices());
    refreshVoices();
    if (canSpeak()) window.speechSynthesis.addEventListener?.("voiceschanged", refreshVoices);
    return () => {
      if (canSpeak()) window.speechSynthesis.removeEventListener?.("voiceschanged", refreshVoices);
    };
  }, []);

  const updateVoiceSettings = (patch: Partial<VoiceSettings>) => {
    const next = { ...voiceSettings, ...patch };
    setVoiceSettingsState(next);
    setVoiceSettings(next); // applies to all speech + persists
  };

  const testVoice = () => {
    cancelSpeech();
    speak("Red. 12. Striped. Go!");
  };
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
  // Pause support: while paused, hits are ignored, speech stops, and (timed
  // mode) the pending callout timers are disarmed. timedElapsedMsRef tracks how
  // much of the timed schedule already ran, so resuming re-arms the remaining
  // callouts with their remaining delays; pausedAtRef shifts the reaction clock
  // so pause time never counts against the shooter.
  const [paused, setPaused] = useState(false);
  const pausedAtRef = useRef(0);
  const armedAtRef = useRef(0);
  const timedElapsedMsRef = useRef(0);
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

    const apply = (linked: ScenarioZone[], seedValue: number | null, sourceLabel: string) => {
      if (!linked.length) return;
      setZones(linked);
      setZoneCount(linked.length);
      setLength(linked.length);
      setAttrs(attributesFromZones(linked));
      setSeed(seedValue);
      // Scanned/linked target: lock the layout so it keeps matching the print.
      setLockedSource(sourceLabel);
    };

    // 1) Saved locally → exact zones (works fully offline).
    const stored = id ? getTarget(id) : null;
    if (stored?.zones?.length) {
      apply(stored.zones, stored.drillSeed ?? null, stored.name || id || "saved target");
      return;
    }
    // 2) Recipe carried inline (self-contained QR or the landing-page link).
    const recipe = decodeRecipe(rawZ);
    const inline = recipe ? zonesFromRecipe(recipe, paletteVersion) : decodeZones(rawZ, paletteVersion);
    if (inline?.length) {
      apply(inline, recipe?.seed ?? null, "scanned target");
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
        apply(zones, fetched?.seed ?? null, data.name ?? "scanned target");
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
    // Layout is locked to a scanned/saved target — no regeneration.
    if (lockedSource) return;
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
    if (lockedSource) return;
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
    if (!timedRun || paused) return;
    const { steps, calloutAtMs, endAtMs } = timedRun;
    // Re-arm relative to how much of the schedule already ran before a pause
    // (0 on a fresh start): each pending callout keeps its remaining delay.
    const alreadyMs = timedElapsedMsRef.current;
    armedAtRef.current = Date.now();

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
      const delayMs = calloutAtMs[index] - alreadyMs;
      if (delayMs <= 0) continue; // fired before the pause
      timers.push(window.setTimeout(() => fireCallout(index), delayMs));
    }
    timers.push(window.setTimeout(finish, Math.max(0, endAtMs - alreadyMs)));
    return () => {
      for (const timer of timers) window.clearTimeout(timer);
    };
  }, [timedRun, paused]);

  // Pause: freeze the drill. Speech stops, hits are ignored, and the reaction
  // clock (plus the timed schedule) stops counting until Resume.
  const pauseDrill = () => {
    if (phase !== "playing" || paused) return;
    cancelSpeech();
    pausedAtRef.current = Date.now();
    if (mode === "timed" && timedRun) {
      timedElapsedMsRef.current += Math.max(0, Date.now() - armedAtRef.current);
    }
    setPaused(true);
  };

  const resumeDrill = () => {
    if (!paused) return;
    setPaused(false);
    // Re-speak the pending call so the shooter doesn't resume blind.
    if (mode === "reactive" && drill) {
      const step = currentStep(drill);
      if (step) announceStep(step);
    } else if (mode === "timed" && drill && drill.index === announcedIndexRef.current) {
      const step = drill.steps[announcedIndexRef.current];
      if (step) speak(step.spoken);
    }
  };

  // On resume, restart the current step's reaction window — pause time never
  // counts against the shooter. (Effect scope so the clock read is legal.)
  useEffect(() => {
    if (paused || pausedAtRef.current === 0) return;
    pausedAtRef.current = 0;
    stepStartRef.current = Date.now();
  }, [paused]);

  // Stop: abandon the run entirely and go back to setup (zones untouched).
  const stopDrill = () => {
    cancelSpeech();
    setTimedRun(null);
    setPaused(false);
    timedElapsedMsRef.current = 0;
    pausedAtRef.current = 0;
    announcedIndexRef.current = -1;
    setAnnouncedIndex(-1);
    drillRef.current = null;
    setDrill(null);
    setPhase("setup");
    setFeedback({});
  };

  const start = () => {
    cancelSpeech();
    setTimedRun(null);
    setPaused(false);
    timedElapsedMsRef.current = 0;
    pausedAtRef.current = 0;
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
    if (phase !== "playing" || !drill || paused) return;
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
    // Loaded targets are (potentially printed) fixed layouts — lock them too.
    setLockedSource(target.name || target.id);
    setDrill(null);
    setPhase("setup");
    setFeedback({});
  };

  // Detach from the scanned/saved target: keep the current zones but allow
  // editing again (any change from here diverges from the print, deliberately).
  const unlockLayout = () => {
    setLockedSource(null);
  };

  // Ad-hoc callout: pressing any zone immediately speaks ONE random property of
  // it (among the enabled attributes) — as if that target were called on the
  // spot. Outside a run it replaces any pending speech; during a run it queues
  // behind the live callouts so they're never cut off.
  const handleZonePress = (zoneId: string) => {
    if (phase === "announcing" || paused) return;
    const zone = zones.find((candidate) => candidate.id === zoneId);
    if (!zone) return;
    if (phase !== "playing") cancelSpeech();
    speak(randomZoneStep(zone, attrs).spoken);
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
            ) : phase === "playing" && paused ? (
              <p className="text-lg font-semibold text-sky-200">⏸ Paused — hits are ignored until you resume.</p>
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
            onZonePress={handleZonePress}
          />
          <p className="text-[11px] text-gray-500">
            Tap/click a zone to register a hit. (This same engine accepts live shot-detection impacts — that wiring is
            the next step.)
          </p>
        </section>

        {/* Controls */}
        <aside className="space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h1 className="text-xl font-semibold">Scenario drill</h1>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowVoiceSettings((prev) => !prev)}
                aria-expanded={showVoiceSettings}
                aria-label="Voice settings"
                title="Voice settings"
                className="rounded-md border border-gray-700 px-2.5 py-1.5 text-sm text-gray-300 transition hover:bg-neutral-900"
              >
                ⚙️ Voice
              </button>
              {showVoiceSettings ? (
                <div className="absolute right-0 top-full z-20 mt-2 w-72 space-y-3 rounded-lg border border-gray-700 bg-neutral-950 p-3 shadow-xl">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold text-gray-200">Callout voice</p>
                    <button
                      type="button"
                      onClick={() => setShowVoiceSettings(false)}
                      aria-label="Close voice settings"
                      className="text-gray-500 transition hover:text-gray-200"
                    >
                      ✕
                    </button>
                  </div>
                  <label className="block">
                    <span className="text-xs text-gray-400">Voice</span>
                    <select
                      value={voiceSettings.voiceURI ?? ""}
                      onChange={(event) => updateVoiceSettings({ voiceURI: event.target.value || null })}
                      className="mt-1 w-full rounded-md border border-gray-700 bg-black px-2 py-1.5 text-xs"
                    >
                      <option value="">System default</option>
                      {voices.map((voice) => (
                        <option key={voice.voiceURI} value={voice.voiceURI}>
                          {voice.name} ({voice.lang}){voice.default ? " · default" : ""}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-[10px] leading-relaxed text-gray-500">
                      Male, female, and accent options come from the voices installed on this device.
                    </p>
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-400">Speed: {voiceSettings.rate.toFixed(2)}×</span>
                    <input
                      type="range"
                      min={0.5}
                      max={2}
                      step={0.05}
                      value={voiceSettings.rate}
                      onChange={(event) => updateVoiceSettings({ rate: Number(event.target.value) })}
                      className="mt-1 w-full"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-400">Pitch: {voiceSettings.pitch.toFixed(2)}</span>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={voiceSettings.pitch}
                      onChange={(event) => updateVoiceSettings({ pitch: Number(event.target.value) })}
                      className="mt-1 w-full"
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-400">Volume: {Math.round(voiceSettings.volume * 100)}%</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={voiceSettings.volume}
                      onChange={(event) => updateVoiceSettings({ volume: Number(event.target.value) })}
                      className="mt-1 w-full"
                    />
                  </label>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={testVoice}
                      className="flex-1 rounded-md border border-sky-400/40 bg-sky-500/15 px-2 py-1.5 text-xs font-medium text-sky-100 transition hover:bg-sky-500/25"
                    >
                      ▶ Test voice
                    </button>
                    <button
                      type="button"
                      onClick={() => updateVoiceSettings({ ...DEFAULT_VOICE_SETTINGS })}
                      className="rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          <div className="space-y-3 rounded-lg border border-gray-800 p-3">
            {lockedSource ? (
              <div className="flex items-start justify-between gap-2 rounded-md border border-amber-400/30 bg-amber-500/10 px-2 py-1.5">
                <p className="text-[11px] leading-relaxed text-amber-100">
                  Layout locked to <span className="font-semibold">{lockedSource}</span> so it keeps matching the
                  printed target — zones, attributes, and shuffle are disabled.
                </p>
                <button
                  type="button"
                  onClick={unlockLayout}
                  className="shrink-0 text-[11px] font-medium text-amber-200 underline hover:text-amber-100"
                >
                  Unlock
                </button>
              </div>
            ) : null}
            <label className="block">
              <span className="text-xs text-gray-400">
                Zones: {zoneCount}
                {lockedSource ? " (from target)" : ""}
              </span>
              <input
                type="range"
                min={2}
                max={MAX_SCENARIO_ZONES}
                value={zoneCount}
                disabled={Boolean(lockedSource)}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setZoneCount(value);
                  regenerate(value);
                }}
                className="mt-1 w-full disabled:opacity-40"
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
                      disabled={Boolean(lockedSource)}
                      className={`rounded-md border px-2 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 ${
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
                  onClick={() => chooseMode("timed")}
                  className={`col-span-2 rounded-md border px-2 py-1.5 text-xs ${
                    mode === "timed" ? "border-sky-400/50 bg-sky-500/15 text-sky-100" : "border-gray-700 text-gray-300"
                  }`}
                >
                  Random timing — calls at surprise moments
                </button>
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
              {phase === "playing" ? (
                <button
                  type="button"
                  onClick={paused ? resumeDrill : pauseDrill}
                  className="rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-2 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
                >
                  {paused ? "▶ Resume" : "⏸ Pause"}
                </button>
              ) : null}
              {phase === "playing" || phase === "announcing" ? (
                <button
                  type="button"
                  onClick={stopDrill}
                  className="rounded-md border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-100 transition hover:bg-rose-500/20"
                >
                  ⏹ Stop
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => regenerate()}
                disabled={Boolean(lockedSource)}
                title={lockedSource ? "Locked to the scanned target — Unlock above to edit the layout" : undefined}
                className="rounded-md border border-gray-600 px-3 py-2 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-40"
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
