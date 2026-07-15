// Dry-fire click trigger.
//
// Listens to the microphone for the sharp transient of a striker/hammer
// falling on an empty chamber and fires a callback — that's the "shot break"
// signal for dry-fire training (no laser pulse to see). Detection is on the
// PEAK sample amplitude, not RMS: a striker click is a ~2 ms transient, and
// averaging it across the ~21 ms analysis window dilutes it ~3× — peak keeps
// the full spike. Adaptive threshold: a click must jump well above the
// rolling ambient peak level, with a refractory period so one click can't
// double-fire.

import { isSpeaking } from "@/app/lib/targets/speech";

export type TriggerSensitivity = "low" | "normal" | "high";

// k = required sigmas above ambient; floor = absolute minimum threshold.
const TRIGGER_GATES: Record<TriggerSensitivity, { k: number; floor: number }> = {
  low: { k: 5, floor: 0.04 }, // noisy rooms — only loud, sharp clicks
  normal: { k: 3, floor: 0.018 },
  high: { k: 2.2, floor: 0.008 }, // quiet rooms / distant mic (default)
};

export type ClickTrigger = {
  start: () => Promise<boolean>;
  stop: () => void;
  setSensitivity: (s: TriggerSensitivity) => void;
  // Calibrated absolute floor (from the trigger-click calibration routine) —
  // replaces the preset's floor while set; null reverts to the preset.
  setCustomFloor: (floor: number | null) => void;
  // CAPTURE MODE (calibration): the discriminator is bypassed and every
  // isolated above-noise spike is delivered raw through onClick, in order —
  // the caller assigns meaning by sequence (first = slide, then clicks).
  setCaptureMode: (on: boolean) => void;
  // Spectral fingerprints from calibration: once the CLICK fingerprint is
  // set, a candidate transient must spectrally match your striker (and beat
  // the rack fingerprint, if provided) to fire. null disables matching.
  setFingerprints: (click: number[] | null, rack: number[] | null) => void;
  // Latest level / threshold (0..~1), for a small UI meter. peakHold decays
  // over ~1 s so a 2 ms click stays visible to a slow UI poll; ambient is
  // the rolling quiet-room level the threshold is derived from. suppressed
  // is true while a noisy action (slide rack) is holding the trigger;
  // rejectedAtMs is when the last noisy event was discarded. lastFp is the
  // spectral fingerprint of the most recent loud transient (calibration
  // harvests these); clickSim is how well the last candidate matched the
  // calibrated click fingerprint (null until one is set and a candidate
  // has been scored).
  getLevel: () => {
    rms: number;
    threshold: number;
    armed: boolean;
    peakHold: number;
    ambient: number;
    suppressed: boolean;
    rejectedAtMs: number;
    lastFp: number[] | null;
    clickSim: number | null;
  };
};

// ---- Spectral fingerprint ---------------------------------------------------
// A transient's identity, as FINGERPRINT_BANDS log-spaced band energies
// normalized to unit length. A striker fall is a short broadband metallic
// snap with strong highs; a rack is low-mid clunk and scrape; a voice or
// door slam is different again. Cosine similarity separates them where
// amplitude alone can't.
export const FINGERPRINT_BANDS = 16;

export function fingerprintSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) dot += a[i] * b[i];
  return dot; // inputs are unit-normalized
}

// Mean of several fingerprints, renormalized — averages calibration clicks.
export function meanFingerprint(fps: number[][]): number[] | null {
  if (fps.length === 0) return null;
  const out = new Array<number>(fps[0].length).fill(0);
  for (const fp of fps) for (let i = 0; i < out.length; i += 1) out[i] += fp[i];
  let norm = 0;
  for (const v of out) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  return out.map((v) => v / norm);
}

// ---- Click vs rack discrimination -----------------------------------------
// A striker fall is ONE sharp transient bracketed by silence. Racking the
// slide is a CLUSTER: pull, rear-stop impact, release, slam home — several
// transients across a few hundred ms, plus scraping. So a candidate click
// must be (a) preceded by quiet, and (b) followed by quiet through a short
// confirmation window; any follow-up transient reclassifies the whole event
// as a noisy action and suppresses triggering until the room settles.
const PRE_QUIET_POLLS = 2; // ~100 ms of history that must be calm before a click
const PRE_QUIET_FACTOR = 0.7; // "calm" = below this × threshold
const CONFIRM_MS = 120; // post-click quiet window before firing
const RETRIGGER_FACTOR = 0.5; // follow-up transient above this × candidate peak = rack
const NOISY_QUIET_MS = 250; // silence needed to leave the suppressed state

export function createClickTrigger(
  onClick: (atMs: number, peak: number, fingerprint: number[] | null) => void,
  sensitivity: TriggerSensitivity = "high",
): ClickTrigger {
  let audioContext: AudioContext | null = null;
  let stream: MediaStream | null = null;
  let analyser: AnalyserNode | null = null;
  let timer: number | null = null;
  let buffer: Float32Array<ArrayBuffer> | null = null;
  let freqBuf: Float32Array<ArrayBuffer> | null = null;

  let gate = TRIGGER_GATES[sensitivity];
  let customFloor: number | null = null;
  let emaMean = 0.005;
  let emaVar = 1e-6;
  let lastFireMs = 0;
  let peakHold = 0;
  // Fingerprint state.
  let clickFp: number[] | null = null;
  let rackFp: number[] | null = null;
  let lastFp: number[] | null = null;
  let lastFpAtMs = 0;
  let lastSim: number | null = null;
  const SIM_MIN = 0.65; // candidate must match the calibrated click this well
  // Discrimination state.
  const recent: number[] = []; // last few poll peaks (pre-quiet check)
  let candidate: { atMs: number; peak: number; fp: number[] | null } | null = null;
  let noisyUntilQuietMs = 0; // > now while suppressed by a noisy action
  let suppressedSinceMs = 0; // when the current suppression began
  let rejectedAtMs = 0;
  // Capture mode (calibration): raw sequential spikes, no classification.
  let captureMode = false;
  let lastSpikeMs = 0;
  let level = {
    rms: 0,
    threshold: 1,
    armed: false,
    peakHold: 0,
    ambient: 0,
    suppressed: false,
    rejectedAtMs: 0,
    lastFp: null as number[] | null,
    clickSim: null as number | null,
  };

  const REFRACTORY_MS = 350;
  // 50 ms poll. The analysis window (fftSize 4096 ≈ 85 ms at 48 kHz) is
  // wider than the poll gap, so no transient falls between polls — and the
  // window is long enough that a snapshot taken up to one poll after the
  // spike still contains its spectrum.
  const POLL_MS = 50;

  // Spectral snapshot of whatever is in the analysis window right now, as
  // FINGERPRINT_BANDS log-spaced band energies, unit-normalized.
  const fingerprintNow = (): number[] | null => {
    if (!analyser || !freqBuf) return null;
    analyser.getFloatFrequencyData(freqBuf);
    const n = freqBuf.length;
    const bands = new Array<number>(FINGERPRINT_BANDS).fill(0);
    const first = 3; // skip DC / rumble bins
    for (let b = 0; b < FINGERPRINT_BANDS; b += 1) {
      const i0 = Math.round(first * Math.pow(n / first, b / FINGERPRINT_BANDS));
      const i1 = Math.max(i0 + 1, Math.round(first * Math.pow(n / first, (b + 1) / FINGERPRINT_BANDS)));
      let acc = 0;
      for (let i = i0; i < Math.min(n, i1); i += 1) {
        const db = Math.max(-100, freqBuf[i]);
        acc += Math.pow(10, db / 10);
      }
      bands[b] = Math.sqrt(acc / Math.max(1, i1 - i0));
    }
    let norm = 0;
    for (const v of bands) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return null;
    return bands.map((v) => v / norm);
  };

  const poll = () => {
    if (!analyser || !buffer) return;
    analyser.getFloatTimeDomainData(buffer);
    let peak = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const a = Math.abs(buffer[i]);
      if (a > peak) peak = a;
    }
    const std = Math.sqrt(emaVar);
    const threshold = Math.max(emaMean + gate.k * std, customFloor ?? gate.floor);
    const now = performance.now();
    const armed = now - lastFireMs > REFRACTORY_MS;
    peakHold = Math.max(peak, peakHold * 0.88); // ~1 s tail at 50 ms polls

    const loud = peak > threshold;
    // "Quiet" must be judged against the ROOM, not the click threshold: a
    // calibrated floor can sit below ambient peaks, and a quiet bar derived
    // from it would then be unreachable — suppression would never release
    // and candidates could never form.
    const quietLevel = Math.max(emaMean * 3, threshold * PRE_QUIET_FACTOR);
    const calm = peak < quietLevel;
    const preQuiet = recent.every((p) => p < quietLevel);
    const suppressed = noisyUntilQuietMs > 0;

    // Fingerprint any decently-loud transient (calibration harvests these
    // for the rack template; the click path snapshots its own candidate).
    if (peak > Math.max(emaMean * 4, threshold * 0.5, 0.003) && now - lastFpAtMs > 80) {
      const fp = fingerprintNow();
      if (fp) {
        lastFp = fp;
        lastFpAtMs = now;
      }
    }

    // Our own voice prompts play through the speakers and land on this mic.
    // While speaking: no spike capture, and — critically — no ambient
    // learning, or a long prompt inflates the "room noise" estimate 4-10×
    // and the gate sits above a real rack for seconds afterward.
    const speakingNow = isSpeaking();
    if (captureMode) {
      // Raw sequential capture: any isolated above-noise spike is an event.
      // 600 ms of spacing keeps one rack (several impacts) as ONE spike.
      if (!speakingNow && peak > Math.max(emaMean * 4, 0.004) && now - lastSpikeMs > 600) {
        lastSpikeMs = now;
        onClick(now, peak, fingerprintNow());
      }
    } else if (suppressed) {
      // Waiting out a noisy action (rack): need sustained calm to re-arm.
      if (suppressedSinceMs === 0) suppressedSinceMs = now;
      if (calm) {
        if (now >= noisyUntilQuietMs) {
          noisyUntilQuietMs = 0;
          suppressedSinceMs = 0;
        }
      } else if (now - suppressedSinceMs > 3000) {
        // Hard cap: 3 s of continuous "noise" means the ROOM changed, not
        // that a rack is still going — release and let the adaptive
        // threshold absorb the new level. Suppression can never wedge.
        noisyUntilQuietMs = 0;
        suppressedSinceMs = 0;
      } else {
        noisyUntilQuietMs = now + NOISY_QUIET_MS;
      }
    } else if (candidate) {
      // Confirmation window: any follow-up transient means this was the
      // first impact of a compound action, not an isolated click.
      if (peak > Math.max(threshold, candidate.peak * RETRIGGER_FACTOR)) {
        candidate = null;
        rejectedAtMs = now;
        noisyUntilQuietMs = now + NOISY_QUIET_MS;
      } else if (now - candidate.atMs >= CONFIRM_MS) {
        // FINGERPRINT GATE: with a calibrated click template, the candidate
        // must sound like YOUR striker or it's rejected (a knuckle rap, a
        // dropped pen). CAUTION on the rack comparison: the click and the
        // rack are the SAME gun's metal — their templates typically score
        // 0.9+ against each other, so "closer to rack than click" is a coin
        // flip for a genuine dry-fire. Reject as rack only when the
        // transient is BOTH decisively rack-matching in absolute terms AND
        // clearly beats the click match; a striker-quality match fires.
        let simOk = true;
        let rackLike = false;
        if (clickFp && candidate.fp) {
          const sim = fingerprintSimilarity(candidate.fp, clickFp);
          lastSim = sim;
          const rackSim = rackFp ? fingerprintSimilarity(candidate.fp, rackFp) : -1;
          rackLike = rackSim > 0.92 && rackSim > sim + 0.02;
          simOk = sim >= SIM_MIN && !rackLike;
        }
        const atMs = candidate.atMs; // original click time — aim sampling uses it
        const clickPeak = candidate.peak;
        const clickFingerprint = candidate.fp;
        candidate = null;
        if (simOk) {
          lastFireMs = now; // refractory belongs to real fires only
          onClick(atMs, clickPeak, clickFingerprint);
        } else {
          rejectedAtMs = now;
          // A transient that sounds like YOUR RACK is the start of a slide
          // pull — enter the hold state until the room settles, so the UI's
          // "HOLDING (noisy)" is true for characterized racks too, and the
          // rack's later impacts can't sneak through as candidates.
          if (rackLike) noisyUntilQuietMs = now + NOISY_QUIET_MS;
        }
      }
    } else if (armed && loud) {
      if (preQuiet) {
        candidate = { atMs: now, peak, fp: fingerprintNow() };
      } else {
        // Loud without preceding quiet — mid-rumble; suppress.
        rejectedAtMs = now;
        noisyUntilQuietMs = now + NOISY_QUIET_MS;
      }
    }

    if (!loud && !speakingNow) {
      // Only track ambient while quiet AND not narrating — neither a click
      // nor our own voice may raise the bar.
      const alpha = 0.15; // matched to the 50 ms poll (~330 ms time constant)
      emaMean += alpha * (peak - emaMean);
      emaVar += alpha * ((peak - emaMean) * (peak - emaMean) - emaVar);
    }

    recent.push(peak);
    if (recent.length > PRE_QUIET_POLLS) recent.shift();
    level = {
      rms: peak,
      threshold,
      armed,
      peakHold,
      ambient: emaMean,
      suppressed: noisyUntilQuietMs > 0,
      rejectedAtMs,
      lastFp,
      clickSim: lastSim,
    };
  };

  return {
    start: async () => {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) return false;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
          video: false,
        });
      } catch {
        return false;
      }
      audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      // 4096 ≈ 85 ms window at 48 kHz — wider than the 50 ms poll gap, and
      // 2048 frequency bins for the spectral fingerprint.
      analyser.fftSize = 4096;
      analyser.smoothingTimeConstant = 0; // fingerprints want the raw frame
      buffer = new Float32Array(analyser.fftSize);
      freqBuf = new Float32Array(analyser.frequencyBinCount);
      source.connect(analyser);
      await audioContext.resume().catch(() => undefined);
      timer = window.setInterval(poll, POLL_MS);
      return true;
    },
    stop: () => {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
        stream = null;
      }
      if (audioContext) {
        void audioContext.close().catch(() => undefined);
        audioContext = null;
      }
      analyser = null;
      buffer = null;
      freqBuf = null;
    },
    setSensitivity: (s: TriggerSensitivity) => {
      gate = TRIGGER_GATES[s];
    },
    setCustomFloor: (floor: number | null) => {
      customFloor = floor;
    },
    setCaptureMode: (on: boolean) => {
      captureMode = on;
      candidate = null;
      noisyUntilQuietMs = 0;
      suppressedSinceMs = 0;
      lastSpikeMs = 0;
    },
    setFingerprints: (click: number[] | null, rack: number[] | null) => {
      clickFp = click;
      rackFp = rack;
      lastSim = null;
    },
    getLevel: () => level,
  };
}
