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
  low: { k: 4, floor: 0.03 }, // noisy rooms — only loud, sharp clicks
  normal: { k: 2.2, floor: 0.011 },
  high: { k: 1.5, floor: 0.0045 }, // quiet rooms / distant mic (default)
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
  // PRIMARY gate: the calibrated striker-click DURATION (ms the transient
  // stays loud). A click's duration is consistent shot-to-shot, so it's the
  // most reliable discriminator — a candidate much longer than this is a
  // scrape/voice/rack, not a click. 0 = fall back to a generic short cap.
  setClickDuration: (ms: number) => void;
  // Calibrated rack inter-impact gap (pull-back → slam), ms. Sizes the
  // two-peak confirmation wait: a candidate only fires after this window
  // passes with NO second peak. 0 = generic window.
  setRackGap: (ms: number) => void;
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

// AUDITION a stored fingerprint: we don't keep raw recordings, but the
// 16-band spectrum IS the sound's identity — resynthesize it as band-shaped
// noise bursts with a percussive envelope. A click plays as one short snap;
// a slide as two bursts (pull-back … slam). Close enough to "hear what the
// calibration captured" and judge whether it still matches reality.
export function playFingerprintSample(
  fp: number[],
  opts: { durMs?: number; peak?: number; bursts?: number } = {},
): void {
  if (typeof window === "undefined" || fp.length === 0) return;
  const durMs = opts.durMs ?? 90;
  const bursts = Math.max(1, opts.bursts ?? 1);
  const gapMs = 160;
  const ctx = new AudioContext();
  const nyq = ctx.sampleRate / 2;
  const n = fp.length;
  const level = Math.min(0.9, 0.25 + (opts.peak ?? 0.1) * 2);
  for (let burst = 0; burst < bursts; burst += 1) {
    const t0 = ctx.currentTime + 0.03 + (burst * gapMs) / 1000;
    const len = Math.max(1, Math.round(ctx.sampleRate * (durMs / 1000)));
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i += 1) data[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const env = ctx.createGain();
    env.gain.setValueAtTime(level, t0);
    env.gain.exponentialRampToValueAtTime(0.001, t0 + durMs / 1000);
    env.connect(ctx.destination);
    // Parallel bandpass bank matching the fingerprint's log-spaced bands
    // (bins 3..2048 of a 4096-point FFT at capture time).
    for (let b = 0; b < n; b += 1) {
      const e0 = (3 * Math.pow(2048 / 3, b / n)) / 2048;
      const e1 = (3 * Math.pow(2048 / 3, (b + 1) / n)) / 2048;
      const f0 = Math.sqrt(e0 * e1) * nyq;
      if (f0 < 40 || f0 > nyq * 0.95) continue;
      const bp = ctx.createBiquadFilter();
      bp.type = "bandpass";
      bp.frequency.value = f0;
      bp.Q.value = 2;
      const g = ctx.createGain();
      g.gain.value = fp[b];
      src.connect(bp);
      bp.connect(g);
      g.connect(env);
    }
    src.start(t0);
    src.stop(t0 + durMs / 1000 + 0.02);
  }
  window.setTimeout(() => void ctx.close().catch(() => undefined), bursts * gapMs + durMs + 300);
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
// TWO PEAKS vs ONE: a slide rack ALWAYS lands as two impacts (pull-back,
// then slam forward); a trigger pull is a single snap. So a candidate is
// confirmed only after a wait long enough to have SEEN the rack's second
// impact — calibrated from the shooter's own measured inter-impact gap, or
// this generic window before calibration. The wait delays the shot callout
// slightly; aim is unaffected (sampling keys off the click's own timestamp).
const GENERIC_CONFIRM_MS = 420;
const RING_GAP_MS = 110; // loud polls this close together = same transient ringing
const NOISY_QUIET_MS = 250; // silence needed to leave the suppressed state

export function createClickTrigger(
  onClick: (atMs: number, peak: number, fingerprint: number[] | null, durationMs: number) => void,
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
  // DURATION gate (primary). clickDurMs = calibrated striker duration.
  let clickDurMs = 0;
  const GENERIC_MAX_DUR = 190; // pre-calibration cap on a "click" transient
  // Calibrated rack inter-impact gap (pull-back → slam), ms. Sets how long
  // a candidate waits for a possible second peak before confirming.
  let rackGapMs = 0;
  const confirmWindowMs = (): number =>
    rackGapMs > 0 ? Math.min(700, Math.max(250, rackGapMs * 1.5 + 100)) : GENERIC_CONFIRM_MS;
  // Discrimination state.
  const recent: number[] = []; // last few poll peaks (pre-quiet check)
  let candidate: { atMs: number; peak: number; fp: number[] | null; lastLoudMs: number } | null = null;
  let noisyUntilQuietMs = 0; // > now while suppressed by a noisy action
  let suppressedSinceMs = 0; // when the current suppression began
  let rejectedAtMs = 0;
  // Capture mode (calibration): sequential spikes reported at run END so the
  // event's DURATION can be measured (pull-back vs slam durations for a rack,
  // the striker duration for a click).
  let captureMode = false;
  let lastSpikeMs = 0;
  let capStart = 0; // start of the current capture-mode loud run (0 = none)
  let capLastLoud = 0;
  let capPeak = 0;
  let capFp: number[] | null = null;
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
      // Sequential capture, reported at run END so DURATION is measured.
      // A loud run starts on the first above-noise poll and ends after
      // ~90 ms of quiet; its length is the event's duration. Each rack
      // impact (pull-back, slam) is its own run → its own duration.
      // 3× the (now honest) noise floor: the old 4× was tuned against a
      // floor that wedged low; against the real room level it over-gated.
      const capLoud = !speakingNow && peak > Math.max(emaMean * 3, 0.004);
      if (capLoud) {
        if (capStart === 0) {
          capStart = now;
          capPeak = peak;
          capFp = fingerprintNow();
        } else {
          capPeak = Math.max(capPeak, peak);
        }
        capLastLoud = now;
      } else if (capStart > 0 && now - capLastLoud > 90) {
        const durMs = capLastLoud - capStart + POLL_MS;
        if (now - lastSpikeMs > 220) {
          lastSpikeMs = now;
          onClick(capStart, capPeak, capFp, durMs);
        }
        capStart = 0;
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
      // Extend the candidate's loud run while its OWN ring continues —
      // contiguous loud polls only. A loud poll after a real gap is not
      // ring, it's a new sound (see secondPeak below).
      const stillLoud = peak > threshold * 0.6;
      const ringContinues = stillLoud && now - candidate.lastLoudMs <= RING_GAP_MS;
      if (ringContinues) {
        candidate.lastLoudMs = now;
        candidate.peak = Math.max(candidate.peak, peak);
      }
      const durMs = candidate.lastLoudMs - candidate.atMs + POLL_MS;
      // DURATION is the PRIMARY gate: a striker click is a short, consistent
      // snap — a candidate ringing past the calibrated click length (×2 +
      // margin) is a scrape / voice / rack.
      const maxDur = clickDurMs > 0 ? clickDurMs * 2 + 70 : GENERIC_MAX_DUR;
      // TWO-PEAK TEST: the candidate's ring has ended, and a NEW burst
      // arrives — that's the rack's second impact (slam after pull-back).
      // The slam is often QUIETER than the pull-back, so the bar is
      // deliberately generous: well above calm, or a real fraction of the
      // first peak.
      const ringOver = now - candidate.lastLoudMs > RING_GAP_MS;
      const secondPeak = ringOver && peak > Math.max(threshold * 0.75, candidate.peak * 0.3);
      if (durMs > maxDur || secondPeak) {
        candidate = null;
        rejectedAtMs = now;
        noisyUntilQuietMs = now + NOISY_QUIET_MS;
      } else if (now - candidate.atMs >= confirmWindowMs() && !stillLoud) {
        // Clean short transient, confirmed quiet. FINGERPRINT is now only a
        // SECONDARY veto: reject only when the sound is decisively rack-like
        // (both the click and rack are the same gun's metal, so their
        // templates score 0.9+ against each other — never trust a close
        // call). Duration already did the primary work.
        let rackLike = false;
        if (clickFp && candidate.fp) {
          const sim = fingerprintSimilarity(candidate.fp, clickFp);
          lastSim = sim;
          const rackSim = rackFp ? fingerprintSimilarity(candidate.fp, rackFp) : -1;
          rackLike = rackSim > 0.94 && rackSim > sim + 0.05;
        }
        const atMs = candidate.atMs; // original click time — aim sampling uses it
        const clickPeak = candidate.peak;
        const clickFingerprint = candidate.fp;
        candidate = null;
        if (!rackLike) {
          lastFireMs = now; // refractory belongs to real fires only
          onClick(atMs, clickPeak, clickFingerprint, durMs);
        } else {
          rejectedAtMs = now;
          noisyUntilQuietMs = now + NOISY_QUIET_MS;
        }
      }
    } else if (armed && loud) {
      if (preQuiet) {
        candidate = { atMs: now, peak, fp: fingerprintNow(), lastLoudMs: now };
      } else {
        // Loud without preceding quiet — mid-rumble; suppress.
        rejectedAtMs = now;
        noisyUntilQuietMs = now + NOISY_QUIET_MS;
      }
    }

    if (!speakingNow) {
      // ASYMMETRIC noise-floor tracker. Falls fast; rises slowly — but it
      // DOES rise even while "loud". The old !loud gate was circular: the
      // threshold comes from the floor, so a room that is continuously
      // louder than the current floor read as "loud" forever, the floor
      // never caught up, and the capture gate sat BELOW the room — one
      // endless run, nothing ever delivered. A brief transient (a click is
      // 1-2 polls) barely moves the slow riser; sustained room noise lifts
      // the floor to reality within a few seconds. Our own voice is still
      // excluded — the mic hears the speakers.
      const alpha = peak < emaMean ? 0.15 : loud ? 0.02 : 0.15;
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
      capStart = 0;
    },
    setFingerprints: (click: number[] | null, rack: number[] | null) => {
      clickFp = click;
      rackFp = rack;
      lastSim = null;
    },
    setClickDuration: (ms: number) => {
      clickDurMs = ms > 0 ? ms : 0;
    },
    setRackGap: (ms: number) => {
      rackGapMs = ms > 0 ? ms : 0;
    },
    getLevel: () => level,
  };
}
