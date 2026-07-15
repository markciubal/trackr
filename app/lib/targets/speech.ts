// Thin wrapper over the Web Speech API for drill callouts. Degrades to no-ops
// (still firing onEnd) when speech synthesis isn't available, so callers can
// rely on onEnd to drive timing either way.
//
// Voice customization: a persisted VoiceSettings (voice, rate, pitch, volume)
// is applied to every utterance. The available voices — male/female, accents,
// languages — come from the device via speechSynthesis.getVoices().

export type VoiceSettings = {
  voiceURI: string | null; // null = the browser/system default voice
  rate: number; // 0.5–2, 1 = normal
  pitch: number; // 0–2, 1 = normal
  volume: number; // 0–1
};

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = { voiceURI: null, rate: 1, pitch: 1, volume: 1 };

const VOICE_STORAGE_KEY = "trackr.drillVoice";

let activeSettings: VoiceSettings | null = null;

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const num = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.max(min, Math.min(max, num));
}

export function loadVoiceSettings(): VoiceSettings {
  if (typeof window === "undefined") return { ...DEFAULT_VOICE_SETTINGS };
  try {
    const raw = window.localStorage.getItem(VOICE_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_VOICE_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<VoiceSettings>;
    return {
      voiceURI: typeof parsed.voiceURI === "string" && parsed.voiceURI ? parsed.voiceURI : null,
      rate: clampNumber(parsed.rate, 0.5, 2, 1),
      pitch: clampNumber(parsed.pitch, 0, 2, 1),
      volume: clampNumber(parsed.volume, 0, 1, 1),
    };
  } catch {
    return { ...DEFAULT_VOICE_SETTINGS };
  }
}

// Set (and persist) the voice used for all subsequent speech.
export function setVoiceSettings(settings: VoiceSettings): void {
  activeSettings = { ...settings };
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(VOICE_STORAGE_KEY, JSON.stringify(activeSettings));
  } catch {
    // Storage full/blocked — settings still apply for this session.
  }
}

function getActiveVoiceSettings(): VoiceSettings {
  if (!activeSettings) activeSettings = loadVoiceSettings();
  return activeSettings;
}

// The device's installed voices. May be empty until the browser fires
// "voiceschanged" — callers should refresh on that event.
export function listVoices(): SpeechSynthesisVoice[] {
  if (!canSpeak()) return [];
  return window.speechSynthesis.getVoices();
}

export function canSpeak(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

// Bumped by cancelSpeech. Sequences capture it when they start and stop dead
// (without firing onEnd) once it changes — otherwise a cancelled utterance's
// `onend` would chain-speak the REST of the sequence on the now-idle
// synthesizer, resurrecting callouts after a Stop.
let speechGeneration = 0;

export function cancelSpeech(): void {
  speechGeneration += 1;
  if (canSpeak()) window.speechSynthesis.cancel();
}

// True while an utterance is speaking or queued. Detection code uses this to
// ignore the microphone picking up our OWN voice prompts — the speakers'
// audio is a spike like any other to the mic.
export function isSpeaking(): boolean {
  return canSpeak() && (window.speechSynthesis.speaking || window.speechSynthesis.pending);
}

export function speak(text: string, opts: { rate?: number; onEnd?: () => void } = {}): void {
  if (!canSpeak() || !text) {
    opts.onEnd?.();
    return;
  }
  const settings = getActiveVoiceSettings();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = opts.rate ?? settings.rate;
  utterance.pitch = settings.pitch;
  utterance.volume = settings.volume;
  if (settings.voiceURI) {
    const voice = window.speechSynthesis.getVoices().find((v) => v.voiceURI === settings.voiceURI);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
  }
  if (opts.onEnd) {
    utterance.onend = () => opts.onEnd?.();
    utterance.onerror = () => opts.onEnd?.();
  }
  window.speechSynthesis.speak(utterance);
}

// Speak each part with a short pause between, then call onEnd once at the end.
export function speakSequence(
  parts: string[],
  opts: { rate?: number; onEnd?: () => void } = {},
): void {
  const queue = parts.filter(Boolean);
  if (!canSpeak() || queue.length === 0) {
    opts.onEnd?.();
    return;
  }
  const generation = speechGeneration;
  let i = 0;
  const next = () => {
    // Cancelled mid-sequence: stop dead, and don't fire onEnd — the run that
    // queued this sequence has been abandoned.
    if (generation !== speechGeneration) return;
    if (i >= queue.length) {
      opts.onEnd?.();
      return;
    }
    const part = queue[i];
    i += 1;
    speak(part, { rate: opts.rate, onEnd: next });
  };
  next();
}
