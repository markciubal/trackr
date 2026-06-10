// Thin wrapper over the Web Speech API for drill callouts. Degrades to no-ops
// (still firing onEnd) when speech synthesis isn't available, so callers can
// rely on onEnd to drive timing either way.

export function canSpeak(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function cancelSpeech(): void {
  if (canSpeak()) window.speechSynthesis.cancel();
}

export function speak(text: string, opts: { rate?: number; onEnd?: () => void } = {}): void {
  if (!canSpeak() || !text) {
    opts.onEnd?.();
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = opts.rate ?? 1;
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
  let i = 0;
  const next = () => {
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
