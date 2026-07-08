// Pluggable "is this patch a bullet hole?" classifier for the hybrid pipeline:
// the change detector proposes candidate regions (cheap, runs everywhere) and a
// classifier confirms them (rejects false positives).
//
// Ships with a pure-JS heuristic so the pipeline works today, with zero model.
// When you train a CNN on the data collected by the dataset exporter, export it
// to ONNX/TF.js, wrap inference in a HoleClassifier, and call setHoleClassifier()
// — the detector picks it up with no other changes. See the note at the bottom.

export type GrayPatch = { data: Uint8Array; width: number; height: number };

export interface HoleClassifier {
  readonly name: string;
  // Probability in [0,1] that the patch is centered on a real bullet hole.
  score(patch: GrayPatch): number;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

// Crop a square grayscale window from a larger gray buffer, clamped to edges.
export function cropGrayWindow(
  source: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  size: number,
): GrayPatch {
  const side = Math.max(4, Math.round(size));
  const half = Math.floor(side / 2);
  const out = new Uint8Array(side * side);
  for (let j = 0; j < side; j += 1) {
    const sy = Math.min(height - 1, Math.max(0, Math.round(centerY - half + j)));
    for (let i = 0; i < side; i += 1) {
      const sx = Math.min(width - 1, Math.max(0, Math.round(centerX - half + i)));
      out[j * side + i] = source[sy * width + sx];
    }
  }
  return { data: out, width: side, height: side };
}

// Heuristic baseline: a hole is a roughly central region whose brightness
// contrasts with the surrounding paper ring (darker for a punched hole, brighter
// for a splatter halo — either way the magnitude of the difference is the cue).
export const heuristicHoleClassifier: HoleClassifier = {
  name: "heuristic-contrast",
  score(patch) {
    const { data, width, height } = patch;
    if (width < 5 || height < 5) return 0;
    const centerX = width / 2;
    const centerY = height / 2;
    const minSide = Math.min(width, height);
    const coreRadius = minSide * 0.22;
    const ringInner = minSide * 0.32;
    const ringOuter = minSide * 0.5;

    let coreSum = 0;
    let coreCount = 0;
    let ringSum = 0;
    let ringCount = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const distance = Math.hypot(x - centerX, y - centerY);
        const value = data[y * width + x];
        if (distance <= coreRadius) {
          coreSum += value;
          coreCount += 1;
        } else if (distance >= ringInner && distance <= ringOuter) {
          ringSum += value;
          ringCount += 1;
        }
      }
    }
    if (coreCount === 0 || ringCount === 0) return 0;
    const contrast = Math.abs(coreSum / coreCount - ringSum / ringCount) / 255;
    // ~0.12 normalized contrast → 0.5; saturates by ~0.24.
    return clamp01(contrast / 0.24);
  },
};

let activeClassifier: HoleClassifier = heuristicHoleClassifier;

export function getHoleClassifier(): HoleClassifier {
  return activeClassifier;
}

export function setHoleClassifier(classifier: HoleClassifier): void {
  activeClassifier = classifier;
}

// --- Dropping in a trained model ---
// 1. Train a small hole/not_hole classifier on the exported ImageFolder dataset
//    (e.g. `yolo classify train model=yol11n-cls.pt data=trackr-holes`).
// 2. Export to ONNX, load it in the browser with onnxruntime-web, and:
//      setHoleClassifier({
//        name: "cnn",
//        score: (patch) => runOnnxModel(patch), // resize → normalize → infer
//      });
//    The change detector will then confirm hits with the CNN instead of the
//    heuristic. Keep score() synchronous (pre-load the session, run sync infer),
//    or extend the interface to async if you prefer a worker.
