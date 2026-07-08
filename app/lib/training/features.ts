// Feature extraction for the shot classifier. A GrayPatch (square-ish grayscale
// window centered on a candidate) is reduced to a small, fixed-length vector
// that both the in-browser trainer (jsonModel.ts) and the runtime classifier
// share — so a model trained on these features scores identically everywhere.
//
// The features are illumination- and polarity-normalized radial statistics: a
// bullet hole reads as a roughly circular core that contrasts with the paper
// ring around it, regardless of overall brightness or whether the hole is darker
// (punched) or brighter (splatter halo). Keeping the vector tiny keeps the model
// tiny (a handful of weights) and the training stable with few samples.

import type { GrayPatch } from "./holeClassifier";

// Number of concentric annuli sampled from center to edge.
export const RADIAL_BINS = 6;
// Total feature vector length: per-ring normalized brightness + a few global
// shape/texture stats appended below. Keep FEATURE_LENGTH in sync if you edit
// extractFeatures — the model stores stats of this exact length.
export const FEATURE_LENGTH = RADIAL_BINS + 3;

// Reduce a grayscale patch to a fixed-length feature vector. Deterministic and
// dependency-free so it runs identically in the trainer and the scanner.
export function extractFeatures(patch: GrayPatch): Float32Array {
  const { data, width, height } = patch;
  const out = new Float32Array(FEATURE_LENGTH);
  if (width < 5 || height < 5 || data.length < width * height) return out;

  // 1) Standardize the patch to zero mean / unit variance → illumination-invariant.
  let sum = 0;
  for (let i = 0; i < width * height; i += 1) sum += data[i];
  const mean = sum / (width * height);
  let varSum = 0;
  for (let i = 0; i < width * height; i += 1) {
    const d = data[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / (width * height)) || 1;

  // 2) Accumulate standardized brightness into radial bins by distance from center.
  const cx = (width - 1) / 2;
  const cy = (height - 1) / 2;
  const maxRadius = Math.min(width, height) / 2;
  const ringSums = new Float64Array(RADIAL_BINS);
  const ringCounts = new Float64Array(RADIAL_BINS);
  let edgePixels = 0; // texture: fraction of pixels well above local mean magnitude
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const z = (data[y * width + x] - mean) / std;
      const dist = Math.hypot(x - cx, y - cy);
      const bin = Math.min(RADIAL_BINS - 1, Math.floor((dist / maxRadius) * RADIAL_BINS));
      ringSums[bin] += z;
      ringCounts[bin] += 1;
      if (Math.abs(z) > 1) edgePixels += 1;
    }
  }

  for (let b = 0; b < RADIAL_BINS; b += 1) {
    out[b] = ringCounts[b] > 0 ? ringSums[b] / ringCounts[b] : 0;
  }
  // 3) Global shape/texture descriptors:
  //    - core-vs-ring contrast (signed): inner bins minus outer bins.
  const core = out[0] + out[1];
  const rim = out[RADIAL_BINS - 1] + out[RADIAL_BINS - 2];
  out[RADIAL_BINS] = core - rim;
  //    - absolute contrast magnitude (polarity-agnostic cue the heuristic uses).
  out[RADIAL_BINS + 1] = Math.abs(core - rim);
  //    - edge density (textured/ragged regions read differently from clean holes).
  out[RADIAL_BINS + 2] = edgePixels / (width * height);
  return out;
}
