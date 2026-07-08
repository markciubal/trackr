// In-browser shot classifier: train a tiny logistic-regression model on labeled
// patches, serialize it to JSON, and rebuild a runtime HoleClassifier from that
// JSON. No Python, no ONNX, no native deps — the whole train→publish→load loop
// runs in the app. The admin panel trains and publishes ModelJSON; every
// scanner fetches the published ModelJSON and calls buildClassifierFromModel().
//
// The model is deliberately small (FEATURE_LENGTH weights + bias) and trained
// with deterministic full-batch gradient descent (zero init, no RNG) so the
// same dataset always yields the same model — reproducible and resume-safe.

import { extractFeatures, FEATURE_LENGTH } from "./features";
import type { GrayPatch, HoleClassifier } from "./holeClassifier";

export const MODEL_KIND = "logreg-radial-v1";

export type LabeledPatch = { patch: GrayPatch; label: "hole" | "not_hole" };

export type ModelJSON = {
  kind: typeof MODEL_KIND;
  // Per-feature standardization (computed over the training set).
  featureMeans: number[];
  featureStds: number[];
  // Logistic-regression parameters over the standardized feature vector.
  weights: number[];
  bias: number;
  // Provenance / quality, surfaced in the admin UI and model history.
  meta: {
    samples: number;
    holes: number;
    negatives: number;
    trainAccuracy: number; // 0..1 on the training set (with threshold 0.5)
    iterations: number;
  };
};

function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export type TrainOptions = {
  iterations?: number;
  learningRate?: number;
  l2?: number; // L2 regularization strength
};

// Train a logistic-regression hole/not_hole classifier. Returns null if the
// dataset is too small or single-class (nothing to learn).
export function trainModel(samples: LabeledPatch[], options: TrainOptions = {}): ModelJSON | null {
  const iterations = options.iterations ?? 400;
  const learningRate = options.learningRate ?? 0.3;
  const l2 = options.l2 ?? 1e-3;

  const holes = samples.filter((s) => s.label === "hole").length;
  const negatives = samples.length - holes;
  if (samples.length < 4 || holes === 0 || negatives === 0) return null;

  // 1) Extract features and labels.
  const X = samples.map((s) => extractFeatures(s.patch));
  const y = samples.map((s) => (s.label === "hole" ? 1 : 0));
  const n = X.length;

  // 2) Standardize each feature across the dataset (stored in the model so the
  //    scanner standardizes incoming patches identically).
  const featureMeans = new Array(FEATURE_LENGTH).fill(0);
  const featureStds = new Array(FEATURE_LENGTH).fill(1);
  for (let f = 0; f < FEATURE_LENGTH; f += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += X[i][f];
    const mean = sum / n;
    let varSum = 0;
    for (let i = 0; i < n; i += 1) {
      const d = X[i][f] - mean;
      varSum += d * d;
    }
    featureMeans[f] = mean;
    featureStds[f] = Math.sqrt(varSum / n) || 1;
  }
  const Z = X.map((row) => {
    const z = new Float64Array(FEATURE_LENGTH);
    for (let f = 0; f < FEATURE_LENGTH; f += 1) z[f] = (row[f] - featureMeans[f]) / featureStds[f];
    return z;
  });

  // 3) Full-batch gradient descent (zero init → deterministic).
  const weights = new Float64Array(FEATURE_LENGTH);
  let bias = 0;
  for (let iter = 0; iter < iterations; iter += 1) {
    const gradW = new Float64Array(FEATURE_LENGTH);
    let gradB = 0;
    for (let i = 0; i < n; i += 1) {
      let dot = bias;
      for (let f = 0; f < FEATURE_LENGTH; f += 1) dot += weights[f] * Z[i][f];
      const error = sigmoid(dot) - y[i];
      for (let f = 0; f < FEATURE_LENGTH; f += 1) gradW[f] += error * Z[i][f];
      gradB += error;
    }
    for (let f = 0; f < FEATURE_LENGTH; f += 1) {
      weights[f] -= learningRate * (gradW[f] / n + l2 * weights[f]);
    }
    bias -= learningRate * (gradB / n);
  }

  // 4) Training accuracy (threshold 0.5).
  let correct = 0;
  for (let i = 0; i < n; i += 1) {
    let dot = bias;
    for (let f = 0; f < FEATURE_LENGTH; f += 1) dot += weights[f] * Z[i][f];
    if ((sigmoid(dot) >= 0.5 ? 1 : 0) === y[i]) correct += 1;
  }

  return {
    kind: MODEL_KIND,
    featureMeans,
    featureStds,
    weights: Array.from(weights),
    bias,
    meta: {
      samples: n,
      holes,
      negatives,
      trainAccuracy: correct / n,
      iterations,
    },
  };
}

// Validate a parsed JSON blob as a ModelJSON (defensive — it crosses the wire).
export function isModelJSON(value: unknown): value is ModelJSON {
  if (!value || typeof value !== "object") return false;
  const m = value as Record<string, unknown>;
  return (
    m.kind === MODEL_KIND &&
    Array.isArray(m.weights) &&
    (m.weights as unknown[]).length === FEATURE_LENGTH &&
    Array.isArray(m.featureMeans) &&
    Array.isArray(m.featureStds) &&
    typeof m.bias === "number"
  );
}

// Rebuild a runtime HoleClassifier from published ModelJSON. Synchronous score()
// matches the existing HoleClassifier contract, so the change detector can swap
// it in via setHoleClassifier() with no other changes.
export function buildClassifierFromModel(model: ModelJSON): HoleClassifier {
  const { weights, bias, featureMeans, featureStds } = model;
  return {
    name: `${model.kind}-v${model.meta?.samples ?? 0}`,
    score(patch) {
      const features = extractFeatures(patch);
      let dot = bias;
      for (let f = 0; f < FEATURE_LENGTH; f += 1) {
        const z = (features[f] - featureMeans[f]) / (featureStds[f] || 1);
        dot += weights[f] * z;
      }
      return sigmoid(dot);
    },
  };
}
