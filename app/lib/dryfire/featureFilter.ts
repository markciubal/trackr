// Constant-velocity feature filter with an outlier gate — replaces the plain
// EMA on the flag's feature vector.
//
// Each feature gets an independent alpha-beta filter (a fixed-gain Kalman
// filter): the state is position + velocity, so fast genuine transitions are
// tracked with far less lag than an EMA of equal smoothness. Before an
// update, the frame's aggregate innovation (residual vs prediction,
// normalized by a running residual scale) is tested — a single wild frame
// (momentary hijack, half-occluded card) is REJECTED rather than blended in.
// Consecutive rejections force a reset so the filter can't lock onto a stale
// state after a real jump.

export type FeatureFilter = {
  // Feed one measurement; returns the filtered feature vector. quality
  // (0..1) scales the correction gains — a frame with a poor geometric fit
  // nudges the state less than a crisp one.
  update: (z: number[], atMs: number, quality?: number) => number[];
  current: () => number[] | null;
  reset: () => void;
};

const ALPHA = 0.45; // position gain
const BETA = 0.06; // velocity gain
const GATE = 6; // reject frames beyond this × the running residual scale
const MAX_REJECTS = 5; // then hard-reset onto the new data

export function createFeatureFilter(): FeatureFilter {
  let x: number[] | null = null; // positions
  let v: number[] | null = null; // velocities (per ms)
  let lastMs = 0;
  let residScale = 0; // running mean |innovation| (per feature, aggregated)
  let rejects = 0;

  const reset = () => {
    x = null;
    v = null;
    residScale = 0;
    rejects = 0;
  };

  return {
    reset,
    current: () => (x ? [...x] : null),
    update: (z: number[], atMs: number, quality = 1) => {
      if (!x || !v || x.length !== z.length) {
        x = [...z];
        v = new Array<number>(z.length).fill(0);
        lastMs = atMs;
        residScale = 0;
        rejects = 0;
        return [...x];
      }
      const dt = Math.min(100, Math.max(1, atMs - lastMs));
      lastMs = atMs;
      // Predict.
      for (let i = 0; i < x.length; i += 1) x[i] += v[i] * dt;
      // Innovation magnitude across the vector.
      let resid = 0;
      for (let i = 0; i < x.length; i += 1) resid += Math.abs(z[i] - x[i]);
      if (residScale > 1e-9 && resid > residScale * GATE) {
        rejects += 1;
        if (rejects > MAX_REJECTS) {
          // The world really did jump — start over on the new data.
          x = [...z];
          v.fill(0);
          residScale = 0;
          rejects = 0;
        }
        return [...x]; // hold the prediction, ignore the outlier
      }
      rejects = 0;
      residScale = residScale > 0 ? residScale + 0.1 * (resid - residScale) : resid;
      // Correct — scaled by the frame's geometric fit quality.
      const q = Math.min(1, Math.max(0.25, quality));
      for (let i = 0; i < x.length; i += 1) {
        const r = z[i] - x[i];
        x[i] += ALPHA * q * r;
        v[i] += ((BETA * q) / dt) * r;
      }
      return [...x];
    },
  };
}
