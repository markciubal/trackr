// Aim calibration model.
//
// Maps flag-tracker feature vectors (flagTracker.ts) to on-screen aim
// coordinates via per-axis ridge-regularized least squares. Because the model
// is FITTED from "hold on this dot" samples, it absorbs everything fixed in
// the setup — camera intrinsics, screen pose, the bore-to-card transform, and
// the shooter's zero — without any explicit 3D math. The honest limitation:
// it is valid for the shooting position it was calibrated from; moving a big
// step sideways needs a recalibration.

export type AimSample = {
  features: number[]; // from FlagObservation.features
  targetX: number; // where the shooter was actually aiming (viewport px)
  targetY: number;
};

export type AimModel = {
  weightsX: number[];
  weightsY: number[];
  rmsErrorPx: number; // training residual (sanity signal, not a guarantee)
  sampleCount: number;
};

// Solve (FᵀF + λI) w = Fᵀy by Gaussian elimination with partial pivoting.
function ridgeSolve(rows: number[][], targets: number[], lambda: number): number[] | null {
  const dims = rows[0].length;
  const m: number[][] = Array.from({ length: dims }, (_, i) =>
    Array.from({ length: dims }, (_, j) => (i === j ? lambda : 0)),
  );
  const rhs = new Array<number>(dims).fill(0);
  for (let s = 0; s < rows.length; s += 1) {
    const f = rows[s];
    for (let i = 0; i < dims; i += 1) {
      rhs[i] += f[i] * targets[s];
      for (let j = 0; j < dims; j += 1) m[i][j] += f[i] * f[j];
    }
  }
  for (let col = 0; col < dims; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < dims; row += 1) if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    if (pivot !== col) {
      const tmpRow = m[col];
      m[col] = m[pivot];
      m[pivot] = tmpRow;
      const tmpB = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = tmpB;
    }
    for (let row = col + 1; row < dims; row += 1) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k < dims; k += 1) m[row][k] -= factor * m[col][k];
      rhs[row] -= factor * rhs[col];
    }
  }
  const out = new Array<number>(dims).fill(0);
  for (let row = dims - 1; row >= 0; row -= 1) {
    let acc = rhs[row];
    for (let k = row + 1; k < dims; k += 1) acc -= m[row][k] * out[k];
    out[row] = acc / m[row][row];
  }
  return out;
}

// minSamples: the full model wants ≥6 dots (7 unknowns per axis); a
// PROVISIONAL model can be fit from as few as 3 — ridge regularization keeps
// the underdetermined fit sane — to preview the aim estimate mid-calibration.
export function fitAimModel(samples: AimSample[], lambda = 1e-4, minSamples = 6): AimModel | null {
  if (samples.length < Math.max(3, minSamples)) return null;
  const rows = samples.map((s) => s.features);
  const weightsX = ridgeSolve(
    rows,
    samples.map((s) => s.targetX),
    lambda,
  );
  const weightsY = ridgeSolve(
    rows,
    samples.map((s) => s.targetY),
    lambda,
  );
  if (!weightsX || !weightsY) return null;
  let errSq = 0;
  for (const s of samples) {
    const p = predictWith(weightsX, weightsY, s.features);
    errSq += (p.x - s.targetX) ** 2 + (p.y - s.targetY) ** 2;
  }
  return {
    weightsX,
    weightsY,
    rmsErrorPx: Math.sqrt(errSq / samples.length),
    sampleCount: samples.length,
  };
}

function predictWith(weightsX: number[], weightsY: number[], features: number[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (let i = 0; i < features.length; i += 1) {
    x += weightsX[i] * features[i];
    y += weightsY[i] * features[i];
  }
  return { x, y };
}

export function predictAim(model: AimModel, features: number[]): { x: number; y: number } {
  return predictWith(model.weightsX, model.weightsY, features);
}

// Mean of a stack of feature vectors (used to average a dwell window into one
// calibration sample).
export function meanFeatures(stack: number[][]): number[] {
  const out = new Array<number>(stack[0].length).fill(0);
  for (const f of stack) for (let i = 0; i < f.length; i += 1) out[i] += f[i];
  for (let i = 0; i < out.length; i += 1) out[i] /= stack.length;
  return out;
}
