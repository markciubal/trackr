// Grid-cell background subtraction for the dry-fire camera.
//
// The camera is stationary and the flag card is handheld, so the card is
// ALWAYS foreground — while the colored clutter that fools acquisition
// (posters, bins, bezels, a shirt on a chair) is static. This model keeps a
// slow per-cell mean-RGB background and reports which cells currently differ
// from it; acquisition uses that as a motion veto (see ForegroundGrid in
// flagTracker.ts) — tried FIRST, with the un-vetoed pass as fallback, so
// background subtraction can only ever add detections.
//
// Design notes, all in service of never hurting the existing pipeline:
// - WARMUP: the first frames seed the model; no grid is returned until the
//   model has actually seen the scene (a fresh model would veto everything).
// - The caller passes an EXCLUDE rect (the tracked card / the photo ring):
//   those cells never adapt, so a steady 30-second hold cannot be absorbed
//   into the background.
// - Foreground cells adapt ~8× slower than background cells: transient
//   objects fade in slowly, lighting drift still tracks.
// - Degenerate grids return null (nothing moving → maybe the card has been
//   static since startup and IS the background; almost everything moving →
//   camera bump / exposure flip, the veto means nothing). Null = the caller
//   simply runs the normal un-vetoed detection.
// - The mask is dilated by one cell so the card's anti-aliased edges and a
//   slow-moving hand survive the veto.

import type { ForegroundGrid } from "./flagTracker";

export type BackgroundModel = {
  // Feed a full processing frame; returns the foreground grid or null when
  // the veto should not be used this frame. Cheap: samples every 2nd pixel.
  step: (
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
    exclude?: { x: number; y: number; w: number; h: number } | null,
  ) => ForegroundGrid | null;
  reset: () => void;
};

const CELL = 16; // px per grid cell (matches the CMY prescan's granularity)
const WARMUP = 6; // model steps before the veto is trusted
const ADAPT_BG = 0.06; // background cells ease toward the current frame
const ADAPT_FG = 0.008; // foreground cells barely adapt (don't absorb the card)
const THR = 22; // mean |ΔRGB| per channel that makes a cell "changed"
const MIN_FG_CELLS = 3; // less movement than this → don't veto
const MAX_FG_FRAC = 0.6; // more of the frame than this changed → don't veto

export function createBackgroundModel(): BackgroundModel {
  let gw = 0;
  let gh = 0;
  let bg: Float32Array | null = null; // 3 floats per cell
  let steps = 0;

  const reset = () => {
    bg = null;
    steps = 0;
  };

  const step = (
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
    exclude: { x: number; y: number; w: number; h: number } | null = null,
  ): ForegroundGrid | null => {
    const ngw = Math.max(1, Math.ceil(width / CELL));
    const ngh = Math.max(1, Math.ceil(height / CELL));
    if (!bg || ngw !== gw || ngh !== gh) {
      gw = ngw;
      gh = ngh;
      bg = null;
      steps = 0;
    }
    const n = gw * gh;

    // Current per-cell mean RGB, sampled at stride 2.
    const sum = new Float32Array(n * 3);
    const cnt = new Float32Array(n);
    for (let y = 0; y < height; y += 2) {
      const rowCell = ((y / CELL) | 0) * gw;
      for (let x = 0; x < width; x += 2) {
        const o = (y * width + x) * 4;
        const gi = rowCell + ((x / CELL) | 0);
        sum[gi * 3] += rgba[o];
        sum[gi * 3 + 1] += rgba[o + 1];
        sum[gi * 3 + 2] += rgba[o + 2];
        cnt[gi] += 1;
      }
    }
    for (let i = 0; i < n; i += 1) {
      const c = Math.max(1, cnt[i]);
      sum[i * 3] /= c;
      sum[i * 3 + 1] /= c;
      sum[i * 3 + 2] /= c;
    }

    if (!bg) {
      bg = sum.slice();
      steps = 1;
      return null;
    }
    steps += 1;

    // Classify cells, then adapt the model (excluded region never adapts).
    const raw = new Uint8Array(n);
    let fgCount = 0;
    for (let gy = 0; gy < gh; gy += 1) {
      for (let gx = 0; gx < gw; gx += 1) {
        const i = gy * gw + gx;
        const d =
          (Math.abs(sum[i * 3] - bg[i * 3]) +
            Math.abs(sum[i * 3 + 1] - bg[i * 3 + 1]) +
            Math.abs(sum[i * 3 + 2] - bg[i * 3 + 2])) /
          3;
        const isFg = d > THR;
        if (isFg) {
          raw[i] = 1;
          fgCount += 1;
        }
        if (exclude) {
          const px = gx * CELL;
          const py = gy * CELL;
          if (px + CELL > exclude.x && px < exclude.x + exclude.w && py + CELL > exclude.y && py < exclude.y + exclude.h) {
            continue;
          }
        }
        const a = isFg ? ADAPT_FG : ADAPT_BG;
        bg[i * 3] += (sum[i * 3] - bg[i * 3]) * a;
        bg[i * 3 + 1] += (sum[i * 3 + 1] - bg[i * 3 + 1]) * a;
        bg[i * 3 + 2] += (sum[i * 3 + 2] - bg[i * 3 + 2]) * a;
      }
    }

    if (steps <= WARMUP) return null;
    if (fgCount < MIN_FG_CELLS || fgCount > n * MAX_FG_FRAC) return null;

    // Dilate by one cell so edges and slow hands survive.
    const mask = new Uint8Array(n);
    for (let gy = 0; gy < gh; gy += 1) {
      for (let gx = 0; gx < gw; gx += 1) {
        if (!raw[gy * gw + gx]) continue;
        for (let dy = -1; dy <= 1; dy += 1) {
          const yy = gy + dy;
          if (yy < 0 || yy >= gh) continue;
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = gx + dx;
            if (xx >= 0 && xx < gw) mask[yy * gw + xx] = 1;
          }
        }
      }
    }
    return { mask, gw, gh, cell: CELL };
  };

  return { step, reset };
}
