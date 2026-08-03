// Body-pose stance guard for aim calibration.
//
// The 9-dot aim model is only valid for the posture it was calibrated from —
// nothing in the card tracker can tell that the SHOOTER moved. This module
// wraps MediaPipe's pose landmarker to (a) draw a live wireframe on the
// camera preview and (b) reduce the body to a compact stance signature so
// the trainer can measure "how far am I from the posture I calibrated in?"
// and prompt a recalibration when the answer is "too far".
//
// Everything degrades gracefully: the model + wasm load lazily from CDN at
// first use, and any failure (offline, unsupported device) just disables the
// guard — detection and training are never blocked on it.

export type PosePoint = { x: number; y: number; v: number }; // frame-normalized + visibility

// Stance signature: absolute shoulder-frame placement (position in frame,
// shoulder width = distance proxy) plus keypoints RELATIVE to the shoulder
// frame (shape). Both matter for aim validity — stepping sideways changes
// the mapping even with identical body shape.
export type StanceSig = {
  mid: readonly [number, number]; // shoulder midpoint, frame units
  w: number; // shoulder width, frame units
  rel: (readonly [number, number] | null)[]; // shape keypoints in shoulder-width units
};

// MediaPipe Pose landmark indices used here.
const NOSE = 0;
const L_SHOULDER = 11;
const R_SHOULDER = 12;
const L_ELBOW = 13;
const R_ELBOW = 14;
const L_WRIST = 15;
const R_WRIST = 16;
const L_HIP = 23;
const R_HIP = 24;

// Shape keypoints for the signature (relative to the shoulder frame).
const SIG_POINTS = [NOSE, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP] as const;

// Upper-body wireframe edges for the preview overlay.
export const WIREFRAME_EDGES: ReadonlyArray<readonly [number, number]> = [
  [L_SHOULDER, R_SHOULDER],
  [L_SHOULDER, L_ELBOW],
  [L_ELBOW, L_WRIST],
  [R_SHOULDER, R_ELBOW],
  [R_ELBOW, R_WRIST],
  [L_SHOULDER, L_HIP],
  [R_SHOULDER, R_HIP],
  [L_HIP, R_HIP],
  [NOSE, L_SHOULDER],
  [NOSE, R_SHOULDER],
];
export const WIREFRAME_JOINTS: readonly number[] = [
  NOSE,
  L_SHOULDER,
  R_SHOULDER,
  L_ELBOW,
  R_ELBOW,
  L_WRIST,
  R_WRIST,
  L_HIP,
  R_HIP,
];

const VIS_MIN = 0.5; // below this a landmark is treated as unseen

// Pinned versions: the wasm bundle and model are fetched from CDN at first
// use (they are far too large to ship in the app bundle).
const TASKS_VISION_VERSION = "0.10.14";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

type Landmarker = {
  detectForVideo: (video: HTMLVideoElement, tsMs: number) => {
    landmarks: { x: number; y: number; visibility?: number }[][];
  };
  close: () => void;
};

let landmarker: Landmarker | null = null;
let initPromise: Promise<boolean> | null = null;
let lastTs = 0;

// Lazy one-time init. Resolves false (and stays false) on any failure.
export function initPoseStance(): Promise<boolean> {
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      const vision = await import("@mediapipe/tasks-vision");
      const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
      // GPU first; some devices only do CPU — try both before giving up.
      for (const delegate of ["GPU", "CPU"] as const) {
        try {
          landmarker = (await vision.PoseLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: MODEL_URL, delegate },
            runningMode: "VIDEO",
            numPoses: 1,
          })) as unknown as Landmarker;
          return true;
        } catch {
          landmarker = null;
        }
      }
      return false;
    } catch {
      return false;
    }
  })();
  return initPromise;
}

export function disposePoseStance(): void {
  try {
    landmarker?.close();
  } catch {
    /* already gone */
  }
  landmarker = null;
  initPromise = null;
  lastTs = 0;
}

// One detection pass. Returns frame-normalized landmarks, or null when no
// body is visible (or the model isn't ready).
export function detectPose(video: HTMLVideoElement, tsMs: number): PosePoint[] | null {
  if (!landmarker) return null;
  // detectForVideo requires strictly increasing timestamps.
  const ts = Math.max(tsMs, lastTs + 1);
  lastTs = ts;
  try {
    const res = landmarker.detectForVideo(video, ts);
    const lm = res.landmarks?.[0];
    if (!lm || lm.length < 25) return null;
    return lm.map((p) => ({ x: p.x, y: p.y, v: p.visibility ?? 1 }));
  } catch {
    return null;
  }
}

// Reduce landmarks to the stance signature. Null when the shoulders — the
// anchor of the whole frame — aren't confidently visible.
export function stanceSignature(lm: PosePoint[]): StanceSig | null {
  const ls = lm[L_SHOULDER];
  const rs = lm[R_SHOULDER];
  if (!ls || !rs || ls.v < VIS_MIN || rs.v < VIS_MIN) return null;
  const mid = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2] as const;
  const w = Math.max(0.02, Math.hypot(ls.x - rs.x, ls.y - rs.y));
  const rel = SIG_POINTS.map((idx) => {
    const p = lm[idx];
    if (!p || p.v < VIS_MIN) return null;
    return [(p.x - mid[0]) / w, (p.y - mid[1]) / w] as const;
  });
  return { mid, w, rel };
}

// How far the current stance is from the reference, in shoulder-width units.
// Combines: body translation in frame, distance-to-camera change (shoulder
// width), and posture shape change. ~0 = same stance; >0.5 = clearly moved.
export function stanceDeviation(ref: StanceSig, cur: StanceSig): number {
  const posShift = Math.hypot(cur.mid[0] - ref.mid[0], cur.mid[1] - ref.mid[1]) / ref.w;
  const scaleShift = Math.abs(cur.w - ref.w) / ref.w;
  let shapeSum = 0;
  let shapeN = 0;
  for (let i = 0; i < ref.rel.length; i += 1) {
    const a = ref.rel[i];
    const b = cur.rel[i];
    if (!a || !b) continue;
    shapeSum += Math.hypot(b[0] - a[0], b[1] - a[1]);
    shapeN += 1;
  }
  const shapeShift = shapeN > 0 ? shapeSum / shapeN : 0;
  return Math.max(posShift, scaleShift * 1.5, shapeShift);
}
