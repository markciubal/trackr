import type { CvApi, CvMat, CvQRCodeDetector } from "@/app/lib/opencv";
import { decodeTargetPayload, pixelsPerInchFromQr, type TargetPayload } from "./payload";

export type Corner = { x: number; y: number };

// The QR as located in the frame, independent of whether it decoded to a Trackr
// target. `qrSidePx` is the calibration reference (printed size is known).
export type DetectedQr = {
  text: string; // raw decoded content ("" if OpenCV located but couldn't decode it)
  corners: Corner[]; // 4 corners, in image pixels
  center: Corner;
  qrSidePx: number; // mean edge length, in pixels
};

// A located QR that decoded to one of our smart-target URLs: carries the random
// target id and (usually) the inline calibration baked into the QR.
export type DetectedTarget = {
  id: string;
  payload: TargetPayload;
  qr: DetectedQr;
  pixelsPerInch: number | null;
};

export type TargetDetection = {
  qr: DetectedQr | null;
  target: DetectedTarget | null;
};

const EMPTY: TargetDetection = { qr: null, target: null };

function readCorners(points: CvMat): Corner[] {
  const data = points.data32F;
  if (!data || data.length < 8) return [];
  const corners: Corner[] = [];
  for (let i = 0; i < 4; i += 1) corners.push({ x: data[i * 2], y: data[i * 2 + 1] });
  return corners;
}

function meanSidePx(corners: Corner[]): number {
  let sum = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    sum += Math.hypot(a.x - b.x, a.y - b.y);
  }
  return sum / 4;
}

function centroid(corners: Corner[]): Corner {
  const sum = corners.reduce((acc, c) => ({ x: acc.x + c.x, y: acc.y + c.y }), { x: 0, y: 0 });
  return { x: sum.x / corners.length, y: sum.y / corners.length };
}

// Locate + decode a smart-target QR in a frame using OpenCV's QRCodeDetector.
// Always returns the QR geometry when one is found (so calibration can proceed
// even if decoding fails); returns `target` only when the content decodes to a
// Trackr `/t/{id}` URL, exposing the random id + inline calibration.
export function detectTargetInImageData(cv: CvApi, imageData: ImageData): TargetDetection {
  if (!cv?.QRCodeDetector) return EMPTY;

  let src: CvMat | null = null;
  let gray: CvMat | null = null;
  let points: CvMat | null = null;
  let detector: CvQRCodeDetector | null = null;
  try {
    src = cv.matFromImageData(imageData);
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    detector = new cv.QRCodeDetector();
    points = new cv.Mat();
    const text = detector.detectAndDecode(gray, points) ?? "";

    const corners = readCorners(points);
    if (corners.length !== 4) return EMPTY;

    const qrSidePx = meanSidePx(corners);
    if (!Number.isFinite(qrSidePx) || qrSidePx <= 0) return EMPTY;

    const qr: DetectedQr = { text, corners, center: centroid(corners), qrSidePx };

    const payload = text ? decodeTargetPayload(text) : null;
    if (!payload) return { qr, target: null };

    const pixelsPerInch =
      typeof payload.qrSizeValue === "number"
        ? pixelsPerInchFromQr(qrSidePx, payload.qrSizeValue, payload.unit)
        : null;

    return { qr, target: { id: payload.id, payload, qr, pixelsPerInch } };
  } catch {
    return EMPTY;
  } finally {
    points?.delete();
    gray?.delete();
    src?.delete();
    detector?.delete();
  }
}

// Convenience wrapper for a canvas already holding the frame.
export function detectTargetInCanvas(cv: CvApi, canvas: HTMLCanvasElement): TargetDetection {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || canvas.width <= 0 || canvas.height <= 0) return EMPTY;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return detectTargetInImageData(cv, imageData);
}
