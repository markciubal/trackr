export type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Calibration = {
  focalX: number;
  focalY: number;
  referenceWidthPx: number;
  referenceHeightPx: number;
  referenceDistance: number;
};

export const clamp = (value: number, min: number, max: number): number =>
  Math.min(Math.max(value, min), max);

export const rectFromPoints = (
  start: { x: number; y: number },
  end: { x: number; y: number },
): Rect => {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(start.x - end.x);
  const height = Math.abs(start.y - end.y);
  return { x, y, width, height };
};

export const pointInRect = (
  point: { x: number; y: number },
  rect: Rect,
): boolean =>
  point.x >= rect.x &&
  point.x <= rect.x + rect.width &&
  point.y >= rect.y &&
  point.y <= rect.y + rect.height;

export const parsePositiveNumber = (raw: string): number | null => {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value;
};

export const parseNonNegativeNumber = (raw: string): number | null => {
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  return value;
};

export const calibrateFromReference = (
  referenceRect: Rect,
  knownWidth: number,
  knownHeight: number,
  knownDistance: number,
): Calibration => ({
  focalX: (referenceRect.width * knownDistance) / knownWidth,
  focalY: (referenceRect.height * knownDistance) / knownHeight,
  referenceWidthPx: referenceRect.width,
  referenceHeightPx: referenceRect.height,
  referenceDistance: knownDistance,
});

const trigSizeFromPixels = (
  pixelSize: number,
  focalLength: number,
  distance: number,
): number => {
  const theta = 2 * Math.atan(pixelSize / (2 * focalLength));
  return 2 * distance * Math.tan(theta / 2);
};

export const estimateRealSize = (
  pixelWidth: number,
  pixelHeight: number,
  calibration: Calibration,
  distance: number,
): { width: number; height: number; diameter: number } => {
  const width = trigSizeFromPixels(pixelWidth, calibration.focalX, distance);
  const height = trigSizeFromPixels(pixelHeight, calibration.focalY, distance);
  return {
    width,
    height,
    diameter: (width + height) / 2,
  };
};

export const formatNumber = (value: number, digits = 3): string =>
  value.toFixed(digits);
