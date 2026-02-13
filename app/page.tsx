"use client";

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type CvMat = {
  rows: number;
  cols: number;
  data: Uint8Array;
  data32F: Float32Array;
  clone: () => CvMat;
  delete: () => void;
};

type CvApi = {
  Mat: new () => CvMat;
  MatVector: new () => {
    push_back: (mat: CvMat) => void;
    size?: () => number;
    get?: (index: number) => CvMat;
    delete: () => void;
  };
  Rect: new (x: number, y: number, width: number, height: number) => {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  TermCriteria: new (type: number, maxCount: number, epsilon: number) => unknown;
  imread: (input: HTMLCanvasElement | HTMLImageElement | string) => CvMat;
  cvtColor: (src: CvMat, dst: CvMat, code: number) => void;
  goodFeaturesToTrack?: (
    image: CvMat,
    corners: CvMat,
    maxCorners: number,
    qualityLevel: number,
    minDistance: number,
    mask?: CvMat,
    blockSize?: number,
    useHarrisDetector?: boolean,
    k?: number,
  ) => void;
  calcOpticalFlowPyrLK?: (
    prevImg: CvMat,
    nextImg: CvMat,
    prevPts: CvMat,
    nextPts: CvMat,
    status: CvMat,
    err: CvMat,
    winSize?: { width: number; height: number },
    maxLevel?: number,
    criteria?: unknown,
    flags?: number,
    minEigThreshold?: number,
  ) => void;
  matchTemplate: (image: CvMat, templ: CvMat, result: CvMat, method: number) => void;
  resize: (
    src: CvMat,
    dst: CvMat,
    dsize: { width: number; height: number },
    fx?: number,
    fy?: number,
    interpolation?: number,
  ) => void;
  calcHist: (
    images: { push_back: (mat: CvMat) => void; delete: () => void },
    channels: number[],
    mask: CvMat,
    hist: CvMat,
    histSize: number[],
    ranges: number[],
    accumulate?: boolean,
  ) => void;
  calcBackProject: (
    images: { push_back: (mat: CvMat) => void; delete: () => void },
    channels: number[],
    hist: CvMat,
    dst: CvMat,
    ranges: number[],
    scale: number,
  ) => void;
  normalize: (
    src: CvMat,
    dst: CvMat,
    alpha: number,
    beta: number,
    normType: number,
    dtype?: number,
    mask?: CvMat,
  ) => void;
  minMaxLoc: (src: CvMat) => { maxVal: number; maxLoc: { x: number; y: number } };
  findContours?: (
    image: CvMat,
    contours: { size?: () => number; get?: (index: number) => CvMat; delete: () => void },
    hierarchy: CvMat,
    mode: number,
    method: number,
  ) => void;
  contourArea?: (contour: CvMat, oriented?: boolean) => number;
  arcLength?: (curve: CvMat, closed: boolean) => number;
  boundingRect?: (contour: CvMat) => { x: number; y: number; width: number; height: number };
  minEnclosingCircle?: (points: CvMat) => { center: { x: number; y: number }; radius: number };
  moments?: (array: CvMat, binaryImage?: boolean) => { m00: number; m10: number; m01: number };
  Size: new (width: number, height: number) => { width: number; height: number };
  COLOR_RGBA2GRAY: number;
  TM_CCOEFF_NORMED: number;
  INTER_AREA: number;
  NORM_MINMAX: number;
  TermCriteria_COUNT: number;
  TermCriteria_EPS: number;
  RETR_EXTERNAL?: number;
  CHAIN_APPROX_SIMPLE?: number;
  meanShift?: (
    probImage: CvMat,
    window: { x: number; y: number; width: number; height: number },
    criteria: unknown,
  ) => number;
  CamShift?: (
    probImage: CvMat,
    window: { x: number; y: number; width: number; height: number },
    criteria: unknown,
  ) => unknown;
  onRuntimeInitialized?: () => void;
};

type DetectionLogEntry = {
  frame: number;
  videoTimeSec: number;
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  estimatedDistanceInches: number | null;
};

type ShotLogEntry = {
  id: string;
  shotNumber: number;
  frame: number;
  videoTimeSec: number;
  timeSincePreviousShotSec: number | null;
  windowStartSec: number;
  windowEndSec: number;
  centerX: number;
  centerY: number;
  radius: number;
  changedPixels: number;
  changeScore: number;
  estimatedDiameterInches: number | null;
  detectionMethod: "pixel_change";
  trackingMode: TrackingMode;
  detectionEnabled: boolean;
  detectionConfidencePct: number;
  detectionThresholdPct: number;
  spikeFocusedWindow: boolean;
  nearestSpikeId: string | null;
  nearestSpikeTimeSec: number | null;
  nearestSpikeDeltaSec: number | null;
  nearestSpikeStrength: number | null;
  patchWidthPx: number;
  patchHeightPx: number;
  drawRectX: number;
  drawRectY: number;
  drawRectWidth: number;
  drawRectHeight: number;
  centerPatchX: number;
  centerPatchY: number;
  spanWidthPx: number;
  spanHeightPx: number;
  estimatedDiameterPx: number;
  changedPixelRatioPct: number;
  brightPixelCount: number | null;
  transitionPixelCount: number | null;
  transitionPurityPct: number | null;
  minBlobPixelsThreshold: number;
  temporalHistoryFramesUsed: number;
  shotCooldownMs: number;
  residualMotionPx: number | null;
  trackedPointCount: number | null;
  estimatedRapidChangeMs: number | null;
  fastColorChangeTrigger: boolean;
  audioDecibelDbfs: number | null;
  audioDeltaFromMeanDb: number | null;
  audioPeakDbfs: number | null;
  audioCorrelationScorePct: number | null;
  nearestSpikeDecibelDbfs: number | null;
};

type RoiRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type TrackingMode = "template" | "meanshift" | "camshift";
type WorkflowStep = "upload_video" | "capture_frame" | "draw_geometry" | "calibrate" | "scan" | "export";

type HowlInstance = {
  seek: (seconds?: number) => number | void;
  play: (sprite?: string) => number;
  stop: () => void;
  unload: () => void;
};

type HowlConstructor = new (options: {
  src: string[];
  html5?: boolean;
  preload?: boolean;
  sprite?: Record<string, [number, number]>;
  onload?: () => void;
  onloaderror?: (_id: number, error: unknown) => void;
}) => HowlInstance;

type SpikeMetadata = {
  id: string;
  timeSec: number;
  strength: number;
  spriteStartMs: number;
  spriteDurationMs: number;
  windowStartSec: number;
  windowEndSec: number;
};

type AudioRmsSample = {
  timeSec: number;
  rms: number;
  dbfs: number;
};

type TimeWindow = {
  start: number;
  end: number;
};

type SpikeShotSummary = {
  shots: ShotLogEntry[];
  count: number;
  meanPointOfImpactX: number | null;
  meanPointOfImpactY: number | null;
  extremeSpreadPx: number | null;
  horizontalSpreadPx: number | null;
  verticalSpreadPx: number | null;
  meanRadiusPx: number | null;
  averageDiameterInches: number | null;
  averageChangeScore: number | null;
};

type ShotClusterSummary = {
  clusterId: number;
  shots: ShotLogEntry[];
  count: number;
  centroidX: number;
  centroidY: number;
  centroidTimeSec: number;
  extremeSpreadPx: number;
  timeSpanSec: number;
  meanTimeBetweenShotsSec: number | null;
  meanDiameterInches: number | null;
};

type ShotClusteringResult = {
  selectedK: number;
  finalK: number;
  closeMergeCount: number;
  objectiveScore: number;
  shotClusterById: Record<string, number>;
  clusters: ShotClusterSummary[];
};

type ClusterGeometry = {
  clusterId: number;
  points: Array<{ x: number; y: number }>;
  centroidX: number;
  centroidY: number;
};

type BinaryBlob = {
  pixelCount: number;
  centerX: number;
  centerY: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  indices: number[];
};

type ChangedContourRegion = {
  pixelCount: number;
  centerX: number;
  centerY: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  corePixels: number;
  blackToColorPixels: number;
  meanSeverity: number;
};

type ContourWindowRegionSnapshot = {
  pixelCount: number;
  centerX: number;
  centerY: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type BinaryMaskRun = {
  start: number;
  length: number;
};

type OpenCvActionState = "ok" | "warn" | "skip" | "idle";

type OpenCvActionStatus = {
  label: string;
  state: OpenCvActionState;
  detail: string;
};

type ContourWindowFrameSnapshot = {
  frame: number;
  videoTimeSec: number;
  patchWidthPx: number;
  patchHeightPx: number;
  changedPixels: number;
  rawMaskRuns: BinaryMaskRun[];
  temporalMaskRuns: BinaryMaskRun[];
  effectiveMaskRuns: BinaryMaskRun[];
  regions: ContourWindowRegionSnapshot[];
  actionStatuses: OpenCvActionStatus[];
};

type PendingShotCandidate = {
  centerX: number;
  centerY: number;
  firstSeenAtMs: number;
  lastSeenAtMs: number;
  bestEntry: ShotLogEntry;
  bestScore: number;
  confirmed: boolean;
};

type LinearUnit = "in" | "ft" | "m" | "cm" | "mm";

const TEMPLATE_REGION_DATA_URL_KEY = "trackr-template-region-data-url";
const TEMPLATE_REGION_IMAGE_NAME_KEY = "trackr-template-region-image-name";
const TEMPLATE_REGION_RECT_KEY = "trackr-template-region-rect";
const PROCESSED_CONTOUR_SNAPSHOT_STORAGE_KEY = "trackr-processed-contour-snapshot-v1";
const GEARS_SETTINGS_STORAGE_KEY = "trackr-gears-settings-v1";
const LINEAR_UNIT_INCH_FACTORS: Record<LinearUnit, number> = {
  in: 1,
  ft: 12,
  m: 39.37007874015748,
  cm: 0.3937007874015748,
  mm: 0.03937007874015748,
};
const LINEAR_UNIT_LABELS: Record<LinearUnit, string> = {
  in: "in",
  ft: "ft",
  m: "m",
  cm: "cm",
  mm: "mm",
};
const LINEAR_INPUT_STEPS: Record<LinearUnit, number> = {
  in: 0.01,
  ft: 0.01,
  m: 0.001,
  cm: 0.1,
  mm: 1,
};

type TweakSettings = {
  templateFitMarginRatio: number;
  centerProbeCoverageRatio: number;
  centerAcceptThresholdMinPct: number;
  centerAcceptThresholdScale: number;
  trackerWidthRatioMin: number;
  trackerWidthRatioMax: number;
  trackerHeightRatioMin: number;
  trackerHeightRatioMax: number;
  trackerMaxCenterDistanceRatio: number;
  spikeIntensiveFocusSec: number;
  spikeIntensiveHistoryFrames: number;
  spikeStandardProbeSize: number;
  spikeIntensiveProbeSize: number;
  spikePatchHistoryMaxFrames: number;
  spikeWindowHalfSec: number;
  spikeSpriteLeadSec: number;
  spikeSpriteDurationMs: number;
  templateTrackerTermCriteriaMaxCount: number;
  templateTrackerTermCriteriaEpsilon: number;
  opticalFlowTermCriteriaMaxCount: number;
  opticalFlowTermCriteriaEpsilon: number;
  opticalFlowWindowSizePx: number;
  shotHistoryMaxCount: number;
  playbackCurrentShotWindowSec: number;
  darkPixelLuminanceMax: number;
  darkPixelChannelMax: number;
  brightPixelMinMaxChannel: number;
  brightPixelMinSaturation: number;
  brightDominanceDelta: number;
  brightGreenMinChannel: number;
  brightRedMinChannel: number;
  brightYellowMinRed: number;
  brightYellowMinGreen: number;
  brightYellowMaxBlue: number;
  brightOrangeMinRed: number;
  brightOrangeMinGreen: number;
  brightOrangeMaxGreen: number;
  brightOrangeMaxBlue: number;
  genericBrightMinChannel: number;
  genericBrightMinSaturation: number;
  transitionLuminanceJumpMin: number;
  temporalDarkVoteRatio: number;
  temporalLuminanceJumpMin: number;
  rapidColorDeltaMin: number;
  rapidLuminanceDeltaMin: number;
  rapidCurrentLuminanceMin: number;
  rapidCurrentMaxChannelMin: number;
  subMillisecondMinRapidPixels: number;
  subMillisecondBypassQualityGate: number;
  minVisibleHitChangeScore: number;
  maxShotDiameterInches: number;
  fallbackMaxDiameterRatio: number;
  fallbackMaxDiameterMinPx: number;
  shotCooldownMs: number;
  shotDetectionThresholdScale: number;
  shotDetectionThresholdMinPct: number;
  minBlobBaseIntensiveFactor: number;
  minBlobBaseStandardFactor: number;
  minBlobFallbackIntensive: number;
  minBlobFallbackStandard: number;
  transitionPurityScoreWeight: number;
  colorSpikeDeltaMaxSec: number;
  colorMinConfidencePct: number;
  colorChangedPixelsMin: number;
  colorChangedPixelsMax: number;
  colorDiameterMinInches: number;
  colorDiameterMaxInches: number;
  colorDiameterMinPx: number;
  colorDiameterMaxPx: number;
  colorAspectRatioMax: number;
  opticalGoodFeaturesMaxCorners: number;
  opticalGoodFeaturesQualityLevel: number;
  opticalGoodFeaturesMinDistance: number;
  opticalResidualThresholdPx: number;
  opticalSignificantMinPoints: number;
  opticalSignificantMinResidual: number;
  opticalSpikeDeltaMaxSec: number;
  opticalTrackedPointsMin: number;
  opticalResidualMotionMin: number;
  opticalResidualMotionMax: number;
  opticalChangedRatioMaxPct: number;
  audioWindowSize: number;
  audioSpikeStdDevMultiplier: number;
  audioSpikeMinGapSec: number;
  audioPeakWindowHalfSec: number;
  audioEnergyOffsetDb: number;
  audioEnergyScaleDb: number;
  audioNoSpikeAlignmentScore: number;
  audioWeightEnergy: number;
  audioWeightTimeAlignment: number;
  audioWeightVisual: number;
  audioVisualScoreFloor: number;
  colorAudioCorrelationMinPct: number;
  colorAudioDeltaMinDb: number;
  opticalAudioCorrelationMinPct: number;
  opticalAudioDeltaMinDb: number;
  kmeansTimeWeight: number;
  kmeansMaxClustersCap: number;
  kmeansComplexityPenalty: number;
  kmeansClosePenaltyWeight: number;
  kmeansSsePenaltyWeight: number;
  kmeansCentroidCloseDistance: number;
  kmeansMergeCombinedDistanceMax: number;
  kmeansMergeSpatialDistanceMax: number;
  kmeansMergeTimeDistanceMax: number;
  kmeansMaxIterations: number;
};

const DEFAULT_TWEAK_SETTINGS: TweakSettings = {
  templateFitMarginRatio: 0.95,
  centerProbeCoverageRatio: 0.58,
  centerAcceptThresholdMinPct: 20,
  centerAcceptThresholdScale: 0.85,
  trackerWidthRatioMin: 0.5,
  trackerWidthRatioMax: 2.5,
  trackerHeightRatioMin: 0.5,
  trackerHeightRatioMax: 2.5,
  trackerMaxCenterDistanceRatio: 0.28,
  spikeIntensiveFocusSec: 1.2,
  spikeIntensiveHistoryFrames: 4,
  spikeStandardProbeSize: 192,
  spikeIntensiveProbeSize: 256,
  spikePatchHistoryMaxFrames: 10,
  spikeWindowHalfSec: 1.4,
  spikeSpriteLeadSec: 0.2,
  spikeSpriteDurationMs: 450,
  templateTrackerTermCriteriaMaxCount: 10,
  templateTrackerTermCriteriaEpsilon: 1,
  opticalFlowTermCriteriaMaxCount: 30,
  opticalFlowTermCriteriaEpsilon: 0.03,
  opticalFlowWindowSizePx: 15,
  shotHistoryMaxCount: 120,
  playbackCurrentShotWindowSec: 0.18,
  darkPixelLuminanceMax: 96,
  darkPixelChannelMax: 128,
  brightPixelMinMaxChannel: 112,
  brightPixelMinSaturation: 18,
  brightDominanceDelta: 8,
  brightGreenMinChannel: 98,
  brightRedMinChannel: 98,
  brightYellowMinRed: 116,
  brightYellowMinGreen: 98,
  brightYellowMaxBlue: 182,
  brightOrangeMinRed: 114,
  brightOrangeMinGreen: 66,
  brightOrangeMaxGreen: 238,
  brightOrangeMaxBlue: 156,
  genericBrightMinChannel: 176,
  genericBrightMinSaturation: 26,
  transitionLuminanceJumpMin: 14,
  temporalDarkVoteRatio: 0.33,
  temporalLuminanceJumpMin: 16,
  rapidColorDeltaMin: 7,
  rapidLuminanceDeltaMin: 1,
  rapidCurrentLuminanceMin: 8,
  rapidCurrentMaxChannelMin: 18,
  subMillisecondMinRapidPixels: 1,
  subMillisecondBypassQualityGate: 0,
  minVisibleHitChangeScore: 0.03,
  maxShotDiameterInches: 2.35,
  fallbackMaxDiameterRatio: 0.18,
  fallbackMaxDiameterMinPx: 4,
  shotCooldownMs: 70,
  shotDetectionThresholdScale: 0.4,
  shotDetectionThresholdMinPct: 5,
  minBlobBaseIntensiveFactor: 0.00003,
  minBlobBaseStandardFactor: 0.00005,
  minBlobFallbackIntensive: 2,
  minBlobFallbackStandard: 4,
  transitionPurityScoreWeight: 0.25,
  colorSpikeDeltaMaxSec: 1.8,
  colorMinConfidencePct: 8,
  colorChangedPixelsMin: 1,
  colorChangedPixelsMax: 6500,
  colorDiameterMinInches: 0.02,
  colorDiameterMaxInches: 2.3,
  colorDiameterMinPx: 2,
  colorDiameterMaxPx: 560,
  colorAspectRatioMax: 8.8,
  opticalGoodFeaturesMaxCorners: 260,
  opticalGoodFeaturesQualityLevel: 0.004,
  opticalGoodFeaturesMinDistance: 2,
  opticalResidualThresholdPx: 0.08,
  opticalSignificantMinPoints: 1,
  opticalSignificantMinResidual: 0.08,
  opticalSpikeDeltaMaxSec: 0.5,
  opticalTrackedPointsMin: 40,
  opticalResidualMotionMin: 0.05,
  opticalResidualMotionMax: 3.5,
  opticalChangedRatioMaxPct: 0.5,
  audioWindowSize: 2048,
  audioSpikeStdDevMultiplier: 2.5,
  audioSpikeMinGapSec: 0.35,
  audioPeakWindowHalfSec: 0.09,
  audioEnergyOffsetDb: 6,
  audioEnergyScaleDb: 18,
  audioNoSpikeAlignmentScore: 0.45,
  audioWeightEnergy: 0.52,
  audioWeightTimeAlignment: 0.28,
  audioWeightVisual: 0.2,
  audioVisualScoreFloor: 0.5,
  colorAudioCorrelationMinPct: 0,
  colorAudioDeltaMinDb: -30,
  opticalAudioCorrelationMinPct: 0,
  opticalAudioDeltaMinDb: -30,
  kmeansTimeWeight: 1.15,
  kmeansMaxClustersCap: 8,
  kmeansComplexityPenalty: 0.035,
  kmeansClosePenaltyWeight: 0.16,
  kmeansSsePenaltyWeight: 0.012,
  kmeansCentroidCloseDistance: 1.1,
  kmeansMergeCombinedDistanceMax: 1.05,
  kmeansMergeSpatialDistanceMax: 0.95,
  kmeansMergeTimeDistanceMax: 0.8,
  kmeansMaxIterations: 30,
};

type StoredGearsSettings = {
  unitConversionEnabled?: boolean;
  displayLinearUnit?: LinearUnit;
  tweakSettings?: Partial<TweakSettings>;
  isGearsExpanded?: boolean;
};

function isLinearUnit(value: unknown): value is LinearUnit {
  return typeof value === "string" && value in LINEAR_UNIT_LABELS;
}

function sanitizeStoredTweakSettings(value: unknown): Partial<TweakSettings> | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Partial<Record<keyof TweakSettings, unknown>>;
  const sanitized: Partial<TweakSettings> = {};
  for (const key of Object.keys(DEFAULT_TWEAK_SETTINGS) as Array<keyof TweakSettings>) {
    const candidate = raw[key];
    if (typeof candidate !== "number" || !Number.isFinite(candidate)) continue;
    sanitized[key] = candidate;
  }
  return sanitized;
}

type TweakFieldConfig = {
  key: keyof TweakSettings;
  label: string;
  min?: number;
  max?: number;
  step: number;
};

type TweakCategoryConfig = {
  title: string;
  fields: TweakFieldConfig[];
};

const TWEAK_CATEGORIES: TweakCategoryConfig[] = [
  {
    title: "Template and Windows",
    fields: [
      { key: "templateFitMarginRatio", label: "Template Fit Margin Ratio", min: 0.5, max: 1, step: 0.01 },
      { key: "centerProbeCoverageRatio", label: "Center Probe Coverage", min: 0.2, max: 1, step: 0.01 },
      { key: "centerAcceptThresholdMinPct", label: "Center Accept Min (%)", min: 0, max: 100, step: 1 },
      { key: "centerAcceptThresholdScale", label: "Center Accept Scale", min: 0.1, max: 1.5, step: 0.01 },
      { key: "spikeIntensiveFocusSec", label: "Spike Focus Window (s)", min: 0.05, max: 3, step: 0.01 },
      { key: "spikeIntensiveHistoryFrames", label: "Intensive History Frames", min: 1, max: 30, step: 1 },
      { key: "spikeStandardProbeSize", label: "Standard Probe Size (px)", min: 32, max: 512, step: 1 },
      { key: "spikeIntensiveProbeSize", label: "Intensive Probe Size (px)", min: 32, max: 512, step: 1 },
      { key: "spikePatchHistoryMaxFrames", label: "Patch History Max Frames", min: 1, max: 60, step: 1 },
      { key: "spikeWindowHalfSec", label: "Spike Window Half Size (s)", min: 0, max: 10, step: 0.01 },
      { key: "spikeSpriteLeadSec", label: "Spike Sprite Lead (s)", min: 0, max: 2, step: 0.01 },
      { key: "spikeSpriteDurationMs", label: "Spike Sprite Duration (ms)", min: 50, max: 5000, step: 1 },
      { key: "templateTrackerTermCriteriaMaxCount", label: "Template Term Max Count", min: 1, max: 200, step: 1 },
      { key: "templateTrackerTermCriteriaEpsilon", label: "Template Term Epsilon", min: 0.0001, max: 10, step: 0.0001 },
      { key: "opticalFlowTermCriteriaMaxCount", label: "Optical Term Max Count", min: 1, max: 200, step: 1 },
      { key: "opticalFlowTermCriteriaEpsilon", label: "Optical Term Epsilon", min: 0.0001, max: 10, step: 0.0001 },
      { key: "opticalFlowWindowSizePx", label: "Optical Window Size (px)", min: 3, max: 128, step: 1 },
      { key: "shotHistoryMaxCount", label: "Shot History Max Count", min: 1, max: 5000, step: 1 },
      { key: "playbackCurrentShotWindowSec", label: "Playback Current Shot Window (s)", min: 0.01, max: 2, step: 0.01 },
      { key: "shotCooldownMs", label: "Shot Cooldown (ms)", min: 0, max: 2000, step: 1 },
    ],
  },
  {
    title: "Color and Transition",
    fields: [
      { key: "darkPixelLuminanceMax", label: "Dark Pixel Luminance Max", min: 0, max: 255, step: 1 },
      { key: "darkPixelChannelMax", label: "Dark Pixel Channel Max", min: 0, max: 255, step: 1 },
      { key: "brightPixelMinMaxChannel", label: "Bright Pixel Max Channel Min", min: 0, max: 255, step: 1 },
      { key: "brightPixelMinSaturation", label: "Bright Pixel Saturation Min", min: 0, max: 255, step: 1 },
      { key: "brightDominanceDelta", label: "Dominance Delta", min: 0, max: 64, step: 1 },
      { key: "brightGreenMinChannel", label: "Green Min Channel", min: 0, max: 255, step: 1 },
      { key: "brightRedMinChannel", label: "Red Min Channel", min: 0, max: 255, step: 1 },
      { key: "brightYellowMinRed", label: "Yellow Min Red", min: 0, max: 255, step: 1 },
      { key: "brightYellowMinGreen", label: "Yellow Min Green", min: 0, max: 255, step: 1 },
      { key: "brightYellowMaxBlue", label: "Yellow Max Blue", min: 0, max: 255, step: 1 },
      { key: "brightOrangeMinRed", label: "Orange Min Red", min: 0, max: 255, step: 1 },
      { key: "brightOrangeMinGreen", label: "Orange Min Green", min: 0, max: 255, step: 1 },
      { key: "brightOrangeMaxGreen", label: "Orange Max Green", min: 0, max: 255, step: 1 },
      { key: "brightOrangeMaxBlue", label: "Orange Max Blue", min: 0, max: 255, step: 1 },
      { key: "genericBrightMinChannel", label: "Generic Bright Min Channel", min: 0, max: 255, step: 1 },
      { key: "genericBrightMinSaturation", label: "Generic Bright Min Saturation", min: 0, max: 255, step: 1 },
      { key: "transitionLuminanceJumpMin", label: "Luminance Jump Min", min: 0, max: 255, step: 1 },
      { key: "temporalDarkVoteRatio", label: "Temporal Dark Vote Ratio", min: 0, max: 1, step: 0.01 },
      { key: "temporalLuminanceJumpMin", label: "Temporal Luminance Jump Min", min: 0, max: 255, step: 1 },
      { key: "rapidColorDeltaMin", label: "Rapid Color Delta Min", min: 0, max: 765, step: 1 },
      { key: "rapidLuminanceDeltaMin", label: "Rapid Luminance Delta Min", min: 0, max: 255, step: 1 },
      { key: "rapidCurrentLuminanceMin", label: "Rapid Current Luminance Min", min: 0, max: 255, step: 1 },
      { key: "rapidCurrentMaxChannelMin", label: "Rapid Current Max Channel Min", min: 0, max: 255, step: 1 },
      { key: "subMillisecondMinRapidPixels", label: "Fast Trigger Min Rapid Pixels", min: 1, max: 5000, step: 1 },
      { key: "subMillisecondBypassQualityGate", label: "Fast Trigger Bypass Quality (0/1)", min: 0, max: 1, step: 1 },
      { key: "transitionPurityScoreWeight", label: "Transition Purity Score Weight", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    title: "Shot Gate",
    fields: [
      { key: "minVisibleHitChangeScore", label: "Min Visible Hit Change Score", min: 0, max: 10, step: 0.01 },
      { key: "maxShotDiameterInches", label: "Max Shot Diameter (in)", min: 0.1, max: 10, step: 0.01 },
      { key: "fallbackMaxDiameterRatio", label: "Fallback Max Diameter Ratio", min: 0.01, max: 1, step: 0.01 },
      { key: "fallbackMaxDiameterMinPx", label: "Fallback Max Diameter Min (px)", min: 1, max: 100, step: 1 },
      { key: "shotDetectionThresholdScale", label: "Shot Detection Threshold Scale", min: 0, max: 2, step: 0.01 },
      { key: "shotDetectionThresholdMinPct", label: "Shot Detection Threshold Min (%)", min: 0, max: 100, step: 0.1 },
      { key: "colorSpikeDeltaMaxSec", label: "Color Spike Delta Max (s)", min: 0, max: 2, step: 0.01 },
      { key: "colorMinConfidencePct", label: "Color Min Confidence (%)", min: 0, max: 100, step: 0.1 },
      { key: "colorChangedPixelsMin", label: "Color Changed Pixels Min", min: 0, max: 5000, step: 1 },
      { key: "colorChangedPixelsMax", label: "Color Changed Pixels Max", min: 0, max: 10000, step: 1 },
      { key: "colorDiameterMinInches", label: "Color Diameter Min (in)", min: 0, max: 5, step: 0.01 },
      { key: "colorDiameterMaxInches", label: "Color Diameter Max (in)", min: 0.1, max: 10, step: 0.01 },
      { key: "colorDiameterMinPx", label: "Color Diameter Min (px)", min: 0, max: 500, step: 1 },
      { key: "colorDiameterMaxPx", label: "Color Diameter Max (px)", min: 1, max: 2000, step: 1 },
      { key: "colorAspectRatioMax", label: "Color Aspect Ratio Max", min: 1, max: 10, step: 0.01 },
      { key: "minBlobBaseIntensiveFactor", label: "Min Blob Intensive Factor", min: 0, max: 0.05, step: 0.00001 },
      { key: "minBlobBaseStandardFactor", label: "Min Blob Standard Factor", min: 0, max: 0.05, step: 0.00001 },
      { key: "minBlobFallbackIntensive", label: "Min Blob Fallback Intensive", min: 0, max: 200, step: 1 },
      { key: "minBlobFallbackStandard", label: "Min Blob Fallback Standard", min: 0, max: 200, step: 1 },
    ],
  },
  {
    title: "Optical Flow and Tracker",
    fields: [
      { key: "opticalGoodFeaturesMaxCorners", label: "Good Features Max Corners", min: 1, max: 2000, step: 1 },
      { key: "opticalGoodFeaturesQualityLevel", label: "Good Features Quality Level", min: 0.0001, max: 1, step: 0.0001 },
      { key: "opticalGoodFeaturesMinDistance", label: "Good Features Min Distance", min: 0, max: 100, step: 0.1 },
      { key: "opticalResidualThresholdPx", label: "Optical Residual Threshold (px)", min: 0, max: 10, step: 0.01 },
      { key: "opticalSignificantMinPoints", label: "Optical Significant Min Points", min: 1, max: 1000, step: 1 },
      { key: "opticalSignificantMinResidual", label: "Optical Significant Min Residual", min: 0, max: 10, step: 0.01 },
      { key: "opticalSpikeDeltaMaxSec", label: "Optical Spike Delta Max (s)", min: 0, max: 2, step: 0.01 },
      { key: "opticalTrackedPointsMin", label: "Optical Tracked Points Min", min: 0, max: 5000, step: 1 },
      { key: "opticalResidualMotionMin", label: "Optical Residual Motion Min", min: 0, max: 10, step: 0.01 },
      { key: "opticalResidualMotionMax", label: "Optical Residual Motion Max", min: 0, max: 20, step: 0.01 },
      { key: "opticalChangedRatioMaxPct", label: "Optical Changed Ratio Max (%)", min: 0, max: 100, step: 0.01 },
      { key: "trackerWidthRatioMin", label: "Tracker Width Ratio Min", min: 0, max: 5, step: 0.01 },
      { key: "trackerWidthRatioMax", label: "Tracker Width Ratio Max", min: 0.1, max: 10, step: 0.01 },
      { key: "trackerHeightRatioMin", label: "Tracker Height Ratio Min", min: 0, max: 5, step: 0.01 },
      { key: "trackerHeightRatioMax", label: "Tracker Height Ratio Max", min: 0.1, max: 10, step: 0.01 },
      { key: "trackerMaxCenterDistanceRatio", label: "Tracker Max Center Dist Ratio", min: 0, max: 1, step: 0.01 },
    ],
  },
  {
    title: "Audio Correlation",
    fields: [
      { key: "audioWindowSize", label: "Audio RMS Window Size", min: 64, max: 16384, step: 1 },
      { key: "audioSpikeStdDevMultiplier", label: "Audio Spike StdDev Multiplier", min: 0.1, max: 10, step: 0.01 },
      { key: "audioSpikeMinGapSec", label: "Audio Spike Min Gap (s)", min: 0, max: 2, step: 0.01 },
      { key: "audioPeakWindowHalfSec", label: "Audio Peak Half Window (s)", min: 0, max: 1, step: 0.005 },
      { key: "audioEnergyOffsetDb", label: "Audio Energy Offset (dB)", min: -40, max: 40, step: 0.1 },
      { key: "audioEnergyScaleDb", label: "Audio Energy Scale (dB)", min: 0.1, max: 80, step: 0.1 },
      { key: "audioNoSpikeAlignmentScore", label: "No-Spike Alignment Score", min: 0, max: 1, step: 0.01 },
      { key: "audioWeightEnergy", label: "Audio Weight Energy", min: 0, max: 1, step: 0.01 },
      { key: "audioWeightTimeAlignment", label: "Audio Weight Time", min: 0, max: 1, step: 0.01 },
      { key: "audioWeightVisual", label: "Audio Weight Visual", min: 0, max: 1, step: 0.01 },
      { key: "audioVisualScoreFloor", label: "Audio Visual Score Floor", min: 0.01, max: 10, step: 0.01 },
      { key: "colorAudioCorrelationMinPct", label: "Color Audio Correlation Min (%)", min: 0, max: 100, step: 0.1 },
      { key: "colorAudioDeltaMinDb", label: "Color Audio Delta Min (dB)", min: -80, max: 20, step: 0.1 },
      { key: "opticalAudioCorrelationMinPct", label: "Optical Audio Correlation Min (%)", min: 0, max: 100, step: 0.1 },
      { key: "opticalAudioDeltaMinDb", label: "Optical Audio Delta Min (dB)", min: -80, max: 20, step: 0.1 },
    ],
  },
  {
    title: "K-Means Clustering",
    fields: [
      { key: "kmeansTimeWeight", label: "K-Means Time Weight", min: 0.1, max: 10, step: 0.01 },
      { key: "kmeansMaxClustersCap", label: "K-Means Max Clusters Cap", min: 1, max: 50, step: 1 },
      { key: "kmeansComplexityPenalty", label: "K-Means Complexity Penalty", min: 0, max: 1, step: 0.001 },
      { key: "kmeansClosePenaltyWeight", label: "K-Means Close Penalty Weight", min: 0, max: 5, step: 0.001 },
      { key: "kmeansSsePenaltyWeight", label: "K-Means SSE Penalty Weight", min: 0, max: 5, step: 0.001 },
      { key: "kmeansCentroidCloseDistance", label: "Centroid Close Distance", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeCombinedDistanceMax", label: "Merge Combined Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeSpatialDistanceMax", label: "Merge Spatial Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeTimeDistanceMax", label: "Merge Time Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMaxIterations", label: "K-Means Max Iterations", min: 1, max: 500, step: 1 },
    ],
  },
];

const TWEAK_UI_KEYS = TWEAK_CATEGORIES.flatMap((category) => category.fields.map((field) => field.key));
const TWEAK_UI_KEY_SET = new Set<keyof TweakSettings>(TWEAK_UI_KEYS);
const MISSING_TWEAK_UI_KEYS = (Object.keys(DEFAULT_TWEAK_SETTINGS) as Array<keyof TweakSettings>).filter(
  (key) => !TWEAK_UI_KEY_SET.has(key),
);
const DUPLICATE_TWEAK_UI_KEYS = TWEAK_UI_KEYS.filter((key, index) => TWEAK_UI_KEYS.indexOf(key) !== index);
if (process.env.NODE_ENV !== "production" && (MISSING_TWEAK_UI_KEYS.length > 0 || DUPLICATE_TWEAK_UI_KEYS.length > 0)) {
  const missing = MISSING_TWEAK_UI_KEYS.length > 0 ? `Missing in settings UI: ${MISSING_TWEAK_UI_KEYS.join(", ")}` : "";
  const duplicates =
    DUPLICATE_TWEAK_UI_KEYS.length > 0 ? `Duplicate settings keys: ${Array.from(new Set(DUPLICATE_TWEAK_UI_KEYS)).join(", ")}` : "";
  const details = [missing, duplicates].filter(Boolean).join(" | ");
  throw new Error(`Gears and Tweaks coverage issue. ${details}`);
}

function convertFromInches(valueInches: number, unit: LinearUnit): number {
  return valueInches / LINEAR_UNIT_INCH_FACTORS[unit];
}

function convertToInches(valueInUnit: number, unit: LinearUnit): number {
  return valueInUnit * LINEAR_UNIT_INCH_FACTORS[unit];
}

function UnitConverterSettings({
  enabled,
  onEnabledChange,
  unit,
  onUnitChange,
}: {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  unit: LinearUnit;
  onUnitChange: (unit: LinearUnit) => void;
}) {
  return (
    <div className="rounded-md border border-gray-700 bg-black p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-gray-300">Units</p>
        <label className="flex items-center gap-2 text-xs text-gray-200">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => onEnabledChange(event.target.checked)}
          />
          Enable Unit Conversion
        </label>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-[11px] text-gray-400">Display Unit</span>
        <select
          value={unit}
          onChange={(event) => onUnitChange(event.target.value as LinearUnit)}
          disabled={!enabled}
          className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1 text-sm text-white disabled:opacity-60"
        >
          <option value="in">Inches (in)</option>
          <option value="ft">Feet (ft)</option>
          <option value="m">Meters (m)</option>
          <option value="cm">Centimeters (cm)</option>
          <option value="mm">Millimeters (mm)</option>
        </select>
        <p className="text-[11px] text-gray-400">
          Keep internal calculations in inches and convert values across the interface.
        </p>
      </div>
    </div>
  );
}

function GearsAndTweaksSection({
  values,
  onValueChange,
  onReset,
  changedCount,
}: {
  values: TweakSettings;
  onValueChange: (key: keyof TweakSettings, value: number) => void;
  onReset: () => void;
  changedCount: number;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-gray-400">
          {changedCount > 0 ? `${changedCount} settings changed from defaults.` : "Using default settings."}
        </p>
        <button
          type="button"
          onClick={onReset}
          className="rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-200 transition hover:bg-neutral-800"
        >
          Reset Defaults
        </button>
      </div>
      <p className="rounded-md border border-gray-700 bg-neutral-900 px-3 py-2 text-xs text-gray-200">
        Warning: changing these settings can materially alter detection behavior and make runs less comparable.
      </p>
      {changedCount > 0 ? (
        <p className="rounded-md border border-gray-700 bg-neutral-900 px-3 py-2 text-xs text-gray-200">
          Warning: custom tuning is active. Validate results before relying on shot analytics.
        </p>
      ) : null}
      <div className="space-y-3">
        {TWEAK_CATEGORIES.map((category) => (
          <details key={category.title} className="rounded-md border border-gray-700 bg-black p-3" open>
            <summary className="cursor-pointer text-xs uppercase tracking-wide text-gray-300">{category.title}</summary>
            <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {category.fields.map((field) => (
                <label key={String(field.key)} className="flex flex-col gap-1">
                  <span className="text-[11px] text-gray-400">{field.label}</span>
                  <input
                    type="number"
                    value={values[field.key]}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    onChange={(event) => onValueChange(field.key, Number(event.target.value))}
                    className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                </label>
              ))}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}

const MAX_GLOBAL_CHANGE_RATIO_PCT = 46;
const MIN_EVENT_CHANGE_RATIO_PCT = 0.005;
const MAX_EVENT_CHANGE_RATIO_PCT = 55;
const SHOT_PERSISTENCE_MIN_MS = 1000;
const SHOT_PERSISTENCE_MAX_GAP_MS = 850;
const SIMPLE_CONTOUR_DIFF_THRESHOLD = 50;
const SIMPLE_CONTOUR_OPEN_KERNEL_SIZE = 5;
const SIMPLE_CONTOUR_MIN_AREA = 10;
const SIMPLE_CONTOUR_MAX_AREA = 500;
const SIMPLE_CONTOUR_CIRCULARITY_MIN = 0.5;
const SIMPLE_CONTOUR_CIRCULARITY_MAX = 2;
const SIMPLE_CONTOUR_MASK_HISTORY_MAX = 4;

function countMaskPixels(mask: Uint8Array): number {
  let count = 0;
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] === 1) count += 1;
  }
  return count;
}

function getNearestSpikeAtTime(timeSec: number, spikes: SpikeMetadata[]): { spike: SpikeMetadata | null; deltaSec: number } {
  if (spikes.length === 0) return { spike: null, deltaSec: Number.POSITIVE_INFINITY };

  let nearest: SpikeMetadata | null = null;
  let nearestDelta = Number.POSITIVE_INFINITY;
  for (const spike of spikes) {
    const delta = Math.abs(spike.timeSec - timeSec);
    if (delta < nearestDelta) {
      nearest = spike;
      nearestDelta = delta;
    }
  }
  return { spike: nearest, deltaSec: nearestDelta };
}

function estimatePatchBlobDiameter(
  pixelCount: number,
  spanX: number,
  spanY: number,
  patchWidth: number,
  patchHeight: number,
  drawWidth: number,
  drawHeight: number,
  pixelsPerInch: number,
): { diameterPx: number; diameterInches: number | null } {
  const safePixelCount = Math.max(1, pixelCount);
  const scaleX = drawWidth / Math.max(1, patchWidth);
  const scaleY = drawHeight / Math.max(1, patchHeight);
  const areaEquivalentDiameterPx = 2 * Math.sqrt((safePixelCount * scaleX * scaleY) / Math.PI);
  const spanEquivalentDiameterPx = Math.max(1, Math.max(spanX * scaleX, spanY * scaleY));
  const diameterPx = Math.max(areaEquivalentDiameterPx, spanEquivalentDiameterPx);
  const diameterInches = pixelsPerInch > 0 ? diameterPx / pixelsPerInch : null;
  return { diameterPx, diameterInches };
}

function isSubTwoInchShot(
  diameterPx: number,
  diameterInches: number | null,
  drawWidth: number,
  drawHeight: number,
  tweaks: TweakSettings,
): boolean {
  if (!Number.isFinite(diameterPx) || diameterPx <= 0) return false;
  const maxDiameterInches = Math.max(0.1, Math.max(2.3, tweaks.maxShotDiameterInches));
  if (diameterInches !== null) return diameterInches <= maxDiameterInches;
  const fallbackMaxPx = Math.max(
    Math.max(1, Math.max(4, tweaks.fallbackMaxDiameterMinPx)),
    Math.min(drawWidth, drawHeight) * Math.max(0.01, Math.max(0.18, tweaks.fallbackMaxDiameterRatio)),
  );
  return diameterPx <= fallbackMaxPx;
}

function spanAspectRatio(width: number, height: number): number {
  const longest = Math.max(width, height);
  const shortest = Math.max(1, Math.min(width, height));
  return longest / shortest;
}

function isLikelyGoodShot(entry: ShotLogEntry, tweaks: TweakSettings): boolean {
  // Keep detections constrained to their active spike window/time gate.
  if (entry.videoTimeSec < entry.windowStartSec || entry.videoTimeSec > entry.windowEndSec) return false;
  if (entry.windowEndSec < entry.windowStartSec) return false;
  if (entry.nearestSpikeTimeSec !== null) {
    const spikeDelta = Math.abs(entry.videoTimeSec - entry.nearestSpikeTimeSec);
    if (spikeDelta > Math.max(0.05, tweaks.colorSpikeDeltaMaxSec)) return false;
  }
  if (
    entry.nearestSpikeId !== null &&
    (entry.nearestSpikeDeltaSec === null || entry.nearestSpikeDeltaSec > Math.max(0.05, tweaks.colorSpikeDeltaMaxSec))
  ) {
    return false;
  }

  if (entry.changedPixels <= 0) return false;
  const minVisibleScore = Math.max(0, Math.min(0.03, tweaks.minVisibleHitChangeScore));
  if (entry.changeScore < minVisibleScore) return false;
  const minChangedPixels = Math.max(1, Math.round(Math.min(1, tweaks.colorChangedPixelsMin)));
  const maxChangedPixels = Math.max(minChangedPixels, Math.round(Math.max(6500, tweaks.colorChangedPixelsMax)));
  if (entry.changedPixels < minChangedPixels || entry.changedPixels > maxChangedPixels) return false;
  if (entry.changedPixelRatioPct < MIN_EVENT_CHANGE_RATIO_PCT || entry.changedPixelRatioPct > MAX_EVENT_CHANGE_RATIO_PCT) {
    return false;
  }
  const maxShotDiameterInches = Math.max(0.1, Math.max(2.3, tweaks.maxShotDiameterInches));
  if (entry.estimatedDiameterInches !== null) {
    const minDiameterInches = Math.max(0, Math.min(0.02, tweaks.colorDiameterMinInches));
    const maxDiameterInches = Math.max(0.1, Math.max(2.3, tweaks.colorDiameterMaxInches));
    if (
      entry.estimatedDiameterInches < minDiameterInches ||
      entry.estimatedDiameterInches > maxDiameterInches
    ) {
      return false;
    }
    if (entry.estimatedDiameterInches > maxShotDiameterInches) return false;
  } else {
    const minDiameterPx = Math.max(0, Math.min(2, tweaks.colorDiameterMinPx));
    const maxDiameterPx = Math.max(minDiameterPx + 1, Math.max(560, tweaks.colorDiameterMaxPx));
    if (entry.estimatedDiameterPx < minDiameterPx || entry.estimatedDiameterPx > maxDiameterPx) return false;
    const fallbackMaxPx = Math.max(
      Math.max(1, Math.max(4, tweaks.fallbackMaxDiameterMinPx)),
      Math.min(entry.drawRectWidth, entry.drawRectHeight) * Math.max(0.01, Math.max(0.18, tweaks.fallbackMaxDiameterRatio)),
    );
    if (entry.estimatedDiameterPx > fallbackMaxPx) return false;
  }
  const aspectRatioOk =
    spanAspectRatio(entry.spanWidthPx, entry.spanHeightPx) <= Math.max(1, Math.max(8.8, tweaks.colorAspectRatioMax));
  if (!aspectRatioOk) return false;
  if (entry.transitionPurityPct !== null) {
    const minTemporalPurityPct = Math.max(20, Math.min(95, tweaks.temporalDarkVoteRatio * 100));
    if (entry.transitionPurityPct < minTemporalPurityPct) return false;
  }
  if (entry.audioCorrelationScorePct !== null && entry.audioCorrelationScorePct < Math.max(0, tweaks.colorAudioCorrelationMinPct)) {
    return false;
  }
  if (entry.audioDeltaFromMeanDb !== null && entry.audioDeltaFromMeanDb < tweaks.colorAudioDeltaMinDb) return false;
  return true;
}

type NormalizedShotPoint = {
  shot: ShotLogEntry;
  nx: number;
  ny: number;
  nt: number;
};

type KMeansModel = {
  k: number;
  assignments: number[];
  centroids: Array<{ x: number; y: number; t: number }>;
  sse: number;
};

function euclideanDistance3(aX: number, aY: number, aT: number, bX: number, bY: number, bT: number): number {
  return Math.hypot(aX - bX, aY - bY, aT - bT);
}

function meanAndStd(values: number[]): { mean: number; std: number } {
  if (values.length === 0) return { mean: 0, std: 1 };
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  return { mean, std: std > 1e-6 ? std : 1 };
}

function initializeKMeansCentroids(points: NormalizedShotPoint[], k: number): Array<{ x: number; y: number; t: number }> {
  const centroids: Array<{ x: number; y: number; t: number }> = [];
  if (points.length === 0 || k <= 0) return centroids;

  const meanX = points.reduce((sum, point) => sum + point.nx, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.ny, 0) / points.length;
  const meanT = points.reduce((sum, point) => sum + point.nt, 0) / points.length;

  let firstIdx = 0;
  let firstDist = -1;
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const distance = euclideanDistance3(point.nx, point.ny, point.nt, meanX, meanY, meanT);
    if (distance > firstDist) {
      firstDist = distance;
      firstIdx = i;
    }
  }
  centroids.push({ x: points[firstIdx].nx, y: points[firstIdx].ny, t: points[firstIdx].nt });

  const selected = new Set<number>([firstIdx]);
  while (centroids.length < k) {
    let nextIdx = -1;
    let maxNearest = -1;
    for (let i = 0; i < points.length; i += 1) {
      if (selected.has(i)) continue;
      const point = points[i];
      let nearest = Number.POSITIVE_INFINITY;
      for (const centroid of centroids) {
        const distance = euclideanDistance3(point.nx, point.ny, point.nt, centroid.x, centroid.y, centroid.t);
        if (distance < nearest) nearest = distance;
      }
      if (nearest > maxNearest) {
        maxNearest = nearest;
        nextIdx = i;
      }
    }
    if (nextIdx < 0) break;
    selected.add(nextIdx);
    centroids.push({ x: points[nextIdx].nx, y: points[nextIdx].ny, t: points[nextIdx].nt });
  }

  while (centroids.length < k) {
    const fallbackIdx = centroids.length % points.length;
    centroids.push({ x: points[fallbackIdx].nx, y: points[fallbackIdx].ny, t: points[fallbackIdx].nt });
  }

  return centroids;
}

function runKMeans(points: NormalizedShotPoint[], k: number, maxIterations = 30): KMeansModel {
  const safeK = Math.max(1, Math.min(k, Math.max(1, points.length)));
  const assignments = new Array<number>(points.length).fill(0);
  const centroids = initializeKMeansCentroids(points, safeK);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let changed = false;
    for (let i = 0; i < points.length; i += 1) {
      const point = points[i];
      let bestCluster = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let cluster = 0; cluster < centroids.length; cluster += 1) {
        const centroid = centroids[cluster];
        const distance = euclideanDistance3(point.nx, point.ny, point.nt, centroid.x, centroid.y, centroid.t);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = cluster;
        }
      }
      if (assignments[i] !== bestCluster) {
        assignments[i] = bestCluster;
        changed = true;
      }
    }

    const sums = centroids.map(() => ({ x: 0, y: 0, t: 0, count: 0 }));
    for (let i = 0; i < points.length; i += 1) {
      const cluster = assignments[i];
      sums[cluster].x += points[i].nx;
      sums[cluster].y += points[i].ny;
      sums[cluster].t += points[i].nt;
      sums[cluster].count += 1;
    }

    for (let cluster = 0; cluster < centroids.length; cluster += 1) {
      if (sums[cluster].count === 0) {
        let farthestIdx = 0;
        let farthestDistance = -1;
        for (let i = 0; i < points.length; i += 1) {
          const assignedCentroid = centroids[assignments[i]];
          const distance = euclideanDistance3(
            points[i].nx,
            points[i].ny,
            points[i].nt,
            assignedCentroid.x,
            assignedCentroid.y,
            assignedCentroid.t,
          );
          if (distance > farthestDistance) {
            farthestDistance = distance;
            farthestIdx = i;
          }
        }
        centroids[cluster] = { x: points[farthestIdx].nx, y: points[farthestIdx].ny, t: points[farthestIdx].nt };
        changed = true;
        continue;
      }

      centroids[cluster] = {
        x: sums[cluster].x / sums[cluster].count,
        y: sums[cluster].y / sums[cluster].count,
        t: sums[cluster].t / sums[cluster].count,
      };
    }

    if (!changed) break;
  }

  let sse = 0;
  for (let i = 0; i < points.length; i += 1) {
    const centroid = centroids[assignments[i]];
    const distance = euclideanDistance3(points[i].nx, points[i].ny, points[i].nt, centroid.x, centroid.y, centroid.t);
    sse += distance * distance;
  }

  return { k: safeK, assignments, centroids, sse };
}

function silhouetteScore(points: NormalizedShotPoint[], assignments: number[], k: number): number {
  if (points.length <= 1 || k <= 1) return 0;

  const clusters: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < assignments.length; i += 1) {
    clusters[assignments[i]].push(i);
  }

  let silhouetteSum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const ownCluster = assignments[i];
    const ownMembers = clusters[ownCluster];
    let a = 0;
    if (ownMembers.length > 1) {
      for (const memberIdx of ownMembers) {
        if (memberIdx === i) continue;
        a += euclideanDistance3(
          points[i].nx,
          points[i].ny,
          points[i].nt,
          points[memberIdx].nx,
          points[memberIdx].ny,
          points[memberIdx].nt,
        );
      }
      a /= ownMembers.length - 1;
    }

    let b = Number.POSITIVE_INFINITY;
    for (let cluster = 0; cluster < k; cluster += 1) {
      if (cluster === ownCluster || clusters[cluster].length === 0) continue;
      let averageDistance = 0;
      for (const memberIdx of clusters[cluster]) {
        averageDistance += euclideanDistance3(
          points[i].nx,
          points[i].ny,
          points[i].nt,
          points[memberIdx].nx,
          points[memberIdx].ny,
          points[memberIdx].nt,
        );
      }
      averageDistance /= clusters[cluster].length;
      if (averageDistance < b) b = averageDistance;
    }

    const denominator = Math.max(a, b, 1e-6);
    const s = Number.isFinite(b) ? (b - a) / denominator : 0;
    silhouetteSum += s;
  }

  return silhouetteSum / points.length;
}

function centroidClosenessPenalty(centroids: Array<{ x: number; y: number; t: number }>, tweaks: TweakSettings): number {
  let penalty = 0;
  const closeDistance = tweaks.kmeansCentroidCloseDistance;
  for (let i = 0; i < centroids.length; i += 1) {
    for (let j = i + 1; j < centroids.length; j += 1) {
      const distance = euclideanDistance3(
        centroids[i].x,
        centroids[i].y,
        centroids[i].t,
        centroids[j].x,
        centroids[j].y,
        centroids[j].t,
      );
      if (distance < closeDistance) {
        penalty += (closeDistance - distance) ** 2;
      }
    }
  }
  return penalty;
}

function clusterShotsBySpaceTime(shots: ShotLogEntry[], tweaks: TweakSettings): ShotClusteringResult {
  if (shots.length === 0) {
    return {
      selectedK: 0,
      finalK: 0,
      closeMergeCount: 0,
      objectiveScore: 0,
      shotClusterById: {},
      clusters: [],
    };
  }

  const sortedShots = [...shots].sort((a, b) => a.videoTimeSec - b.videoTimeSec);
  const xStats = meanAndStd(sortedShots.map((shot) => shot.centerX));
  const yStats = meanAndStd(sortedShots.map((shot) => shot.centerY));
  const tStats = meanAndStd(sortedShots.map((shot) => shot.videoTimeSec));
  const points: NormalizedShotPoint[] = sortedShots.map((shot) => ({
    shot,
    nx: (shot.centerX - xStats.mean) / xStats.std,
    ny: (shot.centerY - yStats.mean) / yStats.std,
    nt: ((shot.videoTimeSec - tStats.mean) / tStats.std) * tweaks.kmeansTimeWeight,
  }));

  const maxK = Math.max(
    1,
    Math.min(points.length, Math.min(Math.max(1, Math.round(tweaks.kmeansMaxClustersCap)), Math.floor(Math.sqrt(points.length)) + 1)),
  );
  let bestModel = runKMeans(points, 1, Math.max(1, Math.round(tweaks.kmeansMaxIterations)));
  let bestScore = -Infinity;
  for (let k = 1; k <= maxK; k += 1) {
    const model = runKMeans(points, k, Math.max(1, Math.round(tweaks.kmeansMaxIterations)));
    const silhouette = silhouetteScore(points, model.assignments, model.k);
    const closePenalty = centroidClosenessPenalty(model.centroids, tweaks);
    const normalizedSse = model.sse / Math.max(1, points.length);
    const objective =
      silhouette -
      tweaks.kmeansComplexityPenalty * (k - 1) -
      tweaks.kmeansClosePenaltyWeight * closePenalty -
      tweaks.kmeansSsePenaltyWeight * normalizedSse;
    if (objective > bestScore) {
      bestScore = objective;
      bestModel = model;
    }
  }

  const parents = Array.from({ length: bestModel.k }, (_, index) => index);
  const findRoot = (index: number): number => {
    let root = index;
    while (parents[root] !== root) {
      root = parents[root];
    }
    let current = index;
    while (parents[current] !== current) {
      const next = parents[current];
      parents[current] = root;
      current = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const rootA = findRoot(a);
    const rootB = findRoot(b);
    if (rootA !== rootB) parents[rootB] = rootA;
  };

  for (let i = 0; i < bestModel.centroids.length; i += 1) {
    for (let j = i + 1; j < bestModel.centroids.length; j += 1) {
      const spatialDistance = Math.hypot(
        bestModel.centroids[i].x - bestModel.centroids[j].x,
        bestModel.centroids[i].y - bestModel.centroids[j].y,
      );
      const timeDistance = Math.abs(bestModel.centroids[i].t - bestModel.centroids[j].t);
      const combinedDistance = euclideanDistance3(
        bestModel.centroids[i].x,
        bestModel.centroids[i].y,
        bestModel.centroids[i].t,
        bestModel.centroids[j].x,
        bestModel.centroids[j].y,
        bestModel.centroids[j].t,
      );
      if (
        combinedDistance < tweaks.kmeansMergeCombinedDistanceMax ||
        (spatialDistance < tweaks.kmeansMergeSpatialDistanceMax && timeDistance < tweaks.kmeansMergeTimeDistanceMax)
      ) {
        union(i, j);
      }
    }
  }

  const rootToClusterId = new Map<number, number>();
  const mergedAssignments: number[] = [];
  let nextClusterId = 1;
  for (let i = 0; i < bestModel.assignments.length; i += 1) {
    const root = findRoot(bestModel.assignments[i]);
    if (!rootToClusterId.has(root)) {
      rootToClusterId.set(root, nextClusterId);
      nextClusterId += 1;
    }
    const mappedClusterId = rootToClusterId.get(root) ?? 1;
    mergedAssignments[i] = mappedClusterId;
  }

  const shotsByCluster = new Map<number, ShotLogEntry[]>();
  for (let i = 0; i < points.length; i += 1) {
    const clusterId = mergedAssignments[i];
    const bucket = shotsByCluster.get(clusterId) ?? [];
    bucket.push(points[i].shot);
    shotsByCluster.set(clusterId, bucket);
  }

  const shotClusterById: Record<string, number> = {};
  const clusters: ShotClusterSummary[] = [];
  const orderedClusters = [...shotsByCluster.entries()].sort((a, b) => a[0] - b[0]);
  for (const [clusterId, clusterShots] of orderedClusters) {
    const sortedClusterShots = [...clusterShots].sort((a, b) => a.videoTimeSec - b.videoTimeSec);
    const count = sortedClusterShots.length;
    for (const shot of sortedClusterShots) {
      shotClusterById[shot.id] = clusterId;
    }

    const centroidX = sortedClusterShots.reduce((sum, shot) => sum + shot.centerX, 0) / count;
    const centroidY = sortedClusterShots.reduce((sum, shot) => sum + shot.centerY, 0) / count;
    const centroidTimeSec = sortedClusterShots.reduce((sum, shot) => sum + shot.videoTimeSec, 0) / count;
    let extremeSpreadPx = 0;
    for (let i = 0; i < sortedClusterShots.length; i += 1) {
      for (let j = i + 1; j < sortedClusterShots.length; j += 1) {
        const spread = Math.hypot(
          sortedClusterShots[i].centerX - sortedClusterShots[j].centerX,
          sortedClusterShots[i].centerY - sortedClusterShots[j].centerY,
        );
        if (spread > extremeSpreadPx) extremeSpreadPx = spread;
      }
    }

    const timeSpanSec =
      sortedClusterShots.length > 1
        ? sortedClusterShots[sortedClusterShots.length - 1].videoTimeSec - sortedClusterShots[0].videoTimeSec
        : 0;
    let meanTimeBetweenShotsSec: number | null = null;
    if (sortedClusterShots.length > 1) {
      let sumDelta = 0;
      for (let i = 1; i < sortedClusterShots.length; i += 1) {
        sumDelta += sortedClusterShots[i].videoTimeSec - sortedClusterShots[i - 1].videoTimeSec;
      }
      meanTimeBetweenShotsSec = sumDelta / (sortedClusterShots.length - 1);
    }

    const diameterInches = sortedClusterShots
      .map((shot) => shot.estimatedDiameterInches)
      .filter((value): value is number => value !== null);
    const meanDiameterInches =
      diameterInches.length === 0 ? null : diameterInches.reduce((sum, value) => sum + value, 0) / diameterInches.length;

    clusters.push({
      clusterId,
      shots: sortedClusterShots,
      count,
      centroidX,
      centroidY,
      centroidTimeSec,
      extremeSpreadPx,
      timeSpanSec,
      meanTimeBetweenShotsSec,
      meanDiameterInches,
    });
  }

  return {
    selectedK: bestModel.k,
    finalK: rootToClusterId.size,
    closeMergeCount: Math.max(0, bestModel.k - rootToClusterId.size),
    objectiveScore: bestScore,
    shotClusterById,
    clusters,
  };
}

const CLUSTER_COLOR_PALETTE = [
  "#22d3ee",
  "#f97316",
  "#34d399",
  "#f43f5e",
  "#a78bfa",
  "#facc15",
  "#60a5fa",
  "#fb7185",
  "#4ade80",
  "#e879f9",
];

function clusterColorForId(clusterId: number): string {
  if (!Number.isFinite(clusterId) || clusterId <= 0) return "#f87171";
  return CLUSTER_COLOR_PALETTE[(clusterId - 1) % CLUSTER_COLOR_PALETTE.length];
}

function hexToRgba(hex: string, alpha: number): string {
  const safeAlpha = Math.max(0, Math.min(1, alpha));
  const sanitized = hex.replace("#", "");
  if (sanitized.length !== 6) return `rgba(248, 113, 113, ${safeAlpha})`;
  const r = Number.parseInt(sanitized.slice(0, 2), 16);
  const g = Number.parseInt(sanitized.slice(2, 4), 16);
  const b = Number.parseInt(sanitized.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${safeAlpha})`;
}

function convexHull(points: Array<{ x: number; y: number }>): Array<{ x: number; y: number }> {
  if (points.length <= 1) return [...points];
  const unique = new Map<string, { x: number; y: number }>();
  for (const point of points) {
    unique.set(`${point.x.toFixed(4)}:${point.y.toFixed(4)}`, point);
  }
  const sorted = [...unique.values()].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  if (sorted.length <= 2) return sorted;

  const cross = (o: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Array<{ x: number; y: number }> = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper: Array<{ x: number; y: number }> = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const point = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function clusterGeometryFromShots(shots: ShotLogEntry[], shotClusterById: Record<string, number>): ClusterGeometry[] {
  const grouped = new Map<number, ShotLogEntry[]>();
  for (const shot of shots) {
    const clusterId = shotClusterById[shot.id];
    if (clusterId === undefined) continue;
    const bucket = grouped.get(clusterId) ?? [];
    bucket.push(shot);
    grouped.set(clusterId, bucket);
  }

  const geometry: ClusterGeometry[] = [];
  for (const [clusterId, clusterShots] of grouped.entries()) {
    const points = clusterShots.map((shot) => ({ x: shot.centerX, y: shot.centerY }));
    const centroidX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
    const centroidY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    geometry.push({ clusterId, points, centroidX, centroidY });
  }

  geometry.sort((a, b) => a.clusterId - b.clusterId);
  return geometry;
}

function drawClusterGeometry(
  context: CanvasRenderingContext2D,
  clusterGeometry: ClusterGeometry[],
  clusterColorById: Record<number, string>,
  scaleX: number,
  scaleY: number,
) {
  for (const cluster of clusterGeometry) {
    if (cluster.points.length === 0) continue;
    const clusterColor = clusterColorById[cluster.clusterId] ?? clusterColorForId(cluster.clusterId);
    const hull = convexHull(cluster.points);
    if (hull.length >= 3) {
      context.beginPath();
      context.moveTo(hull[0].x * scaleX, hull[0].y * scaleY);
      for (let i = 1; i < hull.length; i += 1) {
        context.lineTo(hull[i].x * scaleX, hull[i].y * scaleY);
      }
      context.closePath();
      context.fillStyle = hexToRgba(clusterColor, 0.11);
      context.fill();
      context.strokeStyle = hexToRgba(clusterColor, 0.9);
      context.lineWidth = 2;
      context.stroke();
    } else if (hull.length === 2) {
      context.beginPath();
      context.moveTo(hull[0].x * scaleX, hull[0].y * scaleY);
      context.lineTo(hull[1].x * scaleX, hull[1].y * scaleY);
      context.strokeStyle = hexToRgba(clusterColor, 0.9);
      context.lineWidth = 2;
      context.stroke();
    } else {
      context.beginPath();
      context.arc(hull[0].x * scaleX, hull[0].y * scaleY, 7, 0, Math.PI * 2);
      context.strokeStyle = hexToRgba(clusterColor, 0.9);
      context.lineWidth = 2;
      context.stroke();
    }

    const labelX = cluster.centroidX * scaleX;
    const labelY = cluster.centroidY * scaleY;
    context.fillStyle = "rgba(0, 0, 0, 0.72)";
    context.fillRect(labelX - 14, labelY - 20, 28, 14);
    context.fillStyle = clusterColor;
    context.font = "10px sans-serif";
    context.fillText(`C${cluster.clusterId}`, labelX - 10, labelY - 10);
  }
}

function windowIndexAtTime(timeSec: number, windows: TimeWindow[]): number {
  for (let i = 0; i < windows.length; i += 1) {
    if (timeSec >= windows[i].start && timeSec <= windows[i].end) return i;
  }
  return -1;
}

type ProcessedContourOverlayMasks = {
  rawMask: Uint8Array | null;
  temporalMask: Uint8Array | null;
  effectiveMask: Uint8Array | null;
  positiveDiffMap: Float32Array | null;
};

function clampColorChannel(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function openCvActionStatusColor(state: OpenCvActionState): string {
  if (state === "ok") return "#86efac";
  if (state === "warn") return "#fcd34d";
  if (state === "skip") return "#93c5fd";
  return "#cbd5e1";
}

function drawOpenCvActionStatuses(
  context: CanvasRenderingContext2D,
  statuses: OpenCvActionStatus[],
  maxWidth: number,
  startY: number,
): void {
  if (statuses.length === 0) return;
  const safeWidth = Math.max(0, maxWidth);
  if (safeWidth < 40) return;
  const panelX = 4;
  const rowHeight = 10;
  const panelHeight = statuses.length * rowHeight + 6;
  const panelWidth = Math.min(Math.max(110, safeWidth - 8), 380);
  context.fillStyle = "rgba(2, 6, 23, 0.72)";
  context.fillRect(panelX, startY, panelWidth, panelHeight);

  context.save();
  context.beginPath();
  context.rect(panelX, startY, panelWidth, panelHeight);
  context.clip();
  context.font = "10px sans-serif";
  for (let i = 0; i < statuses.length; i += 1) {
    const status = statuses[i];
    context.fillStyle = openCvActionStatusColor(status.state);
    context.fillText(`${status.label}: ${status.detail}`, panelX + 4, startY + 10 + i * rowHeight);
  }
  context.restore();
}

function drawContourMethodLegend(
  context: CanvasRenderingContext2D,
  patchWidth: number,
  patchHeight: number,
): void {
  if (patchWidth < 56 || patchHeight < 14) return;
  const legendY = Math.max(10, patchHeight - 4);
  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, Math.max(0, patchHeight - 12), Math.min(patchWidth, 340), 12);
  context.font = "9px sans-serif";
  context.fillStyle = "#7dd3fc";
  context.fillText("raw", 4, legendY);
  context.fillStyle = "#fbbf24";
  context.fillText("temporal", 30, legendY);
  context.fillStyle = "#86efac";
  context.fillText("effective", 74, legendY);
  context.fillStyle = "#f9a8d4";
  context.fillText("diff", 121, legendY);
}

function paintMaskRuns(
  imageData: ImageData,
  pixelCount: number,
  runs: BinaryMaskRun[],
  r: number,
  g: number,
  b: number,
): void {
  for (const run of runs) {
    const safeStart = Math.max(0, Math.min(pixelCount - 1, Math.floor(run.start)));
    const safeLength = Math.max(0, Math.floor(run.length));
    const endExclusive = Math.min(pixelCount, safeStart + safeLength);
    for (let i = safeStart; i < endExclusive; i += 1) {
      const rgbaIndex = i * 4;
      imageData.data[rgbaIndex] = clampColorChannel(imageData.data[rgbaIndex] + r);
      imageData.data[rgbaIndex + 1] = clampColorChannel(imageData.data[rgbaIndex + 1] + g);
      imageData.data[rgbaIndex + 2] = clampColorChannel(imageData.data[rgbaIndex + 2] + b);
      imageData.data[rgbaIndex + 3] = 255;
    }
  }
}

function drawProcessedContourView(
  context: CanvasRenderingContext2D,
  patchRgba: Uint8ClampedArray,
  baselineRgba: Uint8ClampedArray | null,
  patchWidth: number,
  patchHeight: number,
  overlayMasks: ProcessedContourOverlayMasks,
  regions: ChangedContourRegion[],
  actionStatuses: OpenCvActionStatus[],
): void {
  const pixelCount = patchWidth * patchHeight;
  const rawMask = overlayMasks.rawMask && overlayMasks.rawMask.length === pixelCount ? overlayMasks.rawMask : null;
  const temporalMask =
    overlayMasks.temporalMask && overlayMasks.temporalMask.length === pixelCount ? overlayMasks.temporalMask : null;
  const effectiveMask =
    overlayMasks.effectiveMask && overlayMasks.effectiveMask.length === pixelCount ? overlayMasks.effectiveMask : null;
  const positiveDiffMap =
    overlayMasks.positiveDiffMap && overlayMasks.positiveDiffMap.length === pixelCount ? overlayMasks.positiveDiffMap : null;
  const hasBaseline = !!baselineRgba && baselineRgba.length === patchRgba.length;
  let changedCount = 0;
  for (let i = 0; i < pixelCount; i += 1) {
    if (effectiveMask && effectiveMask[i] === 1) changedCount += 1;
  }

  const imageData = context.createImageData(patchWidth, patchHeight);
  for (let i = 0; i < pixelCount; i += 1) {
    const rgbaIndex = i * 4;
    let r = patchRgba[rgbaIndex] * (hasBaseline ? 0.24 : 0.35);
    let g = patchRgba[rgbaIndex + 1] * (hasBaseline ? 0.24 : 0.35);
    let b = patchRgba[rgbaIndex + 2] * (hasBaseline ? 0.24 : 0.35);

    if (hasBaseline && baselineRgba) {
      const deltaR = Math.abs(patchRgba[rgbaIndex] - baselineRgba[rgbaIndex]);
      const deltaG = Math.abs(patchRgba[rgbaIndex + 1] - baselineRgba[rgbaIndex + 1]);
      const deltaB = Math.abs(patchRgba[rgbaIndex + 2] - baselineRgba[rgbaIndex + 2]);
      r += deltaR * 0.16;
      g += deltaG * 0.16;
      b += deltaB * 0.16;
    }

    if (positiveDiffMap) {
      const diffStrength = Math.max(
        0,
        Math.min(1, (positiveDiffMap[i] ?? 0) / Math.max(1, SIMPLE_CONTOUR_DIFF_THRESHOLD * 1.8)),
      );
      r += 110 * diffStrength;
      b += 130 * diffStrength;
    }
    if (rawMask && rawMask[i] === 1) {
      g += 88;
      b += 160;
    }
    if (temporalMask && temporalMask[i] === 1) {
      r += 160;
      g += 110;
    }
    if (effectiveMask && effectiveMask[i] === 1) {
      r += 25;
      g += 170;
      b += 44;
    }

    imageData.data[rgbaIndex] = clampColorChannel(r);
    imageData.data[rgbaIndex + 1] = clampColorChannel(g);
    imageData.data[rgbaIndex + 2] = clampColorChannel(b);
    imageData.data[rgbaIndex + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);

  const maxRegionsToDraw = 10;
  const visibleRegions = regions.slice(0, maxRegionsToDraw);
  for (let i = 0; i < visibleRegions.length; i += 1) {
    const region = visibleRegions[i];
    const rectWidth = Math.max(1, region.maxX - region.minX + 1);
    const rectHeight = Math.max(1, region.maxY - region.minY + 1);
    context.strokeStyle = "rgba(248, 113, 113, 0.95)";
    context.lineWidth = 1.5;
    context.strokeRect(region.minX, region.minY, rectWidth, rectHeight);
    drawCenterCross(
      context,
      region.centerX,
      region.centerY,
      3,
      "rgba(255, 255, 255, 0.95)",
      1.25,
    );
    context.fillStyle = "rgba(248, 113, 113, 0.95)";
    context.font = "10px sans-serif";
    context.fillText(
      `${i + 1}`,
      Math.max(0, region.minX),
      Math.max(10, region.minY - 2),
    );
  }

  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(360, patchWidth), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  const modeLabel = hasBaseline ? "Contour hybrid overlay" : "Contour overlay (baseline pending)";
  context.fillText(`${modeLabel} | changed=${changedCount} | regions=${regions.length}`, 4, 11);
  drawOpenCvActionStatuses(context, actionStatuses, patchWidth, 18);
  drawContourMethodLegend(context, patchWidth, patchHeight);
}

function encodeBinaryMaskRuns(mask: Uint8Array): BinaryMaskRun[] {
  const runs: BinaryMaskRun[] = [];
  let runStart = -1;
  for (let i = 0; i < mask.length; i += 1) {
    const isChanged = mask[i] === 1;
    if (isChanged && runStart < 0) {
      runStart = i;
      continue;
    }
    if (!isChanged && runStart >= 0) {
      runs.push({ start: runStart, length: i - runStart });
      runStart = -1;
    }
  }
  if (runStart >= 0) {
    runs.push({ start: runStart, length: mask.length - runStart });
  }
  return runs;
}

function drawPersistedContourWindowView(
  context: CanvasRenderingContext2D,
  snapshot: ContourWindowFrameSnapshot,
): void {
  const width = Math.max(1, Math.round(snapshot.patchWidthPx));
  const height = Math.max(1, Math.round(snapshot.patchHeightPx));
  if (context.canvas.width !== width || context.canvas.height !== height) {
    context.canvas.width = width;
    context.canvas.height = height;
  }

  const pixelCount = width * height;
  const imageData = context.createImageData(width, height);
  for (let i = 0; i < pixelCount; i += 1) {
    const rgbaIndex = i * 4;
    imageData.data[rgbaIndex] = 8;
    imageData.data[rgbaIndex + 1] = 12;
    imageData.data[rgbaIndex + 2] = 20;
    imageData.data[rgbaIndex + 3] = 255;
  }
  paintMaskRuns(imageData, pixelCount, snapshot.rawMaskRuns, 0, 88, 160);
  paintMaskRuns(imageData, pixelCount, snapshot.temporalMaskRuns, 160, 110, 0);
  paintMaskRuns(imageData, pixelCount, snapshot.effectiveMaskRuns, 25, 170, 44);
  context.putImageData(imageData, 0, 0);

  const maxRegionsToDraw = 10;
  const visibleRegions = snapshot.regions.slice(0, maxRegionsToDraw);
  for (let i = 0; i < visibleRegions.length; i += 1) {
    const region = visibleRegions[i];
    const rectWidth = Math.max(1, region.maxX - region.minX + 1);
    const rectHeight = Math.max(1, region.maxY - region.minY + 1);
    context.strokeStyle = "rgba(248, 113, 113, 0.95)";
    context.lineWidth = 1.5;
    context.strokeRect(region.minX, region.minY, rectWidth, rectHeight);
    drawCenterCross(
      context,
      region.centerX,
      region.centerY,
      3,
      "rgba(255, 255, 255, 0.95)",
      1.25,
    );
    context.fillStyle = "rgba(248, 113, 113, 0.95)";
    context.font = "10px sans-serif";
    context.fillText(
      `${i + 1}`,
      Math.max(0, region.minX),
      Math.max(10, region.minY - 2),
    );
  }

  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(360, width), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  context.fillText(
    `Contour playback hybrid | changed=${snapshot.changedPixels} | regions=${snapshot.regions.length}`,
    4,
    11,
  );
  drawOpenCvActionStatuses(context, snapshot.actionStatuses, width, 18);
  drawContourMethodLegend(context, width, height);
}

function sanitizeBinaryMaskRuns(value: unknown): BinaryMaskRun[] {
  if (!Array.isArray(value)) return [];
  const runs: BinaryMaskRun[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const candidate = entry as { start?: unknown; length?: unknown };
    const start = Number(candidate.start);
    const length = Number(candidate.length);
    if (!Number.isFinite(start) || !Number.isFinite(length) || length <= 0) continue;
    runs.push({ start: Math.max(0, Math.floor(start)), length: Math.max(1, Math.floor(length)) });
  }
  return runs;
}

function sanitizeContourWindowRegions(value: unknown): ContourWindowRegionSnapshot[] {
  if (!Array.isArray(value)) return [];
  const regions: ContourWindowRegionSnapshot[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const candidate = entry as Partial<ContourWindowRegionSnapshot>;
    const pixelCount = Number(candidate.pixelCount);
    const centerX = Number(candidate.centerX);
    const centerY = Number(candidate.centerY);
    const minX = Number(candidate.minX);
    const minY = Number(candidate.minY);
    const maxX = Number(candidate.maxX);
    const maxY = Number(candidate.maxY);
    if (
      !Number.isFinite(pixelCount) ||
      !Number.isFinite(centerX) ||
      !Number.isFinite(centerY) ||
      !Number.isFinite(minX) ||
      !Number.isFinite(minY) ||
      !Number.isFinite(maxX) ||
      !Number.isFinite(maxY)
    ) {
      continue;
    }
    regions.push({
      pixelCount: Math.max(0, Math.floor(pixelCount)),
      centerX,
      centerY,
      minX,
      minY,
      maxX,
      maxY,
    });
  }
  return regions;
}

function sanitizeOpenCvActionStatuses(value: unknown): OpenCvActionStatus[] {
  if (!Array.isArray(value)) return [];
  const statuses: OpenCvActionStatus[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const candidate = entry as Partial<OpenCvActionStatus>;
    const label = typeof candidate.label === "string" ? candidate.label : "";
    const detail = typeof candidate.detail === "string" ? candidate.detail : "";
    const state = candidate.state;
    if (!label || !detail) continue;
    if (state !== "ok" && state !== "warn" && state !== "skip" && state !== "idle") continue;
    statuses.push({
      label: label.slice(0, 80),
      detail: detail.slice(0, 180),
      state,
    });
  }
  return statuses;
}

function sanitizeContourWindowSnapshot(value: unknown): ContourWindowFrameSnapshot | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<ContourWindowFrameSnapshot>;
  const frame = Number(candidate.frame);
  const videoTimeSec = Number(candidate.videoTimeSec);
  const patchWidthPx = Number(candidate.patchWidthPx);
  const patchHeightPx = Number(candidate.patchHeightPx);
  const changedPixels = Number(candidate.changedPixels);
  if (
    !Number.isFinite(frame) ||
    !Number.isFinite(videoTimeSec) ||
    !Number.isFinite(patchWidthPx) ||
    !Number.isFinite(patchHeightPx) ||
    !Number.isFinite(changedPixels)
  ) {
    return null;
  }

  const rawMaskRuns = sanitizeBinaryMaskRuns(candidate.rawMaskRuns);
  const temporalMaskRuns = sanitizeBinaryMaskRuns(candidate.temporalMaskRuns);
  const effectiveMaskRuns = sanitizeBinaryMaskRuns(candidate.effectiveMaskRuns);
  const regions = sanitizeContourWindowRegions(candidate.regions);
  const actionStatuses = sanitizeOpenCvActionStatuses(candidate.actionStatuses);

  return {
    frame: Math.max(0, Math.floor(frame)),
    videoTimeSec: Math.max(0, videoTimeSec),
    patchWidthPx: Math.max(1, Math.floor(patchWidthPx)),
    patchHeightPx: Math.max(1, Math.floor(patchHeightPx)),
    changedPixels: Math.max(0, Math.floor(changedPixels)),
    rawMaskRuns,
    temporalMaskRuns,
    effectiveMaskRuns,
    regions,
    actionStatuses,
  };
}

function drawCenterCross(
  context: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  halfSize: number,
  color: string,
  lineWidth: number,
) {
  context.beginPath();
  context.moveTo(centerX - halfSize, centerY);
  context.lineTo(centerX + halfSize, centerY);
  context.moveTo(centerX, centerY - halfSize);
  context.lineTo(centerX, centerY + halfSize);
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.stroke();
}

declare global {
  interface Window {
    cv?: CvApi;
    Howl?: HowlConstructor;
  }
}

function loadImageFromUrl(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load reference image for OpenCV processing."));
    image.src = src;
  });
}

function revokeBlobUrl(url: string | null) {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function clearTemplateRegionCache() {
  if (typeof window === "undefined") return;
  sessionStorage.removeItem(TEMPLATE_REGION_DATA_URL_KEY);
  sessionStorage.removeItem(TEMPLATE_REGION_IMAGE_NAME_KEY);
  sessionStorage.removeItem(TEMPLATE_REGION_RECT_KEY);
}

async function createTemplateRegionDataUrl(imageSrc: string, roi: RoiRect): Promise<string> {
  const sourceImage = await loadImageFromUrl(imageSrc);
  if (sourceImage.naturalWidth === 0 || sourceImage.naturalHeight === 0) {
    throw new Error("Unable to load source image for template selection.");
  }

  const sourceCanvas = imageToCanvas(sourceImage);
  const sx = Math.max(0, Math.floor(roi.x * sourceCanvas.width));
  const sy = Math.max(0, Math.floor(roi.y * sourceCanvas.height));
  const sw = Math.max(1, Math.floor(roi.width * sourceCanvas.width));
  const sh = Math.max(1, Math.floor(roi.height * sourceCanvas.height));

  const regionCanvas = document.createElement("canvas");
  regionCanvas.width = sw;
  regionCanvas.height = sh;
  const regionCtx = regionCanvas.getContext("2d");
  if (!regionCtx) throw new Error("Failed to capture selected template region.");
  regionCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, sw, sh);

  return regionCanvas.toDataURL("image/png");
}

async function getSelectionPixelSize(imageSrc: string, roi: RoiRect): Promise<{ widthPx: number; heightPx: number }> {
  const sourceImage = await loadImageFromUrl(imageSrc);
  if (sourceImage.naturalWidth === 0 || sourceImage.naturalHeight === 0) {
    throw new Error("Unable to load source image for pixel calibration.");
  }
  const sourceCanvas = imageToCanvas(sourceImage);
  const widthPx = Math.max(1, Math.floor(roi.width * sourceCanvas.width));
  const heightPx = Math.max(1, Math.floor(roi.height * sourceCanvas.height));
  return { widthPx, heightPx };
}

function imageToCanvas(image: HTMLImageElement): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;

  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to prepare reference image canvas.");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  return canvas;
}

function toCsv(logEntries: DetectionLogEntry[]): string {
  const header = "frame,video_time_sec,x,y,width,height,score_pct,estimated_distance_in";
  const rows = logEntries.map((entry) =>
    [
      entry.frame,
      entry.videoTimeSec.toFixed(3),
      entry.x,
      entry.y,
      entry.width,
      entry.height,
      entry.score.toFixed(2),
      entry.estimatedDistanceInches === null ? "" : entry.estimatedDistanceInches.toFixed(2),
    ].join(","),
  );
  return [header, ...rows].join("\n");
}

function toShotCsv(shotEntries: ShotLogEntry[]): string {
  const header =
    "id,shot_number,frame,video_time_sec,time_since_prev_shot_sec,window_start_sec,window_end_sec,center_x,center_y,radius_px,changed_pixels,change_score,estimated_diameter_in,detection_method,tracking_mode,detection_enabled,detection_confidence_pct,detection_threshold_pct,spike_focused_window,nearest_spike_id,nearest_spike_time_sec,nearest_spike_delta_sec,nearest_spike_strength,audio_dbfs,audio_peak_dbfs,audio_delta_from_mean_db,audio_correlation_score_pct,nearest_spike_dbfs,patch_width_px,patch_height_px,draw_rect_x,draw_rect_y,draw_rect_width,draw_rect_height,center_patch_x,center_patch_y,span_width_px,span_height_px,estimated_diameter_px,changed_pixel_ratio_pct,bright_pixel_count,transition_pixel_count,transition_purity_pct,estimated_rapid_change_ms,fast_color_change_trigger,min_blob_pixels_threshold,temporal_history_frames_used,shot_cooldown_ms,residual_motion_px,tracked_point_count";
  const rows = shotEntries.map((entry) =>
    [
      entry.id,
      entry.shotNumber,
      entry.frame,
      entry.videoTimeSec.toFixed(3),
      entry.timeSincePreviousShotSec === null ? "" : entry.timeSincePreviousShotSec.toFixed(3),
      entry.windowStartSec.toFixed(3),
      entry.windowEndSec.toFixed(3),
      entry.centerX,
      entry.centerY,
      entry.radius,
      entry.changedPixels,
      entry.changeScore.toFixed(3),
      entry.estimatedDiameterInches === null ? "" : entry.estimatedDiameterInches.toFixed(3),
      entry.detectionMethod,
      entry.trackingMode,
      entry.detectionEnabled ? "1" : "0",
      entry.detectionConfidencePct.toFixed(2),
      entry.detectionThresholdPct.toFixed(2),
      entry.spikeFocusedWindow ? "1" : "0",
      entry.nearestSpikeId ?? "",
      entry.nearestSpikeTimeSec === null ? "" : entry.nearestSpikeTimeSec.toFixed(3),
      entry.nearestSpikeDeltaSec === null ? "" : entry.nearestSpikeDeltaSec.toFixed(3),
      entry.nearestSpikeStrength === null ? "" : entry.nearestSpikeStrength.toFixed(4),
      entry.audioDecibelDbfs === null ? "" : entry.audioDecibelDbfs.toFixed(3),
      entry.audioPeakDbfs === null ? "" : entry.audioPeakDbfs.toFixed(3),
      entry.audioDeltaFromMeanDb === null ? "" : entry.audioDeltaFromMeanDb.toFixed(3),
      entry.audioCorrelationScorePct === null ? "" : entry.audioCorrelationScorePct.toFixed(2),
      entry.nearestSpikeDecibelDbfs === null ? "" : entry.nearestSpikeDecibelDbfs.toFixed(3),
      entry.patchWidthPx,
      entry.patchHeightPx,
      entry.drawRectX,
      entry.drawRectY,
      entry.drawRectWidth,
      entry.drawRectHeight,
      entry.centerPatchX.toFixed(3),
      entry.centerPatchY.toFixed(3),
      entry.spanWidthPx,
      entry.spanHeightPx,
      entry.estimatedDiameterPx.toFixed(3),
      entry.changedPixelRatioPct.toFixed(3),
      entry.brightPixelCount === null ? "" : entry.brightPixelCount,
      entry.transitionPixelCount === null ? "" : entry.transitionPixelCount,
      entry.transitionPurityPct === null ? "" : entry.transitionPurityPct.toFixed(3),
      entry.estimatedRapidChangeMs === null ? "" : entry.estimatedRapidChangeMs.toFixed(4),
      entry.fastColorChangeTrigger ? "1" : "0",
      entry.minBlobPixelsThreshold,
      entry.temporalHistoryFramesUsed,
      entry.shotCooldownMs,
      entry.residualMotionPx === null ? "" : entry.residualMotionPx.toFixed(3),
      entry.trackedPointCount === null ? "" : entry.trackedPointCount,
    ].join(","),
  );
  return [header, ...rows].join("\n");
}

function mergeWindows(windows: TimeWindow[]): TimeWindow[] {
  if (windows.length === 0) return [];
  const sorted = [...windows].sort((a, b) => a.start - b.start);
  const merged: TimeWindow[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i += 1) {
    const current = sorted[i];
    const last = merged[merged.length - 1];

    if (current.start <= last.end) {
      last.end = Math.max(last.end, current.end);
    } else {
      merged.push({ ...current });
    }
  }

  return merged;
}

function openBinaryMask(mask: Uint8Array, width: number, height: number, kernelSize = 3): Uint8Array {
  const halfKernel = Math.max(1, Math.floor(kernelSize / 2));
  const eroded = new Uint8Array(mask.length);
  const opened = new Uint8Array(mask.length);

  for (let y = halfKernel; y < height - halfKernel; y += 1) {
    for (let x = halfKernel; x < width - halfKernel; x += 1) {
      let keep = 1;
      for (let dy = -halfKernel; dy <= halfKernel && keep === 1; dy += 1) {
        for (let dx = -halfKernel; dx <= halfKernel; dx += 1) {
          const idx = (y + dy) * width + (x + dx);
          if (mask[idx] === 0) {
            keep = 0;
            break;
          }
        }
      }
      eroded[y * width + x] = keep;
    }
  }

  for (let y = halfKernel; y < height - halfKernel; y += 1) {
    for (let x = halfKernel; x < width - halfKernel; x += 1) {
      let on = 0;
      for (let dy = -halfKernel; dy <= halfKernel && on === 0; dy += 1) {
        for (let dx = -halfKernel; dx <= halfKernel; dx += 1) {
          const idx = (y + dy) * width + (x + dx);
          if (eroded[idx] === 1) {
            on = 1;
            break;
          }
        }
      }
      opened[y * width + x] = on;
    }
  }

  return opened;
}

function buildSimpleBackgroundSubtractMask(
  currentRgba: Uint8ClampedArray,
  baselineRgba: Uint8ClampedArray,
  width: number,
  height: number,
  threshold = SIMPLE_CONTOUR_DIFF_THRESHOLD,
): Uint8Array {
  const pixelCount = width * height;
  const mask = new Uint8Array(pixelCount);
  if (pixelCount <= 0 || currentRgba.length < pixelCount * 4 || baselineRgba.length < pixelCount * 4) {
    return mask;
  }

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const currentGray =
      0.299 * currentRgba[idx] + 0.587 * currentRgba[idx + 1] + 0.114 * currentRgba[idx + 2];
    const baselineGray =
      0.299 * baselineRgba[idx] + 0.587 * baselineRgba[idx + 1] + 0.114 * baselineRgba[idx + 2];
    const diff = Math.max(0, baselineGray - currentGray);
    if (diff >= threshold) mask[i] = 1;
  }

  return openBinaryMask(mask, width, height, SIMPLE_CONTOUR_OPEN_KERNEL_SIZE);
}

function buildPositiveBackgroundDiffMap(
  currentRgba: Uint8ClampedArray,
  baselineRgba: Uint8ClampedArray,
  width: number,
  height: number,
): Float32Array {
  const pixelCount = width * height;
  const diffMap = new Float32Array(pixelCount);
  if (pixelCount <= 0 || currentRgba.length < pixelCount * 4 || baselineRgba.length < pixelCount * 4) {
    return diffMap;
  }

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const currentGray = 0.299 * currentRgba[idx] + 0.587 * currentRgba[idx + 1] + 0.114 * currentRgba[idx + 2];
    const baselineGray = 0.299 * baselineRgba[idx] + 0.587 * baselineRgba[idx + 1] + 0.114 * baselineRgba[idx + 2];
    diffMap[i] = Math.max(0, baselineGray - currentGray);
  }

  return diffMap;
}

function buildTemporalPersistenceMask(
  currentMask: Uint8Array,
  historyMasks: Uint8Array[],
  requiredVotes: number,
): Uint8Array {
  const safeVotes = Math.max(1, requiredVotes);
  const persistent = new Uint8Array(currentMask.length);
  for (let i = 0; i < currentMask.length; i += 1) {
    let votes = currentMask[i] === 1 ? 1 : 0;
    for (const historyMask of historyMasks) {
      if (historyMask.length !== currentMask.length) continue;
      if (historyMask[i] === 1) votes += 1;
    }
    if (votes >= safeVotes) persistent[i] = 1;
  }
  return persistent;
}

function summarizeRegionTemporalSupport(
  region: ChangedContourRegion,
  rawMask: Uint8Array,
  persistentMask: Uint8Array,
  positiveDiffMap: Float32Array,
  width: number,
  height: number,
): { rawPixels: number; persistentPixels: number; supportRatio: number; meanDiff: number } {
  const minX = Math.max(0, Math.min(width - 1, Math.floor(region.minX)));
  const maxX = Math.max(0, Math.min(width - 1, Math.floor(region.maxX)));
  const minY = Math.max(0, Math.min(height - 1, Math.floor(region.minY)));
  const maxY = Math.max(0, Math.min(height - 1, Math.floor(region.maxY)));
  if (minX > maxX || minY > maxY) {
    return { rawPixels: 0, persistentPixels: 0, supportRatio: 0, meanDiff: 0 };
  }

  let rawPixels = 0;
  let persistentPixels = 0;
  let diffTotal = 0;
  for (let y = minY; y <= maxY; y += 1) {
    const rowOffset = y * width;
    for (let x = minX; x <= maxX; x += 1) {
      const index = rowOffset + x;
      if (rawMask[index] === 1) rawPixels += 1;
      if (persistentMask[index] === 1) {
        persistentPixels += 1;
        diffTotal += positiveDiffMap[index] ?? 0;
      }
    }
  }

  const supportRatio = persistentPixels / Math.max(1, rawPixels);
  const meanDiff = persistentPixels > 0 ? diffTotal / persistentPixels : 0;
  return { rawPixels, persistentPixels, supportRatio, meanDiff };
}

function findBlobs(mask: Uint8Array, width: number, height: number, minPixels: number): BinaryBlob[] {
  const visited = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  const blobs: BinaryBlob[] = [];

  for (let startIdx = 0; startIdx < mask.length; startIdx += 1) {
    if (mask[startIdx] === 0 || visited[startIdx] === 1) continue;

    let stackSize = 0;
    stack[stackSize] = startIdx;
    stackSize += 1;
    visited[startIdx] = 1;

    let pixelCount = 0;
    let sumX = 0;
    let sumY = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    const indices: number[] = [];

    while (stackSize > 0) {
      stackSize -= 1;
      const idx = stack[stackSize];
      const x = idx % width;
      const y = Math.floor(idx / width);

      pixelCount += 1;
      indices.push(idx);
      sumX += x;
      sumY += y;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;

      if (x > 0) {
        const leftIdx = idx - 1;
        if (mask[leftIdx] === 1 && visited[leftIdx] === 0) {
          visited[leftIdx] = 1;
          stack[stackSize] = leftIdx;
          stackSize += 1;
        }
      }
      if (x < width - 1) {
        const rightIdx = idx + 1;
        if (mask[rightIdx] === 1 && visited[rightIdx] === 0) {
          visited[rightIdx] = 1;
          stack[stackSize] = rightIdx;
          stackSize += 1;
        }
      }
      if (y > 0) {
        const upIdx = idx - width;
        if (mask[upIdx] === 1 && visited[upIdx] === 0) {
          visited[upIdx] = 1;
          stack[stackSize] = upIdx;
          stackSize += 1;
        }
      }
      if (y < height - 1) {
        const downIdx = idx + width;
        if (mask[downIdx] === 1 && visited[downIdx] === 0) {
          visited[downIdx] = 1;
          stack[stackSize] = downIdx;
          stackSize += 1;
        }
      }
    }

    if (pixelCount >= minPixels) {
      blobs.push({
        pixelCount,
        centerX: sumX / pixelCount,
        centerY: sumY / pixelCount,
        minX,
        minY,
        maxX,
        maxY,
        indices,
      });
    }
  }

  blobs.sort((a, b) => b.pixelCount - a.pixelCount);
  return blobs;
}

function findChangedRegionsByContours(
  cv: CvApi,
  mask: Uint8Array,
  width: number,
  height: number,
  minPixels: number,
  maxPixels: number,
): ChangedContourRegion[] {
  const canUseContours =
    !!cv.findContours &&
    !!cv.contourArea &&
    !!cv.arcLength &&
    !!cv.boundingRect &&
    !!cv.moments &&
    cv.RETR_EXTERNAL !== undefined &&
    cv.CHAIN_APPROX_SIMPLE !== undefined;

  if (!canUseContours || typeof document === "undefined") {
    return findBlobs(mask, width, height, minPixels)
      .filter((blob) => blob.pixelCount < maxPixels)
      .map((blob) => ({
        pixelCount: blob.pixelCount,
        centerX: blob.centerX,
        centerY: blob.centerY,
        minX: blob.minX,
        minY: blob.minY,
        maxX: blob.maxX,
        maxY: blob.maxY,
        corePixels: blob.pixelCount,
        blackToColorPixels: blob.pixelCount,
        meanSeverity: 1,
      }));
  }

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d");
  if (!maskContext) {
    return findBlobs(mask, width, height, minPixels)
      .filter((blob) => blob.pixelCount < maxPixels)
      .map((blob) => ({
        pixelCount: blob.pixelCount,
        centerX: blob.centerX,
        centerY: blob.centerY,
        minX: blob.minX,
        minY: blob.minY,
        maxX: blob.maxX,
        maxY: blob.maxY,
        corePixels: blob.pixelCount,
        blackToColorPixels: blob.pixelCount,
        meanSeverity: 1,
      }));
  }

  const maskImageData = maskContext.createImageData(width, height);
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i] === 1 ? 255 : 0;
    const rgbaIndex = i * 4;
    maskImageData.data[rgbaIndex] = value;
    maskImageData.data[rgbaIndex + 1] = value;
    maskImageData.data[rgbaIndex + 2] = value;
    maskImageData.data[rgbaIndex + 3] = 255;
  }
  maskContext.putImageData(maskImageData, 0, 0);

  const sourceMat = cv.imread(maskCanvas);
  const grayMaskMat = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const regions: ChangedContourRegion[] = [];

  try {
    cv.cvtColor(sourceMat, grayMaskMat, cv.COLOR_RGBA2GRAY);
    cv.findContours?.(
      grayMaskMat,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL ?? 0,
      cv.CHAIN_APPROX_SIMPLE ?? 0,
    );

    const contourCount = contours.size ? contours.size() : 0;
    for (let i = 0; i < contourCount; i += 1) {
      const contour = contours.get?.(i);
      if (!contour) continue;

      try {
        const contourArea = Math.max(0, cv.contourArea?.(contour, false) ?? 0);
        const contourPixels = Math.max(0, Math.round(contourArea));
        if (contourPixels <= minPixels || contourPixels >= maxPixels) continue;
        const perimeter = Math.max(0, cv.arcLength?.(contour, true) ?? 0);
        if (perimeter <= 0) continue;
        const circularity = (perimeter * perimeter) / (4 * Math.PI * Math.max(1, contourArea));
        if (circularity <= SIMPLE_CONTOUR_CIRCULARITY_MIN || circularity >= SIMPLE_CONTOUR_CIRCULARITY_MAX) continue;

        const boundingRect = cv.boundingRect?.(contour);
        if (!boundingRect) continue;
        const minX = Math.max(0, Math.min(width - 1, Math.floor(boundingRect.x)));
        const minY = Math.max(0, Math.min(height - 1, Math.floor(boundingRect.y)));
        const maxX = Math.max(0, Math.min(width - 1, Math.ceil(boundingRect.x + boundingRect.width - 1)));
        const maxY = Math.max(0, Math.min(height - 1, Math.ceil(boundingRect.y + boundingRect.height - 1)));
        if (minX > maxX || minY > maxY) continue;

        let changedPixels = 0;
        let sumX = 0;
        let sumY = 0;

        for (let y = minY; y <= maxY; y += 1) {
          const rowOffset = y * width;
          for (let x = minX; x <= maxX; x += 1) {
            const index = rowOffset + x;
            if (mask[index] !== 1) continue;
            changedPixels += 1;
            sumX += x;
            sumY += y;
          }
        }
        if (changedPixels <= minPixels || changedPixels >= maxPixels) continue;

        const circle = cv.minEnclosingCircle?.(contour);
        const moments = !circle ? cv.moments?.(contour, false) : null;
        const centerFromMomentsX =
          circle?.center?.x ??
          (moments && Number.isFinite(moments.m00) && moments.m00 !== 0
            ? moments.m10 / moments.m00
            : minX + (maxX - minX) / 2);
        const centerFromMomentsY =
          circle?.center?.y ??
          (moments && Number.isFinite(moments.m00) && moments.m00 !== 0
            ? moments.m01 / moments.m00
            : minY + (maxY - minY) / 2);
        const centerX = changedPixels > 0 ? sumX / changedPixels : centerFromMomentsX;
        const centerY = changedPixels > 0 ? sumY / changedPixels : centerFromMomentsY;

        regions.push({
          pixelCount: changedPixels,
          centerX,
          centerY,
          minX,
          minY,
          maxX,
          maxY,
          corePixels: changedPixels,
          blackToColorPixels: changedPixels,
          meanSeverity: 1,
        });
      } finally {
        contour.delete();
      }
    }
  } finally {
    hierarchy.delete();
    contours.delete();
    grayMaskMat.delete();
    sourceMat.delete();
  }

  regions.sort((a, b) => b.pixelCount - a.pixelCount);
  return regions;
}

function rmsToDbfs(rms: number): number {
  return 20 * Math.log10(Math.max(rms, 1e-6));
}

function nearestAudioSampleAtTime(timeSec: number, samples: AudioRmsSample[]): AudioRmsSample | null {
  if (samples.length === 0) return null;
  let nearest: AudioRmsSample | null = null;
  let nearestDelta = Number.POSITIVE_INFINITY;
  for (const sample of samples) {
    const delta = Math.abs(sample.timeSec - timeSec);
    if (delta < nearestDelta) {
      nearest = sample;
      nearestDelta = delta;
    }
  }
  return nearest;
}

function peakAudioDbAroundTime(timeSec: number, samples: AudioRmsSample[], halfWindowSec = 0.09): number | null {
  if (samples.length === 0) return null;
  let peakDb = Number.NEGATIVE_INFINITY;
  let found = false;
  for (const sample of samples) {
    if (Math.abs(sample.timeSec - timeSec) > halfWindowSec) continue;
    if (sample.dbfs > peakDb) peakDb = sample.dbfs;
    found = true;
  }
  if (!found) {
    const nearest = nearestAudioSampleAtTime(timeSec, samples);
    return nearest ? nearest.dbfs : null;
  }
  return peakDb;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function computeShotAudioMetrics(
  shotTimeSec: number,
  changeScore: number,
  nearestSpikeDeltaSec: number | null,
  nearestSpikeStrengthRms: number | null,
  audioTimeline: AudioRmsSample[],
  audioMeanDbfs: number,
  tweaks: TweakSettings,
): {
  audioDecibelDbfs: number | null;
  audioDeltaFromMeanDb: number | null;
  audioPeakDbfs: number | null;
  audioCorrelationScorePct: number | null;
  nearestSpikeDecibelDbfs: number | null;
} {
  const nearestSpikeDecibelDbfs = nearestSpikeStrengthRms === null ? null : rmsToDbfs(nearestSpikeStrengthRms);
  const nearestDb = nearestAudioSampleAtTime(shotTimeSec, audioTimeline)?.dbfs ?? null;
  const peakDb = peakAudioDbAroundTime(shotTimeSec, audioTimeline, tweaks.audioPeakWindowHalfSec);
  const selectedDb = peakDb ?? nearestDb ?? nearestSpikeDecibelDbfs;
  if (selectedDb === null) {
    return {
      audioDecibelDbfs: null,
      audioDeltaFromMeanDb: null,
      audioPeakDbfs: null,
      audioCorrelationScorePct: null,
      nearestSpikeDecibelDbfs,
    };
  }

  const deltaFromMeanDb = selectedDb - audioMeanDbfs;
  const audioEnergyScore = clamp01((deltaFromMeanDb + tweaks.audioEnergyOffsetDb) / tweaks.audioEnergyScaleDb);
  const timeAlignmentScore =
    nearestSpikeDeltaSec === null ? tweaks.audioNoSpikeAlignmentScore : clamp01(1 - nearestSpikeDeltaSec / tweaks.spikeIntensiveFocusSec);
  const visualScore = clamp01(changeScore / Math.max(tweaks.audioVisualScoreFloor, tweaks.minVisibleHitChangeScore * 4));
  const correlationPct =
    (audioEnergyScore * tweaks.audioWeightEnergy +
      timeAlignmentScore * tweaks.audioWeightTimeAlignment +
      visualScore * tweaks.audioWeightVisual) *
    100;

  return {
    audioDecibelDbfs: selectedDb,
    audioDeltaFromMeanDb: deltaFromMeanDb,
    audioPeakDbfs: peakDb,
    audioCorrelationScorePct: correlationPct,
    nearestSpikeDecibelDbfs,
  };
}

function detectAudioSpikes(audioBuffer: AudioBuffer, videoDurationSec: number, tweaks: TweakSettings) {
  const channelData = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const windowSize = Math.max(64, Math.round(tweaks.audioWindowSize));

  const rmsSamples: AudioRmsSample[] = [];

  for (let i = 0; i + windowSize < channelData.length; i += windowSize) {
    let total = 0;
    for (let j = 0; j < windowSize; j += 1) {
      const sample = channelData[i + j];
      total += sample * sample;
    }

    const rms = Math.sqrt(total / windowSize);
    const timeSec = i / sampleRate;
    rmsSamples.push({ timeSec, rms, dbfs: rmsToDbfs(rms) });
  }

  if (rmsSamples.length === 0) {
    return {
      spikes: [] as SpikeMetadata[],
      spriteMap: {} as Record<string, [number, number]>,
      rmsTimeline: [] as AudioRmsSample[],
      meanDbfs: -120,
      thresholdDbfs: -120,
    };
  }

  const mean = rmsSamples.reduce((sum, entry) => sum + entry.rms, 0) / rmsSamples.length;
  const variance =
    rmsSamples.reduce((sum, entry) => sum + (entry.rms - mean) ** 2, 0) / rmsSamples.length;
  const stdDev = Math.sqrt(variance);
  const threshold = mean + stdDev * tweaks.audioSpikeStdDevMultiplier;
  const meanDbfs = rmsToDbfs(mean);
  const thresholdDbfs = rmsToDbfs(threshold);

  const spikes: SpikeMetadata[] = [];
  let lastAccepted = -1;

  for (const sample of rmsSamples) {
    if (sample.rms < threshold) continue;
    if (sample.timeSec - lastAccepted < tweaks.audioSpikeMinGapSec) continue;

    const windowStartSec = Math.max(0, sample.timeSec - tweaks.spikeWindowHalfSec);
    const windowEndSec = Math.min(videoDurationSec, sample.timeSec + tweaks.spikeWindowHalfSec);
    const spriteStartSec = Math.max(0, sample.timeSec - tweaks.spikeSpriteLeadSec);
    const spriteDurationMs = Math.max(1, Math.round(tweaks.spikeSpriteDurationMs));

    const spike: SpikeMetadata = {
      id: `spike_${spikes.length + 1}`,
      timeSec: sample.timeSec,
      strength: sample.rms,
      spriteStartMs: Math.round(spriteStartSec * 1000),
      spriteDurationMs,
      windowStartSec,
      windowEndSec,
    };

    spikes.push(spike);
    lastAccepted = sample.timeSec;
  }

  const spriteMap: Record<string, [number, number]> = {};
  for (const spike of spikes) {
    spriteMap[spike.id] = [spike.spriteStartMs, spike.spriteDurationMs];
  }

  return { spikes, spriteMap, rmsTimeline: rmsSamples, meanDbfs, thresholdDbfs };
}

function seekVideo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const targetTime = Math.max(0, Math.min(timeSec, video.duration || timeSec));
    if (Math.abs(video.currentTime - targetTime) < 0.01) {
      resolve();
      return;
    }

    const onSeeked = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(new Error("Video seek failed during spike-window scan."));
    };

    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out seeking video frame for analysis."));
    }, 2000);

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };

    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = targetTime;
  });
}

async function playVideoForAnalysis(video: HTMLVideoElement): Promise<void> {
  try {
    await video.play();
  } catch (error) {
    // Retry muted to satisfy autoplay policies on stricter browsers.
    video.muted = true;
    video.defaultMuted = true;
    try {
      await video.play();
    } catch {
      throw error;
    }
  }
}

function clampRectToFrame(
  x: number,
  y: number,
  width: number,
  height: number,
  frameWidth: number,
  frameHeight: number,
) {
  const clampedWidth = Math.max(1, Math.min(width, frameWidth));
  const clampedHeight = Math.max(1, Math.min(height, frameHeight));
  const clampedX = Math.max(0, Math.min(x, frameWidth - clampedWidth));
  const clampedY = Math.max(0, Math.min(y, frameHeight - clampedHeight));

  return { x: clampedX, y: clampedY, width: clampedWidth, height: clampedHeight };
}

function estimateDistanceInchesFromDetection(
  targetWidthInches: number,
  targetHeightInches: number,
  pixelsPerInch: number,
  detectedWidthPixels: number,
  detectedHeightPixels: number,
  focalScalePxIn: number,
  manualDistanceOverrideInches: number,
): number | null {
  if (manualDistanceOverrideInches > 0) return manualDistanceOverrideInches;

  const focalEstimates: number[] = [];
  if (focalScalePxIn > 0) {
    if (targetWidthInches > 0 && detectedWidthPixels > 0) {
      focalEstimates.push((focalScalePxIn * targetWidthInches) / detectedWidthPixels);
    }
    if (targetHeightInches > 0 && detectedHeightPixels > 0) {
      focalEstimates.push((focalScalePxIn * targetHeightInches) / detectedHeightPixels);
    }
  }
  if (focalEstimates.length > 0) {
    return focalEstimates.reduce((sum, value) => sum + value, 0) / focalEstimates.length;
  }

  if (pixelsPerInch <= 0) return null;

  const estimates: number[] = [];

  const estimateSingleAxis = (realInches: number, observedPixels: number) => {
    if (realInches <= 0 || observedPixels <= 0) return;
    const distanceInches = (realInches * pixelsPerInch) / observedPixels;
    if (Number.isFinite(distanceInches) && distanceInches > 0) {
      estimates.push(distanceInches);
    }
  };

  estimateSingleAxis(targetWidthInches, detectedWidthPixels);
  estimateSingleAxis(targetHeightInches, detectedHeightPixels);

  if (estimates.length === 0) return null;
  return estimates.reduce((sum, value) => sum + value, 0) / estimates.length;
}

export default function Home() {
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);
  const [selectedImagePreviewUrl, setSelectedImagePreviewUrl] = useState<string | null>(null);
  const [selectedVideoName, setSelectedVideoName] = useState<string | null>(null);
  const [selectedVideoPreviewUrl, setSelectedVideoPreviewUrl] = useState<string | null>(null);
  const [captureMode, setCaptureMode] = useState<"upload" | "stream">("upload");
  const [streamUrl, setStreamUrl] = useState("");
  const [opencvReady, setOpenCvReady] = useState(false);
  const [opencvError, setOpenCvError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState("Idle");
  const [detectionEnabled, setDetectionEnabled] = useState(false);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const [targetWidthInches, setTargetWidthInches] = useState(12);
  const [targetHeightInches, setTargetHeightInches] = useState(12);
  const [pixelsPerInch, setPixelsPerInch] = useState(0);
  const [calibrationDistanceInches, setCalibrationDistanceInches] = useState(252);
  const [manualDistanceOverrideInches] = useState(0);
  const [focalScalePxIn, setFocalScalePxIn] = useState(0);
  const [matchThreshold, setMatchThreshold] = useState(20);
  const [trackingMode, setTrackingMode] = useState<TrackingMode>("template");
  const [logEntries, setLogEntries] = useState<DetectionLogEntry[]>([]);
  const [shotLogEntries, setShotLogEntries] = useState<ShotLogEntry[]>([]);
  const [lastDetection, setLastDetection] = useState<DetectionLogEntry | null>(null);
  const [lastShot, setLastShot] = useState<ShotLogEntry | null>(null);
  const [roiRect, setRoiRect] = useState<RoiRect | null>(null);
  const [isSelectingRoi, setIsSelectingRoi] = useState(false);
  const [howlerReady, setHowlerReady] = useState(false);
  const [howlerError, setHowlerError] = useState<string | null>(null);
  const [spikeMetadata, setSpikeMetadata] = useState<SpikeMetadata[]>([]);
  const [audioSprites, setAudioSprites] = useState<Record<string, [number, number]>>({});
  const [spritesReady, setSpritesReady] = useState(false);
  const [expandedSpikeIds, setExpandedSpikeIds] = useState<string[]>([]);
  const [unitConversionEnabled, setUnitConversionEnabled] = useState(false);
  const [displayLinearUnit, setDisplayLinearUnit] = useState<LinearUnit>("in");
  const [tweakSettings, setTweakSettings] = useState<TweakSettings>(DEFAULT_TWEAK_SETTINGS);
  const [isGearsExpanded, setIsGearsExpanded] = useState(false);

  const activeLinearUnit: LinearUnit = unitConversionEnabled ? displayLinearUnit : "in";
  const activeLinearUnitLabel = LINEAR_UNIT_LABELS[activeLinearUnit];
  const activeLinearInputStep = LINEAR_INPUT_STEPS[activeLinearUnit];
  const toDisplayLinearValue = (valueInches: number, fractionDigits = 3) => {
    const converted = convertFromInches(valueInches, activeLinearUnit);
    if (!Number.isFinite(converted)) return 0;
    return Number(converted.toFixed(fractionDigits));
  };
  const fromDisplayLinearValue = (raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return 0;
    return Math.max(0, convertToInches(parsed, activeLinearUnit));
  };
  const formatLinearFromInches = (valueInches: number, fractionDigits = 1) =>
    `${toDisplayLinearValue(valueInches, fractionDigits).toFixed(fractionDigits)} ${activeLinearUnitLabel}`;

  const shotClustering = useMemo<ShotClusteringResult>(
    () => clusterShotsBySpaceTime(shotLogEntries, tweakSettings),
    [shotLogEntries, tweakSettings],
  );
  const clusterColorById = useMemo<Record<number, string>>(() => {
    const map: Record<number, string> = {};
    for (const cluster of shotClustering.clusters) {
      map[cluster.clusterId] = clusterColorForId(cluster.clusterId);
    }
    return map;
  }, [shotClustering.clusters]);
  const changedTweakCount = useMemo(
    () =>
      (Object.keys(DEFAULT_TWEAK_SETTINGS) as Array<keyof TweakSettings>).filter(
        (key) => tweakSettings[key] !== DEFAULT_TWEAK_SETTINGS[key],
      ).length,
    [tweakSettings],
  );
  const hasUploadedVideo = captureMode === "upload" && !!selectedVideoPreviewUrl;
  const hasReferenceFrame = !!selectedImagePreviewUrl;
  const hasDrawnGeometry = !!roiRect && roiRect.width >= 0.01 && roiRect.height >= 0.01;
  const hasScaleCalibration = pixelsPerInch > 0 || focalScalePxIn > 0;
  const hasResultData = logEntries.length > 0 || shotLogEntries.length > 0;
  const workflowStep: WorkflowStep =
    captureMode !== "upload"
      ? "upload_video"
      : !hasUploadedVideo
        ? "upload_video"
        : !hasReferenceFrame
          ? "capture_frame"
          : !hasDrawnGeometry
            ? "draw_geometry"
            : !hasScaleCalibration
              ? "calibrate"
              : isScanning || !hasResultData
                ? "scan"
                : "export";
  const highlightActionClass =
    "ring-2 ring-amber-300 border-amber-300 bg-amber-500/10 text-amber-100 shadow-[0_0_0_1px_rgba(252,211,77,0.35)]";

  const updateTweakSetting = (key: keyof TweakSettings, value: number) => {
    if (!Number.isFinite(value)) return;
    setTweakSettings((current) => ({ ...current, [key]: value }));
  };

  const spikeShotSummaryById = useMemo<Record<string, SpikeShotSummary>>(() => {
    const shotsBySpikeId = new Map<string, ShotLogEntry[]>();
    for (const spike of spikeMetadata) {
      shotsBySpikeId.set(spike.id, []);
    }

    for (const shot of shotLogEntries) {
      let linkedSpikeId = shot.nearestSpikeId;
      if (!linkedSpikeId || !shotsBySpikeId.has(linkedSpikeId)) {
        let nearestWindowSpikeId: string | null = null;
        let nearestDelta = Number.POSITIVE_INFINITY;
        for (const spike of spikeMetadata) {
          if (shot.videoTimeSec < spike.windowStartSec || shot.videoTimeSec > spike.windowEndSec) continue;
          const delta = Math.abs(shot.videoTimeSec - spike.timeSec);
          if (delta < nearestDelta) {
            nearestDelta = delta;
            nearestWindowSpikeId = spike.id;
          }
        }
        linkedSpikeId = nearestWindowSpikeId;
      }

      if (!linkedSpikeId) continue;
      const bucket = shotsBySpikeId.get(linkedSpikeId);
      if (!bucket) continue;
      bucket.push(shot);
    }

    const summaryById: Record<string, SpikeShotSummary> = {};
    for (const spike of spikeMetadata) {
      const shots = [...(shotsBySpikeId.get(spike.id) ?? [])].sort((a, b) => a.videoTimeSec - b.videoTimeSec);
      const count = shots.length;
      if (count === 0) {
        summaryById[spike.id] = {
          shots,
          count,
          meanPointOfImpactX: null,
          meanPointOfImpactY: null,
          extremeSpreadPx: null,
          horizontalSpreadPx: null,
          verticalSpreadPx: null,
          meanRadiusPx: null,
          averageDiameterInches: null,
          averageChangeScore: null,
        };
        continue;
      }

      let sumX = 0;
      let sumY = 0;
      let sumChangeScore = 0;
      let minX = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      const diametersInches: number[] = [];

      for (const shot of shots) {
        sumX += shot.centerX;
        sumY += shot.centerY;
        sumChangeScore += shot.changeScore;
        if (shot.centerX < minX) minX = shot.centerX;
        if (shot.centerX > maxX) maxX = shot.centerX;
        if (shot.centerY < minY) minY = shot.centerY;
        if (shot.centerY > maxY) maxY = shot.centerY;
        if (shot.estimatedDiameterInches !== null) diametersInches.push(shot.estimatedDiameterInches);
      }

      const meanPointOfImpactX = sumX / count;
      const meanPointOfImpactY = sumY / count;
      let meanRadiusPx = 0;
      for (const shot of shots) {
        meanRadiusPx += Math.hypot(shot.centerX - meanPointOfImpactX, shot.centerY - meanPointOfImpactY);
      }
      meanRadiusPx /= count;

      let extremeSpreadPx = 0;
      for (let i = 0; i < shots.length; i += 1) {
        for (let j = i + 1; j < shots.length; j += 1) {
          const spread = Math.hypot(shots[i].centerX - shots[j].centerX, shots[i].centerY - shots[j].centerY);
          if (spread > extremeSpreadPx) extremeSpreadPx = spread;
        }
      }

      summaryById[spike.id] = {
        shots,
        count,
        meanPointOfImpactX,
        meanPointOfImpactY,
        extremeSpreadPx,
        horizontalSpreadPx: maxX - minX,
        verticalSpreadPx: maxY - minY,
        meanRadiusPx,
        averageDiameterInches:
          diametersInches.length > 0
            ? diametersInches.reduce((sum, value) => sum + value, 0) / diametersInches.length
            : null,
        averageChangeScore: sumChangeScore / count,
      };
    }

    return summaryById;
  }, [shotLogEntries, spikeMetadata]);

  useEffect(() => {
    setExpandedSpikeIds((current) => current.filter((id) => spikeMetadata.some((spike) => spike.id === id)));
  }, [spikeMetadata]);

  const toggleSpikeExpanded = (spikeId: string) => {
    setExpandedSpikeIds((current) =>
      current.includes(spikeId) ? current.filter((id) => id !== spikeId) : [...current, spikeId],
    );
  };

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const processedContourCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imagePreviewRef = useRef<HTMLImageElement | null>(null);
  const videoSourceSectionRef = useRef<HTMLElement | null>(null);
  const captureFrameButtonRef = useRef<HTMLButtonElement | null>(null);
  const roiContainerRef = useRef<HTMLDivElement | null>(null);
  const calibrationSectionRef = useRef<HTMLElement | null>(null);
  const startScanButtonRef = useRef<HTMLButtonElement | null>(null);
  const exportButtonsRef = useRef<HTMLDivElement | null>(null);
  const previousWorkflowStepRef = useRef<WorkflowStep | null>(null);
  const howlRef = useRef<HowlInstance | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const spikeWindowsRef = useRef<TimeWindow[]>([]);
  const audioReadyRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const playPauseTimeoutRef = useRef<number | null>(null);
  const frameIndexRef = useRef(0);
  const lastLoggedAtMsRef = useRef(0);
  const lastDetectionSampleAtMsRef = useRef(0);
  const lastShotAtMsRef = useRef(0);
  const roiRectRef = useRef<RoiRect | null>(null);
  const roiStartRef = useRef<{ x: number; y: number } | null>(null);
  const spikeEventsRef = useRef<SpikeMetadata[]>([]);
  const scanVideoRef = useRef<HTMLVideoElement | null>(null);
  const analyzedPlaybackVideoRef = useRef<HTMLVideoElement | null>(null);
  const analyzedPlaybackRafRef = useRef<number | null>(null);
  const detectionTimelineRef = useRef<DetectionLogEntry[]>([]);
  const contourWindowTimelineRef = useRef<ContourWindowFrameSnapshot[]>([]);
  const lastProcessedContourSnapshotRef = useRef<ContourWindowFrameSnapshot | null>(null);
  const shotMarkersRef = useRef<ShotLogEntry[]>([]);
  const shotSequenceRef = useRef(0);
  const liveShotClusteringRef = useRef<ShotClusteringResult>({
    selectedK: 0,
    finalK: 0,
    closeMergeCount: 0,
    objectiveScore: 0,
    shotClusterById: {},
    clusters: [],
  });
  const liveClusterColorByIdRef = useRef<Record<number, string>>({});
  const liveClusterShotCountRef = useRef(0);
  const audioRmsTimelineRef = useRef<AudioRmsSample[]>([]);
  const audioMeanDbfsRef = useRef(-120);
  const audioThresholdDbfsRef = useRef(-120);
  const activeSpikeWindowIndexRef = useRef<number>(-1);
  const pendingShotCandidateRef = useRef<PendingShotCandidate | null>(null);
  const lastUiDetectionEnabledRef = useRef(false);
  const lastUiDetectionConfidenceRef = useRef(-1);
  const scanTaskActiveRef = useRef(false);
  const restartScanRequestedRef = useRef(false);
  const lastAutoCalibrationKeyRef = useRef<string>("");
  const gearsSettingsHydratedRef = useRef(false);

  const drawContourSnapshot = useCallback((snapshot: ContourWindowFrameSnapshot | null): boolean => {
    if (!snapshot) return false;
    const contourCanvas = processedContourCanvasRef.current;
    const contourContext = contourCanvas?.getContext("2d") ?? null;
    if (!contourCanvas || !contourContext) return false;
    drawPersistedContourWindowView(contourContext, snapshot);
    return true;
  }, []);

  const persistContourSnapshot = useCallback((snapshot: ContourWindowFrameSnapshot | null) => {
    if (!snapshot) return;
    lastProcessedContourSnapshotRef.current = snapshot;
    if (typeof window === "undefined") return;
    try {
      sessionStorage.setItem(PROCESSED_CONTOUR_SNAPSHOT_STORAGE_KEY, JSON.stringify(snapshot));
    } catch {
      // Ignore storage write failures.
    }
  }, []);

  const restorePersistedContourSnapshot = useCallback(() => {
    if (drawContourSnapshot(lastProcessedContourSnapshotRef.current)) return true;
    if (typeof window === "undefined") return false;
    try {
      const raw = sessionStorage.getItem(PROCESSED_CONTOUR_SNAPSHOT_STORAGE_KEY);
      if (!raw) return false;
      const parsed = sanitizeContourWindowSnapshot(JSON.parse(raw));
      if (!parsed) return false;
      lastProcessedContourSnapshotRef.current = parsed;
      if (contourWindowTimelineRef.current.length === 0) {
        contourWindowTimelineRef.current = [parsed];
      }
      return drawContourSnapshot(parsed);
    } catch {
      return false;
    }
  }, [drawContourSnapshot]);

  useEffect(() => {
    restorePersistedContourSnapshot();
  }, [restorePersistedContourSnapshot]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    try {
      const raw = window.localStorage.getItem(GEARS_SETTINGS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as StoredGearsSettings;
      if (typeof parsed.unitConversionEnabled === "boolean") {
        setUnitConversionEnabled(parsed.unitConversionEnabled);
      }
      if (isLinearUnit(parsed.displayLinearUnit)) {
        setDisplayLinearUnit(parsed.displayLinearUnit);
      }
      const sanitizedTweaks = sanitizeStoredTweakSettings(parsed.tweakSettings);
      if (sanitizedTweaks) {
        setTweakSettings((current) => ({ ...current, ...sanitizedTweaks }));
      }
      if (typeof parsed.isGearsExpanded === "boolean") {
        setIsGearsExpanded(parsed.isGearsExpanded);
      }
    } catch {
      // Ignore malformed cached settings.
    } finally {
      gearsSettingsHydratedRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !gearsSettingsHydratedRef.current) return;

    const payload: StoredGearsSettings = {
      unitConversionEnabled,
      displayLinearUnit,
      tweakSettings,
      isGearsExpanded,
    };
    try {
      window.localStorage.setItem(GEARS_SETTINGS_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Ignore storage write errors (e.g., quota/privacy mode).
    }
  }, [unitConversionEnabled, displayLinearUnit, tweakSettings, isGearsExpanded]);

  useEffect(() => {
    return () => {
      revokeBlobUrl(selectedImagePreviewUrl);
    };
  }, [selectedImagePreviewUrl]);

  useEffect(() => {
    return () => {
      revokeBlobUrl(selectedVideoPreviewUrl);
    };
  }, [selectedVideoPreviewUrl]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const markReady = () => {
      setOpenCvReady(true);
      setOpenCvError(null);
      setScanStatus("OpenCV.js ready");
    };

    const existingCv = window.cv;
    if (existingCv?.Mat) {
      markReady();
      return;
    }

    const existingScript = document.getElementById("opencv-script") as HTMLScriptElement | null;
    if (existingScript) return;

    const script = document.createElement("script");
    script.id = "opencv-script";
    script.src = "https://docs.opencv.org/4.x/opencv.js";
    script.async = true;

    script.onerror = () => {
      setOpenCvError("Failed to load OpenCV.js.");
      setScanStatus("OpenCV.js failed to load");
    };

    script.onload = () => {
      const cv = window.cv;
      if (!cv) {
        setOpenCvError("OpenCV.js loaded but cv is unavailable.");
        return;
      }

      if (cv.Mat) {
        markReady();
        return;
      }

      cv.onRuntimeInitialized = markReady;
    };

    document.body.appendChild(script);
  }, []);

  useEffect(() => {
    const previousStep = previousWorkflowStepRef.current;
    if (previousStep === null) {
      previousWorkflowStepRef.current = workflowStep;
      return;
    }
    if (previousStep === workflowStep) return;
    previousWorkflowStepRef.current = workflowStep;
    if (previousStep === "draw_geometry" && (workflowStep === "calibrate" || workflowStep === "scan")) {
      return;
    }

    const targetElement =
      workflowStep === "upload_video"
        ? videoSourceSectionRef.current
        : workflowStep === "capture_frame"
          ? captureFrameButtonRef.current
          : workflowStep === "draw_geometry"
            ? roiContainerRef.current ?? captureFrameButtonRef.current
            : workflowStep === "calibrate"
              ? calibrationSectionRef.current
              : workflowStep === "scan"
                ? startScanButtonRef.current
                : exportButtonsRef.current;

    if (!targetElement) return;

    window.requestAnimationFrame(() => {
      targetElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "nearest",
      });
    });
  }, [workflowStep]);

  useEffect(() => {
    return () => {
      if (howlRef.current) {
        howlRef.current.unload();
        howlRef.current = null;
      }
      if (analyzedPlaybackRafRef.current) {
        cancelAnimationFrame(analyzedPlaybackRafRef.current);
        analyzedPlaybackRafRef.current = null;
      }
      if (analyzedPlaybackVideoRef.current) {
        analyzedPlaybackVideoRef.current.pause();
        analyzedPlaybackVideoRef.current = null;
      }
      if (playPauseTimeoutRef.current) {
        window.clearTimeout(playPauseTimeoutRef.current);
        playPauseTimeoutRef.current = null;
      }
    };
  }, []);

  const stopAnalyzedPlayback = () => {
    if (analyzedPlaybackRafRef.current) {
      cancelAnimationFrame(analyzedPlaybackRafRef.current);
      analyzedPlaybackRafRef.current = null;
    }
    if (analyzedPlaybackVideoRef.current) {
      analyzedPlaybackVideoRef.current.pause();
      analyzedPlaybackVideoRef.current = null;
    }
  };

  const resetShotFlowState = () => {
    activeSpikeWindowIndexRef.current = -1;
    pendingShotCandidateRef.current = null;
    lastShotAtMsRef.current = 0;
  };

  useEffect(() => {
    if (typeof window === "undefined") return;

    if (window.Howl) {
      setHowlerReady(true);
      setHowlerError(null);
      return;
    }

    const existingScript = document.getElementById("howler-script") as HTMLScriptElement | null;
    if (existingScript) return;

    const script = document.createElement("script");
    script.id = "howler-script";
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.4/howler.min.js";
    script.async = true;

    script.onerror = () => {
      setHowlerError("Failed to load Howler.js.");
      setHowlerReady(false);
    };

    script.onload = () => {
      if (window.Howl) {
        setHowlerReady(true);
        setHowlerError(null);
      } else {
        setHowlerError("Howler.js loaded but Howl constructor is unavailable.");
      }
    };

    document.body.appendChild(script);
  }, []);

  const onFileSelection =
    (type: "image" | "video") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null;
      if (!file) return;

      const previewUrl = URL.createObjectURL(file);

      if (type === "image") {
        revokeBlobUrl(selectedImagePreviewUrl);
        setSelectedImageName(file.name);
        setSelectedImagePreviewUrl(previewUrl);
        setRoiRect(null);
        roiRectRef.current = null;
        setFocalScalePxIn(0);
        clearTemplateRegionCache();
        return;
      }

      revokeBlobUrl(selectedVideoPreviewUrl);
      setSelectedVideoName(file.name);
      setSelectedVideoFile(file);
      setSelectedVideoPreviewUrl(previewUrl);
      setSelectedImageName(null);
      revokeBlobUrl(selectedImagePreviewUrl);
      setSelectedImagePreviewUrl(null);
      setRoiRect(null);
      roiRectRef.current = null;
      setFocalScalePxIn(0);
      clearTemplateRegionCache();
      setSpikeMetadata([]);
      setAudioSprites({});
      setSpritesReady(false);
      spikeEventsRef.current = [];
    };

  const captureReferenceFrameFromVideo = () => {
    const videoEl = videoRef.current;
    if (!videoEl || !selectedVideoPreviewUrl) {
      setScanStatus("Upload a video first, then capture a reference frame.");
      return;
    }

    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    if (width <= 0 || height <= 0) {
      setScanStatus("Video frame unavailable. Play or seek the video first, then capture.");
      return;
    }

    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = width;
    frameCanvas.height = height;
    const frameCtx = frameCanvas.getContext("2d");
    if (!frameCtx) {
      setScanStatus("Unable to capture reference frame.");
      return;
    }

    frameCtx.drawImage(videoEl, 0, 0, width, height);
    const frameDataUrl = frameCanvas.toDataURL("image/png");
    revokeBlobUrl(selectedImagePreviewUrl);
    setSelectedImageName(
      `${(selectedVideoName ?? "video").replace(/\.[^/.]+$/, "")}-frame-${videoEl.currentTime.toFixed(2)}s.png`,
    );
    setSelectedImagePreviewUrl(frameDataUrl);
    setRoiRect(null);
    roiRectRef.current = null;
    setFocalScalePxIn(0);
    clearTemplateRegionCache();
    setScanStatus("Reference frame captured. Drag to select target geometry.");
    window.alert("Reference frame captured. Please draw a box around the target.");
  };

  const applyCalibrationFromSelection = useCallback(
    async ({
      suppressPreconditionStatus = false,
      successStatus,
      failureStatus,
    }: {
      suppressPreconditionStatus?: boolean;
      successStatus: string;
      failureStatus: string;
    }) => {
      const activeRoi = roiRectRef.current ?? roiRect;
      if (!selectedImagePreviewUrl || !activeRoi) {
        if (!suppressPreconditionStatus) {
          setScanStatus("Capture a reference frame and draw target geometry first.");
        }
        return false;
      }
      if (targetWidthInches <= 0 && targetHeightInches <= 0) {
        if (!suppressPreconditionStatus) {
          setScanStatus("Enter target width or height before calibration.");
        }
        return false;
      }

      try {
        const { widthPx, heightPx } = await getSelectionPixelSize(selectedImagePreviewUrl, activeRoi);
        const ppiEstimates: number[] = [];
        if (targetWidthInches > 0) ppiEstimates.push(widthPx / targetWidthInches);
        if (targetHeightInches > 0) ppiEstimates.push(heightPx / targetHeightInches);

        if (ppiEstimates.length > 0) {
          const nextPpi = ppiEstimates.reduce((sum, value) => sum + value, 0) / ppiEstimates.length;
          setPixelsPerInch(nextPpi);
        }

        if (calibrationDistanceInches > 0) {
          const focalEstimates: number[] = [];
          if (targetWidthInches > 0) focalEstimates.push((widthPx * calibrationDistanceInches) / targetWidthInches);
          if (targetHeightInches > 0) focalEstimates.push((heightPx * calibrationDistanceInches) / targetHeightInches);
          if (focalEstimates.length > 0) {
            const nextFocalScale = focalEstimates.reduce((sum, value) => sum + value, 0) / focalEstimates.length;
            setFocalScalePxIn(nextFocalScale);
          }
        } else {
          setFocalScalePxIn(0);
        }

        setScanStatus(successStatus);
        return true;
      } catch {
        setScanStatus(failureStatus);
        return false;
      }
    },
    [roiRect, selectedImagePreviewUrl, targetWidthInches, targetHeightInches, calibrationDistanceInches],
  );

  const goToNextAfterGeometrySelection = () => {
    const nextTarget = hasScaleCalibration ? startScanButtonRef.current : calibrationSectionRef.current;
    if (!nextTarget) return;

    nextTarget.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
    setScanStatus(
      hasScaleCalibration
        ? "Geometry selected. Proceed to Start Scan when ready."
        : "Geometry selected. Set target dimensions and calibration values next.",
    );
  };

  useEffect(() => {
    const activeRoi = roiRectRef.current ?? roiRect;
    if (!selectedImagePreviewUrl || !activeRoi) return;
    if (targetWidthInches <= 0 && targetHeightInches <= 0) return;

    const autoCalibrationKey = [
      selectedImagePreviewUrl,
      activeRoi.x.toFixed(6),
      activeRoi.y.toFixed(6),
      activeRoi.width.toFixed(6),
      activeRoi.height.toFixed(6),
      targetWidthInches.toFixed(4),
      targetHeightInches.toFixed(4),
      calibrationDistanceInches.toFixed(4),
    ].join("|");
    if (lastAutoCalibrationKeyRef.current === autoCalibrationKey) return;
    lastAutoCalibrationKeyRef.current = autoCalibrationKey;

    void applyCalibrationFromSelection({
      suppressPreconditionStatus: true,
      successStatus: "Calibration auto-updated from drawn target geometry.",
      failureStatus: "Failed to auto-calibrate from selected target geometry.",
    });
  }, [
    applyCalibrationFromSelection,
    calibrationDistanceInches,
    roiRect,
    selectedImagePreviewUrl,
    targetHeightInches,
    targetWidthInches,
  ]);

  const persistTemplateRegionSelection = async (nextRect: RoiRect) => {
    if (typeof window === "undefined") return;
    if (!selectedImagePreviewUrl || !selectedImageName) return;

    try {
      const templateRegionDataUrl = await createTemplateRegionDataUrl(selectedImagePreviewUrl, nextRect);
      sessionStorage.setItem(TEMPLATE_REGION_DATA_URL_KEY, templateRegionDataUrl);
      sessionStorage.setItem(TEMPLATE_REGION_IMAGE_NAME_KEY, selectedImageName);
      sessionStorage.setItem(TEMPLATE_REGION_RECT_KEY, JSON.stringify(nextRect));
    } catch {
      clearTemplateRegionCache();
    }
  };

  const clearRoiSelection = () => {
    roiRectRef.current = null;
    setRoiRect(null);
    setFocalScalePxIn(0);
    clearTemplateRegionCache();
  };

  const startRoiSelection = (clientX: number, clientY: number) => {
    const container = roiContainerRef.current;
    if (!container) return;

    const bounds = container.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;

    const x = Math.min(Math.max(clientX - bounds.left, 0), bounds.width);
    const y = Math.min(Math.max(clientY - bounds.top, 0), bounds.height);

    roiStartRef.current = { x, y };
    setIsSelectingRoi(true);
    const nextRect = { x: x / bounds.width, y: y / bounds.height, width: 0, height: 0 };
    roiRectRef.current = nextRect;
    setRoiRect(nextRect);
  };

  const updateRoiSelection = (clientX: number, clientY: number) => {
    const container = roiContainerRef.current;
    const start = roiStartRef.current;
    if (!container || !start || !isSelectingRoi) return;

    const bounds = container.getBoundingClientRect();
    const currentX = Math.min(Math.max(clientX - bounds.left, 0), bounds.width);
    const currentY = Math.min(Math.max(clientY - bounds.top, 0), bounds.height);

    const left = Math.min(start.x, currentX);
    const top = Math.min(start.y, currentY);
    const width = Math.abs(currentX - start.x);
    const height = Math.abs(currentY - start.y);
    const nextRect = {
      x: left / bounds.width,
      y: top / bounds.height,
      width: width / bounds.width,
      height: height / bounds.height,
    };
    roiRectRef.current = nextRect;
    setRoiRect(nextRect);
  };

  const endRoiSelection = () => {
    setIsSelectingRoi(false);
    roiStartRef.current = null;
    const current = roiRectRef.current;
    if (!current || current.width < 0.01 || current.height < 0.01) {
      clearRoiSelection();
      return;
    }
    setRoiRect(current);
    void persistTemplateRegionSelection(current);
  };

  const stopScan = () => {
    stopRequestedRef.current = true;
    stopAnalyzedPlayback();
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (videoRef.current) videoRef.current.pause();
    if (scanVideoRef.current) {
      scanVideoRef.current.pause();
      scanVideoRef.current = null;
    }
    if (howlRef.current) {
      howlRef.current.stop();
    }
    if (playPauseTimeoutRef.current) {
      window.clearTimeout(playPauseTimeoutRef.current);
      playPauseTimeoutRef.current = null;
    }
    const overlayCanvas = overlayCanvasRef.current;
    if (overlayCanvas) {
      const overlayCtx = overlayCanvas.getContext("2d");
      overlayCtx?.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    const processedContourCanvas = processedContourCanvasRef.current;
    if (processedContourCanvas) {
      const restored = restorePersistedContourSnapshot();
      if (!restored) {
        const processedCtx = processedContourCanvas.getContext("2d");
        processedCtx?.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
      }
    }
    resetShotFlowState();
    shotMarkersRef.current = [];
    shotSequenceRef.current = 0;
    liveShotClusteringRef.current = {
      selectedK: 0,
      finalK: 0,
      closeMergeCount: 0,
      objectiveScore: 0,
      shotClusterById: {},
      clusters: [],
    };
    liveClusterColorByIdRef.current = {};
    liveClusterShotCountRef.current = 0;
    setIsScanning(false);
    setDetectionEnabled(false);
    setDetectionConfidence(null);
    lastUiDetectionEnabledRef.current = false;
    lastUiDetectionConfidenceRef.current = -1;
    setScanStatus("Scan stopped");
  };

  const startScan = async () => {
    if (!opencvReady || !window.cv || !howlerReady || !window.Howl) {
      setScanStatus("OpenCV.js or Howler.js is not ready yet.");
      return;
    }

    if (captureMode !== "upload") {
      setScanStatus("Video scanning is implemented for uploaded video in this step.");
      return;
    }

    const previewVideoEl = videoRef.current;
    const canvasEl = processingCanvasRef.current;
    const overlayEl = overlayCanvasRef.current;

    if (!previewVideoEl || !canvasEl || !overlayEl || !selectedVideoFile || !selectedVideoPreviewUrl) {
      setScanStatus("Upload a reference video first.");
      return;
    }

    if (!selectedImagePreviewUrl) {
      setScanStatus("Capture a reference frame from the video, then draw the target geometry.");
      return;
    }

    const cv = window.cv;
    if (!cv) {
      setScanStatus("OpenCV runtime is unavailable.");
      return;
    }

    if (scanTaskActiveRef.current) {
      restartScanRequestedRef.current = true;
      setScanStatus("Scan rerun queued...");
      return;
    }
    scanTaskActiveRef.current = true;

    try {
      stopAnalyzedPlayback();
      const activeTemplateCanvas = document.createElement("canvas");
      let usedStoredTemplateRegion = false;

      if (typeof window !== "undefined") {
        const cachedTemplateDataUrl = sessionStorage.getItem(TEMPLATE_REGION_DATA_URL_KEY);
        const cachedTemplateImageName = sessionStorage.getItem(TEMPLATE_REGION_IMAGE_NAME_KEY);
        if (
          cachedTemplateDataUrl &&
          cachedTemplateImageName &&
          selectedImageName &&
          cachedTemplateImageName === selectedImageName
        ) {
          try {
            const cachedTemplateImage = await loadImageFromUrl(cachedTemplateDataUrl);
            if (cachedTemplateImage.naturalWidth > 0 && cachedTemplateImage.naturalHeight > 0) {
              activeTemplateCanvas.width = cachedTemplateImage.naturalWidth;
              activeTemplateCanvas.height = cachedTemplateImage.naturalHeight;
              const cachedCtx = activeTemplateCanvas.getContext("2d");
              if (!cachedCtx) throw new Error("Failed to prepare stored template canvas.");
              cachedCtx.drawImage(cachedTemplateImage, 0, 0);
              usedStoredTemplateRegion = true;
            }
          } catch {
            clearTemplateRegionCache();
          }
        }
      }

      if (!usedStoredTemplateRegion) {
        const templateImage = await loadImageFromUrl(selectedImagePreviewUrl);
        if (templateImage.naturalWidth === 0 || templateImage.naturalHeight === 0) {
          throw new Error("Reference image did not decode correctly.");
        }

        const templateSourceCanvas = imageToCanvas(templateImage);
        const activeRoi = roiRectRef.current ?? roiRect;

        if (activeRoi && activeRoi.width > 0 && activeRoi.height > 0) {
          const sx = Math.max(0, Math.floor(activeRoi.x * templateSourceCanvas.width));
          const sy = Math.max(0, Math.floor(activeRoi.y * templateSourceCanvas.height));
          const sw = Math.max(1, Math.floor(activeRoi.width * templateSourceCanvas.width));
          const sh = Math.max(1, Math.floor(activeRoi.height * templateSourceCanvas.height));

          activeTemplateCanvas.width = sw;
          activeTemplateCanvas.height = sh;
          const roiCtx = activeTemplateCanvas.getContext("2d");
          if (!roiCtx) throw new Error("Failed to prepare selected template area.");
          roiCtx.drawImage(templateSourceCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
        } else {
          activeTemplateCanvas.width = templateSourceCanvas.width;
          activeTemplateCanvas.height = templateSourceCanvas.height;
          const fullCtx = activeTemplateCanvas.getContext("2d");
          if (!fullCtx) throw new Error("Failed to prepare template canvas.");
          fullCtx.drawImage(templateSourceCanvas, 0, 0);
        }
      }

      previewVideoEl.pause();

      const videoEl = document.createElement("video");
      videoEl.src = selectedVideoPreviewUrl;
      videoEl.preload = "auto";
      videoEl.muted = true;
      videoEl.playsInline = true;
      scanVideoRef.current = videoEl;

      if (videoEl.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise<void>((resolve, reject) => {
          const onLoadedMetadata = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error("Unable to load video metadata for scanning."));
          };
          const cleanup = () => {
            videoEl.removeEventListener("loadedmetadata", onLoadedMetadata);
            videoEl.removeEventListener("error", onError);
          };

          videoEl.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
          videoEl.addEventListener("error", onError, { once: true });
        });
      }

      const templateMat = cv.imread(activeTemplateCanvas);
      let templateGray = new cv.Mat();
      cv.cvtColor(templateMat, templateGray, cv.COLOR_RGBA2GRAY);
      templateMat.delete();

      let trackingHist: CvMat | null = null;
      if (trackingMode !== "template") {
        const templateImages = new cv.MatVector();
        templateImages.push_back(templateGray);
        const hist = new cv.Mat();
        const emptyMask = new cv.Mat();
        cv.calcHist(templateImages, [0], emptyMask, hist, [64], [0, 256], false);
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX);
        emptyMask.delete();
        templateImages.delete();
        trackingHist = hist;
      }

      videoEl.pause();
      stopRequestedRef.current = false;
      frameIndexRef.current = 0;
      lastLoggedAtMsRef.current = 0;
      lastDetectionSampleAtMsRef.current = 0;
      spikeWindowsRef.current = [];
      audioReadyRef.current = false;
      detectionTimelineRef.current = [];
      contourWindowTimelineRef.current = [];
      shotMarkersRef.current = [];
      shotSequenceRef.current = 0;
      liveShotClusteringRef.current = {
        selectedK: 0,
        finalK: 0,
        closeMergeCount: 0,
        objectiveScore: 0,
        shotClusterById: {},
        clusters: [],
      };
      liveClusterColorByIdRef.current = {};
      liveClusterShotCountRef.current = 0;
      audioRmsTimelineRef.current = [];
      audioMeanDbfsRef.current = -120;
      audioThresholdDbfsRef.current = -120;
      resetShotFlowState();
      setLogEntries([]);
      setShotLogEntries([]);
      setLastDetection(null);
      setLastShot(null);
      setDetectionEnabled(false);
      setDetectionConfidence(0);
      lastUiDetectionEnabledRef.current = false;
      lastUiDetectionConfidenceRef.current = 0;
      setSpikeMetadata([]);
      setAudioSprites({});
      setSpritesReady(false);
      setIsScanning(true);
      setScanStatus("Starting concurrent audio + OpenCV analysis...");

      const durationSec = Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : 0;
      if (durationSec <= 0) {
        templateGray.delete();
        if (trackingHist) trackingHist.delete();
        setIsScanning(false);
        setScanStatus("Video metadata unavailable. Press play once, then retry.");
        return;
      }

      const HowlCtor = window.Howl;
      const audioTask = (async () => {
        const audioContext = new AudioContext();
        const fileBuffer = await selectedVideoFile.arrayBuffer();
        const decodedAudio = await audioContext.decodeAudioData(fileBuffer.slice(0));
        await audioContext.close();

        const { spikes, spriteMap, rmsTimeline, meanDbfs, thresholdDbfs } = detectAudioSpikes(
          decodedAudio,
          durationSec,
          tweakSettings,
        );
        setSpikeMetadata(spikes);
        setAudioSprites(spriteMap);
        spikeEventsRef.current = spikes;
        audioRmsTimelineRef.current = rmsTimeline;
        audioMeanDbfsRef.current = meanDbfs;
        audioThresholdDbfsRef.current = thresholdDbfs;

        if (howlRef.current) {
          howlRef.current.unload();
          howlRef.current = null;
        }

        if (spikes.length === 0) {
          spikeWindowsRef.current = [{ start: 0, end: durationSec }];
          audioReadyRef.current = true;
          setScanStatus("No audio spikes detected; scanning full video range.");
          return;
        }

        const howl = await new Promise<HowlInstance>((resolve, reject) => {
          const instance = new HowlCtor({
            src: [selectedVideoPreviewUrl],
            html5: true,
            preload: true,
            sprite: spriteMap,
            onload: () => resolve(instance),
            onloaderror: (_id, error) =>
              reject(new Error(`Howler failed to load audio for sprites: ${String(error)}`)),
          });
        });

        howlRef.current = howl;
        spikeWindowsRef.current = mergeWindows(
          spikes.map((spike) => ({ start: spike.windowStartSec, end: spike.windowEndSec })),
        );
        audioReadyRef.current = true;
        setSpritesReady(true);
        setScanStatus("Audio spikes ready. Running detection in +/-1s windows with intensive analysis near each spike.");
      })().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        audioReadyRef.current = true;
        spikeWindowsRef.current = [{ start: 0, end: durationSec }];
        setSpritesReady(false);
        setScanStatus(`Audio spike analysis failed (${message}). Scanning full video range.`);
      });

      await audioTask;

      const videoWidth = videoEl.videoWidth;
      const videoHeight = videoEl.videoHeight;
      if (videoWidth > 0 && videoHeight > 0 && (templateGray.cols > videoWidth || templateGray.rows > videoHeight)) {
        const ratio = Math.min(videoWidth / templateGray.cols, videoHeight / templateGray.rows) * tweakSettings.templateFitMarginRatio;
        const resized = new cv.Mat();
        const nextWidth = Math.max(1, Math.floor(templateGray.cols * ratio));
        const nextHeight = Math.max(1, Math.floor(templateGray.rows * ratio));
        cv.resize(templateGray, resized, new cv.Size(nextWidth, nextHeight), 0, 0, cv.INTER_AREA);
        templateGray.delete();
        templateGray = resized;
        setScanStatus("Scanning... Template auto-resized to fit video frame.");
      }

      let trackingWindow: { x: number; y: number; width: number; height: number } | null = null;
      const termCriteria = new cv.TermCriteria(
        cv.TermCriteria_EPS + cv.TermCriteria_COUNT,
        Math.max(1, Math.round(tweakSettings.templateTrackerTermCriteriaMaxCount)),
        tweakSettings.templateTrackerTermCriteriaEpsilon,
      );
      let trackingFallbackNoted = false;
      const centerProbeCanvas = document.createElement("canvas");
      const changeProbeCanvas = document.createElement("canvas");
      changeProbeCanvas.width = Math.max(16, Math.round(tweakSettings.spikeStandardProbeSize));
      changeProbeCanvas.height = Math.max(16, Math.round(tweakSettings.spikeStandardProbeSize));
      const templateBaselineCanvas = document.createElement("canvas");
      const templateBaselineContext = templateBaselineCanvas.getContext("2d");
      let templateBaselinePatch: Uint8ClampedArray | null = null;
      let contourMaskHistory: Uint8Array[] = [];
      const syncTemplateBaselinePatch = () => {
        if (!templateBaselineContext) return false;
        if (
          templateBaselineCanvas.width !== changeProbeCanvas.width ||
          templateBaselineCanvas.height !== changeProbeCanvas.height
        ) {
          templateBaselineCanvas.width = changeProbeCanvas.width;
          templateBaselineCanvas.height = changeProbeCanvas.height;
        }
        templateBaselineContext.clearRect(0, 0, templateBaselineCanvas.width, templateBaselineCanvas.height);
        templateBaselineContext.drawImage(
          activeTemplateCanvas,
          0,
          0,
          activeTemplateCanvas.width,
          activeTemplateCanvas.height,
          0,
          0,
          templateBaselineCanvas.width,
          templateBaselineCanvas.height,
        );
        const baselineImage = templateBaselineContext.getImageData(
          0,
          0,
          templateBaselineCanvas.width,
          templateBaselineCanvas.height,
        );
        templateBaselinePatch = new Uint8ClampedArray(baselineImage.data);
        contourMaskHistory = [];
        return true;
      };
      syncTemplateBaselinePatch();
      try {
        await videoEl.play();
      } catch {
        templateGray.delete();
        if (trackingHist) trackingHist.delete();
        setIsScanning(false);
        setScanStatus("Unable to play video for concurrent scan.");
        return;
      }

      await new Promise<void>((resolve) => {
        const processFrame = () => {
          if (stopRequestedRef.current || videoEl.ended) {
            resolve();
            return;
          }

          const width = videoEl.videoWidth;
          const height = videoEl.videoHeight;
          if (width <= 0 || height <= 0) {
            animationFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }

          canvasEl.width = width;
          canvasEl.height = height;
          overlayEl.width = videoEl.clientWidth;
          overlayEl.height = videoEl.clientHeight;
          const context = canvasEl.getContext("2d");
          const overlayContext = overlayEl.getContext("2d");
          if (!context) {
            setScanStatus("Canvas context unavailable");
            resolve();
            return;
          }
          if (!overlayContext) {
            setScanStatus("Overlay canvas context unavailable");
            resolve();
            return;
          }

          context.drawImage(videoEl, 0, 0, width, height);
          overlayContext.clearRect(0, 0, overlayEl.width, overlayEl.height);

          const currentSec = videoEl.currentTime;
          const windows = spikeWindowsRef.current;

          const spikeTimeline = spikeEventsRef.current;
          const nearestSpikeInfo = getNearestSpikeAtTime(currentSec, spikeTimeline);
          const desiredProbeSize = Math.max(16, Math.round(tweakSettings.spikeStandardProbeSize));
          const probeSizeChanged =
            changeProbeCanvas.width !== desiredProbeSize || changeProbeCanvas.height !== desiredProbeSize;
          if (probeSizeChanged) {
            changeProbeCanvas.width = desiredProbeSize;
            changeProbeCanvas.height = desiredProbeSize;
            syncTemplateBaselinePatch();
          }

          const frameMat = cv.imread(canvasEl);
          const frameGray = new cv.Mat();
          cv.cvtColor(frameMat, frameGray, cv.COLOR_RGBA2GRAY);

          if (templateGray.rows > frameGray.rows || templateGray.cols > frameGray.cols) {
            const ratio = Math.min(frameGray.cols / templateGray.cols, frameGray.rows / templateGray.rows) * tweakSettings.templateFitMarginRatio;
            if (ratio > 0 && ratio < 1) {
              const resized = new cv.Mat();
              const nextWidth = Math.max(1, Math.floor(templateGray.cols * ratio));
              const nextHeight = Math.max(1, Math.floor(templateGray.rows * ratio));
              cv.resize(templateGray, resized, new cv.Size(nextWidth, nextHeight), 0, 0, cv.INTER_AREA);
              templateGray.delete();
              templateGray = resized;
            } else {
              frameMat.delete();
              frameGray.delete();
              setScanStatus("Unable to fit selected template into video frames.");
              resolve();
              return;
            }
          }

          let bestScore = -1;
          let bestLoc = { x: 0, y: 0 };
          const centerCoverage = tweakSettings.centerProbeCoverageRatio;
          const centerWidth = Math.max(templateGray.cols, Math.floor(width * centerCoverage));
          const centerHeight = Math.max(templateGray.rows, Math.floor(height * centerCoverage));
          const centerX = Math.max(0, Math.floor((width - centerWidth) / 2));
          const centerY = Math.max(0, Math.floor((height - centerHeight) / 2));
          const canUseCenterProbe = centerWidth >= templateGray.cols && centerHeight >= templateGray.rows;

          if (canUseCenterProbe) {
            centerProbeCanvas.width = centerWidth;
            centerProbeCanvas.height = centerHeight;
            const centerContext = centerProbeCanvas.getContext("2d");
            if (centerContext) {
              centerContext.drawImage(canvasEl, centerX, centerY, centerWidth, centerHeight, 0, 0, centerWidth, centerHeight);
              const centerFrameMat = cv.imread(centerProbeCanvas);
              const centerFrameGray = new cv.Mat();
              cv.cvtColor(centerFrameMat, centerFrameGray, cv.COLOR_RGBA2GRAY);
              const centerResult = new cv.Mat();
              cv.matchTemplate(centerFrameGray, templateGray, centerResult, cv.TM_CCOEFF_NORMED);
              const centerMinMax = cv.minMaxLoc(centerResult);
              centerResult.delete();
              centerFrameGray.delete();
              centerFrameMat.delete();

              bestScore = centerMinMax.maxVal;
              bestLoc = {
                x: centerX + centerMinMax.maxLoc.x,
                y: centerY + centerMinMax.maxLoc.y,
              };
            }
          }

          const centerAcceptThresholdPct = Math.max(
            tweakSettings.centerAcceptThresholdMinPct,
            matchThreshold * tweakSettings.centerAcceptThresholdScale,
          );
          if (bestScore * 100 < centerAcceptThresholdPct) {
            const result = new cv.Mat();
            cv.matchTemplate(frameGray, templateGray, result, cv.TM_CCOEFF_NORMED);
            const minMax = cv.minMaxLoc(result);
            result.delete();
            bestScore = minMax.maxVal;
            bestLoc = minMax.maxLoc;
          }

          frameIndexRef.current += 1;
          const rawConfidence = Number(bestScore);
          const safeConfidence = Number.isFinite(rawConfidence) ? rawConfidence : 0;
          const confidencePct = Math.max(0, Math.min(100, safeConfidence * 100));
          const isAboveThreshold = confidencePct >= matchThreshold;
          if (
            isAboveThreshold !== lastUiDetectionEnabledRef.current ||
            Math.abs(confidencePct - lastUiDetectionConfidenceRef.current) >= 0.1
          ) {
            lastUiDetectionEnabledRef.current = isAboveThreshold;
            lastUiDetectionConfidenceRef.current = confidencePct;
            setDetectionEnabled(isAboveThreshold);
            setDetectionConfidence(confidencePct);
          }

          if (trackingMode !== "template" && !trackingWindow && isAboveThreshold) {
            trackingWindow = clampRectToFrame(
              bestLoc.x,
              bestLoc.y,
              templateGray.cols,
              templateGray.rows,
              frameGray.cols,
              frameGray.rows,
            );
          }

          const templateRect = {
            x: bestLoc.x,
            y: bestLoc.y,
            width: templateGray.cols,
            height: templateGray.rows,
          };
          let drawRect = templateRect;

          if (trackingMode !== "template" && trackingWindow && trackingHist) {
            const frameImages = new cv.MatVector();
            frameImages.push_back(frameGray);
            const backProj = new cv.Mat();
            cv.calcBackProject(frameImages, [0], trackingHist, backProj, [0, 256], 1);
            frameImages.delete();

            const hasCamShift = typeof cv.CamShift === "function";
            const hasMeanShift = typeof cv.meanShift === "function";

            if (trackingMode === "camshift" && hasCamShift) {
              cv.CamShift!(backProj, trackingWindow, termCriteria);
            } else if (hasMeanShift) {
              cv.meanShift?.(backProj, trackingWindow, termCriteria);
            } else if (!trackingFallbackNoted) {
              trackingFallbackNoted = true;
              setScanStatus("meanShift/CamShift unavailable in this OpenCV.js build. Using template only.");
            }

            backProj.delete();

            const bounded = clampRectToFrame(
              trackingWindow.x,
              trackingWindow.y,
              trackingWindow.width,
              trackingWindow.height,
              frameGray.cols,
              frameGray.rows,
            );
            trackingWindow = bounded;

            const trackerWidthRatio = bounded.width / Math.max(1, templateGray.cols);
            const trackerHeightRatio = bounded.height / Math.max(1, templateGray.rows);
            const templateCenterX = templateRect.x + templateRect.width / 2;
            const templateCenterY = templateRect.y + templateRect.height / 2;
            const trackerCenterX = bounded.x + bounded.width / 2;
            const trackerCenterY = bounded.y + bounded.height / 2;
            const centerDistance = Math.hypot(templateCenterX - trackerCenterX, templateCenterY - trackerCenterY);
            const maxReasonableDistance = Math.max(frameGray.cols, frameGray.rows) * tweakSettings.trackerMaxCenterDistanceRatio;
            const trackerLooksSane =
              trackerWidthRatio >= tweakSettings.trackerWidthRatioMin &&
              trackerWidthRatio <= tweakSettings.trackerWidthRatioMax &&
              trackerHeightRatio >= tweakSettings.trackerHeightRatioMin &&
              trackerHeightRatio <= tweakSettings.trackerHeightRatioMax &&
              centerDistance <= maxReasonableDistance;

            // Prefer template match location; only use tracker when it remains close and size-consistent.
            drawRect = trackerLooksSane ? bounded : templateRect;
            if (!trackerLooksSane) {
              trackingWindow = clampRectToFrame(
                templateRect.x,
                templateRect.y,
                templateRect.width,
                templateRect.height,
                frameGray.cols,
                frameGray.rows,
              );
            }
          }
          drawRect = clampRectToFrame(
            drawRect.x,
            drawRect.y,
            drawRect.width,
            drawRect.height,
            frameGray.cols,
            frameGray.rows,
          );
          const estimatedDistanceInches = estimateDistanceInchesFromDetection(
            targetWidthInches,
            targetHeightInches,
            pixelsPerInch,
            drawRect.width,
            drawRect.height,
            focalScalePxIn,
            manualDistanceOverrideInches,
          );
          const nearestSpike = nearestSpikeInfo.spike;
          const nearestSpikeDeltaSec = Number.isFinite(nearestSpikeInfo.deltaSec) ? nearestSpikeInfo.deltaSec : null;
          const spikeFocusedWindow =
            nearestSpikeDeltaSec !== null && nearestSpikeDeltaSec <= Math.max(0.05, tweakSettings.spikeIntensiveFocusSec);
          let shotDetectedThisFrame: ShotLogEntry | null = null;
          let contourWindowSnapshotForFrame: ContourWindowFrameSnapshot | null = null;
          const changeProbeContext = changeProbeCanvas.getContext("2d");
          const processedContourCanvas = processedContourCanvasRef.current;
          const processedContourContext = processedContourCanvas?.getContext("2d") ?? null;
          if (changeProbeContext) {
            try {
              changeProbeContext.drawImage(
                canvasEl,
                drawRect.x,
                drawRect.y,
                drawRect.width,
                drawRect.height,
                0,
                0,
                changeProbeCanvas.width,
                changeProbeCanvas.height,
              );
              const patchWidth = changeProbeCanvas.width;
              const patchHeight = changeProbeCanvas.height;
              const probeImageData = changeProbeContext.getImageData(0, 0, patchWidth, patchHeight);
              const currentPatchRgba = probeImageData.data;
              if (processedContourCanvas) {
                processedContourCanvas.width = patchWidth;
                processedContourCanvas.height = patchHeight;
              }
              contourWindowSnapshotForFrame = {
                frame: frameIndexRef.current,
                videoTimeSec: videoEl.currentTime,
                patchWidthPx: patchWidth,
                patchHeightPx: patchHeight,
                changedPixels: 0,
                rawMaskRuns: [],
                temporalMaskRuns: [],
                effectiveMaskRuns: [],
                regions: [],
                actionStatuses: [],
              };

              if (
                (!templateBaselinePatch || templateBaselinePatch.length !== currentPatchRgba.length) &&
                !syncTemplateBaselinePatch()
              ) {
                resetShotFlowState();
                const contourActionStatuses: OpenCvActionStatus[] = [
                  { label: "Baseline", state: "warn", detail: "missing reference patch" },
                  { label: "Background Subtract", state: "idle", detail: "waiting for baseline" },
                  { label: "Temporal Persistence", state: "idle", detail: "waiting for baseline" },
                  { label: "Find Contours", state: "idle", detail: "waiting for baseline" },
                ];
                if (contourWindowSnapshotForFrame) {
                  contourWindowSnapshotForFrame.actionStatuses = contourActionStatuses;
                }
                if (processedContourContext) {
                  drawProcessedContourView(
                    processedContourContext,
                    currentPatchRgba,
                    null,
                    patchWidth,
                    patchHeight,
                    {
                      rawMask: null,
                      temporalMask: null,
                      effectiveMask: null,
                      positiveDiffMap: null,
                    },
                    [],
                    contourActionStatuses,
                  );
                }
              } else {
                const activeBaselinePatch =
                  templateBaselinePatch && templateBaselinePatch.length === currentPatchRgba.length
                    ? templateBaselinePatch
                    : null;
                if (!activeBaselinePatch) {
                  contourMaskHistory = [];
                  resetShotFlowState();
                  const contourActionStatuses: OpenCvActionStatus[] = [
                    { label: "Baseline", state: "warn", detail: "reference patch mismatch" },
                    { label: "Background Subtract", state: "idle", detail: "baseline not usable" },
                    { label: "Temporal Persistence", state: "idle", detail: "baseline not usable" },
                    { label: "Find Contours", state: "idle", detail: "baseline not usable" },
                  ];
                  if (contourWindowSnapshotForFrame) {
                    contourWindowSnapshotForFrame.actionStatuses = contourActionStatuses;
                  }
                  if (processedContourContext) {
                    drawProcessedContourView(
                      processedContourContext,
                      currentPatchRgba,
                      null,
                      patchWidth,
                      patchHeight,
                      {
                        rawMask: null,
                        temporalMask: null,
                        effectiveMask: null,
                        positiveDiffMap: null,
                      },
                      [],
                      contourActionStatuses,
                    );
                  }
                } else {
                  const shotCooldownMs = Math.max(0, Math.round(tweakSettings.shotCooldownMs));
                  const nowMs = performance.now();

                  const pending = pendingShotCandidateRef.current;
                  if (pending && nowMs - pending.lastSeenAtMs > SHOT_PERSISTENCE_MAX_GAP_MS) {
                    pendingShotCandidateRef.current = null;
                  }

                  const openedChangeMask = buildSimpleBackgroundSubtractMask(
                    currentPatchRgba,
                    activeBaselinePatch,
                    patchWidth,
                    patchHeight,
                  );
                  const openedChangedPixels = countMaskPixels(openedChangeMask);
                  const requiredPersistenceVotes = Math.max(
                    2,
                    Math.min(
                      SIMPLE_CONTOUR_MASK_HISTORY_MAX,
                      Math.round(Math.max(2, tweakSettings.spikeIntensiveHistoryFrames * 0.5)),
                    ),
                  );
                  const hasTemporalHistory = contourMaskHistory.length > 0;
                  const persistentMask = buildTemporalPersistenceMask(
                    openedChangeMask,
                    contourMaskHistory,
                    requiredPersistenceVotes,
                  );
                  const persistentChangedPixels = countMaskPixels(persistentMask);
                  const effectiveMask = hasTemporalHistory ? persistentMask : openedChangeMask;
                  const effectiveChangedPixels = hasTemporalHistory ? persistentChangedPixels : openedChangedPixels;
                  const effectiveChangedRatioPct = (effectiveChangedPixels / (patchWidth * patchHeight)) * 100;
                  const positiveDiffMap = buildPositiveBackgroundDiffMap(
                    currentPatchRgba,
                    activeBaselinePatch,
                    patchWidth,
                    patchHeight,
                  );
                  const minBlobPixels = SIMPLE_CONTOUR_MIN_AREA;
                  const maxBlobPixels = Math.min(
                    SIMPLE_CONTOUR_MAX_AREA,
                    Math.max(SIMPLE_CONTOUR_MIN_AREA + 1, Math.floor(patchWidth * patchHeight * 0.25)),
                  );
                  const contourSkippedByGlobalChange = effectiveChangedRatioPct > MAX_GLOBAL_CHANGE_RATIO_PCT;
                  const changedRegions =
                    effectiveChangedPixels > 0 && !contourSkippedByGlobalChange
                      ? findChangedRegionsByContours(
                          cv,
                          effectiveMask,
                          patchWidth,
                          patchHeight,
                          minBlobPixels,
                          maxBlobPixels,
                        )
                      : [];
                  const temporalFramesUsed = contourMaskHistory.length + 1;
                  const contourActionStatuses: OpenCvActionStatus[] = [
                    { label: "Baseline", state: "ok", detail: "reference patch ready" },
                    { label: "Background Subtract", state: "ok", detail: `${openedChangedPixels}px after opening` },
                    {
                      label: "Temporal Persistence",
                      state: hasTemporalHistory ? "ok" : "idle",
                      detail: hasTemporalHistory
                        ? `${persistentChangedPixels}px | votes ${requiredPersistenceVotes}/${temporalFramesUsed}`
                        : `warmup ${temporalFramesUsed}/${requiredPersistenceVotes} votes`,
                    },
                    {
                      label: "Find Contours",
                      state:
                        effectiveChangedPixels === 0
                          ? "idle"
                          : contourSkippedByGlobalChange
                            ? "skip"
                            : "ok",
                      detail:
                        effectiveChangedPixels === 0
                          ? "no changed pixels"
                          : contourSkippedByGlobalChange
                            ? `skipped ${effectiveChangedRatioPct.toFixed(1)}% > ${MAX_GLOBAL_CHANGE_RATIO_PCT.toFixed(1)}%`
                            : `${changedRegions.length} regions | bounds ${minBlobPixels}-${maxBlobPixels}px`,
                    },
                    {
                      label: "Hybrid Mask",
                      state: effectiveChangedPixels > 0 ? "ok" : "idle",
                      detail: `${hasTemporalHistory ? "temporal" : "raw"} output ${effectiveChangedPixels}px`,
                    },
                  ];
                  const contourRegionsForPlayback = changedRegions.slice(0, 10).map((region) => ({
                    pixelCount: region.pixelCount,
                    centerX: region.centerX,
                    centerY: region.centerY,
                    minX: region.minX,
                    minY: region.minY,
                    maxX: region.maxX,
                    maxY: region.maxY,
                  }));
                  contourWindowSnapshotForFrame = {
                    frame: frameIndexRef.current,
                    videoTimeSec: videoEl.currentTime,
                    patchWidthPx: patchWidth,
                    patchHeightPx: patchHeight,
                    changedPixels: effectiveChangedPixels,
                    rawMaskRuns: encodeBinaryMaskRuns(openedChangeMask),
                    temporalMaskRuns: hasTemporalHistory ? encodeBinaryMaskRuns(persistentMask) : [],
                    effectiveMaskRuns: encodeBinaryMaskRuns(effectiveMask),
                    regions: contourRegionsForPlayback,
                    actionStatuses: contourActionStatuses,
                  };

                  if (processedContourContext) {
                    drawProcessedContourView(
                      processedContourContext,
                      currentPatchRgba,
                      activeBaselinePatch,
                      patchWidth,
                      patchHeight,
                      {
                        rawMask: openedChangeMask,
                        temporalMask: hasTemporalHistory ? persistentMask : null,
                        effectiveMask,
                        positiveDiffMap,
                      },
                      changedRegions,
                      contourActionStatuses,
                    );
                  }

                  const activeWindowIndexForFrame = windowIndexAtTime(currentSec, windows);
                  const activeWindowForFrame =
                    activeWindowIndexForFrame >= 0 && activeWindowIndexForFrame < windows.length
                      ? windows[activeWindowIndexForFrame]
                      : null;

                  let bestCandidate: { entry: ShotLogEntry; score: number } | null = null;
                  for (const region of changedRegions.slice(0, 10)) {
                    if (!activeWindowForFrame) continue;
                    const temporalStats = summarizeRegionTemporalSupport(
                      region,
                      openedChangeMask,
                      effectiveMask,
                      positiveDiffMap,
                      patchWidth,
                      patchHeight,
                    );
                    const minSupportRatio = Math.max(0.2, Math.min(0.95, tweakSettings.temporalDarkVoteRatio));
                    if (temporalStats.rawPixels > 0 && temporalStats.supportRatio < minSupportRatio) continue;
                    const minMeanPositiveDiff = Math.max(6, Math.min(80, tweakSettings.transitionLuminanceJumpMin));
                    if (temporalStats.meanDiff < minMeanPositiveDiff) continue;

                    const spanX = Math.max(1, region.maxX - region.minX + 1);
                    const spanY = Math.max(1, region.maxY - region.minY + 1);
                    const blobAreaPx = Math.max(1, spanX * spanY);
                    const blobFillPct = (region.pixelCount / blobAreaPx) * 100;
                    const blobAspectRatio = spanAspectRatio(spanX, spanY);
                    const maxBlobAspectRatio = Math.max(3, Math.max(8.8, tweakSettings.colorAspectRatioMax) + 1.2);
                    if (blobFillPct < 5 || blobAspectRatio > maxBlobAspectRatio) continue;

                    const { diameterPx, diameterInches } = estimatePatchBlobDiameter(
                      region.pixelCount,
                      spanX,
                      spanY,
                      patchWidth,
                      patchHeight,
                      drawRect.width,
                      drawRect.height,
                      pixelsPerInch,
                    );
                    if (!isSubTwoInchShot(diameterPx, diameterInches, drawRect.width, drawRect.height, tweakSettings)) continue;

                    const transitionRatioPct = (region.pixelCount / (patchWidth * patchHeight)) * 100;
                    const normalizedTransitionRatio = clamp01(transitionRatioPct / 4.5);
                    const normalizedFill = clamp01(blobFillPct / 100);
                    const normalizedCompactness = clamp01(1 / Math.max(1, blobAspectRatio));
                    const normalizedTemporalSupport = clamp01(temporalStats.supportRatio);
                    const normalizedMeanDiff = clamp01(temporalStats.meanDiff / Math.max(1, tweakSettings.transitionLuminanceJumpMin * 2));
                    const candidateScore =
                      normalizedTransitionRatio * 0.35 +
                      normalizedFill * 0.2 +
                      normalizedCompactness * 0.1 +
                      normalizedTemporalSupport * 0.2 +
                      normalizedMeanDiff * 0.15;

                    const previousShot = shotMarkersRef.current[shotMarkersRef.current.length - 1] ?? null;
                    const timeSincePreviousShotSec =
                      previousShot === null ? null : Math.max(0, videoEl.currentTime - previousShot.videoTimeSec);
                    const audioMetrics = computeShotAudioMetrics(
                      videoEl.currentTime,
                      candidateScore,
                      nearestSpikeDeltaSec,
                      nearestSpike?.strength ?? null,
                      audioRmsTimelineRef.current,
                      audioMeanDbfsRef.current,
                      tweakSettings,
                    );
                    const centerX = Math.round(drawRect.x + ((region.centerX + 0.5) / patchWidth) * drawRect.width);
                    const centerY = Math.round(drawRect.y + ((region.centerY + 0.5) / patchHeight) * drawRect.height);
                    const radius = Math.max(4, Math.round(diameterPx / 2));
                    const fastColorChangeTrigger = false;

                    const candidateShot: ShotLogEntry = {
                      id: "",
                      shotNumber: 0,
                      frame: frameIndexRef.current,
                      videoTimeSec: videoEl.currentTime,
                      timeSincePreviousShotSec,
                      windowStartSec: activeWindowForFrame.start,
                      windowEndSec: activeWindowForFrame.end,
                      centerX,
                      centerY,
                      radius,
                      changedPixels: region.pixelCount,
                      changeScore: candidateScore,
                      estimatedDiameterInches: diameterInches,
                      detectionMethod: "pixel_change",
                      trackingMode,
                      detectionEnabled: isAboveThreshold,
                      detectionConfidencePct: confidencePct,
                      detectionThresholdPct: matchThreshold,
                      spikeFocusedWindow,
                      nearestSpikeId: nearestSpike?.id ?? null,
                      nearestSpikeTimeSec: nearestSpike?.timeSec ?? null,
                      nearestSpikeDeltaSec,
                      nearestSpikeStrength: nearestSpike?.strength ?? null,
                      patchWidthPx: patchWidth,
                      patchHeightPx: patchHeight,
                      drawRectX: Math.round(drawRect.x),
                      drawRectY: Math.round(drawRect.y),
                      drawRectWidth: Math.round(drawRect.width),
                      drawRectHeight: Math.round(drawRect.height),
                      centerPatchX: region.centerX + 0.5,
                      centerPatchY: region.centerY + 0.5,
                      spanWidthPx: spanX,
                      spanHeightPx: spanY,
                      estimatedDiameterPx: diameterPx,
                      changedPixelRatioPct: transitionRatioPct,
                      brightPixelCount: temporalStats.persistentPixels,
                      transitionPixelCount: temporalStats.rawPixels,
                      transitionPurityPct: temporalStats.supportRatio * 100,
                      minBlobPixelsThreshold: minBlobPixels,
                      temporalHistoryFramesUsed: hasTemporalHistory ? contourMaskHistory.length + 1 : 1,
                      shotCooldownMs,
                      residualMotionPx: null,
                      trackedPointCount: null,
                      estimatedRapidChangeMs: null,
                      fastColorChangeTrigger,
                      audioDecibelDbfs: audioMetrics.audioDecibelDbfs,
                      audioDeltaFromMeanDb: audioMetrics.audioDeltaFromMeanDb,
                      audioPeakDbfs: audioMetrics.audioPeakDbfs,
                      audioCorrelationScorePct: audioMetrics.audioCorrelationScorePct,
                      nearestSpikeDecibelDbfs: audioMetrics.nearestSpikeDecibelDbfs,
                    };
                    if (!isLikelyGoodShot(candidateShot, tweakSettings)) continue;
                    if (!bestCandidate || candidateScore > bestCandidate.score) {
                      bestCandidate = { entry: candidateShot, score: candidateScore };
                    }
                  }

                if (bestCandidate) {
                  const activePending = pendingShotCandidateRef.current;
                  const matchDistancePx = Math.max(10, Math.min(64, bestCandidate.entry.radius * 2.2));
                  const extendsPending =
                    !!activePending &&
                    nowMs - activePending.lastSeenAtMs <= SHOT_PERSISTENCE_MAX_GAP_MS &&
                    Math.hypot(
                      bestCandidate.entry.centerX - activePending.centerX,
                      bestCandidate.entry.centerY - activePending.centerY,
                    ) <= matchDistancePx;

                  if (extendsPending && activePending) {
                    activePending.centerX = bestCandidate.entry.centerX;
                    activePending.centerY = bestCandidate.entry.centerY;
                    activePending.lastSeenAtMs = nowMs;
                    if (bestCandidate.score >= activePending.bestScore) {
                      activePending.bestScore = bestCandidate.score;
                      activePending.bestEntry = bestCandidate.entry;
                    }

                    const persistedMs = activePending.lastSeenAtMs - activePending.firstSeenAtMs;
                    const shouldConfirm = !activePending.confirmed && persistedMs >= SHOT_PERSISTENCE_MIN_MS;
                    if (shouldConfirm && nowMs - lastShotAtMsRef.current > shotCooldownMs) {
                      const nextShotNumber = shotSequenceRef.current + 1;
                      shotSequenceRef.current = nextShotNumber;
                      const acceptedShot: ShotLogEntry = {
                        ...activePending.bestEntry,
                        shotNumber: nextShotNumber,
                        id: `shot_${nextShotNumber}`,
                      };
                      activePending.confirmed = true;
                      lastShotAtMsRef.current = nowMs;
                      shotDetectedThisFrame = acceptedShot;
                      shotMarkersRef.current = [...shotMarkersRef.current, acceptedShot].slice(
                        -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
                      );
                    }
                  } else {
                    pendingShotCandidateRef.current = {
                      centerX: bestCandidate.entry.centerX,
                      centerY: bestCandidate.entry.centerY,
                      firstSeenAtMs: nowMs,
                      lastSeenAtMs: nowMs,
                      bestEntry: bestCandidate.entry,
                      bestScore: bestCandidate.score,
                      confirmed: false,
                    };
                  }
                }
                contourMaskHistory = [...contourMaskHistory, new Uint8Array(openedChangeMask)].slice(
                  -SIMPLE_CONTOUR_MASK_HISTORY_MAX,
                );
              }
              activeSpikeWindowIndexRef.current = -1;
            }
            } catch {
              resetShotFlowState();
              if (processedContourContext && processedContourCanvas) {
                const restored = restorePersistedContourSnapshot();
                if (!restored) {
                  processedContourContext.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
                }
              }
            }
          }
          const overlayLabel =
            `${trackingMode.toUpperCase()} ${confidencePct.toFixed(1)}%` +
            (estimatedDistanceInches === null ? "" : ` | ~${formatLinearFromInches(estimatedDistanceInches, 1)}`);
          const labelWidth = estimatedDistanceInches === null ? 165 : 250;

          const scaleX = overlayEl.width / width;
          const scaleY = overlayEl.height / height;
          const isBlinkOn = Math.floor(performance.now() / 500) % 2 === 0;
          const boundaryColor = isBlinkOn ? "#ffffff" : "#000000";
          overlayContext.strokeStyle = boundaryColor;
          overlayContext.lineWidth = 3;
          overlayContext.strokeRect(
            drawRect.x * scaleX,
            drawRect.y * scaleY,
            drawRect.width * scaleX,
            drawRect.height * scaleY,
          );
          context.strokeStyle = boundaryColor;
          context.lineWidth = 3;
          context.strokeRect(drawRect.x, drawRect.y, drawRect.width, drawRect.height);
          overlayContext.fillStyle = "rgba(15, 23, 42, 0.82)";
          overlayContext.fillRect(drawRect.x * scaleX, Math.max(drawRect.y * scaleY - 20, 0), labelWidth, 18);
          overlayContext.fillStyle = isAboveThreshold ? "#86efac" : "#fcd34d";
          overlayContext.font = "12px sans-serif";
          overlayContext.fillText(
            overlayLabel,
            drawRect.x * scaleX + 6,
            Math.max(drawRect.y * scaleY - 7, 12),
          );
          context.fillStyle = "rgba(0, 0, 0, 0.72)";
          context.fillRect(drawRect.x, Math.max(drawRect.y - 20, 0), labelWidth, 18);
          context.fillStyle = isBlinkOn ? "#ffffff" : "#d4d4d4";
          context.font = "12px sans-serif";
          context.fillText(
            overlayLabel,
            drawRect.x + 6,
            Math.max(drawRect.y - 7, 12),
          );

          // Draw one corner dot for each detected spike up to the current playback time.
          const seenSpikeCount = spikeEventsRef.current.filter((spike) => spike.timeSec <= currentSec).length;
          const maxVisibleDots = 20;
          const visibleDots = Math.min(seenSpikeCount, maxVisibleDots);
          const dotRadius = 4;
          const dotGap = 6;
          const dotStartX = overlayEl.width - 16;
          const dotStartY = 16;

          for (let i = 0; i < visibleDots; i += 1) {
            overlayContext.beginPath();
            overlayContext.arc(dotStartX - i * (dotRadius * 2 + dotGap), dotStartY, dotRadius, 0, Math.PI * 2);
            overlayContext.fillStyle = "#e5e7eb";
            overlayContext.fill();
          }

          if (seenSpikeCount > maxVisibleDots) {
            overlayContext.fillStyle = "#cbd5e1";
            overlayContext.font = "11px sans-serif";
            overlayContext.fillText(`+${seenSpikeCount - maxVisibleDots}`, dotStartX - maxVisibleDots * 14 - 18, 20);
          }

          const visibleShotMarkers = shotMarkersRef.current;
          if (shotMarkersRef.current.length !== liveClusterShotCountRef.current) {
            const liveClustering = clusterShotsBySpaceTime(shotMarkersRef.current, tweakSettings);
            liveShotClusteringRef.current = liveClustering;
            const liveColorMap: Record<number, string> = {};
            for (const cluster of liveClustering.clusters) {
              liveColorMap[cluster.clusterId] = clusterColorForId(cluster.clusterId);
            }
            liveClusterColorByIdRef.current = liveColorMap;
            liveClusterShotCountRef.current = shotMarkersRef.current.length;
          }
          const liveClusterByShotId = liveShotClusteringRef.current.shotClusterById;
          const liveClusterColorById = liveClusterColorByIdRef.current;
          const visibleClusterGeometry = clusterGeometryFromShots(visibleShotMarkers, liveClusterByShotId);
          drawClusterGeometry(overlayContext, visibleClusterGeometry, liveClusterColorById, scaleX, scaleY);
          drawClusterGeometry(context, visibleClusterGeometry, liveClusterColorById, 1, 1);
          const oneInchRadiusPx = pixelsPerInch > 0 ? pixelsPerInch / 2 : 0;
          const oneInchRadiusOverlay = oneInchRadiusPx * Math.max(scaleX, scaleY);
          for (const shotMarker of visibleShotMarkers) {
            const clusterId = liveClusterByShotId[shotMarker.id];
            const clusterColor = clusterId === undefined ? "#f87171" : liveClusterColorById[clusterId] ?? clusterColorForId(clusterId);
            const shotRadiusOverlay = shotMarker.radius * Math.max(scaleX, scaleY);
            overlayContext.beginPath();
            overlayContext.arc(shotMarker.centerX * scaleX, shotMarker.centerY * scaleY, shotRadiusOverlay, 0, Math.PI * 2);
            overlayContext.strokeStyle = hexToRgba(clusterColor, 0.95);
            overlayContext.lineWidth = 2;
            overlayContext.stroke();
            if (oneInchRadiusOverlay > 0) {
              overlayContext.beginPath();
              overlayContext.arc(
                shotMarker.centerX * scaleX,
                shotMarker.centerY * scaleY,
                oneInchRadiusOverlay,
                0,
                Math.PI * 2,
              );
              overlayContext.strokeStyle = "rgba(56, 189, 248, 0.95)";
              overlayContext.lineWidth = 1.5;
              overlayContext.stroke();
            }
            const overlayCenterX = shotMarker.centerX * scaleX;
            const overlayCenterY = shotMarker.centerY * scaleY;
            const overlayCrossHalfSize = Math.max(4, Math.min(12, shotRadiusOverlay * 0.35));
            drawCenterCross(
              overlayContext,
              overlayCenterX,
              overlayCenterY,
              overlayCrossHalfSize,
              "rgba(255, 255, 255, 0.95)",
              1.75,
            );
            if (clusterId !== undefined) {
              overlayContext.fillStyle = hexToRgba(clusterColor, 0.95);
              overlayContext.font = "10px sans-serif";
              overlayContext.fillText(
                `C${clusterId} S${shotMarker.shotNumber}`,
                overlayCenterX + shotRadiusOverlay + 2,
                overlayCenterY - shotRadiusOverlay - 2,
              );
            }

            context.beginPath();
            context.arc(shotMarker.centerX, shotMarker.centerY, shotMarker.radius, 0, Math.PI * 2);
            context.strokeStyle = hexToRgba(clusterColor, 0.95);
            context.lineWidth = 2;
            context.stroke();
            if (oneInchRadiusPx > 0) {
              context.beginPath();
              context.arc(shotMarker.centerX, shotMarker.centerY, oneInchRadiusPx, 0, Math.PI * 2);
              context.strokeStyle = "rgba(56, 189, 248, 0.95)";
              context.lineWidth = 1.5;
              context.stroke();
            }
            const crossHalfSize = Math.max(4, Math.min(12, shotMarker.radius * 0.35));
            drawCenterCross(
              context,
              shotMarker.centerX,
              shotMarker.centerY,
              crossHalfSize,
              "rgba(255, 255, 255, 0.95)",
              1.75,
            );
            if (clusterId !== undefined) {
              context.fillStyle = hexToRgba(clusterColor, 0.95);
              context.font = "10px sans-serif";
              context.fillText(
                `C${clusterId} S${shotMarker.shotNumber}`,
                shotMarker.centerX + shotMarker.radius + 2,
                shotMarker.centerY - shotMarker.radius - 2,
              );
            }
          }

          const now = performance.now();
          if (now - lastLoggedAtMsRef.current > 120) {
            lastLoggedAtMsRef.current = now;
            const entry: DetectionLogEntry = {
              frame: frameIndexRef.current,
              videoTimeSec: videoEl.currentTime,
              x: Math.round(drawRect.x),
              y: Math.round(drawRect.y),
              width: Math.round(drawRect.width),
              height: Math.round(drawRect.height),
              score: confidencePct,
              estimatedDistanceInches,
            };
            setLastDetection(entry);
            setLogEntries((current) => [...current, entry]);
          }
          if (shotDetectedThisFrame) {
            setLastShot(shotDetectedThisFrame);
            setShotLogEntries((current) => [...current, shotDetectedThisFrame]);
          }
          const sampleNow = performance.now();
          if (sampleNow - lastDetectionSampleAtMsRef.current > 50) {
            lastDetectionSampleAtMsRef.current = sampleNow;
            detectionTimelineRef.current.push({
              frame: frameIndexRef.current,
              videoTimeSec: videoEl.currentTime,
              x: Math.round(drawRect.x),
              y: Math.round(drawRect.y),
              width: Math.round(drawRect.width),
              height: Math.round(drawRect.height),
              score: confidencePct,
              estimatedDistanceInches,
            });
            if (detectionTimelineRef.current.length > 4000) {
              detectionTimelineRef.current.shift();
            }
            if (contourWindowSnapshotForFrame) {
              contourWindowTimelineRef.current.push(contourWindowSnapshotForFrame);
              if (contourWindowTimelineRef.current.length > 4000) {
                contourWindowTimelineRef.current.shift();
              }
              persistContourSnapshot(contourWindowSnapshotForFrame);
            }
          }

          frameGray.delete();
          frameMat.delete();

          animationFrameRef.current = requestAnimationFrame(processFrame);
        };

        animationFrameRef.current = requestAnimationFrame(processFrame);
      });

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      videoEl.pause();
      scanVideoRef.current = null;
      const overlayCtx = overlayEl.getContext("2d");
      overlayCtx?.clearRect(0, 0, overlayEl.width, overlayEl.height);
      const processedContourCanvas = processedContourCanvasRef.current;
      if (processedContourCanvas) {
        const restored = restorePersistedContourSnapshot();
        if (!restored) {
          const processedCtx = processedContourCanvas.getContext("2d");
          processedCtx?.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
        }
      }
      templateGray.delete();
      if (trackingHist) trackingHist.delete();
      resetShotFlowState();
      shotMarkersRef.current = [];
      shotSequenceRef.current = 0;
      liveShotClusteringRef.current = {
        selectedK: 0,
        finalK: 0,
        closeMergeCount: 0,
        objectiveScore: 0,
        shotClusterById: {},
        clusters: [],
      };
      liveClusterColorByIdRef.current = {};
      liveClusterShotCountRef.current = 0;
      setIsScanning(false);
      setDetectionEnabled(false);
      setDetectionConfidence(null);
      lastUiDetectionEnabledRef.current = false;
      lastUiDetectionConfidenceRef.current = -1;
      setScanStatus(stopRequestedRef.current ? "Scan stopped" : "Scan complete (concurrent audio + OpenCV)");
    } catch (error) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (overlayEl) {
        const overlayCtx = overlayEl.getContext("2d");
        overlayCtx?.clearRect(0, 0, overlayEl.width, overlayEl.height);
      }
      const processedContourCanvas = processedContourCanvasRef.current;
      if (processedContourCanvas) {
        const restored = restorePersistedContourSnapshot();
        if (!restored) {
          const processedCtx = processedContourCanvas.getContext("2d");
          processedCtx?.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
        }
      }
      if (scanVideoRef.current) {
        scanVideoRef.current.pause();
        scanVideoRef.current = null;
      }
      resetShotFlowState();
      shotMarkersRef.current = [];
      shotSequenceRef.current = 0;
      liveShotClusteringRef.current = {
        selectedK: 0,
        finalK: 0,
        closeMergeCount: 0,
        objectiveScore: 0,
        shotClusterById: {},
        clusters: [],
      };
      liveClusterColorByIdRef.current = {};
      liveClusterShotCountRef.current = 0;
      setIsScanning(false);
      setDetectionEnabled(false);
      setDetectionConfidence(null);
      lastUiDetectionEnabledRef.current = false;
      lastUiDetectionConfidenceRef.current = -1;
      setScanStatus(error instanceof Error ? error.message : "Failed to start scan.");
    } finally {
      scanTaskActiveRef.current = false;
      if (restartScanRequestedRef.current) {
        restartScanRequestedRef.current = false;
        window.setTimeout(() => {
          void startScan();
        }, 0);
      }
    }
  };

  const requestStartScan = () => {
    if (isScanning || scanTaskActiveRef.current) {
      restartScanRequestedRef.current = true;
      if (isScanning) {
        stopScan();
      } else {
        setScanStatus("Scan rerun queued...");
      }
      return;
    }
    void startScan();
  };

  const downloadLog = () => {
    if (logEntries.length === 0) {
      setScanStatus("No detections available to export.");
      return;
    }

    const csv = toCsv(logEntries);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const fileUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = fileUrl;
    a.download = "target-detections.csv";
    a.click();
    URL.revokeObjectURL(fileUrl);
  };

  const downloadShotLog = () => {
    if (shotLogEntries.length === 0) {
      setScanStatus("No significant shot-change events available to export.");
      return;
    }

    const csv = toShotCsv(shotLogEntries);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const fileUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = fileUrl;
    a.download = "shot-centers.csv";
    a.click();
    URL.revokeObjectURL(fileUrl);
  };

  const playSpikeSprite = async (spikeId: string) => {
    const spike = spikeMetadata.find((entry) => entry.id === spikeId);
    if (!spike) {
      setScanStatus("Spike metadata not found.");
      return;
    }

    const previewVideo = videoRef.current;
    if (previewVideo) previewVideo.pause();

    const canvasEl = processingCanvasRef.current;
    if (!canvasEl || !selectedVideoPreviewUrl) {
      setScanStatus("Analyzed video canvas or video source unavailable.");
      return;
    }

    try {
      stopAnalyzedPlayback();
      if (playPauseTimeoutRef.current) {
        window.clearTimeout(playPauseTimeoutRef.current);
        playPauseTimeoutRef.current = null;
      }

      const playbackVideo = document.createElement("video");
      playbackVideo.src = selectedVideoPreviewUrl;
      playbackVideo.preload = "auto";
      // Audio sprite playback is handled by Howler; keep video muted for reliable canvas playback.
      playbackVideo.muted = true;
      playbackVideo.defaultMuted = true;
      playbackVideo.playsInline = true;
      analyzedPlaybackVideoRef.current = playbackVideo;

      if (playbackVideo.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise<void>((resolve, reject) => {
          const onLoadedMetadata = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error("Unable to load analyzed video for spike playback."));
          };
          const cleanup = () => {
            playbackVideo.removeEventListener("loadedmetadata", onLoadedMetadata);
            playbackVideo.removeEventListener("error", onError);
          };

          playbackVideo.addEventListener("loadedmetadata", onLoadedMetadata, { once: true });
          playbackVideo.addEventListener("error", onError, { once: true });
        });
      }

      await seekVideo(playbackVideo, spike.windowStartSec);
      await playVideoForAnalysis(playbackVideo);

      const renderAnalyzedFrame = () => {
        const activeVideo = analyzedPlaybackVideoRef.current;
        const context = canvasEl.getContext("2d");
        if (!activeVideo || !context) {
          stopAnalyzedPlayback();
          return;
        }

        const width = activeVideo.videoWidth;
        const height = activeVideo.videoHeight;
        if (width <= 0 || height <= 0) {
          analyzedPlaybackRafRef.current = requestAnimationFrame(renderAnalyzedFrame);
          return;
        }

        canvasEl.width = width;
        canvasEl.height = height;
        context.drawImage(activeVideo, 0, 0, width, height);

        const nowSec = activeVideo.currentTime;
        const candidateDetections = detectionTimelineRef.current.length > 0 ? detectionTimelineRef.current : logEntries;
        const nearestDetection = candidateDetections.reduce<DetectionLogEntry | null>((closest, entry) => {
          if (entry.videoTimeSec < spike.windowStartSec || entry.videoTimeSec > spike.windowEndSec) return closest;
          const diff = Math.abs(entry.videoTimeSec - nowSec);
          if (!closest) return entry;
          return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
        }, null);

        if (nearestDetection) {
          const isBlinkOn = Math.floor(performance.now() / 500) % 2 === 0;
          const playbackDistanceSuffix =
            nearestDetection.estimatedDistanceInches === null
              ? ""
              : ` | ~${formatLinearFromInches(nearestDetection.estimatedDistanceInches, 1)}`;
          const playbackLabel = `SPIKE ${nearestDetection.score.toFixed(1)}%${playbackDistanceSuffix}`;
          const playbackLabelWidth = nearestDetection.estimatedDistanceInches === null ? 165 : 250;
          context.strokeStyle = isBlinkOn ? "#ffffff" : "#000000";
          context.lineWidth = 3;
          context.strokeRect(
            nearestDetection.x,
            nearestDetection.y,
            Math.max(1, nearestDetection.width),
            Math.max(1, nearestDetection.height),
          );
          context.fillStyle = "rgba(0, 0, 0, 0.72)";
          context.fillRect(nearestDetection.x, Math.max(nearestDetection.y - 20, 0), playbackLabelWidth, 18);
          context.fillStyle = isBlinkOn ? "#ffffff" : "#d4d4d4";
          context.font = "12px sans-serif";
          context.fillText(
            playbackLabel,
            nearestDetection.x + 6,
            Math.max(nearestDetection.y - 7, 12),
          );
        }
        const contourCanvas = processedContourCanvasRef.current;
        const contourContext = contourCanvas?.getContext("2d") ?? null;
        if (contourCanvas && contourContext) {
          const nearestContourSnapshot = contourWindowTimelineRef.current.reduce<ContourWindowFrameSnapshot | null>(
            (closest, entry) => {
              if (entry.videoTimeSec < spike.windowStartSec || entry.videoTimeSec > spike.windowEndSec) return closest;
              const diff = Math.abs(entry.videoTimeSec - nowSec);
              if (!closest) return entry;
              return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
            },
            null,
          );
          if (nearestContourSnapshot) {
            drawPersistedContourWindowView(contourContext, nearestContourSnapshot);
            persistContourSnapshot(nearestContourSnapshot);
          } else {
            const restored = restorePersistedContourSnapshot();
            if (!restored) {
              contourContext.clearRect(0, 0, contourCanvas.width, contourCanvas.height);
            }
          }
        }

        const spikeShotMarkers = shotLogEntries;
        const spikeClusterGeometry = clusterGeometryFromShots(spikeShotMarkers, shotClustering.shotClusterById);
        drawClusterGeometry(context, spikeClusterGeometry, clusterColorById, 1, 1);
        const oneInchRadiusPx = pixelsPerInch > 0 ? pixelsPerInch / 2 : 0;
        for (const shotMarker of spikeShotMarkers) {
          const clusterId = shotClustering.shotClusterById[shotMarker.id];
          const clusterColor = clusterId === undefined ? "#f87171" : clusterColorById[clusterId] ?? clusterColorForId(clusterId);
          const isCurrentShot = Math.abs(shotMarker.videoTimeSec - nowSec) <= tweakSettings.playbackCurrentShotWindowSec;
          context.beginPath();
          context.arc(shotMarker.centerX, shotMarker.centerY, shotMarker.radius, 0, Math.PI * 2);
          context.strokeStyle = isCurrentShot ? hexToRgba(clusterColor, 1) : hexToRgba(clusterColor, 0.45);
          context.lineWidth = isCurrentShot ? 2.5 : 1.5;
          context.stroke();
          if (oneInchRadiusPx > 0) {
            context.beginPath();
            context.arc(shotMarker.centerX, shotMarker.centerY, oneInchRadiusPx, 0, Math.PI * 2);
            context.strokeStyle = isCurrentShot ? "rgba(56, 189, 248, 1)" : "rgba(56, 189, 248, 0.5)";
            context.lineWidth = isCurrentShot ? 2 : 1.25;
            context.stroke();
          }
          const crossHalfSize = Math.max(4, Math.min(12, shotMarker.radius * 0.35));
          drawCenterCross(
            context,
            shotMarker.centerX,
            shotMarker.centerY,
            crossHalfSize,
            isCurrentShot ? "rgba(255, 255, 255, 0.95)" : "rgba(255, 255, 255, 0.65)",
            isCurrentShot ? 1.75 : 1.5,
          );
          if (clusterId !== undefined) {
            context.fillStyle = isCurrentShot ? hexToRgba(clusterColor, 1) : hexToRgba(clusterColor, 0.75);
            context.font = "10px sans-serif";
            context.fillText(
              `C${clusterId} S${shotMarker.shotNumber}`,
              shotMarker.centerX + shotMarker.radius + 2,
              shotMarker.centerY - shotMarker.radius - 2,
            );
          }
        }

        if (activeVideo.ended || activeVideo.currentTime >= spike.windowEndSec) {
          stopAnalyzedPlayback();
          return;
        }

        analyzedPlaybackRafRef.current = requestAnimationFrame(renderAnalyzedFrame);
      };

      analyzedPlaybackRafRef.current = requestAnimationFrame(renderAnalyzedFrame);
      setScanStatus(
        `Playing analyzed spike ${spike.id} (${spike.windowStartSec.toFixed(2)}s-${spike.windowEndSec.toFixed(2)}s)`,
      );
    } catch (error) {
      stopAnalyzedPlayback();
      setScanStatus(
        error instanceof Error
          ? `Unable to play analyzed video segment for selected spike: ${error.message}`
          : "Unable to play analyzed video segment for selected spike.",
      );
    }

    if (howlRef.current) {
      howlRef.current.stop();
      howlRef.current.play(spikeId);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <main className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-3 py-4 sm:gap-6 sm:px-8 sm:py-8">
        <section className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-4 sm:p-6">
          <h2 className="text-base font-semibold text-amber-100 sm:text-lg">Workflow Guide</h2>
          <p className="mt-2 text-sm text-amber-50/90">
            Follow sections top-to-bottom. Highlighted controls are the next action to press.
          </p>
          <p className="mt-1 text-xs text-amber-200">
            Current step:{" "}
            {workflowStep === "upload_video"
              ? "Upload a reference video"
              : workflowStep === "capture_frame"
                ? "Press Use Current Video Frame"
                : workflowStep === "draw_geometry"
                  ? "Draw target geometry on the captured frame"
                  : workflowStep === "calibrate"
                    ? "Set target dimensions and calibration values"
                    : workflowStep === "scan"
                      ? "Press Start Scan"
                      : "Export results (Download buttons)"}
          </p>
          <ol className="mt-3 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "upload_video"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : hasUploadedVideo
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                    : "border-gray-700 text-gray-300"
              }`}
            >
              1. Upload Video
            </li>
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "capture_frame"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : hasReferenceFrame
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                    : "border-gray-700 text-gray-300"
              }`}
            >
              2. Use Current Video Frame
            </li>
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "draw_geometry"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : hasDrawnGeometry
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                    : "border-gray-700 text-gray-300"
              }`}
            >
              3. Draw Geometry
            </li>
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "calibrate"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : hasScaleCalibration
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                    : "border-gray-700 text-gray-300"
              }`}
            >
              4. Calibrate
            </li>
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "scan"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : hasResultData
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
                    : "border-gray-700 text-gray-300"
              }`}
            >
              5. Start Scan
            </li>
            <li
              className={`rounded border px-2 py-1 ${
                workflowStep === "export"
                  ? "border-amber-300/80 bg-amber-500/20 text-amber-100"
                  : "border-gray-700 text-gray-300"
              }`}
            >
              6. Download Results
            </li>
          </ol>
        </section>

        <section className="rounded-xl border border-gray-700 bg-neutral-950 p-4 sm:p-6">
          <h1 className="text-xl font-semibold text-white sm:text-2xl">Gears and Tweaks</h1>
          <div className="mt-4 flex items-center justify-between gap-2">
            <p className="text-xs uppercase tracking-wide text-gray-300">Configure Units and Advanced Settings</p>
            <button
              type="button"
              onClick={() => setIsGearsExpanded((current) => !current)}
              className="rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-200 transition hover:bg-neutral-800"
            >
              {isGearsExpanded ? "Collapse" : "Expand"}
            </button>
          </div>
          <p className="mt-2 text-xs text-gray-400">
            Gears and tweaks include units, detection, audio correlation, tracking, and clustering settings.
          </p>
          {changedTweakCount > 0 ? (
            <p className="mt-2 rounded-md border border-gray-700 bg-neutral-900 px-3 py-2 text-xs text-gray-200">
              Warning: {changedTweakCount} custom tweak settings are active.
            </p>
          ) : null}
          {isGearsExpanded ? (
            <div className="mt-3 space-y-3">
              <UnitConverterSettings
                enabled={unitConversionEnabled}
                onEnabledChange={setUnitConversionEnabled}
                unit={displayLinearUnit}
                onUnitChange={setDisplayLinearUnit}
              />
              <GearsAndTweaksSection
                values={tweakSettings}
                onValueChange={updateTweakSetting}
                onReset={() => setTweakSettings(DEFAULT_TWEAK_SETTINGS)}
                changedCount={changedTweakCount}
              />
            </div>
          ) : null}
        </section>

        <section ref={videoSourceSectionRef} className="rounded-xl border border-gray-700 bg-neutral-950 p-4 sm:p-6">
          <h2 className="text-base font-semibold text-white sm:text-lg">Video Source and Reference Video</h2>
          <div className="mt-4 space-y-3">
            <fieldset className="space-y-2">
                <legend className="text-xs uppercase tracking-wide text-gray-300">Video Source</legend>
                <label className="flex items-center gap-2 py-1 text-base sm:text-sm">
                  <input
                    type="radio"
                    name="captureMode"
                    value="upload"
                    checked={captureMode === "upload"}
                    onChange={() => setCaptureMode("upload")}
                  />
                  Upload Video
                </label>
                <label className="flex items-center gap-2 py-1 text-base sm:text-sm">
                  <input
                    type="radio"
                    name="captureMode"
                    value="stream"
                    checked={captureMode === "stream"}
                    onChange={() => setCaptureMode("stream")}
                  />
                  Stream (WebSocket/RTSP URL)
                </label>
              </fieldset>

              {captureMode === "upload" ? (
                <label className="flex flex-col gap-1">
                  <span className="text-xs uppercase tracking-wide text-gray-300">Reference Video</span>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={onFileSelection("video")}
                    className={`rounded-md border border-dashed bg-black px-3 py-2 text-sm ${
                      workflowStep === "upload_video"
                        ? "border-amber-300 ring-2 ring-amber-300/70"
                        : "border-gray-700"
                    }`}
                  />
                  {selectedVideoName ? <span className="text-xs text-gray-400">{selectedVideoName}</span> : null}
                  {selectedVideoPreviewUrl ? (
                    <div className="relative mt-2">
                      <video
                        ref={videoRef}
                        src={selectedVideoPreviewUrl}
                        controls
                        className="max-h-56 w-full rounded-md border border-gray-700 sm:max-h-72"
                      />
                      <canvas
                        ref={overlayCanvasRef}
                        className="pointer-events-none absolute inset-0 h-full w-full rounded-md"
                      />
                    </div>
                  ) : null}
                </label>
              ) : (
                <label className="flex flex-col gap-1">
                  <span className="text-xs uppercase tracking-wide text-gray-300">Stream URL</span>
                  <input
                    type="text"
                    value={streamUrl}
                    onChange={(event) => setStreamUrl(event.target.value)}
                    placeholder="ws://... or rtsp://..."
                    className="rounded-md border border-gray-700 bg-black px-3 py-2 text-base outline-none ring-gray-400/40 focus:ring-1 sm:text-sm"
                  />
                </label>
              )}

          </div>
        </section>

        <section className="rounded-xl border border-gray-700 bg-neutral-950 p-4 sm:p-6">
          <h2 className="text-base font-semibold text-white sm:text-lg">Capture Image From Video</h2>
          <div className="mt-4">
            {captureMode === "upload" && selectedVideoPreviewUrl ? (
                <div className="rounded-md border border-gray-700 bg-black p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs uppercase tracking-wide text-gray-300">Capture Reference Image From Video</p>
                    <button
                      ref={captureFrameButtonRef}
                      type="button"
                      onClick={captureReferenceFrameFromVideo}
                      className={`rounded-md border border-sky-400/35 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/10 ${
                        workflowStep === "capture_frame" ? highlightActionClass : ""
                      }`}
                    >
                      Use Current Video Frame
                    </button>
                  </div>
                  {selectedImageName ? <span className="mt-2 block text-xs text-gray-400">{selectedImageName}</span> : null}
                  {selectedImagePreviewUrl ? (
                    <div
                      ref={roiContainerRef}
                      className="relative mt-2 cursor-crosshair touch-none"
                      onPointerDown={(event) => {
                        event.preventDefault();
                        startRoiSelection(event.clientX, event.clientY);
                      }}
                      onPointerMove={(event) => {
                        event.preventDefault();
                        updateRoiSelection(event.clientX, event.clientY);
                      }}
                      onPointerUp={endRoiSelection}
                      onPointerLeave={endRoiSelection}
                    >
                      <img
                        ref={imagePreviewRef}
                        src={selectedImagePreviewUrl}
                        alt="Reference preview"
                        className="w-full rounded-md border border-gray-700"
                      />
                      {roiRect ? (
                        <div
                          className="pointer-events-none absolute border-2"
                          style={{
                            left: `${roiRect.x * 100}%`,
                            top: `${roiRect.y * 100}%`,
                            width: `${roiRect.width * 100}%`,
                            height: `${roiRect.height * 100}%`,
                            borderColor: "#22c55e",
                            backgroundColor: "rgba(34, 197, 94, 0.22)",
                          }}
                        />
                      ) : null}
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-gray-400">
                      Scrub the video to the frame you want, then click Use Current Video Frame.
                    </p>
                  )}
                  {hasDrawnGeometry ? (
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={clearRoiSelection}
                        className="rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-300 transition hover:bg-neutral-800"
                      >
                        Clear Selection
                      </button>
                      <button
                        type="button"
                        onClick={goToNextAfterGeometrySelection}
                        className="rounded-md border border-emerald-400/45 px-3 py-1 text-xs text-emerald-100 transition hover:bg-emerald-500/10"
                      >
                        Next
                      </button>
                    </div>
                  ) : null}
                </div>
            ) : (
              <p className="text-sm text-gray-300">Upload a reference video to enable frame capture and ROI selection.</p>
            )}
          </div>
        </section>

        <section
          ref={calibrationSectionRef}
          className={`rounded-xl border bg-neutral-950 p-4 sm:p-6 ${
            workflowStep === "calibrate" ? "border-amber-300/70 ring-2 ring-amber-300/35" : "border-gray-700"
          }`}
        >
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-base font-semibold text-white sm:text-lg">Target Dimensions and Calibration</h2>
            {focalScalePxIn > 0 ? (
              <span className="text-[11px] text-gray-400">Calibrated scale: {focalScalePxIn.toFixed(1)}</span>
            ) : null}
          </div>
          <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Target Width ({activeLinearUnitLabel})</span>
              <input
                type="number"
                min="0"
                step={activeLinearInputStep}
                value={toDisplayLinearValue(targetWidthInches)}
                onChange={(event) => setTargetWidthInches(fromDisplayLinearValue(event.target.value))}
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Target Height ({activeLinearUnitLabel})</span>
              <input
                type="number"
                min="0"
                step={activeLinearInputStep}
                value={toDisplayLinearValue(targetHeightInches)}
                onChange={(event) => setTargetHeightInches(fromDisplayLinearValue(event.target.value))}
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Pixels Per Inch</span>
              <input
                type="number"
                min="0"
                step="0.1"
                value={pixelsPerInch}
                onChange={(event) => setPixelsPerInch(Math.max(0, Number(event.target.value) || 0))}
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Known Distance At Calibration ({activeLinearUnitLabel})</span>
              <input
                type="number"
                min="0"
                step={activeLinearInputStep}
                value={toDisplayLinearValue(calibrationDistanceInches)}
                onChange={(event) => setCalibrationDistanceInches(fromDisplayLinearValue(event.target.value))}
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
              />
            </label>
            {/* <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Manual Distance Override (in)</span>
              <input
                type="number"
                min="0"
                step="0.1"
                value={manualDistanceOverrideInches}
                onChange={(event) =>
                  setManualDistanceOverrideInches(Math.max(0, Number(event.target.value) || 0))
                }
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
              />
            </label> */}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <p className="text-[11px] text-gray-400">
              Draw the target in the capture section; calibration auto-runs when geometry and target dimensions are valid.
            </p>
          </div>
        </section>

        <section className="rounded-xl border border-gray-700 bg-neutral-950 p-4 sm:p-6">
          <h2 className="text-base font-semibold text-white sm:text-lg">Scan</h2>
          <p className="mt-2 text-sm text-gray-300">
            Uses Howler.js audio spikes to scan only within +/- 1 second windows, then logs OpenCV matches.
          </p>

          <div className="mt-4 space-y-3">
            <fieldset className="space-y-2">
              <legend className="text-xs uppercase tracking-wide text-gray-300">Tracking Mode</legend>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                <label className="flex items-center gap-2 rounded-md border border-gray-700 px-2 py-1.5 text-sm">
                  <input
                    type="radio"
                    name="trackingMode"
                    value="template"
                    checked={trackingMode === "template"}
                    onChange={() => setTrackingMode("template")}
                  />
                  Template
                </label>
                <label className="flex items-center gap-2 rounded-md border border-gray-700 px-2 py-1.5 text-sm">
                  <input
                    type="radio"
                    name="trackingMode"
                    value="meanshift"
                    checked={trackingMode === "meanshift"}
                    onChange={() => setTrackingMode("meanshift")}
                  />
                  meanShift
                </label>
                <label className="flex items-center gap-2 rounded-md border border-gray-700 px-2 py-1.5 text-sm">
                  <input
                    type="radio"
                    name="trackingMode"
                    value="camshift"
                    checked={trackingMode === "camshift"}
                    onChange={() => setTrackingMode("camshift")}
                  />
                  CamShift
                </label>
              </div>
            </fieldset>

            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-gray-300">
                Match Threshold ({matchThreshold.toFixed(0)}%)
              </span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={matchThreshold}
                onChange={(event) => setMatchThreshold(Number(event.target.value))}
                className="accent-gray-400"
              />
            </label>

            <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
              <button
                ref={startScanButtonRef}
                type="button"
                onClick={requestStartScan}
                disabled={!opencvReady || !howlerReady}
                className={`w-full rounded-md border border-sky-400/35 px-3 py-2.5 text-sm text-sky-100 transition hover:bg-sky-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto ${
                  workflowStep === "scan" ? highlightActionClass : ""
                }`}
              >
                {isScanning ? "Restart Scan" : "Start Scan"}
              </button>
              <button
                type="button"
                onClick={stopScan}
                disabled={!isScanning}
                className="w-full rounded-md border border-amber-400/35 px-3 py-2.5 text-sm text-amber-100 transition hover:bg-amber-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
              >
                Stop
              </button>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-gray-700 bg-neutral-950 p-4 sm:p-6">
          <h2 className="text-base font-semibold text-white sm:text-lg">Resulting Data</h2>
          <div className="mt-4 space-y-3">
            <div ref={exportButtonsRef} className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
              <button
                type="button"
                onClick={downloadLog}
                disabled={logEntries.length === 0}
                className={`w-full rounded-md border border-emerald-400/35 px-3 py-2.5 text-sm text-emerald-100 transition hover:bg-emerald-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto ${
                  workflowStep === "export" ? highlightActionClass : ""
                }`}
              >
                Download Log File
              </button>
              <button
                type="button"
                onClick={downloadShotLog}
                disabled={shotLogEntries.length === 0}
                className={`w-full rounded-md border border-rose-400/35 px-3 py-2.5 text-sm text-rose-100 transition hover:bg-rose-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto ${
                  workflowStep === "export" ? highlightActionClass : ""
                }`}
              >
                Download Shot Centers
              </button>
            </div>

            <p className="text-sm text-gray-300">
              Status: {scanStatus} | Detection:{" "}
              {isScanning && detectionConfidence === null ? (
                <span className="text-gray-300">initializing</span>
              ) : detectionConfidence === null ? (
                <span className="text-gray-300">idle</span>
              ) : detectionEnabled ? (
                <span className="text-gray-300">enabled ({detectionConfidence.toFixed(1)}%)</span>
              ) : (
                <span className="text-gray-300">searching ({detectionConfidence.toFixed(1)}%)</span>
              )}
            </p>
            {opencvError ? <p className="text-sm text-gray-300">{opencvError}</p> : null}
            {howlerError ? <p className="text-sm text-gray-300">{howlerError}</p> : null}
            <p className="text-sm text-gray-300">
              Howler: {howlerReady ? "ready" : "loading"} | Audio spikes: {spikeMetadata.length} | Sprites:{" "}
              {spritesReady ? "ready" : "pending"}
            </p>
            {audioRmsTimelineRef.current.length > 0 ? (
              <p className="text-sm text-gray-300">
                Audio baseline: {audioMeanDbfsRef.current.toFixed(1)} dBFS | spike threshold:{" "}
                {audioThresholdDbfsRef.current.toFixed(1)} dBFS
              </p>
            ) : null}
            <p className="text-sm text-gray-300">Log entries: {logEntries.length}</p>
            <p className="text-sm text-gray-300">Shot changes: {shotLogEntries.length}</p>
            {lastDetection ? (
              <p className="text-sm text-gray-300">
                Last detection at ({lastDetection.x}, {lastDetection.y}) with score{" "}
                {lastDetection.score.toFixed(1)}%
                {lastDetection.estimatedDistanceInches === null
                  ? ""
                  : ` | Estimated distance ~${formatLinearFromInches(lastDetection.estimatedDistanceInches, 1)}`}
              </p>
            ) : null}
            {lastShot ? (
              <p className="text-sm text-gray-300">
                Last shot center: ({lastShot.centerX}, {lastShot.centerY}) | change {lastShot.changeScore.toFixed(1)} | px{" "}
                {lastShot.changedPixels}
                {lastShot.timeSincePreviousShotSec === null
                  ? ""
                  : ` | dt ${lastShot.timeSincePreviousShotSec.toFixed(3)}s`}
                {lastShot.audioDecibelDbfs === null ? "" : ` | audio ${lastShot.audioDecibelDbfs.toFixed(1)} dBFS`}
                {lastShot.audioCorrelationScorePct === null
                  ? ""
                  : ` | audio-match ${lastShot.audioCorrelationScorePct.toFixed(0)}%`}
                {lastShot.estimatedDiameterInches === null
                  ? ""
                  : ` | est diameter ${formatLinearFromInches(lastShot.estimatedDiameterInches, 2)}`}
              </p>
            ) : null}
            {shotClustering.clusters.length > 0 ? (
              <div className="rounded-md border border-gray-700 bg-black p-3">
                <p className="text-xs uppercase tracking-wide text-gray-300">Shot Clusters (K-Means: X, Y, Time)</p>
                <p className="mt-1 text-[11px] text-gray-400">
                  Auto K selected: {shotClustering.selectedK} | Final clusters: {shotClustering.finalK}
                  {shotClustering.closeMergeCount > 0 ? ` | merged close clusters: ${shotClustering.closeMergeCount}` : ""}
                </p>
                <div className="mt-2 max-h-44 space-y-1 overflow-auto">
                  {shotClustering.clusters.map((cluster) => {
                    const clusterColor = clusterColorById[cluster.clusterId] ?? clusterColorForId(cluster.clusterId);
                    return (
                      <div
                        key={cluster.clusterId}
                        className="rounded border px-2 py-1 text-[11px] text-gray-200"
                        style={{
                          borderColor: hexToRgba(clusterColor, 0.5),
                          backgroundColor: hexToRgba(clusterColor, 0.08),
                        }}
                      >
                        <p className="flex items-center gap-1">
                          <span
                            className="inline-block h-2 w-2 rounded-full"
                            style={{ backgroundColor: hexToRgba(clusterColor, 0.95) }}
                          />
                          C{cluster.clusterId} | shots={cluster.count} | centroid=({cluster.centroidX.toFixed(1)},{" "}
                          {cluster.centroidY.toFixed(1)}) px | t={cluster.centroidTimeSec.toFixed(3)}s
                        </p>
                        <p className="text-gray-300">
                          spread={cluster.extremeSpreadPx.toFixed(1)} px | span={cluster.timeSpanSec.toFixed(3)}s | mean dt=
                          {cluster.meanTimeBetweenShotsSec === null ? "n/a" : `${cluster.meanTimeBetweenShotsSec.toFixed(3)}s`} | size=
                          {cluster.meanDiameterInches === null
                            ? "n/a"
                            : formatLinearFromInches(cluster.meanDiameterInches, 2)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}

            {spikeMetadata.length > 0 ? (
              <div className="rounded-md border border-gray-700 bg-black p-3">
                <p className="text-xs uppercase tracking-wide text-gray-300">Spike Metadata</p>
                <div className="mt-2 max-h-72 overflow-auto space-y-1">
                  {spikeMetadata.map((spike) => (
                    <div key={spike.id} className="rounded border border-gray-800 px-2 py-2 text-xs text-gray-300">
                      {(() => {
                        const isExpanded = expandedSpikeIds.includes(spike.id);
                        const shotCount = spikeShotSummaryById[spike.id]?.count ?? 0;
                        return (
                          <>
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex items-start gap-2">
                                <button
                                  type="button"
                                  onClick={() => toggleSpikeExpanded(spike.id)}
                                  className="mt-0.5 h-5 w-5 rounded border border-gray-600 text-center text-xs leading-4 text-gray-200 transition hover:bg-neutral-800"
                                  aria-expanded={isExpanded}
                                  aria-label={`${isExpanded ? "Collapse" : "Expand"} details for ${spike.id}`}
                                >
                                  {isExpanded ? "-" : "+"}
                                </button>
                                <div>
                                  <p>
                                    {spike.id} | t={spike.timeSec.toFixed(2)}s | rms={spike.strength.toFixed(4)} | db=
                                    {rmsToDbfs(spike.strength).toFixed(1)} dBFS
                                  </p>
                                  <p className="text-[11px] text-gray-400">
                                    Window: {spike.windowStartSec.toFixed(2)}s - {spike.windowEndSec.toFixed(2)}s
                                  </p>
                                </div>
                              </div>
                              <div className="flex flex-col items-end gap-1">
                                <button
                                  type="button"
                                  onClick={() => playSpikeSprite(spike.id)}
                                  disabled={isScanning}
                                  className="rounded border border-sky-500/30 px-2 py-0.5 text-[11px] text-sky-100 hover:bg-sky-500/10 disabled:opacity-50"
                                >
                                  Play
                                </button>
                                <p className="text-[11px] text-gray-300">Shots: {shotCount}</p>
                              </div>
                            </div>

                            {isExpanded ? (
                              shotCount > 0 ? (
                                <div className="mt-2 rounded border border-gray-800 bg-neutral-900/50 p-2 text-[11px] text-gray-300">
                                  <p>
                                    MPI: ({(spikeShotSummaryById[spike.id]?.meanPointOfImpactX ?? 0).toFixed(1)},{" "}
                                    {(spikeShotSummaryById[spike.id]?.meanPointOfImpactY ?? 0).toFixed(1)}) px | Extreme Spread:{" "}
                                    {(spikeShotSummaryById[spike.id]?.extremeSpreadPx ?? 0).toFixed(1)} px | Mean Radius:{" "}
                                    {(spikeShotSummaryById[spike.id]?.meanRadiusPx ?? 0).toFixed(1)} px
                                  </p>
                                  <p className="mt-0.5">
                                    H Spread: {(spikeShotSummaryById[spike.id]?.horizontalSpreadPx ?? 0).toFixed(1)} px | V Spread:{" "}
                                    {(spikeShotSummaryById[spike.id]?.verticalSpreadPx ?? 0).toFixed(1)} px | Avg Size:{" "}
                                    {spikeShotSummaryById[spike.id]?.averageDiameterInches === null
                                      ? "n/a"
                                      : formatLinearFromInches(spikeShotSummaryById[spike.id]?.averageDiameterInches ?? 0, 2)}
                                    {" | "}Avg Change: {(spikeShotSummaryById[spike.id]?.averageChangeScore ?? 0).toFixed(3)}
                                  </p>
                                  <div className="mt-1 space-y-0.5">
                                    {spikeShotSummaryById[spike.id]?.shots.map((shot) => {
                                      const shotClusterId = shotClustering.shotClusterById[shot.id];
                                      const shotClusterColor =
                                        shotClusterId === undefined
                                          ? null
                                          : clusterColorById[shotClusterId] ?? clusterColorForId(shotClusterId);
                                      return (
                                        <p key={shot.id} className="text-gray-400">
                                          S{shot.shotNumber} | t={shot.videoTimeSec.toFixed(3)}s
                                          {shot.timeSincePreviousShotSec === null
                                            ? ""
                                            : ` | dt=${shot.timeSincePreviousShotSec.toFixed(3)}s`}
                                          {shotClusterId === undefined ? null : (
                                            <span
                                              className="mx-1 inline-flex items-center rounded border px-1 py-0 text-[10px]"
                                              style={{
                                                borderColor: hexToRgba(shotClusterColor ?? "#f87171", 0.65),
                                                backgroundColor: hexToRgba(shotClusterColor ?? "#f87171", 0.15),
                                                color: hexToRgba(shotClusterColor ?? "#f87171", 0.95),
                                              }}
                                            >
                                              C{shotClusterId} S{shot.shotNumber}
                                            </span>
                                          )}
                                          {shot.audioDecibelDbfs === null ? "" : ` | ${shot.audioDecibelDbfs.toFixed(1)}dBFS`}
                                          {shot.audioCorrelationScorePct === null
                                            ? ""
                                            : ` | a-match=${shot.audioCorrelationScorePct.toFixed(0)}%`}
                                          {" | "}x={shot.centerX}, y={shot.centerY} | size{" "}
                                          {shot.estimatedDiameterInches === null
                                            ? `${shot.estimatedDiameterPx.toFixed(1)} px`
                                            : formatLinearFromInches(shot.estimatedDiameterInches, 2)}
                                        </p>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : (
                                <p className="mt-1 text-[11px] text-gray-500">No shots linked to this spike.</p>
                              )
                            ) : null}
                          </>
                        );
                      })()}
                    </div>
                  ))}
                </div>
                <p className="mt-2 text-xs text-gray-400">
                  Sprites generated: {Object.keys(audioSprites).length}
                </p>
              </div>
            ) : null}

            <canvas ref={processingCanvasRef} className="mt-2 w-full rounded-md border border-gray-700" />
            <p className="text-xs text-gray-400">
              Processed contour view (hybrid raw/temporal/effective masks + contour regions + OpenCV step status)
            </p>
            <canvas ref={processedContourCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
          </div>
        </section>
      </main>
    </div>
  );
}
