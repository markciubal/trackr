"use client";

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { OnboardingGuide } from "@/app/components/onboarding/OnboardingGuide";
import { InfoPopover } from "@/app/components/onboarding/InfoPopover";
import { detectTargetInCanvas } from "@/app/lib/targets/detect";
import { useColorBlindMode } from "@/app/lib/accessibility";
import { buildDatasetZip, type TrainingLabel, type TrainingSample, type TrainingSource } from "@/app/lib/training/dataset";
import { cropGrayWindow, getHoleClassifier, setHoleClassifier } from "@/app/lib/training/holeClassifier";
import { buildClassifierFromModel, isModelJSON } from "@/app/lib/training/jsonModel";
import { pixelsPerInchFromQr, toInches, type LinearUnit as TargetLinearUnit } from "@/app/lib/targets/payload";
import { getTarget } from "@/app/lib/targets/store";

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
  threshold?: (src: CvMat, dst: CvMat, thresh: number, maxval: number, type: number) => number;
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
  minMaxLoc: (src: CvMat) => {
    maxVal: number;
    maxLoc: { x: number; y: number };
    minVal?: number;
    minLoc?: { x: number; y: number };
  };
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
  THRESH_BINARY?: number;
  THRESH_BINARY_INV?: number;
  THRESH_OTSU?: number;
  TM_CCOEFF_NORMED: number;
  TM_SQDIFF_NORMED?: number;
  INTER_AREA: number;
  NORM_MINMAX: number;
  TermCriteria_COUNT: number;
  TermCriteria_EPS: number;
  RETR_EXTERNAL?: number;
  RETR_TREE?: number;
  CHAIN_APPROX_SIMPLE?: number;
  CHAIN_APPROX_NONE?: number;
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
  detectionMethod: "pixel_change" | "change_detect" | "manual";
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
  // Temporal persistence (filled in at scan end): a real impact keeps showing up
  // in the baseline-difference mask for the rest of the clip. `persistenceRatio`
  // is the fraction of post-detection frames the mark was still present; shots
  // that fade early (`persistent === false`) are transient false positives.
  persistenceRatio?: number | null;
  persistent?: boolean | null;
};

// Build a complete ShotLogEntry from a few known fields, defaulting the ~50
// detection-metadata fields. Used by manual taps and the change detector, which
// don't have the full per-frame analysis context the legacy pipeline produces.
function makeShotEntry(
  overrides: Partial<ShotLogEntry> & Pick<ShotLogEntry, "centerX" | "centerY" | "videoTimeSec">,
): ShotLogEntry {
  return {
    id: "",
    shotNumber: 0,
    frame: 0,
    timeSincePreviousShotSec: null,
    windowStartSec: 0,
    windowEndSec: 0,
    radius: 6,
    changedPixels: 0,
    changeScore: 1,
    estimatedDiameterInches: null,
    detectionMethod: "manual",
    trackingMode: "template",
    detectionEnabled: false,
    detectionConfidencePct: 0,
    detectionThresholdPct: 0,
    spikeFocusedWindow: false,
    nearestSpikeId: null,
    nearestSpikeTimeSec: null,
    nearestSpikeDeltaSec: null,
    nearestSpikeStrength: null,
    patchWidthPx: 0,
    patchHeightPx: 0,
    drawRectX: 0,
    drawRectY: 0,
    drawRectWidth: 0,
    drawRectHeight: 0,
    centerPatchX: 0,
    centerPatchY: 0,
    spanWidthPx: 0,
    spanHeightPx: 0,
    estimatedDiameterPx: 0,
    changedPixelRatioPct: 0,
    brightPixelCount: null,
    transitionPixelCount: null,
    transitionPurityPct: null,
    minBlobPixelsThreshold: 0,
    temporalHistoryFramesUsed: 0,
    shotCooldownMs: 0,
    residualMotionPx: null,
    trackedPointCount: null,
    estimatedRapidChangeMs: null,
    fastColorChangeTrigger: false,
    audioDecibelDbfs: null,
    audioDeltaFromMeanDb: null,
    audioPeakDbfs: null,
    audioCorrelationScorePct: null,
    nearestSpikeDecibelDbfs: null,
    ...overrides,
  };
}

type RoiRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type NormalizedMeasurementLine = {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
};

type TrackingMode = "template" | "meanshift" | "camshift";
type WorkflowStep = "upload_video" | "capture_frame" | "draw_geometry" | "calibrate" | "scan" | "export";

type HowlInstance = {
  seek: (seconds?: number) => number | void;
  play: (sprite?: string) => number;
  stop: () => void;
  unload: () => void;
  rate: (rate?: number, id?: number) => number | void;
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
  subPeakTimesSec: number[];
  signatureId: number;
  signatureKey: string;
};

type AudioSignatureCatalogEntry = {
  signatureId: number;
  signatureKey: string;
  count: number;
  meanPeakDbfs: number;
  meanSubPeakCount: number;
  meanSubPeakSpreadSec: number;
  spikeIds: string[];
};

type AudioRmsSample = {
  timeSec: number;
  rms: number;
  dbfs: number;
};

type AudioCaptureInfo = {
  sampleRate: number;
  channels: number;
  durationSec: number;
  totalSamples: number;
  rmsSampleCount: number;
  rmsHopSec: number;
  meanDbfs: number;
  thresholdDbfs: number;
  minDbfs: number;
  maxDbfs: number;
  spikeCount: number;
  signatureCount: number;
};

// A target outline the user drew (ROI), saved as a reusable detection template.
type TargetTemplate = {
  id: string;
  dataUrl: string;
  aspect: number;
  sourceName: string;
  targetWidthInches: number;
  targetHeightInches: number;
  createdAt: number;
  roi?: RoiRect;
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

type ContourRegionOverlay = Pick<
  ContourWindowRegionSnapshot,
  "centerX" | "centerY" | "minX" | "minY" | "maxX" | "maxY"
>;

type ContourWindowFrameSnapshot = {
  frame: number;
  videoTimeSec: number;
  patchWidthPx: number;
  patchHeightPx: number;
  changedPixels: number;
  maskRuns: Array<{ start: number; length: number }>;
  regions: ContourWindowRegionSnapshot[];
};

type YellowGreenHit = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
  area: number;
  centroidX: number;
  centroidY: number;
  meanV: number;
};

type YellowGreenFrameSnapshot = {
  frame: number;
  videoTimeSec: number;
  patchWidthPx: number;
  patchHeightPx: number;
  changedPixels: number;
  maskRuns: Array<{ start: number; length: number }>;
  hits: YellowGreenHit[];
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

type StoredAnalysisVideoMeta = {
  key: string;
  name: string;
  size: number;
  lastModified: number;
  type: string;
};

type LinearUnit = "in" | "ft" | "m" | "cm" | "mm";

const TEMPLATE_REGION_DATA_URL_KEY = "trackr-template-region-data-url";
const TEMPLATE_REGION_IMAGE_NAME_KEY = "trackr-template-region-image-name";
const TEMPLATE_REGION_RECT_KEY = "trackr-template-region-rect";
const TARGET_TEMPLATE_LIBRARY_KEY = "trackr-target-template-library";
const AUTO_ADVANCE_STORAGE_KEY = "trackr-auto-advance";
const MAX_TARGET_TEMPLATES = 12;
// Bundled sample video (public/_short.mp4) preloaded as the default source.
const PRELOADED_VIDEO_URL = "/_short.mp4";
const PRELOADED_VIDEO_NAME = "_short.mp4";
const ANALYSIS_VIDEO_DATA_URL_KEY = "trackr-analysis-video-data-url";
const ANALYSIS_VIDEO_META_KEY = "trackr-analysis-video-meta";
const ANALYSIS_VIDEO_SESSION_MAX_BYTES = 2_800_000;
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
  expectedHoleDiameterInches: number; // caliber/hole size for calibrated size gating (0 = off)
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
  // ±0.25s → a 0.5s analysis window centered on each detected bang.
  spikeWindowHalfSec: 0.25,
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
  expectedHoleDiameterInches: 0,
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
  // Low floor — the prominence gate (not this gap) prevents double-counting,
  // so rapid strings of fire still register one spike per shot.
  audioSpikeMinGapSec: 0.15,
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
      { key: "expectedHoleDiameterInches", label: "Expected Hole/Caliber (in, 0=off)", min: 0, max: 2, step: 0.01 },
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
    title: "DBSCAN Clustering",
    fields: [
      { key: "kmeansTimeWeight", label: "DBSCAN Time Weight", min: 0.1, max: 10, step: 0.01 },
      { key: "kmeansMaxClustersCap", label: "DBSCAN Max Clusters Cap", min: 1, max: 50, step: 1 },
      { key: "kmeansComplexityPenalty", label: "DBSCAN Cluster Penalty", min: 0, max: 1, step: 0.001 },
      { key: "kmeansClosePenaltyWeight", label: "DBSCAN Epsilon Scale", min: 0, max: 5, step: 0.001 },
      { key: "kmeansSsePenaltyWeight", label: "DBSCAN Dispersion Penalty", min: 0, max: 5, step: 0.001 },
      { key: "kmeansCentroidCloseDistance", label: "DBSCAN Epsilon Radius", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeCombinedDistanceMax", label: "DBSCAN Merge Combined Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeSpatialDistanceMax", label: "DBSCAN Merge Spatial Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMergeTimeDistanceMax", label: "DBSCAN Merge Time Distance Max", min: 0, max: 10, step: 0.01 },
      { key: "kmeansMaxIterations", label: "DBSCAN Min Samples (scaled)", min: 1, max: 500, step: 1 },
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

// Numeric input for unit-aware linear values (caliber, target size, etc.). A
// plain controlled type="number" bound to a rounded number can't accept decimals
// — typing "0." collapses back to "0" — so this keeps a text draft while focused
// and only reformats to the canonical value on blur.
function LinearNumberInput({
  valueInches,
  toDisplay,
  fromDisplay,
  onChangeInches,
  className,
  placeholder,
}: {
  valueInches: number;
  toDisplay: (valueInches: number) => number;
  fromDisplay: (raw: string) => number;
  onChangeInches: (valueInches: number) => void;
  className?: string;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const display = draft ?? String(toDisplay(valueInches));
  return (
    <input
      type="text"
      inputMode="decimal"
      value={display}
      placeholder={placeholder}
      className={className}
      onChange={(event) => {
        const raw = event.target.value;
        setDraft(raw);
        if (raw.trim() === "") {
          onChangeInches(0);
          return;
        }
        if (Number.isFinite(Number(raw))) onChangeInches(fromDisplay(raw));
      }}
      onBlur={() => setDraft(null)}
    />
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
                    step="any"
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

const MIN_EVENT_CHANGE_RATIO_PCT = 0.005;
const MAX_EVENT_CHANGE_RATIO_PCT = 55;
const SHOT_PERSISTENCE_MIN_MS = 180;
const SHOT_PERSISTENCE_MAX_GAP_MS = 850;
// Calibrated size gate: accept holes within [min, max] x the expected caliber size.
const EXPECTED_HOLE_MIN_FACTOR = 0.5;
const EXPECTED_HOLE_MAX_FACTOR = 2.5;
// Sub-pixel-ish patch registration: align the baseline to each frame's probe patch
// (small integer-shift search) before differencing, so residual jitter/perspective
// from the tracker stops faking hits.
const USE_PATCH_REGISTRATION = true;
const PATCH_REGISTRATION_MAX_SHIFT = 4; // px search radius
const PATCH_REGISTRATION_STRIDE = 3; // sample stride for the alignment SAD (speed)
// Auto-pick (scrub video to find/box the target automatically) is hidden for now.
const SHOW_AUTO_PICK = false;
// Simplified detection UI: lock grouping to quadtree and hide the change-detector
// toggles for now (flip these to re-expose the alternatives).
const SHOW_GROUPING_MODE_TOGGLE = false;
const SHOW_DETECTOR_TOGGLES = false;
const RELAXED_SHOT_MIN_SCORE = 0.01;
const TEMPLATE_PRIMARY_WEIGHT = 0.72;
const TEMPLATE_SECONDARY_WEIGHT = 0.28;
const FORCE_OPEN_SHOT_GATES = true;
const YELLOW_GREEN_HUE_MIN = 18;
const YELLOW_GREEN_HUE_MAX = 45;
const YELLOW_GREEN_SAT_MIN = 80;
const YELLOW_GREEN_VAL_MIN = 140;
const YELLOW_GREEN_TOPHAT_THRESHOLD = 25;
const YELLOW_GREEN_TOPHAT_RADIUS = 8;
const YELLOW_GREEN_COMPONENT_MIN_AREA = 8;
const YELLOW_GREEN_COMPONENT_MAX_AREA = 800;
const YELLOW_GREEN_COMPONENT_ASPECT_MIN = 0.4;
const YELLOW_GREEN_COMPONENT_ASPECT_MAX = 2.5;
const YELLOW_GREEN_COMPONENT_EXTENT_MIN = 0.25;
const YELLOW_GREEN_COMPONENT_MEAN_V_MIN = 170;
const USE_CONTOUR_RENDER_HITS_AS_SHOTS = true;
const CONTOUR_SHOT_DEDUP_DISTANCE_FACTOR = 1.8;
// Temporal confirmation for the live commit path: a candidate must be sighted
// at roughly the same spot in this many (not necessarily consecutive) frames
// before it becomes a shot. A real hole is permanent so it re-appears on the
// very next frame; one-frame flashes (ejected brass, glare, muzzle smoke) don't.
// Splatter hits are high-precision (2 sightings); baseline-change hits are
// noisier (3 sightings).
const CONTOUR_SHOT_MIN_SIGHTINGS_YELLOW_GREEN = 2;
const CONTOUR_SHOT_MIN_SIGHTINGS_CHANGE = 3;
// A pending track is dropped when not re-sighted within this many frames.
const CONTOUR_SHOT_TRACK_MAX_GAP_FRAMES = 10;
// ---- Contrast-primary hole detection (the main driver) ----
// Polarity-agnostic |current − baseline| threshold for the primary hole mask.
// Kept well above sensor/compression noise; the shape screen and significance
// gate do the fine filtering.
const CONTRAST_HOLE_DIFF_THRESHOLD = 16;
// A candidate region's mean |diff| must be at least this strong. Real impacts
// are high-contrast events (typically 60+); slow lighting drift hovers just
// above the mask threshold and is rejected here.
const CONTRAST_REGION_MIN_MEAN_DIFF = 20;
// Bright-splatter hits must also be NEW versus the scan-start baseline —
// printed yellow/green target graphics are bright but unchanged, so they can
// never register as hits.
const SPLATTER_REGION_MIN_MEAN_DIFF = 12;
// Splatter size band versus the calibrated bullet. The bright mark is the hole
// PLUS the flaked-off ring around it, so it runs larger than the caliber — but
// a fleck far smaller than the bullet, or a splash far larger, isn't a hit.
const SPLATTER_MIN_DIAMETER_SCALE = 0.4;
const SPLATTER_MAX_DIAMETER_SCALE = 4;
// When more than this fraction of the patch changed at once it's a reframe or
// lighting snap, not bullet holes — no contrast candidates from such frames.
const CONTRAST_MAX_GLOBAL_CHANGE_PCT = 35;
// Probe-resolution guard: pick the analysis patch size so the expected hole is
// at least this many pixels across. The cap matters: every mask pass is
// O(px²) and the splatter top-hat is O(px² · disk area) — 448px probes were
// taking seconds per frame, which reads as "the scan is dead".
const PROBE_MIN_HOLE_DIAMETER_PX = 5;
const PROBE_MAX_SIZE_PX = 320;
// Manual-override sentinel for "false positive": removes the shot from the map,
// groups, stats and count entirely (recoverable via Reset, unlike the hard Delete
// tool). Real group ids are >= 1; 0 = stray.
const FALSE_POSITIVE_OVERRIDE = -1;

// Wizard step indices. The flow is split into many small steps so no single step
// stacks too much vertical content (better on mobile). Calibration is split into
// "method" + "measurements"; the old Analysis step is split into Scan / Map /
// Review. Keep these in sync with the sectionSteps array.
const STEP_VIDEO = 0;
const STEP_TARGET = 1;
const STEP_CALIB_METHOD = 2;
const STEP_CALIB_VALUES = 3;
const STEP_SCAN = 4;
const STEP_MAP = 5;
const STEP_REVIEW = 6;
const STEP_SAVE = 7;
// The map canvas is live/visible across Scan + Map (and the RAF draw runs there).
const STEP_ANALYSIS_FIRST = STEP_SCAN;
const STEP_ANALYSIS_LAST = STEP_REVIEW;
const LIVE_GROUP_UPDATE_INTERVAL_MS = 500;
const CLUSTER_MIN_VISIBLE_AGE_SEC = 1;
const AUDIO_SUBPEAK_RELATIVE_THRESHOLD = 0.45;
const AUDIO_SUBPEAK_STDDEV_FACTOR = 0.55;
// Peak prominence: a candidate spike must rise at least this many standard
// deviations above the quietest point (valley) since the previous accepted
// spike. The decaying tail of one bang never dips, so its ripples can't count
// twice — while two real shots in quick succession, with a dip between them,
// both register.
const AUDIO_SPIKE_MIN_PROMINENCE_STDDEV = 1.5;

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

function isLikelyGoodShot(
  entry: ShotLogEntry,
  tweaks: TweakSettings,
  options?: { enforceRealGate?: boolean },
): boolean {
  // FORCE_OPEN_SHOT_GATES logs every detection so nothing is missed during analysis, but
  // clustering only wants shots that pass the real qualification criteria. Pass
  // enforceRealGate to evaluate those criteria even while the gates are forced open.
  if (FORCE_OPEN_SHOT_GATES && !options?.enforceRealGate) {
    return entry.changedPixels > 0 && entry.changeScore >= 0;
  }
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
    // Calibrated size gate: when an expected caliber/hole size is set, reject
    // candidates well outside it (noise that's too small, artifacts too big).
    const expectedHoleInches = tweaks.expectedHoleDiameterInches;
    if (expectedHoleInches > 0) {
      if (
        entry.estimatedDiameterInches < expectedHoleInches * EXPECTED_HOLE_MIN_FACTOR ||
        entry.estimatedDiameterInches > expectedHoleInches * EXPECTED_HOLE_MAX_FACTOR
      ) {
        return false;
      }
    }
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

// A shot "makes it" (and is eligible for space-time/quadtree grouping) when it passes the
// real qualification gate, regardless of whether FORCE_OPEN_SHOT_GATES is logging everything.
function shotMakesIt(entry: ShotLogEntry, tweaks: TweakSettings): boolean {
  return isLikelyGoodShot(entry, tweaks, { enforceRealGate: true });
}

// Caliber size gate for the legacy DBSCAN / quadtree grouping: when an expected
// hole/caliber is set (and the shot is calibrated to inches), a hit smaller than
// the caliber's accepted minimum — the same floor the change detector uses — is
// kept and DRAWN, but excluded from grouping so it shows as a stray rather than
// joining a group.
function passesCaliberGate(entry: ShotLogEntry, tweaks: TweakSettings): boolean {
  if (tweaks.expectedHoleDiameterInches <= 0) return true;
  if (entry.estimatedDiameterInches === null) return true;
  return entry.estimatedDiameterInches >= tweaks.expectedHoleDiameterInches * CHANGE_HOLE_MIN_DIAMETER_SCALE;
}

// Temporal-persistence gate. A real bullet hole keeps differing from the baseline
// frame for the rest of the clip, so during the scan we sample each logged shot's
// location in the per-frame baseline-difference mask. A detection that stops
// showing up (present in fewer than this fraction of its post-detection frames)
// was a transient flash — glare, motion, someone walking through — and is kept
// and drawn but excluded from grouping, categorized as "transient" rather than a
// real impact. Shots fired too near the end (fewer than MIN_FRAMES of footage
// after them) get the benefit of the doubt — the last shots are expected to last.
const SHOT_PERSISTENCE_MIN_RATIO = 0.7;
const SHOT_PERSISTENCE_MIN_FRAMES = 3;
// Flicker gate: an area whose presence toggles on/off this many times AND in at
// least this fraction of its sampled frames is unstable (glare, foliage, a moving
// object), not a fixed hole — auto-strayed (treated as transient) even if it
// happened to clear the persistence ratio.
const SHOT_FLICKER_MIN_TRANSITIONS = 4;
const SHOT_FLICKER_MIN_RATE = 0.25;
function isPersistentShot(entry: ShotLogEntry): boolean {
  return entry.persistent !== false;
}

// Audio false-positive gate. The loudest bangs are the real shots (the shooter);
// a visual hit that isn't backed by a loud-enough report is likely a false
// positive. We take the loudest shot as the reference and lower that dB threshold
// by 10% five times to set a tolerant floor; shots quieter than the floor are
// excluded from grouping (flagged stray). Shots with no audio data aren't judged.
function audioStrayThresholdDbfs(entries: ShotLogEntry[]): number | null {
  const louds = entries
    .map((entry) => entry.audioDecibelDbfs)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  if (louds.length < 3) return null; // not enough audio to establish the shooter level
  let threshold = Math.max(...louds); // the loudest bang ≈ the shooter
  for (let i = 0; i < 5; i += 1) threshold *= 1.1; // dBFS is negative → 10% lower, ×5
  return threshold;
}

function passesAudioConsistency(entry: ShotLogEntry, thresholdDbfs: number | null): boolean {
  if (thresholdDbfs === null) return true; // not enough audio data to judge
  if (entry.audioDecibelDbfs === null || !Number.isFinite(entry.audioDecibelDbfs)) return true; // no audio for this hit
  return entry.audioDecibelDbfs >= thresholdDbfs; // loud enough to match a real shot
}

// Positive loudness signal: the detection lines up with a real bang in the synced
// audio — at least this many dB above the running ambient mean. Unlike the
// shooter-relative consistency filter, this works on a single shot (no need for a
// few bangs to set a reference), so it can affirmatively confirm a hit.
const SHOT_MIN_AUDIO_DELTA_DB = 6;
function hasLoudBang(entry: ShotLogEntry): boolean {
  const delta = entry.audioDeltaFromMeanDb;
  if (delta === null || !Number.isFinite(delta)) return false; // no usable audio → can't confirm by sound
  return delta >= SHOT_MIN_AUDIO_DELTA_DB;
}

// The shot gate. A detection is treated as a real shot when it is BIG enough
// (caliber size) and STICKS AROUND enough (persists, no flicker) — both always
// required — and then either:
//   • it's backed by a LOUD bang in the synced audio (sound is the decisive gate:
//     loud + big + sticks ⇒ shot, even below the confidence noise filter), or
//   • lacking a loud bang to confirm, it still clears the audio-consistency and
//     confidence noise filters.
function qualifiesAsShot(
  entry: ShotLogEntry,
  tweaks: TweakSettings,
  audioThresholdDbfs: number | null,
  minConfidencePct: number,
): boolean {
  if (!passesCaliberGate(entry, tweaks)) return false; // big enough
  if (!isPersistentShot(entry)) return false; // sticks around enough
  if (hasLoudBang(entry)) return true; // loud enough → it's a shot
  return passesAudioConsistency(entry, audioThresholdDbfs) && shotConfidencePct(entry) >= minConfidencePct;
}

type ShotDetailRow = { label: string; value: string; emphasis?: boolean };

type ShotTableSortKey = "num" | "group" | "conf" | "size" | "time" | "loud" | "persist";

// Traffic-light coloring for table cells: green good → amber → red poor.
function confColorClass(pct: number): string {
  if (pct >= 70) return "text-emerald-400";
  if (pct >= 40) return "text-amber-400";
  return "text-rose-400";
}
function persistColorClass(ratio: number | null): string {
  if (ratio === null) return "text-gray-500";
  if (ratio >= SHOT_PERSISTENCE_MIN_RATIO) return "text-emerald-400";
  if (ratio >= 0.4) return "text-amber-400";
  return "text-rose-400";
}
function loudColorClass(dbfs: number | null): string {
  if (dbfs === null) return "text-gray-500";
  const t = Math.max(0, Math.min(1, (dbfs + 60) / 50)); // -60 dBFS quiet → -10 loud
  if (t >= 0.66) return "text-emerald-400";
  if (t >= 0.33) return "text-amber-400";
  return "text-rose-400";
}

// Everything known about a single shot, as labeled rows for the pinned detail
// card. Null/unknown audio fields are omitted (we only show what's actually
// measured); the core position/size/timing/detection fields are always shown.
function buildShotDetailRows(
  shot: ShotLogEntry,
  ctx: {
    groupId: number | undefined;
    groupLabel: (id: number) => string;
    conf: number;
    formatLinearFromInches: (inches: number, digits?: number) => string;
  },
): ShotDetailRow[] {
  const rows: ShotDetailRow[] = [];

  // Category / grouping
  const category =
    shot.persistent === false
      ? "Transient (auto-stray)"
      : ctx.groupId
        ? ctx.groupLabel(ctx.groupId)
        : "Stray";
  rows.push({ label: "Category", value: category, emphasis: true });
  rows.push({ label: "Confidence", value: `${ctx.conf.toFixed(0)}%` });
  if (shot.persistenceRatio !== null && shot.persistenceRatio !== undefined) {
    rows.push({ label: "Stays visible", value: `${Math.round(shot.persistenceRatio * 100)}% of frames` });
  }

  // Position & size
  rows.push({ label: "Position", value: `(${shot.centerX.toFixed(0)}, ${shot.centerY.toFixed(0)}) px` });
  rows.push({
    label: "Size",
    value:
      shot.estimatedDiameterInches !== null
        ? `${ctx.formatLinearFromInches(shot.estimatedDiameterInches, 2)} (${shot.estimatedDiameterPx.toFixed(0)} px)`
        : `${shot.estimatedDiameterPx.toFixed(0)} px`,
  });
  if (shot.spanWidthPx > 0 || shot.spanHeightPx > 0) {
    rows.push({ label: "Span", value: `${shot.spanWidthPx}×${shot.spanHeightPx} px` });
  }

  // Timing
  rows.push({ label: "Time", value: `${shot.videoTimeSec.toFixed(2)} s` });
  rows.push({ label: "Frame", value: String(shot.frame) });
  if (shot.timeSincePreviousShotSec !== null) {
    rows.push({ label: "Since previous", value: `${shot.timeSincePreviousShotSec.toFixed(2)} s` });
  }

  // Detection
  const methodLabel =
    shot.detectionMethod === "manual"
      ? "Manual"
      : shot.detectionMethod === "change_detect"
        ? "Change detector"
        : "Pixel change";
  rows.push({ label: "Detected by", value: `${methodLabel} · ${shot.trackingMode}` });
  rows.push({ label: "Changed px", value: `${shot.changedPixels} (${shot.changedPixelRatioPct.toFixed(1)}%)` });
  rows.push({ label: "Change score", value: shot.changeScore.toFixed(2) });

  // Audio (only what was measured)
  if (shot.audioDecibelDbfs !== null) rows.push({ label: "Loudness", value: `${shot.audioDecibelDbfs.toFixed(0)} dBFS` });
  if (shot.audioDeltaFromMeanDb !== null) {
    rows.push({ label: "Above ambient", value: `${shot.audioDeltaFromMeanDb.toFixed(0)} dB` });
  }
  if (shot.audioPeakDbfs !== null) rows.push({ label: "Peak", value: `${shot.audioPeakDbfs.toFixed(0)} dBFS` });
  if (shot.audioCorrelationScorePct !== null) {
    rows.push({ label: "Audio match", value: `${shot.audioCorrelationScorePct.toFixed(0)}%` });
  }
  if (shot.nearestSpikeDeltaSec !== null) {
    rows.push({ label: "Nearest bang", value: `${(shot.nearestSpikeDeltaSec * 1000).toFixed(0)} ms away` });
  }
  if (shot.audioDecibelDbfs !== null || shot.audioDeltaFromMeanDb !== null) {
    rows.push({ label: "Loud bang", value: hasLoudBang(shot) ? "yes" : "no" });
  }

  return rows;
}

// A 0-100 "this is a real shot, not noise" confidence. Prefers the audio
// correlation (real shots line up with an audio spike) and falls back to the
// visual/tracking confidence when there's no usable audio.
function shotConfidencePct(entry: ShotLogEntry): number {
  const audio = entry.audioCorrelationScorePct;
  if (audio !== null && Number.isFinite(audio)) {
    return Math.max(0, Math.min(100, audio));
  }
  return Math.max(0, Math.min(100, entry.detectionConfidencePct));
}

function isLikelyGoodShotRelaxed(entry: ShotLogEntry, tweaks: TweakSettings): boolean {
  if (FORCE_OPEN_SHOT_GATES) {
    return entry.changedPixels > 0 && entry.changeScore >= 0;
  }
  if (entry.videoTimeSec < entry.windowStartSec || entry.videoTimeSec > entry.windowEndSec) return false;
  if (entry.windowEndSec < entry.windowStartSec) return false;
  if (
    entry.nearestSpikeId !== null &&
    (entry.nearestSpikeDeltaSec === null ||
      entry.nearestSpikeDeltaSec > Math.max(0.1, Math.max(2.5, tweaks.colorSpikeDeltaMaxSec * 2.2)))
  ) {
    return false;
  }

  if (entry.changedPixels <= 0) return false;
  if (entry.changeScore < Math.max(RELAXED_SHOT_MIN_SCORE, tweaks.minVisibleHitChangeScore * 0.25)) return false;
  if (entry.changedPixelRatioPct < 0.001 || entry.changedPixelRatioPct > 90) return false;

  const relaxedAspectMax = Math.max(16, Math.max(10, tweaks.colorAspectRatioMax) * 1.7);
  if (spanAspectRatio(entry.spanWidthPx, entry.spanHeightPx) > relaxedAspectMax) return false;

  if (entry.estimatedDiameterInches !== null) {
    const relaxedMaxInches = Math.max(6, tweaks.maxShotDiameterInches * 3);
    if (entry.estimatedDiameterInches > relaxedMaxInches) return false;
  } else {
    const relaxedMaxPx = Math.max(
      1200,
      tweaks.colorDiameterMaxPx * 2.5,
      Math.min(entry.drawRectWidth, entry.drawRectHeight) * 0.85,
    );
    if (entry.estimatedDiameterPx > relaxedMaxPx) return false;
  }

  if (entry.audioCorrelationScorePct !== null && entry.audioCorrelationScorePct < Math.max(0, tweaks.colorAudioCorrelationMinPct * 0.4)) {
    return false;
  }
  if (entry.audioDeltaFromMeanDb !== null && entry.audioDeltaFromMeanDb < tweaks.colorAudioDeltaMinDb - 10) return false;

  return true;
}

type NormalizedShotPoint = {
  shot: ShotLogEntry;
  nx: number;
  ny: number;
  nt: number;
};

type KdTreeNode = {
  pointIndex: number;
  axis: 0 | 1 | 2;
  left: KdTreeNode | null;
  right: KdTreeNode | null;
};

type DbscanClusterModel = {
  assignments: number[];
  centroids: Array<{ x: number; y: number; t: number }>;
  initialClusterCount: number;
  noiseCount: number;
  objectiveScore: number;
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

function kdPointCoordinate(point: NormalizedShotPoint, axis: 0 | 1 | 2): number {
  if (axis === 0) return point.nx;
  if (axis === 1) return point.ny;
  return point.nt;
}

function buildKdTree(pointIndices: number[], points: NormalizedShotPoint[], depth = 0): KdTreeNode | null {
  if (pointIndices.length === 0) return null;
  const axis = (depth % 3) as 0 | 1 | 2;
  const sorted = [...pointIndices].sort(
    (a, b) => kdPointCoordinate(points[a], axis) - kdPointCoordinate(points[b], axis),
  );
  const median = Math.floor(sorted.length / 2);
  return {
    pointIndex: sorted[median],
    axis,
    left: buildKdTree(sorted.slice(0, median), points, depth + 1),
    right: buildKdTree(sorted.slice(median + 1), points, depth + 1),
  };
}

function kdRadiusSearch(
  node: KdTreeNode | null,
  points: NormalizedShotPoint[],
  targetX: number,
  targetY: number,
  targetT: number,
  radiusSq: number,
  hits: number[],
): void {
  if (!node) return;
  const point = points[node.pointIndex];
  const dx = point.nx - targetX;
  const dy = point.ny - targetY;
  const dt = point.nt - targetT;
  const distanceSq = dx * dx + dy * dy + dt * dt;
  if (distanceSq <= radiusSq) {
    hits.push(node.pointIndex);
  }

  const targetCoord = node.axis === 0 ? targetX : node.axis === 1 ? targetY : targetT;
  const nodeCoord = kdPointCoordinate(point, node.axis);
  const diff = targetCoord - nodeCoord;
  const nearChild = diff <= 0 ? node.left : node.right;
  const farChild = diff <= 0 ? node.right : node.left;
  kdRadiusSearch(nearChild, points, targetX, targetY, targetT, radiusSq, hits);
  if (diff * diff <= radiusSq) {
    kdRadiusSearch(farChild, points, targetX, targetY, targetT, radiusSq, hits);
  }
}

function centroidFromPointIndices(
  points: NormalizedShotPoint[],
  pointIndices: number[],
): { x: number; y: number; t: number } {
  let sumX = 0;
  let sumY = 0;
  let sumT = 0;
  for (const pointIndex of pointIndices) {
    sumX += points[pointIndex].nx;
    sumY += points[pointIndex].ny;
    sumT += points[pointIndex].nt;
  }
  const count = Math.max(1, pointIndices.length);
  return { x: sumX / count, y: sumY / count, t: sumT / count };
}

function runDbscanClustering(points: NormalizedShotPoint[], tweaks: TweakSettings): DbscanClusterModel {
  if (points.length === 0) {
    return {
      assignments: [],
      centroids: [],
      initialClusterCount: 0,
      noiseCount: 0,
      objectiveScore: 0,
    };
  }

  const radiusScale = Math.max(0.25, tweaks.kmeansClosePenaltyWeight);
  const radius = Math.max(0.05, tweaks.kmeansCentroidCloseDistance * radiusScale);
  const radiusSq = radius * radius;
  const minSamples = Math.max(2, Math.min(40, Math.round(Math.max(2, tweaks.kmeansMaxIterations * 0.12))));
  const maxClustersCap = Math.max(1, Math.round(tweaks.kmeansMaxClustersCap));
  const pointIndices = Array.from({ length: points.length }, (_, index) => index);
  const kdTree = buildKdTree(pointIndices, points);
  if (!kdTree) {
    return {
      assignments: [],
      centroids: [],
      initialClusterCount: 0,
      noiseCount: points.length,
      objectiveScore: 0,
    };
  }

  const UNVISITED = -2;
  const NOISE = -1;
  const neighborCache = new Map<number, number[]>();
  const regionQuery = (pointIndex: number): number[] => {
    const cached = neighborCache.get(pointIndex);
    if (cached) return cached;
    const hits: number[] = [];
    const point = points[pointIndex];
    kdRadiusSearch(kdTree, points, point.nx, point.ny, point.nt, radiusSq, hits);
    neighborCache.set(pointIndex, hits);
    return hits;
  };

  const assignments = new Array<number>(points.length).fill(UNVISITED);
  let clusterIndex = 0;
  for (let i = 0; i < points.length; i += 1) {
    if (assignments[i] !== UNVISITED) continue;
    const neighbors = regionQuery(i);
    if (neighbors.length < minSamples) {
      assignments[i] = NOISE;
      continue;
    }

    assignments[i] = clusterIndex;
    const queue = neighbors.filter((neighborIndex) => neighborIndex !== i);
    const queued = new Set<number>(queue);
    for (let head = 0; head < queue.length; head += 1) {
      const neighborIndex = queue[head];
      queued.delete(neighborIndex);
      if (assignments[neighborIndex] === NOISE) {
        assignments[neighborIndex] = clusterIndex;
      }
      if (assignments[neighborIndex] !== UNVISITED && assignments[neighborIndex] !== clusterIndex) {
        continue;
      }
      if (assignments[neighborIndex] === UNVISITED) {
        assignments[neighborIndex] = clusterIndex;
      }

      const neighborNeighbors = regionQuery(neighborIndex);
      if (neighborNeighbors.length < minSamples) continue;
      for (const candidate of neighborNeighbors) {
        if (assignments[candidate] !== UNVISITED && assignments[candidate] !== NOISE) continue;
        if (queued.has(candidate)) continue;
        queue.push(candidate);
        queued.add(candidate);
      }
    }
    clusterIndex += 1;
  }

  const initialClusterCount = clusterIndex;
  if (initialClusterCount > maxClustersCap) {
    const clusterMembers = Array.from({ length: initialClusterCount }, () => [] as number[]);
    for (let i = 0; i < assignments.length; i += 1) {
      const clusterId = assignments[i];
      if (clusterId >= 0) clusterMembers[clusterId].push(i);
    }
    const keepClusterIds = clusterMembers
      .map((members, clusterId) => ({ clusterId, size: members.length }))
      .sort((a, b) => {
        const sizeDelta = b.size - a.size;
        if (sizeDelta !== 0) return sizeDelta;
        return a.clusterId - b.clusterId;
      })
      .slice(0, maxClustersCap)
      .map((entry) => entry.clusterId);
    const oldToNew = new Map<number, number>();
    for (let i = 0; i < keepClusterIds.length; i += 1) {
      oldToNew.set(keepClusterIds[i], i);
    }
    for (let i = 0; i < assignments.length; i += 1) {
      const clusterId = assignments[i];
      if (clusterId < 0) continue;
      assignments[i] = oldToNew.get(clusterId) ?? NOISE;
    }
  }

  const oldToNewCluster = new Map<number, number>();
  const compactAssignments = new Array<number>(assignments.length).fill(NOISE);
  for (let i = 0; i < assignments.length; i += 1) {
    const clusterId = assignments[i];
    if (clusterId < 0) continue;
    let remappedClusterId = oldToNewCluster.get(clusterId);
    if (remappedClusterId === undefined) {
      remappedClusterId = oldToNewCluster.size;
      oldToNewCluster.set(clusterId, remappedClusterId);
    }
    compactAssignments[i] = remappedClusterId;
  }

  const compactBuckets: number[][] = Array.from({ length: oldToNewCluster.size }, () => []);
  for (let i = 0; i < compactAssignments.length; i += 1) {
    const clusterId = compactAssignments[i];
    if (clusterId < 0) continue;
    compactBuckets[clusterId].push(i);
  }

  const centroids = compactBuckets.map((bucket) => centroidFromPointIndices(points, bucket));
  let dispersion = 0;
  let assignedCount = 0;
  let noiseCount = 0;
  for (let i = 0; i < points.length; i += 1) {
    const clusterId = compactAssignments[i];
    if (clusterId < 0) {
      noiseCount += 1;
      continue;
    }
    const centroid = centroids[clusterId];
    const distance = euclideanDistance3(points[i].nx, points[i].ny, points[i].nt, centroid.x, centroid.y, centroid.t);
    dispersion += distance * distance;
    assignedCount += 1;
  }
  const coverage = assignedCount / Math.max(1, points.length);
  const noiseRatio = noiseCount / Math.max(1, points.length);
  const normalizedDispersion = dispersion / Math.max(1, assignedCount);
  const objectiveScore =
    coverage -
    tweaks.kmeansComplexityPenalty * Math.max(0, centroids.length - 1) -
    tweaks.kmeansSsePenaltyWeight * normalizedDispersion -
    tweaks.kmeansClosePenaltyWeight * noiseRatio;

  return {
    assignments: compactAssignments,
    centroids,
    initialClusterCount,
    noiseCount,
    objectiveScore,
  };
}

function filterShotsByAnalysisAge(
  shots: ShotLogEntry[],
  analysisTimeSec: number,
  minVisibleAgeSec = CLUSTER_MIN_VISIBLE_AGE_SEC,
): ShotLogEntry[] {
  const safeMinAge = Math.max(0, minVisibleAgeSec);
  if (safeMinAge <= 0 || !Number.isFinite(analysisTimeSec)) return shots;
  return shots.filter((shot) => analysisTimeSec - shot.videoTimeSec >= safeMinAge);
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
  // Increase temporal influence so timing separation drives grouping more strongly.
  const effectiveTimeWeight = Math.max(0, tweaks.kmeansTimeWeight) * 4;
  const points: NormalizedShotPoint[] = sortedShots.map((shot) => ({
    shot,
    nx: (shot.centerX - xStats.mean) / xStats.std,
    ny: (shot.centerY - yStats.mean) / yStats.std,
    nt: ((shot.videoTimeSec - tStats.mean) / tStats.std) * effectiveTimeWeight,
  }));

  const dbscanModel = runDbscanClustering(points, tweaks);
  if (dbscanModel.assignments.length !== points.length) {
    return {
      selectedK: 0,
      finalK: 0,
      closeMergeCount: 0,
      objectiveScore: 0,
      shotClusterById: {},
      clusters: [],
    };
  }
  if (dbscanModel.centroids.length === 0) {
    return {
      selectedK: dbscanModel.initialClusterCount,
      finalK: 0,
      closeMergeCount: 0,
      objectiveScore: dbscanModel.objectiveScore,
      shotClusterById: {},
      clusters: [],
    };
  }

  const parents = Array.from({ length: dbscanModel.centroids.length }, (_, index) => index);
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

  for (let i = 0; i < dbscanModel.centroids.length; i += 1) {
    for (let j = i + 1; j < dbscanModel.centroids.length; j += 1) {
      const spatialDistance = Math.hypot(
        dbscanModel.centroids[i].x - dbscanModel.centroids[j].x,
        dbscanModel.centroids[i].y - dbscanModel.centroids[j].y,
      );
      const timeDistance = Math.abs(dbscanModel.centroids[i].t - dbscanModel.centroids[j].t);
      const combinedDistance = euclideanDistance3(
        dbscanModel.centroids[i].x,
        dbscanModel.centroids[i].y,
        dbscanModel.centroids[i].t,
        dbscanModel.centroids[j].x,
        dbscanModel.centroids[j].y,
        dbscanModel.centroids[j].t,
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
  for (let i = 0; i < dbscanModel.assignments.length; i += 1) {
    const sourceCluster = dbscanModel.assignments[i];
    if (sourceCluster < 0) {
      mergedAssignments[i] = -1;
      continue;
    }
    const root = findRoot(sourceCluster);
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
    if (clusterId <= 0) continue;
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
    selectedK: dbscanModel.initialClusterCount,
    finalK: rootToClusterId.size,
    closeMergeCount: Math.max(0, dbscanModel.initialClusterCount - rootToClusterId.size),
    objectiveScore: dbscanModel.objectiveScore,
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

// Color-blind-safe categorical palette (Okabe–Ito + IBM safe hues), used for the
// shot groups when the color-blind option is on. Ordered for max separation.
const CLUSTER_COLOR_PALETTE_CB = [
  "#0072b2",
  "#e69f00",
  "#009e73",
  "#d55e00",
  "#785ef0",
  "#56b4e9",
  "#f0e442",
  "#cc79a7",
  "#999999",
  "#117733",
];

// Module mirror of the persisted color-blind preference so clusterColorForId
// (called from many module-level draw helpers) can pick the palette without
// threading the flag through every call site. Synced from React state on mount
// and whenever the toggle changes.
let clusterPaletteColorBlind = false;

function clusterColorForId(clusterId: number): string {
  const palette = clusterPaletteColorBlind ? CLUSTER_COLOR_PALETTE_CB : CLUSTER_COLOR_PALETTE;
  if (!Number.isFinite(clusterId) || clusterId <= 0) return palette[3];
  return palette[(clusterId - 1) % palette.length];
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
  labelPrefix = "DB",
  labelFor?: (clusterId: number) => string,
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
    context.fillText(labelFor ? labelFor(cluster.clusterId) : `${labelPrefix}${cluster.clusterId}`, labelX - 13, labelY - 10);
  }
}

type QuadPoint = { x: number; y: number; id: string };
type QuadRect = { x: number; y: number; w: number; h: number };

function quadRectContains(rect: QuadRect, px: number, py: number): boolean {
  return px >= rect.x && px < rect.x + rect.w && py >= rect.y && py < rect.y + rect.h;
}

function quadRectsIntersect(a: QuadRect, b: QuadRect): boolean {
  return !(b.x > a.x + a.w || b.x + b.w < a.x || b.y > a.y + a.h || b.y + b.h < a.y);
}

// A point quadtree used to spatially index shots for fast neighbor queries.
class ShotQuadtree {
  private readonly rect: QuadRect;
  private readonly capacity: number;
  private readonly depth: number;
  private readonly maxDepth: number;
  private points: QuadPoint[] = [];
  private divided = false;
  private children: ShotQuadtree[] = [];

  constructor(rect: QuadRect, capacity: number, depth: number, maxDepth: number) {
    this.rect = rect;
    this.capacity = capacity;
    this.depth = depth;
    this.maxDepth = maxDepth;
  }

  insert(point: QuadPoint): boolean {
    if (!quadRectContains(this.rect, point.x, point.y)) return false;
    if (this.points.length < this.capacity || this.depth >= this.maxDepth) {
      this.points.push(point);
      return true;
    }
    if (!this.divided) this.subdivide();
    for (const child of this.children) {
      if (child.insert(point)) return true;
    }
    this.points.push(point); // numeric edge cases: keep at this node
    return true;
  }

  private subdivide() {
    const { x, y, w, h } = this.rect;
    const hw = w / 2;
    const hh = h / 2;
    const nextDepth = this.depth + 1;
    this.children = [
      new ShotQuadtree({ x, y, w: hw, h: hh }, this.capacity, nextDepth, this.maxDepth),
      new ShotQuadtree({ x: x + hw, y, w: hw, h: hh }, this.capacity, nextDepth, this.maxDepth),
      new ShotQuadtree({ x, y: y + hh, w: hw, h: hh }, this.capacity, nextDepth, this.maxDepth),
      new ShotQuadtree({ x: x + hw, y: y + hh, w: hw, h: hh }, this.capacity, nextDepth, this.maxDepth),
    ];
    this.divided = true;
  }

  queryRange(range: QuadRect, found: QuadPoint[]) {
    if (!quadRectsIntersect(this.rect, range)) return;
    for (const point of this.points) {
      if (quadRectContains(range, point.x, point.y)) found.push(point);
    }
    if (this.divided) {
      for (const child of this.children) child.queryRange(range, found);
    }
  }

  collectOccupiedLeaves(out: QuadRect[]) {
    if (this.divided) {
      for (const child of this.children) child.collectOccupiedLeaves(out);
    } else if (this.points.length > 0) {
      out.push(this.rect);
    }
  }
}

type QuadtreeGroupingResult = {
  shotClusterById: Record<string, number>;
  groupCount: number;
  leafRects: QuadRect[];
  radius: number;
};

// Group shots spatially: index them in a quadtree, then single-link union any shots
// within a radius of each other. Groups are numbered by descending size. When a
// baseRadiusPx is supplied (e.g. 6 inches × pixels-per-inch) it drives the grouping
// distance; otherwise an adaptive radius based on shot spread is used.
function groupShotsByQuadtree(
  shots: ShotLogEntry[],
  radiusScale: number,
  baseRadiusPx?: number,
): QuadtreeGroupingResult {
  if (shots.length === 0) return { shotClusterById: {}, groupCount: 0, leafRects: [], radius: 0 };

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const shot of shots) {
    if (shot.centerX < minX) minX = shot.centerX;
    if (shot.centerY < minY) minY = shot.centerY;
    if (shot.centerX > maxX) maxX = shot.centerX;
    if (shot.centerY > maxY) maxY = shot.centerY;
  }
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const pad = Math.max(spanX, spanY) * 0.02 + 1;
  const bounds: QuadRect = { x: minX - pad, y: minY - pad, w: spanX + pad * 2, h: spanY + pad * 2 };

  const tree = new ShotQuadtree(bounds, 4, 0, 12);
  const points: QuadPoint[] = shots.map((shot) => ({ x: shot.centerX, y: shot.centerY, id: shot.id }));
  for (const point of points) tree.insert(point);

  const scale = Math.max(0.1, radiusScale);
  const radius =
    baseRadiusPx && baseRadiusPx > 0
      ? Math.max(2, baseRadiusPx * scale)
      : Math.max(2, Math.hypot(spanX, spanY) * 0.06 * scale);

  const indexById = new Map<string, number>();
  points.forEach((point, index) => indexById.set(point.id, index));
  const parent = points.map((_, index) => index);
  const findRoot = (i: number): number => {
    let root = i;
    while (parent[root] !== root) root = parent[root];
    let cursor = i;
    while (parent[cursor] !== cursor) {
      const next = parent[cursor];
      parent[cursor] = root;
      cursor = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const rootA = findRoot(a);
    const rootB = findRoot(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  };

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const neighbors: QuadPoint[] = [];
    tree.queryRange({ x: point.x - radius, y: point.y - radius, w: radius * 2, h: radius * 2 }, neighbors);
    for (const neighbor of neighbors) {
      if (neighbor.id === point.id) continue;
      if (Math.hypot(point.x - neighbor.x, point.y - neighbor.y) <= radius) {
        const neighborIndex = indexById.get(neighbor.id);
        if (neighborIndex !== undefined) union(i, neighborIndex);
      }
    }
  }

  const members = new Map<number, string[]>();
  for (let i = 0; i < points.length; i += 1) {
    const root = findRoot(i);
    const bucket = members.get(root) ?? [];
    bucket.push(points[i].id);
    members.set(root, bucket);
  }
  const ordered = [...members.entries()].sort((a, b) => b[1].length - a[1].length);
  const shotClusterById: Record<string, number> = {};
  ordered.forEach(([, ids], groupIndex) => {
    for (const id of ids) shotClusterById[id] = groupIndex + 1;
  });

  const leafRects: QuadRect[] = [];
  tree.collectOccupiedLeaves(leafRects);

  return { shotClusterById, groupCount: ordered.length, leafRects, radius };
}

function windowIndexAtTime(timeSec: number, windows: TimeWindow[]): number {
  for (let i = 0; i < windows.length; i += 1) {
    if (timeSec >= windows[i].start && timeSec <= windows[i].end) return i;
  }
  return -1;
}

function buildContourRegionClusterVisuals(
  snapshot: ContourWindowFrameSnapshot,
  sourceRect: { x: number; y: number; width: number; height: number },
  shots: ShotLogEntry[],
  shotClusterById: Record<string, number>,
  clusterColorById: Record<number, string>,
): { regionColors: string[]; regionGroupLabels: string[] } {
  const safeSourceWidth = Math.max(1, sourceRect.width);
  const safeSourceHeight = Math.max(1, sourceRect.height);
  const regionColors: string[] = [];
  const regionGroupLabels: string[] = [];
  const visibleRegions = snapshot.regions.slice(0, 10);

  for (let i = 0; i < visibleRegions.length; i += 1) {
    const region = visibleRegions[i];
    let nearestShot: ShotLogEntry | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;

    for (const shot of shots) {
      const patchShotX = ((shot.centerX - sourceRect.x) / safeSourceWidth) * snapshot.patchWidthPx;
      const patchShotY = ((shot.centerY - sourceRect.y) / safeSourceHeight) * snapshot.patchHeightPx;
      const distance = Math.hypot(region.centerX - patchShotX, region.centerY - patchShotY);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestShot = shot;
      }
    }

    const clusterId = nearestShot ? shotClusterById[nearestShot.id] : undefined;
    if (clusterId !== undefined && Number.isFinite(clusterId) && clusterId > 0) {
      const normalizedClusterId = Math.round(clusterId);
      regionColors.push(clusterColorById[normalizedClusterId] ?? clusterColorForId(normalizedClusterId));
      regionGroupLabels.push(`DB${normalizedClusterId}`);
      continue;
    }

    regionColors.push(clusterColorForId(i + 1));
    regionGroupLabels.push(`R${i + 1}`);
  }

  return { regionColors, regionGroupLabels };
}

function drawContourRegionWindowOverlays(
  context: CanvasRenderingContext2D,
  regions: ContourRegionOverlay[],
  regionColors?: string[],
  regionGroupLabels?: string[],
): void {
  const maxRegionsToDraw = 10;
  const visibleRegions = regions.slice(0, maxRegionsToDraw);
  for (let i = 0; i < visibleRegions.length; i += 1) {
    const region = visibleRegions[i];
    const rectWidth = Math.max(1, region.maxX - region.minX + 1);
    const rectHeight = Math.max(1, region.maxY - region.minY + 1);
    const radius = Math.max(2, Math.round(Math.max(rectWidth, rectHeight) / 2));
    const color = regionColors?.[i] ?? clusterColorForId(i + 1);
    const groupLabel = regionGroupLabels?.[i] ?? `${i + 1}`;
    context.strokeStyle = hexToRgba(color, 0.95);
    context.lineWidth = 2;
    context.beginPath();
    context.arc(region.centerX, region.centerY, radius, 0, Math.PI * 2);
    context.stroke();
    drawCenterCross(
      context,
      region.centerX,
      region.centerY,
      3,
      "rgba(255, 255, 255, 0.95)",
      1.25,
    );

    context.font = "10px sans-serif";
    const textWidth = context.measureText(groupLabel).width;
    const labelX = Math.max(0, Math.min(region.centerX - 10, context.canvas.width - textWidth - 4));
    const labelY = Math.max(10, Math.min(region.centerY + 10, context.canvas.height - 2));
    context.fillStyle = "rgba(0, 0, 0, 0.72)";
    context.fillRect(labelX - 2, labelY - 9, textWidth + 4, 12);
    context.fillStyle = hexToRgba(color, 0.98);
    context.fillText(groupLabel, labelX, labelY);
  }
}

function drawProcessedContourView(
  context: CanvasRenderingContext2D,
  patchRgba: Uint8ClampedArray,
  baselineRgba: Uint8ClampedArray | null,
  patchWidth: number,
  patchHeight: number,
  mask: Uint8Array | null,
  regions: ChangedContourRegion[],
  regionColors?: string[],
  regionGroupLabels?: string[],
): void {
  let changedCount = 0;
  const hasBaseline = !!baselineRgba && baselineRgba.length === patchRgba.length;
  for (let i = 0; i < patchWidth * patchHeight; i += 1) {
    if (mask && mask[i] === 1) changedCount += 1;
  }

  const imageData = context.createImageData(patchWidth, patchHeight);
  for (let i = 0; i < patchWidth * patchHeight; i += 1) {
    const rgbaIndex = i * 4;
    const changed = mask ? mask[i] === 1 : false;

    if (changed && changedCount > 0 && hasBaseline && baselineRgba) {
      // Difference image: suppress baseline content and amplify only changed pixels.
      const deltaR = Math.abs(patchRgba[rgbaIndex] - baselineRgba[rgbaIndex]);
      const deltaG = Math.abs(patchRgba[rgbaIndex + 1] - baselineRgba[rgbaIndex + 1]);
      const deltaB = Math.abs(patchRgba[rgbaIndex + 2] - baselineRgba[rgbaIndex + 2]);
      const gain = 6;
      imageData.data[rgbaIndex] = Math.min(255, deltaR * gain);
      imageData.data[rgbaIndex + 1] = Math.min(255, deltaG * gain);
      imageData.data[rgbaIndex + 2] = Math.min(255, deltaB * gain);
    } else {
      const baseGain = hasBaseline ? 0.22 : 0.32;
      imageData.data[rgbaIndex] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex] * baseGain)));
      imageData.data[rgbaIndex + 1] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex + 1] * baseGain)));
      imageData.data[rgbaIndex + 2] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex + 2] * baseGain)));
    }
    imageData.data[rgbaIndex + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);

  drawContourRegionWindowOverlays(context, regions, regionColors, regionGroupLabels);

  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(280, patchWidth), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  const modeLabel = hasBaseline ? "Contour mask vs original" : "Contour mask";
  context.fillText(`${modeLabel} | changed=${changedCount} | regions=${regions.length}`, 4, 11);
}

function drawPatchWindowView(
  context: CanvasRenderingContext2D,
  source: CanvasImageSource,
  sourceRect: { x: number; y: number; width: number; height: number },
): void {
  const safeX = Math.max(0, Math.floor(sourceRect.x));
  const safeY = Math.max(0, Math.floor(sourceRect.y));
  const safeWidth = Math.max(1, Math.round(sourceRect.width));
  const safeHeight = Math.max(1, Math.round(sourceRect.height));
  if (context.canvas.width !== safeWidth || context.canvas.height !== safeHeight) {
    context.canvas.width = safeWidth;
    context.canvas.height = safeHeight;
  }
  context.drawImage(
    source,
    safeX,
    safeY,
    safeWidth,
    safeHeight,
    0,
    0,
    safeWidth,
    safeHeight,
  );
}

function drawProbePatchOverlayView(
  context: CanvasRenderingContext2D,
  patchWidth: number,
  patchHeight: number,
  overlayLabel: string,
  blinkOn: boolean,
  snapshot: ContourWindowFrameSnapshot | null,
  regionColors: string[] | undefined,
  pixelsPerInch: number,
  formatLinearFromInches: (valueInches: number, fractionDigits?: number) => string,
  regionGroupLabels?: string[],
): void {
  const safeWidth = Math.max(1, Math.round(patchWidth));
  const safeHeight = Math.max(1, Math.round(patchHeight));
  const boundaryColor = blinkOn ? "#ffffff" : "#000000";
  const labelWidth = Math.min(safeWidth, Math.max(120, Math.min(420, 14 + overlayLabel.length * 6)));

  context.strokeStyle = boundaryColor;
  context.lineWidth = 3;
  context.strokeRect(0, 0, safeWidth, safeHeight);
  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, labelWidth, 18);
  context.fillStyle = blinkOn ? "#ffffff" : "#d4d4d4";
  context.font = "12px sans-serif";
  context.fillText(overlayLabel, 6, 12);

  if (!snapshot || snapshot.regions.length === 0) return;
  drawContourRegionsOnTargetView(
    context,
    snapshot,
    { x: 0, y: 0, width: safeWidth, height: safeHeight },
    blinkOn,
    regionColors,
    pixelsPerInch,
    formatLinearFromInches,
    regionGroupLabels,
  );
}

function drawBinaryMaskWindowView(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  mask: Uint8Array | null,
  title: string,
  patchRgba?: Uint8ClampedArray | null,
): void {
  const safeWidth = Math.max(1, Math.round(width));
  const safeHeight = Math.max(1, Math.round(height));
  if (context.canvas.width !== safeWidth || context.canvas.height !== safeHeight) {
    context.canvas.width = safeWidth;
    context.canvas.height = safeHeight;
  }
  const imageData = context.createImageData(safeWidth, safeHeight);
  let changed = 0;
  for (let i = 0; i < safeWidth * safeHeight; i += 1) {
    const rgbaIndex = i * 4;
    const isChanged = mask ? mask[i] === 1 : false;
    if (isChanged) changed += 1;
    if (isChanged) {
      imageData.data[rgbaIndex] = 255;
      imageData.data[rgbaIndex + 1] = 255;
      imageData.data[rgbaIndex + 2] = 255;
    } else if (patchRgba && patchRgba.length >= safeWidth * safeHeight * 4) {
      const gray =
        0.299 * patchRgba[rgbaIndex] +
        0.587 * patchRgba[rgbaIndex + 1] +
        0.114 * patchRgba[rgbaIndex + 2];
      const dim = Math.max(10, Math.min(180, Math.round(gray * 0.35)));
      imageData.data[rgbaIndex] = dim;
      imageData.data[rgbaIndex + 1] = dim;
      imageData.data[rgbaIndex + 2] = dim;
    } else {
      imageData.data[rgbaIndex] = 18;
      imageData.data[rgbaIndex + 1] = 18;
      imageData.data[rgbaIndex + 2] = 18;
    }
    imageData.data[rgbaIndex + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(280, safeWidth), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  context.fillText(`${title} | changed=${changed}`, 4, 11);
}

function drawBinaryMaskSnapshotWindowView(
  context: CanvasRenderingContext2D,
  snapshot: ContourWindowFrameSnapshot,
  title: string,
): void {
  const width = Math.max(1, Math.round(snapshot.patchWidthPx));
  const height = Math.max(1, Math.round(snapshot.patchHeightPx));
  const mask = decodeMaskRuns(width, height, snapshot.maskRuns);
  drawBinaryMaskWindowView(context, width, height, mask, title);
}

function drawYellowGreenWindowView(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  mask: Uint8Array | null,
  hits: YellowGreenHit[],
  title: string,
  patchRgba?: Uint8ClampedArray | null,
): void {
  const safeWidth = Math.max(1, Math.round(width));
  const safeHeight = Math.max(1, Math.round(height));
  if (context.canvas.width !== safeWidth || context.canvas.height !== safeHeight) {
    context.canvas.width = safeWidth;
    context.canvas.height = safeHeight;
  }

  const imageData = context.createImageData(safeWidth, safeHeight);
  let changed = 0;
  for (let i = 0; i < safeWidth * safeHeight; i += 1) {
    const rgbaIndex = i * 4;
    const changedPixel = mask ? mask[i] === 1 : false;
    if (changedPixel) changed += 1;

    if (patchRgba && patchRgba.length >= safeWidth * safeHeight * 4) {
      imageData.data[rgbaIndex] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex] * 0.5)));
      imageData.data[rgbaIndex + 1] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex + 1] * 0.5)));
      imageData.data[rgbaIndex + 2] = Math.max(8, Math.min(255, Math.round(patchRgba[rgbaIndex + 2] * 0.5)));
    } else {
      imageData.data[rgbaIndex] = 16;
      imageData.data[rgbaIndex + 1] = 16;
      imageData.data[rgbaIndex + 2] = 16;
    }

    if (changedPixel) {
      imageData.data[rgbaIndex] = Math.max(imageData.data[rgbaIndex], 235);
      imageData.data[rgbaIndex + 1] = Math.max(imageData.data[rgbaIndex + 1], 215);
      imageData.data[rgbaIndex + 2] = Math.max(imageData.data[rgbaIndex + 2], 55);
    }
    imageData.data[rgbaIndex + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);

  const visibleHits = hits.slice(0, 12);
  for (let i = 0; i < visibleHits.length; i += 1) {
    const hit = visibleHits[i];
    context.strokeStyle = "rgba(34, 197, 94, 0.95)";
    context.lineWidth = 2;
    context.strokeRect(hit.minX, hit.minY, Math.max(1, hit.width), Math.max(1, hit.height));
    drawCenterCross(context, hit.centroidX, hit.centroidY, 3, "rgba(255, 255, 255, 0.95)", 1.25);
    context.fillStyle = "rgba(239, 68, 68, 0.95)";
    context.font = "10px sans-serif";
    context.fillText(`${i + 1}`, Math.max(0, hit.centroidX - 10), Math.max(10, hit.centroidY + 10));
  }

  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(340, safeWidth), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  context.fillText(`${title} | changed=${changed} | hits=${hits.length}`, 4, 11);
}

function drawYellowGreenSnapshotWindowView(
  context: CanvasRenderingContext2D,
  snapshot: YellowGreenFrameSnapshot,
  title: string,
): void {
  const width = Math.max(1, Math.round(snapshot.patchWidthPx));
  const height = Math.max(1, Math.round(snapshot.patchHeightPx));
  const mask = decodeMaskRuns(width, height, snapshot.maskRuns);
  drawYellowGreenWindowView(context, width, height, mask, snapshot.hits, title);
}

function decodeMaskRuns(
  width: number,
  height: number,
  runs: Array<{ start: number; length: number }>,
): Uint8Array {
  const mask = new Uint8Array(Math.max(0, width * height));
  for (const run of runs) {
    const safeStart = Math.max(0, Math.min(mask.length - 1, Math.floor(run.start)));
    const safeLength = Math.max(0, Math.floor(run.length));
    const endExclusive = Math.min(mask.length, safeStart + safeLength);
    for (let i = safeStart; i < endExclusive; i += 1) {
      mask[i] = 1;
    }
  }
  return mask;
}

function encodeBinaryMaskRuns(mask: Uint8Array): Array<{ start: number; length: number }> {
  const runs: Array<{ start: number; length: number }> = [];
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
  regionColors?: string[],
  regionGroupLabels?: string[],
): void {
  const width = Math.max(1, Math.round(snapshot.patchWidthPx));
  const height = Math.max(1, Math.round(snapshot.patchHeightPx));
  if (context.canvas.width !== width || context.canvas.height !== height) {
    context.canvas.width = width;
    context.canvas.height = height;
  }

  const imageData = context.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const rgbaIndex = i * 4;
    imageData.data[rgbaIndex] = 16;
    imageData.data[rgbaIndex + 1] = 16;
    imageData.data[rgbaIndex + 2] = 16;
    imageData.data[rgbaIndex + 3] = 255;
  }
  for (const run of snapshot.maskRuns) {
    const safeStart = Math.max(0, Math.min(width * height - 1, Math.floor(run.start)));
    const safeLength = Math.max(0, Math.floor(run.length));
    const endExclusive = Math.min(width * height, safeStart + safeLength);
    for (let i = safeStart; i < endExclusive; i += 1) {
      const rgbaIndex = i * 4;
      imageData.data[rgbaIndex] = 30;
      imageData.data[rgbaIndex + 1] = 240;
      imageData.data[rgbaIndex + 2] = 60;
      imageData.data[rgbaIndex + 3] = 255;
    }
  }
  context.putImageData(imageData, 0, 0);

  drawContourRegionWindowOverlays(context, snapshot.regions, regionColors, regionGroupLabels);

  context.fillStyle = "rgba(0, 0, 0, 0.72)";
  context.fillRect(0, 0, Math.min(280, width), 16);
  context.fillStyle = "#e5e7eb";
  context.font = "10px sans-serif";
  context.fillText(
    `Contour playback | changed=${snapshot.changedPixels} | regions=${snapshot.regions.length}`,
    4,
    11,
  );
}

function drawContourRegionsOnTargetView(
  context: CanvasRenderingContext2D,
  snapshot: ContourWindowFrameSnapshot,
  targetRect: { x: number; y: number; width: number; height: number },
  blinkOn: boolean,
  regionColors?: string[],
  pixelsPerInch = 0,
  formatLinearFromInches?: (valueInches: number, fractionDigits?: number) => string,
  regionGroupLabels?: string[],
): void {
  const patchWidth = Math.max(1, snapshot.patchWidthPx);
  const patchHeight = Math.max(1, snapshot.patchHeightPx);
  const scaleX = targetRect.width / patchWidth;
  const scaleY = targetRect.height / patchHeight;
  const scale = Math.max(scaleX, scaleY);
  const visibleRegions = snapshot.regions.slice(0, 10);

  for (let i = 0; i < visibleRegions.length; i += 1) {
    const region = visibleRegions[i];
    const rectWidth = Math.max(1, region.maxX - region.minX + 1);
    const rectHeight = Math.max(1, region.maxY - region.minY + 1);
    const { diameterPx, diameterInches } = estimatePatchBlobDiameter(
      region.pixelCount,
      rectWidth,
      rectHeight,
      patchWidth,
      patchHeight,
      targetRect.width,
      targetRect.height,
      pixelsPerInch,
    );
    const radiusPatch = Math.max(2, Math.round(Math.max(rectWidth, rectHeight) / 2));
    const centerX = targetRect.x + ((region.centerX + 0.5) / patchWidth) * targetRect.width;
    const centerY = targetRect.y + ((region.centerY + 0.5) / patchHeight) * targetRect.height;
    const radius = Math.max(2, radiusPatch * scale);
    const color = regionColors?.[i] ?? clusterColorForId(i + 1);

    context.beginPath();
    context.arc(centerX, centerY, radius, 0, Math.PI * 2);
    context.fillStyle = hexToRgba(color, 0.25);
    context.fill();
    context.strokeStyle = hexToRgba(color, 0.95);
    context.lineWidth = 1;
    context.stroke();

    drawCenterCross(
      context,
      centerX,
      centerY,
      Math.max(3, Math.min(8, radius * 0.35)),
      blinkOn ? "rgba(255, 255, 255, 0.95)" : "rgba(0, 0, 0, 0.95)",
      1.25,
    );

    const groupLabelPrefix = regionGroupLabels?.[i] ?? `R${i + 1}`;
    const sizeLabel =
      diameterInches !== null && formatLinearFromInches
        ? `${groupLabelPrefix} D${formatLinearFromInches(diameterInches, 2)}`
        : `${groupLabelPrefix} D${diameterPx.toFixed(1)} px`;
    context.font = "10px sans-serif";
    const textWidth = context.measureText(sizeLabel).width;
    const labelPaddingX = 4;
    const labelHeight = 12;
    const rawLabelX = centerX + radius + 4;
    const rawLabelY = centerY - radius - labelHeight - 2;
    const labelX = Math.max(
      targetRect.x + 1,
      Math.min(rawLabelX, targetRect.x + targetRect.width - (textWidth + labelPaddingX * 2) - 1),
    );
    const labelY = Math.max(
      targetRect.y + 1,
      Math.min(rawLabelY, targetRect.y + targetRect.height - labelHeight - 1),
    );
    context.fillStyle = "rgba(0, 0, 0, 0.72)";
    context.fillRect(labelX, labelY, textWidth + labelPaddingX * 2, labelHeight);
    context.fillStyle = hexToRgba(color, 0.98);
    context.fillText(sizeLabel, labelX + labelPaddingX, labelY + 9);
  }
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

function clearAnalysisVideoCache() {
  if (typeof window === "undefined") return;
  sessionStorage.removeItem(ANALYSIS_VIDEO_DATA_URL_KEY);
  sessionStorage.removeItem(ANALYSIS_VIDEO_META_KEY);
}

function buildAnalysisVideoCacheKey(file: File | null, videoName: string | null): string | null {
  if (file) {
    return `${file.name}|${file.size}|${file.lastModified}|${file.type || "video/unknown"}`;
  }
  if (videoName) return `${videoName}|unknown`;
  return null;
}

// Grab the first frame of a video URL as a PNG data URL using a detached video
// element. Robust against the visible preview being hidden / not yet loaded
// (used for the preloaded sample so step 2 always has a reference image).
function captureFirstFrameDataUrl(url: string): Promise<string | null> {
  if (typeof document === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    (video as HTMLVideoElement & { playsInline?: boolean }).playsInline = true;
    let settled = false;
    const finish = (result: string | null) => {
      if (settled) return;
      settled = true;
      video.removeAttribute("src");
      try {
        video.load();
      } catch {
        /* ignore */
      }
      resolve(result);
    };
    const grab = () => {
      if (video.videoWidth <= 0 || video.videoHeight <= 0) {
        finish(null);
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        finish(null);
        return;
      }
      ctx.drawImage(video, 0, 0);
      try {
        finish(canvas.toDataURL("image/png"));
      } catch {
        finish(null);
      }
    };
    video.onloadeddata = () => {
      // At HAVE_CURRENT_DATA the frame is decoded, so drawImage works directly.
      // (Don't wait on requestVideoFrameCallback — it never fires for a detached,
      // never-played video and would hang until the timeout.)
      const run = () => requestAnimationFrame(() => grab());
      if (video.currentTime > 0.05) {
        video.onseeked = run;
        try {
          video.currentTime = 0;
        } catch {
          run();
        }
      } else {
        run();
      }
    };
    video.onerror = () => finish(null);
    window.setTimeout(() => finish(null), 8000);
    video.src = url;
  });
}

// Per-video saved profile: the captured reference frame plus the target
// calibration/geometry, keyed by the video's identity. Persisted to
// localStorage so re-uploading the same clip can restore prior setup.
const VIDEO_PROFILE_PREFIX = "trackr-vprofile:";

type VideoProfile = {
  imageName: string | null;
  imageDataUrl: string;
  roiRect: RoiRect | null;
  pixelsPerInch: number;
  focalScalePxIn: number;
  calibrationDistanceInches: number;
  targetWidthInches: number;
  targetHeightInches: number;
  expectedHoleDiameterInches: number;
  savedAt: number;
};

function loadVideoProfile(key: string | null): VideoProfile | null {
  if (!key || typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(VIDEO_PROFILE_PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<VideoProfile>;
    if (!parsed || typeof parsed.imageDataUrl !== "string" || !parsed.imageDataUrl.startsWith("data:")) return null;
    return parsed as VideoProfile;
  } catch {
    return null;
  }
}

function saveVideoProfile(key: string, profile: VideoProfile): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(VIDEO_PROFILE_PREFIX + key, JSON.stringify(profile));
  } catch {
    // Quota exceeded (the reference frame can be large) — skip silently.
  }
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error("Failed to encode video for session playback."));
      }
    };
    reader.onerror = () => reject(new Error("Failed reading video file."));
    reader.readAsDataURL(file);
  });
}

function readStoredAnalysisVideoMeta(): StoredAnalysisVideoMeta | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(ANALYSIS_VIDEO_META_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredAnalysisVideoMeta>;
    if (
      !parsed ||
      typeof parsed.key !== "string" ||
      typeof parsed.name !== "string" ||
      typeof parsed.size !== "number" ||
      typeof parsed.lastModified !== "number" ||
      typeof parsed.type !== "string"
    ) {
      return null;
    }
    return parsed as StoredAnalysisVideoMeta;
  } catch {
    return null;
  }
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

function closeBinaryMask(mask: Uint8Array, width: number, height: number, kernelSize = 3): Uint8Array {
  const inverted = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) {
    inverted[i] = mask[i] === 1 ? 0 : 1;
  }
  const openedInverted = openBinaryMask(inverted, width, height, kernelSize);
  const closed = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) {
    closed[i] = openedInverted[i] === 1 ? 0 : 1;
  }
  return closed;
}

function rgbaToGrayChannel(rgba: Uint8ClampedArray, width: number, height: number): Uint8Array {
  const pixelCount = width * height;
  const gray = new Uint8Array(pixelCount);
  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const value = 0.299 * rgba[idx] + 0.587 * rgba[idx + 1] + 0.114 * rgba[idx + 2];
    gray[i] = Math.max(0, Math.min(255, Math.round(value)));
  }
  return gray;
}

function buildGrayDifferenceMask(currentGray: Uint8Array, referenceGray: Uint8Array, threshold: number): Uint8Array {
  const size = Math.min(currentGray.length, referenceGray.length);
  const mask = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    if (Math.abs(currentGray[i] - referenceGray[i]) >= threshold) mask[i] = 1;
  }
  return mask;
}

function mergeBinaryMasks(masks: Uint8Array[]): Uint8Array {
  if (masks.length === 0) return new Uint8Array(0);
  const size = masks.reduce((min, mask) => Math.min(min, mask.length), masks[0].length);
  const merged = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < masks.length; j += 1) {
      if (masks[j][i] === 1) {
        merged[i] = 1;
        break;
      }
    }
  }
  return merged;
}

const diskOffsetsCache = new Map<number, Array<{ dx: number; dy: number }>>();

function getDiskOffsets(radius: number): Array<{ dx: number; dy: number }> {
  const safeRadius = Math.max(1, Math.round(radius));
  const cached = diskOffsetsCache.get(safeRadius);
  if (cached) return cached;
  const offsets: Array<{ dx: number; dy: number }> = [];
  const radiusSq = safeRadius * safeRadius;
  for (let dy = -safeRadius; dy <= safeRadius; dy += 1) {
    for (let dx = -safeRadius; dx <= safeRadius; dx += 1) {
      if (dx * dx + dy * dy <= radiusSq) offsets.push({ dx, dy });
    }
  }
  diskOffsetsCache.set(safeRadius, offsets);
  return offsets;
}

function rgbToOpenCvHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const cMax = Math.max(rn, gn, bn);
  const cMin = Math.min(rn, gn, bn);
  const delta = cMax - cMin;

  let hueDeg = 0;
  if (delta > 0) {
    if (cMax === rn) {
      hueDeg = 60 * (((gn - bn) / delta) % 6);
    } else if (cMax === gn) {
      hueDeg = 60 * (((bn - rn) / delta) + 2);
    } else {
      hueDeg = 60 * (((rn - gn) / delta) + 4);
    }
  }
  if (!Number.isFinite(hueDeg)) hueDeg = 0;
  if (hueDeg < 0) hueDeg += 360;

  const saturation = cMax <= 0 ? 0 : (delta / cMax) * 255;
  const value = cMax * 255;
  return {
    h: Math.max(0, Math.min(179, Math.round(hueDeg / 2))),
    s: Math.max(0, Math.min(255, Math.round(saturation))),
    v: Math.max(0, Math.min(255, Math.round(value))),
  };
}

function gaussianBlur5x5Channel(channel: Uint8Array, width: number, height: number): Uint8Array {
  const kernel = [1, 4, 6, 4, 1];
  const temp = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      let weighted = 0;
      for (let k = -2; k <= 2; k += 1) {
        const sx = Math.max(0, Math.min(width - 1, x + k));
        weighted += channel[rowOffset + sx] * kernel[k + 2];
      }
      temp[rowOffset + x] = weighted / 16;
    }
  }

  const blurred = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let weighted = 0;
      for (let k = -2; k <= 2; k += 1) {
        const sy = Math.max(0, Math.min(height - 1, y + k));
        weighted += temp[sy * width + x] * kernel[k + 2];
      }
      blurred[y * width + x] = Math.max(0, Math.min(255, Math.round(weighted / 16)));
    }
  }

  return blurred;
}

function erodeGrayWithOffsets(
  channel: Uint8Array,
  width: number,
  height: number,
  offsets: Array<{ dx: number; dy: number }>,
): Uint8Array {
  const eroded = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let minValue = 255;
      for (const offset of offsets) {
        const sx = Math.max(0, Math.min(width - 1, x + offset.dx));
        const sy = Math.max(0, Math.min(height - 1, y + offset.dy));
        const value = channel[sy * width + sx];
        if (value < minValue) minValue = value;
      }
      eroded[y * width + x] = minValue;
    }
  }
  return eroded;
}

function dilateGrayWithOffsets(
  channel: Uint8Array,
  width: number,
  height: number,
  offsets: Array<{ dx: number; dy: number }>,
): Uint8Array {
  const dilated = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let maxValue = 0;
      for (const offset of offsets) {
        const sx = Math.max(0, Math.min(width - 1, x + offset.dx));
        const sy = Math.max(0, Math.min(height - 1, y + offset.dy));
        const value = channel[sy * width + sx];
        if (value > maxValue) maxValue = value;
      }
      dilated[y * width + x] = maxValue;
    }
  }
  return dilated;
}

function detectHitsBrightYellowGreen(
  patchRgba: Uint8ClampedArray,
  width: number,
  height: number,
): { mask: Uint8Array; hits: YellowGreenHit[] } {
  const pixelCount = width * height;
  const empty = { mask: new Uint8Array(pixelCount), hits: [] as YellowGreenHit[] };
  if (pixelCount <= 0 || patchRgba.length < pixelCount * 4) return empty;

  const hue = new Uint8Array(pixelCount);
  const sat = new Uint8Array(pixelCount);
  const val = new Uint8Array(pixelCount);
  const colorMask = new Uint8Array(pixelCount);

  for (let i = 0; i < pixelCount; i += 1) {
    const rgbaIndex = i * 4;
    const hsv = rgbToOpenCvHsv(
      patchRgba[rgbaIndex],
      patchRgba[rgbaIndex + 1],
      patchRgba[rgbaIndex + 2],
    );
    hue[i] = hsv.h;
    sat[i] = hsv.s;
    val[i] = hsv.v;
    if (
      hsv.h >= YELLOW_GREEN_HUE_MIN &&
      hsv.h <= YELLOW_GREEN_HUE_MAX &&
      hsv.s >= YELLOW_GREEN_SAT_MIN &&
      hsv.v >= YELLOW_GREEN_VAL_MIN
    ) {
      colorMask[i] = 1;
    }
  }

  const blurredV = gaussianBlur5x5Channel(val, width, height);
  const topHatOffsets = getDiskOffsets(YELLOW_GREEN_TOPHAT_RADIUS);
  const openedV = dilateGrayWithOffsets(
    erodeGrayWithOffsets(blurredV, width, height, topHatOffsets),
    width,
    height,
    topHatOffsets,
  );

  const brightMask = new Uint8Array(pixelCount);
  for (let i = 0; i < pixelCount; i += 1) {
    if (blurredV[i] - openedV[i] >= YELLOW_GREEN_TOPHAT_THRESHOLD) brightMask[i] = 1;
  }

  const mask = new Uint8Array(pixelCount);
  for (let i = 0; i < pixelCount; i += 1) {
    if (colorMask[i] === 1 && brightMask[i] === 1) mask[i] = 1;
  }

  const openedMask = openBinaryMask(mask, width, height, 3);
  const cleanedMask = closeBinaryMask(openedMask, width, height, 5);

  const hits: YellowGreenHit[] = [];
  const components = findBlobs(cleanedMask, width, height, 1);
  for (const component of components) {
    const area = component.pixelCount;
    if (area < YELLOW_GREEN_COMPONENT_MIN_AREA || area > YELLOW_GREEN_COMPONENT_MAX_AREA) continue;

    const boxWidth = Math.max(1, component.maxX - component.minX + 1);
    const boxHeight = Math.max(1, component.maxY - component.minY + 1);
    const aspectRatio = boxWidth / boxHeight;
    if (aspectRatio < YELLOW_GREEN_COMPONENT_ASPECT_MIN || aspectRatio > YELLOW_GREEN_COMPONENT_ASPECT_MAX) continue;

    const extent = area / (boxWidth * boxHeight);
    if (extent < YELLOW_GREEN_COMPONENT_EXTENT_MIN) continue;

    let totalV = 0;
    for (const idx of component.indices) {
      totalV += val[idx];
    }
    const meanV = totalV / Math.max(1, component.indices.length);
    if (meanV < YELLOW_GREEN_COMPONENT_MEAN_V_MIN) continue;

    hits.push({
      minX: component.minX,
      minY: component.minY,
      maxX: component.maxX,
      maxY: component.maxY,
      width: boxWidth,
      height: boxHeight,
      area,
      centroidX: component.centerX,
      centroidY: component.centerY,
      meanV,
    });
  }

  hits.sort((a, b) => b.area - a.area);
  return { mask: cleanedMask, hits };
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


// Per-pixel frame-to-frame change: |frame[t] - frame[t-1]| >= threshold. A real
// impact produces a sharp jump here; slow drift/lighting does not.


// Find the integer (dx, dy) that best aligns the baseline gray patch to the current
// gray patch (minimizes mean abs difference over a strided interior sample). A small
// regularizer prefers no shift on ties. Cheap and dependency-free.
function estimatePatchShift(
  currentGray: Uint8Array,
  baselineGray: Uint8Array,
  width: number,
  height: number,
  maxShift: number,
  stride: number,
): { dx: number; dy: number } {
  if (width <= maxShift * 2 + 2 || height <= maxShift * 2 + 2 || currentGray.length !== baselineGray.length) {
    return { dx: 0, dy: 0 };
  }
  let bestDx = 0;
  let bestDy = 0;
  let bestScore = Infinity;
  const step = Math.max(1, stride);
  for (let dy = -maxShift; dy <= maxShift; dy += 1) {
    for (let dx = -maxShift; dx <= maxShift; dx += 1) {
      let sad = 0;
      let count = 0;
      for (let y = maxShift; y < height - maxShift; y += step) {
        const cRow = y * width;
        const bRow = (y - dy) * width;
        for (let x = maxShift; x < width - maxShift; x += step) {
          const c = currentGray[cRow + x];
          const b = baselineGray[bRow + (x - dx)];
          sad += c > b ? c - b : b - c;
          count += 1;
        }
      }
      if (count === 0) continue;
      const score = sad / count + (Math.abs(dx) + Math.abs(dy)) * 0.001;
      if (score < bestScore) {
        bestScore = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }
  return { dx: bestDx, dy: bestDy };
}

// Shift a single-channel patch by (dx, dy); exposed borders replicate the unshifted
// value so the diff there stays ~0 instead of producing edge artifacts.
function shiftGrayPatch(src: Uint8Array, width: number, height: number, dx: number, dy: number): Uint8Array {
  if (dx === 0 && dy === 0) return src;
  const out = new Uint8Array(src.length);
  for (let y = 0; y < height; y += 1) {
    const outRow = y * width;
    const sy = y - dy;
    const inRange = sy >= 0 && sy < height;
    const srcRow = inRange ? sy * width : outRow;
    for (let x = 0; x < width; x += 1) {
      const sx = x - dx;
      out[outRow + x] = inRange && sx >= 0 && sx < width ? src[srcRow + sx] : src[outRow + x];
    }
  }
  return out;
}

// Shift an RGBA patch by (dx, dy) with the same border-replication rule.
function shiftRgbaPatch(src: Uint8ClampedArray, width: number, height: number, dx: number, dy: number): Uint8ClampedArray {
  if (dx === 0 && dy === 0) return src;
  const out = new Uint8ClampedArray(src.length);
  for (let y = 0; y < height; y += 1) {
    const sy = y - dy;
    const inRange = sy >= 0 && sy < height;
    for (let x = 0; x < width; x += 1) {
      const sx = x - dx;
      const outIdx = (y * width + x) * 4;
      const useShift = inRange && sx >= 0 && sx < width;
      const srcIdx = useShift ? (sy * width + sx) * 4 : outIdx;
      out[outIdx] = src[srcIdx];
      out[outIdx + 1] = src[srcIdx + 1];
      out[outIdx + 2] = src[srcIdx + 2];
      out[outIdx + 3] = src[srcIdx + 3];
    }
  }
  return out;
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


function findChangedRegionsByConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number,
  minPixels: number,
  maxPixels: number,
): ChangedContourRegion[] {
  return findBlobs(mask, width, height, minPixels)
    .filter((blob) => blob.pixelCount <= maxPixels)
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

// Touching-bullet separation: given one merged change region, erode its mask by a
// fraction of the caliber radius. Two touching holes form a figure-8 that erosion
// pinches into TWO blobs; a single hole (even an elongated smear) stays one blob
// or vanishes. Returns the centroid (in patch coords) of each surviving blob — i.e.
// one point per actual hole — so we never invent hits where there's no real
// separate change concentration.
function findTouchingHoleCenters(
  mask: Uint8Array,
  patchWidth: number,
  patchHeight: number,
  region: ChangedContourRegion,
  holeRadiusPx: number,
): { px: number; py: number }[] {
  const erodeR = Math.max(1, Math.round(holeRadiusPx * 0.45));
  const pad = erodeR + 1;
  const x0 = Math.max(0, Math.floor(region.minX) - pad);
  const y0 = Math.max(0, Math.floor(region.minY) - pad);
  const x1 = Math.min(patchWidth - 1, Math.ceil(region.maxX) + pad);
  const y1 = Math.min(patchHeight - 1, Math.ceil(region.maxY) + pad);
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  if (bw <= 2 || bh <= 2) return [];
  const sub = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y += 1) {
    for (let x = 0; x < bw; x += 1) {
      sub[y * bw + x] = mask[(y0 + y) * patchWidth + (x0 + x)];
    }
  }
  const eroded = erodeGrayWithOffsets(sub, bw, bh, getDiskOffsets(erodeR));
  const minBlobPx = Math.max(2, Math.round(erodeR * erodeR));
  return findChangedRegionsByConnectedComponents(eroded, bw, bh, minBlobPx, bw * bh).map((blob) => ({
    px: x0 + blob.centerX,
    py: y0 + blob.centerY,
  }));
}

// Hole-shape screen for baseline-change regions feeding the live commit path:
// a bullet hole is solid, roughly round, and about the caliber's size (same
// criteria as the from-scratch change detector). Anything ragged, elongated,
// or far off the expected size is lighting/motion noise, not a hole.
function isHoleShapedRegion(
  region: ChangedContourRegion,
  expectedDiameterPx: number,
  patchWidth: number,
  patchHeight: number,
): boolean {
  // Absolute floor: below ~3px a blob carries no shape information (sensor
  // noise and 1px edge shifts would pass every ratio test trivially).
  if (region.pixelCount < 3) return false;
  const boxWidth = Math.max(1, region.maxX - region.minX + 1);
  const boxHeight = Math.max(1, region.maxY - region.minY + 1);
  const compactness = clamp01(region.pixelCount / (boxWidth * boxHeight));
  const aspect = Math.max(boxWidth, boxHeight) / Math.min(boxWidth, boxHeight);
  if (compactness < 0.45) return false;
  if (aspect > 2.4) return false;
  if (expectedDiameterPx > 0) {
    const diameterPx = 2 * Math.sqrt(region.pixelCount / Math.PI);
    if (
      diameterPx < expectedDiameterPx * CHANGE_HOLE_MIN_DIAMETER_SCALE ||
      diameterPx > expectedDiameterPx * CHANGE_HOLE_MAX_DIAMETER_SCALE
    ) {
      return false;
    }
  } else {
    // Uncalibrated ceiling: a hole is small relative to the target patch; a
    // compact blob spanning a large fraction of it is a shadow/lighting patch.
    const maxSpanPx = Math.max(8, Math.min(patchWidth, patchHeight) * 0.12);
    if (Math.max(boxWidth, boxHeight) > maxSpanPx) return false;
  }
  return true;
}

// Sub-pixel hole center: weight each masked pixel in the region's box by how
// strongly it darkened versus the baseline. The binary-mask centroid drifts
// with ragged mask edges; the diff-weighted centroid tracks the hole's core.
// Returns null (caller keeps the mask centroid) when the region isn't darker
// than baseline — e.g. daylight showing through the hole.
function refineRegionCenterByDiff(
  region: ChangedContourRegion,
  mask: Uint8Array,
  diffMap: Float32Array,
  width: number,
  height: number,
): { px: number; py: number } | null {
  const minX = Math.max(0, Math.floor(region.minX));
  const maxX = Math.min(width - 1, Math.ceil(region.maxX));
  const minY = Math.max(0, Math.floor(region.minY));
  const maxY = Math.min(height - 1, Math.ceil(region.maxY));
  let weightTotal = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = minY; y <= maxY; y += 1) {
    const rowOffset = y * width;
    for (let x = minX; x <= maxX; x += 1) {
      const index = rowOffset + x;
      if (mask[index] !== 1) continue;
      const weight = diffMap[index] ?? 0;
      if (weight <= 0) continue;
      weightTotal += weight;
      sumX += (x + 0.5) * weight;
      sumY += (y + 0.5) * weight;
    }
  }
  if (weightTotal <= 1e-3) return null;
  const px = sumX / weightTotal;
  const py = sumY / weightTotal;
  // Stay inside the region: a degenerate map shouldn't drag the center off the blob.
  if (px < region.minX || px > region.maxX + 1 || py < region.minY || py > region.maxY + 1) return null;
  return { px, py };
}

// Mean |current − baseline| over a region's masked pixels: how STRONG the
// change is, independent of polarity. Real impacts are high-contrast; gradual
// lighting drift barely clears the mask threshold; unchanged printed graphics
// score near zero.
function regionMeanAbsDiff(
  region: ChangedContourRegion,
  mask: Uint8Array,
  currentGray: Uint8Array,
  baselineGray: Uint8Array,
  width: number,
  height: number,
): number {
  const minX = Math.max(0, Math.floor(region.minX));
  const maxX = Math.min(width - 1, Math.ceil(region.maxX));
  const minY = Math.max(0, Math.floor(region.minY));
  const maxY = Math.min(height - 1, Math.ceil(region.maxY));
  let total = 0;
  let count = 0;
  for (let y = minY; y <= maxY; y += 1) {
    const rowOffset = y * width;
    for (let x = minX; x <= maxX; x += 1) {
      const index = rowOffset + x;
      if (mask[index] !== 1) continue;
      total += Math.abs(currentGray[index] - baselineGray[index]);
      count += 1;
    }
  }
  return count > 0 ? total / count : 0;
}

// ---- Detection-trail visualization ----
// Live, step-by-step view of the pipeline: start frame → stabilized live frame
// → subtraction bit-mask → blobs/tracks → end-frame check, plus the sound
// track. Drawn throttled from the scan loop so users can SEE how each hit is
// found (and why a non-hit isn't).
type TrailInfo = {
  shiftPx: number;
  changedPixels: number;
  changedPct: number;
  blobCount: number;
  acceptedCount: number;
  splatterCount: number;
  pendingTracks: number;
  shotCount: number;
  inWindow: boolean;
  timeSec: number;
  endGateActive: boolean;
};

function ensureTrailContext(
  canvas: HTMLCanvasElement | null,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  if (!canvas || width <= 0 || height <= 0) return null;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas.getContext("2d");
}

function drawTrailGray(canvas: HTMLCanvasElement | null, gray: Uint8Array, width: number, height: number): void {
  const ctx = ensureTrailContext(canvas, width, height);
  if (!ctx) return;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const g = gray[i];
    img.data[i * 4] = g;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawTrailRgba(canvas: HTMLCanvasElement | null, rgba: Uint8ClampedArray, width: number, height: number): void {
  const ctx = ensureTrailContext(canvas, width, height);
  if (!ctx) return;
  const img = ctx.createImageData(width, height);
  img.data.set(rgba.subarray(0, width * height * 4));
  ctx.putImageData(img, 0, 0);
}

// The bit mask, over a dimmed base image: red = pixel differs from baseline.
function drawTrailMask(
  canvas: HTMLCanvasElement | null,
  baseGray: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
): void {
  const ctx = ensureTrailContext(canvas, width, height);
  if (!ctx) return;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    if (mask[i] === 1) {
      img.data[i * 4] = 248;
      img.data[i * 4 + 1] = 60;
      img.data[i * 4 + 2] = 60;
    } else {
      const dim = baseGray[i] * 0.35;
      img.data[i * 4] = dim;
      img.data[i * 4 + 1] = dim;
      img.data[i * 4 + 2] = dim;
    }
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Blob stage: amber box = raw candidate blob, green box = hole-shaped and
// strong enough, cyan box = splatter hit; circles = sighting tracks (gray with
// a count while pending, green once committed as a shot).
function drawTrailBlobs(
  canvas: HTMLCanvasElement | null,
  rgba: Uint8ClampedArray,
  allRegions: ChangedContourRegion[],
  acceptedRegions: ChangedContourRegion[],
  splatterRegions: ChangedContourRegion[],
  tracks: { px: number; py: number; rPatch: number; count: number; committed: boolean }[],
  width: number,
  height: number,
): void {
  const ctx = ensureTrailContext(canvas, width, height);
  if (!ctx) return;
  const img = ctx.createImageData(width, height);
  img.data.set(rgba.subarray(0, width * height * 4));
  ctx.putImageData(img, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(0, 0, width, height);
  ctx.lineWidth = 1;
  const box = (region: ChangedContourRegion, color: string) => {
    ctx.strokeStyle = color;
    ctx.strokeRect(region.minX + 0.5, region.minY + 0.5, Math.max(1, region.maxX - region.minX), Math.max(1, region.maxY - region.minY));
  };
  for (const region of allRegions) box(region, "#f59e0b");
  for (const region of acceptedRegions) box(region, "#34d399");
  for (const region of splatterRegions) box(region, "#22d3ee");
  ctx.font = "9px monospace";
  for (const track of tracks) {
    ctx.beginPath();
    ctx.arc(track.px, track.py, Math.max(3, track.rPatch * 0.6), 0, Math.PI * 2);
    ctx.strokeStyle = track.committed ? "#4ade80" : "#cbd5e1";
    ctx.lineWidth = track.committed ? 2 : 1;
    ctx.stroke();
    if (!track.committed) {
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(String(track.count), track.px + 4, track.py - 4);
    }
  }
}

// Sound track: loudness curve, adaptive spike threshold, detected spikes,
// the 0.5s scan windows, and the playhead.
function drawTrailAudio(
  canvas: HTMLCanvasElement | null,
  samples: AudioRmsSample[],
  meanDbfs: number,
  thresholdDbfs: number,
  spikes: SpikeMetadata[],
  windows: TimeWindow[],
  currentSec: number,
  durationSec: number,
): void {
  const W = 640;
  const H = 96;
  const ctx = ensureTrailContext(canvas, W, H);
  if (!ctx) return;
  ctx.fillStyle = "#09090b";
  ctx.fillRect(0, 0, W, H);
  const span = Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 0;
  if (span <= 0 || samples.length === 0) {
    ctx.fillStyle = "#71717a";
    ctx.font = "11px sans-serif";
    ctx.fillText(samples.length === 0 ? "No audio data (live stream or silent clip) — scanning every frame." : "Waiting for audio…", 8, H / 2);
    return;
  }
  const xAt = (t: number) => (Math.max(0, Math.min(span, t)) / span) * W;
  const DB_MIN = -90;
  const DB_MAX = -5;
  const yAt = (db: number) => H - ((Math.max(DB_MIN, Math.min(DB_MAX, db)) - DB_MIN) / (DB_MAX - DB_MIN)) * (H - 14) - 2;
  // Scan windows (where the detector actually looks).
  ctx.fillStyle = "rgba(56, 189, 248, 0.16)";
  for (const window of windows) {
    if (!Number.isFinite(window.end)) continue;
    ctx.fillRect(xAt(window.start), 0, Math.max(2, xAt(window.end) - xAt(window.start)), H);
  }
  // Loudness curve.
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  samples.forEach((sample, i) => {
    const x = xAt(sample.timeSec);
    const y = yAt(sample.dbfs);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // Mean + adaptive spike threshold.
  ctx.strokeStyle = "#52525b";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, yAt(meanDbfs));
  ctx.lineTo(W, yAt(meanDbfs));
  ctx.stroke();
  ctx.strokeStyle = "#f59e0b";
  ctx.beginPath();
  ctx.moveTo(0, yAt(thresholdDbfs));
  ctx.lineTo(W, yAt(thresholdDbfs));
  ctx.stroke();
  ctx.setLineDash([]);
  // Detected spikes (the bangs).
  ctx.fillStyle = "#ef4444";
  for (const spike of spikes) {
    const x = xAt(spike.timeSec);
    ctx.beginPath();
    ctx.arc(x, yAt(rmsToDbfs(spike.strength)), 3, 0, Math.PI * 2);
    ctx.fill();
  }
  // Playhead.
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1.5;
  const px = xAt(currentSec);
  ctx.beginPath();
  ctx.moveTo(px, 0);
  ctx.lineTo(px, H);
  ctx.stroke();
  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px monospace";
  ctx.fillText(`${currentSec.toFixed(1)}s / ${span.toFixed(1)}s · ${spikes.length} spike${spikes.length === 1 ? "" : "s"}`, 6, 11);
}

// Is there any set pixel within `radius` px of (px, py)? Used against the
// permanent-change mask (first-vs-last background subtraction): a real hole
// must still show change there; a transient never does.
function maskHasChangeNear(
  mask: Uint8Array,
  width: number,
  height: number,
  px: number,
  py: number,
  radius: number,
): boolean {
  const r = Math.max(1, Math.ceil(radius));
  const cx = Math.round(px);
  const cy = Math.round(py);
  for (let dy = -r; dy <= r; dy += 1) {
    const y = cy + dy;
    if (y < 0 || y >= height) continue;
    const rowOffset = y * width;
    for (let dx = -r; dx <= r; dx += 1) {
      const x = cx + dx;
      if (x < 0 || x >= width) continue;
      if (mask[rowOffset + x] === 1) return true;
    }
  }
  return false;
}

// ---- From-scratch hit detector: "persistent localized change" ----
//
// Polarity-agnostic and simple by design. A real bullet hole is a small region
// that (1) changes versus a stable reference — darker OR brighter, we don't care
// which — (2) is roughly round and solid, (3) is about the caliber's size, and
// (4) stays put across several frames. Transient motion/lighting fails (4); big
// reframes fail (2)/(3). Temporal persistence + dedup are handled by the caller
// (a small tracker) so this function stays pure.
// Accepted size band, as multiples of the Expected hole / caliber: a blob counts
// only if its diameter is between 0.4x and 2.6x the expected size. Shared with the
// UI so the on-screen range matches what the detector actually keeps.
const CHANGE_HOLE_MIN_DIAMETER_SCALE = 0.4;
const CHANGE_HOLE_MAX_DIAMETER_SCALE = 2.6;

// Common bullet diameters (in inches — the storage unit). Surfaced next to the
// Expected hole / caliber field, converted to whatever unit is active, so you
// can set the size gate without converting calibers by hand.
const CALIBER_PRESETS: { label: string; inches: number }[] = [
  { label: ".22 LR", inches: 0.223 },
  { label: ".223 / 5.56", inches: 0.224 },
  { label: ".243 / 6mm", inches: 0.243 },
  { label: ".308 / 7.62", inches: 0.308 },
  { label: "9mm", inches: 0.355 },
  { label: ".40 S&W", inches: 0.4 },
  { label: ".45 ACP", inches: 0.452 },
];

type ChangeHoleHit = {
  centerX: number; // patch-pixel coordinates
  centerY: number;
  pixelCount: number;
  diameterPx: number;
  compactness: number;
};

function detectChangeHoles(
  currentGray: Uint8Array,
  baselineGray: Uint8Array,
  width: number,
  height: number,
  opts: { diffThreshold: number; minPixels: number; maxPixels: number; expectedDiameterPx: number },
): ChangeHoleHit[] {
  // |current - baseline| > threshold — both increases and decreases count.
  const changeMask = buildGrayDifferenceMask(currentGray, baselineGray, opts.diffThreshold);
  const opened = openBinaryMask(changeMask, width, height, 3);
  const cleaned = closeBinaryMask(opened, width, height, 3);
  const regions = findChangedRegionsByConnectedComponents(cleaned, width, height, opts.minPixels, opts.maxPixels);

  const hits: ChangeHoleHit[] = [];
  for (const region of regions) {
    const boxWidth = Math.max(1, region.maxX - region.minX + 1);
    const boxHeight = Math.max(1, region.maxY - region.minY + 1);
    const compactness = clamp01(region.pixelCount / (boxWidth * boxHeight)); // 1 = solid blob
    const aspect = Math.max(boxWidth, boxHeight) / Math.min(boxWidth, boxHeight);
    const diameterPx = 2 * Math.sqrt(region.pixelCount / Math.PI);
    if (compactness < 0.45) continue; // ragged → not a hole
    if (aspect > 2.4) continue; // streak/edge → not a hole
    if (opts.expectedDiameterPx > 0) {
      // Keep blobs within the caliber size band (see CHANGE_HOLE_*_DIAMETER_SCALE).
      if (
        diameterPx < opts.expectedDiameterPx * CHANGE_HOLE_MIN_DIAMETER_SCALE ||
        diameterPx > opts.expectedDiameterPx * CHANGE_HOLE_MAX_DIAMETER_SCALE
      )
        continue;
    }
    hits.push({ centerX: region.centerX, centerY: region.centerY, pixelCount: region.pixelCount, diameterPx, compactness });
  }
  return hits;
}

// ---- Change-detector debug instrumentation ----
//
// Mirrors detectChangeHoles but keeps EVERY connected component (down to a low
// floor) and tags why each one is or isn't accepted, plus the binary mask the
// detector actually thresholds. Debug-only — called from the scan loop solely
// when the detailed views are open, so it never costs anything during normal use.
type ChangeHoleDebugReason = "accepted" | "too_small" | "too_large" | "ragged" | "streak" | "off_size";

type ChangeHoleDebugCandidate = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  centerX: number;
  centerY: number;
  pixelCount: number;
  diameterPx: number;
  compactness: number;
  aspect: number;
  reason: ChangeHoleDebugReason;
};

const CHANGE_DEBUG_COLORS: Record<ChangeHoleDebugReason, string> = {
  accepted: "#34d399", // green — kept
  too_small: "#60a5fa", // blue — below min pixels
  too_large: "#a855f7", // purple — above max pixels
  ragged: "#fbbf24", // amber — not solid enough (compactness)
  streak: "#fb923c", // orange — too elongated (aspect)
  off_size: "#f87171", // red — outside the caliber size band
};

function analyzeChangeHoleCandidates(
  currentGray: Uint8Array,
  baselineGray: Uint8Array,
  width: number,
  height: number,
  opts: { diffThreshold: number; minPixels: number; maxPixels: number; expectedDiameterPx: number },
): { mask: Uint8Array; candidates: ChangeHoleDebugCandidate[] } {
  const changeMask = buildGrayDifferenceMask(currentGray, baselineGray, opts.diffThreshold);
  const opened = openBinaryMask(changeMask, width, height, 3);
  const cleaned = closeBinaryMask(opened, width, height, 3);
  // Low floor so even below-min blobs show up (that's the point of debugging).
  const floor = Math.max(2, Math.min(opts.minPixels, 6));
  const candidates = findBlobs(cleaned, width, height, floor).map<ChangeHoleDebugCandidate>((blob) => {
    const boxWidth = Math.max(1, blob.maxX - blob.minX + 1);
    const boxHeight = Math.max(1, blob.maxY - blob.minY + 1);
    const compactness = clamp01(blob.pixelCount / (boxWidth * boxHeight));
    const aspect = Math.max(boxWidth, boxHeight) / Math.min(boxWidth, boxHeight);
    const diameterPx = 2 * Math.sqrt(blob.pixelCount / Math.PI);
    let reason: ChangeHoleDebugReason = "accepted";
    if (blob.pixelCount < opts.minPixels) reason = "too_small";
    else if (blob.pixelCount > opts.maxPixels) reason = "too_large";
    else if (compactness < 0.45) reason = "ragged";
    else if (aspect > 2.4) reason = "streak";
    else if (
      opts.expectedDiameterPx > 0 &&
      (diameterPx < opts.expectedDiameterPx * CHANGE_HOLE_MIN_DIAMETER_SCALE ||
        diameterPx > opts.expectedDiameterPx * CHANGE_HOLE_MAX_DIAMETER_SCALE)
    )
      reason = "off_size";
    return {
      minX: blob.minX,
      minY: blob.minY,
      maxX: blob.maxX,
      maxY: blob.maxY,
      centerX: blob.centerX,
      centerY: blob.centerY,
      pixelCount: blob.pixelCount,
      diameterPx,
      compactness,
      aspect,
      reason,
    };
  });
  return { mask: cleaned, candidates };
}

// Repaint the "mask" and "probe" detail views with the change detector's view:
// the thresholded change mask (red over a dimmed frame) and every candidate blob
// boxed by reject reason, plus the persistence tracker state and live counts.
function drawChangeDetectorDebugViews(
  maskCtx: CanvasRenderingContext2D | null,
  patchCtx: CanvasRenderingContext2D | null,
  gray: Uint8Array,
  width: number,
  height: number,
  mask: Uint8Array,
  candidates: ChangeHoleDebugCandidate[],
  tracks: { x: number; y: number; count: number; committed: boolean }[],
  info: { changedPixels: number; bandMinPx: number; bandMaxPx: number; persistence: number },
): void {
  if (maskCtx) {
    const canvas = maskCtx.canvas;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const img = maskCtx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      if (mask[i]) {
        img.data[i * 4] = 255;
        img.data[i * 4 + 1] = 60;
        img.data[i * 4 + 2] = 60;
      } else {
        const dim = gray[i] * 0.35;
        img.data[i * 4] = dim;
        img.data[i * 4 + 1] = dim;
        img.data[i * 4 + 2] = dim;
      }
      img.data[i * 4 + 3] = 255;
    }
    maskCtx.putImageData(img, 0, 0);
    maskCtx.font = "10px monospace";
    maskCtx.fillStyle = "rgba(0,0,0,0.6)";
    maskCtx.fillRect(0, 0, width, 14);
    maskCtx.fillStyle = "#fff";
    maskCtx.fillText(`change mask · ${info.changedPixels}px changed vs baseline`, 3, 10);
  }

  if (patchCtx) {
    const canvas = patchCtx.canvas;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const img = patchCtx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      const g = gray[i];
      img.data[i * 4] = g;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = g;
      img.data[i * 4 + 3] = 255;
    }
    patchCtx.putImageData(img, 0, 0);

    patchCtx.lineWidth = 1;
    for (const c of candidates) {
      patchCtx.strokeStyle = CHANGE_DEBUG_COLORS[c.reason];
      patchCtx.strokeRect(c.minX + 0.5, c.minY + 0.5, c.maxX - c.minX, c.maxY - c.minY);
    }
    // Persistence tracker: green = committed, amber = persisted enough, gray = building.
    for (const t of tracks) {
      patchCtx.beginPath();
      patchCtx.arc(t.x, t.y, 4, 0, Math.PI * 2);
      patchCtx.strokeStyle = t.committed ? "#34d399" : t.count >= info.persistence ? "#fbbf24" : "#94a3b8";
      patchCtx.lineWidth = t.committed ? 2 : 1;
      patchCtx.stroke();
    }

    const accepted = candidates.filter((c) => c.reason === "accepted").length;
    const band = info.bandMaxPx > 0 ? `${Math.round(info.bandMinPx)}-${Math.round(info.bandMaxPx)}px` : "off";
    patchCtx.font = "9px monospace";
    patchCtx.fillStyle = "rgba(0,0,0,0.6)";
    patchCtx.fillRect(0, 0, width, 13);
    patchCtx.fillStyle = "#fff";
    patchCtx.fillText(`blobs ${candidates.length} · accepted ${accepted} · size band ${band}`, 3, 10);
  }
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

function findAudioSubPeakTimesSec(
  rmsSamples: AudioRmsSample[],
  windowStartSec: number,
  windowEndSec: number,
  peakTimeSec: number,
  peakRms: number,
  meanRms: number,
  stdDevRms: number,
): number[] {
  const windowSamples = rmsSamples.filter((sample) => sample.timeSec >= windowStartSec && sample.timeSec <= windowEndSec);
  if (windowSamples.length < 3) return [];

  const sampleStepSec =
    windowSamples.length >= 2 ? Math.max(1e-4, windowSamples[1].timeSec - windowSamples[0].timeSec) : 0.02;
  const minRmsForSubPeak = Math.max(
    peakRms * AUDIO_SUBPEAK_RELATIVE_THRESHOLD,
    meanRms + stdDevRms * AUDIO_SUBPEAK_STDDEV_FACTOR,
  );
  const subPeakTimesSec: number[] = [];
  for (let i = 1; i < windowSamples.length - 1; i += 1) {
    const prev = windowSamples[i - 1];
    const current = windowSamples[i];
    const next = windowSamples[i + 1];
    const isLocalMax = current.rms >= prev.rms && current.rms >= next.rms;
    if (!isLocalMax) continue;
    if (current.rms < minRmsForSubPeak) continue;
    if (Math.abs(current.timeSec - peakTimeSec) <= sampleStepSec * 1.15) continue;
    subPeakTimesSec.push(current.timeSec);
  }
  return subPeakTimesSec.slice(0, 16);
}

function buildAudioSignatureKey(
  peakDbfs: number,
  subPeakTimesSec: number[],
  windowStartSec: number,
  windowEndSec: number,
): string {
  const subPeakCount = subPeakTimesSec.length;
  const spreadSec =
    subPeakCount <= 1 ? 0 : Math.max(0, subPeakTimesSec[subPeakCount - 1] - subPeakTimesSec[0]);
  const peakBucket = Math.round((peakDbfs + 120) / 3);
  const subPeakBucket = Math.min(8, subPeakCount);
  const spreadBucket = Math.round(spreadSec / 0.06);
  const windowSpanSec = Math.max(1e-3, windowEndSec - windowStartSec);
  const densityBucket = Math.round((subPeakCount / windowSpanSec) * 10);
  return `${peakBucket}:${subPeakBucket}:${spreadBucket}:${densityBucket}`;
}

function buildAudioSignatureCatalog(spikes: SpikeMetadata[]): AudioSignatureCatalogEntry[] {
  const grouped = new Map<number, SpikeMetadata[]>();
  for (const spike of spikes) {
    const bucket = grouped.get(spike.signatureId) ?? [];
    bucket.push(spike);
    grouped.set(spike.signatureId, bucket);
  }

  const catalog: AudioSignatureCatalogEntry[] = [];
  for (const [signatureId, group] of grouped.entries()) {
    const peakDbValues = group.map((spike) => rmsToDbfs(spike.strength));
    const subPeakCounts = group.map((spike) => spike.subPeakTimesSec.length);
    const subPeakSpreads = group.map((spike) =>
      spike.subPeakTimesSec.length <= 1
        ? 0
        : spike.subPeakTimesSec[spike.subPeakTimesSec.length - 1] - spike.subPeakTimesSec[0],
    );
    catalog.push({
      signatureId,
      signatureKey: group[0]?.signatureKey ?? "",
      count: group.length,
      meanPeakDbfs: peakDbValues.reduce((sum, value) => sum + value, 0) / Math.max(1, peakDbValues.length),
      meanSubPeakCount: subPeakCounts.reduce((sum, value) => sum + value, 0) / Math.max(1, subPeakCounts.length),
      meanSubPeakSpreadSec: subPeakSpreads.reduce((sum, value) => sum + value, 0) / Math.max(1, subPeakSpreads.length),
      spikeIds: group.map((spike) => spike.id),
    });
  }

  catalog.sort((a, b) => (b.count === a.count ? a.signatureId - b.signatureId : b.count - a.count));
  return catalog;
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
      signatureCatalog: [] as AudioSignatureCatalogEntry[],
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

  // Peak picking over the RMS envelope: a gunshot is a LOCAL MAXIMUM that (a)
  // clears the adaptive tolerance (mean + k·σ of the whole clip) and (b) has
  // real prominence — it rises well above the quietest local minimum (valley)
  // since the previous accepted spike. Centering the spike on the peak (rather
  // than the first threshold crossing) centers the analysis window on the bang.
  const spikes: SpikeMetadata[] = [];
  let lastAcceptedTimeSec = Number.NEGATIVE_INFINITY;
  let valleyRms = Number.POSITIVE_INFINITY;
  const minProminenceRms = stdDev * AUDIO_SPIKE_MIN_PROMINENCE_STDDEV;

  for (let i = 0; i < rmsSamples.length; i += 1) {
    const sample = rmsSamples[i];
    if (sample.rms < valleyRms) valleyRms = sample.rms;
    const prev = i > 0 ? rmsSamples[i - 1] : undefined;
    const next = i < rmsSamples.length - 1 ? rmsSamples[i + 1] : undefined;
    const isLocalMax =
      (prev === undefined || sample.rms >= prev.rms) && (next === undefined || sample.rms > next.rms);
    if (!isLocalMax) continue;
    if (sample.rms < threshold) continue;
    if (sample.timeSec - lastAcceptedTimeSec < tweaks.audioSpikeMinGapSec) continue;
    // Prominence gate (the first spike is exempt — its valley is the clip start).
    if (spikes.length > 0 && sample.rms - valleyRms < minProminenceRms) continue;

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
      subPeakTimesSec: findAudioSubPeakTimesSec(
        rmsSamples,
        windowStartSec,
        windowEndSec,
        sample.timeSec,
        sample.rms,
        mean,
        stdDev,
      ),
      signatureId: 0,
      signatureKey: "",
    };
    spike.signatureKey = buildAudioSignatureKey(
      rmsToDbfs(spike.strength),
      spike.subPeakTimesSec,
      spike.windowStartSec,
      spike.windowEndSec,
    );

    spikes.push(spike);
    lastAcceptedTimeSec = sample.timeSec;
    // Valley now tracks the dip AFTER this spike, for the next one's prominence.
    valleyRms = sample.rms;
  }

  const signatureGroups = new Map<string, SpikeMetadata[]>();
  for (const spike of spikes) {
    const bucket = signatureGroups.get(spike.signatureKey) ?? [];
    bucket.push(spike);
    signatureGroups.set(spike.signatureKey, bucket);
  }
  const orderedSignatures = [...signatureGroups.entries()].sort((a, b) => {
    if (b[1].length !== a[1].length) return b[1].length - a[1].length;
    return a[0].localeCompare(b[0]);
  });
  const signatureIdByKey = new Map<string, number>();
  for (let i = 0; i < orderedSignatures.length; i += 1) {
    signatureIdByKey.set(orderedSignatures[i][0], i + 1);
  }
  for (const spike of spikes) {
    spike.signatureId = signatureIdByKey.get(spike.signatureKey) ?? 1;
  }

  const spriteMap: Record<string, [number, number]> = {};
  for (const spike of spikes) {
    spriteMap[spike.id] = [spike.spriteStartMs, spike.spriteDurationMs];
  }
  const signatureCatalog = buildAudioSignatureCatalog(spikes);

  return { spikes, spriteMap, signatureCatalog, rmsTimeline: rmsSamples, meanDbfs, thresholdDbfs };
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

type DominantObjectBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  areaFraction: number;
};

// Find the largest plausible object in a frame via Otsu thresholding + contours.
// Runs both polarities so it works for light-on-dark and dark-on-light subjects.
function detectDominantObjectBox(cv: CvApi, canvas: HTMLCanvasElement): DominantObjectBox | null {
  if (
    !cv.findContours ||
    !cv.contourArea ||
    !cv.boundingRect ||
    !cv.threshold ||
    cv.RETR_EXTERNAL == null ||
    cv.CHAIN_APPROX_SIMPLE == null ||
    cv.THRESH_BINARY == null ||
    cv.THRESH_BINARY_INV == null ||
    cv.THRESH_OTSU == null
  ) {
    return null;
  }

  const frameWidth = canvas.width;
  const frameHeight = canvas.height;
  if (frameWidth <= 0 || frameHeight <= 0) return null;
  const frameArea = frameWidth * frameHeight;

  const src = cv.imread(canvas);
  const gray = new cv.Mat();
  const binary = new cv.Mat();
  let best: DominantObjectBox | null = null;

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const evaluatePolarity = (thresholdType: number) => {
      cv.threshold!(gray, binary, 0, 255, thresholdType | cv.THRESH_OTSU!);
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      try {
        cv.findContours!(binary, contours, hierarchy, cv.RETR_EXTERNAL!, cv.CHAIN_APPROX_SIMPLE!);
        const count = contours.size?.() ?? 0;
        for (let i = 0; i < count; i += 1) {
          const contour = contours.get?.(i);
          if (!contour) continue;
          try {
            const fraction = cv.contourArea!(contour, false) / frameArea;
            // Skip noise (too small) and whole-frame/background blobs (too large).
            if (fraction < 0.02 || fraction > 0.85) continue;
            const rect = cv.boundingRect!(contour);
            if (rect.width >= frameWidth * 0.98 && rect.height >= frameHeight * 0.98) continue;
            if (!best || fraction > best.areaFraction) {
              best = {
                x: rect.x / frameWidth,
                y: rect.y / frameHeight,
                width: rect.width / frameWidth,
                height: rect.height / frameHeight,
                areaFraction: fraction,
              };
            }
          } finally {
            contour.delete();
          }
        }
      } finally {
        hierarchy.delete();
        contours.delete();
      }
    };

    evaluatePolarity(cv.THRESH_BINARY!);
    evaluatePolarity(cv.THRESH_BINARY_INV!);
  } finally {
    src.delete();
    gray.delete();
    binary.delete();
  }

  return best;
}

type FrameTemplate = {
  gray: CvMat;
  aspect: number;
  targetWidthInches: number;
  targetHeightInches: number;
};

type TemplateMatchResult = {
  score: number;
  box: { x: number; y: number; width: number; height: number };
  targetWidthInches: number;
  targetHeightInches: number;
};

// Slide each saved target template (at several scales) over a frame and keep the
// strongest normalized-cross-correlation match. Returns a normalized box or null.
function bestTemplateMatchForFrame(
  cv: CvApi,
  frameGray: CvMat,
  frameWidth: number,
  frameHeight: number,
  templates: FrameTemplate[],
  scales: number[],
): TemplateMatchResult | null {
  let best: TemplateMatchResult | null = null;
  for (const template of templates) {
    for (const scale of scales) {
      const width = Math.round(scale * frameWidth);
      const height = Math.round(width / Math.max(template.aspect, 1e-6));
      if (width < 12 || height < 12 || width >= frameWidth || height >= frameHeight) continue;
      const resized = new cv.Mat();
      const result = new cv.Mat();
      try {
        cv.resize(template.gray, resized, new cv.Size(width, height), 0, 0, cv.INTER_AREA);
        cv.matchTemplate(frameGray, resized, result, cv.TM_CCOEFF_NORMED);
        const match = cv.minMaxLoc(result);
        if (!best || match.maxVal > best.score) {
          best = {
            score: match.maxVal,
            box: {
              x: match.maxLoc.x / frameWidth,
              y: match.maxLoc.y / frameHeight,
              width: width / frameWidth,
              height: height / frameHeight,
            },
            targetWidthInches: template.targetWidthInches,
            targetHeightInches: template.targetHeightInches,
          };
        }
      } finally {
        resized.delete();
        result.delete();
      }
    }
  }
  return best;
}

type DominantObjectSample = { timeSec: number; box: DominantObjectBox };

// Cluster per-frame object boxes by center + size, then return the representative
// frame of the largest (most common) cluster — its clearest/closest instance.
function pickMostCommonObjectFrame(samples: DominantObjectSample[]): DominantObjectSample | null {
  if (samples.length === 0) return null;

  const clusters: DominantObjectSample[][] = [];
  for (const sample of samples) {
    const sampleCenterX = sample.box.x + sample.box.width / 2;
    const sampleCenterY = sample.box.y + sample.box.height / 2;
    let placed = false;
    for (const cluster of clusters) {
      const anchor = cluster[0].box;
      const anchorCenterX = anchor.x + anchor.width / 2;
      const anchorCenterY = anchor.y + anchor.height / 2;
      const centerClose = Math.hypot(sampleCenterX - anchorCenterX, sampleCenterY - anchorCenterY) < 0.18;
      const widthRatio = sample.box.width / Math.max(anchor.width, 1e-6);
      const heightRatio = sample.box.height / Math.max(anchor.height, 1e-6);
      const sizeClose = widthRatio > 0.5 && widthRatio < 2 && heightRatio > 0.5 && heightRatio < 2;
      if (centerClose && sizeClose) {
        cluster.push(sample);
        placed = true;
        break;
      }
    }
    if (!placed) clusters.push([sample]);
  }

  clusters.sort((a, b) => b.length - a.length);
  const dominant = clusters[0];
  return dominant.reduce<DominantObjectSample | null>(
    (best, candidate) => (!best || candidate.box.areaFraction > best.box.areaFraction ? candidate : best),
    null,
  );
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

function clampTemplateLocToFrame(
  x: number,
  y: number,
  frameWidth: number,
  frameHeight: number,
  templateWidth: number,
  templateHeight: number,
): { x: number; y: number } {
  const maxX = Math.max(0, frameWidth - Math.max(1, templateWidth));
  const maxY = Math.max(0, frameHeight - Math.max(1, templateHeight));
  return {
    x: Math.max(0, Math.min(Math.round(x), maxX)),
    y: Math.max(0, Math.min(Math.round(y), maxY)),
  };
}

function computeSadTemplateSimilarity(frameGray: CvMat, templateGray: CvMat, locX: number, locY: number): number {
  const templateWidth = Math.max(1, templateGray.cols);
  const templateHeight = Math.max(1, templateGray.rows);
  const frameWidth = Math.max(1, frameGray.cols);
  const frameHeight = Math.max(1, frameGray.rows);
  if (templateWidth > frameWidth || templateHeight > frameHeight) return 0;

  const loc = clampTemplateLocToFrame(locX, locY, frameWidth, frameHeight, templateWidth, templateHeight);
  const templatePixels = templateWidth * templateHeight;
  const stride = templatePixels > 160_000 ? 4 : templatePixels > 70_000 ? 3 : templatePixels > 25_000 ? 2 : 1;
  let absDiffTotal = 0;
  let sampled = 0;

  for (let y = 0; y < templateHeight; y += stride) {
    const frameRow = (loc.y + y) * frameWidth + loc.x;
    const templateRow = y * templateWidth;
    for (let x = 0; x < templateWidth; x += stride) {
      const frameValue = frameGray.data[frameRow + x] ?? 0;
      const templateValue = templateGray.data[templateRow + x] ?? 0;
      absDiffTotal += Math.abs(frameValue - templateValue);
      sampled += 1;
    }
  }

  if (sampled <= 0) return 0;
  const normalizedDiff = absDiffTotal / (255 * sampled);
  return Math.max(0, Math.min(1, 1 - normalizedDiff));
}

function blendTemplateSimilarity(primaryNccScore: number, secondarySadScore: number): number {
  const safePrimary = Math.max(0, Math.min(1, primaryNccScore));
  const safeSecondary = Math.max(0, Math.min(1, secondarySadScore));
  return Math.max(0, Math.min(1, safePrimary * TEMPLATE_PRIMARY_WEIGHT + safeSecondary * TEMPLATE_SECONDARY_WEIGHT));
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

type StartScanOptions = {
  forcedWindow?: TimeWindow;
  forcedSpike?: SpikeMetadata;
};

export default function Home() {
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);
  const [selectedImagePreviewUrl, setSelectedImagePreviewUrl] = useState<string | null>(null);
  const [choosingDifferentFrame, setChoosingDifferentFrame] = useState(false);
  const [selectedVideoName, setSelectedVideoName] = useState<string | null>(null);
  const [selectedVideoPreviewUrl, setSelectedVideoPreviewUrl] = useState<string | null>(null);
  // True once the uploaded video is actually decodable (dimensions + duration
  // known, frame data buffered) — i.e. step 2 can grab a frame from it. Reset
  // whenever a new video / capture mode is chosen so Next re-gates.
  const [videoReady, setVideoReady] = useState(false);
  const [captureMode, setCaptureMode] = useState<"upload" | "stream">("upload");
  const [streamCameraActive, setStreamCameraActive] = useState(false);
  const [streamCameraFacingMode, setStreamCameraFacingMode] = useState<"environment" | "user">("environment");
  const [streamCameraError, setStreamCameraError] = useState<string | null>(null);
  const [opencvReady, setOpenCvReady] = useState(false);
  const [opencvError, setOpenCvError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState("Idle");
  const [detectionEnabled, setDetectionEnabled] = useState(false);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  // Target stabilization: how many pixels of camera shake we're correcting each
  // frame (from baseline-to-frame registration), shown live in the UI.
  const [stabilizationPx, setStabilizationPx] = useState<number | null>(null);
  const lastStabilizationPxRef = useRef(-1);
  // Play the clip's audio aloud during scan/rescan/replay (default on). Toggling
  // updates the live scan/replay video elements immediately.
  const [scanAudioMuted, setScanAudioMuted] = useState(false);
  const toggleScanAudio = () => {
    setScanAudioMuted((prev) => {
      const next = !prev;
      if (scanVideoRef.current) scanVideoRef.current.muted = next || captureMode === "stream";
      if (mapReplayVideoRef.current) mapReplayVideoRef.current.muted = next;
      return next;
    });
  };
  // Playback volume (0–1) shared by the scan and replay audio. A ref mirrors it so
  // the imperative scan/replay start code can read the latest value.
  const [playbackVolume, setPlaybackVolume] = useState(0.8);
  const playbackVolumeRef = useRef(0.8);
  const changePlaybackVolume = (next: number) => {
    const clamped = Math.max(0, Math.min(1, next));
    playbackVolumeRef.current = clamped;
    setPlaybackVolume(clamped);
    if (scanVideoRef.current) scanVideoRef.current.volume = clamped;
    if (mapReplayVideoRef.current) mapReplayVideoRef.current.volume = clamped;
  };
  const [targetWidthInches, setTargetWidthInches] = useState(12);
  const [targetHeightInches, setTargetHeightInches] = useState(12);
  const [pixelsPerInch, setPixelsPerInch] = useState(0);
  const [calibrationDistanceInches, setCalibrationDistanceInches] = useState(252);
  const [manualDistanceOverrideInches] = useState(0);
  const [focalScalePxIn, setFocalScalePxIn] = useState(0);
  // Target opened via its QR (the /?t={id} handoff from the phone-camera scan).
  // Holds the QR's known printed size, which is the calibration reference now
  // that the QR itself no longer carries it inline.
  const [qrTarget, setQrTarget] = useState<{
    id: string;
    name: string | null;
    qrSizeValue: number;
    unit: TargetLinearUnit;
  } | null>(null);
  const [qrCalibrationStatus, setQrCalibrationStatus] = useState<string | null>(null);
  // Manual QR option: the QR's printed side length (stored in inches), entered by
  // hand so "Calibrate from QR" works without the /?t={id} handoff. When > 0 it
  // overrides any resolved target's size.
  const [manualQrSizeInches, setManualQrSizeInches] = useState(0);
  // Which calibration method the user is filling in (caliber is shown separately,
  // always). Defaults to the line tool; auto-switches to QR if one is detected.
  const [calibMethod, setCalibMethod] = useState<"line" | "dimensions" | "qr" | "manual">("line");
  const userPickedCalibMethodRef = useRef(false);
  // Result of auto-scanning the ROI for a QR on step 3: found, and whether the QR
  // encodes its own printed size (so the manual "printed size" field isn't needed).
  const [qrAutoScan, setQrAutoScan] = useState<{ found: boolean; hasEncodedSize: boolean } | null>(null);
  // Last QR detection, in ROI-canvas pixel space, drawn as a red highlight over
  // the reference (ROI) image. knownSize/ppi are 0 when the printed size is unknown.
  const [qrRoiHighlight, setQrRoiHighlight] = useState<{
    corners: { x: number; y: number }[];
    center: { x: number; y: number };
    qrSidePx: number; // source-pixel side length (same space as shots)
    knownSize: number;
    sizeUnit: TargetLinearUnit;
    ppi: number;
  } | null>(null);
  const matchThreshold = 20;
  const trackingMode: TrackingMode = "template";
  const [logEntries, setLogEntries] = useState<DetectionLogEntry[]>([]);
  const [shotLogEntries, setShotLogEntries] = useState<ShotLogEntry[]>([]);
  const [lastDetection, setLastDetection] = useState<DetectionLogEntry | null>(null);
  const [lastShot, setLastShot] = useState<ShotLogEntry | null>(null);
  const [roiRect, setRoiRect] = useState<RoiRect | null>(null);
  const [referenceImageSize, setReferenceImageSize] = useState<{ width: number; height: number } | null>(null);
  const [isSelectingRoi, setIsSelectingRoi] = useState(false);
  const [roiMeasurementLengthInches, setRoiMeasurementLengthInches] = useState(1);
  const [roiMagnifiedDataUrl, setRoiMagnifiedDataUrl] = useState<string | null>(null);
  const [roiSelectionPixelSize, setRoiSelectionPixelSize] = useState<{ widthPx: number; heightPx: number } | null>(null);
  const [roiMeasurementLine, setRoiMeasurementLine] = useState<NormalizedMeasurementLine | null>(null);
  const [isDrawingRoiMeasurementLine, setIsDrawingRoiMeasurementLine] = useState(false);
  const [roiZoom, setRoiZoom] = useState(1); // calibration image zoom (1 = fit width)
  // ROI image interaction tool: measure a line, pan the zoomed image, or drag a
  // marquee box to zoom into a region.
  const [roiTool, setRoiTool] = useState<"measure" | "pan" | "zoom">("measure");
  const roiScrollRef = useRef<HTMLDivElement | null>(null);
  const roiPanStartRef = useRef<{ clientX: number; clientY: number; scrollLeft: number; scrollTop: number } | null>(null);
  const roiZoomStartRef = useRef<{ x: number; y: number } | null>(null);
  const [roiZoomBox, setRoiZoomBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  // Capture step: the in-progress drag is a *draft* until confirmed in a modal.
  const [draftRoiRect, setDraftRoiRect] = useState<RoiRect | null>(null);
  const [confirmRoiOpen, setConfirmRoiOpen] = useState(false);
  const [confirmRoiPreview, setConfirmRoiPreview] = useState<string | null>(null);
  const [howlerReady, setHowlerReady] = useState(false);
  const [howlerError, setHowlerError] = useState<string | null>(null);
  const [spikeMetadata, setSpikeMetadata] = useState<SpikeMetadata[]>([]);
  const [audioSignatureCatalog, setAudioSignatureCatalog] = useState<AudioSignatureCatalogEntry[]>([]);
  const [audioSprites, setAudioSprites] = useState<Record<string, [number, number]>>({});
  const [spritesReady, setSpritesReady] = useState(false);
  const [audioCaptureInfo, setAudioCaptureInfo] = useState<AudioCaptureInfo | null>(null);
  const [audioCaptureError, setAudioCaptureError] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [targetTemplates, setTargetTemplates] = useState<TargetTemplate[]>([]);
  const [expandedSpikeIds, setExpandedSpikeIds] = useState<string[]>([]);
  const [analysisVideoCacheStatus, setAnalysisVideoCacheStatus] = useState<
    "idle" | "cached" | "too_large" | "unavailable"
  >("idle");
  const [unitConversionEnabled, setUnitConversionEnabled] = useState(false);
  const [displayLinearUnit, setDisplayLinearUnit] = useState<LinearUnit>("in");
  const [tweakSettings, setTweakSettings] = useState<TweakSettings>(DEFAULT_TWEAK_SETTINGS);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  // Color-blind friendly palette (persisted). Mirror it into the module-level
  // flag during render so clusterColorForId — used by memos below and by the
  // imperative canvas draw helpers — picks the right palette immediately.
  const [colorBlindMode, setColorBlindMode] = useColorBlindMode();
  clusterPaletteColorBlind = colorBlindMode;
  const [isAutoPicking, setIsAutoPicking] = useState(false);
  // Sortable shots table (below the group stats).
  const [showShotTable, setShowShotTable] = useState(false);
  const [shotTableSort, setShotTableSort] = useState<{ key: ShotTableSortKey; dir: "asc" | "desc" }>({
    key: "num",
    dir: "asc",
  });
  const [showDetailedViews, setShowDetailedViews] = useState(false);
  // Mirror to a ref so the long-running scan loop can skip the (expensive) per-frame
  // detail-view drawing when those views are hidden, even if toggled mid-scan.
  const showDetailedViewsRef = useRef(false);
  useEffect(() => {
    showDetailedViewsRef.current = showDetailedViews;
  }, [showDetailedViews]);
  // Audio-gated scanning: skip playback straight to the 0.5s windows around
  // detected gunshot spikes. Turn off to scan every frame — the fallback when
  // the clip's audio doesn't line up with the visible impacts.
  const [audioGatedScan, setAudioGatedScan] = useState(true);
  const audioGatedScanRef = useRef(true);
  useEffect(() => {
    audioGatedScanRef.current = audioGatedScan;
  }, [audioGatedScan]);
  // Detection trail: live step-by-step visualization of the pipeline (start
  // frame, stabilized frame, bit mask, blobs, end-frame check, sound track).
  const [showDetectionTrail, setShowDetectionTrail] = useState(true);
  const showDetectionTrailRef = useRef(true);
  useEffect(() => {
    showDetectionTrailRef.current = showDetectionTrail;
  }, [showDetectionTrail]);
  const [trailInfo, setTrailInfo] = useState<TrailInfo | null>(null);
  const trailBaselineCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trailCurrentCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trailMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trailBlobsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trailEndCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trailAudioCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrailDrawAtMsRef = useRef(0);
  // Optional secondary detector: bright yellow-green splatter (shoot-n-see
  // targets). Contrast/change detection is always the primary driver; this adds
  // the color detector on top for reactive targets.
  const [detectBrightColors, setDetectBrightColors] = useState(true);
  const detectBrightColorsRef = useRef(true);
  useEffect(() => {
    detectBrightColorsRef.current = detectBrightColors;
  }, [detectBrightColors]);
  // New from-scratch detector: when on, hits come only from persistent localized
  // change (see detectChangeHoles), not the legacy multi-signal pipeline.
  const [useChangeDetector, setUseChangeDetector] = useState(false);
  const useChangeDetectorRef = useRef(false);
  useEffect(() => {
    useChangeDetectorRef.current = useChangeDetector;
  }, [useChangeDetector]);
  // Per-track persistence state for the change detector: a candidate must recur
  // in roughly the same spot across frames before it's committed as a shot.
  const changeHoleTrackerRef = useRef<
    { x: number; y: number; count: number; lastFrame: number; committed: boolean }[]
  >([]);
  // Hybrid pipeline: confirm change-detector candidates with the patch classifier
  // (heuristic until a trained model is plugged in) to reject false positives.
  const [confirmWithClassifier, setConfirmWithClassifier] = useState(false);
  const confirmWithClassifierRef = useRef(false);
  useEffect(() => {
    confirmWithClassifierRef.current = confirmWithClassifier;
  }, [confirmWithClassifier]);
  // Pull the latest model published from the admin panel and install it as the
  // active hole classifier. This is the consumer side of "pushed out to users":
  // when no model is published the detector keeps using the heuristic baseline.
  const [publishedModelVersion, setPublishedModelVersion] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch("/api/classifier", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((json: { version?: number; model?: unknown } | null) => {
        if (cancelled || !json || !isModelJSON(json.model)) return;
        setHoleClassifier(buildClassifierFromModel(json.model));
        setPublishedModelVersion(json.version ?? null);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  // Training-data collection: harvest labeled hole/not-hole patches as you scan,
  // exportable as an ImageFolder dataset to train a classifier on.
  const [collectTrainingData, setCollectTrainingData] = useState(false);
  const collectTrainingDataRef = useRef(false);
  useEffect(() => {
    collectTrainingDataRef.current = collectTrainingData;
  }, [collectTrainingData]);
  const [trainingSamples, setTrainingSamples] = useState<TrainingSample[]>([]);
  const trainingIdRef = useRef(0);
  const patchCaptureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [collapsedSections, setCollapsedSections] = useState({ source: false, capture: false, calibrate: false });
  const [currentStep, setCurrentStep] = useState(0);
  const [autoAdvance, setAutoAdvance] = useState(false);
  // Account membership (for the Save step) + save UI state.
  const [membership, setMembership] = useState<{ configured: boolean; signedIn: boolean; isPro: boolean; email: string | null } | null>(
    null,
  );
  const [saveSessionName, setSaveSessionName] = useState("");
  const [savingSession, setSavingSession] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [shotGroupMode, setShotGroupMode] = useState<"dbscan" | "quadtree">("quadtree");
  // Shot group map background: "reference" = static frame updated as shots arrive; "live" = streamed cropped video.
  const [shotMapLiveStream, setShotMapLiveStream] = useState(true);
  // After a scan, the persistent reference background becomes the first frame with
  // the last frame composited at 50% opacity (so impacts show over the clean target).
  const [scanBlendBackgroundUrl, setScanBlendBackgroundUrl] = useState<string | null>(null);
  // Map replay: play the analyzed clip on the map and reveal shots up to the current time.
  const mapReplayVideoRef = useRef<HTMLVideoElement | null>(null);
  const mapReplayActiveRef = useRef(false);
  const mapRevealTimeRef = useRef<number>(Number.POSITIVE_INFINITY);
  const [mapReplayActive, setMapReplayActive] = useState(false);
  const [mapReplayPlaying, setMapReplayPlaying] = useState(false);
  const [mapReplayTimeSec, setMapReplayTimeSec] = useState(0);
  const [mapReplayDurationSec, setMapReplayDurationSec] = useState(0);
  // Shot tooltip: hover preview, plus a pinned one that stays open on click/tap.
  const [hoverShotInfo, setHoverShotInfo] = useState<{ shot: ShotLogEntry; left: number; top: number } | null>(null);
  const [pinnedShotInfo, setPinnedShotInfo] = useState<{ shot: ShotLogEntry; left: number; top: number } | null>(null);
  const [quadtreeRadiusScale, setQuadtreeRadiusScale] = useState(1);
  const [showQuadtreeCells, setShowQuadtreeCells] = useState(true);
  // Noise filter: shots below this confidence (0-100) are demoted to strays.
  const [minShotConfidence, setMinShotConfidence] = useState(0);
  // Manual grouping: shotId -> groupId (0 = stray). Overrides the automatic grouping.
  const [manualGroupOverrides, setManualGroupOverrides] = useState<Record<string, number>>({});
  // Optional custom names per (stable) group id; otherwise groups show a
  // contiguous "Group N" number that renumbers as groups are added/removed.
  const [groupNames, setGroupNames] = useState<Record<number, string>>({});
  const [manualEditMode, setManualEditMode] = useState(false);
  // Tap-to-add: when on, tapping the live video or the shot map drops a hit at
  // that point (works while the video plays).
  const [manualMarkMode, setManualMarkMode] = useState(false);
  // Remove mode: tap a shot on the map to delete it entirely (false hit cleanup).
  const [manualRemoveMode, setManualRemoveMode] = useState(false);
  const [activeManualGroup, setActiveManualGroup] = useState(1);
  // Box-select: ids currently selected, plus the in-progress drag rectangle (shot-space).
  const [manualSelectedIds, setManualSelectedIds] = useState<string[]>([]);
  const [selectionBlinkOn, setSelectionBlinkOn] = useState(true);
  const [manualSelectionRect, setManualSelectionRect] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(
    null,
  );
  // Per-group aim points (full-frame shot-pixel space, like shot.centerX/Y): where
  // each group was aiming, so the offset of its centroid can be reported. When
  // aimPointGroupId is non-null, the next tap on the map sets that group's aim point.
  const [groupAimPoints, setGroupAimPoints] = useState<Record<number, { x: number; y: number }>>({});
  const groupAimPointsRef = useRef(groupAimPoints);
  useEffect(() => {
    groupAimPointsRef.current = groupAimPoints;
  }, [groupAimPoints]);
  const [aimPointGroupId, setAimPointGroupId] = useState<number | null>(null);
  // Group timeline: scrub through time to watch each group form on an X/Y plot.
  // null cursor = follow the latest time (show all).
  const [timelineCursorSec, setTimelineCursorSec] = useState<number | null>(null);
  // Reveal window start (null = from the beginning of the clip). Paired with the
  // cursor (end) to narrow the map to shots in a [start, end] time window.
  const [timelineStartSec, setTimelineStartSec] = useState<number | null>(null);
  // Dragging the middle of the reveal slider shifts the whole window at a fixed
  // duration. The track element (for px→time) and the in-flight drag baseline.
  const revealTrackRef = useRef<HTMLDivElement | null>(null);
  const revealDragRef = useRef<{ startX: number; s0: number; dur: number; width: number } | null>(null);
  const sectionAutoCollapsedRef = useRef({ source: false, capture: false, calibrate: false });

  const activeLinearUnit: LinearUnit = unitConversionEnabled ? displayLinearUnit : "in";
  const activeLinearUnitLabel = LINEAR_UNIT_LABELS[activeLinearUnit];
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

  // The shot group map shows only the target ROI. Compute that region in image-pixel space,
  // and restrict the shots used for grouping/stats/drawing to the ones inside it.
  const roiPixelRect = useMemo(() => {
    if (!roiRect || !referenceImageSize) return null;
    const { width: imgW, height: imgH } = referenceImageSize;
    if (imgW <= 0 || imgH <= 0) return null;
    return {
      sx: Math.max(0, roiRect.x * imgW),
      sy: Math.max(0, roiRect.y * imgH),
      sw: Math.max(1, roiRect.width * imgW),
      sh: Math.max(1, roiRect.height * imgH),
    };
  }, [roiRect, referenceImageSize]);
  const mapShots = useMemo<ShotLogEntry[]>(() => {
    const inRoi = !roiPixelRect
      ? shotLogEntries
      : shotLogEntries.filter((shot) => {
          const { sx, sy, sw, sh } = roiPixelRect;
          return shot.centerX >= sx && shot.centerX <= sx + sw && shot.centerY >= sy && shot.centerY <= sy + sh;
        });
    // Size filter runs in the shot filters too: a match smaller than the caliber
    // is ignored entirely (dropped here, not shown), not just left ungrouped.
    // Hand-marked false positives are dropped here too (gone from map/groups/stats).
    return inRoi.filter(
      (shot) =>
        passesCaliberGate(shot, tweakSettings) && manualGroupOverrides[shot.id] !== FALSE_POSITIVE_OVERRIDE,
    );
  }, [shotLogEntries, roiPixelRect, tweakSettings, manualGroupOverrides]);

  // Audio false-positive floor: the loudest bang (the shooter) lowered 10% ×5.
  const audioStrayThreshold = useMemo(() => audioStrayThresholdDbfs(mapShots), [mapShots]);

  // Shots that pass the real gate AND clear the confidence noise filter. These
  // feed grouping/stats; everything else is shown as a (dimmed) stray.
  const eligibleMapShots = useMemo<ShotLogEntry[]>(
    () => mapShots.filter((shot) => qualifiesAsShot(shot, tweakSettings, audioStrayThreshold, minShotConfidence)),
    [mapShots, tweakSettings, minShotConfidence, audioStrayThreshold],
  );

  // Confidence overview for the Analysis UI.
  const confidenceSummary = useMemo(() => {
    const gated = mapShots.filter((shot) => shotMakesIt(shot, tweakSettings));
    const confidences = gated.map((shot) => shotConfidencePct(shot));
    const kept = confidences.filter((c) => c >= minShotConfidence);
    const avgKept = kept.length > 0 ? kept.reduce((sum, c) => sum + c, 0) / kept.length : 0;
    return {
      avgConfidence: avgKept,
      keptCount: kept.length,
      filteredCount: gated.length - kept.length,
      totalGated: gated.length,
    };
  }, [mapShots, tweakSettings, minShotConfidence]);

  const shotClustering = useMemo<ShotClusteringResult>(() => {
    // The results map is a final, static view, so no min-visible-age filter here (that's a
    // live-scan heuristic). Group every eligible ROI shot — matching quadtree.
    return clusterShotsBySpaceTime(eligibleMapShots, tweakSettings);
  }, [eligibleMapShots, tweakSettings]);
  const clusterColorById = useMemo<Record<number, string>>(() => {
    const map: Record<number, string> = {};
    for (const cluster of shotClustering.clusters) {
      map[cluster.clusterId] = clusterColorForId(cluster.clusterId);
    }
    return map;
  }, [shotClustering.clusters]);
  // Default the quadtree grouping distance to 3 inches when calibrated; the slider scales it.
  const quadtreeGroupInches = 3;
  const quadtreeBaseRadiusPx = pixelsPerInch > 0 ? quadtreeGroupInches * pixelsPerInch : 0;
  const quadtreeGrouping = useMemo<QuadtreeGroupingResult>(
    () => groupShotsByQuadtree(eligibleMapShots, quadtreeRadiusScale, quadtreeBaseRadiusPx),
    [eligibleMapShots, quadtreeRadiusScale, quadtreeBaseRadiusPx],
  );

  // Effective grouping = the active automatic method, with any manual overrides applied
  // on top (groupId >= 1 joins that group; 0 marks the shot as a stray and removes it).
  const autoGroupByShotId = shotGroupMode === "quadtree" ? quadtreeGrouping.shotClusterById : shotClustering.shotClusterById;
  const effectiveGroupByShotId = useMemo<Record<string, number>>(() => {
    const map: Record<string, number> = { ...autoGroupByShotId };
    for (const [id, group] of Object.entries(manualGroupOverrides)) {
      if (group >= 1) map[id] = group;
      else delete map[id];
    }
    return map;
  }, [autoGroupByShotId, manualGroupOverrides]);
  const effectiveGroupIds = useMemo<number[]>(() => {
    const ids = new Set<number>();
    for (const id of Object.values(effectiveGroupByShotId)) ids.add(id);
    return [...ids].sort((a, b) => a - b);
  }, [effectiveGroupByShotId]);
  const effectiveGroupColorById = useMemo<Record<number, string>>(() => {
    const map: Record<number, string> = {};
    for (const id of effectiveGroupIds) map[id] = clusterColorForId(id);
    return map;
  }, [effectiveGroupIds, colorBlindMode]);
  // Stable group id → contiguous display number (1, 2, 3 …), recomputed whenever
  // groups are added/removed so the numbering never has gaps.
  const groupDisplayNumberById = useMemo<Record<number, number>>(() => {
    const map: Record<number, number> = {};
    effectiveGroupIds.forEach((id, index) => {
      map[id] = index + 1;
    });
    return map;
  }, [effectiveGroupIds]);
  // Full label (custom name, else "Group N") and the short chip label ("N").
  const groupLabel = useCallback(
    (id: number) => groupNames[id]?.trim() || `Group ${groupDisplayNumberById[id] ?? id}`,
    [groupNames, groupDisplayNumberById],
  );
  const groupShortLabel = useCallback(
    (id: number) => groupNames[id]?.trim() || String(groupDisplayNumberById[id] ?? id),
    [groupNames, groupDisplayNumberById],
  );

  // Per-group statistics for the shot group map, recomputed from the effective grouping
  // (so manual edits, the active method, and the radius all flow through immediately).
  const effectiveGroupStats = useMemo(() => {
    const byGroup = new Map<number, ShotLogEntry[]>();
    let strayCount = 0;
    let transientCount = 0;
    for (const shot of mapShots) {
      // Transient (faded) detections are categorized separately from real strays.
      if (shot.persistent === false) {
        transientCount += 1;
        continue;
      }
      const groupId = effectiveGroupByShotId[shot.id];
      if (!groupId) {
        strayCount += 1;
        continue;
      }
      const bucket = byGroup.get(groupId) ?? [];
      bucket.push(shot);
      byGroup.set(groupId, bucket);
    }
    const groups = [...byGroup.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([groupId, shots]) => {
        const count = shots.length;
        let sumX = 0;
        let sumY = 0;
        for (const shot of shots) {
          sumX += shot.centerX;
          sumY += shot.centerY;
        }
        const centroidX = sumX / count;
        const centroidY = sumY / count;
        let extremeSpreadPx = 0;
        for (let i = 0; i < shots.length; i += 1) {
          for (let j = i + 1; j < shots.length; j += 1) {
            const dist = Math.hypot(shots[i].centerX - shots[j].centerX, shots[i].centerY - shots[j].centerY);
            if (dist > extremeSpreadPx) extremeSpreadPx = dist;
          }
        }
        let radialSum = 0;
        for (const shot of shots) {
          radialSum += Math.hypot(shot.centerX - centroidX, shot.centerY - centroidY);
        }
        const meanRadialPx = radialSum / count;
        const diameters = shots
          .map((shot) => shot.estimatedDiameterInches)
          .filter((value): value is number => value !== null);
        const meanDiameterInches =
          diameters.length > 0 ? diameters.reduce((sum, value) => sum + value, 0) / diameters.length : null;
        const timeSorted = [...shots].sort((a, b) => a.videoTimeSec - b.videoTimeSec);
        const timeSpanSec = timeSorted[count - 1].videoTimeSec - timeSorted[0].videoTimeSec;
        const meanConfidence = shots.reduce((sum, shot) => sum + shotConfidencePct(shot), 0) / count;
        return {
          groupId,
          count,
          centroidX,
          centroidY,
          extremeSpreadPx,
          meanRadialPx,
          meanDiameterInches,
          timeSpanSec,
          meanConfidence,
        };
      });
    return { groups, strayCount, transientCount };
  }, [mapShots, effectiveGroupByShotId]);

  // How many shots are hand-marked as false positives (dropped from everything).
  const falsePositiveCount = useMemo(
    () => Object.values(manualGroupOverrides).filter((value) => value === FALSE_POSITIVE_OVERRIDE).length,
    [manualGroupOverrides],
  );

  // Rows for the sortable shots table. Each row carries the raw shot plus the
  // derived values shown/sorted in the table. Null metrics sort to the bottom.
  const shotTableRows = useMemo(() => {
    const rows = mapShots.map((shot) => {
      const groupId = effectiveGroupByShotId[shot.id];
      const category =
        shot.persistent === false ? "Transient" : groupId ? groupLabel(groupId) : "Stray";
      return {
        shot,
        groupId,
        category,
        conf: shotConfidencePct(shot),
        sizeIn: shot.estimatedDiameterInches,
        sizePx: shot.estimatedDiameterPx,
        time: shot.videoTimeSec,
        loud: shot.audioDecibelDbfs,
        persist: shot.persistenceRatio ?? null,
        loudBang: hasLoudBang(shot),
      };
    });
    const dir = shotTableSort.dir === "asc" ? 1 : -1;
    const num = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? null : v);
    const keyOf = (r: (typeof rows)[number]): number | string | null => {
      switch (shotTableSort.key) {
        case "num":
          return r.shot.shotNumber;
        case "group":
          return r.category;
        case "conf":
          return r.conf;
        case "size":
          return num(r.sizeIn) ?? num(r.sizePx);
        case "time":
          return r.time;
        case "loud":
          return num(r.loud);
        case "persist":
          return num(r.persist);
        default:
          return r.shot.shotNumber;
      }
    };
    return [...rows].sort((a, b) => {
      const av = keyOf(a);
      const bv = keyOf(b);
      // Nulls always last, regardless of direction.
      if (av === null && bv === null) return a.shot.shotNumber - b.shot.shotNumber;
      if (av === null) return 1;
      if (bv === null) return -1;
      if (typeof av === "string" || typeof bv === "string") {
        return String(av).localeCompare(String(bv)) * dir;
      }
      return (av - bv) * dir;
    });
  }, [mapShots, effectiveGroupByShotId, groupLabel, shotTableSort]);

  // Time-ordered grouped shots (with bounds) backing the X/Y group-timeline plot.
  const groupTimeline = useMemo(() => {
    const points: { id: string; x: number; y: number; t: number; group: number }[] = [];
    let tMin = Infinity;
    let tMax = -Infinity;
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const shot of mapShots) {
      const group = effectiveGroupByShotId[shot.id];
      if (!group) continue;
      points.push({ id: shot.id, x: shot.centerX, y: shot.centerY, t: shot.videoTimeSec, group });
      if (shot.videoTimeSec < tMin) tMin = shot.videoTimeSec;
      if (shot.videoTimeSec > tMax) tMax = shot.videoTimeSec;
      if (shot.centerX < xMin) xMin = shot.centerX;
      if (shot.centerX > xMax) xMax = shot.centerX;
      if (shot.centerY < yMin) yMin = shot.centerY;
      if (shot.centerY > yMax) yMax = shot.centerY;
    }
    points.sort((a, b) => a.t - b.t);
    return {
      points,
      hasData: points.length > 0,
      tMin: Number.isFinite(tMin) ? tMin : 0,
      tMax: Number.isFinite(tMax) ? tMax : 0,
      xMin: Number.isFinite(xMin) ? xMin : 0,
      xMax: Number.isFinite(xMax) ? xMax : 1,
      yMin: Number.isFinite(yMin) ? yMin : 0,
      yMax: Number.isFinite(yMax) ? yMax : 1,
    };
  }, [mapShots, effectiveGroupByShotId]);
  const timelineCursor = timelineCursorSec ?? groupTimeline.tMax;
  const timelineStart = timelineStartSec ?? groupTimeline.tMin;
  const timelineWindowPoints = groupTimeline.points.filter(
    (point) => point.t >= timelineStart && point.t <= timelineCursor,
  );
  const timelineUpToCount = timelineWindowPoints.length;
  const timelineGroupsActive = new Set(timelineWindowPoints.map((point) => point.group)).size;
  // Handle positions (0–100%) for the dual-thumb reveal slider.
  const timelineSpanSec = Math.max(1e-6, groupTimeline.tMax - groupTimeline.tMin);
  const timelineStartPct = Math.max(0, Math.min(100, ((timelineStart - groupTimeline.tMin) / timelineSpanSec) * 100));
  const timelineEndPct = Math.max(0, Math.min(100, ((timelineCursor - groupTimeline.tMin) / timelineSpanSec) * 100));
  // Mirror the reveal window for the live-stream RAF (reads it fresh each frame
  // without restarting the loop on every scrub tick).
  const timelineCursorRef = useRef<number | null>(null);
  timelineCursorRef.current = timelineCursorSec;
  const timelineStartRef = useRef<number | null>(null);
  timelineStartRef.current = timelineStartSec;
  // True while the replay overlay was switched on by setting a reveal window (vs.
  // the user pressing Play), so clearing the window can switch it back off.
  const windowDrivenReplayRef = useRef(false);

  const changedTweakCount = useMemo(
    () =>
      (Object.keys(DEFAULT_TWEAK_SETTINGS) as Array<keyof TweakSettings>).filter(
        (key) => tweakSettings[key] !== DEFAULT_TWEAK_SETTINGS[key],
      ).length,
    [tweakSettings],
  );
  const hasUploadedVideo = captureMode === "upload" ? !!selectedVideoPreviewUrl : streamCameraActive;
  const hasReferenceFrame = !!selectedImagePreviewUrl;
  // Step 1 → 2 gate: step 2 needs the captured reference frame. Don't advance on
  // mere video-readiness — wait until the first frame is actually captured and
  // available, or step 2 shows a hanging "Preparing…" placeholder. Stream mode
  // waits for a live camera (its frame is captured in step 2).
  const canLeaveSourceStep = captureMode === "upload" ? hasReferenceFrame : streamCameraActive;
  const hasDrawnGeometry = !!roiRect && roiRect.width >= 0.01 && roiRect.height >= 0.01;
  const hasScaleCalibration = pixelsPerInch > 0 || focalScalePxIn > 0;
  const hasResultData = logEntries.length > 0 || shotLogEntries.length > 0;
  const workflowStep: WorkflowStep =
    !hasUploadedVideo
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
  const toggleSectionCollapsed = (key: "source" | "capture" | "calibrate") =>
    setCollapsedSections((current) => ({ ...current, [key]: !current[key] }));
  const roiMeasurementMetrics = useMemo(() => {
    if (!roiMeasurementLine || !roiSelectionPixelSize) return null;
    const dxPx = (roiMeasurementLine.endX - roiMeasurementLine.startX) * roiSelectionPixelSize.widthPx;
    const dyPx = (roiMeasurementLine.endY - roiMeasurementLine.startY) * roiSelectionPixelSize.heightPx;
    const pixelLength = Math.hypot(dxPx, dyPx);
    if (!Number.isFinite(pixelLength) || pixelLength <= 0) return null;
    const knownLengthInches = Math.max(0, roiMeasurementLengthInches);
    const nextPixelsPerInch = knownLengthInches > 0 ? pixelLength / knownLengthInches : null;
    return {
      pixelLength,
      knownLengthInches,
      pixelsPerInch: nextPixelsPerInch,
    };
  }, [roiMeasurementLine, roiMeasurementLengthInches, roiSelectionPixelSize]);

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
  const processedPatchCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const processedMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const processedYellowGreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imagePreviewRef = useRef<HTMLImageElement | null>(null);
  const roiMeasurementCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoSourceSectionRef = useRef<HTMLElement | null>(null);
  const captureSectionRef = useRef<HTMLElement | null>(null);
  const captureFrameButtonRef = useRef<HTMLButtonElement | null>(null);
  const autoCapturedVideoUrlRef = useRef<string | null>(null);
  const roiContainerRef = useRef<HTMLDivElement | null>(null);
  const calibrationSectionRef = useRef<HTMLElement | null>(null);
  const scanSectionRef = useRef<HTMLDivElement | null>(null);
  const audioSectionRef = useRef<HTMLElement | null>(null);
  const analysisSectionRef = useRef<HTMLElement | null>(null);
  const startScanButtonRef = useRef<HTMLButtonElement | null>(null);
  const exportButtonsRef = useRef<HTMLDivElement | null>(null);
  const previousWorkflowStepRef = useRef<WorkflowStep | null>(null);
  const suppressWorkflowScrollRef = useRef(false);
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
  const draftRoiRectRef = useRef<RoiRect | null>(null);
  const roiMeasurementLineStartRef = useRef<{ x: number; y: number } | null>(null);
  const roiMagnifiedImageRef = useRef<HTMLImageElement | null>(null);
  const streamMediaRef = useRef<MediaStream | null>(null);
  const spikeEventsRef = useRef<SpikeMetadata[]>([]);
  const scanVideoRef = useRef<HTMLVideoElement | null>(null);
  const analyzedPlaybackVideoRef = useRef<HTMLVideoElement | null>(null);
  const analyzedPlaybackRafRef = useRef<number | null>(null);
  const playbackSpeedRef = useRef(2);
  const detectionTimelineRef = useRef<DetectionLogEntry[]>([]);
  const contourWindowTimelineRef = useRef<ContourWindowFrameSnapshot[]>([]);
  const yellowGreenTimelineRef = useRef<YellowGreenFrameSnapshot[]>([]);
  const shotMarkersRef = useRef<ShotLogEntry[]>([]);
  const shotSequenceRef = useRef(0);
  // Per-shot persistence tally, keyed by shot id: how many frames after detection
  // the shot's mark was sampled, how many of those it was still present in the
  // baseline-difference mask, and how many times presence toggled on/off (flicker).
  // Resolved into persistenceRatio/persistent at scan end.
  const shotPresenceRef = useRef<
    Map<string, { frames: number; present: number; transitions: number; lastPresent: boolean }>
  >(new Map());
  // Every committed shot's patch-space location (uncapped, whole scan). A bullet
  // hole is permanent, so a later detection at the same spot is the same hole —
  // we de-dup against this so persistent holes aren't re-counted frame after frame.
  const committedShotPointsRef = useRef<{ px: number; py: number; rPatch: number }[]>([]);
  // Pending contour-shot tracks: candidate locations awaiting temporal
  // confirmation (CONTOUR_SHOT_MIN_SIGHTINGS_* frames) before commit. Committed
  // tracks stay in the list while sighted so a persistent hole keeps matching
  // its own track instead of spawning a fresh one every frame.
  const contourShotTracksRef = useRef<
    {
      px: number;
      py: number;
      rPatch: number;
      count: number;
      lastFrame: number;
      committed: boolean;
      detector: "yellow_green" | "change";
      bestScore: number;
      bestEntry: ShotLogEntry;
    }[]
  >([]);
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
  const liveClusterLastUpdatedAtMsRef = useRef(0);
  const liveContourGroupVisualsRef = useRef<{
    updatedAtMs: number;
    regionColors: string[];
    regionGroupLabels: string[];
  }>({
    updatedAtMs: 0,
    regionColors: [],
    regionGroupLabels: [],
  });
  const audioRmsTimelineRef = useRef<AudioRmsSample[]>([]);
  const audioMeanDbfsRef = useRef(-120);
  const audioThresholdDbfsRef = useRef(-120);
  const audioTimelineCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const shotGroupMapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const shotGroupMapImageRef = useRef<{ url: string; img: HTMLImageElement } | null>(null);
  const mapTransformRef = useRef<{
    scaleX: number;
    scaleY: number;
    originX: number;
    originY: number;
    canvasWidth: number;
    canvasHeight: number;
  } | null>(null);
  const manualDragRef = useRef<{ startX: number; startY: number; startClientX: number; startClientY: number; moved: boolean } | null>(
    null,
  );
  // Long-press-to-drag a single shot in edit mode: arm on pointerdown over a shot,
  // enter drag mode after the hold (with beep + haptic), then move it to reposition.
  const longPressTimerRef = useRef<number | null>(null);
  const pendingShotDragRef = useRef<{ shotId: string; startClientX: number; startClientY: number } | null>(null);
  const draggingShotIdRef = useRef<string | null>(null);
  const [draggingShotId, setDraggingShotId] = useState<string | null>(null);
  const dragAudioCtxRef = useRef<AudioContext | null>(null);
  useEffect(
    () => () => {
      if (longPressTimerRef.current !== null) window.clearTimeout(longPressTimerRef.current);
    },
    [],
  );
  const activeSpikeWindowIndexRef = useRef<number>(-1);
  const pendingShotCandidateRef = useRef<PendingShotCandidate | null>(null);
  const lastUiDetectionEnabledRef = useRef(false);
  const lastUiDetectionConfidenceRef = useRef(-1);
  const scanTaskActiveRef = useRef(false);
  const restartScanRequestedRef = useRef(false);
  const lastAutoCalibrationKeyRef = useRef<string>("");
  const gearsSettingsHydratedRef = useRef(false);
  const analysisVideoCacheRequestIdRef = useRef(0);
  const relaxedShotGateNoticeRef = useRef(false);
  const lastHistogramDeltaPctRef = useRef(0);

  const stopStreamCamera = useCallback(() => {
    const stream = streamMediaRef.current;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      streamMediaRef.current = null;
    }

    const videoEl = videoRef.current;
    if (videoEl?.srcObject) {
      videoEl.pause();
      videoEl.srcObject = null;
    }
    const overlayEl = overlayCanvasRef.current;
    if (overlayEl) {
      const overlayContext = overlayEl.getContext("2d");
      overlayContext?.clearRect(0, 0, overlayEl.width, overlayEl.height);
    }
    setStreamCameraActive(false);
  }, []);

  const startStreamCamera = useCallback(
    async (preferredFacingMode?: "environment" | "user") => {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        // The usual cause on phones: the page is served over plain http:// on a
        // LAN IP, which isn't a secure context, so the browser hides the camera
        // API entirely. localhost is exempt; a LAN address needs https://.
        const insecure = typeof window !== "undefined" && !window.isSecureContext;
        const message = insecure
          ? "The camera needs a secure (HTTPS) connection. Open Trackr over https:// (or via localhost) — phone browsers block the camera on plain http:// addresses."
          : "Camera streaming isn't available in this browser.";
        setStreamCameraError(message);
        setScanStatus(message);
        return false;
      }

      const facingMode = preferredFacingMode ?? streamCameraFacingMode;
      const constraintsWithFacing: MediaStreamConstraints = {
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      };
      const fallbackConstraints: MediaStreamConstraints = {
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      };

      stopStreamCamera();
      try {
        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraintsWithFacing);
        } catch {
          stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        }

        streamMediaRef.current = stream;
        const videoEl = videoRef.current;
        if (!videoEl) {
          setScanStatus("Camera preview element unavailable.");
          stopStreamCamera();
          return false;
        }

        videoEl.srcObject = stream;
        videoEl.playsInline = true;
        videoEl.muted = true;
        await videoEl.play().catch(() => undefined);

        setStreamCameraFacingMode(facingMode);
        setStreamCameraActive(true);
        setStreamCameraError(null);
        setScanStatus(`Device camera stream ready (${facingMode === "environment" ? "rear" : "front"} camera).`);
        return true;
      } catch (error) {
        const name = error instanceof DOMException ? error.name : "";
        let message: string;
        if (name === "NotAllowedError" || name === "SecurityError") {
          message =
            "Camera permission was blocked. Allow camera access for this site in your browser settings, then retry.";
        } else if (name === "NotFoundError" || name === "OverconstrainedError") {
          message = "No camera was found on this device.";
        } else if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") {
          message = "The camera is in use by another app. Close anything else using it and retry.";
        } else {
          message = error instanceof Error ? error.message : "Unable to access device camera.";
        }
        setStreamCameraError(message);
        setScanStatus(`Camera access failed: ${message}`);
        setStreamCameraActive(false);
        return false;
      }
    },
    [stopStreamCamera, streamCameraFacingMode],
  );

  const toggleStreamCameraFacingMode = useCallback(async () => {
    const nextFacingMode = streamCameraFacingMode === "environment" ? "user" : "environment";
    setStreamCameraFacingMode(nextFacingMode);
    if (captureMode === "stream" && streamCameraActive) {
      await startStreamCamera(nextFacingMode);
    }
  }, [captureMode, startStreamCamera, streamCameraActive, streamCameraFacingMode]);

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
    };
    try {
      window.localStorage.setItem(GEARS_SETTINGS_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Ignore storage write errors (e.g., quota/privacy mode).
    }
  }, [unitConversionEnabled, displayLinearUnit, tweakSettings]);

  useEffect(() => {
    if (!isSettingsModalOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsSettingsModalOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isSettingsModalOpen]);

  // Collapse each section once the user has moved past it; re-arm if they undo that progress.
  useEffect(() => {
    setCollapsedSections((current) => {
      const next = { ...current };
      let changed = false;
      const arm = (key: "source" | "capture" | "calibrate", done: boolean) => {
        if (done) {
          if (!sectionAutoCollapsedRef.current[key]) {
            sectionAutoCollapsedRef.current[key] = true;
            if (!next[key]) {
              next[key] = true;
              changed = true;
            }
          }
        } else {
          sectionAutoCollapsedRef.current[key] = false;
        }
      };
      arm("source", hasReferenceFrame);
      arm("capture", hasScaleCalibration);
      arm("calibrate", hasResultData);
      return changed ? next : current;
    });
  }, [hasReferenceFrame, hasScaleCalibration, hasResultData]);

  // Render the captured RMS/dBFS timeline with the spike threshold, mean, and per-spike markers.
  useEffect(() => {
    const canvas = audioTimelineCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width, height);

    const timeline = audioRmsTimelineRef.current;
    if (!audioCaptureInfo || timeline.length === 0) {
      ctx.fillStyle = "rgba(148,163,184,0.7)";
      ctx.font = "12px monospace";
      ctx.fillText("No audio captured yet — run a scan to populate the timeline.", 12, height / 2);
      return;
    }

    const minDb = Math.min(audioCaptureInfo.minDbfs, audioCaptureInfo.thresholdDbfs) - 2;
    const maxDb = Math.max(audioCaptureInfo.maxDbfs, audioCaptureInfo.thresholdDbfs) + 2;
    const durationSec = Math.max(audioCaptureInfo.durationSec, 1e-6);
    const dbRange = Math.max(maxDb - minDb, 1e-6);
    const toX = (timeSec: number) => (timeSec / durationSec) * width;
    const toY = (dbfs: number) => height - ((dbfs - minDb) / dbRange) * height;

    // Spike windows (faint amber bands).
    ctx.fillStyle = "rgba(251,191,36,0.10)";
    for (const spike of spikeMetadata) {
      const startX = toX(spike.windowStartSec);
      const endX = toX(spike.windowEndSec);
      ctx.fillRect(startX, 0, Math.max(1, endX - startX), height);
    }

    // Threshold line (red dashed) and mean line (sky).
    ctx.strokeStyle = "rgba(248,113,113,0.8)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, toY(audioCaptureInfo.thresholdDbfs));
    ctx.lineTo(width, toY(audioCaptureInfo.thresholdDbfs));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(125,211,252,0.55)";
    ctx.beginPath();
    ctx.moveTo(0, toY(audioCaptureInfo.meanDbfs));
    ctx.lineTo(width, toY(audioCaptureInfo.meanDbfs));
    ctx.stroke();

    // RMS dBFS curve (emerald).
    ctx.strokeStyle = "rgba(52,211,153,0.95)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < timeline.length; i += 1) {
      const x = toX(timeline[i].timeSec);
      const y = toY(timeline[i].dbfs);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Spike peak markers (amber verticals).
    ctx.fillStyle = "rgba(251,191,36,0.95)";
    for (const spike of spikeMetadata) {
      ctx.fillRect(toX(spike.timeSec) - 1, 0, 2, height);
    }
  }, [audioCaptureInfo, spikeMetadata]);

  // Keep the playback-speed ref in sync and apply it live to any in-progress analyzed playback.
  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
    if (analyzedPlaybackVideoRef.current) {
      analyzedPlaybackVideoRef.current.playbackRate = playbackSpeed;
    }
    if (howlRef.current?.rate) {
      howlRef.current.rate(playbackSpeed);
    }
  }, [playbackSpeed]);

  // Load the saved target-template library (ROIs the user has drawn before) for auto-detect.
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(TARGET_TEMPLATE_LIBRARY_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return;
      const valid = parsed.filter(
        (entry): entry is TargetTemplate =>
          !!entry &&
          typeof (entry as TargetTemplate).dataUrl === "string" &&
          typeof (entry as TargetTemplate).aspect === "number" &&
          (entry as TargetTemplate).aspect > 0,
      );
      setTargetTemplates(valid.slice(0, MAX_TARGET_TEMPLATES));
    } catch {
      // Ignore a malformed library.
    }
  }, []);

  // Track the reference image's natural size so the ROI (normalized) can be converted to
  // pixel space for cropping the shot group map and filtering shots to the target region.
  useEffect(() => {
    if (!selectedImagePreviewUrl) {
      setReferenceImageSize(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) setReferenceImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = selectedImagePreviewUrl;
    return () => {
      cancelled = true;
    };
  }, [selectedImagePreviewUrl]);

  // Shared overlay: quadtree cells, group hulls, shot markers, and the manual selection,
  // drawn in shot-space and shifted by the ROI origin. Used by both the static reference
  // map and the live cropped-video stream.
  const drawShotMapOverlay = (
    ctx: CanvasRenderingContext2D,
    scaleX: number,
    scaleY: number,
    originX: number,
    originY: number,
    revealTimeSec: number = Number.POSITIVE_INFINITY,
    revealStartSec: number = Number.NEGATIVE_INFINITY,
  ) => {
    const isQuad = shotGroupMode === "quadtree";
    const groupByShotId = effectiveGroupByShotId;
    const colorById = effectiveGroupColorById;
    // Reveal only the shots inside the [start, end] time window (defaults span the
    // whole clip). During replay the end follows the playback cursor; the Reveal
    // scrubber can also pull in the start to narrow to a window.
    const windowed = Number.isFinite(revealTimeSec) || Number.isFinite(revealStartSec);
    const visibleShots = windowed
      ? mapShots.filter((shot) => shot.videoTimeSec >= revealStartSec && shot.videoTimeSec <= revealTimeSec)
      : mapShots;

    ctx.save();
    ctx.translate(-originX * scaleX, -originY * scaleY);

    if (isQuad && showQuadtreeCells) {
      ctx.strokeStyle = "rgba(148,163,184,0.28)";
      ctx.lineWidth = 1;
      for (const cell of quadtreeGrouping.leafRects) {
        ctx.strokeRect(cell.x * scaleX, cell.y * scaleY, cell.w * scaleX, cell.h * scaleY);
      }
    }

    const geometry = clusterGeometryFromShots(visibleShots, groupByShotId);
    drawClusterGeometry(ctx, geometry, colorById, scaleX, scaleY, isQuad ? "G" : "DB", groupShortLabel);

    const selectedIdSet = new Set(manualSelectedIds);
    for (const shot of visibleShots) {
      const cx = shot.centerX * scaleX;
      const cy = shot.centerY * scaleY;
      const groupId = groupByShotId[shot.id];
      const isSelected = selectedIdSet.has(shot.id);
      // Circle sized to the estimated bullet hole, drawn over the actual feature.
      const bulletRadius = Math.max(3, (shot.estimatedDiameterPx / 2) * scaleX);
      // Loudness halo: a ring scaled + brightened by how loud the shot's bang was
      // (from the synced audio), drawn behind the hole marker.
      if (shot.audioDecibelDbfs !== null && Number.isFinite(shot.audioDecibelDbfs)) {
        const loud = Math.max(0, Math.min(1, (shot.audioDecibelDbfs + 60) / 55)); // -60 dB → 0, -5 dB → 1
        if (loud > 0.02) {
          const haloColor = groupId ? colorById[groupId] ?? clusterColorForId(groupId) : "#94a3b8";
          ctx.beginPath();
          ctx.arc(cx, cy, bulletRadius + 4 + loud * 16, 0, Math.PI * 2);
          ctx.strokeStyle = hexToRgba(haloColor, 0.18 + loud * 0.5);
          ctx.lineWidth = 1 + loud * 2;
          ctx.stroke();
        }
      }
      if (!groupId) {
        // Transient (faded) detections: amber dashed ring so they read as a
        // distinct category from real strays (solid gray).
        const isTransient = shot.persistent === false;
        ctx.beginPath();
        ctx.arc(cx, cy, bulletRadius, 0, Math.PI * 2);
        ctx.lineWidth = 1.5;
        if (isTransient) ctx.setLineDash([3, 3]);
        ctx.strokeStyle = isTransient ? "rgba(251,191,36,0.75)" : "rgba(148,163,184,0.6)";
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        const color = colorById[groupId] ?? clusterColorForId(groupId);
        ctx.beginPath();
        ctx.arc(cx, cy, bulletRadius, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(color, 0.18);
        ctx.fill();
        ctx.lineWidth = manualEditMode && groupId === activeManualGroup ? 2.5 : 1.75;
        ctx.strokeStyle = manualEditMode && groupId === activeManualGroup ? "#ffffff" : color;
        ctx.stroke();
      }
      // Small center mark for the precise impact point.
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fillStyle = groupId ? colorById[groupId] ?? clusterColorForId(groupId) : "rgba(148,163,184,0.85)";
      ctx.fill();
      if (isSelected) {
        const ring = bulletRadius + 3;
        const blinkFill = selectionBlinkOn ? "#ffffff" : "#000000";
        const blinkEdge = selectionBlinkOn ? "#000000" : "#ffffff";
        ctx.beginPath();
        ctx.arc(cx, cy, ring, 0, Math.PI * 2);
        ctx.lineWidth = 3;
        ctx.strokeStyle = blinkEdge;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, ring, 0, Math.PI * 2);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = blinkFill;
        ctx.stroke();
      }
    }

    if (manualSelectionRect) {
      const rx0 = Math.min(manualSelectionRect.x0, manualSelectionRect.x1) * scaleX;
      const ry0 = Math.min(manualSelectionRect.y0, manualSelectionRect.y1) * scaleY;
      const rw = Math.abs(manualSelectionRect.x1 - manualSelectionRect.x0) * scaleX;
      const rh = Math.abs(manualSelectionRect.y1 - manualSelectionRect.y0) * scaleY;
      ctx.fillStyle = selectionBlinkOn ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.3)";
      ctx.fillRect(rx0, ry0, rw, rh);
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = selectionBlinkOn ? "#ffffff" : "#000000";
      ctx.strokeRect(rx0, ry0, rw, rh);
      ctx.setLineDash([]);
    }

    // Aim points: a reticle where each group was aiming, with a dashed vector to
    // the group's (revealed) centroid — the on-screen twin of the offset stat.
    const aimPoints = groupAimPointsRef.current;
    for (const [groupIdStr, aim] of Object.entries(aimPoints)) {
      const groupId = Number(groupIdStr);
      let sumX = 0;
      let sumY = 0;
      let n = 0;
      for (const shot of visibleShots) {
        if (groupByShotId[shot.id] === groupId) {
          sumX += shot.centerX;
          sumY += shot.centerY;
          n += 1;
        }
      }
      const ax = aim.x * scaleX;
      const ay = aim.y * scaleY;
      const color = colorById[groupId] ?? clusterColorForId(groupId);
      if (n > 0) {
        const cX = (sumX / n) * scaleX;
        const cY = (sumY / n) * scaleY;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(cX, cY);
        ctx.strokeStyle = hexToRgba(color, 0.9);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // Reticle: white ring + crosshair so it reads over any background, with a
      // group-colored center dot.
      const r = 7;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(ax, ay, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ax - r - 3, ay);
      ctx.lineTo(ax + r + 3, ay);
      ctx.moveTo(ax, ay - r - 3);
      ctx.lineTo(ax, ay + r + 3);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(ax, ay, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    ctx.restore();
  };

  // Draw the reference frame with every shot plotted, colored by group, and a convex
  // hull around each group (DBSCAN space+time, or quadtree spatial grouping).
  useEffect(() => {
    if (shotMapLiveStream) return; // live mode is handled by the streaming effect below
    const canvas = shotGroupMapCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = (image: HTMLImageElement | null) => {
      const imageWidth = image?.naturalWidth ?? 0;
      const imageHeight = image?.naturalHeight ?? 0;
      const hasImage = imageWidth > 0 && imageHeight > 0;

      // Region to show: the target ROI in image-pixel space, or the full frame as a fallback.
      let originX = 0;
      let originY = 0;
      let regionW = imageWidth;
      let regionH = imageHeight;
      if (roiPixelRect && hasImage) {
        originX = Math.max(0, Math.min(roiPixelRect.sx, imageWidth - 1));
        originY = Math.max(0, Math.min(roiPixelRect.sy, imageHeight - 1));
        regionW = Math.max(1, Math.min(roiPixelRect.sw, imageWidth - originX));
        regionH = Math.max(1, Math.min(roiPixelRect.sh, imageHeight - originY));
      }

      const displayWidth = 760;
      const canvasWidth = displayWidth;
      const canvasHeight = hasImage
        ? Math.round((displayWidth * regionH) / regionW)
        : Math.round(displayWidth * 0.6);
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      if (hasImage) {
        // Draw only the ROI crop, stretched to fill the canvas.
        ctx.drawImage(image as HTMLImageElement, originX, originY, regionW, regionH, 0, 0, canvasWidth, canvasHeight);
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      } else {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      }

      if (mapShots.length === 0) {
        ctx.fillStyle = "rgba(148,163,184,0.85)";
        ctx.font = "16px sans-serif";
        ctx.fillText(
          roiPixelRect ? "No shots in the target region yet." : "No shots yet — run a scan to see grouped shot clusters.",
          16,
          canvasHeight / 2,
        );
        return;
      }

      // Shots are in frame-pixel space; reference dims = the shown region (ROI or full frame).
      let referenceWidth = regionW;
      let referenceHeight = regionH;
      if (!hasImage) {
        let maxShotX = 1;
        let maxShotY = 1;
        for (const shot of mapShots) {
          if (shot.centerX > maxShotX) maxShotX = shot.centerX;
          if (shot.centerY > maxShotY) maxShotY = shot.centerY;
        }
        referenceWidth = maxShotX * 1.05;
        referenceHeight = maxShotY * 1.05;
      }
      const scaleX = canvasWidth / referenceWidth;
      const scaleY = canvasHeight / referenceHeight;

      // Remember the draw transform (incl. ROI origin) so manual-edit taps map screen → shot space.
      mapTransformRef.current = { scaleX, scaleY, originX, originY, canvasWidth, canvasHeight };
      drawShotMapOverlay(
        ctx,
        scaleX,
        scaleY,
        originX,
        originY,
        timelineCursorSec ?? Number.POSITIVE_INFINITY,
        timelineStartSec ?? Number.NEGATIVE_INFINITY,
      );
    };

    // After a scan, prefer the first+last(50%) blended background so impacts show.
    const url = scanBlendBackgroundUrl ?? selectedImagePreviewUrl;
    if (!url) {
      render(null);
      return;
    }
    if (shotGroupMapImageRef.current?.url === url) {
      render(shotGroupMapImageRef.current.img);
      return;
    }
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      shotGroupMapImageRef.current = { url, img: image };
      render(image);
    };
    image.onerror = () => {
      if (!cancelled) render(null);
    };
    image.src = url;
    return () => {
      cancelled = true;
    };
  }, [
    selectedImagePreviewUrl,
    mapShots,
    roiPixelRect,
    shotGroupMode,
    quadtreeGrouping,
    showQuadtreeCells,
    effectiveGroupByShotId,
    effectiveGroupColorById,
    manualEditMode,
    activeManualGroup,
    manualSelectedIds,
    manualSelectionRect,
    selectionBlinkOn,
    shotMapLiveStream,
    groupAimPoints,
    colorBlindMode,
    timelineCursorSec,
    timelineStartSec,
    scanBlendBackgroundUrl,
    groupShortLabel,
  ]);

  // Live mode: stream the cropped ROI of the active video onto the shot group map and
  // overlay the same groups/shots, refreshing every animation frame.
  useEffect(() => {
    if (!shotMapLiveStream) return;
    // Only run the per-frame loop when the map is actually relevant on screen
    // (any of the Scan/Map/Review steps), or while scanning/replaying.
    const onAnalysisStep = currentStep >= STEP_ANALYSIS_FIRST && currentStep <= STEP_ANALYSIS_LAST;
    if (!onAnalysisStep && !isScanning && !mapReplayActive) return;
    const canvas = shotGroupMapCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      const replayActive = mapReplayActiveRef.current;
      const replayVideo = mapReplayVideoRef.current;
      const video =
        replayActive && replayVideo
          ? replayVideo
          : (analyzedPlaybackVideoRef.current ?? scanVideoRef.current ?? videoRef.current);
      // While the replay is actively playing, reveal shots up to the playhead.
      // Otherwise (paused replay or static) honor the reveal window [start, end] —
      // the replay frame itself is parked on the window's midpoint.
      const playingThrough = replayActive && replayVideo !== null && !replayVideo.paused;
      const revealStart = playingThrough ? Number.NEGATIVE_INFINITY : timelineStartRef.current ?? Number.NEGATIVE_INFINITY;
      const revealEnd = playingThrough
        ? (replayVideo as HTMLVideoElement).currentTime
        : timelineCursorRef.current ?? Number.POSITIVE_INFINITY;
      mapRevealTimeRef.current = revealEnd;
      const vW = video?.videoWidth ?? 0;
      const vH = video?.videoHeight ?? 0;
      const displayWidth = 760;

      if (!video || vW <= 0 || vH <= 0) {
        canvas.width = displayWidth;
        canvas.height = Math.round(displayWidth * 0.6);
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "rgba(148,163,184,0.85)";
        ctx.font = "16px sans-serif";
        ctx.fillText("Waiting for video — start a scan or play a spike.", 16, canvas.height / 2);
        raf = requestAnimationFrame(draw);
        return;
      }

      // ROI crop in video-pixel space (matches the reference frame the shots live in).
      let originX = 0;
      let originY = 0;
      let regionW = vW;
      let regionH = vH;
      if (roiRect) {
        originX = Math.max(0, Math.min(roiRect.x * vW, vW - 1));
        originY = Math.max(0, Math.min(roiRect.y * vH, vH - 1));
        regionW = Math.max(1, Math.min(roiRect.width * vW, vW - originX));
        regionH = Math.max(1, Math.min(roiRect.height * vH, vH - originY));
      }
      const canvasWidth = displayWidth;
      const canvasHeight = Math.round((displayWidth * regionH) / regionW);
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      ctx.drawImage(video, originX, originY, regionW, regionH, 0, 0, canvasWidth, canvasHeight);
      ctx.fillStyle = "rgba(0,0,0,0.18)";
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      const scaleX = canvasWidth / regionW;
      const scaleY = canvasHeight / regionH;
      mapTransformRef.current = { scaleX, scaleY, originX, originY, canvasWidth, canvasHeight };
      drawShotMapOverlay(ctx, scaleX, scaleY, originX, originY, revealEnd, revealStart);

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    shotMapLiveStream,
    currentStep,
    isScanning,
    mapReplayActive,
    roiRect,
    mapShots,
    shotGroupMode,
    quadtreeGrouping,
    showQuadtreeCells,
    effectiveGroupByShotId,
    effectiveGroupColorById,
    manualEditMode,
    activeManualGroup,
    manualSelectedIds,
    manualSelectionRect,
    selectionBlinkOn,
  ]);

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
    if (captureMode !== "stream") {
      setStreamCameraError(null);
      stopStreamCamera();
    }
  }, [captureMode, stopStreamCamera]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const markReady = () => {
      setOpenCvReady(true);
      setOpenCvError(null);
      setScanStatus("OpenCV.js ready");
    };

    const existingCv = window.cv as unknown as CvApi | undefined;
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
      const cv = window.cv as unknown as CvApi | undefined;
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
    previousWorkflowStepRef.current = workflowStep;
    // Navigation is now explicit via the step wizard, so no auto-scroll on step change.
    suppressWorkflowScrollRef.current = false;
  }, [workflowStep]);

  useEffect(() => {
    return () => {
      stopStreamCamera();
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
  }, [stopStreamCamera]);

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

  const cacheAnalysisVideoForSessionPlayback = useCallback(async (file: File) => {
    if (typeof window === "undefined") return;
    const requestId = analysisVideoCacheRequestIdRef.current + 1;
    analysisVideoCacheRequestIdRef.current = requestId;

    const cacheKey = buildAnalysisVideoCacheKey(file, file.name);
    if (!cacheKey) {
      setAnalysisVideoCacheStatus("unavailable");
      return;
    }

    if (file.size > ANALYSIS_VIDEO_SESSION_MAX_BYTES) {
      clearAnalysisVideoCache();
      if (analysisVideoCacheRequestIdRef.current === requestId) {
        setAnalysisVideoCacheStatus("too_large");
      }
      return;
    }

    const existingMeta = readStoredAnalysisVideoMeta();
    const existingDataUrl = sessionStorage.getItem(ANALYSIS_VIDEO_DATA_URL_KEY);
    if (existingMeta && existingDataUrl && existingMeta.key === cacheKey) {
      if (analysisVideoCacheRequestIdRef.current === requestId) {
        setAnalysisVideoCacheStatus("cached");
      }
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      if (analysisVideoCacheRequestIdRef.current !== requestId) return;
      const meta: StoredAnalysisVideoMeta = {
        key: cacheKey,
        name: file.name,
        size: file.size,
        lastModified: file.lastModified,
        type: file.type || "video/unknown",
      };
      sessionStorage.setItem(ANALYSIS_VIDEO_DATA_URL_KEY, dataUrl);
      sessionStorage.setItem(ANALYSIS_VIDEO_META_KEY, JSON.stringify(meta));
      setAnalysisVideoCacheStatus("cached");
    } catch {
      clearAnalysisVideoCache();
      if (analysisVideoCacheRequestIdRef.current === requestId) {
        setAnalysisVideoCacheStatus("unavailable");
      }
    }
  }, []);

  const resolvePlaybackVideoSource = useCallback((): { src: string | null; fromSession: boolean } => {
    if (typeof window === "undefined") return { src: selectedVideoPreviewUrl, fromSession: false };
    const cacheKey = buildAnalysisVideoCacheKey(selectedVideoFile, selectedVideoName);
    const meta = readStoredAnalysisVideoMeta();
    const dataUrl = sessionStorage.getItem(ANALYSIS_VIDEO_DATA_URL_KEY);
    if (cacheKey && meta && dataUrl && meta.key === cacheKey) {
      return { src: dataUrl, fromSession: true };
    }
    return { src: selectedVideoPreviewUrl, fromSession: false };
  }, [selectedVideoFile, selectedVideoName, selectedVideoPreviewUrl]);

  // --- Shot group map replay: play the analyzed clip on the map, revealing shots over time. ---
  const ensureMapReplaySource = (): boolean => {
    const video = mapReplayVideoRef.current;
    if (!video) return false;
    const source = resolvePlaybackVideoSource();
    if (!source.src) return false;
    if (video.src !== source.src) {
      video.src = source.src;
      video.load();
    }
    return true;
  };

  const startMapReplay = () => {
    const video = mapReplayVideoRef.current;
    if (!video || !ensureMapReplaySource()) {
      setScanStatus("No video available to replay. Upload a clip and scan first.");
      return;
    }
    setShotMapLiveStream(true);
    mapReplayActiveRef.current = true;
    setMapReplayActive(true);
    windowDrivenReplayRef.current = false; // user pressed Play — they own the replay now
    // Restart from the top if we were parked at the end.
    if (mapReplayDurationSec > 0 && video.currentTime >= mapReplayDurationSec - 0.05) {
      video.currentTime = 0;
    }
    video.playbackRate = playbackSpeed;
    video.muted = scanAudioMuted; // play the clip's audio during replay unless muted
    video.volume = playbackVolumeRef.current;
    void video.play().catch(() => {
      setScanStatus("Couldn't start replay playback.");
    });
  };

  const toggleMapReplay = () => {
    if (mapReplayPlaying) {
      mapReplayVideoRef.current?.pause();
    } else {
      startMapReplay();
    }
  };

  const exitMapReplay = () => {
    mapReplayVideoRef.current?.pause();
    mapReplayActiveRef.current = false;
    setMapReplayActive(false);
    setMapReplayPlaying(false);
    mapRevealTimeRef.current = Number.POSITIVE_INFINITY;
  };

  const scrubMapReplay = (timeSec: number) => {
    const video = mapReplayVideoRef.current;
    if (!video) return;
    if (!mapReplayActiveRef.current) {
      if (!ensureMapReplaySource()) return;
      setShotMapLiveStream(true);
      mapReplayActiveRef.current = true;
      setMapReplayActive(true);
    }
    windowDrivenReplayRef.current = false; // user grabbed the replay scrubber
    video.currentTime = timeSec;
    mapRevealTimeRef.current = timeSec;
    setMapReplayTimeSec(timeSec);
  };

  // Sync the replay to the reveal window: park the replay frame on the window's
  // midpoint as you drag it (a paused background frame), and switch the replay
  // overlay back off when the window is cleared (if we were the ones who set it).
  const revealWindowActive = timelineStartSec !== null || timelineCursorSec !== null;
  useEffect(() => {
    const video = mapReplayVideoRef.current;
    if (!video) return;
    if (revealWindowActive) {
      if (!ensureMapReplaySource()) return;
      const midpoint = (timelineStart + timelineCursor) / 2;
      if (!mapReplayActiveRef.current) {
        windowDrivenReplayRef.current = true;
        setShotMapLiveStream(true);
        mapReplayActiveRef.current = true;
        setMapReplayActive(true);
      }
      video.pause();
      if (Number.isFinite(midpoint)) {
        video.currentTime = midpoint;
        setMapReplayTimeSec(midpoint);
      }
    } else if (windowDrivenReplayRef.current) {
      windowDrivenReplayRef.current = false;
      video.pause();
      mapReplayActiveRef.current = false;
      setMapReplayActive(false);
      setMapReplayPlaying(false);
      mapRevealTimeRef.current = Number.POSITIVE_INFINITY;
      setShotMapLiveStream(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revealWindowActive, timelineStartSec, timelineCursorSec]);

  const clearAnalysisCanvases = () => {
    const processedContourCanvas = processedContourCanvasRef.current;
    if (processedContourCanvas) {
      const processedCtx = processedContourCanvas.getContext("2d");
      processedCtx?.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
    }
    const processedPatchCanvas = processedPatchCanvasRef.current;
    if (processedPatchCanvas) {
      const processedPatchCtx = processedPatchCanvas.getContext("2d");
      processedPatchCtx?.clearRect(0, 0, processedPatchCanvas.width, processedPatchCanvas.height);
    }
    const processedMaskCanvas = processedMaskCanvasRef.current;
    if (processedMaskCanvas) {
      const processedMaskCtx = processedMaskCanvas.getContext("2d");
      processedMaskCtx?.clearRect(0, 0, processedMaskCanvas.width, processedMaskCanvas.height);
    }
    const processedYellowGreenCanvas = processedYellowGreenCanvasRef.current;
    if (processedYellowGreenCanvas) {
      const processedYellowGreenCtx = processedYellowGreenCanvas.getContext("2d");
      processedYellowGreenCtx?.clearRect(0, 0, processedYellowGreenCanvas.width, processedYellowGreenCanvas.height);
    }
  };

  // Persist the detailed views after a scan ends: instead of blanking the debug
  // canvases, repaint the last recorded snapshot from each timeline so the final
  // contour/binary-mask/yellow-green frames stay on screen (and are there when the
  // panel is expanded after the fact). Falls back to clearing if nothing was
  // recorded. Scrubbing the map replay still re-renders these per-frame.
  const renderPersistedDetailedViews = () => {
    const lastContour =
      contourWindowTimelineRef.current[contourWindowTimelineRef.current.length - 1] ?? null;
    const contourCanvas = processedContourCanvasRef.current;
    const contourContext = contourCanvas?.getContext("2d") ?? null;
    if (contourCanvas && contourContext) {
      if (lastContour) drawPersistedContourWindowView(contourContext, lastContour);
      else contourContext.clearRect(0, 0, contourCanvas.width, contourCanvas.height);
    }
    const maskCanvas = processedMaskCanvasRef.current;
    const maskContext = maskCanvas?.getContext("2d") ?? null;
    if (maskCanvas && maskContext) {
      if (lastContour) drawBinaryMaskSnapshotWindowView(maskContext, lastContour, "Binary mask (final frame)");
      else maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    const lastYellowGreen =
      yellowGreenTimelineRef.current[yellowGreenTimelineRef.current.length - 1] ?? null;
    const yellowGreenCanvas = processedYellowGreenCanvasRef.current;
    const yellowGreenContext = yellowGreenCanvas?.getContext("2d") ?? null;
    if (yellowGreenCanvas && yellowGreenContext) {
      if (lastYellowGreen)
        drawYellowGreenSnapshotWindowView(yellowGreenContext, lastYellowGreen, "Yellow-green / top-hat (final frame)");
      else yellowGreenContext.clearRect(0, 0, yellowGreenCanvas.width, yellowGreenCanvas.height);
    }
  };

  const clearRoiLineCalibrationState = useCallback(() => {
    setRoiMagnifiedDataUrl(null);
    setRoiSelectionPixelSize(null);
    setRoiMeasurementLine(null);
    setIsDrawingRoiMeasurementLine(false);
    roiMeasurementLineStartRef.current = null;
    roiMagnifiedImageRef.current = null;
    const canvas = roiMeasurementCanvasRef.current;
    if (canvas) {
      const context = canvas.getContext("2d");
      context?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const renderRoiMeasurementCanvas = useCallback(() => {
    const canvas = roiMeasurementCanvasRef.current;
    const image = roiMagnifiedImageRef.current;
    if (!canvas || !image || !roiSelectionPixelSize) return;

    const width = Math.max(1, Math.round(roiSelectionPixelSize.widthPx));
    const height = Math.max(1, Math.round(roiSelectionPixelSize.heightPx));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.imageSmoothingEnabled = false;
    context.drawImage(image, 0, 0, width, height);

    // Red QR highlight on the reference image (corners are already in this canvas's
    // pixel space), with the measurements labeled just beneath the box.
    if (qrRoiHighlight && qrRoiHighlight.corners.length === 4) {
      const pts = qrRoiHighlight.corners;
      context.beginPath();
      context.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < 4; i += 1) context.lineTo(pts[i].x, pts[i].y);
      context.closePath();
      context.fillStyle = "rgba(239,68,68,0.22)";
      context.fill();
      context.lineWidth = Math.max(1.5, Math.min(4, Math.max(width, height) * 0.01));
      context.strokeStyle = "#ef4444";
      context.stroke();
      context.fillStyle = "#ef4444";
      const dotR = Math.max(2, Math.min(6, Math.max(width, height) * 0.015));
      for (const p of pts) {
        context.beginPath();
        context.arc(p.x, p.y, dotR, 0, Math.PI * 2);
        context.fill();
      }
      const label =
        qrRoiHighlight.ppi > 0
          ? `${qrRoiHighlight.qrSidePx.toFixed(0)} px = ${qrRoiHighlight.knownSize} ${qrRoiHighlight.sizeUnit} · ${qrRoiHighlight.ppi.toFixed(1)} px/in`
          : `${qrRoiHighlight.qrSidePx.toFixed(0)} px · printed size unknown`;
      context.font = "12px monospace";
      const textWidth = context.measureText(label).width;
      const maxY = Math.max(...pts.map((p) => p.y));
      const labelX = Math.min(Math.max(2, qrRoiHighlight.center.x - textWidth / 2), Math.max(2, width - textWidth - 4));
      const labelY = Math.min(maxY + 16, height - 4);
      context.fillStyle = "rgba(0,0,0,0.7)";
      context.fillRect(labelX - 3, labelY - 12, textWidth + 6, 16);
      context.fillStyle = "#fecaca";
      context.fillText(label, labelX, labelY);
    }

    if (!roiMeasurementLine) return;
    const x1 = clamp01(roiMeasurementLine.startX) * width;
    const y1 = clamp01(roiMeasurementLine.startY) * height;
    const x2 = clamp01(roiMeasurementLine.endX) * width;
    const y2 = clamp01(roiMeasurementLine.endY) * height;
    const pixelLength = Math.hypot(x2 - x1, y2 - y1);

    context.strokeStyle = "rgba(248, 113, 113, 0.98)";
    context.lineWidth = Math.max(1.5, Math.min(4, Math.max(width, height) * 0.01));
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();

    const endpointRadius = Math.max(2, Math.min(6, Math.max(width, height) * 0.015));
    context.fillStyle = "rgba(248, 113, 113, 0.98)";
    context.beginPath();
    context.arc(x1, y1, endpointRadius, 0, Math.PI * 2);
    context.fill();
    context.beginPath();
    context.arc(x2, y2, endpointRadius, 0, Math.PI * 2);
    context.fill();

    const label = `${pixelLength.toFixed(1)} px`;
    const labelX = Math.max(3, Math.min(width - 90, (x1 + x2) / 2 - 32));
    const labelY = Math.max(14, Math.min(height - 4, (y1 + y2) / 2 - 6));
    context.fillStyle = "rgba(0, 0, 0, 0.72)";
    context.fillRect(labelX - 3, labelY - 10, 72, 14);
    context.fillStyle = "#fca5a5";
    context.font = "10px sans-serif";
    context.fillText(label, labelX, labelY);
  }, [roiMeasurementLine, roiSelectionPixelSize, qrRoiHighlight]);

  useEffect(() => {
    if (isSelectingRoi) return;
    const activeRoi = roiRectRef.current ?? roiRect;
    if (!selectedImagePreviewUrl || !activeRoi || activeRoi.width < 0.01 || activeRoi.height < 0.01) {
      clearRoiLineCalibrationState();
      return;
    }

    let isCancelled = false;
    void (async () => {
      try {
        const [dataUrl, pixelSize] = await Promise.all([
          createTemplateRegionDataUrl(selectedImagePreviewUrl, activeRoi),
          getSelectionPixelSize(selectedImagePreviewUrl, activeRoi),
        ]);
        if (isCancelled) return;
        setRoiMagnifiedDataUrl(dataUrl);
        setRoiSelectionPixelSize(pixelSize);
        setRoiMeasurementLine(null);
        setQrRoiHighlight(null);
      } catch {
        if (!isCancelled) {
          clearRoiLineCalibrationState();
        }
      }
    })();

    return () => {
      isCancelled = true;
    };
  }, [clearRoiLineCalibrationState, isSelectingRoi, roiRect, selectedImagePreviewUrl]);

  useEffect(() => {
    if (!roiMagnifiedDataUrl) {
      roiMagnifiedImageRef.current = null;
      renderRoiMeasurementCanvas();
      return;
    }

    let isCancelled = false;
    const image = new Image();
    image.onload = () => {
      if (isCancelled) return;
      roiMagnifiedImageRef.current = image;
      renderRoiMeasurementCanvas();
    };
    image.onerror = () => {
      if (isCancelled) return;
      roiMagnifiedImageRef.current = null;
      renderRoiMeasurementCanvas();
    };
    image.src = roiMagnifiedDataUrl;

    return () => {
      isCancelled = true;
    };
  }, [renderRoiMeasurementCanvas, roiMagnifiedDataUrl]);

  useEffect(() => {
    renderRoiMeasurementCanvas();
  }, [renderRoiMeasurementCanvas]);

  const linePointFromClient = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const canvas = roiMeasurementCanvasRef.current;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return null;
    return {
      x: clamp01((clientX - bounds.left) / bounds.width),
      y: clamp01((clientY - bounds.top) / bounds.height),
    };
  };

  const startRoiMeasurementLineSelection = (clientX: number, clientY: number) => {
    const point = linePointFromClient(clientX, clientY);
    if (!point) return;
    roiMeasurementLineStartRef.current = point;
    setIsDrawingRoiMeasurementLine(true);
    setRoiMeasurementLine({
      startX: point.x,
      startY: point.y,
      endX: point.x,
      endY: point.y,
    });
  };

  const updateRoiMeasurementLineSelection = (clientX: number, clientY: number) => {
    if (!isDrawingRoiMeasurementLine) return;
    const start = roiMeasurementLineStartRef.current;
    if (!start) return;
    const point = linePointFromClient(clientX, clientY);
    if (!point) return;
    setRoiMeasurementLine({
      startX: start.x,
      startY: start.y,
      endX: point.x,
      endY: point.y,
    });
  };

  const endRoiMeasurementLineSelection = () => {
    setIsDrawingRoiMeasurementLine(false);
    roiMeasurementLineStartRef.current = null;
    setRoiMeasurementLine((current) => {
      if (!current || !roiSelectionPixelSize) return null;
      const dxPx = (current.endX - current.startX) * roiSelectionPixelSize.widthPx;
      const dyPx = (current.endY - current.startY) * roiSelectionPixelSize.heightPx;
      return Math.hypot(dxPx, dyPx) >= 2 ? current : null;
    });
  };

  // Marquee zoom: turn the dragged box into a zoom level + scroll so the box
  // fills the viewport.
  const applyRoiZoomBox = () => {
    const box = roiZoomBox;
    roiZoomStartRef.current = null;
    setRoiZoomBox(null);
    const canvas = roiMeasurementCanvasRef.current;
    const container = roiScrollRef.current;
    if (!box || !canvas || !container) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const fx0 = clamp01((Math.min(box.x0, box.x1) - rect.left) / rect.width);
    const fx1 = clamp01((Math.max(box.x0, box.x1) - rect.left) / rect.width);
    const fy0 = clamp01((Math.min(box.y0, box.y1) - rect.top) / rect.height);
    const fy1 = clamp01((Math.max(box.y0, box.y1) - rect.top) / rect.height);
    const fxw = fx1 - fx0;
    const fyh = fy1 - fy0;
    if (fxw < 0.02 || fyh < 0.02) return; // ignore an accidental tap
    const viewportW = container.clientWidth;
    const viewportH = container.clientHeight;
    const displayAspect = rect.height / rect.width;
    const zoomToFit = Math.min(1 / fxw, viewportH / (fyh * viewportW * displayAspect));
    const nextZoom = Math.max(1, Math.min(6, Math.round(zoomToFit * 10) / 10));
    const newCanvasW = viewportW * nextZoom;
    const newCanvasH = newCanvasW * displayAspect;
    const left = (fx0 + fxw / 2) * newCanvasW - viewportW / 2;
    const top = (fy0 + fyh / 2) * newCanvasH - viewportH / 2;
    setRoiZoom(nextZoom);
    // Scroll after the canvas re-lays-out at the new width.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        roiScrollRef.current?.scrollTo({ left: Math.max(0, left), top: Math.max(0, top) });
      }),
    );
  };

  const onRoiPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    if (roiTool === "measure") {
      startRoiMeasurementLineSelection(event.clientX, event.clientY);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    if (roiTool === "pan") {
      const c = roiScrollRef.current;
      if (c) roiPanStartRef.current = { clientX: event.clientX, clientY: event.clientY, scrollLeft: c.scrollLeft, scrollTop: c.scrollTop };
    } else {
      roiZoomStartRef.current = { x: event.clientX, y: event.clientY };
      setRoiZoomBox({ x0: event.clientX, y0: event.clientY, x1: event.clientX, y1: event.clientY });
    }
  };

  const onRoiPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (roiTool === "measure") {
      event.preventDefault();
      updateRoiMeasurementLineSelection(event.clientX, event.clientY);
    } else if (roiTool === "pan") {
      const p = roiPanStartRef.current;
      const c = roiScrollRef.current;
      if (p && c) {
        c.scrollLeft = p.scrollLeft - (event.clientX - p.clientX);
        c.scrollTop = p.scrollTop - (event.clientY - p.clientY);
      }
    } else {
      const s = roiZoomStartRef.current;
      if (s) setRoiZoomBox({ x0: s.x, y0: s.y, x1: event.clientX, y1: event.clientY });
    }
  };

  const onRoiPointerUp = () => {
    if (roiTool === "measure") endRoiMeasurementLineSelection();
    else if (roiTool === "pan") roiPanStartRef.current = null;
    else applyRoiZoomBox();
  };

  const onRoiPointerLeave = () => {
    if (roiTool === "measure") endRoiMeasurementLineSelection();
    else if (roiTool === "pan") roiPanStartRef.current = null;
    else {
      roiZoomStartRef.current = null;
      setRoiZoomBox(null);
    }
  };

  // Auto-apply line calibration as soon as a line is drawn (and a length is set),
  // so the user doesn't have to press a separate "Apply" button.
  useEffect(() => {
    if (isDrawingRoiMeasurementLine) return;
    const ppi = roiMeasurementMetrics?.pixelsPerInch;
    if (!ppi || ppi <= 0) return;
    setPixelsPerInch(ppi);
    setFocalScalePxIn(calibrationDistanceInches > 0 ? ppi * calibrationDistanceInches : 0);
  }, [isDrawingRoiMeasurementLine, roiMeasurementMetrics, calibrationDistanceInches]);

  // Stage 2 of the QR handoff: a phone camera scanned the target's id-only QR
  // and opened Trackr at /?t={id}. Resolve that id (local store first, else the
  // catalog) to learn the QR's printed size + target dimensions, so the in-app
  // scanner can calibrate by measuring the QR. Runs once on mount.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("t")?.trim();
    if (!id) return;

    const adopt = (name: string | null, qrSizeValue: number, unit: TargetLinearUnit, w?: number, h?: number) => {
      if (!(qrSizeValue > 0)) return;
      setQrTarget({ id, name, qrSizeValue, unit });
      if (w && w > 0) setTargetWidthInches(toInches(w, unit));
      if (h && h > 0) setTargetHeightInches(toInches(h, unit));
      setQrCalibrationStatus(`Loaded ${name ?? id}. Point the camera at it and calibrate from the QR.`);
    };

    const local = getTarget(id);
    if (local) {
      adopt(local.name, local.qrSizeValue, local.unit, local.widthValue, local.heightValue);
      return;
    }
    let cancelled = false;
    fetch(`/api/targets?id=${encodeURIComponent(id)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { name?: string | null; qrSizeValue?: number; unit?: TargetLinearUnit; widthValue?: number; heightValue?: number } | null) => {
        if (cancelled || !data) return;
        const unit: TargetLinearUnit = data.unit === "mm" || data.unit === "cm" || data.unit === "in" ? data.unit : "in";
        adopt(data.name ?? null, Number(data.qrSizeValue), unit, data.widthValue, data.heightValue);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // One-shot calibrate: detect the target QR within the SELECTED ROI (the
  // reference image shown for calibration), turn its measured pixel side length
  // into pixels-per-inch using the QR's known printed size, and highlight the QR
  // on that reference image. Working in ROI space keeps the highlight and the
  // shots in the same pixel coordinates.
  const calibrateFromQr = useCallback(() => {
    const cv = window.cv;
    if (!opencvReady || !cv?.QRCodeDetector) {
      setQrCalibrationStatus("OpenCV is still loading — try again in a moment.");
      return;
    }
    const roiImage = roiMagnifiedImageRef.current;
    if (!roiImage || !roiSelectionPixelSize || roiImage.naturalWidth <= 0) {
      setQrCalibrationStatus("Draw and select the target region first, then calibrate from its QR.");
      return;
    }
    // Detect at the reference image's native resolution for accuracy.
    const natW = roiImage.naturalWidth;
    const natH = roiImage.naturalHeight;
    const canvas = document.createElement("canvas");
    canvas.width = natW;
    canvas.height = natH;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(roiImage, 0, 0, natW, natH);

    const { qr, target } = detectTargetInCanvas(cv, canvas);
    if (!qr || qr.qrSidePx <= 0) {
      setQrRoiHighlight(null);
      setQrCalibrationStatus("No QR found in the selected region — adjust the region so the QR is fully inside it.");
      return;
    }

    // Map detection (native-res) → ROI-canvas/source-pixel space (roiSelectionPixelSize).
    const sx = roiSelectionPixelSize.widthPx / natW;
    const sy = roiSelectionPixelSize.heightPx / natH;
    const cornersRoi = qr.corners.map((corner) => ({ x: corner.x * sx, y: corner.y * sy }));
    const centerRoi = { x: qr.center.x * sx, y: qr.center.y * sy };
    const qrSidePxSource = qr.qrSidePx * ((sx + sy) / 2);

    // A manually-entered QR size wins; otherwise prefer the resolved target's
    // printed size, then a legacy QR carrying it inline.
    let sizeUnit: TargetLinearUnit;
    let knownSize: number;
    if (manualQrSizeInches > 0) {
      sizeUnit = "in";
      knownSize = manualQrSizeInches;
    } else {
      sizeUnit = qrTarget?.unit ?? target?.payload.unit ?? "in";
      knownSize = qrTarget?.qrSizeValue ?? target?.payload.qrSizeValue ?? 0;
    }

    const ppi = knownSize > 0 ? pixelsPerInchFromQr(qrSidePxSource, knownSize, sizeUnit) : 0;
    setQrRoiHighlight({ corners: cornersRoi, center: centerRoi, qrSidePx: qrSidePxSource, knownSize, sizeUnit, ppi: ppi || 0 });

    if (!(knownSize > 0)) {
      setQrCalibrationStatus(
        "Found a QR (highlighted on the reference image), but its printed size is unknown — enter the QR Code Printed Size field above to calibrate.",
      );
      return;
    }
    if (!ppi || ppi <= 0) {
      setQrCalibrationStatus("Couldn't compute scale from the QR.");
      return;
    }
    setPixelsPerInch(ppi);
    setFocalScalePxIn(calibrationDistanceInches > 0 ? ppi * calibrationDistanceInches : 0);
    const sourceNote = manualQrSizeInches > 0 ? " (manual)" : "";
    setQrCalibrationStatus(`Calibrated: ${ppi.toFixed(1)} px/in from a ${knownSize} ${sizeUnit} QR${sourceNote}.`);
  }, [opencvReady, qrTarget, calibrationDistanceInches, manualQrSizeInches, roiSelectionPixelSize]);

  // If auto-scan finds a QR without an encoded size, jump to the QR method so its
  // printed-size field is front-and-center — unless the user already picked a tab.
  useEffect(() => {
    if (!userPickedCalibMethodRef.current && qrAutoScan?.found && !qrAutoScan.hasEncodedSize) {
      setCalibMethod("qr");
    }
  }, [qrAutoScan]);

  // Auto-scan the selected ROI for a QR on step 3. If one is found but doesn't
  // encode its own printed size, the "QR Code Printed Size" field stays open so
  // the user can supply it; if the QR carries its size, no manual entry needed.
  useEffect(() => {
    if (currentStep !== STEP_CALIB_METHOD && currentStep !== STEP_CALIB_VALUES) return;
    const cv = window.cv;
    if (!opencvReady || !cv?.QRCodeDetector) return;
    if (!roiMagnifiedDataUrl) {
      setQrAutoScan(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      const { qr, target } = detectTargetInCanvas(cv, canvas);
      if (cancelled) return;
      if (!qr) {
        setQrAutoScan({ found: false, hasEncodedSize: false });
        return;
      }
      const hasEncodedSize = (target?.payload?.qrSizeValue ?? 0) > 0 || (qrTarget?.qrSizeValue ?? 0) > 0;
      setQrAutoScan({ found: true, hasEncodedSize });
    };
    img.src = roiMagnifiedDataUrl;
    return () => {
      cancelled = true;
    };
  }, [currentStep, opencvReady, roiMagnifiedDataUrl, qrTarget]);

  const hasAnalysisCoverageForWindow = useCallback(
    (windowStartSec: number, windowEndSec: number): boolean => {
      const safeStart = Math.min(windowStartSec, windowEndSec);
      const safeEnd = Math.max(windowStartSec, windowEndSec);
      return (
        shotLogEntries.some((entry) => entry.videoTimeSec >= safeStart && entry.videoTimeSec <= safeEnd) ||
        detectionTimelineRef.current.some((entry) => entry.videoTimeSec >= safeStart && entry.videoTimeSec <= safeEnd) ||
        contourWindowTimelineRef.current.some((entry) => entry.videoTimeSec >= safeStart && entry.videoTimeSec <= safeEnd) ||
        yellowGreenTimelineRef.current.some((entry) => entry.videoTimeSec >= safeStart && entry.videoTimeSec <= safeEnd)
      );
    },
    [shotLogEntries],
  );

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
        clearRoiLineCalibrationState();
        setFocalScalePxIn(0);
        clearTemplateRegionCache();
        return;
      }

      // Same video as a prior session? Offer to restore its saved reference frame
      // + calibration instead of starting over.
      const savedProfile = loadVideoProfile(buildAnalysisVideoCacheKey(file, file.name));
      const restore =
        savedProfile !== null &&
        window.confirm(
          `Found a saved reference frame and calibration for "${file.name}". Restore them?\n\n` +
            `OK = restore your previous setup · Cancel = start fresh.`,
        );

      revokeBlobUrl(selectedVideoPreviewUrl);
      setSelectedVideoName(file.name);
      setSelectedVideoFile(file);
      setSelectedVideoPreviewUrl(previewUrl);
      setAnalysisVideoCacheStatus("idle");
      setScanBlendBackgroundUrl(null);
      setGroupNames({});
      setGroupAimPoints({});
      setChoosingDifferentFrame(false);
      clearTemplateRegionCache();
      setSpikeMetadata([]);
      setAudioSignatureCatalog([]);
      setAudioSprites({});
      setSpritesReady(false);
      spikeEventsRef.current = [];
      clearAnalysisVideoCache();

      revokeBlobUrl(selectedImagePreviewUrl);
      if (restore && savedProfile) {
        // Restore prior reference frame + target calibration/geometry.
        setSelectedImageName(savedProfile.imageName);
        setSelectedImagePreviewUrl(savedProfile.imageDataUrl);
        setRoiRect(savedProfile.roiRect);
        roiRectRef.current = savedProfile.roiRect;
        setPixelsPerInch(savedProfile.pixelsPerInch);
        setFocalScalePxIn(savedProfile.focalScalePxIn);
        setCalibrationDistanceInches(savedProfile.calibrationDistanceInches);
        setTargetWidthInches(savedProfile.targetWidthInches);
        setTargetHeightInches(savedProfile.targetHeightInches);
        updateTweakSetting("expectedHoleDiameterInches", savedProfile.expectedHoleDiameterInches);
        // Keep the auto-capture effect from overwriting the restored frame.
        autoCapturedVideoUrlRef.current = previewUrl;
        setScanStatus("Restored your saved reference frame and calibration for this video.");
      } else {
        // Fresh: clear reference frame + calibration (auto-capture will re-grab).
        setSelectedImageName(null);
        setSelectedImagePreviewUrl(null);
        autoCapturedVideoUrlRef.current = null;
        setRoiRect(null);
        roiRectRef.current = null;
        clearRoiLineCalibrationState();
        setFocalScalePxIn(0);
      }
      void cacheAnalysisVideoForSessionPlayback(file);
    };

  const captureReferenceFrameFromVideo = (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;
    const videoEl = videoRef.current;
    if (!videoEl || (captureMode === "upload" ? !selectedVideoPreviewUrl : !streamCameraActive)) {
      setScanStatus(
        captureMode === "upload"
          ? "Upload a video first, then capture a reference frame."
          : "Start the device camera stream first, then capture a reference frame.",
      );
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
    const frameSourceName =
      captureMode === "upload"
        ? (selectedVideoName ?? "video").replace(/\.[^/.]+$/, "")
        : `stream-${streamCameraFacingMode === "environment" ? "rear" : "front"}-camera`;
    setSelectedImageName(
      `${frameSourceName}-frame-${videoEl.currentTime.toFixed(2)}s.png`,
    );
    setSelectedImagePreviewUrl(frameDataUrl);
    setRoiRect(null);
    roiRectRef.current = null;
    clearRoiLineCalibrationState();
    setFocalScalePxIn(0);
    clearTemplateRegionCache();
    setChoosingDifferentFrame(false);
    setScanStatus(
      silent
        ? "Using the first video frame as the reference image. Drag a rectangle around the target."
        : "Reference frame captured. Drag a rectangle around the target.",
    );
    if (!silent) {
      window.alert("Reference frame captured. Drag a rectangle around the target.");
    }
  };

  // A freshly chosen video / mode is not ready until it decodes (re-gates Next).
  useEffect(() => {
    setVideoReady(false);
  }, [selectedVideoPreviewUrl, captureMode]);

  // Persist this video's reference frame + calibration so re-uploading the same
  // clip can restore it. Debounced; only saves once a real captured frame exists.
  const videoProfileKey = useMemo(
    () => buildAnalysisVideoCacheKey(selectedVideoFile, selectedVideoName),
    [selectedVideoFile, selectedVideoName],
  );
  useEffect(() => {
    if (captureMode !== "upload" || !videoProfileKey) return;
    if (!selectedImagePreviewUrl || !selectedImagePreviewUrl.startsWith("data:")) return;
    const key = videoProfileKey;
    const imageDataUrl = selectedImagePreviewUrl;
    const handle = window.setTimeout(() => {
      saveVideoProfile(key, {
        imageName: selectedImageName,
        imageDataUrl,
        roiRect,
        pixelsPerInch,
        focalScalePxIn,
        calibrationDistanceInches,
        targetWidthInches,
        targetHeightInches,
        expectedHoleDiameterInches: tweakSettings.expectedHoleDiameterInches,
        savedAt: Date.now(),
      });
    }, 600);
    return () => window.clearTimeout(handle);
  }, [
    captureMode,
    videoProfileKey,
    selectedImagePreviewUrl,
    selectedImageName,
    roiRect,
    pixelsPerInch,
    focalScalePxIn,
    calibrationDistanceInches,
    targetWidthInches,
    targetHeightInches,
    tweakSettings.expectedHoleDiameterInches,
  ]);

  // Mark the uploaded video ready once it's decodable enough for step 2 to grab a
  // frame: dimensions + duration known and at least the current frame buffered.
  const markUploadedVideoReady = useCallback(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    if (
      videoEl.videoWidth > 0 &&
      videoEl.videoHeight > 0 &&
      Number.isFinite(videoEl.duration) &&
      videoEl.duration > 0 &&
      videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      setVideoReady(true);
    }
  }, []);

  // Preload a bundled sample video so the app opens with a working source that
  // flows through every step. The URL is set immediately (the element + frame
  // capture start right away); the File is fetched in the background because the
  // scan/audio pipeline needs it (it reads arrayBuffer). Skips if the user or a
  // restored session already provided a source.
  const didPreloadSampleRef = useRef(false);
  useEffect(() => {
    if (didPreloadSampleRef.current) return;
    didPreloadSampleRef.current = true;
    if (selectedVideoPreviewUrl || selectedImagePreviewUrl || selectedVideoFile) return;

    setSelectedVideoName(PRELOADED_VIDEO_NAME);
    setSelectedVideoPreviewUrl(PRELOADED_VIDEO_URL);
    setAnalysisVideoCacheStatus("idle");
    // We capture the first frame ourselves (below), so suppress the visible-element
    // auto-capture for this URL; if our capture fails we clear it so that takes over.
    autoCapturedVideoUrlRef.current = PRELOADED_VIDEO_URL;

    let cancelled = false;
    // Reference frame: grab frame 0 from a detached video so step 2 always has an
    // image even though the visible preview may be hidden once you navigate.
    void (async () => {
      const dataUrl = await captureFirstFrameDataUrl(PRELOADED_VIDEO_URL);
      if (cancelled) return;
      if (dataUrl) {
        setSelectedImagePreviewUrl((prev) => prev ?? dataUrl);
        setSelectedImageName((prev) => prev ?? `${PRELOADED_VIDEO_NAME.replace(/\.[^/.]+$/, "")}-frame-0.00s.png`);
      } else {
        // Let the visible-element auto-capture effect try instead.
        autoCapturedVideoUrlRef.current = null;
      }
    })();
    // File for the scan/audio pipeline (it reads arrayBuffer).
    void (async () => {
      try {
        const res = await fetch(PRELOADED_VIDEO_URL);
        if (!res.ok || cancelled) return;
        const blob = await res.blob();
        if (cancelled) return;
        const file = new File([blob], PRELOADED_VIDEO_NAME, { type: blob.type || "video/mp4" });
        setSelectedVideoFile(file);
        void cacheAnalysisVideoForSessionPlayback(file);
      } catch {
        /* leave the URL set; scanning just needs a re-pick if the fetch failed */
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Default the reference image to the video's first frame as soon as an uploaded
  // video is ready, unless the user has already captured/chosen one.
  useEffect(() => {
    if (captureMode !== "upload") return;
    if (!selectedVideoPreviewUrl) return;
    if (selectedImagePreviewUrl) return;
    if (autoCapturedVideoUrlRef.current === selectedVideoPreviewUrl) return;
    const videoEl = videoRef.current;
    if (!videoEl) return;

    autoCapturedVideoUrlRef.current = selectedVideoPreviewUrl;
    let cancelled = false;

    const captureNow = () => {
      if (cancelled || selectedImagePreviewUrl) return;
      if (videoEl.videoWidth <= 0 || videoEl.videoHeight <= 0) return;
      captureReferenceFrameFromVideo({ silent: true });
    };

    // `seeked`/`loadeddata` can fire before the decoded frame is actually painted, which
    // produces a black capture. Wait for a presented frame via requestVideoFrameCallback,
    // with a timeout fallback for browsers that lack it.
    const grab = () => {
      if (cancelled) return;
      let done = false;
      const fire = () => {
        if (done) return;
        done = true;
        captureNow();
      };
      const rvfc = (
        videoEl as HTMLVideoElement & {
          requestVideoFrameCallback?: (cb: () => void) => number;
        }
      ).requestVideoFrameCallback;
      if (typeof rvfc === "function") {
        rvfc.call(videoEl, fire);
      }
      window.setTimeout(fire, 250);
    };

    const onSeeked = () => {
      grab();
      cleanup();
    };
    const onLoadedData = () => {
      // Use the very first frame as the reference image.
      if (videoEl.currentTime > 0.01) {
        videoEl.addEventListener("seeked", onSeeked, { once: true });
        try {
          videoEl.currentTime = 0;
        } catch {
          videoEl.removeEventListener("seeked", onSeeked);
          grab();
          cleanup();
        }
      } else {
        grab();
        cleanup();
      }
    };
    const cleanup = () => {
      videoEl.removeEventListener("loadeddata", onLoadedData);
      videoEl.removeEventListener("seeked", onSeeked);
    };

    if (videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      onLoadedData();
    } else {
      videoEl.addEventListener("loadeddata", onLoadedData, { once: true });
    }

    return () => {
      cancelled = true;
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureMode, selectedVideoPreviewUrl, selectedImagePreviewUrl]);

  const autoPickReferenceFrameFromVideo = async () => {
    if (captureMode !== "upload") {
      setScanStatus("Auto-pick is available for uploaded videos only.");
      return;
    }
    const videoEl = videoRef.current;
    const cv = window.cv as unknown as CvApi | undefined;
    if (!videoEl || !selectedVideoPreviewUrl) {
      setScanStatus("Upload a video first, then auto-pick a reference frame.");
      return;
    }
    if (!opencvReady || !cv) {
      setScanStatus("OpenCV is still loading. Try auto-pick again in a moment.");
      return;
    }
    const duration = videoEl.duration;
    if (!Number.isFinite(duration) || duration <= 0 || videoEl.videoWidth <= 0) {
      setScanStatus("Video metadata not ready yet. Play or seek the video, then auto-pick.");
      return;
    }

    setIsAutoPicking(true);
    suppressWorkflowScrollRef.current = true;
    const loadedTemplates: FrameTemplate[] = [];
    try {
      videoEl.pause();

      const sampleCount = 24;
      const detectionScale = Math.min(1, 480 / videoEl.videoWidth);
      const detectionCanvas = document.createElement("canvas");
      detectionCanvas.width = Math.max(1, Math.round(videoEl.videoWidth * detectionScale));
      detectionCanvas.height = Math.max(1, Math.round(videoEl.videoHeight * detectionScale));
      const frameWidth = detectionCanvas.width;
      const frameHeight = detectionCanvas.height;
      const detectionCtx = detectionCanvas.getContext("2d", { willReadFrequently: true });
      if (!detectionCtx) {
        setScanStatus("Unable to prepare frame analysis canvas for auto-pick.");
        return;
      }

      const timeForSample = (index: number) =>
        duration * (0.04 + (0.92 * index) / Math.max(1, sampleCount - 1));

      // Pad a detected box slightly so the target isn't clipped, then clamp to [0,1].
      const padBox = (box: { x: number; y: number; width: number; height: number }): RoiRect => {
        const pad = 0.04;
        const rect: RoiRect = {
          x: Math.max(0, box.x - pad),
          y: Math.max(0, box.y - pad),
          width: Math.min(1, box.width + pad * 2),
          height: Math.min(1, box.height + pad * 2),
        };
        rect.width = Math.min(rect.width, 1 - rect.x);
        rect.height = Math.min(rect.height, 1 - rect.y);
        return rect;
      };

      // Seek to the chosen time, capture the full-res frame, and wire it up as the reference.
      const finalizeReference = async (
        timeSec: number,
        autoRect: RoiRect,
        dims: { widthInches: number; heightInches: number } | null,
      ): Promise<boolean> => {
        await seekVideo(videoEl, timeSec);
        const width = videoEl.videoWidth;
        const height = videoEl.videoHeight;
        const frameCanvas = document.createElement("canvas");
        frameCanvas.width = width;
        frameCanvas.height = height;
        const frameCtx = frameCanvas.getContext("2d");
        if (!frameCtx) {
          setScanStatus("Unable to capture the auto-picked reference frame.");
          return false;
        }
        frameCtx.drawImage(videoEl, 0, 0, width, height);
        const frameDataUrl = frameCanvas.toDataURL("image/png");
        const frameSourceName = (selectedVideoName ?? "video").replace(/\.[^/.]+$/, "");
        const frameName = `${frameSourceName}-auto-${timeSec.toFixed(2)}s.png`;

        revokeBlobUrl(selectedImagePreviewUrl);
        clearRoiLineCalibrationState();
        setFocalScalePxIn(0);
        setSelectedImageName(frameName);
        setSelectedImagePreviewUrl(frameDataUrl);
        roiRectRef.current = autoRect;
        setRoiRect(autoRect);
        lastAutoCalibrationKeyRef.current = "";
        // Reuse the dimensions saved with the matched template so calibration can auto-run.
        if (dims && dims.widthInches > 0 && dims.heightInches > 0 && targetWidthInches <= 0 && targetHeightInches <= 0) {
          setTargetWidthInches(dims.widthInches);
          setTargetHeightInches(dims.heightInches);
        }

        try {
          const templateRegionDataUrl = await createTemplateRegionDataUrl(frameDataUrl, autoRect);
          sessionStorage.setItem(TEMPLATE_REGION_DATA_URL_KEY, templateRegionDataUrl);
          sessionStorage.setItem(TEMPLATE_REGION_IMAGE_NAME_KEY, frameName);
          sessionStorage.setItem(TEMPLATE_REGION_RECT_KEY, JSON.stringify(autoRect));
        } catch {
          clearTemplateRegionCache();
        }
        return true;
      };

      // Pass 1: match against the target outlines the user has drawn before.
      for (const template of targetTemplates.slice(0, 5)) {
        try {
          const image = await loadImageFromUrl(template.dataUrl);
          if (image.naturalWidth <= 0 || image.naturalHeight <= 0) continue;
          const templateCanvas = document.createElement("canvas");
          const downscale = Math.min(1, 240 / image.naturalWidth);
          templateCanvas.width = Math.max(1, Math.round(image.naturalWidth * downscale));
          templateCanvas.height = Math.max(1, Math.round(image.naturalHeight * downscale));
          const templateCtx = templateCanvas.getContext("2d");
          if (!templateCtx) continue;
          templateCtx.drawImage(image, 0, 0, templateCanvas.width, templateCanvas.height);
          const srcMat = cv.imread(templateCanvas);
          const gray = new cv.Mat();
          cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
          srcMat.delete();
          loadedTemplates.push({
            gray,
            aspect: templateCanvas.width / Math.max(templateCanvas.height, 1),
            targetWidthInches: template.targetWidthInches,
            targetHeightInches: template.targetHeightInches,
          });
        } catch {
          // Skip a template that fails to load.
        }
      }

      if (loadedTemplates.length > 0) {
        const scales = [0.25, 0.38, 0.52, 0.66];
        let best: (TemplateMatchResult & { timeSec: number }) | null = null;
        for (let i = 0; i < sampleCount; i += 1) {
          const timeSec = timeForSample(i);
          setScanStatus(`Auto-pick: matching your ${loadedTemplates.length} saved target(s)… ${i + 1}/${sampleCount}`);
          try {
            await seekVideo(videoEl, timeSec);
          } catch {
            continue;
          }
          detectionCtx.drawImage(videoEl, 0, 0, frameWidth, frameHeight);
          const frameMat = cv.imread(detectionCanvas);
          const frameGray = new cv.Mat();
          cv.cvtColor(frameMat, frameGray, cv.COLOR_RGBA2GRAY);
          frameMat.delete();
          const match = bestTemplateMatchForFrame(cv, frameGray, frameWidth, frameHeight, loadedTemplates, scales);
          frameGray.delete();
          if (match && (!best || match.score > best.score)) best = { ...match, timeSec };
          if (best && best.score >= 0.85) break; // strong enough, stop early
        }

        if (best && best.score >= 0.45) {
          const finalized = await finalizeReference(best.timeSec, padBox(best.box), {
            widthInches: best.targetWidthInches,
            heightInches: best.targetHeightInches,
          });
          if (finalized) {
            setScanStatus(
              `Auto-picked at ${best.timeSec.toFixed(2)}s by matching a target you drew before (${(
                best.score * 100
              ).toFixed(0)}% match). Adjust the box if needed.`,
            );
          }
          return;
        }
        setScanStatus("No strong match to your saved targets — falling back to shape detection…");
      }

      // Pass 2: fall back to finding the most common object by shape.
      const samples: DominantObjectSample[] = [];
      for (let i = 0; i < sampleCount; i += 1) {
        const timeSec = timeForSample(i);
        setScanStatus(`Auto-pick: detecting most common object… ${i + 1}/${sampleCount}`);
        try {
          await seekVideo(videoEl, timeSec);
        } catch {
          continue;
        }
        detectionCtx.drawImage(videoEl, 0, 0, frameWidth, frameHeight);
        const box = detectDominantObjectBox(cv, detectionCanvas);
        if (box) samples.push({ timeSec, box });
      }

      const chosen = pickMostCommonObjectFrame(samples);
      if (!chosen) {
        setScanStatus("Auto-pick could not find a recurring object. Capture a frame manually instead.");
        return;
      }
      const finalized = await finalizeReference(chosen.timeSec, padBox(chosen.box), null);
      if (finalized) {
        setScanStatus(
          `Auto-picked reference frame at ${chosen.timeSec.toFixed(2)}s and placed a target box around the most common object. Adjust the box or set target dimensions next.`,
        );
      }
    } catch {
      setScanStatus("Auto-pick failed while scanning the video. Capture a frame manually instead.");
    } finally {
      for (const template of loadedTemplates) template.gray.delete();
      setIsAutoPicking(false);
      // Fallback release in case auto-pick didn't change the workflow step (so the
      // step effect never ran to consume the flag).
      window.setTimeout(() => {
        suppressWorkflowScrollRef.current = false;
      }, 600);
    }
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

  // Wizard steps, in the same order they appear in the DOM. Kept granular so each
  // step holds a small, focused amount of content (mobile-friendly).
  const sectionSteps: { label: string; short: string }[] = [
    { label: "Video", short: "1" },
    { label: "Target", short: "2" },
    { label: "Calibration", short: "3" },
    { label: "Measurements", short: "4" },
    { label: "Scan", short: "5" },
    { label: "Map", short: "6" },
    { label: "Review", short: "7" },
    { label: "Save", short: "8" },
  ];

  const goToStep = (index: number) => {
    // Can't move past the video-source step until the reference frame is ready.
    if (index > STEP_VIDEO && currentStep === STEP_VIDEO && !canLeaveSourceStep) return;
    setCurrentStep(Math.max(0, Math.min(sectionSteps.length - 1, index)));
  };

  // Probe account membership once for the Save step.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/me", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setMembership(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Save the current session's bullet annotations to the signed-in account.
  const saveAnnotationsToAccount = async () => {
    const shots = mapShots.map((shot) => ({
      n: shot.shotNumber,
      x: Math.round(shot.centerX),
      y: Math.round(shot.centerY),
      dpx: Math.round(shot.estimatedDiameterPx),
      din: shot.estimatedDiameterInches,
      t: shot.videoTimeSec,
      method: shot.detectionMethod,
      group: effectiveGroupByShotId[shot.id] ?? null,
      conf: Math.round(shotConfidencePct(shot)),
    }));
    if (shots.length === 0) {
      setSaveStatus("No shots to save yet — run a scan first.");
      return;
    }
    setSavingSession(true);
    setSaveStatus(null);
    try {
      const res = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: saveSessionName.trim() || selectedVideoName || null,
          target: {
            widthInches: targetWidthInches,
            heightInches: targetHeightInches,
            pixelsPerInch,
            unit: activeLinearUnit,
            roi: roiRect,
            // Per-group aim points (where each group was aiming) — saved so the
            // shot library can normalize/align stats on intent, not just impact.
            aimPoints: Object.fromEntries(
              Object.entries(groupAimPoints).map(([g, p]) => [g, { x: Math.round(p.x), y: Math.round(p.y) }]),
            ),
          },
          shots,
        }),
      });
      const json = (await res.json().catch(() => ({}))) as { ok?: boolean; shotCount?: number; error?: string };
      if (!res.ok || !json.ok) {
        setSaveStatus(json.error ?? "Save failed.");
      } else {
        setSaveStatus(`Saved ${json.shotCount ?? shots.length} shots to your account.`);
        setSaveSessionName("");
      }
    } catch {
      setSaveStatus("Save failed (network).");
    } finally {
      setSavingSession(false);
    }
  };

  // Each step is its own card; jump back to the top when switching steps.
  useEffect(() => {
    if (typeof window !== "undefined") window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentStep]);

  // Auto-advance preference persists in localStorage.
  useEffect(() => {
    if (typeof window === "undefined") return;
    setAutoAdvance(window.localStorage.getItem(AUTO_ADVANCE_STORAGE_KEY) === "true");
  }, []);
  const handleAutoAdvanceChange = (next: boolean) => {
    setAutoAdvance(next);
    if (typeof window !== "undefined") window.localStorage.setItem(AUTO_ADVANCE_STORAGE_KEY, String(next));
  };

  // When enabled, advance to the next step the moment the current step's milestone is met
  // (rising edge only, so going back manually doesn't bounce you forward again).
  const autoAdvancePrevRef = useRef({ uploaded: false, geometry: false, calib: false, scanning: false });
  useEffect(() => {
    const prev = autoAdvancePrevRef.current;
    if (autoAdvance) {
      if (!prev.uploaded && canLeaveSourceStep && currentStep === STEP_VIDEO) goToStep(STEP_TARGET);
      else if (!prev.geometry && hasDrawnGeometry && currentStep === STEP_TARGET) goToStep(STEP_CALIB_METHOD);
      else if (!prev.calib && hasScaleCalibration && currentStep === STEP_CALIB_VALUES) goToStep(STEP_SCAN);
    }
    autoAdvancePrevRef.current = {
      uploaded: canLeaveSourceStep,
      geometry: hasDrawnGeometry,
      calib: hasScaleCalibration,
      scanning: isScanning,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoAdvance, canLeaveSourceStep, hasDrawnGeometry, hasScaleCalibration, hasResultData, isScanning, currentStep]);

  // While a scan runs, collapse the detailed analysis views down to just the Shot Group Map.
  useEffect(() => {
    if (isScanning) setShowDetailedViews(false);
  }, [isScanning]);

  // Blink the manual-edit selection (black/white) so it stands out over any background.
  useEffect(() => {
    const active = manualEditMode && (manualSelectedIds.length > 0 || manualSelectionRect !== null);
    if (!active) return;
    const id = window.setInterval(() => setSelectionBlinkOn((value) => !value), 450);
    return () => window.clearInterval(id);
  }, [manualEditMode, manualSelectedIds, manualSelectionRect]);

  // Map a pointer position to shot-space (reference-frame px) using the last draw transform.
  const mapClientToShot = (clientX: number, clientY: number) => {
    const canvas = shotGroupMapCanvasRef.current;
    const transform = mapTransformRef.current;
    if (!canvas || !transform) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const internalX = (clientX - rect.left) * (canvas.width / rect.width);
    const internalY = (clientY - rect.top) * (canvas.height / rect.height);
    // Canvas x = (shotX - originX) * scaleX, so invert with the ROI origin offset.
    return {
      shotX: internalX / transform.scaleX + transform.originX,
      shotY: internalY / transform.scaleY + transform.originY,
    };
  };

  // Tap a single shot to move it into the active group (0 = stray).
  const assignSingleShotAtPoint = (clientX: number, clientY: number) => {
    const point = mapClientToShot(clientX, clientY);
    const transform = mapTransformRef.current;
    if (!point || !transform) return;
    let nearestId: string | null = null;
    let nearestDist = Infinity;
    for (const shot of mapShots) {
      const dist = Math.hypot((shot.centerX - point.shotX) * transform.scaleX, (shot.centerY - point.shotY) * transform.scaleY);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestId = shot.id;
      }
    }
    if (!nearestId || nearestDist > 20) return;
    const id = nearestId;
    setManualGroupOverrides((current) => ({ ...current, [id]: activeManualGroup }));
  };

  // Tap a shot to delete it from the log (e.g. a confirmed false positive).
  const removeShotAtPoint = (clientX: number, clientY: number) => {
    const point = mapClientToShot(clientX, clientY);
    const transform = mapTransformRef.current;
    if (!point || !transform) return;
    let nearestId: string | null = null;
    let nearestDist = Infinity;
    for (const shot of mapShots) {
      const dist = Math.hypot((shot.centerX - point.shotX) * transform.scaleX, (shot.centerY - point.shotY) * transform.scaleY);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestId = shot.id;
      }
    }
    if (!nearestId || nearestDist > 20) return;
    const id = nearestId;
    // Drop it from both the live markers ref and the committed log so it can't
    // reappear, and tidy any manual group override that referenced it.
    shotMarkersRef.current = shotMarkersRef.current.filter((shot) => shot.id !== id);
    setShotLogEntries((current) => current.filter((shot) => shot.id !== id));
    setManualSelectedIds((current) => current.filter((selectedId) => selectedId !== id));
    setManualGroupOverrides((current) => {
      if (!(id in current)) return current;
      const next = { ...current };
      delete next[id];
      return next;
    });
    if (pinnedShotInfo?.shot.id === id) setPinnedShotInfo(null);
  };

  // Drop a hit at a full-frame video-pixel coordinate (same space detected shots
  // use). Works during playback; integrates with the same shot list/markers.
  const addManualShot = (videoX: number, videoY: number) => {
    const video = videoRef.current;
    const t = video?.currentTime ?? 0;
    const caliber = tweakSettings.expectedHoleDiameterInches;
    const diameterPx = pixelsPerInch > 0 && caliber > 0 ? pixelsPerInch * caliber : 0;
    const radius = Math.max(4, Math.round((diameterPx || 12) / 2));
    const nextShotNumber = shotSequenceRef.current + 1;
    shotSequenceRef.current = nextShotNumber;
    const previous = shotMarkersRef.current[shotMarkersRef.current.length - 1] ?? null;
    const entry = makeShotEntry({
      id: `manual_${nextShotNumber}`,
      shotNumber: nextShotNumber,
      frame: frameIndexRef.current,
      videoTimeSec: t,
      timeSincePreviousShotSec: previous ? Math.max(0, t - previous.videoTimeSec) : null,
      centerX: Math.round(videoX),
      centerY: Math.round(videoY),
      radius,
      estimatedDiameterPx: diameterPx,
      estimatedDiameterInches: caliber > 0 ? caliber : null,
      detectionMethod: "manual",
      changeScore: 1,
    });
    shotMarkersRef.current = [...shotMarkersRef.current, entry].slice(
      -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
    );
    setShotLogEntries((current) => [...current, entry]);
  };

  // Tap the live video to place a hit. The overlay fills the video box, so a
  // normalized box coordinate maps straight to a full-frame video pixel.
  const handleVideoMarkTap = (event: React.PointerEvent<HTMLElement>) => {
    if (!manualMarkMode) return;
    const video = videoRef.current;
    if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    const rect = video.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    event.preventDefault();
    const nx = clamp01((event.clientX - rect.left) / rect.width);
    const ny = clamp01((event.clientY - rect.top) / rect.height);
    addManualShot(nx * video.videoWidth, ny * video.videoHeight);
  };

  // Short beep + haptic tick to confirm a shot drag has been engaged.
  const playDragFeedback = () => {
    try {
      navigator.vibrate?.(30);
    } catch {
      // vibrate unsupported
    }
    try {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return;
      const ctx = dragAudioCtxRef.current ?? new Ctor();
      dragAudioCtxRef.current = ctx;
      if (ctx.state === "suspended") void ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.12);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.13);
    } catch {
      // audio unsupported
    }
  };

  // Move a shot to a new full-frame position (clamped to the ROI so it stays on the
  // map), updating both the live markers ref and the committed log.
  const moveShotTo = (shotId: string, shotX: number, shotY: number) => {
    let x = Math.round(shotX);
    let y = Math.round(shotY);
    if (roiPixelRect) {
      x = Math.min(Math.max(x, roiPixelRect.sx), roiPixelRect.sx + roiPixelRect.sw);
      y = Math.min(Math.max(y, roiPixelRect.sy), roiPixelRect.sy + roiPixelRect.sh);
    }
    shotMarkersRef.current = shotMarkersRef.current.map((shot) =>
      shot.id === shotId ? { ...shot, centerX: x, centerY: y } : shot,
    );
    setShotLogEntries((current) => current.map((shot) => (shot.id === shotId ? { ...shot, centerX: x, centerY: y } : shot)));
  };

  const clearLongPress = () => {
    if (longPressTimerRef.current !== null) {
      window.clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
    pendingShotDragRef.current = null;
  };

  const handleMapPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (manualMarkMode) {
      // Tap-to-add on the shot map (mapClientToShot returns full-frame video px).
      event.preventDefault();
      const tapped = mapClientToShot(event.clientX, event.clientY);
      if (tapped) addManualShot(tapped.shotX, tapped.shotY);
      return;
    }
    if (!manualEditMode) return;
    event.preventDefault();
    const point = mapClientToShot(event.clientX, event.clientY);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    // If the press landed on a shot, arm a long-press to enter single-shot drag.
    const shot = shotInfoAtClient(event.clientX, event.clientY)?.shot ?? null;
    clearLongPress();
    if (shot) {
      pendingShotDragRef.current = { shotId: shot.id, startClientX: event.clientX, startClientY: event.clientY };
      longPressTimerRef.current = window.setTimeout(() => {
        longPressTimerRef.current = null;
        const pending = pendingShotDragRef.current;
        pendingShotDragRef.current = null;
        if (!pending) return;
        draggingShotIdRef.current = pending.shotId;
        setDraggingShotId(pending.shotId);
        setManualSelectedIds([pending.shotId]);
        setManualSelectionRect(null);
        manualDragRef.current = null; // dragging a shot cancels box-select
        playDragFeedback();
      }, 450);
    }
    manualDragRef.current = {
      startX: point.shotX,
      startY: point.shotY,
      startClientX: event.clientX,
      startClientY: event.clientY,
      moved: false,
    };
    setManualSelectionRect(null);
  };

  const handleMapPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!manualEditMode) return;
    // Single-shot drag in progress: move the shot to follow the pointer.
    if (draggingShotIdRef.current) {
      const point = mapClientToShot(event.clientX, event.clientY);
      if (point) moveShotTo(draggingShotIdRef.current, point.shotX, point.shotY);
      return;
    }
    const drag = manualDragRef.current;
    if (!drag) return;
    const movedFar = Math.hypot(event.clientX - drag.startClientX, event.clientY - drag.startClientY) > 6;
    if (movedFar) {
      drag.moved = true;
      clearLongPress(); // moving before the hold cancels shot-drag → falls back to box select
    }
    if (!drag.moved) return;
    const point = mapClientToShot(event.clientX, event.clientY);
    if (!point) return;
    setManualSelectionRect({ x0: drag.startX, y0: drag.startY, x1: point.shotX, y1: point.shotY });
  };

  const handleMapPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = manualDragRef.current;
    manualDragRef.current = null;
    clearLongPress();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    // Drop a dragged shot (its position was committed live during the move).
    if (draggingShotIdRef.current) {
      draggingShotIdRef.current = null;
      setDraggingShotId(null);
      try {
        navigator.vibrate?.(15);
      } catch {
        // vibrate unsupported
      }
      return;
    }
    if (!manualEditMode || !drag) return;
    if (!drag.moved) {
      // A tap: assign the nearest shot to the active group.
      assignSingleShotAtPoint(event.clientX, event.clientY);
      return;
    }
    // A drag: select every shot inside the rectangle.
    const point = mapClientToShot(event.clientX, event.clientY);
    if (!point) {
      setManualSelectionRect(null);
      return;
    }
    const minX = Math.min(drag.startX, point.shotX);
    const maxX = Math.max(drag.startX, point.shotX);
    const minY = Math.min(drag.startY, point.shotY);
    const maxY = Math.max(drag.startY, point.shotY);
    const ids = mapShots
      .filter((shot) => shot.centerX >= minX && shot.centerX <= maxX && shot.centerY >= minY && shot.centerY <= maxY)
      .map((shot) => shot.id);
    setManualSelectedIds(ids);
    setManualSelectionRect(null);
  };

  // Hit-test the shot map for a pointer: returns the nearest shot under the
  // cursor (within its drawn radius + a little slack) plus its CSS position
  // relative to the canvas, so a tooltip can be anchored over it.
  const shotInfoAtClient = (
    clientX: number,
    clientY: number,
  ): { shot: ShotLogEntry; left: number; top: number } | null => {
    const canvas = shotGroupMapCanvasRef.current;
    const transform = mapTransformRef.current;
    if (!canvas || !transform) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const internalX = (clientX - rect.left) * (canvas.width / rect.width);
    const internalY = (clientY - rect.top) * (canvas.height / rect.height);
    const shotX = internalX / transform.scaleX + transform.originX;
    const shotY = internalY / transform.scaleY + transform.originY;
    const revealTime = mapRevealTimeRef.current;
    let best: ShotLogEntry | null = null;
    let bestDist = Infinity;
    for (const shot of mapShots) {
      // Only hit shots that are currently drawn (revealed up to the playhead).
      if (Number.isFinite(revealTime) && shot.videoTimeSec > revealTime) continue;
      const dxCanvas = (shot.centerX - shotX) * transform.scaleX;
      const dyCanvas = (shot.centerY - shotY) * transform.scaleY;
      const distCanvas = Math.hypot(dxCanvas, dyCanvas);
      const bulletRadiusCanvas = Math.max(3, (shot.estimatedDiameterPx / 2) * transform.scaleX);
      const hitRadius = Math.max(bulletRadiusCanvas + 4, 12);
      if (distCanvas <= hitRadius && distCanvas < bestDist) {
        best = shot;
        bestDist = distCanvas;
      }
    }
    if (!best) return null;
    const cssPerInternalX = rect.width / canvas.width;
    const cssPerInternalY = rect.height / canvas.height;
    const left = (best.centerX - transform.originX) * transform.scaleX * cssPerInternalX;
    const top = (best.centerY - transform.originY) * transform.scaleY * cssPerInternalY;
    return { shot: best, left, top };
  };

  // Pointer wrappers for the map canvas. Hovering shows a transient tooltip;
  // clicking/tapping a shot pins it open (toggle); clicking empty space clears
  // the pin. In manual-edit mode the existing drag/select handlers still run.
  const onMapPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (manualEditMode) handleMapPointerMove(event);
    // No hover tooltip while dragging a shot.
    if (!draggingShotIdRef.current) setHoverShotInfo(shotInfoAtClient(event.clientX, event.clientY));
  };
  const onMapPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (aimPointGroupId !== null) {
      // Placing an aim point: drop it where tapped, then leave aim-set mode.
      event.preventDefault();
      const point = mapClientToShot(event.clientX, event.clientY);
      if (point) {
        const groupId = aimPointGroupId;
        setGroupAimPoints((current) => ({ ...current, [groupId]: { x: point.shotX, y: point.shotY } }));
      }
      setAimPointGroupId(null);
      return;
    }
    if (manualRemoveMode) {
      event.preventDefault();
      removeShotAtPoint(event.clientX, event.clientY);
      return;
    }
    if (manualMarkMode || manualEditMode) {
      handleMapPointerDown(event);
      return;
    }
    const info = shotInfoAtClient(event.clientX, event.clientY);
    if (info) {
      setPinnedShotInfo((prev) => (prev?.shot.id === info.shot.id ? null : info));
    } else {
      setPinnedShotInfo(null);
    }
  };
  const onMapPointerLeave = () => setHoverShotInfo(null);

  // Clicking a group/Stray/New: assign the current selection to it, or (with no selection)
  // just make it the active group for tap-to-assign.
  const chooseManualGroup = (groupId: number) => {
    if (manualSelectedIds.length > 0) {
      const ids = manualSelectedIds;
      setManualGroupOverrides((current) => {
        const next = { ...current };
        for (const id of ids) next[id] = groupId;
        return next;
      });
      setManualSelectedIds([]);
    }
    setActiveManualGroup(groupId);
  };

  const nextManualGroupId = () =>
    (effectiveGroupIds.length > 0 ? effectiveGroupIds[effectiveGroupIds.length - 1] : 0) + 1;

  const resetManualGroupOverrides = () => {
    setManualGroupOverrides({});
    setManualSelectedIds([]);
    setManualSelectionRect(null);
  };

  const goToNextAfterGeometrySelection = () => {
    // Advance the wizard: to Scan if already calibrated, otherwise to Calibration.
    goToStep(hasScaleCalibration ? STEP_SCAN : STEP_CALIB_METHOD);
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

  // Save a drawn ROI as a reusable target template that auto-detect can match against later.
  const addTargetTemplate = async (nextRect: RoiRect) => {
    if (typeof window === "undefined" || !selectedImagePreviewUrl) return;
    if (nextRect.width < 0.02 || nextRect.height < 0.02) {
      setScanStatus("Target box is too small to save for Auto-pick — draw a slightly larger box.");
      return;
    }
    try {
      const dataUrl = await createTemplateRegionDataUrl(selectedImagePreviewUrl, nextRect);
      const cropped = await loadImageFromUrl(dataUrl);
      const aspect =
        cropped.naturalWidth > 0 && cropped.naturalHeight > 0
          ? cropped.naturalWidth / cropped.naturalHeight
          : nextRect.width / Math.max(nextRect.height, 1e-6);
      const template: TargetTemplate = {
        id: `tpl-${Date.now().toString(36)}-${Math.round(nextRect.x * 1000)}`,
        dataUrl,
        aspect,
        sourceName: selectedVideoName ?? selectedImageName ?? "frame",
        targetWidthInches,
        targetHeightInches,
        createdAt: Date.now(),
        roi: nextRect,
      };
      // Only replace a box drawn over essentially the same spot on the same source, so
      // distinct draws accumulate. Keep newest first, capped.
      const sameSpot = (existing: TargetTemplate) => {
        if (existing.sourceName !== template.sourceName) return false;
        if (Math.abs(existing.aspect - template.aspect) >= 0.05) return false;
        const a = existing.roi;
        if (!a) return true; // legacy entries without a stored ROI: treat as the same spot
        const aCx = a.x + a.width / 2;
        const aCy = a.y + a.height / 2;
        const bCx = nextRect.x + nextRect.width / 2;
        const bCy = nextRect.y + nextRect.height / 2;
        return Math.hypot(aCx - bCx, aCy - bCy) < 0.05;
      };
      const next = [template, ...targetTemplates.filter((existing) => !sameSpot(existing))].slice(
        0,
        MAX_TARGET_TEMPLATES,
      );
      setTargetTemplates(next);
      try {
        window.localStorage.setItem(TARGET_TEMPLATE_LIBRARY_KEY, JSON.stringify(next));
        setScanStatus(`Saved target to Auto-pick library (${next.length} saved).`);
      } catch {
        setScanStatus("Target box drawn, but saving to the Auto-pick library failed (storage full?).");
      }
    } catch {
      setScanStatus("Could not save the target box for Auto-pick (image crop failed).");
    }
  };

  const clearTargetTemplates = () => {
    setTargetTemplates([]);
    if (typeof window === "undefined") return;
    try {
      window.localStorage.removeItem(TARGET_TEMPLATE_LIBRARY_KEY);
    } catch {
      // Ignore.
    }
  };

  const clearRoiSelection = () => {
    roiRectRef.current = null;
    setRoiRect(null);
    draftRoiRectRef.current = null;
    setDraftRoiRect(null);
    clearRoiLineCalibrationState();
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

    roiStartRef.current = { x, y }; // the top-left corner of the box
    setIsSelectingRoi(true);
    // Only a *draft* while dragging — the real ROI isn't committed until confirm.
    const nextRect = { x: x / bounds.width, y: y / bounds.height, width: 0, height: 0 };
    draftRoiRectRef.current = nextRect;
    setDraftRoiRect(nextRect);
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
    draftRoiRectRef.current = nextRect;
    setDraftRoiRect(nextRect);
  };

  // Drag finished (mouseup/leave): open the confirm modal with a crop preview.
  // Nothing downstream advances until the user confirms.
  const endRoiSelection = () => {
    if (!isSelectingRoi) return;
    setIsSelectingRoi(false);
    roiStartRef.current = null;
    const draft = draftRoiRectRef.current;
    if (!draft || draft.width < 0.01 || draft.height < 0.01) {
      draftRoiRectRef.current = null;
      setDraftRoiRect(null);
      return;
    }
    setConfirmRoiPreview(null);
    setConfirmRoiOpen(true);
    if (selectedImagePreviewUrl) {
      void createTemplateRegionDataUrl(selectedImagePreviewUrl, draft)
        .then((url) => setConfirmRoiPreview(url))
        .catch(() => setConfirmRoiPreview(null));
    }
  };

  const confirmRoiSelection = () => {
    const draft = draftRoiRectRef.current;
    setConfirmRoiOpen(false);
    setConfirmRoiPreview(null);
    if (!draft || draft.width < 0.01 || draft.height < 0.01) return;
    roiRectRef.current = draft;
    setRoiRect(draft);
    lastAutoCalibrationKeyRef.current = "";
    void persistTemplateRegionSelection(draft);
    void addTargetTemplate(draft);
    draftRoiRectRef.current = null;
    setDraftRoiRect(null);
    // Confirming the target area advances straight to the next step.
    goToNextAfterGeometrySelection();
  };

  const retryRoiSelection = () => {
    draftRoiRectRef.current = null;
    setDraftRoiRect(null);
    setConfirmRoiOpen(false);
    setConfirmRoiPreview(null);
  };

  // Composite the scan's persistent background: the first (reference) frame with
  // the given last frame drawn over it at 50% opacity. Runs at scan end.
  const buildScanBlendBackground = (video: HTMLVideoElement | null) => {
    if (!video || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    const lastCanvas = document.createElement("canvas");
    lastCanvas.width = w;
    lastCanvas.height = h;
    const lastCtx = lastCanvas.getContext("2d");
    if (!lastCtx) return;
    try {
      lastCtx.drawImage(video, 0, 0, w, h);
    } catch {
      return;
    }
    const firstUrl = selectedImagePreviewUrl;
    if (!firstUrl) {
      try {
        setScanBlendBackgroundUrl(lastCanvas.toDataURL("image/png"));
      } catch {
        /* ignore */
      }
      return;
    }
    const firstImg = new Image();
    firstImg.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(firstImg, 0, 0, w, h); // first frame (base)
      ctx.globalAlpha = 0.5;
      ctx.drawImage(lastCanvas, 0, 0); // last frame at 50%
      ctx.globalAlpha = 1;
      try {
        setScanBlendBackgroundUrl(canvas.toDataURL("image/png"));
      } catch {
        /* same-origin frames only; ignore taint errors */
      }
    };
    firstImg.src = firstUrl;
  };

  const stopScan = () => {
    // A genuine Stop (not a restart): persist the blended result and show it.
    if (!restartScanRequestedRef.current) {
      buildScanBlendBackground(scanVideoRef.current ?? analyzedPlaybackVideoRef.current ?? videoRef.current);
      setShotMapLiveStream(false);
    }
    stopRequestedRef.current = true;
    setStabilizationPx(null);
    lastStabilizationPxRef.current = -1;
    stopAnalyzedPlayback();
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (videoRef.current && captureMode === "upload") videoRef.current.pause();
    if (scanVideoRef.current) {
      const shouldPauseScanVideo = !(captureMode === "stream" && streamCameraActive && scanVideoRef.current === videoRef.current);
      if (shouldPauseScanVideo) {
        scanVideoRef.current.pause();
      }
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
    clearAnalysisCanvases();
    relaxedShotGateNoticeRef.current = false;
    lastHistogramDeltaPctRef.current = 0;
    resetShotFlowState();
    shotMarkersRef.current = [];
    shotSequenceRef.current = 0;
    shotPresenceRef.current = new Map();
    committedShotPointsRef.current = [];
    contourShotTracksRef.current = [];
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
    liveClusterLastUpdatedAtMsRef.current = 0;
    liveContourGroupVisualsRef.current = { updatedAtMs: 0, regionColors: [], regionGroupLabels: [] };
    setIsScanning(false);
    setDetectionEnabled(false);
    setDetectionConfidence(null);
    lastUiDetectionEnabledRef.current = false;
    lastUiDetectionConfidenceRef.current = -1;
    setScanStatus("Scan stopped");
  };

  const startScan = async (options?: StartScanOptions) => {
    const isStreamScan = captureMode === "stream";
    const requestedWindow = isStreamScan ? null : options?.forcedWindow ?? null;
    const requiresAudioPipeline = !requestedWindow && !isStreamScan;
    const runtimeCv = window.cv as unknown as CvApi | undefined;

    if (!opencvReady || !runtimeCv) {
      setScanStatus("OpenCV.js is not ready yet.");
      return;
    }
    if (requiresAudioPipeline && (!howlerReady || !window.Howl)) {
      setScanStatus("Howler.js is not ready yet.");
      return;
    }

    const previewVideoEl = videoRef.current;
    const canvasEl = processingCanvasRef.current;
    const overlayEl = overlayCanvasRef.current;

    if (!previewVideoEl || !canvasEl || !overlayEl) {
      setScanStatus("Video preview or analysis canvases are unavailable.");
      return;
    }
    if (!isStreamScan && (!selectedVideoFile || !selectedVideoPreviewUrl)) {
      setScanStatus("Upload a reference video first.");
      return;
    }
    if (isStreamScan && (!streamCameraActive || !previewVideoEl.srcObject)) {
      setScanStatus("Start the device camera stream first.");
      return;
    }

    if (!selectedImagePreviewUrl) {
      setScanStatus("Capture a reference frame from the video, then draw the target geometry.");
      return;
    }

    const cv = window.cv as unknown as CvApi | undefined;
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

      let videoEl: HTMLVideoElement;
      if (isStreamScan) {
        videoEl = previewVideoEl;
        videoEl.muted = true;
        videoEl.playsInline = true;
        scanVideoRef.current = videoEl;
      } else {
        previewVideoEl.pause();
        videoEl = document.createElement("video");
        videoEl.src = selectedVideoPreviewUrl!;
        videoEl.preload = "auto";
        // Start muted so autoplay is allowed (the play() call happens after several
        // awaits, past the user-gesture window). We unmute right after play()
        // succeeds so the clip's audio is still audible during the scan.
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
      yellowGreenTimelineRef.current = [];
      shotMarkersRef.current = [];
      shotSequenceRef.current = 0;
      shotPresenceRef.current = new Map();
      committedShotPointsRef.current = [];
      contourShotTracksRef.current = [];
      changeHoleTrackerRef.current = [];
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
      liveClusterLastUpdatedAtMsRef.current = 0;
      liveContourGroupVisualsRef.current = { updatedAtMs: 0, regionColors: [], regionGroupLabels: [] };
      audioRmsTimelineRef.current = [];
      audioMeanDbfsRef.current = -120;
      audioThresholdDbfsRef.current = -120;
      relaxedShotGateNoticeRef.current = false;
      lastHistogramDeltaPctRef.current = 0;
      resetShotFlowState();
      setLogEntries([]);
      setShotLogEntries([]);
      setManualGroupOverrides({});
      setManualSelectedIds([]);
      setManualSelectionRect(null);
      setTimelineCursorSec(null);
      setLastDetection(null);
      setLastShot(null);
      setDetectionEnabled(false);
      setDetectionConfidence(0);
      lastUiDetectionEnabledRef.current = false;
      lastUiDetectionConfidenceRef.current = 0;
      setSpikeMetadata([]);
      setAudioSignatureCatalog([]);
      setAudioSprites({});
      setSpritesReady(false);
      setAudioCaptureInfo(null);
      setAudioCaptureError(null);
      setIsScanning(true);
      setScanStatus(
        isStreamScan
          ? "Starting live device-camera OpenCV analysis..."
          : requestedWindow
          ? `Preparing focused analysis (${requestedWindow.start.toFixed(2)}s-${requestedWindow.end.toFixed(2)}s)...`
          : FORCE_OPEN_SHOT_GATES
            ? "Starting concurrent audio + OpenCV analysis (all shot gates open)..."
            : "Starting concurrent audio + OpenCV analysis...",
      );

      const durationSec = isStreamScan
        ? Number.POSITIVE_INFINITY
        : Number.isFinite(videoEl.duration) && videoEl.duration > 0
          ? videoEl.duration
          : 0;
      if (!isStreamScan && durationSec <= 0) {
        templateGray.delete();
        if (trackingHist) trackingHist.delete();
        setIsScanning(false);
        setScanStatus("Video metadata unavailable. Press play once, then retry.");
        return;
      }

      if (isStreamScan) {
        spikeEventsRef.current = [];
        spikeWindowsRef.current = [{ start: 0, end: Number.POSITIVE_INFINITY }];
        setSpikeMetadata([]);
        setAudioSignatureCatalog([]);
        setAudioSprites({});
        setSpritesReady(false);
        audioReadyRef.current = true;
        audioRmsTimelineRef.current = [];
        audioMeanDbfsRef.current = -120;
        audioThresholdDbfsRef.current = -120;
        if (howlRef.current) {
          howlRef.current.unload();
          howlRef.current = null;
        }
        setScanStatus("Live camera stream ready. Running OpenCV analysis.");
      } else if (requestedWindow) {
        const boundedStart = Math.max(0, Math.min(durationSec, requestedWindow.start));
        const boundedEnd = Math.max(boundedStart, Math.min(durationSec, requestedWindow.end));
        const fallbackTimeSec = boundedStart + (boundedEnd - boundedStart) / 2;
        const requestedSpike = options?.forcedSpike;
        const normalizedSpike: SpikeMetadata = {
          id:
            requestedSpike?.id ??
            `spike_window_${boundedStart.toFixed(3).replace(".", "_")}_${boundedEnd.toFixed(3).replace(".", "_")}`,
          timeSec: requestedSpike ? Math.max(boundedStart, Math.min(boundedEnd, requestedSpike.timeSec)) : fallbackTimeSec,
          strength: requestedSpike?.strength ?? 0,
          spriteStartMs: Math.round(Math.max(0, boundedStart) * 1000),
          spriteDurationMs: Math.max(1, Math.round((boundedEnd - boundedStart) * 1000)),
          windowStartSec: boundedStart,
          windowEndSec: boundedEnd,
          subPeakTimesSec: requestedSpike?.subPeakTimesSec ?? [],
          signatureId: requestedSpike?.signatureId ?? 1,
          signatureKey: requestedSpike?.signatureKey ?? "forced_window",
        };
        setSpikeMetadata([normalizedSpike]);
        setAudioSignatureCatalog([]);
        spikeEventsRef.current = [normalizedSpike];
        spikeWindowsRef.current = [{ start: boundedStart, end: boundedEnd }];
        setAudioSprites({});
        setSpritesReady(false);
        audioReadyRef.current = true;
        audioRmsTimelineRef.current = [];
        audioMeanDbfsRef.current = -120;
        audioThresholdDbfsRef.current = -120;
        if (howlRef.current) {
          howlRef.current.unload();
          howlRef.current = null;
        }
        setScanStatus(`Running focused analysis in ${boundedStart.toFixed(2)}s-${boundedEnd.toFixed(2)}s window.`);
      } else {
        const HowlCtor = window.Howl;
        if (!HowlCtor) {
          throw new Error("Howler constructor is unavailable.");
        }
        const audioTask = (async () => {
          const audioContext = new AudioContext();
          const fileBuffer = await selectedVideoFile!.arrayBuffer();
          let decodedAudio: AudioBuffer;
          try {
            decodedAudio = await audioContext.decodeAudioData(fileBuffer.slice(0));
          } catch (decodeError) {
            await audioContext.close();
            const detail = decodeError instanceof Error ? decodeError.message : String(decodeError);
            throw new Error(
              `Browser could not decode audio from "${selectedVideoFile!.name}". The file may have no audio track or an unsupported codec. (${detail})`,
            );
          }
          await audioContext.close();

          const { spikes, spriteMap, signatureCatalog, rmsTimeline, meanDbfs, thresholdDbfs } = detectAudioSpikes(
            decodedAudio,
            durationSec,
            tweakSettings,
          );
          setSpikeMetadata(spikes);
          setAudioSignatureCatalog(signatureCatalog);
          setAudioSprites(spriteMap);
          spikeEventsRef.current = spikes;
          audioRmsTimelineRef.current = rmsTimeline;
          audioMeanDbfsRef.current = meanDbfs;
          audioThresholdDbfsRef.current = thresholdDbfs;

          let minDbfs = Number.POSITIVE_INFINITY;
          let maxDbfs = Number.NEGATIVE_INFINITY;
          for (const sample of rmsTimeline) {
            if (sample.dbfs < minDbfs) minDbfs = sample.dbfs;
            if (sample.dbfs > maxDbfs) maxDbfs = sample.dbfs;
          }
          const rmsHopSec =
            rmsTimeline.length > 1
              ? (rmsTimeline[rmsTimeline.length - 1].timeSec - rmsTimeline[0].timeSec) / (rmsTimeline.length - 1)
              : 0;
          setAudioCaptureInfo({
            sampleRate: decodedAudio.sampleRate,
            channels: decodedAudio.numberOfChannels,
            durationSec: decodedAudio.duration,
            totalSamples: decodedAudio.length,
            rmsSampleCount: rmsTimeline.length,
            rmsHopSec,
            meanDbfs,
            thresholdDbfs,
            minDbfs: Number.isFinite(minDbfs) ? minDbfs : meanDbfs,
            maxDbfs: Number.isFinite(maxDbfs) ? maxDbfs : meanDbfs,
            spikeCount: spikes.length,
            signatureCount: signatureCatalog.length,
          });

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
              src: [selectedVideoPreviewUrl!],
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
          setScanStatus(
            `Detected ${spikes.length} audio spike${spikes.length === 1 ? "" : "s"}. Scanning ${(
              tweakSettings.spikeWindowHalfSec * 2
            ).toFixed(1)}s change-detection windows around each one (dead time is skipped).`,
          );
        })().catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          console.error("[audio] spike pipeline failed:", error);
          audioReadyRef.current = true;
          spikeWindowsRef.current = [{ start: 0, end: durationSec }];
          setSpritesReady(false);
          setAudioCaptureError(message);
          setScanStatus(`Audio spike analysis failed (${message}). Scanning full video range.`);
        });

        await audioTask;
      }

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
      // Splatter-pass cache: the HSV top-hat is by far the most expensive stage,
      // and splatter marks are permanent — recomputing every other frame is
      // plenty, and halves the heaviest per-frame cost.
      let lastYellowGreen: { mask: Uint8Array; hits: YellowGreenHit[]; width: number; height: number; frame: number } | null =
        null;
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
        return true;
      };
      // First-vs-last background subtraction: every real bullet hole is
      // PERMANENT, so it must appear when the LAST frame is subtracted from the
      // scan-start baseline; transients (brass in flight, glare, hands, smoke)
      // never do. Capture the end state up front: seek to the end, locate the
      // target (template match = coarse alignment), grab its patch, fine-align
      // it to the baseline with a shift search, then diff. During the scan a
      // track only commits if its location also changed in this end-state mask.
      // Best-effort: any failure (or a degenerate mask) simply disables the gate.
      let permanentChangeMask: Uint8Array | null = null;
      let permanentChangeMaskWidth = 0;
      let permanentChangeMaskHeight = 0;
      if (!isStreamScan && Number.isFinite(durationSec) && durationSec > 0.2) {
        try {
          setScanStatus("Capturing end-state frame for permanent-change validation...");
          await seekVideo(videoEl, Math.max(0, durationSec - 0.05));
          const lastFrameWidth = videoEl.videoWidth;
          const lastFrameHeight = videoEl.videoHeight;
          if (lastFrameWidth > 0 && lastFrameHeight > 0) {
            const lastFrameCanvas = document.createElement("canvas");
            lastFrameCanvas.width = lastFrameWidth;
            lastFrameCanvas.height = lastFrameHeight;
            const lastFrameContext = lastFrameCanvas.getContext("2d", { willReadFrequently: true });
            if (lastFrameContext) {
              lastFrameContext.drawImage(videoEl, 0, 0, lastFrameWidth, lastFrameHeight);
              // Coarse alignment: find the target in the last frame.
              const lastFrameMat = cv.imread(lastFrameCanvas);
              const lastFrameGrayMat = new cv.Mat();
              cv.cvtColor(lastFrameMat, lastFrameGrayMat, cv.COLOR_RGBA2GRAY);
              lastFrameMat.delete();
              const lastMatchResult = new cv.Mat();
              cv.matchTemplate(lastFrameGrayMat, templateGray, lastMatchResult, cv.TM_CCOEFF_NORMED);
              const lastMatchMinMax = cv.minMaxLoc(lastMatchResult);
              lastMatchResult.delete();
              lastFrameGrayMat.delete();
              const lastLoc = clampTemplateLocToFrame(
                lastMatchMinMax.maxLoc.x,
                lastMatchMinMax.maxLoc.y,
                lastFrameWidth,
                lastFrameHeight,
                templateGray.cols,
                templateGray.rows,
              );
              // Same adaptive probe sizing the scan loop uses, so the mask's
              // coordinates line up 1:1 with the live patch coordinates.
              let lastProbeSize = Math.max(16, Math.round(tweakSettings.spikeStandardProbeSize));
              const lastHoleFullPx =
                pixelsPerInch > 0 && tweakSettings.expectedHoleDiameterInches > 0
                  ? pixelsPerInch * tweakSettings.expectedHoleDiameterInches
                  : 0;
              if (lastHoleFullPx > 0 && templateGray.cols > 0) {
                lastProbeSize = Math.min(
                  PROBE_MAX_SIZE_PX,
                  Math.max(lastProbeSize, Math.ceil((PROBE_MIN_HOLE_DIAMETER_PX * templateGray.cols) / lastHoleFullPx)),
                );
              }
              const lastPatchCanvas = document.createElement("canvas");
              lastPatchCanvas.width = lastProbeSize;
              lastPatchCanvas.height = lastProbeSize;
              const lastPatchContext = lastPatchCanvas.getContext("2d", { willReadFrequently: true });
              const endBaselineCanvas = document.createElement("canvas");
              endBaselineCanvas.width = lastProbeSize;
              endBaselineCanvas.height = lastProbeSize;
              const endBaselineContext = endBaselineCanvas.getContext("2d", { willReadFrequently: true });
              if (lastPatchContext && endBaselineContext) {
                lastPatchContext.drawImage(
                  lastFrameCanvas,
                  lastLoc.x,
                  lastLoc.y,
                  templateGray.cols,
                  templateGray.rows,
                  0,
                  0,
                  lastProbeSize,
                  lastProbeSize,
                );
                const lastPatchGray = rgbaToGrayChannel(
                  lastPatchContext.getImageData(0, 0, lastProbeSize, lastProbeSize).data,
                  lastProbeSize,
                  lastProbeSize,
                );
                endBaselineContext.drawImage(
                  activeTemplateCanvas,
                  0,
                  0,
                  activeTemplateCanvas.width,
                  activeTemplateCanvas.height,
                  0,
                  0,
                  lastProbeSize,
                  lastProbeSize,
                );
                const endBaselineGray = rgbaToGrayChannel(
                  endBaselineContext.getImageData(0, 0, lastProbeSize, lastProbeSize).data,
                  lastProbeSize,
                  lastProbeSize,
                );
                // Fine alignment: integer-shift registration with a wider search
                // than the per-frame stabilizer (one-time cost, and first↔last
                // drift can exceed the live ±4px budget).
                const endShift = estimatePatchShift(
                  lastPatchGray,
                  endBaselineGray,
                  lastProbeSize,
                  lastProbeSize,
                  Math.max(PATCH_REGISTRATION_MAX_SHIFT, 12),
                  2,
                );
                const alignedEndBaseline =
                  endShift.dx === 0 && endShift.dy === 0
                    ? endBaselineGray
                    : shiftGrayPatch(endBaselineGray, lastProbeSize, lastProbeSize, endShift.dx, endShift.dy);
                const endStateMask = closeBinaryMask(
                  openBinaryMask(
                    buildGrayDifferenceMask(lastPatchGray, alignedEndBaseline, CONTRAST_HOLE_DIFF_THRESHOLD),
                    lastProbeSize,
                    lastProbeSize,
                    3,
                  ),
                  lastProbeSize,
                  lastProbeSize,
                  3,
                );
                // Sanity: an EMPTY mask means the capture failed (blank frame,
                // lost target) — a shot video must show change. A mostly-lit
                // mask means misalignment or a lighting snap. Either way the
                // gate is unreliable, so it turns itself off.
                const endChangedPixels = countMaskPixels(endStateMask);
                const endChangedPct = (endChangedPixels / Math.max(1, lastProbeSize * lastProbeSize)) * 100;
                if (endChangedPixels > 0 && endChangedPct <= 50) {
                  permanentChangeMask = endStateMask;
                  permanentChangeMaskWidth = lastProbeSize;
                  permanentChangeMaskHeight = lastProbeSize;
                }
                // Detection trail: show the end frame with its permanent changes.
                drawTrailMask(trailEndCanvasRef.current, lastPatchGray, endStateMask, lastProbeSize, lastProbeSize);
              }
            }
          }
        } catch {
          permanentChangeMask = null;
        }
        try {
          await seekVideo(videoEl, 0);
        } catch {
          // If the rewind fails the play() below starts wherever we are; the
          // scan loop's window logic copes, so don't abort over it.
        }
      }

      syncTemplateBaselinePatch();
      try {
        await videoEl.play();
        videoEl.volume = playbackVolumeRef.current;
        // Now that it's playing (muted autoplay always allowed), try to unmute so
        // the clip's audio is audible during the scan. If a stricter browser blocks
        // unmuted playback and pauses the element, fall back to a silent muted scan
        // so it never stalls. Stream scans stay muted (mic→speaker feedback).
        if (!isStreamScan && !scanAudioMuted) {
          videoEl.muted = false;
          void videoEl.play().catch(() => {
            videoEl.muted = true;
            void videoEl.play().catch(() => undefined);
          });
        }
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

          // Spike-gated scanning: with audio spikes available, only the short
          // window around each bang needs frame-by-frame change detection —
          // between windows nothing new can appear on the target. Stutter
          // through the video by seeking straight to the next window instead of
          // playing the dead time (detection state and the scan-start baseline
          // carry across the jump). Stream scans and no-spike fallbacks use a
          // single full-range window, so this never skips for them.
          if (
            !isStreamScan &&
            audioGatedScanRef.current &&
            windows.length > 0 &&
            windowIndexAtTime(currentSec, windows) === -1
          ) {
            if (!videoEl.seeking) {
              let nextWindowStartSec = Number.POSITIVE_INFINITY;
              for (const window of windows) {
                if (window.start > currentSec && window.start < nextWindowStartSec) {
                  nextWindowStartSec = window.start;
                }
              }
              // Past the last window: jump to the end so the scan wraps up.
              videoEl.currentTime = Number.isFinite(nextWindowStartSec)
                ? Math.min(nextWindowStartSec, Math.max(0, durationSec))
                : Math.max(0, durationSec);
            }
            // Keep the detection-trail sound track moving while skipping.
            if (showDetectionTrailRef.current) {
              const trailNowMs = performance.now();
              if (trailNowMs - lastTrailDrawAtMsRef.current >= 200) {
                lastTrailDrawAtMsRef.current = trailNowMs;
                drawTrailAudio(
                  trailAudioCanvasRef.current,
                  audioRmsTimelineRef.current,
                  audioMeanDbfsRef.current,
                  audioThresholdDbfsRef.current,
                  spikeEventsRef.current,
                  windows,
                  currentSec,
                  durationSec,
                );
                setTrailInfo((prev) => (prev ? { ...prev, inWindow: false, timeSec: currentSec } : prev));
              }
            }
            animationFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }

          const spikeTimeline = spikeEventsRef.current;
          const nearestSpikeInfo = getNearestSpikeAtTime(currentSec, spikeTimeline);
          // Resolution guard: enlarge the analysis patch when the calibrated
          // hole would land below ~5px across at the configured size —
          // otherwise small-caliber holes vanish into the downsampling and
          // "obvious" changes never form a detectable blob.
          let desiredProbeSize = Math.max(16, Math.round(tweakSettings.spikeStandardProbeSize));
          const expectedHoleFullPx =
            pixelsPerInch > 0 && tweakSettings.expectedHoleDiameterInches > 0
              ? pixelsPerInch * tweakSettings.expectedHoleDiameterInches
              : 0;
          if (expectedHoleFullPx > 0 && templateGray.cols > 0) {
            const neededForResolution = Math.ceil((PROBE_MIN_HOLE_DIAMETER_PX * templateGray.cols) / expectedHoleFullPx);
            desiredProbeSize = Math.min(PROBE_MAX_SIZE_PX, Math.max(desiredProbeSize, neededForResolution));
          }
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
          let sqdiffScore = -1;
          let sqdiffLoc: { x: number; y: number } | null = null;
          let sqdiffUsed = false;
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

          const clampedPrimaryLoc = clampTemplateLocToFrame(
            bestLoc.x,
            bestLoc.y,
            frameGray.cols,
            frameGray.rows,
            templateGray.cols,
            templateGray.rows,
          );
          const primaryNccScore = Math.max(0, Math.min(1, Number.isFinite(bestScore) ? bestScore : 0));
          const primarySadScore = computeSadTemplateSimilarity(frameGray, templateGray, clampedPrimaryLoc.x, clampedPrimaryLoc.y);
          let fusedTemplateScore = blendTemplateSimilarity(primaryNccScore, primarySadScore);
          let fusedTemplateLoc = clampedPrimaryLoc;

          const runSqDiffAssist = primaryNccScore * 100 < Math.max(centerAcceptThresholdPct + 4, matchThreshold + 6);
          if (runSqDiffAssist && cv.TM_SQDIFF_NORMED !== undefined) {
            const sqdiffResult = new cv.Mat();
            cv.matchTemplate(frameGray, templateGray, sqdiffResult, cv.TM_SQDIFF_NORMED);
            const sqdiffMinMax = cv.minMaxLoc(sqdiffResult);
            sqdiffResult.delete();
            if (Number.isFinite(sqdiffMinMax.minVal ?? Number.NaN) && sqdiffMinMax.minLoc) {
              sqdiffScore = Math.max(0, Math.min(1, 1 - (sqdiffMinMax.minVal ?? 1)));
              sqdiffLoc = clampTemplateLocToFrame(
                sqdiffMinMax.minLoc.x,
                sqdiffMinMax.minLoc.y,
                frameGray.cols,
                frameGray.rows,
                templateGray.cols,
                templateGray.rows,
              );
              const sqdiffSadScore = computeSadTemplateSimilarity(frameGray, templateGray, sqdiffLoc.x, sqdiffLoc.y);
              const sqdiffFusedScore = blendTemplateSimilarity(sqdiffScore, sqdiffSadScore);
              if (sqdiffFusedScore > fusedTemplateScore + 0.015) {
                fusedTemplateScore = sqdiffFusedScore;
                fusedTemplateLoc = sqdiffLoc;
                sqdiffUsed = true;
              }
            }
          }

          bestScore = fusedTemplateScore;
          bestLoc = fusedTemplateLoc;

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
          const shotsDetectedThisFrame: ShotLogEntry[] = [];
          let histogramDeltaPctForFrame = lastHistogramDeltaPctRef.current;
          let yellowGreenHitCountForFrame = 0;
          let contourWindowSnapshotForFrame: ContourWindowFrameSnapshot | null = null;
          let yellowGreenSnapshotForFrame: YellowGreenFrameSnapshot | null = null;
          // When the new change detector + detailed views are both on, we repaint
          // the mask/probe views with the detector's own debug overlay (captured
          // here, drawn after the legacy views so it wins).
          let changeDetectorDebugForFrame:
            | {
                gray: Uint8Array;
                width: number;
                height: number;
                mask: Uint8Array;
                candidates: ChangeHoleDebugCandidate[];
                info: { changedPixels: number; bandMinPx: number; bandMaxPx: number; persistence: number };
              }
            | null = null;
          const changeProbeContext = changeProbeCanvas.getContext("2d");
          // The four "Probe / Contour / Mask / Yellow-green" views are pure debug
          // visualization. Skip all their per-frame drawing while they're hidden
          // (the default) — this is what made scanning slow.
          const detailedViewsVisible = showDetailedViewsRef.current;
          const processedContourCanvas = processedContourCanvasRef.current;
          const processedContourContext = detailedViewsVisible ? (processedContourCanvas?.getContext("2d") ?? null) : null;
          const processedPatchCanvas = processedPatchCanvasRef.current;
          const processedPatchContext = detailedViewsVisible ? (processedPatchCanvas?.getContext("2d") ?? null) : null;
          const processedMaskCanvas = processedMaskCanvasRef.current;
          const processedMaskContext = detailedViewsVisible ? (processedMaskCanvas?.getContext("2d") ?? null) : null;
          const processedYellowGreenCanvas = processedYellowGreenCanvasRef.current;
          const processedYellowGreenContext = detailedViewsVisible
            ? (processedYellowGreenCanvas?.getContext("2d") ?? null)
            : null;
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
              if (processedYellowGreenCanvas) {
                processedYellowGreenCanvas.width = patchWidth;
                processedYellowGreenCanvas.height = patchHeight;
              }
              contourWindowSnapshotForFrame = {
                frame: frameIndexRef.current,
                videoTimeSec: videoEl.currentTime,
                patchWidthPx: patchWidth,
                patchHeightPx: patchHeight,
                changedPixels: 0,
                maskRuns: [],
                regions: [],
              };
              yellowGreenSnapshotForFrame = {
                frame: frameIndexRef.current,
                videoTimeSec: videoEl.currentTime,
                patchWidthPx: patchWidth,
                patchHeightPx: patchHeight,
                changedPixels: 0,
                maskRuns: [],
                hits: [],
              };

              // Bright-splatter (shoot-n-see) color detection is optional —
              // contrast is the primary driver. Skip the HSV/top-hat pass when
              // off, and when on, decimate it (every 2nd processed frame,
              // reusing the last result) — it dwarfs every other stage.
              let yellowGreenDetection: { mask: Uint8Array; hits: YellowGreenHit[] };
              if (!detectBrightColorsRef.current) {
                yellowGreenDetection = { mask: new Uint8Array(patchWidth * patchHeight), hits: [] };
              } else if (
                lastYellowGreen &&
                lastYellowGreen.width === patchWidth &&
                lastYellowGreen.height === patchHeight &&
                frameIndexRef.current - lastYellowGreen.frame < 2
              ) {
                yellowGreenDetection = { mask: lastYellowGreen.mask, hits: lastYellowGreen.hits };
              } else {
                yellowGreenDetection = detectHitsBrightYellowGreen(currentPatchRgba, patchWidth, patchHeight);
                lastYellowGreen = {
                  mask: yellowGreenDetection.mask,
                  hits: yellowGreenDetection.hits,
                  width: patchWidth,
                  height: patchHeight,
                  frame: frameIndexRef.current,
                };
              }
              const yellowGreenMask = yellowGreenDetection.mask;
              const yellowGreenHits = yellowGreenDetection.hits;
              yellowGreenHitCountForFrame = yellowGreenHits.length;
              const yellowGreenChangedPixels = countMaskPixels(yellowGreenMask);
              yellowGreenSnapshotForFrame = {
                frame: frameIndexRef.current,
                videoTimeSec: videoEl.currentTime,
                patchWidthPx: patchWidth,
                patchHeightPx: patchHeight,
                changedPixels: yellowGreenChangedPixels,
                maskRuns: encodeBinaryMaskRuns(yellowGreenMask),
                hits: yellowGreenHits.slice(0, 20),
              };
              if (processedYellowGreenContext) {
                drawYellowGreenWindowView(
                  processedYellowGreenContext,
                  patchWidth,
                  patchHeight,
                  yellowGreenMask,
                  yellowGreenHits,
                  "Yellow-green / top-hat (live)",
                  currentPatchRgba,
                );
              }

              if (
                (!templateBaselinePatch || templateBaselinePatch.length !== currentPatchRgba.length) &&
                !syncTemplateBaselinePatch()
              ) {
                lastHistogramDeltaPctRef.current = 0;
                histogramDeltaPctForFrame = 0;
                resetShotFlowState();
                if (processedContourContext) {
                  drawProcessedContourView(
                    processedContourContext,
                    currentPatchRgba,
                    null,
                    patchWidth,
                    patchHeight,
                    null,
                    [],
                  );
                }
                if (processedMaskContext) {
                  drawBinaryMaskWindowView(
                    processedMaskContext,
                    patchWidth,
                    patchHeight,
                    null,
                    "Binary mask (live)",
                    currentPatchRgba,
                  );
                }
              } else {
                const activeBaselinePatch =
                  templateBaselinePatch && templateBaselinePatch.length === currentPatchRgba.length
                    ? templateBaselinePatch
                    : null;
                if (!activeBaselinePatch) {
                  lastHistogramDeltaPctRef.current = 0;
                  histogramDeltaPctForFrame = 0;
                  resetShotFlowState();
                  if (processedContourContext) {
                    drawProcessedContourView(
                      processedContourContext,
                      currentPatchRgba,
                      null,
                      patchWidth,
                      patchHeight,
                      null,
                      [],
                    );
                  }
                  if (processedMaskContext) {
                    drawBinaryMaskWindowView(
                      processedMaskContext,
                      patchWidth,
                      patchHeight,
                      null,
                      "Binary mask (live)",
                      currentPatchRgba,
                    );
                  }
                } else {
                  const shotCooldownMs = FORCE_OPEN_SHOT_GATES ? 0 : Math.max(0, Math.round(tweakSettings.shotCooldownMs));
                  const nowMs = performance.now();

                  const pending = pendingShotCandidateRef.current;
                  if (pending && nowMs - pending.lastSeenAtMs > SHOT_PERSISTENCE_MAX_GAP_MS) {
                    pendingShotCandidateRef.current = null;
                  }

                  const currentPatchGray = rgbaToGrayChannel(currentPatchRgba, patchWidth, patchHeight);
                  const baselinePatchGray = rgbaToGrayChannel(activeBaselinePatch, patchWidth, patchHeight);

                  // STAGE 1 — stabilization: register the baseline to this frame's patch
                  // (small integer-shift search) to cancel camera shake / residual
                  // jitter before differencing, so a wobbling camera doesn't read as hits.
                  const patchShift = USE_PATCH_REGISTRATION
                    ? estimatePatchShift(
                        currentPatchGray,
                        baselinePatchGray,
                        patchWidth,
                        patchHeight,
                        PATCH_REGISTRATION_MAX_SHIFT,
                        PATCH_REGISTRATION_STRIDE,
                      )
                    : { dx: 0, dy: 0 };
                  // Surface the correction magnitude to the UI (throttled to whole px).
                  const stabMag = Math.round(Math.hypot(patchShift.dx, patchShift.dy));
                  if (stabMag !== lastStabilizationPxRef.current) {
                    lastStabilizationPxRef.current = stabMag;
                    setStabilizationPx(stabMag);
                  }
                  const alignedBaselineGray =
                    patchShift.dx === 0 && patchShift.dy === 0
                      ? baselinePatchGray
                      : shiftGrayPatch(baselinePatchGray, patchWidth, patchHeight, patchShift.dx, patchShift.dy);
                  const alignedBaselineRgba =
                    patchShift.dx === 0 && patchShift.dy === 0
                      ? activeBaselinePatch
                      : shiftRgbaPatch(activeBaselinePatch, patchWidth, patchHeight, patchShift.dx, patchShift.dy);

                  // STAGE 2 — background subtraction: one clean, polarity-agnostic
                  // |current − baseline| mask at a noise-proof threshold, tidied
                  // with open/close morphology. This single mask is the ONLY
                  // change signal — no multi-mask unions, tile votes, temporal
                  // gates, or histogram fallbacks.
                  const contrastHoleMask = closeBinaryMask(
                    openBinaryMask(
                      buildGrayDifferenceMask(currentPatchGray, alignedBaselineGray, CONTRAST_HOLE_DIFF_THRESHOLD),
                      patchWidth,
                      patchHeight,
                      3,
                    ),
                    patchWidth,
                    patchHeight,
                    3,
                  );
                  const contrastChangedPixels = countMaskPixels(contrastHoleMask);
                  const contrastChangedRatioPct =
                    (contrastChangedPixels / Math.max(1, patchWidth * patchHeight)) * 100;
                  lastHistogramDeltaPctRef.current = 0;
                  histogramDeltaPctForFrame = 0;

                  // From-scratch detector path: hits = persistent localized change.
                  // Runs on the same aligned gray patches; commits its own shots
                  // and the legacy commit sites are suppressed (guarded below).
                  if (useChangeDetectorRef.current) {
                    const expectedDiameterPxPatch =
                      pixelsPerInch > 0 && tweakSettings.expectedHoleDiameterInches > 0 && drawRect.width > 0
                        ? pixelsPerInch * tweakSettings.expectedHoleDiameterInches * (patchWidth / drawRect.width)
                        : 0;
                    const minHolePixels = Math.max(
                      4,
                      Math.round(Math.PI * Math.pow(Math.max(2, (expectedDiameterPxPatch || 6) * 0.4) / 2, 2)),
                    );
                    const maxHolePixels = Math.max(minHolePixels * 6, Math.round(patchWidth * patchHeight * 0.25));
                    const changeHits = detectChangeHoles(currentPatchGray, alignedBaselineGray, patchWidth, patchHeight, {
                      diffThreshold: 18,
                      minPixels: minHolePixels,
                      maxPixels: maxHolePixels,
                      expectedDiameterPx: expectedDiameterPxPatch,
                    });
                    const tracker = changeHoleTrackerRef.current;
                    const matchRadius = Math.max(4, expectedDiameterPxPatch || 10);
                    const frameNow = frameIndexRef.current;
                    const REQUIRED_PERSISTENCE = 3;
                    for (const hit of changeHits) {
                      let track =
                        tracker.find(
                          (candidate) =>
                            !candidate.committed &&
                            Math.hypot(candidate.x - hit.centerX, candidate.y - hit.centerY) <= matchRadius,
                        ) ?? null;
                      if (track) {
                        track.x = (track.x + hit.centerX) / 2;
                        track.y = (track.y + hit.centerY) / 2;
                        track.count += 1;
                        track.lastFrame = frameNow;
                      } else {
                        track = { x: hit.centerX, y: hit.centerY, count: 1, lastFrame: frameNow, committed: false };
                        tracker.push(track);
                      }
                      if (track.committed || track.count < REQUIRED_PERSISTENCE) continue;
                      // Confirm the candidate with the active hole classifier (the
                      // published model, or the heuristic baseline). Reject low-scoring
                      // patches without committing so a later, clearer frame can pass.
                      if (confirmWithClassifierRef.current) {
                        const classifierWindow = Math.max(12, Math.round((expectedDiameterPxPatch || 10) * 2.2));
                        const candidatePatch = cropGrayWindow(
                          currentPatchGray,
                          patchWidth,
                          patchHeight,
                          track.x,
                          track.y,
                          classifierWindow,
                        );
                        if (getHoleClassifier().score(candidatePatch) < 0.5) continue;
                      }
                      const centerXFull = Math.round(drawRect.x + ((track.x + 0.5) / patchWidth) * drawRect.width);
                      const centerYFull = Math.round(drawRect.y + ((track.y + 0.5) / patchHeight) * drawRect.height);
                      const diameterFull = hit.diameterPx * (drawRect.width / patchWidth);
                      track.committed = true;
                      const alreadyLogged = shotMarkersRef.current.some(
                        (shot) => Math.hypot(shot.centerX - centerXFull, shot.centerY - centerYFull) <= Math.max(8, diameterFull),
                      );
                      if (alreadyLogged) continue;
                      const nextNumber = shotSequenceRef.current + 1;
                      shotSequenceRef.current = nextNumber;
                      const previousShot = shotMarkersRef.current[shotMarkersRef.current.length - 1] ?? null;
                      const entry = makeShotEntry({
                        id: `change_${nextNumber}`,
                        shotNumber: nextNumber,
                        frame: frameNow,
                        videoTimeSec: videoEl.currentTime,
                        timeSincePreviousShotSec: previousShot
                          ? Math.max(0, videoEl.currentTime - previousShot.videoTimeSec)
                          : null,
                        centerX: centerXFull,
                        centerY: centerYFull,
                        radius: Math.max(4, Math.round(diameterFull / 2)),
                        changedPixels: hit.pixelCount,
                        changeScore: hit.compactness,
                        estimatedDiameterPx: diameterFull,
                        estimatedDiameterInches: pixelsPerInch > 0 ? diameterFull / pixelsPerInch : null,
                        detectionMethod: "change_detect",
                        patchWidthPx: patchWidth,
                        patchHeightPx: patchHeight,
                        drawRectX: Math.round(drawRect.x),
                        drawRectY: Math.round(drawRect.y),
                        drawRectWidth: Math.round(drawRect.width),
                        drawRectHeight: Math.round(drawRect.height),
                      });
                      shotMarkersRef.current = [...shotMarkersRef.current, entry].slice(
                        -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
                      );
                      shotsDetectedThisFrame.push(entry);
                    }
                    // Expire tracks not re-seen recently so a moved/cleared target resets.
                    changeHoleTrackerRef.current = tracker.filter((candidate) => frameNow - candidate.lastFrame <= 12);

                    // Capture the detector's view for the detail panels (debug only).
                    if (detailedViewsVisible) {
                      const debug = analyzeChangeHoleCandidates(currentPatchGray, alignedBaselineGray, patchWidth, patchHeight, {
                        diffThreshold: 18,
                        minPixels: minHolePixels,
                        maxPixels: maxHolePixels,
                        expectedDiameterPx: expectedDiameterPxPatch,
                      });
                      changeDetectorDebugForFrame = {
                        gray: currentPatchGray,
                        width: patchWidth,
                        height: patchHeight,
                        mask: debug.mask,
                        candidates: debug.candidates,
                        info: {
                          changedPixels: countMaskPixels(debug.mask),
                          bandMinPx: expectedDiameterPxPatch * CHANGE_HOLE_MIN_DIAMETER_SCALE,
                          bandMaxPx: expectedDiameterPxPatch * CHANGE_HOLE_MAX_DIAMETER_SCALE,
                          persistence: REQUIRED_PERSISTENCE,
                        },
                      };
                    }
                  }

                  // The subtraction mask IS the effective change signal.
                  const effectiveMask = contrastHoleMask;
                  const positiveDiffMap = buildPositiveBackgroundDiffMap(
                    currentPatchRgba,
                    alignedBaselineRgba,
                    patchWidth,
                    patchHeight,
                  );
                  const yellowGreenRegions: ChangedContourRegion[] = yellowGreenHits.slice(0, 10).map((hit) => ({
                    pixelCount: hit.area,
                    centerX: hit.centroidX,
                    centerY: hit.centroidY,
                    minX: hit.minX,
                    minY: hit.minY,
                    maxX: hit.maxX,
                    maxY: hit.maxY,
                    corePixels: hit.area,
                    blackToColorPixels: hit.area,
                    meanSeverity: clamp01((hit.meanV - YELLOW_GREEN_COMPONENT_MEAN_V_MIN) / 85),
                  }));
                  // Expected single-hole size in patch pixels (0 when uncalibrated).
                  // Shared by the change-region screen, touching-hole separation,
                  // and the sub-hit sizing below.
                  const expectedHolePatchPx =
                    pixelsPerInch > 0 && tweakSettings.expectedHoleDiameterInches > 0 && drawRect.width > 0
                      ? pixelsPerInch * tweakSettings.expectedHoleDiameterInches * (patchWidth / drawRect.width)
                      : 0;

                  // Detector fusion — CONTRAST IS PRIMARY. Hole-shaped regions of the
                  // clean contrast mask that changed HARD versus the baseline drive
                  // detection on every target type. When the bright-splatter option
                  // is on (shoot-n-see targets), yellow-green top-hat hits are added
                  // as a secondary detector — but only where the pixels also changed
                  // versus the baseline (printed graphics never qualify) — and are
                  // deduped against the contrast hits so one hole isn't counted twice.
                  const contrastMinPixels =
                    expectedHolePatchPx > 0
                      ? Math.max(
                          3,
                          Math.round(Math.PI * Math.pow((expectedHolePatchPx * CHANGE_HOLE_MIN_DIAMETER_SCALE) / 2, 2)),
                        )
                      : 4;
                  const contrastMaxPixels = Math.max(contrastMinPixels + 1, Math.round(patchWidth * patchHeight * 0.25));
                  const contrastRegions =
                    contrastChangedPixels > 0 && contrastChangedRatioPct <= CONTRAST_MAX_GLOBAL_CHANGE_PCT
                      ? findChangedRegionsByConnectedComponents(
                          contrastHoleMask,
                          patchWidth,
                          patchHeight,
                          contrastMinPixels,
                          contrastMaxPixels,
                        )
                      : [];
                  const changeHoleRegions: ChangedContourRegion[] = [];
                  for (const region of contrastRegions) {
                    if (!isHoleShapedRegion(region, expectedHolePatchPx, patchWidth, patchHeight)) continue;
                    // Significance gate: an impact rips the pixels far from baseline;
                    // drift and shadow edges barely clear the mask threshold.
                    const meanDiff = regionMeanAbsDiff(
                      region,
                      contrastHoleMask,
                      currentPatchGray,
                      alignedBaselineGray,
                      patchWidth,
                      patchHeight,
                    );
                    if (meanDiff < CONTRAST_REGION_MIN_MEAN_DIFF) continue;
                    changeHoleRegions.push(region);
                    if (changeHoleRegions.length >= 10) break;
                  }
                  const splatterOverlapRadius = Math.max(6, expectedHolePatchPx);
                  const splatterRegions: ChangedContourRegion[] = detectBrightColorsRef.current
                    ? yellowGreenRegions.filter((yg) => {
                        // Caliber-scaled size band: a splatter mark smaller than
                        // 0.4× the bullet diameter is a fleck, not a hit; larger
                        // than 4× is a splash/graphic, not one hole.
                        if (expectedHolePatchPx > 0) {
                          const splatterDiameterPx = 2 * Math.sqrt(yg.pixelCount / Math.PI);
                          if (
                            splatterDiameterPx < expectedHolePatchPx * SPLATTER_MIN_DIAMETER_SCALE ||
                            splatterDiameterPx > expectedHolePatchPx * SPLATTER_MAX_DIAMETER_SCALE
                          ) {
                            return false;
                          }
                        }
                        return (
                          !changeHoleRegions.some(
                            (hole) =>
                              Math.hypot(hole.centerX - yg.centerX, hole.centerY - yg.centerY) <= splatterOverlapRadius,
                          ) &&
                          regionMeanAbsDiff(
                            yg,
                            yellowGreenMask,
                            currentPatchGray,
                            alignedBaselineGray,
                            patchWidth,
                            patchHeight,
                          ) >= SPLATTER_REGION_MIN_MEAN_DIFF
                        );
                      })
                    : [];
                  const analysisCandidates: {
                    region: ChangedContourRegion;
                    sourceMask: Uint8Array;
                    detector: "yellow_green" | "change";
                  }[] = [
                    ...changeHoleRegions.map((region) => ({
                      region,
                      sourceMask: contrastHoleMask,
                      detector: "change" as const,
                    })),
                    ...splatterRegions.map((region) => ({
                      region,
                      sourceMask: yellowGreenMask,
                      detector: "yellow_green" as const,
                    })),
                  ];
                  const analysisMask =
                    detectBrightColorsRef.current && yellowGreenChangedPixels > 0
                      ? mergeBinaryMasks([contrastHoleMask, yellowGreenMask])
                      : contrastHoleMask;
                  const analysisChangedPixels =
                    contrastChangedPixels + (detectBrightColorsRef.current ? yellowGreenChangedPixels : 0);
                  const analysisRegions = analysisCandidates.map((candidate) => candidate.region);

                  // Detection trail (throttled ~5Hz): every pipeline stage, exactly
                  // as the detector sees it this frame.
                  if (showDetectionTrailRef.current) {
                    const trailNowMs = performance.now();
                    if (trailNowMs - lastTrailDrawAtMsRef.current >= 200) {
                      lastTrailDrawAtMsRef.current = trailNowMs;
                      drawTrailGray(trailBaselineCanvasRef.current, alignedBaselineGray, patchWidth, patchHeight);
                      drawTrailRgba(trailCurrentCanvasRef.current, currentPatchRgba, patchWidth, patchHeight);
                      drawTrailMask(trailMaskCanvasRef.current, currentPatchGray, contrastHoleMask, patchWidth, patchHeight);
                      drawTrailBlobs(
                        trailBlobsCanvasRef.current,
                        currentPatchRgba,
                        contrastRegions,
                        changeHoleRegions,
                        splatterRegions,
                        contourShotTracksRef.current,
                        patchWidth,
                        patchHeight,
                      );
                      drawTrailAudio(
                        trailAudioCanvasRef.current,
                        audioRmsTimelineRef.current,
                        audioMeanDbfsRef.current,
                        audioThresholdDbfsRef.current,
                        spikeEventsRef.current,
                        windows,
                        currentSec,
                        durationSec,
                      );
                      setTrailInfo({
                        shiftPx: Math.round(Math.hypot(patchShift.dx, patchShift.dy)),
                        changedPixels: contrastChangedPixels,
                        changedPct: contrastChangedRatioPct,
                        blobCount: contrastRegions.length,
                        acceptedCount: changeHoleRegions.length,
                        splatterCount: splatterRegions.length,
                        pendingTracks: contourShotTracksRef.current.filter((track) => !track.committed).length,
                        shotCount: shotMarkersRef.current.length,
                        inWindow: true,
                        timeSec: currentSec,
                        endGateActive: permanentChangeMask !== null,
                      });
                    }
                  }

                  const contourRegionsForPlayback = analysisRegions.slice(0, 10).map((region) => ({
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
                    changedPixels: analysisChangedPixels,
                    maskRuns: encodeBinaryMaskRuns(analysisMask),
                    regions: contourRegionsForPlayback,
                  };
                  const displayMask = analysisChangedPixels > 0 ? analysisMask : null;

                  // Persistence sampling: for every shot already logged in a prior
                  // frame, check whether its location still differs from the baseline
                  // in this frame's mask. A real hole keeps showing up; a transient
                  // flash stops. We sample by the shot's PATCH-relative coordinate
                  // (centerPatchX/Y), not its full-frame coordinate — the patch tracks
                  // the target, so a fixed patch coordinate stays on the same point as
                  // the target moves (a full-frame coordinate would drift off it).
                  if (patchWidth > 0 && patchHeight > 0) {
                    for (const loggedShot of shotMarkersRef.current) {
                      const px = Math.round(loggedShot.centerPatchX);
                      const py = Math.round(loggedShot.centerPatchY);
                      if (px < 0 || py < 0 || px >= patchWidth || py >= patchHeight) continue;
                      let present = false;
                      for (let dy = -3; dy <= 3 && !present; dy += 1) {
                        const yy = py + dy;
                        if (yy < 0 || yy >= patchHeight) continue;
                        for (let dx = -3; dx <= 3; dx += 1) {
                          const xx = px + dx;
                          if (xx < 0 || xx >= patchWidth) continue;
                          if (effectiveMask[yy * patchWidth + xx] === 1) {
                            present = true;
                            break;
                          }
                        }
                      }
                      const rec =
                        shotPresenceRef.current.get(loggedShot.id) ??
                        { frames: 0, present: 0, transitions: 0, lastPresent: false };
                      rec.frames += 1;
                      if (present) rec.present += 1;
                      // Count on/off toggles (after the first sample) to flag areas
                      // that flicker in and out rather than staying put.
                      if (rec.frames > 1 && present !== rec.lastPresent) rec.transitions += 1;
                      rec.lastPresent = present;
                      shotPresenceRef.current.set(loggedShot.id, rec);
                    }
                  }

                  if (processedContourContext) {
                    drawProcessedContourView(
                      processedContourContext,
                      currentPatchRgba,
                      activeBaselinePatch,
                      patchWidth,
                      patchHeight,
                      displayMask,
                      analysisRegions,
                    );
                  }
                  if (processedMaskContext && !useChangeDetectorRef.current) {
                    drawBinaryMaskWindowView(
                      processedMaskContext,
                      patchWidth,
                      patchHeight,
                      displayMask,
                      "Binary mask (live)",
                      currentPatchRgba,
                    );
                  }

                  const activeWindowIndexForFrame = windowIndexAtTime(currentSec, windows);
                  const activeWindowForFrame =
                    activeWindowIndexForFrame >= 0 && activeWindowIndexForFrame < windows.length
                      ? windows[activeWindowIndexForFrame]
                      : { start: 0, end: Math.max(0, durationSec) };

                  let bestCandidate: { entry: ShotLogEntry; score: number; source: "strict" } | null = null;
                  let bestRelaxedCandidate: {
                    entry: ShotLogEntry;
                    score: number;
                    source: "relaxed" | "failsafe" | "histogram" | "yellow_green";
                  } | null = null;
                  const contourHitCandidates: Array<{
                    entry: ShotLogEntry;
                    score: number;
                    detector: "yellow_green" | "change";
                  }> = [];
                  for (const { region, sourceMask, detector } of analysisCandidates.slice(0, 12)) {
                    const shouldBypassGates = FORCE_OPEN_SHOT_GATES;
                    const temporalStats = summarizeRegionTemporalSupport(
                      region,
                      sourceMask,
                      sourceMask,
                      positiveDiffMap,
                      patchWidth,
                      patchHeight,
                    );
                    const minSupportRatio = Math.max(0.2, Math.min(0.95, tweakSettings.temporalDarkVoteRatio));
                    const strictSupportOk = shouldBypassGates || temporalStats.rawPixels === 0 || temporalStats.supportRatio >= minSupportRatio;
                    const minMeanPositiveDiff = Math.max(6, Math.min(80, tweakSettings.transitionLuminanceJumpMin));
                    const strictMeanDiffOk = shouldBypassGates || temporalStats.meanDiff >= minMeanPositiveDiff;
                    const relaxedSupportOk = shouldBypassGates || temporalStats.rawPixels === 0 || temporalStats.supportRatio >= 0.05;
                    const relaxedMeanDiffOk = shouldBypassGates || temporalStats.meanDiff >= Math.max(3, minMeanPositiveDiff * 0.35);

                    const spanX = Math.max(1, region.maxX - region.minX + 1);
                    const spanY = Math.max(1, region.maxY - region.minY + 1);
                    const blobAreaPx = Math.max(1, spanX * spanY);
                    const blobFillPct = (region.pixelCount / blobAreaPx) * 100;
                    const blobAspectRatio = spanAspectRatio(spanX, spanY);
                    const maxBlobAspectRatio = Math.max(3, Math.max(8.8, tweakSettings.colorAspectRatioMax) + 1.2);
                    const strictShapeOk = shouldBypassGates || (blobFillPct >= 5 && blobAspectRatio <= maxBlobAspectRatio);
                    const relaxedShapeOk = shouldBypassGates || (blobFillPct >= 1 && blobAspectRatio <= Math.max(16, maxBlobAspectRatio * 1.6));

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
                    const strictSizeOk = isSubTwoInchShot(diameterPx, diameterInches, drawRect.width, drawRect.height, tweakSettings);
                    const relaxedMaxDiameterPx = Math.max(
                      1200,
                      tweakSettings.colorDiameterMaxPx * 2.5,
                      Math.min(drawRect.width, drawRect.height) * 0.85,
                    );
                    const relaxedSizeOk = shouldBypassGates
                      ? true
                      : diameterInches !== null
                        ? diameterInches <= Math.max(6, tweakSettings.maxShotDiameterInches * 3)
                        : diameterPx <= relaxedMaxDiameterPx;
                    if (!relaxedSupportOk || !relaxedMeanDiffOk || !relaxedShapeOk || !relaxedSizeOk) continue;

                    const transitionRatioPct = (region.pixelCount / (patchWidth * patchHeight)) * 100;
                    const normalizedTransitionRatio = clamp01(transitionRatioPct / 4.5);
                    const normalizedFill = clamp01(blobFillPct / 100);
                    const normalizedCompactness = clamp01(1 / Math.max(1, blobAspectRatio));
                    const normalizedTemporalSupport = clamp01(temporalStats.supportRatio);
                    const normalizedMeanDiff = clamp01(temporalStats.meanDiff / Math.max(1, tweakSettings.transitionLuminanceJumpMin * 2));
                    const normalizedHistogramDelta = clamp01(histogramDeltaPctForFrame / 20);
                    const candidateScore = clamp01(
                      normalizedTransitionRatio * 0.35 +
                      normalizedFill * 0.2 +
                      normalizedCompactness * 0.1 +
                      normalizedTemporalSupport * 0.2 +
                      normalizedMeanDiff * 0.15 +
                      normalizedHistogramDelta * 0.2,
                    );

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
                    // Touching bullets merge into one connected blob. When the changed
                    // region is clearly longer AND larger than a single caliber hole,
                    // scan along its long axis and lay out one sub-hit per hole so they
                    // count as separate shots. Needs a calibrated caliber to know the
                    // single-hole size; otherwise the region stays one hit.
                    // Sub-pixel center: for change-sourced hits, refine the binary-mask
                    // centroid by weighting with the darkening magnitude vs baseline.
                    let regionCenterPx = region.centerX + 0.5;
                    let regionCenterPy = region.centerY + 0.5;
                    if (detector === "change") {
                      const refined = refineRegionCenterByDiff(region, sourceMask, positiveDiffMap, patchWidth, patchHeight);
                      if (refined) {
                        regionCenterPx = refined.px;
                        regionCenterPy = refined.py;
                      }
                    }
                    let subHits: { px: number; py: number; split: boolean }[] = [
                      { px: regionCenterPx, py: regionCenterPy, split: false },
                    ];
                    if (expectedHolePatchPx > 0) {
                      const longAxisPx = Math.max(spanX, spanY);
                      const singleHoleAreaPx = Math.PI * Math.pow(expectedHolePatchPx / 2, 2);
                      // Cheap pre-screen (perf): only bother separating when the blob is
                      // clearly bigger than a single caliber hole. Then erode to find the
                      // ACTUAL hole centers — only split if ≥2 real concentrations exist,
                      // so a single elongated smear stays one hit (no phantom markers).
                      const looksOversized =
                        longAxisPx >= expectedHolePatchPx * 1.5 && region.pixelCount >= singleHoleAreaPx * 1.4;
                      if (looksOversized) {
                        const centers = findTouchingHoleCenters(
                          sourceMask,
                          patchWidth,
                          patchHeight,
                          region,
                          expectedHolePatchPx / 2,
                        ).slice(0, 8);
                        if (centers.length >= 2) {
                          subHits = centers.map((c) => ({ px: c.px, py: c.py, split: true }));
                        }
                      }
                    }

                    for (const sub of subHits) {
                      // Split sub-hits get one caliber's worth of size; a lone region
                      // keeps its measured blob diameter.
                      const subDiameterInches =
                        sub.split && tweakSettings.expectedHoleDiameterInches > 0
                          ? tweakSettings.expectedHoleDiameterInches
                          : diameterInches;
                      const subDiameterPx =
                        sub.split && pixelsPerInch > 0 && tweakSettings.expectedHoleDiameterInches > 0
                          ? pixelsPerInch * tweakSettings.expectedHoleDiameterInches
                          : diameterPx;
                      const centerX = Math.round(drawRect.x + (sub.px / patchWidth) * drawRect.width);
                      const centerY = Math.round(drawRect.y + (sub.py / patchHeight) * drawRect.height);
                      const radius = Math.max(4, Math.round(subDiameterPx / 2));
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
                        changedPixels: Math.round(region.pixelCount / subHits.length),
                        changeScore: candidateScore,
                        estimatedDiameterInches: subDiameterInches,
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
                        centerPatchX: sub.px,
                        centerPatchY: sub.py,
                        spanWidthPx: spanX,
                        spanHeightPx: spanY,
                        estimatedDiameterPx: subDiameterPx,
                        changedPixelRatioPct: transitionRatioPct,
                        brightPixelCount: temporalStats.persistentPixels,
                        transitionPixelCount: temporalStats.rawPixels,
                        transitionPurityPct: temporalStats.supportRatio * 100,
                        minBlobPixelsThreshold: detector === "yellow_green" ? YELLOW_GREEN_COMPONENT_MIN_AREA : contrastMinPixels,
                        temporalHistoryFramesUsed: 1,
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
                      const strictGateOk = strictSupportOk && strictMeanDiffOk && strictShapeOk && strictSizeOk;
                      const strictAccepted = strictGateOk && isLikelyGoodShot(candidateShot, tweakSettings);
                      const relaxedAccepted = isLikelyGoodShotRelaxed(candidateShot, tweakSettings);
                      if (strictAccepted || relaxedAccepted) {
                        contourHitCandidates.push({ entry: candidateShot, score: candidateScore, detector });
                      }
                      if (!USE_CONTOUR_RENDER_HITS_AS_SHOTS) {
                        if (strictAccepted) {
                          if (!bestCandidate || candidateScore > bestCandidate.score) {
                            bestCandidate = { entry: candidateShot, score: candidateScore, source: "strict" };
                          }
                        } else if (relaxedAccepted) {
                          if (!bestRelaxedCandidate || candidateScore > bestRelaxedCandidate.score) {
                            bestRelaxedCandidate = { entry: candidateShot, score: candidateScore, source: "relaxed" };
                          }
                        }
                      }
                    }
                  }

                  if (USE_CONTOUR_RENDER_HITS_AS_SHOTS) {
                    const sortedContourCandidates = [...contourHitCandidates].sort((a, b) => b.score - a.score);
                    // Temporal confirmation: this frame's candidates feed per-location
                    // tracks, and a track only commits after it's been sighted enough
                    // frames. A real hole is permanent — it re-appears immediately —
                    // while one-frame flashes (brass in flight, glare, muzzle smoke)
                    // never accumulate sightings. The track center is the running
                    // average of its sightings, so the committed position is steadier
                    // than any single frame's centroid.
                    const contourFrameNow = frameIndexRef.current;
                    const contourTracks = contourShotTracksRef.current;
                    for (const candidate of sortedContourCandidates) {
                      const px = candidate.entry.centerPatchX;
                      const py = candidate.entry.centerPatchY;
                      const toPatch =
                        candidate.entry.drawRectWidth > 0
                          ? candidate.entry.patchWidthPx / candidate.entry.drawRectWidth
                          : 1;
                      const rPatch = Math.max(4, candidate.entry.radius * toPatch);
                      const track = contourTracks.find(
                        (t) => Math.hypot(t.px - px, t.py - py) <= Math.max(rPatch, t.rPatch, 6),
                      );
                      if (track) {
                        if (track.lastFrame !== contourFrameNow) {
                          track.count += 1;
                          track.lastFrame = contourFrameNow;
                        }
                        if (!track.committed) {
                          track.px = (track.px + px) / 2;
                          track.py = (track.py + py) / 2;
                          track.rPatch = Math.max(track.rPatch, rPatch);
                          if (candidate.score >= track.bestScore) {
                            track.bestScore = candidate.score;
                            track.bestEntry = candidate.entry;
                          }
                        }
                      } else {
                        contourTracks.push({
                          px,
                          py,
                          rPatch,
                          count: 1,
                          lastFrame: contourFrameNow,
                          committed: false,
                          detector: candidate.detector,
                          bestScore: candidate.score,
                          bestEntry: candidate.entry,
                        });
                      }
                    }

                    let updatedShots = [...shotMarkersRef.current];
                    const acceptedShots: ShotLogEntry[] = [];
                    if (!useChangeDetectorRef.current) {
                      for (const track of contourTracks) {
                        if (track.committed) continue;
                        const requiredSightings =
                          track.detector === "change"
                            ? CONTOUR_SHOT_MIN_SIGHTINGS_CHANGE
                            : CONTOUR_SHOT_MIN_SIGHTINGS_YELLOW_GREEN;
                        if (track.count < requiredSightings) continue;
                        // End-state validation (first-vs-last background subtraction):
                        // a real hole is permanent, so its location must also show
                        // change in the aligned last-frame diff. Transients don't —
                        // mark them committed so they stop being re-evaluated, but
                        // never become shots.
                        if (
                          permanentChangeMask &&
                          permanentChangeMaskWidth === patchWidth &&
                          permanentChangeMaskHeight === patchHeight &&
                          !maskHasChangeNear(
                            permanentChangeMask,
                            permanentChangeMaskWidth,
                            permanentChangeMaskHeight,
                            track.px,
                            track.py,
                            Math.max(3, track.rPatch),
                          )
                        ) {
                          track.committed = true;
                          continue;
                        }
                        track.committed = true;
                        // De-dup by location over the WHOLE scan (in patch space, which
                        // tracks the target so a fixed hole keeps the same coordinate).
                        // A hole is permanent: once committed, later detections at that
                        // spot are the same hole, not new shots — so persistent holes
                        // aren't re-counted every ~0.35s as they used to be.
                        const duplicate = committedShotPointsRef.current.some(
                          (p) =>
                            Math.hypot(p.px - track.px, p.py - track.py) <=
                            Math.max(track.rPatch, p.rPatch) * CONTOUR_SHOT_DEDUP_DISTANCE_FACTOR,
                        );
                        if (duplicate) continue;

                        const nextShotNumber = shotSequenceRef.current + 1;
                        shotSequenceRef.current = nextShotNumber;
                        const acceptedShot: ShotLogEntry = {
                          ...track.bestEntry,
                          shotNumber: nextShotNumber,
                          id: `shot_${nextShotNumber}`,
                          centerPatchX: track.px,
                          centerPatchY: track.py,
                          centerX: Math.round(drawRect.x + (track.px / patchWidth) * drawRect.width),
                          centerY: Math.round(drawRect.y + (track.py / patchHeight) * drawRect.height),
                        };
                        acceptedShots.push(acceptedShot);
                        committedShotPointsRef.current.push({ px: track.px, py: track.py, rPatch: track.rPatch });
                        updatedShots = [...updatedShots, acceptedShot].slice(
                          -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
                        );
                      }
                    }
                    // Expire tracks not re-sighted recently. Committed tracks stay while
                    // sighted so a persistent hole keeps matching its own track instead
                    // of spawning a fresh one every frame.
                    contourShotTracksRef.current = contourTracks.filter(
                      (t) => contourFrameNow - t.lastFrame <= CONTOUR_SHOT_TRACK_MAX_GAP_FRAMES,
                    );
                    if (acceptedShots.length > 0) {
                      lastShotAtMsRef.current = nowMs;
                      shotMarkersRef.current = updatedShots;
                      shotsDetectedThisFrame.push(...acceptedShots);
                    }
                    pendingShotCandidateRef.current = null;
                  } else {
                    const selectedCandidate = bestCandidate ?? bestRelaxedCandidate;
                    if (selectedCandidate) {
                      if (!bestCandidate && bestRelaxedCandidate && !relaxedShotGateNoticeRef.current) {
                        relaxedShotGateNoticeRef.current = true;
                        setScanStatus(
                          bestRelaxedCandidate.source === "failsafe"
                            ? "Strict shot gate is filtering events; using failsafe mask fallback for shot detection."
                            : bestRelaxedCandidate.source === "histogram"
                              ? "Region extraction is sparse; using histogram probe-window fallback for shot detection."
                              : bestRelaxedCandidate.source === "yellow_green"
                                ? "Using yellow/green top-hat connected-components fallback for shot detection."
                            : "Strict shot gate is filtering events; using relaxed fallback mode for shot detection.",
                        );
                      }
                      const activePending = pendingShotCandidateRef.current;
                      const matchDistancePx = Math.max(10, Math.min(64, selectedCandidate.entry.radius * 2.2));
                      const extendsPending =
                        !!activePending &&
                        nowMs - activePending.lastSeenAtMs <= SHOT_PERSISTENCE_MAX_GAP_MS &&
                        Math.hypot(
                          selectedCandidate.entry.centerX - activePending.centerX,
                          selectedCandidate.entry.centerY - activePending.centerY,
                        ) <= matchDistancePx;

                      if (extendsPending && activePending) {
                        activePending.centerX = selectedCandidate.entry.centerX;
                        activePending.centerY = selectedCandidate.entry.centerY;
                        activePending.lastSeenAtMs = nowMs;
                        if (selectedCandidate.score >= activePending.bestScore) {
                          activePending.bestScore = selectedCandidate.score;
                          activePending.bestEntry = selectedCandidate.entry;
                        }

                        const persistedMs = activePending.lastSeenAtMs - activePending.firstSeenAtMs;
                        const minPersistenceMs = FORCE_OPEN_SHOT_GATES
                          ? 0
                          : selectedCandidate.source === "strict"
                            ? SHOT_PERSISTENCE_MIN_MS
                            : selectedCandidate.source === "relaxed"
                              ? 70
                              : selectedCandidate.source === "histogram"
                                ? 40
                                : selectedCandidate.source === "yellow_green"
                                  ? 40
                              : 0;
                        const shouldConfirm = !activePending.confirmed && persistedMs >= minPersistenceMs;
                        if (shouldConfirm && nowMs - lastShotAtMsRef.current > shotCooldownMs && !useChangeDetectorRef.current) {
                          const nextShotNumber = shotSequenceRef.current + 1;
                          shotSequenceRef.current = nextShotNumber;
                          const acceptedShot: ShotLogEntry = {
                            ...activePending.bestEntry,
                            shotNumber: nextShotNumber,
                            id: `shot_${nextShotNumber}`,
                          };
                          activePending.confirmed = true;
                          lastShotAtMsRef.current = nowMs;
                          shotsDetectedThisFrame.push(acceptedShot);
                          shotMarkersRef.current = [...shotMarkersRef.current, acceptedShot].slice(
                            -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
                          );
                        }
                      } else {
                        const nextPending: PendingShotCandidate = {
                          centerX: selectedCandidate.entry.centerX,
                          centerY: selectedCandidate.entry.centerY,
                          firstSeenAtMs: nowMs,
                          lastSeenAtMs: nowMs,
                          bestEntry: selectedCandidate.entry,
                          bestScore: selectedCandidate.score,
                          confirmed: false,
                        };
                        if (
                          (FORCE_OPEN_SHOT_GATES ||
                            selectedCandidate.source === "failsafe" ||
                            selectedCandidate.source === "histogram" ||
                            selectedCandidate.source === "yellow_green") &&
                          nowMs - lastShotAtMsRef.current > shotCooldownMs &&
                          !useChangeDetectorRef.current
                        ) {
                          const nextShotNumber = shotSequenceRef.current + 1;
                          shotSequenceRef.current = nextShotNumber;
                          const acceptedShot: ShotLogEntry = {
                            ...selectedCandidate.entry,
                            shotNumber: nextShotNumber,
                            id: `shot_${nextShotNumber}`,
                          };
                          lastShotAtMsRef.current = nowMs;
                          shotsDetectedThisFrame.push(acceptedShot);
                          shotMarkersRef.current = [...shotMarkersRef.current, acceptedShot].slice(
                            -Math.max(1, Math.round(tweakSettings.shotHistoryMaxCount)),
                          );
                          pendingShotCandidateRef.current = { ...nextPending, confirmed: true };
                        } else {
                          pendingShotCandidateRef.current = nextPending;
                        }
                      }
                    }
                  }
              }
              activeSpikeWindowIndexRef.current = -1;
            }
            } catch {
              resetShotFlowState();
              if (processedPatchContext && processedPatchCanvas) {
                processedPatchContext.clearRect(0, 0, processedPatchCanvas.width, processedPatchCanvas.height);
              }
              if (processedContourContext && processedContourCanvas) {
                processedContourContext.clearRect(0, 0, processedContourCanvas.width, processedContourCanvas.height);
              }
              if (processedMaskContext && processedMaskCanvas) {
                processedMaskContext.clearRect(0, 0, processedMaskCanvas.width, processedMaskCanvas.height);
              }
              if (processedYellowGreenContext && processedYellowGreenCanvas) {
                processedYellowGreenContext.clearRect(0, 0, processedYellowGreenCanvas.width, processedYellowGreenCanvas.height);
              }
            }
          }
          const matcherSummary = sqdiffUsed
            ? `NCC:${(primaryNccScore * 100).toFixed(0)} SAD:${(primarySadScore * 100).toFixed(0)} SQ:${(
                Math.max(0, sqdiffScore) * 100
              ).toFixed(0)}`
            : `NCC:${(primaryNccScore * 100).toFixed(0)} SAD:${(primarySadScore * 100).toFixed(0)}`;
          const overlayLabel =
            `${trackingMode.toUpperCase()} ${confidencePct.toFixed(1)}% | ${matcherSummary} | HIST ${histogramDeltaPctForFrame.toFixed(
              1,
            )}% | YG ${yellowGreenHitCountForFrame}` +
            (estimatedDistanceInches === null ? "" : ` | ~${formatLinearFromInches(estimatedDistanceInches, 1)}`);
          const labelWidth = Math.max(165, Math.min(420, 14 + overlayLabel.length * 6));

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
          const eligibleLiveShotsForClustering = filterShotsByAnalysisAge(
            visibleShotMarkers,
            currentSec,
            CLUSTER_MIN_VISIBLE_AGE_SEC,
          ).filter((shot) => shotMakesIt(shot, tweakSettings) && passesCaliberGate(shot, tweakSettings));
          const clusterNowMs = performance.now();
          if (eligibleLiveShotsForClustering.length === 0) {
            if (liveClusterShotCountRef.current !== 0) {
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
              liveClusterLastUpdatedAtMsRef.current = 0;
              liveContourGroupVisualsRef.current = { updatedAtMs: 0, regionColors: [], regionGroupLabels: [] };
            }
          } else {
            const shotCountChanged = eligibleLiveShotsForClustering.length !== liveClusterShotCountRef.current;
            const clusterUpdateDue =
              clusterNowMs - liveClusterLastUpdatedAtMsRef.current >= LIVE_GROUP_UPDATE_INTERVAL_MS;
            if (shotCountChanged && clusterUpdateDue) {
              const liveClustering = clusterShotsBySpaceTime(eligibleLiveShotsForClustering, tweakSettings);
              liveShotClusteringRef.current = liveClustering;
              const liveColorMap: Record<number, string> = {};
              for (const cluster of liveClustering.clusters) {
                liveColorMap[cluster.clusterId] = clusterColorForId(cluster.clusterId);
              }
              liveClusterColorByIdRef.current = liveColorMap;
              liveClusterShotCountRef.current = eligibleLiveShotsForClustering.length;
              liveClusterLastUpdatedAtMsRef.current = clusterNowMs;
            }
          }
          const liveClusterByShotId = liveShotClusteringRef.current.shotClusterById;
          const liveClusterColorById = liveClusterColorByIdRef.current;
          const visibleClusterGeometry = clusterGeometryFromShots(visibleShotMarkers, liveClusterByShotId);
          drawClusterGeometry(overlayContext, visibleClusterGeometry, liveClusterColorById, scaleX, scaleY);
          let contourRegionColorsForFrame: string[] | undefined;
          let contourRegionGroupLabelsForFrame: string[] | undefined;
          if (detailedViewsVisible && contourWindowSnapshotForFrame && contourWindowSnapshotForFrame.regions.length > 0) {
            const cachedContourVisuals = liveContourGroupVisualsRef.current;
            const contourVisualsDue =
              cachedContourVisuals.regionColors.length === 0 ||
              clusterNowMs - cachedContourVisuals.updatedAtMs >= LIVE_GROUP_UPDATE_INTERVAL_MS;
            if (contourVisualsDue) {
              const contourVisuals = buildContourRegionClusterVisuals(
                contourWindowSnapshotForFrame,
                drawRect,
                eligibleLiveShotsForClustering,
                liveClusterByShotId,
                liveClusterColorById,
              );
              liveContourGroupVisualsRef.current = {
                updatedAtMs: clusterNowMs,
                regionColors: contourVisuals.regionColors,
                regionGroupLabels: contourVisuals.regionGroupLabels,
              };
            }
            contourRegionColorsForFrame = liveContourGroupVisualsRef.current.regionColors;
            contourRegionGroupLabelsForFrame = liveContourGroupVisualsRef.current.regionGroupLabels;
            const contourCanvas = processedContourCanvasRef.current;
            const contourContext = contourCanvas?.getContext("2d") ?? null;
            if (contourCanvas && contourContext) {
              if (
                contourCanvas.width !== contourWindowSnapshotForFrame.patchWidthPx ||
                contourCanvas.height !== contourWindowSnapshotForFrame.patchHeightPx
              ) {
                contourCanvas.width = contourWindowSnapshotForFrame.patchWidthPx;
                contourCanvas.height = contourWindowSnapshotForFrame.patchHeightPx;
              }
              drawContourRegionWindowOverlays(
                contourContext,
                contourWindowSnapshotForFrame.regions,
                contourRegionColorsForFrame,
                contourRegionGroupLabelsForFrame,
              );
            }
            drawContourRegionsOnTargetView(
              context,
              contourWindowSnapshotForFrame,
              drawRect,
              isBlinkOn,
              contourRegionColorsForFrame,
              pixelsPerInch,
              formatLinearFromInches,
              contourRegionGroupLabelsForFrame,
            );
          }
          if (processedPatchContext && !useChangeDetectorRef.current) {
            drawPatchWindowView(processedPatchContext, videoEl, drawRect);
            drawProbePatchOverlayView(
              processedPatchContext,
              drawRect.width,
              drawRect.height,
              overlayLabel,
              isBlinkOn,
              contourWindowSnapshotForFrame,
              contourRegionColorsForFrame,
              pixelsPerInch,
              formatLinearFromInches,
              contourRegionGroupLabelsForFrame,
            );
          }
          // Change-detector debug repaint: overrides the legacy mask/probe views
          // with what the detector actually sees (drawn last so it wins).
          if (changeDetectorDebugForFrame && (processedMaskContext || processedPatchContext)) {
            drawChangeDetectorDebugViews(
              processedMaskContext,
              processedPatchContext,
              changeDetectorDebugForFrame.gray,
              changeDetectorDebugForFrame.width,
              changeDetectorDebugForFrame.height,
              changeDetectorDebugForFrame.mask,
              changeDetectorDebugForFrame.candidates,
              changeHoleTrackerRef.current,
              changeDetectorDebugForFrame.info,
            );
          }
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
                `DB${clusterId} S${shotMarker.shotNumber}`,
                overlayCenterX + shotRadiusOverlay + 2,
                overlayCenterY - shotRadiusOverlay - 2,
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
          if (shotsDetectedThisFrame.length > 0) {
            const newestShot = shotsDetectedThisFrame[shotsDetectedThisFrame.length - 1];
            setLastShot(newestShot);
            setShotLogEntries((current) => [...current, ...shotsDetectedThisFrame]);
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
            }
            if (yellowGreenSnapshotForFrame) {
              yellowGreenTimelineRef.current.push(yellowGreenSnapshotForFrame);
              if (yellowGreenTimelineRef.current.length > 4000) {
                yellowGreenTimelineRef.current.shift();
              }
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

      // On a genuine finish (not a stop/restart): persist the first+last(50%)
      // blended background and show it on the static map so the result stays up.
      if (!isStreamScan && !stopRequestedRef.current) {
        buildScanBlendBackground(videoEl);
        setShotMapLiveStream(false);
      }
      if (!isStreamScan) {
        videoEl.pause();
      }
      scanVideoRef.current = null;
      // The final shots are exactly what's left in the live overlay at scan end
      // (shotMarkersRef is capped to shotHistoryMaxCount), rather than the full
      // unbounded accumulation. Skip focused per-spike re-scans so they don't
      // replace the full result set.
      if (!requestedWindow) {
        // Resolve each shot's temporal persistence: did its mark keep showing up
        // through the rest of the clip? Shots with too few post-detection frames
        // (fired near the end) get the benefit of the doubt.
        const annotated = shotMarkersRef.current.map((shot) => {
          const rec = shotPresenceRef.current.get(shot.id);
          if (!rec || rec.frames < SHOT_PERSISTENCE_MIN_FRAMES) {
            const ratio = rec && rec.frames > 0 ? rec.present / rec.frames : null;
            return { ...shot, persistenceRatio: ratio, persistent: true };
          }
          const ratio = rec.present / rec.frames;
          // Flicker = many on/off toggles across a meaningful share of frames. Such
          // areas are auto-strayed even if they cleared the persistence ratio.
          const flickerRate = rec.transitions / rec.frames;
          const flickers =
            rec.transitions >= SHOT_FLICKER_MIN_TRANSITIONS && flickerRate >= SHOT_FLICKER_MIN_RATE;
          const persistent = !flickers && ratio >= SHOT_PERSISTENCE_MIN_RATIO;
          return { ...shot, persistenceRatio: ratio, persistent };
        });
        // Fail-safe: if EVERY shot judged transient, the persistence signal likely
        // failed for this clip (e.g. lost tracking) — don't discard the whole
        // result; keep them all rather than show an empty grouping.
        const judged = annotated.filter(
          (shot) => (shotPresenceRef.current.get(shot.id)?.frames ?? 0) >= SHOT_PERSISTENCE_MIN_FRAMES,
        );
        const allTransient = judged.length > 0 && judged.every((shot) => shot.persistent === false);
        const finalShots = allTransient ? annotated.map((shot) => ({ ...shot, persistent: true })) : annotated;
        shotMarkersRef.current = finalShots;
        setShotLogEntries(finalShots);
      }
      const overlayCtx = overlayEl.getContext("2d");
      overlayCtx?.clearRect(0, 0, overlayEl.width, overlayEl.height);
      // Persist the detailed debug views (last frame) instead of blanking them.
      renderPersistedDetailedViews();
      templateGray.delete();
      if (trackingHist) trackingHist.delete();
      resetShotFlowState();
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
      liveClusterLastUpdatedAtMsRef.current = 0;
      liveContourGroupVisualsRef.current = { updatedAtMs: 0, regionColors: [], regionGroupLabels: [] };
      setIsScanning(false);
      setDetectionEnabled(false);
      setDetectionConfidence(null);
      lastUiDetectionEnabledRef.current = false;
      lastUiDetectionConfidenceRef.current = -1;
      setScanStatus(
        stopRequestedRef.current
          ? "Scan stopped"
          : isStreamScan
            ? "Camera stream analysis complete."
          : requestedWindow
            ? "Focused spike-window analysis complete."
            : "Scan complete (concurrent audio + OpenCV)",
      );
    } catch (error) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (overlayEl) {
        const overlayCtx = overlayEl.getContext("2d");
        overlayCtx?.clearRect(0, 0, overlayEl.width, overlayEl.height);
      }
      if (scanVideoRef.current) {
        scanVideoRef.current.pause();
        scanVideoRef.current = null;
      }
      clearAnalysisCanvases();
      resetShotFlowState();
      shotMarkersRef.current = [];
      shotSequenceRef.current = 0;
      shotPresenceRef.current = new Map();
      committedShotPointsRef.current = [];
      contourShotTracksRef.current = [];
      changeHoleTrackerRef.current = [];
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
      liveClusterLastUpdatedAtMsRef.current = 0;
      liveContourGroupVisualsRef.current = { updatedAtMs: 0, regionColors: [], regionGroupLabels: [] };
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
          void startScan(options);
        }, 0);
      }
    }
  };

  const requestStartScan = () => {
    // Fresh scan: drop the previous blended result background.
    setScanBlendBackgroundUrl(null);
    setStabilizationPx(null);
    lastStabilizationPxRef.current = -1;
    setGroupNames({});
    setGroupAimPoints({});
    // Make sure the Scan step is on screen while the scan runs.
    goToStep(STEP_SCAN);
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
    const playbackSource = resolvePlaybackVideoSource();
    if (!canvasEl || !playbackSource.src) {
      setScanStatus("Analyzed video canvas or video source unavailable.");
      return;
    }

    try {
      if (!playbackSource.fromSession && !hasAnalysisCoverageForWindow(spike.windowStartSec, spike.windowEndSec)) {
        setScanStatus(
          `Session playback cache unavailable. Re-running analysis for ${spike.windowStartSec.toFixed(2)}s-${spike.windowEndSec.toFixed(2)}s.`,
        );
        await startScan({
          forcedWindow: { start: spike.windowStartSec, end: spike.windowEndSec },
          forcedSpike: spike,
        });
        if (!hasAnalysisCoverageForWindow(spike.windowStartSec, spike.windowEndSec)) {
          setScanStatus("Unable to regenerate spike-window analysis data for playback.");
          return;
        }
      }

      stopAnalyzedPlayback();
      if (playPauseTimeoutRef.current) {
        window.clearTimeout(playPauseTimeoutRef.current);
        playPauseTimeoutRef.current = null;
      }

      const playbackVideo = document.createElement("video");
      playbackVideo.src = playbackSource.src;
      playbackVideo.preload = "auto";
      playbackVideo.muted = false;
      playbackVideo.volume = previewVideo ? previewVideo.volume : 1;
      playbackVideo.playsInline = true;
      playbackVideo.playbackRate = playbackSpeedRef.current;
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
      await playbackVideo.play();
      playbackVideo.playbackRate = playbackSpeedRef.current;

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
        const inWindow = (timeSec: number) => timeSec >= spike.windowStartSec && timeSec <= spike.windowEndSec;
        const candidateDetections = detectionTimelineRef.current.length > 0 ? detectionTimelineRef.current : logEntries;
        const nearestDetection = candidateDetections.reduce<DetectionLogEntry | null>((closest, entry) => {
          if (!inWindow(entry.videoTimeSec)) return closest;
          const diff = Math.abs(entry.videoTimeSec - nowSec);
          if (!closest) return entry;
          return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
        }, null);

        const isBlinkOn = Math.floor(performance.now() / 500) % 2 === 0;
        let playbackLabel: string | null = null;
        if (nearestDetection) {
          const playbackDistanceSuffix =
            nearestDetection.estimatedDistanceInches === null
              ? ""
              : ` | ~${formatLinearFromInches(nearestDetection.estimatedDistanceInches, 1)}`;
          playbackLabel = `SPIKE ${nearestDetection.score.toFixed(1)}%${playbackDistanceSuffix}`;
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

        const nearestContourSnapshot = contourWindowTimelineRef.current.reduce<ContourWindowFrameSnapshot | null>(
          (closest, entry) => {
            if (!inWindow(entry.videoTimeSec)) return closest;
            const diff = Math.abs(entry.videoTimeSec - nowSec);
            if (!closest) return entry;
            return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
          },
          null,
        );

        const contourCanvas = processedContourCanvasRef.current;
        const contourContext = contourCanvas?.getContext("2d") ?? null;
        if (contourCanvas && contourContext) {
          if (nearestContourSnapshot) {
            drawPersistedContourWindowView(contourContext, nearestContourSnapshot);
          } else {
            contourContext.clearRect(0, 0, contourCanvas.width, contourCanvas.height);
          }
        }

        const maskCanvas = processedMaskCanvasRef.current;
        const maskContext = maskCanvas?.getContext("2d") ?? null;
        if (maskCanvas && maskContext) {
          if (nearestContourSnapshot) {
            drawBinaryMaskSnapshotWindowView(maskContext, nearestContourSnapshot, "Binary mask (playback)");
          } else {
            maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          }
        }

        const nearestYellowGreenSnapshot = yellowGreenTimelineRef.current.reduce<YellowGreenFrameSnapshot | null>(
          (closest, entry) => {
            if (!inWindow(entry.videoTimeSec)) return closest;
            const diff = Math.abs(entry.videoTimeSec - nowSec);
            if (!closest) return entry;
            return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
          },
          null,
        );

        const yellowGreenCanvas = processedYellowGreenCanvasRef.current;
        const yellowGreenContext = yellowGreenCanvas?.getContext("2d") ?? null;
        if (yellowGreenCanvas && yellowGreenContext) {
          if (nearestYellowGreenSnapshot) {
            drawYellowGreenSnapshotWindowView(
              yellowGreenContext,
              nearestYellowGreenSnapshot,
              "Yellow-green / top-hat (playback)",
            );
          } else {
            yellowGreenContext.clearRect(0, 0, yellowGreenCanvas.width, yellowGreenCanvas.height);
          }
        }

        const playbackShotEntries = shotLogEntries.length > 0 ? shotLogEntries : shotMarkersRef.current;
        const spikeShotMarkers = playbackShotEntries.filter((entry) => inWindow(entry.videoTimeSec));
        const eligiblePlaybackShotMarkers = filterShotsByAnalysisAge(
          spikeShotMarkers,
          nowSec,
          CLUSTER_MIN_VISIBLE_AGE_SEC,
        ).filter((shot) => shotMakesIt(shot, tweakSettings) && passesCaliberGate(shot, tweakSettings));
        const playbackClustering = clusterShotsBySpaceTime(eligiblePlaybackShotMarkers, tweakSettings);
        const playbackClusterByShotId = playbackClustering.shotClusterById;
        const playbackClusterColorById: Record<number, string> = {};
        for (const cluster of playbackClustering.clusters) {
          playbackClusterColorById[cluster.clusterId] = clusterColorForId(cluster.clusterId);
        }
        const nearestSpikeShot = spikeShotMarkers.reduce<ShotLogEntry | null>((closest, entry) => {
          const diff = Math.abs(entry.videoTimeSec - nowSec);
          if (!closest) return entry;
          return diff < Math.abs(closest.videoTimeSec - nowSec) ? entry : closest;
        }, null);
        const patchCanvas = processedPatchCanvasRef.current;
        const patchContext = patchCanvas?.getContext("2d") ?? null;
        const patchSourceRect = nearestDetection
          ? clampRectToFrame(
              nearestDetection.x,
              nearestDetection.y,
              nearestDetection.width,
              nearestDetection.height,
              width,
              height,
            )
          : nearestSpikeShot
            ? clampRectToFrame(
                nearestSpikeShot.drawRectX,
                nearestSpikeShot.drawRectY,
                nearestSpikeShot.drawRectWidth,
                nearestSpikeShot.drawRectHeight,
                width,
                height,
              )
            : null;
        if (patchCanvas && patchContext) {
          if (patchSourceRect) {
            drawPatchWindowView(patchContext, activeVideo, patchSourceRect);
          } else {
            patchContext.clearRect(0, 0, patchCanvas.width, patchCanvas.height);
          }
        }
        let contourRegionColorsForFrame: string[] | undefined;
        let contourRegionGroupLabelsForFrame: string[] | undefined;
        if (nearestContourSnapshot && nearestDetection) {
          const contourVisuals = buildContourRegionClusterVisuals(
            nearestContourSnapshot,
            {
              x: nearestDetection.x,
              y: nearestDetection.y,
              width: Math.max(1, nearestDetection.width),
              height: Math.max(1, nearestDetection.height),
            },
            eligiblePlaybackShotMarkers,
            playbackClusterByShotId,
            playbackClusterColorById,
          );
          contourRegionColorsForFrame = contourVisuals.regionColors;
          contourRegionGroupLabelsForFrame = contourVisuals.regionGroupLabels;
          if (contourCanvas && contourContext) {
            drawPersistedContourWindowView(
              contourContext,
              nearestContourSnapshot,
              contourVisuals.regionColors,
              contourVisuals.regionGroupLabels,
            );
          }
          drawContourRegionsOnTargetView(
            context,
            nearestContourSnapshot,
            {
              x: nearestDetection.x,
              y: nearestDetection.y,
              width: Math.max(1, nearestDetection.width),
              height: Math.max(1, nearestDetection.height),
            },
            isBlinkOn,
            contourVisuals.regionColors,
            pixelsPerInch,
            formatLinearFromInches,
            contourVisuals.regionGroupLabels,
          );
        }
        if (patchContext && patchSourceRect && playbackLabel) {
          drawProbePatchOverlayView(
            patchContext,
            patchSourceRect.width,
            patchSourceRect.height,
            playbackLabel,
            isBlinkOn,
            nearestContourSnapshot,
            contourRegionColorsForFrame,
            pixelsPerInch,
            formatLinearFromInches,
            contourRegionGroupLabelsForFrame,
          );
        }

        if (activeVideo.ended || activeVideo.currentTime >= spike.windowEndSec) {
          stopAnalyzedPlayback();
          return;
        }

        analyzedPlaybackRafRef.current = requestAnimationFrame(renderAnalyzedFrame);
      };

      analyzedPlaybackRafRef.current = requestAnimationFrame(renderAnalyzedFrame);
      setScanStatus(
        `Playing analyzed spike ${spike.id} (${spike.windowStartSec.toFixed(2)}s-${spike.windowEndSec.toFixed(2)}s) [${
          playbackSource.fromSession ? "session-cache" : "live-analysis"
        }]`,
      );
    } catch {
      stopAnalyzedPlayback();
      setScanStatus("Unable to play analyzed video segment for selected spike.");
    }

    if (howlRef.current) {
      howlRef.current.stop();
      howlRef.current.rate?.(playbackSpeedRef.current);
      howlRef.current.play(spikeId);
    }
  };

  return (
    <div className="flex h-[100dvh] flex-col bg-black text-white">
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-hidden px-3 sm:px-8">
        <div className="flex shrink-0 items-center justify-end pt-2">
          <OnboardingGuide currentStep={currentStep} />
        </div>
        {confirmRoiOpen ? (
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Confirm selected area"
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
            onClick={retryRoiSelection}
          >
            <div
              className="w-full max-w-sm rounded-xl border border-gray-700 bg-neutral-950 p-4 text-white shadow-2xl shadow-black/60"
              onClick={(event) => event.stopPropagation()}
            >
              <h2 className="text-base font-semibold">Use this selection?</h2>
              <p className="mt-1 text-xs text-gray-400">This is the target area Trackr will measure for shots.</p>
              <div className="mt-3 flex min-h-[8rem] items-center justify-center overflow-hidden rounded-md border border-gray-700 bg-black p-2">
                {confirmRoiPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={confirmRoiPreview} alt="Selected area preview" className="max-h-64 w-auto max-w-full rounded" />
                ) : (
                  <span className="text-xs text-gray-500">Preparing preview…</span>
                )}
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={retryRoiSelection}
                  className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 transition hover:bg-neutral-800"
                >
                  Try again
                </button>
                <button
                  type="button"
                  onClick={confirmRoiSelection}
                  className="rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-1.5 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
                >
                  Confirm
                </button>
              </div>
            </div>
          </div>
        ) : null}
        {isSettingsModalOpen ? (
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Gears and Tweaks settings"
            className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 backdrop-blur-sm sm:items-center"
            onClick={() => setIsSettingsModalOpen(false)}
          >
            <div
              className="my-auto w-full max-w-2xl rounded-xl border border-gray-700 bg-neutral-950 p-4 shadow-2xl shadow-black/60 sm:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold text-white sm:text-xl">Gears and Tweaks</h2>
                  <p className="mt-1 text-xs text-gray-400">
                    Units, detection, audio correlation, tracking, and clustering settings.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setIsSettingsModalOpen(false)}
                  aria-label="Close settings"
                  className="min-h-9 min-w-9 rounded-md border border-gray-600 px-3 py-1.5 text-base leading-none text-gray-300 transition hover:bg-neutral-800 hover:text-gray-100 sm:min-h-0 sm:min-w-0 sm:px-2 sm:py-1 sm:text-sm"
                >
                  &times;
                </button>
              </div>
              {changedTweakCount > 0 ? (
                <p className="mt-3 rounded-md border border-gray-700 bg-neutral-900 px-3 py-2 text-xs text-gray-200">
                  Warning: {changedTweakCount} custom tweak settings are active.
                </p>
              ) : null}
              <div className="mt-4 max-h-[70vh] space-y-3 overflow-y-auto pr-1">
                <div className="rounded-md border border-gray-700 bg-black p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs uppercase tracking-wide text-gray-300">Accessibility</p>
                    <label className="flex items-center gap-2 text-xs text-gray-200">
                      <input
                        type="checkbox"
                        checked={colorBlindMode}
                        onChange={(event) => setColorBlindMode(event.target.checked)}
                      />
                      Color-blind friendly colors
                    </label>
                  </div>
                  <p className="mt-2 text-[11px] text-gray-400">
                    Swaps shot-group and drill-target colors for a higher-contrast, color-blind-safe palette. For drills,
                    shapes / patterns / numbers remain the most reliable way to tell zones apart.
                  </p>
                </div>
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
            </div>
          </div>
        ) : null}

        {/* Centered, scrollable step content — each step fills the viewport between
            the header and the wizard nav, and is centered until it overflows. */}
        <div className="flex-1 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center py-3">
            <div className="flex w-full flex-col gap-3">
        <section
          ref={videoSourceSectionRef}
          className={`flex flex-col rounded-xl border border-gray-700 bg-neutral-950 p-3 sm:p-6 ${
            currentStep === 0 ? "animate-stepIn" : "hidden"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white sm:text-lg">1. Add your video</h2>
              <p className="mt-1 text-xs text-gray-400">Upload a video of your target, or record with your phone.</p>
            </div>
            <button
              type="button"
              onClick={() => toggleSectionCollapsed("source")}
              className="min-h-9 shrink-0 rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800 sm:min-h-0 sm:px-2 sm:py-1"
            >
              {collapsedSections.source ? "Expand" : "Collapse"}
            </button>
          </div>
          {collapsedSections.source ? (
            <p className="mt-3 text-xs text-emerald-200/80">
              {captureMode === "upload"
                ? selectedVideoName
                  ? `Video: ${selectedVideoName}`
                  : "Upload video selected"
                : streamCameraActive
                  ? "Device camera stream active"
                  : "Device camera stream selected"}
            </p>
          ) : null}
          <div className={`mt-4 ${collapsedSections.source ? "hidden" : ""}`}>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {/* Left column: source controls + reference video info. */}
              <div className="space-y-3">
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
                    Device Camera Stream (Phone)
                  </label>
                </fieldset>

                {captureMode === "upload" ? (
                  <div className="space-y-2">
                    <label className="flex flex-col gap-1">
                      <span className="text-xs uppercase tracking-wide text-gray-300">Video</span>
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
                    </label>
                    {selectedVideoPreviewUrl ? (
                      <div className="space-y-1">
                        <p className={`text-[11px] ${hasReferenceFrame ? "text-emerald-300/80" : "text-amber-300/90"}`}>
                          {hasReferenceFrame
                            ? "Reference frame ready — you can continue."
                            : videoReady
                              ? "Preparing the reference frame…"
                              : "Loading video…"}
                        </p>
                        {videoReady && !hasReferenceFrame ? (
                          <button
                            type="button"
                            onClick={() => captureReferenceFrameFromVideo()}
                            className="rounded-md border border-sky-400/40 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/10"
                          >
                            Use current frame as reference
                          </button>
                        ) : null}
                      </div>
                    ) : (
                      <span className="text-[11px] text-gray-500">Choose a clip to load and preview it.</span>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2 rounded-md border border-gray-700 bg-black p-3">
                    <p className="text-xs uppercase tracking-wide text-gray-300">Device Camera Stream</p>
                    <p className="text-[11px] text-gray-400">
                      Open this app on your phone and use the rear camera for best results.
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          void startStreamCamera();
                        }}
                        className="rounded-md border border-sky-400/35 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/10"
                      >
                        {streamCameraActive ? "Restart Camera" : "Start Camera"}
                      </button>
                      <button
                        type="button"
                        onClick={stopStreamCamera}
                        disabled={!streamCameraActive}
                        className="rounded-md border border-amber-400/35 px-3 py-1.5 text-xs text-amber-100 transition hover:bg-amber-500/10 disabled:opacity-50"
                      >
                        Stop Camera
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          void toggleStreamCameraFacingMode();
                        }}
                        disabled={!streamCameraActive}
                        className="rounded-md border border-emerald-400/35 px-3 py-1.5 text-xs text-emerald-100 transition hover:bg-emerald-500/10 disabled:opacity-50"
                      >
                        Switch to {streamCameraFacingMode === "environment" ? "Front" : "Rear"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setManualMarkMode((value) => !value)}
                        className={`rounded-md border px-3 py-1.5 text-xs transition ${
                          manualMarkMode
                            ? "border-amber-300 bg-amber-500/20 text-amber-100"
                            : "border-gray-600 text-gray-200 hover:bg-neutral-800"
                        }`}
                      >
                        {manualMarkMode ? "Marking hits — tap the video" : "Mark hits manually"}
                      </button>
                    </div>
                    {streamCameraError ? <p className="text-xs text-rose-300">{streamCameraError}</p> : null}
                  </div>
                )}
              </div>

              {/* Right column: preview (drops below the controls on mobile). */}
              <div className="space-y-2">
                {captureMode === "upload" ? (
                  selectedVideoPreviewUrl ? (
                    <>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          setManualMarkMode((value) => !value);
                        }}
                        className={`rounded-md border px-3 py-1.5 text-xs transition ${
                          manualMarkMode
                            ? "border-amber-300 bg-amber-500/20 text-amber-100"
                            : "border-gray-600 text-gray-200 hover:bg-neutral-800"
                        }`}
                      >
                        {manualMarkMode ? "Marking hits — tap the video" : "Mark hits manually"}
                      </button>
                      <div className="relative">
                        <video
                          ref={videoRef}
                          src={selectedVideoPreviewUrl}
                          controls
                          onLoadedData={markUploadedVideoReady}
                          onCanPlay={markUploadedVideoReady}
                          onDurationChange={markUploadedVideoReady}
                          onError={() => setVideoReady(false)}
                          className="max-h-72 w-full rounded-md border border-gray-700 sm:max-h-[60vh]"
                        />
                        <canvas
                          ref={overlayCanvasRef}
                          onPointerDown={handleVideoMarkTap}
                          onClick={(event) => {
                            if (manualMarkMode) {
                              event.preventDefault();
                              event.stopPropagation();
                            }
                          }}
                          className={`absolute inset-0 h-full w-full rounded-md ${
                            manualMarkMode ? "cursor-crosshair" : "pointer-events-none"
                          }`}
                        />
                      </div>
                    </>
                  ) : (
                    <div className="flex h-40 items-center justify-center rounded-md border border-dashed border-gray-700 bg-black px-4 text-center text-xs text-gray-500 sm:h-56">
                      Upload a video to preview it here.
                    </div>
                  )
                ) : (
                  <div className="relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      muted
                      playsInline
                      className="max-h-72 w-full rounded-md border border-gray-700 bg-black sm:max-h-[60vh]"
                    />
                    <canvas
                      ref={overlayCanvasRef}
                      onPointerDown={handleVideoMarkTap}
                      className={`absolute inset-0 h-full w-full rounded-md ${
                        manualMarkMode ? "cursor-crosshair" : "pointer-events-none"
                      }`}
                    />
                    {!streamCameraActive ? (
                      <p className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-gray-400">
                        Camera preview inactive
                      </p>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section
          ref={captureSectionRef}
          className={`flex flex-col rounded-xl border border-gray-700 bg-neutral-950 p-3 sm:p-6 ${
            currentStep === 1 ? "animate-stepIn" : "hidden"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white sm:text-lg">2. Mark the target</h2>
              <p className="mt-1 text-xs text-gray-400">Click and drag a box around the target.</p>
            </div>
            <button
              type="button"
              onClick={() => toggleSectionCollapsed("capture")}
              className="min-h-9 shrink-0 rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800 sm:min-h-0 sm:px-2 sm:py-1"
            >
              {collapsedSections.capture ? "Expand" : "Collapse"}
            </button>
          </div>
          {collapsedSections.capture ? (
            <p className="mt-3 text-xs text-emerald-200/80">
              {hasDrawnGeometry ? "Reference frame captured and target box set" : "Reference frame captured"}
            </p>
          ) : null}
          <div className={`mt-4 ${collapsedSections.capture ? "hidden" : ""}`}>
            {(captureMode === "upload" ? !!selectedVideoPreviewUrl : streamCameraActive) ? (
                <div className="rounded-md border border-gray-700 bg-black p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-xs uppercase tracking-wide text-gray-300">Target photo</p>
                    <div className="flex items-center gap-2">
                      {selectedImageName ? <span className="text-[11px] text-gray-400">{selectedImageName}</span> : null}
                      {selectedImagePreviewUrl ? (
                        <button
                          type="button"
                          onClick={() => setChoosingDifferentFrame((value) => !value)}
                          aria-expanded={choosingDifferentFrame}
                          className="min-h-8 rounded-md border border-gray-600 px-3 text-xs text-gray-200 transition hover:bg-neutral-800 sm:min-h-0 sm:py-1"
                        >
                          {choosingDifferentFrame ? "Cancel" : "Choose Different"}
                        </button>
                      ) : null}
                    </div>
                  </div>

                  {selectedImagePreviewUrl ? (
                    <>
                      {/* Reference image is shown by default (first frame). Drag a box, or Auto-pick. */}
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
                        {(() => {
                          const box = draftRoiRect ?? roiRect;
                          if (!box) return null;
                          const isDraft = draftRoiRect !== null;
                          return (
                            <div
                              className="pointer-events-none absolute border-2"
                              style={{
                                left: `${box.x * 100}%`,
                                top: `${box.y * 100}%`,
                                width: `${box.width * 100}%`,
                                height: `${box.height * 100}%`,
                                borderColor: isDraft ? "#38bdf8" : "#22c55e",
                                backgroundColor: isDraft ? "rgba(56, 189, 248, 0.16)" : "rgba(34, 197, 94, 0.22)",
                                borderStyle: isDraft ? "dashed" : "solid",
                              }}
                            />
                          );
                        })()}
                      </div>

                      <p className="mt-2 text-[11px] text-gray-400">
                        {hasDrawnGeometry
                          ? "Target box set. Drag a new rectangle to re-box, or continue."
                          : "Drag a rectangle around the target on the image above."}
                      </p>

                      {/* Target-picking options appear now that the reference image is shown. */}
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        {SHOW_AUTO_PICK && captureMode === "upload" ? (
                          <button
                            type="button"
                            onClick={() => {
                              void autoPickReferenceFrameFromVideo();
                            }}
                            disabled={isAutoPicking || !opencvReady}
                            className="min-h-9 rounded-md border border-emerald-400/40 px-3 py-1.5 text-xs text-emerald-100 transition hover:bg-emerald-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:min-h-0"
                            title={
                              targetTemplates.length > 0
                                ? "Scrub the video and match the target outlines you've drawn before; falls back to shape detection. Picks the best frame and boxes the target automatically."
                                : "Scrub the video to find the most common object and use that frame as the reference, with a target box drawn automatically. Draw ROIs to teach it your targets."
                            }
                          >
                            {isAutoPicking
                              ? "Auto-picking…"
                              : targetTemplates.length > 0
                                ? "Auto-pick target"
                                : "Auto-pick (find common object)"}
                          </button>
                        ) : null}
                        {hasDrawnGeometry ? (
                          <>
                            <button
                              type="button"
                              onClick={clearRoiSelection}
                              className="min-h-9 rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-300 transition hover:bg-neutral-800 sm:min-h-0"
                            >
                              Clear Selection
                            </button>
                            <button
                              type="button"
                              onClick={goToNextAfterGeometrySelection}
                              className={`min-h-9 rounded-md border px-3 py-1.5 text-xs transition sm:min-h-0 ${highlightActionClass}`}
                            >
                              Next
                            </button>
                          </>
                        ) : null}
                      </div>

                      {/* "Choose Different" reveals frame scrubbing/recapture. */}
                      {choosingDifferentFrame ? (
                        <div className="mt-2 rounded-md border border-gray-700 bg-neutral-900/50 p-2">
                          <p className="text-[11px] text-gray-400">
                            {captureMode === "upload"
                              ? "Scrub the reference video in step 1 to the frame you want, then capture it."
                              : "Position the camera, then capture the current frame."}
                          </p>
                          <button
                            ref={captureFrameButtonRef}
                            type="button"
                            onClick={() => captureReferenceFrameFromVideo()}
                            disabled={isAutoPicking}
                            className="mt-2 min-h-9 rounded-md border border-sky-400/35 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:min-h-0"
                          >
                            Use Current Video Frame
                          </button>
                        </div>
                      ) : null}

                      {SHOW_AUTO_PICK && captureMode === "upload" ? (
                        <p className="mt-2 text-[11px] text-gray-400">
                          {targetTemplates.length > 0 ? (
                            <>
                              Auto-pick is learning from{" "}
                              <span className="text-emerald-200">
                                {targetTemplates.length} target{targetTemplates.length === 1 ? "" : "s"}
                              </span>{" "}
                              you&apos;ve drawn.{" "}
                              <button
                                type="button"
                                onClick={clearTargetTemplates}
                                className="text-rose-300 underline-offset-2 hover:underline"
                              >
                                Clear
                              </button>
                            </>
                          ) : (
                            "Each box you draw is saved and matched on future videos to improve Auto-pick."
                          )}
                        </p>
                      ) : null}
                    </>
                  ) : (
                    <div className="mt-2">
                      <p className="text-xs text-gray-400">
                        {captureMode === "upload"
                          ? "Preparing the first video frame as your reference image…"
                          : "Capture a frame from the camera to use as your reference image."}
                      </p>
                      <button
                        ref={captureFrameButtonRef}
                        type="button"
                        onClick={() => captureReferenceFrameFromVideo()}
                        disabled={isAutoPicking}
                        className={`mt-2 min-h-9 rounded-md border border-sky-400/35 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/10 disabled:cursor-not-allowed disabled:opacity-50 sm:min-h-0 ${
                          workflowStep === "capture_frame" ? highlightActionClass : ""
                        }`}
                      >
                        Use Current Video Frame
                      </button>
                    </div>
                  )}
                </div>
            ) : (
              <p className="text-sm text-gray-300">
                {captureMode === "upload"
                  ? "Upload a reference video to enable frame capture and ROI selection."
                  : "Start the device camera stream to enable frame capture and ROI selection."}
              </p>
            )}
          </div>
        </section>

        <section
          ref={calibrationSectionRef}
          className={`flex flex-col rounded-xl border bg-neutral-950 p-3 sm:p-5 ${
            workflowStep === "calibrate" ? "border-amber-300/70 ring-2 ring-amber-300/35" : "border-gray-700"
          } ${
            currentStep === STEP_CALIB_METHOD || currentStep === STEP_CALIB_VALUES ? "animate-stepIn" : "hidden"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white sm:text-lg">
                {currentStep === STEP_CALIB_METHOD ? "3. Set your caliber & method" : "4. Enter the measurements"}
              </h2>
              <p className="mt-1 text-xs text-gray-400">
                {currentStep === STEP_CALIB_METHOD
                  ? "Pick your caliber, then how you'll tell the app the target's real size."
                  : "Enter the values for the method you chose, so sizes come out in real inches."}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {focalScalePxIn > 0 ? (
                <span className="text-[11px] text-gray-400">Scale: {focalScalePxIn.toFixed(1)}</span>
              ) : null}
              <button
                type="button"
                onClick={() => toggleSectionCollapsed("calibrate")}
                className="rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-200 transition hover:bg-neutral-800"
              >
                {collapsedSections.calibrate ? "Expand" : "Collapse"}
              </button>
            </div>
          </div>
          {collapsedSections.calibrate ? (
            <p className="mt-3 text-xs text-emerald-200/80">
              {hasScaleCalibration
                ? `Calibrated${focalScalePxIn > 0 ? ` · scale ${focalScalePxIn.toFixed(1)}` : ""}`
                : "Target dimensions pending"}
            </p>
          ) : null}
          <div className={collapsedSections.calibrate ? "hidden" : ""}>
          {/* Caliber + method picker — the "Calibration" (method) step. */}
          <div className={currentStep === STEP_CALIB_METHOD ? "" : "hidden"}>
          {/* Caliber — shown by default, independent of the calibration method. */}
          <div className="mt-3 rounded-md border border-gray-800 bg-neutral-950/60 p-2">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] text-gray-400">Expected hole / caliber ({activeLinearUnitLabel}, 0 = off)</span>
              <LinearNumberInput
                valueInches={tweakSettings.expectedHoleDiameterInches}
                toDisplay={(v) => toDisplayLinearValue(v)}
                fromDisplay={fromDisplayLinearValue}
                onChangeInches={(v) => updateTweakSetting("expectedHoleDiameterInches", v)}
                className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1 sm:max-w-xs"
              />
            </label>
            <p className="mt-2 text-[11px] text-gray-400">Tap a caliber to fill it — values convert automatically:</p>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {CALIBER_PRESETS.map((preset) => {
                const active = Math.abs(tweakSettings.expectedHoleDiameterInches - preset.inches) < 1e-6;
                return (
                  <button
                    key={preset.label}
                    type="button"
                    aria-pressed={active}
                    onClick={() => updateTweakSetting("expectedHoleDiameterInches", preset.inches)}
                    className={`rounded-full border px-2 py-0.5 text-[11px] transition ${
                      active
                        ? "border-sky-400 bg-sky-500/25 text-sky-100"
                        : "border-gray-700 text-gray-200 hover:border-sky-400/50 hover:bg-sky-500/10"
                    }`}
                  >
                    {preset.label} · {toDisplayLinearValue(preset.inches, 2)} {activeLinearUnitLabel}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Pick a calibration method; the matching controls show below. */}
          <p className="mt-3 text-xs uppercase tracking-wide text-gray-300">Calibration method</p>
          <div className="mt-1.5 inline-flex flex-wrap overflow-hidden rounded-md border border-gray-600 text-xs">
            {([
              { id: "line", label: "Line on image" },
              { id: "dimensions", label: "Target dimensions" },
              { id: "qr", label: "QR code" },
              { id: "manual", label: "Manual" },
            ] as const).map((method, index) => (
              <button
                key={method.id}
                type="button"
                onClick={() => {
                  userPickedCalibMethodRef.current = true;
                  setCalibMethod(method.id);
                }}
                aria-pressed={calibMethod === method.id}
                className={`min-h-9 px-3 transition sm:min-h-0 sm:py-1 ${index > 0 ? "border-l border-gray-600" : ""} ${
                  calibMethod === method.id ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                }`}
              >
                {method.label}
              </button>
            ))}
          </div>
          </div>
          {/* Method-specific value inputs — the "Measurements" step. */}
          <div className={currentStep === STEP_CALIB_VALUES ? "" : "hidden"}>

          {calibMethod === "line" && hasDrawnGeometry ? (
            <div className="mt-2 rounded-md border border-gray-700 bg-black/35 p-3">
              <p className="text-xs uppercase tracking-wide text-gray-300">ROI Line Calibration</p>
              <p className="mt-1 text-[11px] text-gray-400">
                Draw a line on the magnified ROI and enter its real-world length — calibration applies automatically.
              </p>
              {roiMagnifiedDataUrl && roiSelectionPixelSize ? (
                <>
                  <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                    <div className="inline-flex overflow-hidden rounded-md border border-gray-600 text-xs">
                      {([
                        { id: "measure", label: "Measure" },
                        { id: "pan", label: "Pan" },
                        { id: "zoom", label: "Zoom box" },
                      ] as const).map((tool, index) => (
                        <button
                          key={tool.id}
                          type="button"
                          onClick={() => setRoiTool(tool.id)}
                          aria-pressed={roiTool === tool.id}
                          className={`min-h-9 px-3 transition sm:min-h-0 sm:py-1 ${index > 0 ? "border-l border-gray-600" : ""} ${
                            roiTool === tool.id ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                          }`}
                        >
                          {tool.label}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        aria-label="Zoom out"
                        onClick={() => setRoiZoom((z) => Math.max(1, Math.round((z - 0.5) * 10) / 10))}
                        disabled={roiZoom <= 1}
                        className="flex h-10 w-10 items-center justify-center rounded-md border border-gray-600 text-xl leading-none text-gray-100 transition hover:bg-neutral-800 disabled:opacity-40"
                      >
                        −
                      </button>
                      <span className="w-12 text-center text-sm tabular-nums text-gray-300">{roiZoom.toFixed(1)}×</span>
                      <button
                        type="button"
                        aria-label="Zoom in"
                        onClick={() => setRoiZoom((z) => Math.min(6, Math.round((z + 0.5) * 10) / 10))}
                        disabled={roiZoom >= 6}
                        className="flex h-10 w-10 items-center justify-center rounded-md border border-gray-600 text-xl leading-none text-gray-100 transition hover:bg-neutral-800 disabled:opacity-40"
                      >
                        +
                      </button>
                      <button
                        type="button"
                        onClick={() => setRoiZoom(1)}
                        disabled={roiZoom === 1}
                        className="ml-1 min-h-10 rounded-md border border-gray-600 px-3 text-sm text-gray-300 transition hover:bg-neutral-800 disabled:opacity-40"
                      >
                        Fit
                      </button>
                    </div>
                  </div>
                  <p className="mt-1 text-[11px] text-gray-500">
                    {roiTool === "measure"
                      ? "Drag a line across a known length, then enter it below."
                      : roiTool === "pan"
                        ? "Drag to pan the zoomed image."
                        : "Drag a box to zoom into that region."}
                  </p>
                  <div ref={roiScrollRef} className="relative mt-1 max-h-[500px] overflow-auto rounded-md border border-gray-700 bg-black">
                    <canvas
                      ref={roiMeasurementCanvasRef}
                      className={`block h-auto max-w-none touch-none ${
                        roiTool === "measure" ? "cursor-crosshair" : roiTool === "pan" ? "cursor-grab active:cursor-grabbing" : "cursor-zoom-in"
                      }`}
                      style={{ width: `${roiZoom * 100}%`, imageRendering: "pixelated" }}
                      onPointerDown={onRoiPointerDown}
                      onPointerMove={onRoiPointerMove}
                      onPointerUp={onRoiPointerUp}
                      onPointerLeave={onRoiPointerLeave}
                    />
                  </div>
                  {roiTool === "zoom" && roiZoomBox ? (
                    <div
                      className="pointer-events-none fixed z-50 rounded border-2 border-sky-400 bg-sky-400/15"
                      style={{
                        left: Math.min(roiZoomBox.x0, roiZoomBox.x1),
                        top: Math.min(roiZoomBox.y0, roiZoomBox.y1),
                        width: Math.abs(roiZoomBox.x1 - roiZoomBox.x0),
                        height: Math.abs(roiZoomBox.y1 - roiZoomBox.y0),
                      }}
                    />
                  ) : null}
                  <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
                    <label className="flex flex-col gap-1">
                      <span className="text-[11px] text-gray-400">Measured Line Length ({activeLinearUnitLabel})</span>
                      <LinearNumberInput
                        valueInches={roiMeasurementLengthInches}
                        toDisplay={(v) => toDisplayLinearValue(v)}
                        fromDisplay={fromDisplayLinearValue}
                        onChangeInches={setRoiMeasurementLengthInches}
                        className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => setRoiMeasurementLine(null)}
                      className="min-h-9 self-end rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-300 transition hover:bg-neutral-800 sm:min-h-0"
                    >
                      Clear Line
                    </button>
                  </div>
                  {roiMeasurementMetrics?.pixelsPerInch ? (
                    <p className="mt-2 text-[11px] text-emerald-300/90">
                      Auto-applied: {roiMeasurementMetrics.pixelLength.toFixed(1)} px ={" "}
                      {formatLinearFromInches(roiMeasurementMetrics.knownLengthInches, 3)} ·{" "}
                      {roiMeasurementMetrics.pixelsPerInch.toFixed(2)} ppi
                    </p>
                  ) : null}
                  <p className="mt-2 text-[11px] text-gray-400">
                    ROI size: {roiSelectionPixelSize.widthPx}px x {roiSelectionPixelSize.heightPx}px
                    {roiMeasurementMetrics
                      ? ` | line=${roiMeasurementMetrics.pixelLength.toFixed(1)}px | ppi=${(
                          roiMeasurementMetrics.pixelsPerInch ?? 0
                        ).toFixed(2)}`
                      : " | draw line to compute ppi"}
                  </p>
                </>
              ) : (
                <p className="mt-2 text-[11px] text-gray-500">Draw target geometry first to enable magnified line calibration.</p>
              )}
            </div>
          ) : calibMethod === "line" ? (
            <p className="mt-2 text-[11px] text-gray-500">Draw a target box in step 2 first, then measure a line here.</p>
          ) : null}

          {calibMethod === "dimensions" ? (
            <div className="mt-2 rounded-md border border-gray-700 bg-black/35 p-3">
              <p className="text-[11px] text-gray-400">Type the target&apos;s real width and height.</p>
              <div className="mt-2 grid grid-cols-1 gap-2 xs:grid-cols-2">
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] text-gray-400">Target Width ({activeLinearUnitLabel})</span>
                  <LinearNumberInput
                    valueInches={targetWidthInches}
                    toDisplay={(v) => toDisplayLinearValue(v)}
                    fromDisplay={fromDisplayLinearValue}
                    onChangeInches={setTargetWidthInches}
                    className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] text-gray-400">Target Height ({activeLinearUnitLabel})</span>
                  <LinearNumberInput
                    valueInches={targetHeightInches}
                    toDisplay={(v) => toDisplayLinearValue(v)}
                    fromDisplay={fromDisplayLinearValue}
                    onChangeInches={setTargetHeightInches}
                    className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                </label>
              </div>
            </div>
          ) : null}

          {calibMethod === "manual" ? (
            <div className="mt-2 rounded-md border border-gray-700 bg-black/35 p-3">
              <p className="text-[11px] text-gray-400">Enter the scale directly, or the known shooting distance.</p>
              <div className="mt-2 grid grid-cols-1 gap-2 xs:grid-cols-2">
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] text-gray-400">Pixels Per Inch</span>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    value={pixelsPerInch}
                    onChange={(event) => setPixelsPerInch(Math.max(0, Number(event.target.value) || 0))}
                    className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] text-gray-400">Known Distance At Calibration ({activeLinearUnitLabel})</span>
                  <LinearNumberInput
                    valueInches={calibrationDistanceInches}
                    toDisplay={(v) => toDisplayLinearValue(v)}
                    fromDisplay={fromDisplayLinearValue}
                    onChangeInches={setCalibrationDistanceInches}
                    className="rounded-md border border-gray-700 bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                </label>
              </div>
            </div>
          ) : null}
          {/* QR calibration method. */}
          {calibMethod === "qr" ? (
          <div
            className={`mt-2 rounded-md border p-2 ${
              qrAutoScan?.found && !qrAutoScan.hasEncodedSize
                ? "border-amber-400/50 bg-amber-500/5"
                : "border-gray-700 bg-black/35"
            }`}
          >
            {qrAutoScan?.found ? (
              <p className="text-[11px] text-amber-200">
                {qrAutoScan.hasEncodedSize
                  ? "QR detected — it carries its own printed size, so no manual entry is needed."
                  : "QR detected, but it doesn't carry its size — enter the QR's printed size to calibrate from it."}
              </p>
            ) : (
              <p className="text-[11px] text-gray-400">
                Enter the QR&apos;s printed size to calibrate from it, or open a target via its QR.
              </p>
            )}
            <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
              <label className="flex flex-col gap-1">
                <span className="text-[11px] text-gray-400">QR Code Printed Size ({activeLinearUnitLabel}, 0 = none)</span>
                <LinearNumberInput
                  valueInches={manualQrSizeInches}
                  toDisplay={(v) => toDisplayLinearValue(v)}
                  fromDisplay={fromDisplayLinearValue}
                  onChangeInches={setManualQrSizeInches}
                  placeholder="e.g. QR side length"
                  className={`rounded-md border bg-neutral-950 px-2 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1 ${
                    qrAutoScan?.found && !qrAutoScan.hasEncodedSize ? "border-amber-400/60" : "border-gray-700"
                  }`}
                />
              </label>
              <button
                type="button"
                onClick={calibrateFromQr}
                disabled={!opencvReady}
                className="min-h-9 rounded-md border border-sky-400/40 bg-sky-500/15 px-3 py-1.5 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Calibrate from QR
              </button>
            </div>
            <p className="mt-1.5 text-[11px] text-gray-400">
              {qrCalibrationStatus ??
                (manualQrSizeInches > 0
                  ? `Manual QR size set (${formatLinearFromInches(manualQrSizeInches, 2)}). Select the region containing the QR, then calibrate — it'll be highlighted on the reference image.`
                  : "Select the target region containing the QR, then calibrate — the QR is highlighted on the reference image.")}
            </p>
          </div>
          ) : null}
          </div>
          </div>
        </section>

        <section
          ref={analysisSectionRef}
          className={`flex flex-col rounded-xl border border-gray-700 bg-neutral-950 p-3 sm:p-6 ${
            currentStep >= STEP_ANALYSIS_FIRST && currentStep <= STEP_ANALYSIS_LAST ? "animate-stepIn" : "hidden"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white sm:text-lg">
                {currentStep === STEP_SCAN ? "5. Scan for shots" : currentStep === STEP_MAP ? "6. Shot map" : "7. Review"}
              </h2>
              <p className="mt-1 text-xs text-gray-400">
                {currentStep === STEP_SCAN
                  ? "Press Start. Your hits appear on the target below as they're found."
                  : currentStep === STEP_MAP
                    ? "Edit groups, mark strays/false positives, and scrub the reveal window."
                    : "Group stats, the full shot table, and detector detail views."}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setIsSettingsModalOpen(true)}
              className="flex min-h-9 shrink-0 items-center gap-1.5 rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800 sm:min-h-0 sm:px-2.5 sm:py-1"
              aria-haspopup="dialog"
              aria-expanded={isSettingsModalOpen}
            >
              <span aria-hidden="true">⚙</span>
              Settings
              {changedTweakCount > 0 ? (
                <span className="rounded-full bg-amber-500/20 px-1.5 text-[10px] font-semibold text-amber-200">
                  {changedTweakCount}
                </span>
              ) : null}
            </button>
          </div>

          {/* Scan controls (merged in from the old standalone Scan step). */}
          <div
            ref={scanSectionRef}
            className={`mt-4 space-y-3 ${currentStep === STEP_SCAN || isScanning ? "" : "hidden"}`}
          >
            <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
              <button
                ref={startScanButtonRef}
                type="button"
                onClick={requestStartScan}
                disabled={!opencvReady || (captureMode === "upload" && !howlerReady)}
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
              <button
                type="button"
                onClick={toggleScanAudio}
                aria-pressed={!scanAudioMuted}
                title={scanAudioMuted ? "Sound is off during scan & replay" : "Sound is on during scan & replay"}
                className="flex w-full items-center justify-center gap-1.5 rounded-md border border-gray-600 px-3 py-2.5 text-sm text-gray-200 transition hover:bg-neutral-800 sm:w-auto"
              >
                {scanAudioMuted ? "🔇 Sound off" : "🔊 Sound on"}
              </button>
              <label
                className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-600 px-3 py-2.5 text-sm text-gray-300 sm:w-auto"
                title="Playback volume (scan & replay)"
              >
                <span aria-hidden="true">🔉</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={playbackVolume}
                  onChange={(event) => changePlaybackVolume(Number(event.target.value))}
                  aria-label="Playback volume"
                  className="w-24"
                />
                <span className="w-9 shrink-0 text-right tabular-nums">{Math.round(playbackVolume * 100)}%</span>
              </label>
              <label
                className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-600 px-3 py-2.5 text-sm text-gray-300 sm:w-auto"
                title="Contrast change detection always runs; this adds bright yellow-green splatter detection for shoot-n-see targets"
              >
                <input
                  type="checkbox"
                  checked={detectBrightColors}
                  onChange={(event) => setDetectBrightColors(event.target.checked)}
                />
                <span>🎯 Splatter colors</span>
              </label>
              <label
                className="flex w-full items-center justify-center gap-2 rounded-md border border-gray-600 px-3 py-2.5 text-sm text-gray-300 sm:w-auto"
                title="Skip playback to the audio-spike windows. Turn off to scan every frame (use when the clip's sound doesn't line up with the impacts)"
              >
                <input
                  type="checkbox"
                  checked={audioGatedScan}
                  onChange={(event) => setAudioGatedScan(event.target.checked)}
                />
                <span>🎧 Audio-gated</span>
              </label>
            </div>
            {/* Why Start is disabled — so a still-loading button isn't mistaken for broken. */}
            {!isScanning && (!opencvReady || (captureMode === "upload" && !howlerReady)) ? (
              <p className="flex items-center gap-2 text-[11px] text-amber-300/90">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" aria-hidden="true" />
                {!opencvReady
                  ? "Loading the vision engine (OpenCV) from its CDN…"
                  : "Loading the audio engine…"}{" "}
                Start enables once it&apos;s ready. If this never finishes, check your network / ad-blocker (the engines load from a CDN).
              </p>
            ) : null}
            {/* Target stabilization indicator. */}
            {stabilizationPx !== null ? (
              <div className="flex items-center gap-2 rounded-md border border-gray-800 bg-neutral-950 p-2 text-xs text-gray-300">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${stabilizationPx > 6 ? "bg-amber-400" : "bg-emerald-400"}`}
                  aria-hidden="true"
                />
                <span>
                  <span className="font-semibold text-gray-100">Stabilizing target</span> —{" "}
                  {stabilizationPx <= 0
                    ? "camera is steady."
                    : `correcting ~${stabilizationPx}px of camera shake so it isn't counted as hits.`}
                </span>
              </div>
            ) : null}
            {/* Detection trail — the pipeline, step by step, live. */}
            <div className="rounded-lg border border-gray-800 bg-neutral-950 p-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-gray-100">🧠 Detection trail — how hits are found</h3>
                <button
                  type="button"
                  onClick={() => setShowDetectionTrail((prev) => !prev)}
                  className="text-xs font-medium text-sky-300 hover:underline"
                >
                  {showDetectionTrail ? "Hide" : "Show"}
                </button>
              </div>
              {showDetectionTrail ? (
                <div className="mt-2 space-y-3">
                  <p className="text-xs text-gray-400">
                    Stabilize → subtract the start frame → find blobs → confirm across frames → check against the end
                    frame. The sound track below decides <em>where</em> to look.
                  </p>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                    <figure>
                      <canvas
                        ref={trailBaselineCanvasRef}
                        className="w-full rounded border border-gray-800 bg-black [image-rendering:pixelated]"
                      />
                      <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                        <span className="font-semibold text-gray-200">1 · Start frame</span> — the clean target
                        (baseline), re-aligned to each frame
                        {trailInfo ? ` (stabilized ${trailInfo.shiftPx}px)` : ""}
                      </figcaption>
                    </figure>
                    <figure>
                      <canvas
                        ref={trailCurrentCanvasRef}
                        className="w-full rounded border border-gray-800 bg-black [image-rendering:pixelated]"
                      />
                      <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                        <span className="font-semibold text-gray-200">2 · Live frame</span> — the target patch being
                        analyzed{trailInfo ? ` at ${trailInfo.timeSec.toFixed(2)}s` : ""}
                      </figcaption>
                    </figure>
                    <figure>
                      <canvas
                        ref={trailMaskCanvasRef}
                        className="w-full rounded border border-gray-800 bg-black [image-rendering:pixelated]"
                      />
                      <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                        <span className="font-semibold text-gray-200">3 · Bit mask</span> — red = differs from the
                        start frame
                        {trailInfo ? ` (${trailInfo.changedPixels}px, ${trailInfo.changedPct.toFixed(1)}%)` : ""}
                      </figcaption>
                    </figure>
                    <figure>
                      <canvas
                        ref={trailBlobsCanvasRef}
                        className="w-full rounded border border-gray-800 bg-black [image-rendering:pixelated]"
                      />
                      <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                        <span className="font-semibold text-gray-200">4 · Blobs</span> — amber = candidate, green =
                        hole-shaped ✓, cyan = splatter; circles = tracks with their sighting count
                      </figcaption>
                    </figure>
                    <figure>
                      <canvas
                        ref={trailEndCanvasRef}
                        className="w-full rounded border border-gray-800 bg-black [image-rendering:pixelated]"
                      />
                      <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                        <span className="font-semibold text-gray-200">5 · End frame</span> — permanent changes only; a
                        real hit must appear here too
                        {trailInfo ? (trailInfo.endGateActive ? " (check ON)" : " (check off)") : ""}
                      </figcaption>
                    </figure>
                  </div>
                  <figure>
                    <canvas ref={trailAudioCanvasRef} className="w-full rounded border border-gray-800 bg-black" />
                    <figcaption className="mt-1 text-[11px] leading-tight text-gray-400">
                      <span className="font-semibold text-gray-200">Sound</span> — gray curve = loudness, amber line =
                      spike tolerance, red ● = detected bangs, blue bands = the 0.5s scan windows, white line =
                      playhead{trailInfo && !trailInfo.inWindow ? " · skipping dead time…" : ""}
                    </figcaption>
                  </figure>
                  {trailInfo ? (
                    <p className="text-[11px] text-gray-400">
                      This frame: {trailInfo.blobCount} candidate blob{trailInfo.blobCount === 1 ? "" : "s"} →{" "}
                      {trailInfo.acceptedCount} hole-shaped
                      {trailInfo.splatterCount > 0 ? ` + ${trailInfo.splatterCount} splatter` : ""} ·{" "}
                      {trailInfo.pendingTracks} awaiting confirmation ·{" "}
                      <span className="font-semibold text-gray-100">{trailInfo.shotCount} shots committed</span>
                    </p>
                  ) : (
                    <p className="text-[11px] text-gray-500">Start a scan to watch the pipeline live.</p>
                  )}
                </div>
              ) : null}
            </div>
            {SHOW_DETECTOR_TOGGLES && (
              <label className="flex items-start gap-2 rounded-md border border-gray-800 bg-neutral-950 p-2 text-xs text-gray-300">
                <input
                  type="checkbox"
                  checked={useChangeDetector}
                  onChange={(event) => setUseChangeDetector(event.target.checked)}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-semibold text-gray-100">Detect new marks</span> — finds hits by spotting fresh
                  round marks on the target (works without sound).
                  <span className="mt-1 block text-gray-400">
                    {tweakSettings.expectedHoleDiameterInches > 0
                      ? "Only marks about your caliber's size are counted."
                      : "Set your caliber in step 3 so only bullet-size marks count."}
                  </span>
                </span>
              </label>
            )}
            {SHOW_DETECTOR_TOGGLES && useChangeDetector && (
              <label className="flex items-start gap-2 rounded-md border border-gray-800 bg-neutral-950 p-2 text-xs text-gray-300">
                <input
                  type="checkbox"
                  checked={confirmWithClassifier}
                  onChange={(event) => setConfirmWithClassifier(event.target.checked)}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-semibold text-gray-100">Double-check each hit</span> — confirms every mark really
                  looks like a bullet hole before counting it, to cut false hits.
                </span>
              </label>
            )}
          </div>

          <div className="mt-4 rounded-md border border-gray-700 bg-black p-3 sm:p-4">
            {/* Map + tools — shown on the Scan & Map steps, hidden on Review. */}
            <div className={currentStep === STEP_REVIEW ? "hidden" : ""}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-semibold uppercase tracking-wide text-gray-200">
                Shot Group Map — target region
              </p>
              {SHOW_GROUPING_MODE_TOGGLE ? (
              <div className="inline-flex overflow-hidden rounded-md border border-gray-600 text-xs">
                <button
                  type="button"
                  onClick={() => setShotGroupMode("dbscan")}
                  className={`min-h-9 px-3 py-1.5 transition sm:min-h-0 sm:px-2 sm:py-1 ${
                    shotGroupMode === "dbscan" ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                  }`}
                  aria-pressed={shotGroupMode === "dbscan"}
                  title="Group shots by where and when they landed"
                >
                  By time + place
                </button>
                <button
                  type="button"
                  onClick={() => setShotGroupMode("quadtree")}
                  className={`min-h-9 border-l border-gray-600 px-3 py-1.5 transition sm:min-h-0 sm:px-2 sm:py-1 ${
                    shotGroupMode === "quadtree" ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                  }`}
                  aria-pressed={shotGroupMode === "quadtree"}
                  title="Group shots by where they landed"
                >
                  By place
                </button>
              </div>
              ) : null}
              <div className="inline-flex overflow-hidden rounded-md border border-gray-600 text-xs">
                <button
                  type="button"
                  onClick={() => setShotMapLiveStream(false)}
                  className={`min-h-9 px-3 py-1.5 transition sm:min-h-0 sm:px-2 sm:py-1 ${
                    !shotMapLiveStream ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                  }`}
                  aria-pressed={!shotMapLiveStream}
                  title="Static reference frame, updated as shots are detected"
                >
                  Reference
                </button>
                <button
                  type="button"
                  onClick={() => setShotMapLiveStream(true)}
                  className={`min-h-9 border-l border-gray-600 px-3 py-1.5 transition sm:min-h-0 sm:px-2 sm:py-1 ${
                    shotMapLiveStream ? "bg-sky-500/20 text-sky-100" : "text-gray-300 hover:bg-neutral-800"
                  }`}
                  aria-pressed={shotMapLiveStream}
                  title="Stream the cropped ROI of the live video"
                >
                  Live video
                </button>
              </div>
            </div>
            <p className="mt-2 text-sm text-gray-300">
              {effectiveGroupIds.length} group{effectiveGroupIds.length === 1 ? "" : "s"}
              {Object.keys(manualGroupOverrides).length > 0 ? " (edited by hand)" : ""}
              {shotGroupMode === "quadtree" && quadtreeBaseRadiusPx > 0
                ? ` · grouped within ${(quadtreeGroupInches * quadtreeRadiusScale).toFixed(1)} in`
                : ""}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-gray-300">
              <span>
                Avg confidence:{" "}
                <span
                  className={
                    confidenceSummary.avgConfidence >= 60
                      ? "font-semibold text-emerald-300"
                      : confidenceSummary.avgConfidence >= 35
                        ? "font-semibold text-amber-300"
                        : "font-semibold text-rose-300"
                  }
                >
                  {confidenceSummary.avgConfidence.toFixed(0)}%
                </span>
                <span className="text-gray-500"> ({confidenceSummary.keptCount} shots)</span>
              </span>
              <label className="flex items-center gap-1.5">
                <span className="flex items-center gap-1 text-gray-400">
                  Noise filter ≥
                  <InfoPopover title="Noise filter" tipId="tip-noise-filter" placement="top">
                    Hides shots below this confidence (they drop to dimmed strays, not deleted). Raise it to clear out
                    false positives; lower it if real hits disappear.
                  </InfoPopover>
                </span>
                <input
                  type="range"
                  min={0}
                  max={90}
                  step={5}
                  value={minShotConfidence}
                  onChange={(event) => setMinShotConfidence(Number(event.target.value))}
                  className="w-28"
                  aria-label="Minimum shot confidence"
                />
                <span className="tabular-nums">{minShotConfidence}%</span>
              </label>
              {confidenceSummary.filteredCount > 0 ? (
                <span className="text-rose-300/90">
                  {confidenceSummary.filteredCount} filtered as noise (shown as strays)
                </span>
              ) : null}
              <span className="text-gray-500">· bigger ring around a shot = louder bang</span>
            </div>
            <div className="relative mx-auto mt-2 w-fit max-w-full">
              <canvas
                ref={shotGroupMapCanvasRef}
                onPointerDown={onMapPointerDown}
                onPointerMove={onMapPointerMove}
                onPointerUp={manualEditMode ? handleMapPointerUp : undefined}
                onPointerCancel={manualEditMode ? handleMapPointerUp : undefined}
                onPointerLeave={onMapPointerLeave}
                onContextMenu={(event) => event.preventDefault()}
                className={`mx-auto block h-auto max-h-[55vh] w-auto max-w-full rounded-md border border-gray-700 bg-black ${
                  draggingShotId
                    ? "cursor-grabbing touch-none"
                    : manualMarkMode || manualEditMode || manualRemoveMode || aimPointGroupId !== null
                      ? "cursor-crosshair touch-none"
                      : hoverShotInfo
                        ? "cursor-pointer"
                        : ""
                }`}
              />

              {/* Aim-point placement hint. */}
              {aimPointGroupId !== null ? (
                <div className="absolute inset-x-2 top-2 z-30 flex items-center justify-between gap-2 rounded-lg border border-sky-400/40 bg-sky-900/85 px-3 py-1.5 text-[11px] text-sky-100 backdrop-blur">
                  <span>Tap the target where {groupLabel(aimPointGroupId)} was aiming.</span>
                  <button
                    type="button"
                    onClick={() => setAimPointGroupId(null)}
                    className="rounded border border-sky-300/40 px-2 py-0.5 text-sky-100 transition hover:bg-sky-500/20"
                  >
                    Cancel
                  </button>
                </div>
              ) : null}

              {/* Shot-adjustment tools, overlaid on the video window: edit groups,
                  add a hit, or remove a hit (Map step only — not during the live Scan). */}
              <div
                className={`absolute right-2 top-2 z-20 flex flex-col gap-1.5 ${
                  currentStep === STEP_MAP ? "" : "hidden"
                }`}
              >
                <button
                  type="button"
                  onClick={() => {
                    setManualEditMode((value) => !value);
                    setManualMarkMode(false);
                    setManualRemoveMode(false);
                    setAimPointGroupId(null);
                    setManualSelectedIds([]);
                    setManualSelectionRect(null);
                  }}
                  aria-pressed={manualEditMode}
                  title={manualEditMode ? "Done editing groups" : "Edit groups / stray"}
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium backdrop-blur transition ${
                    manualEditMode
                      ? "border-emerald-400/70 bg-emerald-500/25 text-emerald-100"
                      : "border-white/15 bg-neutral-900/80 text-gray-200 hover:bg-neutral-800"
                  }`}
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 20h9" />
                    <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                  </svg>
                  {manualEditMode ? "Done" : "Edit"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setManualMarkMode((value) => !value);
                    setManualEditMode(false);
                    setManualRemoveMode(false);
                    setAimPointGroupId(null);
                  }}
                  aria-pressed={manualMarkMode}
                  title={manualMarkMode ? "Done adding hits" : "Add a hit (tap the map)"}
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium backdrop-blur transition ${
                    manualMarkMode
                      ? "border-amber-300/70 bg-amber-500/25 text-amber-100"
                      : "border-white/15 bg-neutral-900/80 text-gray-200 hover:bg-neutral-800"
                  }`}
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="9" />
                    <path d="M12 8v8M8 12h8" />
                  </svg>
                  {manualMarkMode ? "Done" : "Add hit"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setManualRemoveMode((value) => !value);
                    setManualEditMode(false);
                    setManualMarkMode(false);
                    setAimPointGroupId(null);
                  }}
                  aria-pressed={manualRemoveMode}
                  title={manualRemoveMode ? "Done removing hits" : "Remove a hit (tap a shot to delete)"}
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium backdrop-blur transition ${
                    manualRemoveMode
                      ? "border-rose-400/70 bg-rose-500/25 text-rose-100"
                      : "border-white/15 bg-neutral-900/80 text-gray-200 hover:bg-neutral-800"
                  }`}
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v5M14 11v5" />
                  </svg>
                  {manualRemoveMode ? "Done" : "Delete"}
                </button>
              </div>

              {/* Mode hint for add/remove/edit. */}
              {manualMarkMode || manualRemoveMode || manualEditMode ? (
                <div className="absolute left-2 top-2 z-20 rounded-md border border-white/10 bg-neutral-900/85 px-2 py-1 text-[11px] text-gray-200 backdrop-blur">
                  {manualMarkMode
                    ? "Tap the map to add a hit"
                    : manualRemoveMode
                      ? "Tap a hit to remove it"
                      : draggingShotId
                        ? "Drag to reposition — release to drop"
                        : "Long-press a shot to drag it · drag empty space to select"}
                </div>
              ) : null}

              {(() => {
                const info = pinnedShotInfo ?? hoverShotInfo;
                if (!info) return null;
                const shot = info.shot;
                const pinned = pinnedShotInfo !== null;
                const groupId = effectiveGroupByShotId[shot.id];
                const conf = shotConfidencePct(shot);
                const sizeLabel =
                  shot.estimatedDiameterInches !== null
                    ? `⌀ ${formatLinearFromInches(shot.estimatedDiameterInches, 2)}`
                    : `⌀ ${shot.estimatedDiameterPx.toFixed(0)} px`;
                // Full per-shot detail rows (shown when pinned) — everything known.
                const detailRows = buildShotDetailRows(shot, {
                  groupId,
                  groupLabel,
                  conf,
                  formatLinearFromInches,
                });
                return (
                  <div
                    className={`absolute z-30 rounded-lg border border-gray-600 bg-neutral-900/95 p-2 text-[11px] leading-snug text-gray-200 shadow-xl shadow-black/50 ${
                      pinned ? "w-64 max-h-80 overflow-y-auto" : "w-44 pointer-events-none"
                    }`}
                    style={{
                      left: info.left,
                      top: info.top,
                      transform: "translate(-50%, calc(-100% - 10px))",
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-white">Shot S{shot.shotNumber}</span>
                      {pinned ? (
                        <button
                          type="button"
                          onClick={() => setPinnedShotInfo(null)}
                          aria-label="Close"
                          className="-mr-0.5 -mt-0.5 flex h-4 w-4 items-center justify-center rounded text-gray-400 transition hover:bg-neutral-700 hover:text-white"
                        >
                          ×
                        </button>
                      ) : null}
                    </div>
                    {pinned ? (
                      <dl className="mt-1 divide-y divide-white/5">
                        {detailRows.map((row) => (
                          <div key={row.label} className="flex items-start justify-between gap-3 py-[3px]">
                            <dt className="shrink-0 text-gray-400">{row.label}</dt>
                            <dd className={`text-right ${row.emphasis ? "font-medium text-white" : "text-gray-200"}`}>
                              {row.value}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    ) : (
                      <>
                        <p className="mt-0.5">
                          <span className={groupId ? "text-gray-200" : "text-gray-400"}>
                            {groupId ? groupLabel(groupId) : shot.persistent === false ? "Transient" : "Stray"}
                          </span>{" "}
                          · {conf.toFixed(0)}% conf
                        </p>
                        <p className="text-gray-300">
                          t={shot.videoTimeSec.toFixed(2)}s · {sizeLabel}
                        </p>
                        <p className="text-gray-400">
                          ({shot.centerX.toFixed(0)}, {shot.centerY.toFixed(0)}) px
                          {shot.audioDecibelDbfs !== null ? ` · ${shot.audioDecibelDbfs.toFixed(0)} dBFS` : ""}
                        </p>
                        <p className="mt-0.5 text-[10px] text-gray-500">Click for full details</p>
                      </>
                    )}
                  </div>
                );
              })()}
            </div>
            </div>
            {/* End map + tools wrapper. */}

            {/* Combined controls below the image (out of the frame). */}
            {/* Edit-group palette — shown after clicking the Edit tool (Map step). */}
            {currentStep === STEP_MAP && manualEditMode ? (
              <div className="mt-2 flex flex-wrap items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/5 p-2 text-[11px]">
                <span className="px-1 text-gray-300">
                  {manualSelectedIds.length > 0
                    ? `${manualSelectedIds.length} selected → assign to:`
                    : "Tap/box shots on the map, then assign to:"}
                </span>
                {effectiveGroupIds.map((groupId) => (
                  <button
                    key={groupId}
                    type="button"
                    onClick={() => chooseManualGroup(groupId)}
                    title={groupLabel(groupId)}
                    className={`flex h-7 items-center gap-1 rounded-full border px-2 transition ${
                      manualSelectedIds.length === 0 && activeManualGroup === groupId
                        ? "border-white bg-neutral-800 text-white"
                        : "border-gray-700 text-gray-200 hover:bg-neutral-800"
                    }`}
                  >
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: effectiveGroupColorById[groupId] ?? clusterColorForId(groupId) }}
                    />
                    {groupShortLabel(groupId)}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => chooseManualGroup(nextManualGroupId())}
                  title="New group"
                  className={`flex h-7 items-center rounded-full border px-2 transition ${
                    manualSelectedIds.length === 0 && !effectiveGroupIds.includes(activeManualGroup) && activeManualGroup >= 1
                      ? "border-white bg-neutral-800 text-white"
                      : "border-sky-500/40 text-sky-100 hover:bg-sky-500/10"
                  }`}
                >
                  + New
                </button>
                <button
                  type="button"
                  onClick={() => chooseManualGroup(0)}
                  title="Ungroup (stray)"
                  className={`flex h-7 items-center rounded-full border px-2 transition ${
                    manualSelectedIds.length === 0 && activeManualGroup === 0
                      ? "border-white bg-neutral-800 text-white"
                      : "border-gray-600 text-gray-300 hover:bg-neutral-800"
                  }`}
                >
                  Stray
                </button>
                <button
                  type="button"
                  onClick={() => chooseManualGroup(FALSE_POSITIVE_OVERRIDE)}
                  title="Mark as false positive — removes it from the map and counts (undo with Reset)"
                  className={`flex h-7 items-center gap-1 rounded-full border px-2 transition ${
                    manualSelectedIds.length === 0 && activeManualGroup === FALSE_POSITIVE_OVERRIDE
                      ? "border-rose-300 bg-rose-500/20 text-rose-100"
                      : "border-rose-500/40 text-rose-200 hover:bg-rose-500/10"
                  }`}
                >
                  ✕ False positive
                </button>
                {manualSelectedIds.length > 0 ? (
                  <button
                    type="button"
                    onClick={() => setManualSelectedIds([])}
                    className="flex h-7 items-center rounded-full border border-gray-600 px-2 text-gray-300 transition hover:bg-neutral-800"
                  >
                    Clear ({manualSelectedIds.length})
                  </button>
                ) : null}
                {Object.keys(manualGroupOverrides).length > 0 ? (
                  <button
                    type="button"
                    onClick={resetManualGroupOverrides}
                    className="flex h-7 items-center rounded-full border border-rose-500/40 px-2 text-rose-200 transition hover:bg-rose-500/10"
                  >
                    Reset ({Object.keys(manualGroupOverrides).length})
                  </button>
                ) : null}
              </div>
            ) : null}

            {/* Reveal window scrubber (Map step). */}
            {currentStep === STEP_MAP && groupTimeline.hasData ? (
              <div className="mt-2 text-[11px] text-gray-300">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-semibold uppercase tracking-wide text-gray-400">Reveal</span>
                  <span className="tabular-nums text-gray-400">
                    {timelineStart.toFixed(2)}–{timelineCursor.toFixed(2)}s / {groupTimeline.tMax.toFixed(2)}s ·{" "}
                    {timelineUpToCount} shot
                    {timelineUpToCount === 1 ? "" : "s"} · {timelineGroupsActive} group
                    {timelineGroupsActive === 1 ? "" : "s"}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setTimelineCursorSec(null);
                      setTimelineStartSec(null);
                    }}
                    className="rounded border border-gray-600 px-2 py-0.5 text-gray-300 transition hover:bg-neutral-800"
                  >
                    Show all
                  </button>
                </div>
                {/* Full-width single track, two thumbs (amber = start, sky = end);
                    drag the middle to slide the whole window at a fixed duration. */}
                <div ref={revealTrackRef} className="relative mt-2 h-4 w-full">
                  <div className="pointer-events-none absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-gray-600" />
                  <div
                    className="absolute top-0 bottom-0 cursor-grab touch-none active:cursor-grabbing"
                    style={{ left: `${timelineStartPct}%`, right: `${100 - timelineEndPct}%` }}
                    onPointerDown={(event) => {
                      const track = revealTrackRef.current;
                      if (!track) return;
                      revealDragRef.current = {
                        startX: event.clientX,
                        s0: timelineStart,
                        dur: timelineCursor - timelineStart,
                        width: track.getBoundingClientRect().width,
                      };
                      event.currentTarget.setPointerCapture(event.pointerId);
                    }}
                    onPointerMove={(event) => {
                      const drag = revealDragRef.current;
                      if (!drag || drag.width <= 0) return;
                      const deltaSec = ((event.clientX - drag.startX) / drag.width) * timelineSpanSec;
                      const ns = Math.max(
                        groupTimeline.tMin,
                        Math.min(drag.s0 + deltaSec, groupTimeline.tMax - drag.dur),
                      );
                      const ne = ns + drag.dur;
                      setTimelineStartSec(ns <= groupTimeline.tMin + 1e-6 ? null : ns);
                      setTimelineCursorSec(ne >= groupTimeline.tMax - 1e-6 ? null : ne);
                    }}
                    onPointerUp={(event) => {
                      revealDragRef.current = null;
                      event.currentTarget.releasePointerCapture(event.pointerId);
                    }}
                    aria-label="Drag to move the reveal window"
                  >
                    <div className="pointer-events-none absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-sky-400/70" />
                  </div>
                  <input
                    type="range"
                    min={groupTimeline.tMin}
                    max={groupTimeline.tMax}
                    step={Math.max(0.01, (groupTimeline.tMax - groupTimeline.tMin) / 200)}
                    value={timelineStart}
                    onChange={(event) => {
                      // Start can't pass the end cursor.
                      const next = Math.min(Number(event.target.value), timelineCursor);
                      setTimelineStartSec(next <= groupTimeline.tMin ? null : next);
                    }}
                    aria-label="Reveal window start time"
                    className="range-dual range-dual-start absolute inset-0 h-full w-full"
                  />
                  <input
                    type="range"
                    min={groupTimeline.tMin}
                    max={groupTimeline.tMax}
                    step={Math.max(0.01, (groupTimeline.tMax - groupTimeline.tMin) / 200)}
                    value={timelineCursor}
                    onChange={(event) => {
                      // End can't drop below the start.
                      const next = Math.max(Number(event.target.value), timelineStart);
                      setTimelineCursorSec(next >= groupTimeline.tMax ? null : next);
                    }}
                    aria-label="Reveal window end time"
                    className="range-dual absolute inset-0 h-full w-full"
                  />
                </div>
              </div>
            ) : null}

            {/* Video replay controls (Map step) — the <video> stays mounted below. */}
            <div
              className={`mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-300 ${
                currentStep === STEP_MAP ? "" : "hidden"
              }`}
            >
              <span className="font-semibold uppercase tracking-wide text-gray-400">Replay</span>
              <button
                type="button"
                onClick={toggleMapReplay}
                title={mapReplayPlaying ? "Pause" : "Replay shots over time"}
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-sky-400/40 bg-sky-500/15 text-sky-100 transition hover:bg-sky-500/25"
              >
                {mapReplayPlaying ? "❚❚" : "▶"}
              </button>
              <input
                type="range"
                min={0}
                max={Math.max(0.1, mapReplayDurationSec)}
                step={0.05}
                value={Math.min(mapReplayTimeSec, mapReplayDurationSec || 0)}
                onChange={(event) => scrubMapReplay(Number(event.target.value))}
                disabled={mapReplayDurationSec <= 0}
                aria-label="Replay position"
                className="min-w-0 flex-1 disabled:opacity-40"
              />
              <span className="shrink-0 tabular-nums text-gray-400">
                {Math.floor(Math.max(0, mapReplayTimeSec) / 60)}:
                {String(Math.floor(Math.max(0, mapReplayTimeSec) % 60)).padStart(2, "0")} /{" "}
                {Math.floor(Math.max(0, mapReplayDurationSec) / 60)}:
                {String(Math.floor(Math.max(0, mapReplayDurationSec) % 60)).padStart(2, "0")}
              </span>
              <label className="flex shrink-0 items-center gap-1" title="Replay volume">
                <span aria-hidden="true">🔉</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={playbackVolume}
                  onChange={(event) => changePlaybackVolume(Number(event.target.value))}
                  aria-label="Replay volume"
                  className="w-16"
                />
              </label>
              {mapReplayActive ? (
                <button
                  type="button"
                  onClick={exitMapReplay}
                  title="Exit replay"
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-gray-600 text-gray-200 transition hover:bg-neutral-800"
                >
                  ×
                </button>
              ) : null}
            </div>

            <video
              ref={mapReplayVideoRef}
              playsInline
              preload="metadata"
              className="hidden"
              onLoadedMetadata={(event) => setMapReplayDurationSec(event.currentTarget.duration || 0)}
              onTimeUpdate={(event) => setMapReplayTimeSec(event.currentTarget.currentTime || 0)}
              onPlay={() => setMapReplayPlaying(true)}
              onPause={() => setMapReplayPlaying(false)}
              onEnded={() => setMapReplayPlaying(false)}
            />

            {/* Live per-group statistics (Review step). */}
            {currentStep === STEP_REVIEW && effectiveGroupStats.groups.length > 0 ? (
              <div className="mt-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-sm font-semibold uppercase tracking-wide text-gray-200">Group Statistics</p>
                  <p className="text-xs text-gray-400">
                    {effectiveGroupStats.groups.length} group{effectiveGroupStats.groups.length === 1 ? "" : "s"}
                    {effectiveGroupStats.strayCount > 0 ? ` · ${effectiveGroupStats.strayCount} stray` : ""}
                    {effectiveGroupStats.transientCount > 0 ? ` · ${effectiveGroupStats.transientCount} transient` : ""}
                    {falsePositiveCount > 0 ? ` · ${falsePositiveCount} false positive${falsePositiveCount === 1 ? "" : "s"}` : ""}
                    {pixelsPerInch > 0 ? "" : " · calibrate for inch units"}
                  </p>
                </div>
                <div className="mt-2 grid max-h-[26vh] grid-cols-1 gap-1.5 overflow-y-auto pr-1 sm:grid-cols-2">
                  {effectiveGroupStats.groups.map((group) => {
                    const color = effectiveGroupColorById[group.groupId] ?? clusterColorForId(group.groupId);
                    const spread =
                      pixelsPerInch > 0
                        ? formatLinearFromInches(group.extremeSpreadPx / pixelsPerInch, 2)
                        : `${group.extremeSpreadPx.toFixed(0)} px`;
                    const meanRadius =
                      pixelsPerInch > 0
                        ? formatLinearFromInches(group.meanRadialPx / pixelsPerInch, 2)
                        : `${group.meanRadialPx.toFixed(0)} px`;
                    // Aim-point offset: distance + compass-style bearing (0° = up,
                    // clockwise) from the aim point to the group centroid.
                    const aim = groupAimPoints[group.groupId] ?? null;
                    const settingAim = aimPointGroupId === group.groupId;
                    let aimOffset: { dist: string; deg: number } | null = null;
                    if (aim) {
                      const dx = group.centroidX - aim.x;
                      const dy = group.centroidY - aim.y;
                      const distPx = Math.hypot(dx, dy);
                      const dist =
                        pixelsPerInch > 0 ? formatLinearFromInches(distPx / pixelsPerInch, 2) : `${distPx.toFixed(0)} px`;
                      const deg = (Math.atan2(dx, -dy) * (180 / Math.PI) + 360) % 360;
                      aimOffset = { dist, deg };
                    }
                    return (
                      <div
                        key={group.groupId}
                        className="rounded-md border px-2.5 py-1.5 text-xs text-gray-100"
                        style={{ borderColor: hexToRgba(color, 0.5), backgroundColor: hexToRgba(color, 0.08) }}
                      >
                        <p className="flex items-center gap-2 text-sm">
                          <span className="inline-block h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: color }} />
                          <input
                            type="text"
                            value={groupNames[group.groupId] ?? ""}
                            onChange={(event) =>
                              setGroupNames((prev) => ({ ...prev, [group.groupId]: event.target.value }))
                            }
                            placeholder={`Group ${groupDisplayNumberById[group.groupId] ?? group.groupId}`}
                            aria-label={`Rename ${groupLabel(group.groupId)}`}
                            title="Click to rename this group"
                            className="w-28 shrink rounded border border-transparent bg-transparent px-1 text-sm font-semibold text-gray-100 outline-none hover:border-gray-600 focus:border-sky-400 focus:bg-neutral-900"
                          />
                          <span className="shrink-0 text-xs text-gray-400">
                            · {group.count} shot{group.count === 1 ? "" : "s"}
                          </span>
                          <span
                            className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                              group.meanConfidence >= 60
                                ? "bg-emerald-500/20 text-emerald-200"
                                : group.meanConfidence >= 35
                                  ? "bg-amber-500/20 text-amber-200"
                                  : "bg-rose-500/20 text-rose-200"
                            }`}
                          >
                            {group.meanConfidence.toFixed(0)}% conf
                          </span>
                        </p>
                        <p className="mt-0.5 text-gray-200">
                          spread <span className="font-semibold text-white">{spread}</span> · mean radius {meanRadius}
                          {group.meanDiameterInches !== null
                            ? ` · ⌀ ${formatLinearFromInches(group.meanDiameterInches, 2)}`
                            : ""}
                        </p>
                        <p className="mt-0.5 text-[11px] text-gray-400">
                          center ({group.centroidX.toFixed(0)}, {group.centroidY.toFixed(0)}) px
                          {group.timeSpanSec > 0 ? ` · span ${group.timeSpanSec.toFixed(2)}s` : ""}
                        </p>
                        {aimOffset ? (
                          <p className="mt-0.5 text-gray-200">
                            aim offset <span className="font-semibold text-white">{aimOffset.dist}</span> @{" "}
                            {aimOffset.deg.toFixed(0)}°
                            <span className="text-gray-400"> from aim point</span>
                          </p>
                        ) : null}
                        <div className="mt-1 flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => {
                              setManualEditMode(false);
                              setManualMarkMode(false);
                              setManualRemoveMode(false);
                              setAimPointGroupId((current) => (current === group.groupId ? null : group.groupId));
                            }}
                            className={`rounded border px-2 py-0.5 text-[11px] transition ${
                              settingAim
                                ? "border-sky-300/60 bg-sky-500/20 text-sky-100"
                                : "border-gray-600 text-gray-200 hover:bg-neutral-800"
                            }`}
                          >
                            {settingAim ? "Tap the map…" : aim ? "Move aim point" : "Set aim point"}
                          </button>
                          {aim ? (
                            <button
                              type="button"
                              onClick={() =>
                                setGroupAimPoints((current) => {
                                  const next = { ...current };
                                  delete next[group.groupId];
                                  return next;
                                })
                              }
                              className="rounded border border-gray-700 px-2 py-0.5 text-[11px] text-gray-400 transition hover:bg-neutral-800"
                            >
                              Clear
                            </button>
                          ) : null}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}

            {/* Sortable, color-coded table of every detected shot (Review step). */}
            {currentStep === STEP_REVIEW && mapShots.length > 0 ? (
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => setShowShotTable((value) => !value)}
                  aria-expanded={showShotTable}
                  className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-gray-200 transition hover:text-white"
                >
                  <span className="text-gray-400">{showShotTable ? "▾" : "▸"}</span>
                  Shots table
                  <span className="text-xs font-normal normal-case tracking-normal text-gray-400">
                    ({mapShots.length})
                  </span>
                </button>
                {showShotTable ? (
                  <div className="mt-2 max-h-[40vh] overflow-auto rounded-md border border-gray-700">
                    <table className="w-full border-collapse text-left text-[11px]">
                      <thead className="sticky top-0 bg-neutral-900/95 text-gray-400">
                        <tr>
                          {(
                            [
                              ["num", "#"],
                              ["group", "Group"],
                              ["conf", "Conf"],
                              ["size", "Size"],
                              ["time", "Time"],
                              ["loud", "dBFS"],
                              ["persist", "Stays"],
                            ] as [ShotTableSortKey, string][]
                          ).map(([key, label]) => (
                            <th key={key} className="px-2 py-1.5 font-medium">
                              <button
                                type="button"
                                onClick={() =>
                                  setShotTableSort((prev) =>
                                    prev.key === key
                                      ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
                                      : { key, dir: "asc" },
                                  )
                                }
                                className="flex items-center gap-1 transition hover:text-white"
                              >
                                {label}
                                {shotTableSort.key === key ? (
                                  <span className="text-sky-400">{shotTableSort.dir === "asc" ? "▲" : "▼"}</span>
                                ) : null}
                              </button>
                            </th>
                          ))}
                          <th className="px-2 py-1.5 font-medium">Bang</th>
                        </tr>
                      </thead>
                      <tbody>
                        {shotTableRows.map((row) => {
                          const selected = manualSelectedIds.includes(row.shot.id);
                          const color = row.groupId
                            ? effectiveGroupColorById[row.groupId] ?? clusterColorForId(row.groupId)
                            : null;
                          // Highlight the rows currently shown on the map (inside the
                          // reveal window); dim the rest while a window is active.
                          const inWindow =
                            revealWindowActive && row.time >= timelineStart && row.time <= timelineCursor;
                          return (
                            <tr
                              key={row.shot.id}
                              onClick={() => setManualSelectedIds([row.shot.id])}
                              className={`cursor-pointer border-t border-gray-800 transition hover:bg-neutral-800/60 ${
                                selected
                                  ? "bg-sky-500/15"
                                  : inWindow
                                    ? "bg-amber-500/10"
                                    : revealWindowActive
                                      ? "opacity-40"
                                      : ""
                              }`}
                            >
                              <td className="px-2 py-1 tabular-nums text-gray-300">S{row.shot.shotNumber}</td>
                              <td className="px-2 py-1">
                                {row.groupId ? (
                                  <span
                                    className="inline-flex items-center gap-1"
                                    style={{ color: color ?? undefined }}
                                  >
                                    <span
                                      className="inline-block h-2 w-2 rounded-full"
                                      style={{ backgroundColor: color ?? "transparent" }}
                                    />
                                    {row.category}
                                  </span>
                                ) : (
                                  <span className={row.shot.persistent === false ? "text-amber-400" : "text-gray-400"}>
                                    {row.category}
                                  </span>
                                )}
                              </td>
                              <td className={`px-2 py-1 tabular-nums ${confColorClass(row.conf)}`}>
                                {row.conf.toFixed(0)}%
                              </td>
                              <td className="px-2 py-1 tabular-nums text-gray-300">
                                {row.sizeIn !== null ? formatLinearFromInches(row.sizeIn, 2) : `${row.sizePx.toFixed(0)} px`}
                              </td>
                              <td className="px-2 py-1 tabular-nums text-gray-300">{row.time.toFixed(2)}s</td>
                              <td className={`px-2 py-1 tabular-nums ${loudColorClass(row.loud)}`}>
                                {row.loud !== null ? row.loud.toFixed(0) : "—"}
                              </td>
                              <td className={`px-2 py-1 tabular-nums ${persistColorClass(row.persist)}`}>
                                {row.persist !== null ? `${Math.round(row.persist * 100)}%` : "—"}
                              </td>
                              <td className="px-2 py-1">
                                <span className={row.loudBang ? "text-emerald-400" : "text-gray-500"}>
                                  {row.loudBang ? "✓" : "—"}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </div>
            ) : null}

            {currentStep === STEP_MAP && shotGroupMode === "quadtree" ? (
              <div className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-gray-300">
                <label className="flex items-center gap-2">
                  <span className="text-gray-400">Group radius</span>
                  <input
                    type="range"
                    min="0.2"
                    max="3"
                    step="0.05"
                    value={quadtreeRadiusScale}
                    onChange={(event) => setQuadtreeRadiusScale(Number(event.target.value))}
                  />
                  <span className="tabular-nums">
                    {quadtreeBaseRadiusPx > 0
                      ? `${(quadtreeGroupInches * quadtreeRadiusScale).toFixed(1)} in`
                      : `${quadtreeRadiusScale.toFixed(2)}×`}
                  </span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={showQuadtreeCells}
                    onChange={(event) => setShowQuadtreeCells(event.target.checked)}
                  />
                  Show quadtree cells
                </label>
              </div>
            ) : null}
          </div>

          {/* Detailed detector views (Review step). The container stays mounted so
              the persisted snapshot canvases survive across steps. */}
          {currentStep === STEP_REVIEW ? (
            <button
              type="button"
              onClick={() => setShowDetailedViews((value) => !value)}
              className="mt-3 self-start rounded-md border border-gray-600 px-3 py-1.5 text-xs text-gray-200 transition hover:bg-neutral-800"
              aria-expanded={showDetailedViews}
            >
              {showDetailedViews ? "Hide detailed views" : `Show detailed views${isScanning ? " (scan running)" : ""}`}
            </button>
          ) : null}
          <div className={`mt-4 space-y-3 ${showDetailedViews && currentStep === STEP_REVIEW ? "" : "hidden"}`}>
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-gray-400">Analyzed target view</p>
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-gray-400">Play speed</span>
                  {[0.5, 1, 2, 4].map((speed) => (
                    <button
                      key={speed}
                      type="button"
                      onClick={() => setPlaybackSpeed(speed)}
                      className={`min-h-8 rounded-md border px-3 text-xs transition sm:min-h-0 sm:px-2 sm:py-0.5 sm:text-[11px] ${
                        playbackSpeed === speed
                          ? "border-sky-400 bg-sky-500/20 text-sky-100"
                          : "border-gray-600 text-gray-300 hover:bg-neutral-800"
                      }`}
                      aria-pressed={playbackSpeed === speed}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
              </div>
              <canvas ref={processingCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
            </div>
            <div className="grid max-w-3xl grid-cols-1 gap-2 xs:grid-cols-2 sm:gap-3">
              <div className="space-y-1">
                <p className="text-xs text-gray-400">
                  {useChangeDetector ? "Change candidates (box = blob, color = verdict)" : "Probe patch window (DBSCAN overlays)"}
                </p>
                <canvas ref={processedPatchCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
              </div>
              <div className="space-y-1">
                <p className="text-xs text-gray-400">Contour regions view (DBSCAN groups)</p>
                <canvas ref={processedContourCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
              </div>
              <div className="space-y-1">
                <p className="text-xs text-gray-400">
                  {useChangeDetector ? "Change mask — red = differs from baseline" : "Binary change mask window"}
                </p>
                <canvas ref={processedMaskCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
              </div>
              <div className="space-y-1">
                <p className="text-xs text-gray-400">Yellow-green top-hat mask window</p>
                <canvas ref={processedYellowGreenCanvasRef} className="w-full rounded-md border border-gray-700 bg-black" />
              </div>
            </div>
            {useChangeDetector && (
              <div className="mt-2 flex max-w-3xl flex-wrap gap-x-3 gap-y-1 text-[11px] text-gray-400">
                <span className="font-semibold text-gray-300">Candidate verdicts:</span>
                <span><span className="text-emerald-400">■</span> accepted</span>
                <span><span className="text-sky-400">■</span> too small</span>
                <span><span className="text-purple-400">■</span> too large</span>
                <span><span className="text-amber-400">■</span> ragged (not solid)</span>
                <span><span className="text-orange-400">■</span> streak (elongated)</span>
                <span><span className="text-red-400">■</span> wrong size for caliber</span>
                <span className="text-gray-500">· circles = persistence tracker (green = logged)</span>
              </div>
            )}
          </div>

          <details className="mt-5 rounded-md border border-gray-700 bg-black/35 p-3">
            <summary className="cursor-pointer text-base font-semibold text-white sm:text-lg">Resulting Data</summary>
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
            <p className="text-sm text-gray-300">
              Analysis video cache:{" "}
              {analysisVideoCacheStatus === "cached"
                ? "session ready"
                : analysisVideoCacheStatus === "too_large"
                  ? "video too large for session storage (fallback uses focused re-analysis)"
                  : analysisVideoCacheStatus === "unavailable"
                    ? "unavailable (fallback uses focused re-analysis)"
                    : "pending"}
            </p>
            <p className="text-sm text-gray-300">Log entries: {logEntries.length}</p>
            <p className="text-sm text-gray-300">Shot changes: {shotLogEntries.length}</p>
            <p className="text-sm text-gray-300">
              Probe histogram delta (vs reference): {lastHistogramDeltaPctRef.current.toFixed(2)}%
            </p>
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
                <p className="text-xs uppercase tracking-wide text-gray-300">Shot Clusters (DBSCAN: X, Y, Time)</p>
                <p className="mt-1 text-[11px] text-gray-400">
                  Initial clusters: {shotClustering.selectedK} | Final clusters: {shotClustering.finalK}
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
                          DB{cluster.clusterId} | shots={cluster.count} | centroid=({cluster.centroidX.toFixed(1)},{" "}
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

            </div>
          </details>
        </section>

        {/* Save: persist this session's bullet annotations to the user's account. */}
        <section
          className={`flex flex-col rounded-xl border border-gray-700 bg-neutral-950 p-3 sm:p-6 ${
            currentStep === STEP_SAVE ? "animate-stepIn" : "hidden"
          }`}
        >
          <div>
            <h2 className="text-base font-semibold text-white sm:text-lg">8. Save to account</h2>
            <p className="mt-1 text-xs text-gray-400">
              Save this session&apos;s shots to your account. Saved bullet annotations build a labeled dataset used to
              improve detection over time.
            </p>
          </div>

          <div className="mt-4 rounded-md border border-gray-700 bg-black p-3 sm:p-4">
            <p className="text-sm text-gray-200">
              {confidenceSummary.keptCount} shot{confidenceSummary.keptCount === 1 ? "" : "s"} in this session
              {effectiveGroupIds.length > 0
                ? ` · ${effectiveGroupIds.length} group${effectiveGroupIds.length === 1 ? "" : "s"}`
                : ""}
              {pixelsPerInch > 0 ? " · calibrated" : " · not calibrated"}
            </p>

            {membership && !membership.configured ? (
              <p className="mt-3 text-xs text-amber-100">
                Accounts aren&apos;t configured on this deployment, so saving is disabled.
              </p>
            ) : membership && !membership.signedIn ? (
              <div className="mt-3 space-y-2">
                <p className="text-xs text-gray-300">Sign in to save this session to your account.</p>
                <a
                  href="/login"
                  className="inline-block rounded-md border border-sky-400/45 bg-sky-500/15 px-3 py-1.5 text-sm font-medium text-sky-100 transition hover:bg-sky-500/25"
                >
                  Sign in
                </a>
              </div>
            ) : (
              <div className="mt-3 space-y-2">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                  <input
                    type="text"
                    value={saveSessionName}
                    onChange={(event) => setSaveSessionName(event.target.value)}
                    placeholder={selectedVideoName ?? "Session name (optional)"}
                    className="min-w-0 flex-1 rounded-md border border-gray-700 bg-neutral-950 px-3 py-1.5 text-sm outline-none ring-gray-400/40 focus:ring-1"
                  />
                  <button
                    type="button"
                    onClick={() => void saveAnnotationsToAccount()}
                    disabled={savingSession || confidenceSummary.keptCount === 0 || !membership?.signedIn}
                    className="rounded-md border border-emerald-400/50 bg-emerald-500/15 px-3 py-1.5 text-sm font-medium text-emerald-100 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {savingSession ? "Saving…" : "Save to account"}
                  </button>
                </div>
                {membership?.email ? (
                  <p className="text-[11px] text-gray-500">Signed in as {membership.email}</p>
                ) : null}
              </div>
            )}
            {saveStatus ? <p className="mt-2 text-xs text-emerald-200">{saveStatus}</p> : null}
            <a
              href="/shots"
              className="mt-3 inline-block rounded-md border border-sky-400/40 px-3 py-1.5 text-sm text-sky-100 transition hover:bg-sky-500/10"
            >
              Open shot library — browse &amp; compare saved shots →
            </a>
          </div>
        </section>

        <section
          ref={audioSectionRef}
          className="hidden"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-white sm:text-lg">6. Audio Capture</h2>
              <p className="mt-1 text-xs text-gray-400">
                Every shot is gated by an audio spike. This screen shows exactly what the audio pipeline captures —
                signal properties, the detection threshold, the RMS/dBFS timeline, acoustic signatures, and every
                detected spike.
              </p>
            </div>
            <span className="shrink-0 rounded-md border border-gray-700 px-2 py-1 text-[11px] text-gray-300">
              {spikeMetadata.length} spikes
            </span>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
            <div className="rounded-md border border-gray-700 bg-black p-2">
              <p className="text-gray-400">Howler</p>
              <p className="text-gray-100">{howlerReady ? "ready" : "loading"}</p>
            </div>
            <div className="rounded-md border border-gray-700 bg-black p-2">
              <p className="text-gray-400">Sprites</p>
              <p className="text-gray-100">
                {spritesReady ? "ready" : "pending"} ({Object.keys(audioSprites).length})
              </p>
            </div>
            <div className="rounded-md border border-gray-700 bg-black p-2">
              <p className="text-gray-400">Spikes</p>
              <p className="text-gray-100">{spikeMetadata.length}</p>
            </div>
            <div className="rounded-md border border-gray-700 bg-black p-2">
              <p className="text-gray-400">Signatures</p>
              <p className="text-gray-100">{audioSignatureCatalog.length}</p>
            </div>
          </div>

          <p className="mt-4 text-xs uppercase tracking-wide text-gray-300">Captured Signal</p>
          {audioCaptureInfo ? (
            <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-gray-300 sm:grid-cols-3">
              <p>
                Sample rate: <span className="text-gray-100">{audioCaptureInfo.sampleRate.toLocaleString()} Hz</span>
              </p>
              <p>
                Channels: <span className="text-gray-100">{audioCaptureInfo.channels}</span>
              </p>
              <p>
                Duration: <span className="text-gray-100">{audioCaptureInfo.durationSec.toFixed(2)} s</span>
              </p>
              <p>
                Total samples: <span className="text-gray-100">{audioCaptureInfo.totalSamples.toLocaleString()}</span>
              </p>
              <p>
                RMS frames: <span className="text-gray-100">{audioCaptureInfo.rmsSampleCount.toLocaleString()}</span>
              </p>
              <p>
                RMS hop: <span className="text-gray-100">{(audioCaptureInfo.rmsHopSec * 1000).toFixed(1)} ms</span>
              </p>
              <p>
                Mean level: <span className="text-gray-100">{audioCaptureInfo.meanDbfs.toFixed(1)} dBFS</span>
              </p>
              <p>
                Spike threshold: <span className="text-rose-200">{audioCaptureInfo.thresholdDbfs.toFixed(1)} dBFS</span>
              </p>
              <p>
                Peak / floor:{" "}
                <span className="text-gray-100">
                  {audioCaptureInfo.maxDbfs.toFixed(1)} / {audioCaptureInfo.minDbfs.toFixed(1)} dBFS
                </span>
              </p>
            </div>
          ) : audioCaptureError ? (
            <div className="mt-2 rounded border border-rose-700/60 bg-rose-950/40 p-2 text-[11px] text-rose-200">
              <p className="font-semibold text-rose-100">Audio pipeline failed</p>
              <p className="mt-1 break-words">{audioCaptureError}</p>
            </div>
          ) : (
            <p className="mt-2 text-[11px] text-gray-500">
              Run a scan to capture audio. Detected signal properties will appear here.
            </p>
          )}

          <p className="mt-4 text-xs uppercase tracking-wide text-gray-300">RMS / dBFS Timeline</p>
          <canvas
            ref={audioTimelineCanvasRef}
            width={760}
            height={140}
            className="mt-2 w-full rounded-md border border-gray-700 bg-black"
          />
          <p className="mt-1 text-[10px] text-gray-500">
            <span className="text-emerald-300">emerald</span> = RMS level ·{" "}
            <span className="text-rose-300">red dashed</span> = spike threshold · <span className="text-sky-300">sky</span>{" "}
            = mean · <span className="text-amber-300">amber</span> = detected spikes &amp; windows
          </p>

          <p className="mt-4 text-xs uppercase tracking-wide text-gray-300">Detection Parameters</p>
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-gray-300 sm:grid-cols-3">
            <p>
              Analysis window: <span className="text-gray-100">{tweakSettings.audioWindowSize} samples</span>
            </p>
            <p>
              Threshold: <span className="text-gray-100">mean + {tweakSettings.audioSpikeStdDevMultiplier}σ</span>
            </p>
            <p>
              Min spike gap: <span className="text-gray-100">{tweakSettings.audioSpikeMinGapSec.toFixed(2)} s</span>
            </p>
            <p>
              Peak window: <span className="text-gray-100">±{tweakSettings.audioPeakWindowHalfSec.toFixed(2)} s</span>
            </p>
            <p>
              Energy offset: <span className="text-gray-100">{tweakSettings.audioEnergyOffsetDb} dB</span>
            </p>
            <p>
              Energy scale: <span className="text-gray-100">{tweakSettings.audioEnergyScaleDb} dB</span>
            </p>
            <p>
              Weights E/T/V:{" "}
              <span className="text-gray-100">
                {tweakSettings.audioWeightEnergy}/{tweakSettings.audioWeightTimeAlignment}/
                {tweakSettings.audioWeightVisual}
              </span>
            </p>
            <p>
              Visual floor: <span className="text-gray-100">{tweakSettings.audioVisualScoreFloor}</span>
            </p>
            <p>
              No-spike align: <span className="text-gray-100">{tweakSettings.audioNoSpikeAlignmentScore}</span>
            </p>
          </div>
          <button
            type="button"
            onClick={() => setIsSettingsModalOpen(true)}
            className="mt-2 text-[11px] text-sky-300 underline-offset-2 hover:underline"
          >
            Adjust in Settings →
          </button>

          {audioSignatureCatalog.length > 0 ? (
            <>
              <p className="mt-4 text-xs uppercase tracking-wide text-gray-300">
                Acoustic Signatures ({audioSignatureCatalog.length})
              </p>
              <div className="mt-2 max-h-32 space-y-1 overflow-auto rounded border border-gray-800 bg-neutral-900/40 p-2 text-[11px] text-gray-300">
                {audioSignatureCatalog.map((entry) => (
                  <p key={entry.signatureId}>
                    Sig {entry.signatureId} · key {entry.signatureKey} · spikes={entry.count} · peak=
                    {entry.meanPeakDbfs.toFixed(1)} dBFS · sub-peaks={entry.meanSubPeakCount.toFixed(1)} · spread=
                    {entry.meanSubPeakSpreadSec.toFixed(3)}s
                  </p>
                ))}
              </div>
            </>
          ) : null}

          <p className="mt-4 text-xs uppercase tracking-wide text-gray-300">Detected Spikes ({spikeMetadata.length})</p>
          {spikeMetadata.length > 0 ? (
            <div className="mt-2 max-h-80 space-y-1 overflow-auto">
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
                              className="mt-0.5 h-8 w-8 rounded border border-gray-600 text-center text-base leading-none text-gray-200 transition hover:bg-neutral-800 sm:h-5 sm:w-5 sm:text-xs sm:leading-4"
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
                                Window: {spike.windowStartSec.toFixed(2)}s - {spike.windowEndSec.toFixed(2)}s | Sig{" "}
                                {spike.signatureId} ({spike.signatureKey}) | sub-peaks={spike.subPeakTimesSec.length} |
                                sprite={spike.spriteStartMs.toFixed(0)}ms +{spike.spriteDurationMs.toFixed(0)}ms
                              </p>
                              {spike.subPeakTimesSec.length > 0 ? (
                                <p className="text-[10px] text-gray-500">
                                  sub-peaks @ {spike.subPeakTimesSec.map((t) => `${t.toFixed(3)}s`).join(", ")}
                                </p>
                              ) : null}
                            </div>
                          </div>
                          <div className="flex flex-col items-end gap-1">
                            <button
                              type="button"
                              onClick={() => playSpikeSprite(spike.id)}
                              disabled={isScanning}
                              className="min-h-8 rounded border border-sky-500/30 px-3 text-xs text-sky-100 hover:bg-sky-500/10 disabled:opacity-50 sm:min-h-0 sm:px-2 sm:py-0.5 sm:text-[11px]"
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
                              <div className="mt-1 space-y-0.5">
                                {spikeShotSummaryById[spike.id]?.shots.map((shot) => (
                                  <p key={shot.id} className="text-gray-400">
                                    S{shot.shotNumber} | t={shot.videoTimeSec.toFixed(3)}s
                                    {shot.audioDecibelDbfs === null ? "" : ` | ${shot.audioDecibelDbfs.toFixed(1)}dBFS`}
                                    {shot.audioCorrelationScorePct === null
                                      ? ""
                                      : ` | a-match=${shot.audioCorrelationScorePct.toFixed(0)}%`}
                                    {" | "}x={shot.centerX}, y={shot.centerY}
                                  </p>
                                ))}
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
          ) : (
            <p className="mt-2 text-[11px] text-gray-500">
              No spikes detected yet — run a scan on an uploaded video with audio.
            </p>
          )}
        </section>
            </div>
          </div>
        </div>

        {/* Wizard navigation: move between steps without scrolling. */}
        <nav
          aria-label="Workflow steps"
          className="z-30 -mx-3 shrink-0 border-t border-gray-800 bg-black/90 px-3 py-2 backdrop-blur sm:mx-0 sm:mb-2 sm:rounded-xl sm:border"
        >
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => goToStep(currentStep - 1)}
              disabled={currentStep === 0}
              className="min-h-9 rounded-md border border-gray-600 px-3 text-sm text-gray-200 transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              ← Back
            </button>
            <span className="text-center text-xs text-gray-300 sm:text-sm">
              Step {currentStep + 1} of {sectionSteps.length} · {sectionSteps[currentStep].label}
              {currentStep === 0 && !canLeaveSourceStep ? (
                <span className="ml-1 block text-[11px] text-amber-300/90 sm:inline">
                  {captureMode === "upload" ? "· loading video…" : "· start the camera to continue"}
                </span>
              ) : null}
            </span>
            <button
              type="button"
              onClick={() => goToStep(currentStep + 1)}
              disabled={currentStep === sectionSteps.length - 1 || (currentStep === 0 && !canLeaveSourceStep)}
              title={currentStep === 0 && !canLeaveSourceStep ? "Waiting for the video to finish loading…" : undefined}
              className="min-h-9 rounded-md border border-sky-400/45 px-3 text-sm text-sky-100 transition hover:bg-sky-500/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next →
            </button>
          </div>
          <div className="mt-2 flex flex-wrap items-center justify-center gap-1.5">
            {sectionSteps.map((step, index) => (
              <button
                key={step.short}
                type="button"
                onClick={() => goToStep(index)}
                aria-label={`Step ${index + 1}: ${step.label}`}
                aria-current={currentStep === index}
                className={`h-8 min-w-8 rounded-full border text-xs font-semibold transition ${
                  currentStep === index
                    ? "border-sky-400 bg-sky-500/20 text-sky-100"
                    : "border-gray-700 text-gray-400 hover:bg-neutral-800"
                }`}
              >
                {step.short}
              </button>
            ))}
          </div>
          <label className="mt-2 flex items-center justify-center gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={autoAdvance}
              onChange={(event) => handleAutoAdvanceChange(event.target.checked)}
            />
            Auto-advance to next step when ready
          </label>
        </nav>
      </main>
    </div>
  );
}
