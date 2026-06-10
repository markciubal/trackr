export type CvMat = {
  clone: () => CvMat;
  delete: () => void;
  // Present on real OpenCV mats; optional so existing call sites that only use
  // clone/delete keep type-checking.
  rows?: number;
  cols?: number;
  data32F?: Float32Array;
};

// OpenCV's QR detector+decoder (objdetect module, included in opencv.js).
export type CvQRCodeDetector = {
  // Returns the decoded string ("" if it can't decode) and fills `points`
  // with the 4 corner coordinates of the located QR.
  detectAndDecode: (img: CvMat, points?: CvMat, straightQrCode?: CvMat) => string;
  detect: (img: CvMat, points: CvMat) => boolean;
  delete: () => void;
};

export type CvApi = {
  Mat: new () => CvMat;
  MatVector: new () => {
    size: () => number;
    get: (index: number) => CvMat;
    delete: () => void;
  };
  Point: new (x: number, y: number) => unknown;
  Size: new (width: number, height: number) => unknown;
  QRCodeDetector: new () => CvQRCodeDetector;
  matFromImageData: (imageData: ImageData) => CvMat;
  cvtColor: (src: CvMat, dst: CvMat, code: number, dstCn?: number) => void;
  GaussianBlur: (
    src: CvMat,
    dst: CvMat,
    ksize: unknown,
    sigmaX: number,
    sigmaY: number,
    borderType: number,
  ) => void;
  absdiff: (src1: CvMat, src2: CvMat, dst: CvMat) => void;
  threshold: (
    src: CvMat,
    dst: CvMat,
    thresh: number,
    maxval: number,
    type: number,
  ) => void;
  getStructuringElement: (shape: number, ksize: unknown) => CvMat;
  dilate: (
    src: CvMat,
    dst: CvMat,
    kernel: CvMat,
    anchor: unknown,
    iterations: number,
  ) => void;
  findContours: (
    image: CvMat,
    contours: {
      size: () => number;
      get: (index: number) => CvMat;
      delete: () => void;
    },
    hierarchy: CvMat,
    mode: number,
    method: number,
  ) => void;
  contourArea: (contour: CvMat, oriented?: boolean) => number;
  boundingRect: (contour: CvMat) => {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  COLOR_RGBA2GRAY: number;
  THRESH_BINARY: number;
  THRESH_BINARY_INV: number;
  THRESH_OTSU: number;
  RETR_EXTERNAL: number;
  CHAIN_APPROX_SIMPLE: number;
  MORPH_RECT: number;
  BORDER_DEFAULT: number;
  onRuntimeInitialized?: () => void;
};

declare global {
  interface Window {
    cv?: CvApi;
    __opencvReadyPromise?: Promise<void>;
  }
}

const OPEN_CV_URL = "https://docs.opencv.org/4.x/opencv.js";

export const loadOpenCv = (): Promise<void> => {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Window is not available."));
  }

  if (window.cv?.Mat) {
    return Promise.resolve();
  }

  if (window.__opencvReadyPromise) {
    return window.__opencvReadyPromise;
  }

  window.__opencvReadyPromise = new Promise<void>((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(
      'script[data-opencv="true"]',
    );

    const wireRuntimeCallback = () => {
      if (!window.cv) {
        reject(new Error("OpenCV global did not initialize."));
        return;
      }

      if (window.cv.Mat) {
        resolve();
        return;
      }

      window.cv.onRuntimeInitialized = () => {
        resolve();
      };
    };

    if (existingScript) {
      if (window.cv?.Mat) {
        resolve();
        return;
      }

      if (window.cv && !window.cv.Mat) {
        window.cv.onRuntimeInitialized = () => {
          resolve();
        };
      }

      existingScript.addEventListener("load", wireRuntimeCallback, {
        once: true,
      });
      existingScript.addEventListener(
        "error",
        () => reject(new Error("Failed to load OpenCV script.")),
        { once: true },
      );
      return;
    }

    const script = document.createElement("script");
    script.src = OPEN_CV_URL;
    script.async = true;
    script.dataset.opencv = "true";
    script.onload = wireRuntimeCallback;
    script.onerror = () => reject(new Error("Failed to load OpenCV script."));
    document.body.appendChild(script);
  });

  return window.__opencvReadyPromise;
};
