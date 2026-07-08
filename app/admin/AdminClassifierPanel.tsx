"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildDatasetZip, type TrainingSample } from "@/app/lib/training/dataset";
import { cropGrayWindow, type GrayPatch } from "@/app/lib/training/holeClassifier";
import { buildClassifierFromModel, trainModel, type ModelJSON } from "@/app/lib/training/jsonModel";

// A labeled patch held in the panel: the cropped grayscale window (for training)
// plus a PNG thumbnail (for display/export) and its position on the source image.
type AdminSample = {
  id: string;
  label: "hole" | "not_hole";
  side: number;
  gray: number[]; // length side*side, 0..255
  dataUrl: string;
  x: number; // source-image coordinates of the click
  y: number;
};

type VersionRow = {
  id: string;
  version: number;
  kind: string;
  notes: string | null;
  isPublished: boolean;
  createdAt: string;
  meta: ModelJSON["meta"] | null;
};

const MAX_DISPLAY_WIDTH = 720;
const IDB_NAME = "trackr-admin-classifier";
const IDB_STORE = "samples";

// --- Tiny IndexedDB persistence (samples can be large; localStorage is too small) ---

function idbOpen(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(IDB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(IDB_STORE)) {
        request.result.createObjectStore(IDB_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function idbSave(samples: AdminSample[]): Promise<void> {
  const db = await idbOpen();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).put(samples, "all");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function idbLoad(): Promise<AdminSample[]> {
  const db = await idbOpen();
  const result = await new Promise<AdminSample[]>((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readonly");
    const get = tx.objectStore(IDB_STORE).get("all");
    get.onsuccess = () => resolve(Array.isArray(get.result) ? (get.result as AdminSample[]) : []);
    get.onerror = () => reject(get.error);
  });
  db.close();
  return result;
}

// --- Image helpers ---

// Render an RGBA image to a grayscale buffer (luma) at natural resolution.
function toGray(image: HTMLImageElement): { data: Uint8Array; width: number; height: number } {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(image, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < gray.length; i += 1) {
    gray[i] = (data[i * 4] * 299 + data[i * 4 + 1] * 587 + data[i * 4 + 2] * 114) / 1000;
  }
  return { data: gray, width: canvas.width, height: canvas.height };
}

// Grayscale patch → small PNG data URL (thumbnail + dataset export).
function patchToDataUrl(patch: GrayPatch): string {
  const canvas = document.createElement("canvas");
  canvas.width = patch.width;
  canvas.height = patch.height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(patch.width, patch.height);
  for (let i = 0; i < patch.width * patch.height; i += 1) {
    const v = patch.data[i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function sampleToPatch(sample: AdminSample): GrayPatch {
  return { data: Uint8Array.from(sample.gray), width: sample.side, height: sample.side };
}

export function AdminClassifierPanel() {
  const [samples, setSamples] = useState<AdminSample[]>([]);
  const [labelMode, setLabelMode] = useState<"hole" | "not_hole">("hole");
  const [patchSize, setPatchSize] = useState(32);
  const [model, setModel] = useState<ModelJSON | null>(null);
  const [notes, setNotes] = useState("");
  const [versions, setVersions] = useState<VersionRow[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [lastScore, setLastScore] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const grayRef = useRef<{ data: Uint8Array; width: number; height: number } | null>(null);
  const displayScaleRef = useRef(1); // source px per displayed px
  const idCounter = useRef(0);

  const holeCount = useMemo(() => samples.filter((s) => s.label === "hole").length, [samples]);
  const negativeCount = samples.length - holeCount;
  const canTrain = holeCount > 0 && negativeCount > 0 && samples.length >= 4;

  const refreshVersions = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/classifier", { cache: "no-store" });
      if (!res.ok) return;
      const json = (await res.json()) as { versions?: VersionRow[] };
      setVersions(json.versions ?? []);
    } catch {
      /* ignore */
    }
  }, []);

  // Load persisted samples + published-version history on mount.
  useEffect(() => {
    let cancelled = false;
    idbLoad()
      .then((rows) => {
        if (!cancelled) {
          setSamples(rows);
          idCounter.current = rows.length;
        }
      })
      .catch(() => {})
      .finally(() => !cancelled && setLoaded(true));
    void refreshVersions();
    return () => {
      cancelled = true;
    };
  }, [refreshVersions]);

  // Persist whenever samples change (after the initial load).
  useEffect(() => {
    if (!loaded) return;
    void idbSave(samples).catch(() => {});
  }, [samples, loaded]);

  // Redraw the source image plus labeled markers whenever samples change.
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const scale = displayScaleRef.current || 1;
    for (const s of samples) {
      const dx = s.x / scale;
      const dy = s.y / scale;
      const half = s.side / scale / 2;
      ctx.strokeStyle = s.label === "hole" ? "#34d399" : "#f87171";
      ctx.lineWidth = 2;
      ctx.strokeRect(dx - half, dy - half, half * 2, half * 2);
    }
  }, [samples]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const onFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      const scale = image.naturalWidth > MAX_DISPLAY_WIDTH ? image.naturalWidth / MAX_DISPLAY_WIDTH : 1;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = Math.round(image.naturalWidth / scale);
        canvas.height = Math.round(image.naturalHeight / scale);
      }
      imageRef.current = image;
      grayRef.current = toGray(image);
      displayScaleRef.current = scale;
      URL.revokeObjectURL(url);
      redraw();
    };
    image.src = url;
  }, [redraw]);

  const addSampleAt = useCallback(
    (displayX: number, displayY: number, label: "hole" | "not_hole") => {
      const gray = grayRef.current;
      if (!gray) {
        setStatus("Load an image first.");
        return;
      }
      const scale = displayScaleRef.current || 1;
      const sx = displayX * scale;
      const sy = displayY * scale;
      const patch = cropGrayWindow(gray.data, gray.width, gray.height, sx, sy, patchSize);
      idCounter.current += 1;
      const sample: AdminSample = {
        id: `s${idCounter.current}`,
        label,
        side: patch.width,
        gray: Array.from(patch.data),
        dataUrl: patchToDataUrl(patch),
        x: sx,
        y: sy,
      };
      setSamples((prev) => [...prev, sample]);

      // If a model exists, show its score at this spot — quick sanity check.
      if (model) setLastScore(buildClassifierFromModel(model).score(patch));
    },
    [model, patchSize],
  );

  const onCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * event.currentTarget.width;
      const y = ((event.clientY - rect.top) / rect.height) * event.currentTarget.height;
      // Right-click (or shift) labels the opposite class for fast alternating.
      const label =
        event.button === 2 || event.shiftKey
          ? labelMode === "hole"
            ? "not_hole"
            : "hole"
          : labelMode;
      addSampleAt(x, y, label);
    },
    [addSampleAt, labelMode],
  );

  const removeSample = useCallback((id: string) => {
    setSamples((prev) => prev.filter((s) => s.id !== id));
  }, []);

  const onTrain = useCallback(() => {
    const trained = trainModel(
      samples.map((s) => ({ patch: sampleToPatch(s), label: s.label })),
    );
    if (!trained) {
      setStatus("Need at least 4 samples with both holes and non-holes to train.");
      return;
    }
    setModel(trained);
    setStatus(
      `Trained on ${trained.meta.samples} samples — ${(trained.meta.trainAccuracy * 100).toFixed(1)}% training accuracy.`,
    );
  }, [samples]);

  const onPublish = useCallback(async () => {
    if (!model) return;
    setPublishing(true);
    setStatus(null);
    try {
      const res = await fetch("/api/admin/classifier", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, notes }),
      });
      const json = (await res.json()) as { ok?: boolean; version?: number; error?: string };
      if (!res.ok || !json.ok) {
        setStatus(json.error ?? "Publish failed.");
      } else {
        setStatus(`Published version ${json.version}. Scanners will load it on their next startup.`);
        setNotes("");
        await refreshVersions();
      }
    } catch {
      setStatus("Publish failed (network).");
    } finally {
      setPublishing(false);
    }
  }, [model, notes, refreshVersions]);

  const onExport = useCallback(() => {
    const trainingSamples: TrainingSample[] = samples.map((s) => ({
      id: s.id,
      label: s.label,
      source: "manual",
      dataUrl: s.dataUrl,
      createdAt: 0,
    }));
    const blob = buildDatasetZip(trainingSamples);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "trackr-holes-dataset.zip";
    a.click();
    URL.revokeObjectURL(url);
  }, [samples]);

  const onClear = useCallback(() => {
    setSamples([]);
    setModel(null);
    setLastScore(null);
  }, []);

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-gray-800 bg-neutral-950 p-4">
        <label className="cursor-pointer rounded-lg bg-sky-600 px-3 py-1.5 text-sm font-semibold hover:bg-sky-500">
          Load image
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
          />
        </label>

        <div className="flex overflow-hidden rounded-lg border border-gray-700">
          <button
            type="button"
            onClick={() => setLabelMode("hole")}
            className={`px-3 py-1.5 text-sm font-semibold ${labelMode === "hole" ? "bg-emerald-600" : "bg-neutral-900 text-gray-300"}`}
          >
            Hole
          </button>
          <button
            type="button"
            onClick={() => setLabelMode("not_hole")}
            className={`px-3 py-1.5 text-sm font-semibold ${labelMode === "not_hole" ? "bg-rose-600" : "bg-neutral-900 text-gray-300"}`}
          >
            Not hole
          </button>
        </div>

        <label className="flex items-center gap-2 text-sm text-gray-300">
          Patch
          <input
            type="range"
            min={16}
            max={64}
            step={2}
            value={patchSize}
            onChange={(e) => setPatchSize(Number(e.target.value))}
          />
          <span className="w-8 tabular-nums text-gray-400">{patchSize}px</span>
        </label>

        <span className="ml-auto text-sm text-gray-400">
          <span className="font-semibold text-emerald-300">{holeCount}</span> holes ·{" "}
          <span className="font-semibold text-rose-300">{negativeCount}</span> non-holes
        </span>
      </div>

      <p className="text-xs text-gray-500">
        Left-click to label as <span className="text-gray-300">{labelMode === "hole" ? "hole" : "not hole"}</span>;
        right-click (or shift-click) labels the opposite. Patches persist in this browser until you publish or clear.
      </p>

      {/* Canvas */}
      <div className="overflow-auto rounded-xl border border-gray-800 bg-neutral-900">
        <canvas
          ref={canvasRef}
          onClick={onCanvasClick}
          onContextMenu={(e) => {
            e.preventDefault();
            onCanvasClick(e);
          }}
          className="block max-w-full cursor-crosshair"
        />
      </div>

      {/* Train / publish */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-gray-800 bg-neutral-950 p-4">
        <button
          type="button"
          onClick={onTrain}
          disabled={!canTrain}
          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-semibold hover:bg-indigo-500 disabled:opacity-40"
        >
          Train model
        </button>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Version notes (optional)"
          className="min-w-48 flex-1 rounded-lg border border-gray-700 bg-neutral-900 px-3 py-1.5 text-sm"
        />
        <button
          type="button"
          onClick={onPublish}
          disabled={!model || publishing}
          className="rounded-lg bg-amber-600 px-3 py-1.5 text-sm font-semibold hover:bg-amber-500 disabled:opacity-40"
        >
          {publishing ? "Publishing…" : "Publish to users"}
        </button>
        <button
          type="button"
          onClick={onExport}
          disabled={samples.length === 0}
          className="rounded-lg border border-gray-700 px-3 py-1.5 text-sm hover:bg-neutral-900 disabled:opacity-40"
        >
          Export dataset
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={samples.length === 0}
          className="rounded-lg border border-gray-700 px-3 py-1.5 text-sm text-rose-300 hover:bg-neutral-900 disabled:opacity-40"
        >
          Clear
        </button>
      </div>

      {status && <p className="text-sm text-amber-200">{status}</p>}
      {model && (
        <p className="text-sm text-gray-400">
          Current trained model: {model.meta.holes} holes / {model.meta.negatives} non-holes ·{" "}
          {(model.meta.trainAccuracy * 100).toFixed(1)}% acc
          {lastScore !== null && <> · last click scored {(lastScore * 100).toFixed(0)}%</>}
        </p>
      )}

      {/* Sample thumbnails */}
      {samples.length > 0 && (
        <div className="rounded-xl border border-gray-800 bg-neutral-950 p-4">
          <h2 className="mb-3 text-sm font-semibold text-gray-300">Labeled patches</h2>
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button
                key={s.id}
                type="button"
                title="Remove"
                onClick={() => removeSample(s.id)}
                className={`relative h-12 w-12 overflow-hidden rounded border-2 ${s.label === "hole" ? "border-emerald-500" : "border-rose-500"}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={s.dataUrl} alt={s.label} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Published-version history */}
      <div className="rounded-xl border border-gray-800 bg-neutral-950 p-4">
        <h2 className="mb-3 text-sm font-semibold text-gray-300">Published history</h2>
        {versions.length === 0 ? (
          <p className="text-sm text-gray-500">No models published yet.</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {versions.map((v) => (
              <li key={v.id} className="flex items-center gap-3">
                <span className="font-semibold tabular-nums">v{v.version}</span>
                {v.isPublished && (
                  <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs text-emerald-200">live</span>
                )}
                <span className="text-gray-400">
                  {v.meta ? `${v.meta.holes}/${v.meta.negatives} · ${(v.meta.trainAccuracy * 100).toFixed(0)}% acc` : v.kind}
                </span>
                {v.notes && <span className="truncate text-gray-500">— {v.notes}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
