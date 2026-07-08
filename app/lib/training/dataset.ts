// Browser-side training-data collection for a bullet-hole classifier.
//
// Samples are little image patches (PNG data URLs) labeled hole / not_hole,
// harvested as you use the scanner (confirmed hits → positives, background taps
// → negatives). buildDatasetZip packs them into a standard ImageFolder layout
//   hole/0001.png, not_hole/0001.png, manifest.csv
// that PyTorch's ImageFolder / Ultralytics classification training read directly
// — no labeling step, no dependencies.

export type TrainingLabel = "hole" | "not_hole";
export type TrainingSource = "manual" | "change_detect" | "background";

export type TrainingSample = {
  id: string;
  label: TrainingLabel;
  source: TrainingSource;
  dataUrl: string; // PNG data URL of the cropped patch
  createdAt: number;
};

// ---- Minimal, dependency-free "store" (uncompressed) ZIP writer ----

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function dataUrlToBytes(dataUrl: string): Uint8Array {
  const base64 = dataUrl.split(",")[1] ?? "";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function utf8(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

export function buildDatasetZip(samples: TrainingSample[]): Blob {
  const files: { name: string; data: Uint8Array }[] = [];
  let holeCount = 0;
  let negativeCount = 0;
  const manifest = ["filename,label,source,createdAt"];

  for (const sample of samples) {
    const index = sample.label === "hole" ? (holeCount += 1) : (negativeCount += 1);
    const name = `${sample.label}/${String(index).padStart(4, "0")}.png`;
    files.push({ name, data: dataUrlToBytes(sample.dataUrl) });
    manifest.push(`${name},${sample.label},${sample.source},${sample.createdAt}`);
  }
  files.push({ name: "manifest.csv", data: utf8(manifest.join("\n")) });

  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = utf8(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); // local file header signature
    local.setUint16(4, 20, true); // version needed
    local.setUint16(8, 0, true); // method = store
    local.setUint32(14, crc, true);
    local.setUint32(18, size, true); // compressed size
    local.setUint32(22, size, true); // uncompressed size
    local.setUint16(26, nameBytes.length, true);
    localChunks.push(new Uint8Array(local.buffer), nameBytes, file.data);

    const central = new DataView(new ArrayBuffer(46));
    central.setUint32(0, 0x02014b50, true); // central dir signature
    central.setUint16(4, 20, true); // version made by
    central.setUint16(6, 20, true); // version needed
    central.setUint16(10, 0, true); // method = store
    central.setUint32(16, crc, true);
    central.setUint32(20, size, true);
    central.setUint32(24, size, true);
    central.setUint16(28, nameBytes.length, true);
    central.setUint32(42, offset, true); // local header offset
    centralChunks.push(new Uint8Array(central.buffer), nameBytes);

    offset += 30 + nameBytes.length + size;
  }

  const centralSize = centralChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); // end of central dir signature
  end.setUint16(8, files.length, true); // entries on this disk
  end.setUint16(10, files.length, true); // total entries
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true); // central dir offset

  // Cast through unknown: the chunks are plain ArrayBuffer-backed Uint8Arrays at
  // runtime, but TS widens their type to Uint8Array<ArrayBufferLike>, which the
  // DOM BlobPart type rejects over a (here impossible) SharedArrayBuffer.
  const parts = [...localChunks, ...centralChunks, new Uint8Array(end.buffer)] as unknown as BlobPart[];
  return new Blob(parts, { type: "application/zip" });
}
