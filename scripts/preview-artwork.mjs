// Dev-only visual QA: renders the new target artwork paths (mirrored from
// TargetDesigner.tsx) to PNGs so they can be eyeballed without a browser.
import sharp from "sharp";
import { mkdirSync, writeFileSync } from "node:fs";

const SIL = "#3f3f3f";
const PER_INCH = { in: 1, cm: 2.54, mm: 25.4 };
const fromInches = (v, u) => v * PER_INCH[u];

function fitShape(w, h, unit, shapeW, shapeH) {
  const margin = fromInches(0.5, unit);
  const s = Math.min((w - margin * 2) / shapeW, (h - margin * 2) / shapeH);
  return { s, x: (w - shapeW * s) / 2, y: (h - shapeH * s) / 2 };
}

function b27(w, h, unit) {
  const { s, x, y } = fitShape(w, h, unit, 100, 140);
  const rings = [
    { rx: 27, ry: 34, label: "8", labelY: 113 },
    { rx: 19, ry: 25, label: "9", labelY: 104 },
    { rx: 11.5, ry: 15, label: "10", labelY: 94.5 },
    { rx: 5, ry: 6.5, label: "X", labelY: 83.5 },
  ];
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <path d="M50 4 C61 4 69 12 69 22 C69 28 66 33 61 36 C77 41 87 53 89 70 L92 140 L8 140 L11 70 C13 53 23 41 39 36 C34 33 31 28 31 22 C31 12 39 4 50 4 Z" fill="${SIL}"/>
    ${rings
      .map(
        (r) =>
          `<ellipse cx="50" cy="82" rx="${r.rx}" ry="${r.ry}" fill="none" stroke="#fff" stroke-width="0.7"/>
           <text x="50" y="${r.labelY}" text-anchor="middle" font-size="4" fill="#fff">${r.label}</text>`,
      )
      .join("")}
  </g>`;
}

function uspsa(w, h, unit) {
  const { s, x, y } = fitShape(w, h, unit, 18, 30);
  const lbl = `font-size="1" fill="#777" text-anchor="middle"`;
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <path d="M6 0 H12 V6 L18 10 V30 H0 V10 L6 6 Z" fill="none" stroke="#000" stroke-width="0.08"/>
    <line x1="6" y1="4" x2="12" y2="4" stroke="#555" stroke-width="0.04" stroke-dasharray="0.35 0.25"/>
    <rect x="6" y="13" width="6" height="11" fill="none" stroke="#555" stroke-width="0.04" stroke-dasharray="0.35 0.25"/>
    <rect x="3" y="9" width="12" height="18" fill="none" stroke="#555" stroke-width="0.04" stroke-dasharray="0.35 0.25"/>
    <text x="9" y="19" ${lbl}>A</text><text x="9" y="26.4" ${lbl}>C</text>
    <text x="9" y="29.2" ${lbl}>D</text><text x="9" y="3" ${lbl}>A</text>
  </g>`;
}

function usmc(w, h, unit) {
  const { s, x, y } = fitShape(w, h, unit, 60, 60);
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <path d="M0 60 L0 44 C0 39 4 36 9 36 L21 36 C16.5 31.5 14 25.6 14 19 C14 9.6 21.2 2 30 2 C38.8 2 46 9.6 46 19 C46 25.6 43.5 31.5 39 36 L51 36 C56 36 60 39 60 44 L60 60 Z" fill="${SIL}"/>
    <line x1="26" y1="46" x2="34" y2="46" stroke="#fff" stroke-width="0.7"/>
    <line x1="30" y1="42" x2="30" y2="50" stroke="#fff" stroke-width="0.7"/>
  </g>`;
}

function dots(w, h, unit, q) {
  const r = fromInches(1, unit);
  const gap = fromInches(0.5, unit);
  const margin = fromInches(0.5, unit);
  const pitch = r * 2 + gap;
  const cols = Math.max(1, Math.floor((w - margin * 2 + gap) / pitch));
  const rows = Math.max(1, Math.floor((h - margin * 2 + gap) / pitch));
  const x0 = (w - (cols * r * 2 + (cols - 1) * gap)) / 2 + r;
  const y0 = (h - (rows * r * 2 + (rows - 1) * gap)) / 2 + r;
  const qm = fromInches(0.4, unit);
  const qs = q * 1.3;
  const rect = { x1: qm, y1: h - qm - qs - fromInches(0.3, unit), x2: qm + qs, y2: h - qm };
  let out = "";
  let n = 0;
  for (let row = 0; row < rows; row += 1)
    for (let col = 0; col < cols; col += 1) {
      const x = x0 + col * pitch;
      const y = y0 + row * pitch;
      const nx = Math.min(Math.max(x, rect.x1), rect.x2);
      const ny = Math.min(Math.max(y, rect.y1), rect.y2);
      if (Math.hypot(x - nx, y - ny) < r) continue;
      n += 1;
      out += `<circle cx="${x}" cy="${y}" r="${r}" fill="${SIL}"/>
        <text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size="${r * 0.8}" fill="#fff">${n}</text>`;
    }
  // sketch the QR footprint so overlap is visible
  out += `<rect x="${rect.x1}" y="${rect.y1}" width="${rect.x2 - rect.x1}" height="${rect.y2 - rect.y1}" fill="none" stroke="#38bdf8" stroke-width="0.03"/>`;
  return out;
}

function mpms(w, h, unit) {
  const { s, x, y } = fitShape(w, h, unit, 19.5, 40);
  const zone = `fill="none" stroke="#fff" stroke-width="0.12" stroke-dasharray="0.5 0.35"`;
  const lbl = `font-size="1.6" fill="#fff" text-anchor="middle"`;
  return `<g transform="translate(${x} ${y}) scale(${s})">
    <path d="M9.75 1.2 C12.7 1.2 14.9 3.4 14.9 6.3 C14.9 8.3 14 10 12.6 11 C16.8 12.2 19.5 14.3 19.5 17.5 L19.5 40 L0 40 L0 17.5 C0 14.3 2.7 12.2 6.9 11 C5.5 10 4.6 8.3 4.6 6.3 C4.6 3.4 6.8 1.2 9.75 1.2 Z" fill="${SIL}"/>
    <rect x="7.25" y="3.5" width="5" height="5" ${zone}/><text x="9.75" y="6.8" ${lbl}>A</text>
    <ellipse cx="9.75" cy="18.5" rx="4.5" ry="4" ${zone}/><text x="9.75" y="19.1" ${lbl}>B</text>
    <ellipse cx="9.75" cy="24" rx="8" ry="11" ${zone}/><text x="9.75" y="31.5" ${lbl}>C</text>
    <text x="9.75" y="38.5" ${lbl}>D</text>
  </g>`;
}

// Pattern tiles mirrored from ScenarioBoard's PatternDef, one swatch per pattern.
function patternBoard() {
  const pats = ["solid", "striped", "banded", "dotted", "ringed", "checkered", "hatched", "zigzag"];
  const defs = pats
    .map((p) => {
      if (p === "solid") return "";
      const size = p === "checkered" || p === "ringed" || p === "zigzag" ? 8 : 6;
      const marks =
        p === "striped" || p === "banded"
          ? `<rect x="0" y="0" width="${size}" height="2.2" fill="#fff"/>`
          : p === "dotted"
            ? `<circle cx="${size / 2}" cy="${size / 2}" r="1.5" fill="#fff"/>`
            : p === "ringed"
              ? `<circle cx="${size / 2}" cy="${size / 2}" r="2.4" fill="none" stroke="#fff" stroke-width="1.1"/>`
              : p === "checkered"
                ? `<rect x="0" y="0" width="${size / 2}" height="${size / 2}" fill="#fff"/><rect x="${size / 2}" y="${size / 2}" width="${size / 2}" height="${size / 2}" fill="#fff"/>`
                : p === "hatched"
                  ? `<line x1="0" y1="0" x2="${size}" y2="${size}" stroke="#fff" stroke-width="1"/><line x1="${size}" y1="0" x2="0" y2="${size}" stroke="#fff" stroke-width="1"/>`
                  : `<polyline points="0,${size * 0.7} ${size * 0.25},${size * 0.3} ${size * 0.5},${size * 0.7} ${size * 0.75},${size * 0.3} ${size},${size * 0.7}" fill="none" stroke="#fff" stroke-width="1.1"/>`;
      const rot = p === "striped" ? ` patternTransform="rotate(45)"` : "";
      return `<pattern id="p-${p}" patternUnits="userSpaceOnUse" width="${size}" height="${size}"${rot}><rect width="${size}" height="${size}" fill="#262626"/>${marks}</pattern>`;
    })
    .join("");
  const cells = pats
    .map((p, i) => {
      const cx = (i % 4) * 25 + 12.5;
      const cy = Math.floor(i / 4) * 25 + 12.5;
      const fill = p === "solid" ? "#262626" : `url(#p-${p})`;
      return `<circle cx="${cx}" cy="${cy}" r="9" fill="${fill}" stroke="#0a0a0a" stroke-width="0.6"/>
        <text x="${cx}" y="${cy + 11.5}" text-anchor="middle" font-size="2.2" fill="#000">${p}</text>
        <text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="700" fill="#fff" stroke="#111827" stroke-width="0.8" paint-order="stroke">42</text>`;
    })
    .join("");
  return `<defs>${defs}</defs>${cells}`;
}

const cases = [
  { name: "mpms", w: 19.5, h: 40, unit: "in", body: mpms(19.5, 40, "in") },
  { name: "mpms-7yd", w: 5.46, h: 11.2, unit: "in", body: mpms(5.46, 11.2, "in") },
  { name: "patterns", w: 100, h: 50, unit: "in", body: patternBoard() },
  { name: "b27", w: 12, h: 18, unit: "in", body: b27(12, 18, "in") },
  { name: "uspsa", w: 18, h: 30, unit: "in", body: uspsa(18, 30, "in") },
  { name: "usmc", w: 12, h: 18, unit: "in", body: usmc(12, 18, "in") },
  { name: "dots", w: 8.5, h: 11, unit: "in", body: dots(8.5, 11, "in", 1.5) },
  { name: "b27-letter", w: 8.5, h: 11, unit: "in", body: b27(8.5, 11, "in") },
  { name: "uspsa-a4mm", w: 210, h: 297, unit: "mm", body: uspsa(210, 297, "mm") },
];

mkdirSync("scripts/.preview", { recursive: true });
for (const c of cases) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${c.w} ${c.h}" width="${c.w * 40}" height="${c.h * 40}">
    <rect width="${c.w}" height="${c.h}" fill="#fff"/>
    <rect width="${c.w}" height="${c.h}" fill="none" stroke="#000" stroke-width="${fromInches(0.02, c.unit)}"/>
    ${c.body}
  </svg>`;
  writeFileSync(`scripts/.preview/${c.name}.svg`, svg);
  await sharp(Buffer.from(svg)).png().toFile(`scripts/.preview/${c.name}.png`);
  console.log("wrote", c.name);
}
