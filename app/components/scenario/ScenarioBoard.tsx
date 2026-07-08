"use client";

import { useEffect, useId, useState } from "react";
import { useColorBlindMode } from "@/app/lib/accessibility";
import {
  attributesFromZones,
  colorMeta,
  patternSpoken,
  shapeSpoken,
  zonePattern,
  type ScenarioZone,
  type ZonePattern,
} from "@/app/lib/targets/scenario";

// Visual feedback per zone after a hit is registered.
export type ZoneFeedback = Record<string, "correct" | "wrong">;

// Human-readable trait list for a zone, e.g. "Square · Yellow · Solid · 12".
// Disabled attributes (black fill, number 0) are omitted. When patterns are in
// play for this zone set, "Solid" is a real, callable pattern value and is
// named like any other; when the pattern attribute is off entirely (every zone
// solid), it's omitted as noise.
function zoneTraits(zone: ScenarioZone, includeNumber: boolean, patternsInPlay: boolean): string {
  const parts = [shapeSpoken(zone.shape)];
  if (zone.color !== "black") parts.push(colorMeta(zone.color).spoken);
  const pattern = zonePattern(zone);
  if (patternsInPlay || pattern !== "solid") parts.push(patternSpoken(pattern));
  if (includeNumber && zone.number > 0) parts.push(String(zone.number));
  return parts.join(" · ");
}

// Returns the SVG points for a regular polygon / star / cross inscribed in a
// circle of radius r at (cx, cy). Coordinates are in the 0..100 viewBox space.
function polygonPoints(
  shape: ScenarioZone["shape"],
  cx: number,
  cy: number,
  r: number,
): string {
  const pt = (angle: number, radius: number) =>
    `${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`;
  const top = -Math.PI / 2; // point upward

  if (shape === "triangle") {
    return [0, 1, 2].map((i) => pt(top + (i * 2 * Math.PI) / 3, r)).join(" ");
  }
  if (shape === "diamond") {
    return [0, 1, 2, 3].map((i) => pt(top + (i * Math.PI) / 2, r)).join(" ");
  }
  if (shape === "pentagon") {
    return [0, 1, 2, 3, 4].map((i) => pt(top + (i * 2 * Math.PI) / 5, r)).join(" ");
  }
  if (shape === "hexagon") {
    return [0, 1, 2, 3, 4, 5].map((i) => pt(top + (i * 2 * Math.PI) / 6, r)).join(" ");
  }
  if (shape === "octagon") {
    return [0, 1, 2, 3, 4, 5, 6, 7].map((i) => pt(top + Math.PI / 8 + (i * 2 * Math.PI) / 8, r)).join(" ");
  }
  if (shape === "star") {
    const pts: string[] = [];
    for (let i = 0; i < 10; i += 1) {
      pts.push(pt(top + (i * Math.PI) / 5, i % 2 === 0 ? r : r * 0.45));
    }
    return pts.join(" ");
  }
  // cross / plus
  const a = r * 0.38;
  const b = r;
  return [
    `${cx - a},${cy - b}`,
    `${cx + a},${cy - b}`,
    `${cx + a},${cy - a}`,
    `${cx + b},${cy - a}`,
    `${cx + b},${cy + a}`,
    `${cx + a},${cy + a}`,
    `${cx + a},${cy + b}`,
    `${cx - a},${cy + b}`,
    `${cx - a},${cy + a}`,
    `${cx - b},${cy + a}`,
    `${cx - b},${cy - a}`,
    `${cx - a},${cy - a}`,
  ].join(" ");
}

// One repeating tile per pattern kind, drawn as white marks over the zone's
// base color so it prints legibly on both color and black-and-white printers.
// Tile coordinates are in the board's 0..100 viewBox space.
function PatternDef({ id, pattern, baseHex }: { id: string; pattern: ZonePattern; baseHex: string }) {
  const size = pattern === "checkered" || pattern === "ringed" || pattern === "zigzag" ? 8 : 6;
  const marks =
    pattern === "striped" ? (
      <rect x={0} y={0} width={size} height={2.2} fill="#fff" />
    ) : pattern === "banded" ? (
      <rect x={0} y={0} width={size} height={2.2} fill="#fff" />
    ) : pattern === "dotted" ? (
      <circle cx={size / 2} cy={size / 2} r={1.5} fill="#fff" />
    ) : pattern === "ringed" ? (
      <circle cx={size / 2} cy={size / 2} r={2.4} fill="none" stroke="#fff" strokeWidth={1.1} />
    ) : pattern === "checkered" ? (
      <>
        <rect x={0} y={0} width={size / 2} height={size / 2} fill="#fff" />
        <rect x={size / 2} y={size / 2} width={size / 2} height={size / 2} fill="#fff" />
      </>
    ) : pattern === "hatched" ? (
      <>
        <line x1={0} y1={0} x2={size} y2={size} stroke="#fff" strokeWidth={1} />
        <line x1={size} y1={0} x2={0} y2={size} stroke="#fff" strokeWidth={1} />
      </>
    ) : pattern === "grid" ? (
      <>
        <rect x={0} y={0} width={size} height={1.3} fill="#fff" />
        <rect x={0} y={0} width={1.3} height={size} fill="#fff" />
      </>
    ) : (
      // zigzag
      <polyline
        points={`0,${size * 0.7} ${size * 0.25},${size * 0.3} ${size * 0.5},${size * 0.7} ${size * 0.75},${size * 0.3} ${size},${size * 0.7}`}
        fill="none"
        stroke="#fff"
        strokeWidth={1.1}
      />
    );
  return (
    <pattern
      id={id}
      patternUnits="userSpaceOnUse"
      width={size}
      height={size}
      patternTransform={pattern === "striped" ? "rotate(45)" : undefined}
    >
      <rect width={size} height={size} fill={baseHex} />
      {marks}
    </pattern>
  );
}

function ZoneShapeEl({ zone, fill, stroke }: { zone: ScenarioZone; fill: string; stroke: string }) {
  const cx = zone.cx * 100;
  const cy = zone.cy * 100;
  const r = zone.radius * 100;
  const common = { fill, stroke, strokeWidth: 0.6 } as const;

  if (zone.shape === "circle") return <circle cx={cx} cy={cy} r={r} {...common} />;
  if (zone.shape === "square") {
    const s = r * 1.6;
    return <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} rx={r * 0.12} {...common} />;
  }
  return <polygon points={polygonPoints(zone.shape, cx, cy, r)} {...common} />;
}

export function ScenarioBoard({
  zones,
  onHitZone,
  onMiss,
  onZonePress,
  feedback = {},
  interactive = true,
  bare = false,
  className,
}: {
  zones: ScenarioZone[];
  onHitZone?: (zoneId: string) => void;
  onMiss?: () => void;
  // Fires for EVERY press on a zone, in every phase (unlike onHitZone, which
  // only fires while interactive). Used for ad-hoc feedback like speaking a
  // property of the pressed zone.
  onZonePress?: (zoneId: string) => void;
  feedback?: ZoneFeedback;
  interactive?: boolean;
  // bare = no card chrome/background; for overlaying onto a printable target.
  bare?: boolean;
  className?: string;
}) {
  // Pattern defs need document-unique ids; the board can appear more than once
  // (designer preview + drill page). Sanitized because useId's ":" breaks url(#…).
  const patternPrefix = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [colorBlind] = useColorBlindMode();
  // Click-to-inspect: the last clicked zone's full trait readout, shown in a
  // banner at the top of the board for a moment (works in every phase — during
  // play it rides along with the hit registration).
  const [inspectedId, setInspectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!inspectedId) return;
    const timer = window.setTimeout(() => setInspectedId(null), 2500);
    return () => window.clearTimeout(timer);
  }, [inspectedId]);

  const handleClick = (event: React.MouseEvent<SVGSVGElement>) => {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    // Nearest zone whose radius contains the click.
    let hit: string | null = null;
    let best = Infinity;
    for (const zone of zones) {
      const d = Math.hypot(x - zone.cx, y - zone.cy);
      if (d <= zone.radius && d < best) {
        best = d;
        hit = zone.id;
      }
    }
    setInspectedId(hit);
    if (hit) onZonePress?.(hit);
    if (!interactive) return;
    if (hit) onHitZone?.(hit);
    else onMiss?.();
  };

  const inspectedZone = inspectedId ? (zones.find((zone) => zone.id === inspectedId) ?? null) : null;
  // "Solid" is only worth naming when patterns actually distinguish zones here.
  const patternsInPlay = attributesFromZones(zones).includes("pattern");

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid meet"
      onClick={handleClick}
      className={`${
        bare ? "" : "aspect-square w-full rounded-lg border border-gray-700 bg-neutral-950"
      } ${interactive ? "cursor-crosshair" : ""} ${className ?? ""}`}
    >
      <defs>
        {zones.map((zone) => {
          const pattern = zonePattern(zone);
          if (pattern === "solid") return null;
          return (
            <PatternDef
              key={zone.id}
              id={`${patternPrefix}-${zone.id}`}
              pattern={pattern}
              baseHex={colorMeta(zone.color, colorBlind).hex}
            />
          );
        })}
      </defs>
      {zones.map((zone) => {
        const meta = colorMeta(zone.color, colorBlind);
        const pattern = zonePattern(zone);
        const state = feedback[zone.id];
        const stroke = state === "correct" ? "#22c55e" : state === "wrong" ? "#f43f5e" : "#0a0a0a";
        const cx = zone.cx * 100;
        const cy = zone.cy * 100;
        const fontSize = zone.radius * 100 * 0.9;
        // Yellow needs a dark number for contrast; everything else gets white.
        const textColor = zone.color === "yellow" ? "#111827" : "#ffffff";
        const fill = pattern === "solid" ? meta.hex : `url(#${patternPrefix}-${zone.id})`;
        const r = zone.radius * 100;
        const captionSize = Math.max(1.7, Math.min(2.4, r * 0.17));
        return (
          <g key={zone.id} opacity={state === "wrong" ? 0.55 : 1}>
            <ZoneShapeEl zone={zone} fill={fill} stroke={stroke} />
            {zone.number > 0 ? (
              <text
                x={cx}
                y={cy}
                fontSize={fontSize}
                fontWeight={700}
                fill={textColor}
                // Patterned fills are busy — outline the number so it stays legible.
                stroke={pattern === "solid" ? undefined : textColor === "#ffffff" ? "#111827" : "#ffffff"}
                strokeWidth={pattern === "solid" ? undefined : fontSize * 0.1}
                paintOrder="stroke"
                textAnchor="middle"
                dominantBaseline="central"
                pointerEvents="none"
                style={{ fontFamily: "var(--font-jetbrains-mono), monospace" }}
              >
                {zone.number}
              </text>
            ) : null}
            {/* Very small trait caption under the symbol (shape · color · pattern). */}
            <text
              x={cx}
              y={cy + r + 0.8}
              fontSize={captionSize}
              textAnchor="middle"
              dominantBaseline="hanging"
              pointerEvents="none"
              fill={bare ? "#374151" : "#9ca3af"}
              style={{ fontFamily: "var(--font-jetbrains-mono), monospace" }}
            >
              {zoneTraits(zone, false, patternsInPlay)}
            </text>
          </g>
        );
      })}
      {/* Click-to-inspect banner: the clicked zone's full characteristics. */}
      {inspectedZone ? (
        <g pointerEvents="none">
          <rect
            x={2}
            y={1.5}
            width={96}
            height={6.4}
            rx={1.2}
            fill={bare ? "#ffffff" : "rgba(9, 9, 11, 0.88)"}
            stroke={bare ? "#111827" : "#52525b"}
            strokeWidth={0.25}
          />
          <text
            x={50}
            y={4.7}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={3.2}
            fontWeight={600}
            fill={bare ? "#111827" : "#f9fafb"}
            style={{ fontFamily: "var(--font-jetbrains-mono), monospace" }}
          >
            {zoneTraits(inspectedZone, true, patternsInPlay)}
          </text>
        </g>
      ) : null}
    </svg>
  );
}
