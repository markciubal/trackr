"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { getSeenTips, markTipsSeen } from "@/app/lib/onboarding";

type Placement = "bottom" | "top" | "left" | "right";

const PLACEMENT_CLASS: Record<Placement, string> = {
  bottom: "top-full left-1/2 mt-2 -translate-x-1/2",
  top: "bottom-full left-1/2 mb-2 -translate-x-1/2",
  left: "right-full top-1/2 mr-2 -translate-y-1/2",
  right: "left-full top-1/2 ml-2 -translate-y-1/2",
};

// Reusable, dependency-free popover for short, important instructions. Drop one
// next to any control: <InfoPopover title="..." tipId="...">Body.</InfoPopover>.
// If `autoOpen` + `tipId` are set, it opens once for first-time users.
export function InfoPopover({
  title,
  children,
  tipId,
  autoOpen = false,
  placement = "bottom",
  triggerLabel = "i",
  triggerClassName,
}: {
  title?: string;
  children: ReactNode;
  tipId?: string;
  autoOpen?: boolean;
  placement?: Placement;
  triggerLabel?: ReactNode;
  triggerClassName?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    // One-time auto-open from client-only localStorage; intentional in an effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (autoOpen && tipId && !getSeenTips().has(tipId)) setOpen(true);
  }, [autoOpen, tipId]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const dismiss = () => {
    if (tipId) markTipsSeen([tipId]);
    setOpen(false);
  };

  return (
    <span ref={ref} className="relative inline-flex align-middle">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label={title ? `Help: ${title}` : "Help"}
        aria-expanded={open}
        className={
          triggerClassName ??
          "flex h-4 w-4 items-center justify-center rounded-full border border-gray-600 text-[10px] font-semibold leading-none text-gray-300 transition hover:bg-neutral-800 hover:text-white"
        }
      >
        {triggerLabel}
      </button>
      {open ? (
        <div
          role="dialog"
          className={`absolute z-50 w-60 rounded-lg border border-gray-700 bg-neutral-900 p-3 text-left font-normal normal-case shadow-xl shadow-black/40 ${PLACEMENT_CLASS[placement]}`}
        >
          {title ? <p className="text-xs font-semibold text-white">{title}</p> : null}
          <div className="mt-1 text-[11px] leading-relaxed text-gray-300">{children}</div>
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              onClick={dismiss}
              className="rounded-md border border-gray-600 px-2 py-0.5 text-[11px] text-gray-200 transition hover:bg-neutral-800"
            >
              Got it
            </button>
          </div>
        </div>
      ) : null}
    </span>
  );
}
