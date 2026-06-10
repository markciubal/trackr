"use client";

import { useEffect, useRef, useState } from "react";
import { ONBOARDING_TIPS, markTipsSeen, resetOnboarding } from "@/app/lib/onboarding";

// The onboarding interface: one unobtrusive "Guide" button that opens a popover
// with a welcome + the numbered steps, highlighting the user's current step.
// Stays closed by default — the user opens it on demand; "Restart tour" reopens.
export function OnboardingGuide({ currentStep }: { currentStep: number }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

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

  const welcome = ONBOARDING_TIPS.find((tip) => tip.id === "welcome");
  const steps = ONBOARDING_TIPS.filter((tip) => typeof tip.step === "number");

  const dismiss = () => {
    markTipsSeen(ONBOARDING_TIPS.map((tip) => tip.id));
    setOpen(false);
  };
  const restart = () => {
    resetOnboarding();
    setOpen(true);
  };

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex min-h-8 items-center gap-1 rounded-full border border-gray-600 px-2.5 text-xs text-gray-200 transition hover:bg-neutral-800 sm:min-h-0 sm:py-1"
      >
        <span aria-hidden="true" className="font-semibold">
          ?
        </span>
        Guide
      </button>

      {open ? (
        <div className="absolute right-0 top-full z-50 mt-2 w-72 max-w-[calc(100vw-2rem)] rounded-xl border border-gray-700 bg-neutral-950 p-3 text-left shadow-2xl shadow-black/50">
          {welcome ? (
            <>
              <p className="text-sm font-semibold text-white">{welcome.title}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-gray-300">{welcome.body}</p>
            </>
          ) : null}

          <ol className="mt-3 space-y-1.5">
            {steps.map((tip) => {
              const active = tip.step === currentStep;
              return (
                <li
                  key={tip.id}
                  className={`rounded-md border px-2 py-1.5 ${
                    active ? "border-amber-300/60 bg-amber-500/10" : "border-gray-800"
                  }`}
                >
                  <p className={`text-[11px] font-semibold ${active ? "text-amber-100" : "text-gray-200"}`}>
                    {tip.title}
                    {active ? <span className="ml-1 text-[10px] font-normal text-amber-300/80">— you&apos;re here</span> : null}
                  </p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-gray-400">{tip.body}</p>
                </li>
              );
            })}
          </ol>

          <div className="mt-3 flex items-center justify-between">
            <button type="button" onClick={restart} className="text-[11px] text-gray-400 transition hover:text-gray-200">
              Restart tour
            </button>
            <button
              type="button"
              onClick={dismiss}
              className="rounded-md border border-sky-400/40 bg-sky-500/15 px-2.5 py-1 text-[11px] font-medium text-sky-100 transition hover:bg-sky-500/25"
            >
              Got it
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
