import { TargetDesigner } from "./TargetDesigner";

export const metadata = { title: "Design a target · Trackr" };

// No account anywhere in this flow: the printed QR is fully self-contained
// (drill recipe + calibration ride inside the URL), so the designer needs no
// user context at all.
export default function NewTargetPage() {
  const baseUrl =
    process.env.NEXT_PUBLIC_TARGET_BASE_URL ||
    process.env.NEXT_PUBLIC_SITE_URL ||
    "https://trackr-c8dc72cd850c.herokuapp.com";
  return <TargetDesigner baseUrl={baseUrl} />;
}
