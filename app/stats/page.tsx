import { StatsView } from "./StatsView";

export const metadata = {
  title: "Statistics · Trackr",
  description:
    "Lifetime shooting statistics: every shot measured against where the bullet should have gone — zero error, wobble, stringing, trends, and per-drill breakdowns.",
};

export default function StatsPage() {
  return <StatsView />;
}
