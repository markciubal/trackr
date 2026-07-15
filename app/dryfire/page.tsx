import { DryFireTrainer } from "./DryFireTrainer";

export const metadata = {
  title: "Dry-fire trainer · Trackr",
  description:
    "Point your (verified-empty) gun at the screen: a webcam tracks the printed muzzle flag, shows your live wobble trace, and scores called drills from the trigger click.",
};

export default function DryFirePage() {
  return <DryFireTrainer />;
}
