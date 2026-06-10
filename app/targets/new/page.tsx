import { isSupabaseConfigured } from "@/app/lib/supabase/config";
import { getCurrentUser } from "@/app/lib/supabase/server";
import { TargetDesigner } from "./TargetDesigner";

export const metadata = { title: "Design a target · Trackr" };

export default async function NewTargetPage() {
  const user = isSupabaseConfigured() ? await getCurrentUser() : null;
  const baseUrl =
    process.env.NEXT_PUBLIC_TARGET_BASE_URL || process.env.NEXT_PUBLIC_SITE_URL || "https://trkr.gg";
  return <TargetDesigner canSave={Boolean(user)} baseUrl={baseUrl} />;
}
