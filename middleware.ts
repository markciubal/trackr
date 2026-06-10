import { type NextRequest } from "next/server";
import { updateSession } from "@/app/lib/supabase/middleware";

export async function middleware(request: NextRequest) {
  return updateSession(request);
}

export const config = {
  matcher: [
    // Run on all routes except Next internals, PWA files, and static assets.
    "/((?!_next/static|_next/image|favicon.ico|manifest.webmanifest|sw.js|icon.svg|icon-maskable.svg|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
