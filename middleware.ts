import { NextResponse, type NextRequest } from "next/server";
import { updateSession } from "@/app/lib/supabase/middleware";

export async function middleware(request: NextRequest) {
  // Printed QRs encode an uppercase path (HTTPS://TRKR.GG/T/{ID}) to stay in the
  // compact QR alphanumeric charset. Redirect the uppercase /T/ prefix to the
  // real lowercase route, preserving the (case-sensitive) id verbatim.
  const upper = request.nextUrl.pathname.match(/^\/T\/(.*)$/);
  if (upper) {
    const url = request.nextUrl.clone();
    url.pathname = `/t/${upper[1]}`;
    return NextResponse.redirect(url, 308);
  }
  return updateSession(request);
}

export const config = {
  matcher: [
    // Run on all routes except Next internals, PWA files, and static assets.
    "/((?!_next/static|_next/image|favicon.ico|manifest.webmanifest|sw.js|icon.svg|icon-maskable.svg|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
