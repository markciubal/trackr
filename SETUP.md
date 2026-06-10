# Trackr — Auth & Billing Setup

Trackr now has a Supabase (auth + Postgres) + Stripe (subscription) backend wired in.
**Until you add the env vars below, all of this stays dormant** — the tool runs exactly
as before with full features and no sign-in. Gating only switches on once configured.

Model: **freemium** — anonymous/free users get a limited tier; **Pro** is a recurring
Stripe subscription.

---

## 1. Install dependencies

```bash
npm install
```

(Adds `@supabase/ssr`, `@supabase/supabase-js`, `stripe`, `server-only`.)

## 2. Supabase

1. Create a project at https://supabase.com.
2. **SQL Editor → New query →** paste & run [`supabase/schema.sql`](supabase/schema.sql).
   This creates `profiles` + `subscriptions`, RLS policies, and a trigger that makes a
   profile row on signup.
3. **Project Settings → API**, copy into `.env.local`:
   - Project URL → `NEXT_PUBLIC_SUPABASE_URL`
   - `anon` public key → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `service_role` secret key → `SUPABASE_SERVICE_ROLE_KEY` (server-only; never expose)
4. **Authentication → URL Configuration:**
   - Site URL: `http://localhost:3000` (and your prod URL later)
   - Redirect URLs: add `http://localhost:3000/auth/callback` **and** your prod
     equivalent `https://yourdomain.com/auth/callback`
5. **Authentication → Providers** — Trackr's login is OAuth + passwordless magic link
   (no passwords). Enable the providers you want the buttons to work:
   - **Google:** create an OAuth client in Google Cloud Console, set the authorized
     redirect URI to the Supabase callback shown in the provider panel
     (`https://<project-ref>.supabase.co/auth/v1/callback`), then paste the client ID/secret.
   - **GitHub:** create an OAuth App (Settings → Developer settings), use the same
     Supabase callback URL, paste the client ID/secret.
   - **Email (magic link):** enabled by default — `signInWithOtp` emails a one-click link
     that lands on `/auth/callback`. New users are auto-created on first sign-in.
   - To add another provider, drop a button into `app/login/LoginForm.tsx` (the `Provider`
     union) — the callback already handles every OAuth `code`.

## 3. Stripe

1. **Products →** create a "Trackr Pro" product with a **recurring** monthly price (and an
   annual price if you want). Copy the price IDs (`price_...`):
   - monthly → `STRIPE_PRICE_PRO_MONTHLY`
   - annual → `STRIPE_PRICE_PRO_ANNUAL` (optional)
2. **Developers → API keys →** Secret key → `STRIPE_SECRET_KEY`.
3. Webhook signing secret → `STRIPE_WEBHOOK_SECRET`:
   - **Local:** `stripe login` then
     `stripe listen --forward-to localhost:3000/api/stripe/webhook`
     — it prints a `whsec_...` to use.
   - **Production:** **Developers → Webhooks → Add endpoint** → `https://YOUR_DOMAIN/api/stripe/webhook`,
     subscribe to `checkout.session.completed`, `customer.subscription.created`,
     `customer.subscription.updated`, `customer.subscription.deleted`. Copy its signing secret.
4. (Prod) **Billing → Customer portal →** activate it so "Manage billing" works.

## 4. Environment

```bash
cp .env.example .env.local   # then fill in every value
```

Restart `npm run dev` after editing env.

## 5. Verify

- `/login` — create an account / sign in.
- `/account` — shows plan (Free), **Upgrade to Pro** → Stripe Checkout (use test card
  `4242 4242 4242 4242`). After paying, the webhook flips you to **Pro**.
- **Manage billing** → Stripe Customer Portal (cancel/update).
- The slim header (top of every page) shows your plan + a link to `/account`.

---

## How it fits together

| Piece | File |
| --- | --- |
| Supabase clients (browser/server/admin/middleware) | `app/lib/supabase/*` |
| Session refresh on every request | `middleware.ts` |
| Stripe server client | `app/lib/stripe.ts` |
| Checkout / Portal / Webhook | `app/api/stripe/*` |
| Entitlements (free/pro/anonymous + free-tier caps) | `app/lib/entitlements.ts` |
| Auth pages & actions | `app/login`, `app/account`, `app/auth/*` |
| DB schema | `supabase/schema.sql` |

**Entitlement source of truth:** `getEntitlement()` (server). It returns
`{ plan, isPro, userId, email, configured }`. Use it in Server Components / route
handlers to gate features. Free-tier limits live in `FREE_TIER` in
`app/lib/entitlements.ts`.

## Not done yet (next step)

In-app **feature gating** isn't wired into the tool (`app/page.tsx`) yet — the
entitlement plumbing is ready, but no specific feature is locked behind Pro. Decide which
features are Pro-only (e.g. export, unlimited saved sessions, >N scans/day) and we'll gate
them using `getEntitlement()` + an upgrade prompt. Persisting users' sessions/results to
Supabase (instead of localStorage) is the other natural follow-up.
