-- Trackr database schema. Run this in the Supabase SQL editor (or `supabase db push`).
-- Safe to re-run: uses IF NOT EXISTS / OR REPLACE / DROP-then-CREATE for policies.

-- 1) Profiles: one row per auth user, holds the Stripe customer link.
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text,
  stripe_customer_id text unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2) Subscriptions: mirrors Stripe subscription state. Written only by the
--    Stripe webhook (service-role key, which bypasses RLS).
create table if not exists public.subscriptions (
  id text primary key,                       -- Stripe subscription id (sub_...)
  user_id uuid not null references auth.users (id) on delete cascade,
  status text not null,                      -- active | trialing | past_due | canceled | incomplete | ...
  price_id text,
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists subscriptions_user_id_idx on public.subscriptions (user_id);

-- 3) Row Level Security. Users may read their own rows and update their own
--    profile; subscription writes happen server-side via the service-role key.
alter table public.profiles enable row level security;
alter table public.subscriptions enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

drop policy if exists "subscriptions_select_own" on public.subscriptions;
create policy "subscriptions_select_own" on public.subscriptions
  for select using (auth.uid() = user_id);

-- 4) Auto-create a profile row whenever a new auth user signs up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- 5) Targets catalog: the short id printed in each "smart target" QR resolves
--    to one of these rows (rings, SKU, owner, free/paid). Published targets are
--    world-readable so anyone scanning a printed target can open /t/{id}.
create table if not exists public.targets (
  id text primary key,                       -- short code embedded in the QR
  owner_id uuid references auth.users (id) on delete set null,
  name text,
  width_value numeric not null,
  height_value numeric not null,
  unit text not null default 'in',           -- mm | cm | in
  qr_size_value numeric not null,            -- printed QR side length, in `unit`
  scoring_id text,
  is_paid boolean not null default false,
  is_published boolean not null default true,
  created_at timestamptz not null default now()
);
create index if not exists targets_owner_id_idx on public.targets (owner_id);

-- Drill layout, resolved by id. The QR carries only the short id (kept minimal so
-- it scans from a distance), so a scanned drill target's zones are rebuilt from
-- the recipe stored here via scenario.ts zonesFromParam(recipe, palette_version).
alter table public.targets add column if not exists drill_recipe text;
alter table public.targets add column if not exists drill_palette_version int;

alter table public.targets enable row level security;

drop policy if exists "targets_select_published" on public.targets;
create policy "targets_select_published" on public.targets
  for select using (is_published = true or auth.uid() = owner_id);

drop policy if exists "targets_insert_own" on public.targets;
create policy "targets_insert_own" on public.targets
  for insert with check (auth.uid() = owner_id);

drop policy if exists "targets_update_own" on public.targets;
create policy "targets_update_own" on public.targets
  for update using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

drop policy if exists "targets_delete_own" on public.targets;
create policy "targets_delete_own" on public.targets
  for delete using (auth.uid() = owner_id);

-- 6) Admin flag: hand-set in the Supabase dashboard (update profiles set is_admin
--    = true where email = '...'). Gates the shot-classifier admin panel.
alter table public.profiles add column if not exists is_admin boolean not null default false;

-- 7) Shot-classifier models. Admins label patches in the admin panel, train a
--    small JSON model in the browser, and publish a row here. The single
--    published row is world-readable — that's how the trained classifier is
--    "pushed out" to every scanner (it loads the latest on startup).
create table if not exists public.classifier_models (
  id uuid primary key default gen_random_uuid(),
  version int not null,                       -- monotonically increasing
  kind text not null,                         -- model format tag (e.g. logreg-radial-v1)
  payload jsonb not null,                     -- serialized ModelJSON
  notes text,
  is_published boolean not null default false,
  created_by uuid references auth.users (id) on delete set null,
  created_at timestamptz not null default now()
);
create index if not exists classifier_models_published_idx
  on public.classifier_models (is_published, version desc);

alter table public.classifier_models enable row level security;

-- Anyone (including anonymous scanners) may read published models. Admins may
-- also read their own unpublished drafts.
drop policy if exists "classifier_models_select_published" on public.classifier_models;
create policy "classifier_models_select_published" on public.classifier_models
  for select using (
    is_published = true
    or exists (select 1 from public.profiles p where p.id = auth.uid() and p.is_admin)
  );

-- Only admins may insert/update/delete. Re-checked server-side before publish.
drop policy if exists "classifier_models_admin_write" on public.classifier_models;
create policy "classifier_models_admin_write" on public.classifier_models
  for all
  using (exists (select 1 from public.profiles p where p.id = auth.uid() and p.is_admin))
  with check (exists (select 1 from public.profiles p where p.id = auth.uid() and p.is_admin));

-- 8) Bullet annotations: a signed-in user saves a scanned session (the target
--    calibration + every detected/edited shot) to their account from the Save
--    step. These accumulate a labeled dataset of real shots-on-target for
--    training detectors/classifiers later — hence admins can read them all.
create table if not exists public.bullet_annotations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text,
  target jsonb,                               -- { widthInches, heightInches, pixelsPerInch, roi, ... }
  shots jsonb not null,                       -- array of { n, x, y, dpx, din, t, method, group, conf }
  shot_count int not null default 0,
  created_at timestamptz not null default now()
);
create index if not exists bullet_annotations_user_id_idx on public.bullet_annotations (user_id, created_at desc);

alter table public.bullet_annotations enable row level security;

-- Owners manage their own rows; admins may read everyone's (training corpus).
drop policy if exists "bullet_annotations_select_own_or_admin" on public.bullet_annotations;
create policy "bullet_annotations_select_own_or_admin" on public.bullet_annotations
  for select using (
    auth.uid() = user_id
    or exists (select 1 from public.profiles p where p.id = auth.uid() and p.is_admin)
  );

drop policy if exists "bullet_annotations_insert_own" on public.bullet_annotations;
create policy "bullet_annotations_insert_own" on public.bullet_annotations
  for insert with check (auth.uid() = user_id);

drop policy if exists "bullet_annotations_delete_own" on public.bullet_annotations;
create policy "bullet_annotations_delete_own" on public.bullet_annotations
  for delete using (auth.uid() = user_id);
