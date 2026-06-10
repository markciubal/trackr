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
