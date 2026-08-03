# Trackr dry-fire hardware

## glock17-aim-flag.scad — bore-insert aim flag (Glock 17)

**Direct mount (default): one simple stem + the existing holder and card.**

- **Flag stem** — the rod that fits down the barrel (quick to seat; the
  small tilt slop vs a longer stem is a constant the software zeroing
  absorbs), a collar seating on the crown, and the shared coarse **base
  screw out the front**. The stem prints as a plain vertical rod — no
  delicate features.
- **Head** — `aim-flag-stem-universal-head.stl`, the **click nub pivoted
  45° downward** (solid neck + two flexing prongs with outward barbs) on a
  threaded socket. It screws onto ANY stem — fixed 9 mm, fixed .40, dial,
  or cage — in one turn, and jams snug on the socket's internal ceiling.
  **Print it once.**
- **Holder + Card** — the holder's joint pocket is a **through slot**: push
  the head's prongs in, they squeeze together, then the barbs clear the pad
  and **click over its top edge** — positive retention, no glue, removable
  by pinching the two barb tips together. The card rides at the canted
  attitude the pose solver likes, **pattern facing forward-up at 45°** —
  for a camera at or above screen height — below the sight line ahead of
  the muzzle.

The connector-based 4-part system is still in the .scad — set
`mount_style = "frame"` and render `connector` manually if you want it.

A webcam at the screen computes the card's full pose from the grid's skew and
scale (solvePnP) and back-computes where the gun is pointed — no laser
needed, with a continuous aim trace.

**SAFETY: dry-fire training accessory only. Verified-empty firearm, no
ammunition in the room. The insert blocks the muzzle.**

### Export the STL

1. Install [OpenSCAD](https://openscad.org/downloads.html) (free,
   Win/Mac/Linux).
2. Easiest: export every part (stem, card, card-inlays) as separate STLs in
   one go:

   ```powershell
   .\export-stls.ps1
   ```

   Or manually — set `render_part` in the file to `"stem"`, `"card"`,
   `"card-inlay"` (or `"print"` for a single-color everything plate) and:

   ```
   openscad -o glock17-aim-flag.stl glock17-aim-flag.scad
   ```

   **The card body and card inlay must always be exported as two separate
   files** — rendering them together unions them into one solid and the
   two-filament boundary is lost. They share the same origin and orientation,
   so the slicer drops them into perfect registration.

### Calibers

The **stem is the only caliber-specific part** — the holder and card are
shared. Two stems are exported:

- `glock17-aim-flag-stem.stl` — 9 mm Glock (G17 etc.), bore across lands
  ≈ 8.85 mm, collar ⌀10.6.
- `glock22-aim-flag-stem.stl` — .40 S&W Glock (G22/23), bore across lands
  ≈ 9.91 mm, collar ⌀11.4.

Other calibers: set `caliber` / `bore_land_d` / `collar_d` in the .scad
(calipers on YOUR bore and slide opening beat the typical numbers).

### Universal dial stem (print & assemble, experimental)

A twist-to-expand stem that fits **any bore from 9 mm up to about Ø10.2**
(9 mm, .357 SIG, 10 mm, .40 S&W). Four small parts in two files:

- `aim-flag-stem-universal.stl` — the **replate set**: collet SLEEVE, coned
  SPINE, and knurled DIAL, printed loose on one plate (~15 min). This is the
  part you re-print while tuning fit.
- `aim-flag-stem-universal-head.stl` — the **HEAD**: the 45° click-nub
  connector with a threaded socket. It **screws on and off, so print it
  once** and reuse it across every fit-test iteration of the set.

**Assemble** (no tools): push the spine's rod up through the sleeve from the
tip end — the flats keyed so it only slides, never spins — then spin the
dial onto the screw, and screw the head on until it jams snug against the
socket's internal ceiling. The head's final roll angle doesn't matter: you
aim the nub downward by rotating the whole stem in the bore before locking.

**Use** (verified-empty firearm, as always): slide it in until the sleeve's
knurled flange seats on the crown, point the nub down, pinch the flange, and
twist the dial until snug — it draws the spine's cone into the sleeve's
tapered mouth and expands the four fingers against the bore. A quarter turn
past first resistance is plenty. Twist back to release before pulling out.

**Print**: everything as oriented, 0.2 mm layers, PETG preferred (the
fingers flex ~0.9 mm; brittle PLA can crack). No supports anywhere; brim on
the spine (its bed face is only the Ø8.2 cone tip). Threads bind → raise
`thr_fit` to 0.65; grip weak at .40 → fingers are near max stretch, print
the dedicated G22 stem instead.

The dedicated fixed stems are lighter and stiffer — prefer them for a gun
you use regularly; the universal stem is for trying the system on whatever
is in the safe.

### Cork stem (universal, no threads, no mechanisms — RECOMMENDED)

Two parts, zero fasteners, nothing to bind: a rigid spine with the **click
nub printed integral** (the snap joint this project has already proven),
and a **TPU taper cork** that does all the gripping — exactly like a drum
plug or rubber chamber flag.

- `aim-flag-stem-cork.stl` — the spine (PETG): rod, barb ring, knurled
  grip flange, prong nub. Prints vertically, tip down, generous brim, no
  supports.
- `aim-flag-stem-cork-tpu.stl` — the pistol cork (any TPU; softer = easier
  on big bores). Ø8.5 → Ø11.5 over 25 mm, prints fat-end-down, solid.
- `aim-flag-stem-cork-tpu-rifle.stl` — the **rifle cork** for the same
  spine: Ø6.3 → Ø8.7, covering 6.5 mm–8 mm bores (6.5 Creedmoor, 6.8,
  7 mm-08, .308/.30-06, 7.62×39, 8 mm Mauser). Swap corks by pulling one
  off over the barb ring and popping the other on. The Ø5 rod sets the
  floor — for .22/.223, print a slim spine + cork: `cork_rod_d = 3.6`,
  `cork_barb_d = 4.6`, `cork_bore = 3.2`, cork Ø4.6 → Ø5.9, re-export
  both parts.

**Assemble once**: slide the cork onto the rod from the tip, **fat end
first** (small end pointing at the rod tip) — its bore pops over the barb
ring, which keeps it on the rod when you pull out; the flange is the
shoulder it wedges against. **Use**: verified-empty firearm,
push the stem in until the cone wedges snug — it stops at whatever depth
matches the caliber (9 mm seats shallow, .45 seats deep; zeroing absorbs
the difference). Twist-pull to remove. There is nothing to adjust and
nothing that can cross-thread, strip, or snap.

If the grip feels light in a clean, oily bore, wipe the bore or print the
cork one size up (`cork_d_tip = 9.0`); if the .45 fit runs out of cone,
raise `cork_d_big` to 12.

### Plug stem (universal, squeezed TPU gland)

The test-plug take, and the simplest of the three universals: a **small TPU
ring squeezed lengthwise between two rigid faces bulges out and seals
against the bore** — the same principle as a plumber's pipe test plug or a
bike seatpost expander. Rubber conforms to any bore, grips far better than
hard plastic, releases with zero set, and the expansion sits just past the
crown. Two files:

- `aim-flag-stem-plug.stl` — the RIGID parts (short stud, crown seat, knob,
  taper disc), PETG.
- `aim-flag-stem-plug-gland.stl` — the TPU ring (Ø8.4 × Ø5.2 × 9 mm, a
  five-minute print). Use the **softest TPU you have**: 95A grips
  9 mm/.357 easily, **85A recommended to reach .40** at light knob torque.
  Print it solid (3+ walls / 100 % infill).

**Assemble**: knob onto the base screw → stud through the seat from the
front (flats keyed in the throat) → gland onto the rod → taper disc onto
the tip until it jams → shared head onto the base. **Use**: insert until
the flange seats on the crown, aim the nub down, pinch the knurled flange,
tighten the knob — the disc squeezes the gland against the seat and it
barrels out against the bore. Snug is snug; there's nothing to break by
over-tightening except your patience. Unscrew and it slides straight out.

Total in-bore depth is only ~22 mm and there are no flexing plastic
features at all — the gland does all the conforming. If the grip tops out
before .40 with 95A TPU, print the gland a touch longer (`plug_gland_l`
= 11) or switch to 85A.

### Cage stem (universal, screws at the base)

The **molly-bolt** universal: instead of a cone wedging thin fingers, a
**knob at the base compresses a rib cage lengthwise so its four pre-bowed
ribs bulge outward** and grip the bore with broad mid-span pads. Two files
so the expandable part can print in its own material, plus the shared head:

- `aim-flag-stem-cage.stl` — the RIGID parts (spine, knob, taper disc), PETG.
- `aim-flag-stem-cage-flex.stl` — the CAGE alone: **print it in TPU (~95A)**
  if you have it — rubber ribs grip the bore dramatically better than hard
  plastic, can never crack no matter how hard you crank the knob, and seat
  gently. PETG works too (peak rib strain ~1 %).

- **CAGE** — knurled crown flange, four gently pre-bowed ribs (relaxed
  Ø8.4, necked at the ends so it always slips into a 9 mm bore), solid
  rear ring.
- **SPINE** — a bare double-ended stud: base screw, keyed rod (nothing
  rotates in the barrel), and a small tip thread. Both attachments screw
  on because a fat end on either side could never pass through the cage.
- **DISC** — the tapered tip: screws onto the tip thread and jams on the
  rod's shoulder. This is what the cage's rear ring bears against.
- **KNOB** — knurled base nut (identical thread to the dial stem's).
- **HEAD** — the click-nub connector is the SAME
  `aim-flag-stem-universal-head.stl` the dial stem uses: **print it once
  and share it** between both universal stems and every fit-test replate.

**Assemble, in order**: ① spin the knob onto the base screw and run it
down; ② slide the rod through the cage from the front, flats aligned in
the throat (the tip thread is sized to clear the throat and the rear ring);
③ screw the disc onto the tip until it jams; ④ screw the head onto the
base. **Use**: insert until the flange seats on the crown, aim the nub
down, pinch the knurled flange, tighten the knob — it pulls the spine
outward, the disc squeezes the cage, and the ribs bow out. Molly-bolt
leverage: **an eighth of a turn past snug is fully gripped**, and the hold
is strong at light finger torque. Unscrew and the ribs spring straight for
removal.

**Print**: everything as oriented, no supports; brim under the spine (it
stands on the base screw's end face). The cage prints flange-down in TPU
or PETG — the pre-bow makes the ribs buckle outward deterministically,
never inward. TPU notes: ~95A shore, 3+ perimeters so the thin ribs print
solid; a soft cage needs a bit more knob travel before it bites (the ribs
squash before they bow), and if grip still feels soft, drop `cage_wall`
to 1.0 so the ribs bow easier, or bump `cage_bow` to 0.6. Pick this over
the plug stem if you want an all-hard-plastic grip (no TPU at all in PETG
mode) or a longer engagement; the plug stem is otherwise the simpler,
gentler default.

### Fit-test first

Bores and printers both vary. Print the stem alone first
(`render_part = "stem"`, ~20 min):

1. It should slide into the (verified-empty) bore smoothly and sit without
   wobble, collar seated on the crown.
2. The four sacrificial ribs print slightly proud on purpose — sand them down
   to a snug slide fit. Too loose even before sanding → raise `rib_proud` or
   lower `fit`; won't start at all → raise `fit` by 0.05.
3. The dimensions marked `MEASURE` (bore across lands ≈ 8.85 mm, collar
   clearance inside the slide's muzzle opening) are typical G17 numbers —
   calipers beat defaults.
4. Alignment does NOT need to be perfect: the software zeroing step ("aim at
   the center dot, press a key") absorbs any constant offset. Snug and
   repeatable is the goal.

The head-to-holder joint is a snap fit: push the holder's slot onto the
head's prongs until the barbs CLICK over the pad's top edge — you'll feel and hear
it. Seated, the neck's end face and the barbs sandwich the pad with ~0.15 mm
of designed play, so there's no wobble and no glue. To remove, pinch the two
barb tips (they stick ~3 mm past the pad) and pull. Tuning: prongs won't
start into the slot → raise `snap_fit` to 0.2; the click feels loose →
raise `snap_barb` to 1.0; a prong snaps off (brittle PLA) → print the head
in PETG or widen `snap_slot` to 3.0 for an easier flex. The card still
slides into the holder and clicks past its own snap nubs — tune `card_fit`
(±0.05) and `nub_r` as needed.

### Print settings

- PETG or PLA+, 0.2 mm layers, 4 perimeters, ~40 % infill.
- **Flag stem**: prints vertically, bore tip DOWN, base screw at the top —
  a plain straight print, no supports. The footprint is only the rod's
  ~7 mm tip face, so use a generous brim (8–10 mm) and moderate speed; the
  collar's small overhang ring above the rod prints fine unsupported.
- **Head**: prints socket-down, no supports; one turn seats it on any stem.
- **Holder**: prints upright on its bottom wall, rails vertical. The joint
  pad's foot is flush with the bottom wall, so it stands on the buildplate
  too — no supports needed.
- **Card**: prints flat, pattern face up. Perfect surface, no supports.

### The fiducial pattern (required — a blank face cannot work)

The tracker measures the pattern's known geometry; without a pattern there is
nothing to track. Three card designs are supported — pick the matching mode
on the dry-fire page:

**B/W shape card — recommended (2 filaments).** Same quadrant layout as the
color card, but identity is carried by TOPOLOGY instead of hue: a solid
disk (0 holes), a ring (1 hole), a two-hole patch, and the small dot in the
white quadrant. Hole count survives any viewing angle and rotation, and
black-on-white is immune to white balance and low-light desaturation — no
color-sensitivity tuning at all. Four extra anchor dots at the pattern's
edge midpoints let the tracker least-squares refine the pose over the full
~65 mm baseline once locked (watch `anchors N/4` in the diagnostics).
Print it exactly like the checkerboard: slice
`glock17-aim-flag-card-shape-swap.stl` alone, add one "change filament at
layer" at z = 2.0 mm — white matte base, **bright red** matte above. The
detector looks ONLY for strong red (no darkness fallback — dark sweeps in
half of any room), so the shapes must be red, and nothing else in an
ordinary room will compete. (Or AMS:
`card.stl` + `card-inlay-shapes.stl` as one object, two filaments.) Needs
each ring hole to resolve (~4 px), so its max distance is roughly half the
color card's — move closer or bump camera resolution if `quadrant px` reads
low. Select "●◍◎ Shapes" on the dry-fire page.

**RGB color card (4 filaments).** Four big quadrants — facing the finished
card, you see GREEN top-left, RED top-right, BLUE bottom-right, and the bare
white quadrant with its blue confirmation dot bottom-left. (Don't sweat the
handedness: the tracker only needs the colors' mutual layout, which the STLs
fix.) Saturated color patches are rare in a room and the color IS the
correspondence, so detection is far more robust than the checkerboard (which
must pick its 8 tiles out of every dark blob in the scene), and the ~36 mm
quadrants track from roughly twice the distance. White base + red + green +
blue = exactly four AMS slots.

1. Import `card.stl` + `card-inlay-red.stl` + `card-inlay-green.stl` +
   `card-inlay-blue.stl` together ("one object with multiple parts" → Yes).
2. Assign matte filaments per part (the dot rides in the blue inlay).
3. Print flat. Avoid silk/gloss in all four slots.

**RGB color card, CLICK-IN version (no AMS — one color per print).** The
same pattern, but the four color patches and the dot are separate flat
pieces that press into pockets on the card body and **click in place** under
four small snap ridges per pocket — for single-color printers, or when you'd
rather not burn AMS purge waste on a 2-gram color change:

1. Print `glock17-aim-flag-card-click.stl` — the card body with four snap
   pockets — flat, face up, in **matte white** (the visible rims between
   tiles then read as background to the camera).
2. Print `glock17-aim-flag-card-click-tile.stl` **three times**: once each
   in matte red, green, and blue.
3. Print `glock17-aim-flag-card-click-tile-white.stl` (the tile with the
   dot hole) in matte white, and `glock17-aim-flag-card-click-dot.stl` in
   matte blue. All five inserts print flat in minutes.
4. Assemble: press each tile into a pocket, chamfered face down, until it
   snaps flat under the ridges. Then press the blue dot into the white tile
   **from the front**, flange out — it seats snugly in the front-face
   counterbore, flush with the tile, and the pocket floor behind it stops
   it from ever pushing through. Facing the finished card: GREEN top-left,
   RED top-right, BLUE bottom-right, white tile with the blue dot
   bottom-left.

To swap a tile, push a straightened paperclip through the poke-hole in the
pocket's back and it pops out. Tiles too tight to click in → raise
`insert_fit` to 0.2; tiles rattle → drop it to 0.1 or raise `click_bite`.

**Checkerboard card** (single dark filament, zero-waste single-swap option —
see below). Works, but is the fussier of the two for the camera.

The card body is a flat slab; the checker tiles (plus the round
symmetry-breaking dot) are 0.6 mm of **raised layers on top of the card
face** — wherever the second color isn't, the face reads as a recess. Use
**matte** filaments for both colors (silk/gloss glare kills corner
detection). Two ways to print it, both flat and face up:

**Zero-waste single swap — recommended.** The design guarantees that the
ONLY geometry above the 2.0 mm plate top is the tile layer, so one filament
change does the whole job:

1. Slice `glock17-aim-flag-card-swap.stl` (the merged single mesh) on its
   own, flat as oriented.
2. In the slicer's preview, right-click the layer slider at **z = 2.0 mm**
   (= `card_t`) → "Change filament" / "Add color change".
3. Print: base in light matte, everything above the change in dark matte.
   One swap, no purge tower, no wipe waste — the recesses form themselves
   because no body geometry exists above the swap plane.

**AMS multi-part alternative.** Import `card.stl` and `card-inlay.stl`
together ("one object with multiple parts" → Yes) and assign a filament to
each part. Identical result; the AMS just purges once at the color change.

Either way the card slides into the frame with the tiles inside the lip
clearance (pattern inset 4 mm > lip overlap 3 mm — nothing rubs). The 0.6 mm
tile height is 3 layers at 0.2 mm, opaque even with translucent-ish
filament, and the slight relief doesn't bother the corner detector at the
45° viewing angle.

**Single-color fallback.** Set `printed_pattern = false` to get a card with
one big recess instead, and glue in a paper print of a 4 × 4 checkerboard
(18 mm squares) at exactly 100 % scale on matte paper.

Either way: keep the pattern **coarse** (4 × 4; never finer than 5 × 5) — do
NOT use an actual QR code, whose modules are far too small to resolve across
a room on a webcam. The dot in one white cell is deliberate: a plain
checkerboard looks identical rotated 180°, and the dot removes that
ambiguity for the solver.
