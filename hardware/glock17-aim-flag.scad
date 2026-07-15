// ============================================================================
// Trackr dry-fire aim flag — Glock 17 BORE INSERT version
// ============================================================================
// DIRECT MOUNT (default, mount_style = "direct") — the stem plugs straight
// into the existing HOLDER; no connector.
//   1. FLAG STEM — the rod that slides INTO the muzzle (like a chamber
//      flag; quick to seat, and the software zeroing absorbs the small
//      tilt slop), collar on the crown, and the shared coarse BASE SCREW
//      out the front: the universal HEAD (with the 45°-down click nub)
//      screws on in one turn — one head serves every stem, and the stem
//      itself prints as a plain vertical rod.
//   2. HEAD + HOLDER + CARD — the holder's joint pocket is a THROUGH slot;
//      the head's prongs pass through and the barbs CLICK over the pad's
//      top edge. Positive retention, removable by pinching the barb tips.
//      The card's pattern faces forward-UP at 45° — camera at or above
//      screen height.
//
// FRAME MOUNT (mount_style = "frame") — the older 4-part system:
//   STEM (straight/tilted tang) + CONNECTOR (arm, cant_dir attitude) +
//   HOLDER + slide-in CARD.
//
// A webcam at the screen sees the card; the grid's known size gives the full
// 6-DOF pose (solvePnP) and the bore ray back-computes the aim point. The 45°
// cant is essential: facing the camera square-on is the one orientation where
// tilt (your aim) is nearly unobservable — canted, it's first-order visible.
// `cant_dir` picks which way the 45° leans: "under" tucks the card back
// beneath the barrel (pattern faces forward-UP — for a camera at/above screen
// height), "forward" ramps it ahead of the muzzle (pattern faces forward-DOWN).
//
// SAFETY: dry-fire training accessory only. Use only with a verified-empty
// firearm, no ammunition in the room. The insert blocks the muzzle.
//
// Export:  openscad -o <name>.stl glock17-aim-flag.scad   (set render_part)
//
// FIT: dimensions marked MEASURE are typical G17 numbers; barrels and
// printers vary. Print the stem first and tune `fit`; print the card + a
// frame and tune `card_fit` until the card slides smoothly and the nubs
// click. Mounting error is absorbed by the software zeroing step.
// ============================================================================

// What to render:
//   "assembled"  — preview of everything together
//   "print"      — stem + connector + holder + card plated flat (single color)
//   "stem"       — bore stem, print orientation
//   "stem-universal" — dial-expanding stem, REPLATE SET: collet sleeve +
//                  coned spine + knurled dial, printed loose and assembled;
//                  twisting the dial grips any bore Ø8.8–~10.2 (9 mm–.40)
//   "stem-universal-head" — the screw-on click-nub connector (print ONCE;
//                  it survives every fit-test replate of the set above)
//   "stem-cage"  — molly-bolt universal stem, RIGID parts plate (spine +
//                  knob + taper disc, PETG); a base knob compresses the
//                  rib cage so it bulges out and grips any bore from 9 mm up
//   "stem-cage-flex" — the expandable CAGE alone, as its own file so it can
//                  print in TPU (95A shore — grippier and unbreakable;
//                  PETG also works)
//   "stem-plug"  — squeezed-gland universal stem, RIGID parts plate (short
//                  stud + crown seat + knob + taper disc, PETG); the base
//                  knob squeezes a TPU ring until it bulges and seals
//                  against any bore from 9 mm up
//   "stem-plug-gland" — that TPU ring, alone (print in the softest TPU
//                  available)
//   "connector"  — stem socket + arm + 45° head, print orientation
//   "holder"     — card-holding object, print orientation
//   "card"       — pattern card BODY, printed flat, pattern face up
//   "card-inlay" — the checker tiles, SAME orientation as "card"
//   "card-swap"  — card + tiles merged into ONE mesh, for the ZERO-WASTE
//                  single-swap technique (see below)
//   "card-click"            — card BODY with four snap pockets, for the
//                             CLICK-IN color card (single-color printers:
//                             print each color as its own part, press in)
//   "card-click-tile"       — one square color tile (print 1× each in red,
//                             green, blue)
//   "card-click-tile-white" — the white tile, with the dot hole + counterbore
//   "card-click-dot"        — the blue confirmation dot (flanged disc)
//
// Two ways to print the two-color card, both flat and face up:
//
//  A) ZERO-WASTE single swap (recommended): export "card-swap" (one STL).
//     The only geometry above the 2.0 mm plate top is the tile layer, so add
//     ONE "change filament at layer" in the slicer at z = 2.0 mm (card_t).
//     Base prints in color A, everything above is tiles-only in color B, and
//     the light cells are recesses. One swap, no purge tower, no waste.
//
//  B) AMS multi-part: export "card" and "card-inlay" as two STLs, import both
//     as ONE object (they share the origin), assign a filament to each. Same
//     result, but the AMS purges on the color change.
render_part = "print";

// ---- Fit & quality ---------------------------------------------------------
fit      = 0.15;  // per-side clearance, gun-contact + tang surfaces
card_fit = 0.25;  // clearance around the card in its frame slot
$fn = 96;

// ---- Mount style -----------------------------------------------------------
//   "direct" — stem + holder + card, and the stem CLICKS into the holder
//     (default). The stem's 45°-down nub is a two-prong fork with outward
//     barbs; the holder's joint pocket is a THROUGH slot. Push the prongs
//     in, they flex together, then the barbs clear the pad and CLICK over
//     its top edge — positive retention, no glue, removable by pinching
//     the barb tips. The holder carries the card at the 45° cant, pattern
//     facing forward-UP — for a camera at or above screen height.
//   "frame"  — the original 4-part system (stem/connector/holder/card).
mount_style = "direct";
stem_screw_l = 13; // the fixed stems' base screw: the SAME shared thread as
                  //   the universal stems, so the one screw-on HEAD
                  //   (aim-flag-stem-universal-head.stl) serves every stem —
                  //   and the stem prints as a plain vertical rod, no
                  //   delicate 45° fork
neck_len  = 10;   // solid standoff between collar and prongs (its end face
                  //   is the depth stop; keeps the holder off the slide)
snap_fit  = 0.15; // per-side prong clearance in the pocket
snap_barb = 0.8;  // how far each barb hooks over the pad's top edge
snap_slot = 2.4;  // gap between the prongs (flex room for the click)

// ---- Caliber & bore/muzzle (MEASURE on your gun) ----------------------------
// The STEM is the only caliber-specific part — the holder and card don't
// change. "9mm" fits G17-pattern bores; "40" fits .40 S&W Glocks (G22/23).
caliber      = "9mm"; // "9mm" | "40"
bore_land_d  = (caliber == "40") ? 9.91 : 8.85;
                      // MEASURE: Glock polygonal bore across lands
                      // (9 mm ≈ 8.85, .40 S&W ≈ 9.91)
stem_len     = 20;    // engagement depth (G17 barrel ≈ 114 mm — keep ≤ 95;
                      // 20 mm is a quick slip fit — short engagement means
                      // more tilt slop per clearance, so keep the ribs SNUG;
                      // software zeroing absorbs any constant tilt)
collar_d     = (caliber == "40") ? 11.4 : 10.6;
                      // MEASURE: must clear the slide's muzzle opening; seats
                      // on the barrel crown (barrel OD ≈ 10.9 mm on a G17,
                      // ≈ 12 mm on a G22 — stay a little under it)
collar_t     = 2.5;

// Sacrificial fit ribs on the stem: print slightly proud, sand to fit.
use_ribs   = true;
rib_count  = 4;
rib_proud  = 0.18;
rib_width  = 1.2;

// ---- Tang joint (stem → frame) ----------------------------------------------
// Rectangular so the frame can only mount hanging DOWN (a 180° flip is
// visually obvious). Friction fit; a drop of CA glue makes it permanent.
tang_w   = 8;
tang_h   = 6;
tang_len = 14;

// ---- Holder / connector geometry -----------------------------------------------
socket_wall = 4;
// Card attitude — the connection point rotated in 45°/90° steps:
//   "flat"    — card lies HORIZONTAL under the muzzle, pattern facing
//               straight UP, fully exposed as a face to a camera looking
//               down. The tang inserts horizontally, so gravity no longer
//               seats the joint: GLUE IT (CA on tang + pad face). (default)
//   "under"   — card tucks BACK beneath the barrel/frame at 45°, pattern
//               facing forward-up: camera at/above screen height.
//   "forward" — original: card ramps forward of the muzzle, pattern facing
//               forward-down 45°: for a camera below the aim line.
cant_dir    = "flat";
arm_drop    = (cant_dir == "flat") ? 45
            : (cant_dir == "under") ? 85 : 58; // bore axis → card plane
                    // ("under" is longer so the swept-back card clears the
                    //  frame's dust cover, ~20 mm below the bore; "flat" only
                    //  needs to hang below the slide, so it stays compact)
arm_width   = 14;
// "under" mode only: the arm hangs between the camera and the card, shadowing
// a vertical strip at the card's center. Kept narrow (±5 mm) so the strip
// rides the quadrant boundary — the four color patches and the dot (x ≈ +18)
// stay unobstructed. Dry fire has no recoil, so 10 mm is plenty stiff.
arm_width_u = 10;
arm_depth   = 8;
cant_deg    = 45;   // card lean relative to the bore — keep 45
rail_wall   = 2.5;  // side wall thickness beside the card
lip_w       = 3;    // how far the front lips overlap the card's edges
lip_t       = 1.2;  // front lip thickness
back_t      = 1.8;  // backing panel behind the card
bottom_wall = 2.5;  // closed bottom stop
nub_r       = 0.9;  // snap nub radius (protrudes ~nub_r/2 into the slot)

// Joint between the CONNECTOR (stem socket + arm + 45° head) and the HOLDER
// (the card-holding object): a rectangular tang on the connector slides UP
// into a blind pocket on the holder's back — gravity seats it, and both the
// card insertion force and recoil-free dry fire keep it seated. CA to lock.
joint_w    = 10;
joint_t    = 3.5;
joint_len  = 16;
joint_fit  = 0.2;
joint_wall = 3;    // pad material beside the pocket
pad_z0     = -bottom_wall; // pocket entrance at the holder's BOTTOM face: the
                   // pad's foot is coplanar with the bottom wall, so the
                   // upright print stands on it directly — no supports, and
                   // the seating math shifts the holder out along the 45°
                   // axis automatically (away from the slide — safe side)

// ---- Card & fiducial pattern ---------------------------------------------------
// The pattern is NOT optional: the pose solver tracks the checker corners,
// and their known physical spacing is what turns image skew/scale into real
// orientation/distance. Keep it COARSE and high-contrast.
card_w      = 80;
card_h      = 80;
card_t      = 2;
pattern_rim = 4;     // pattern inset from the card edge (> lip_w so the
                     // frame lips never cover pattern cells)
pattern_depth = 0.6; // pocket depth = 3 layers at 0.2 mm (opaque color change)
printed_pattern = true; // true → checker pockets + inlay (AMS two-color)
                        // false → one big recess for a glued paper grid
pattern_cols = 4;    // 4 × 4 checkerboard → 18 mm cells at the defaults
marker_cell = [0, 3]; // [col, row] white cell that gets the symmetry-break dot

// ---- Derived -------------------------------------------------------------------
// Angle between the bore rod and the tongue. In "flat" mode the tongue exits
// at 135° to the rod (45° downward) so the joint itself starts the "\" of
// the \_ profile; the other modes keep the straight (180°) tongue.
tang_tilt    = (cant_dir == "flat") ? 45 : 0;
// Direct mount: click-together prongs sized for the holder's through slot.
pocket_w     = joint_w + 2 * joint_fit;           // the pocket cross-section
pocket_t     = joint_t + joint_fit;
nub_w        = pocket_w - 2 * snap_fit;           // fork width in the slot
nub_t        = pocket_t - 2 * snap_fit;
prong_w      = (nub_w - snap_slot) / 2;           // each prong's width
pad_through  = joint_len + joint_fit + 2.4;       // pad height above the mouth
prong_len    = pad_through + 0.15;                // neck face → barb latch faces
barb_len     = 3;                                 // barb ramp past the latch
// The pocket mouth seats against the neck's end face.
seat_off     = neck_len;
// Nub frame origin, gun-frame y. The nub lives on the screw-on HEAD: head
// bottom face sits at collar + (screw − 10 mm socket) when jammed on its
// ceiling, the head block is 17 tall, and the nub assembly's neck offset
// is .5·sin45 above its mounting face.
head_seat_y  = collar_t + stem_screw_l - 10;       // head bottom face
nub_o_y      = head_seat_y + 17 + (pocket_t + 3) * 0.3536;
// Holder placement for the assembled preview: pocket mouth at the neck
// face along the 45° axis, card face turned up-forward. Rotation is
// rotate([135, 0, 180]).
pocket_cy    = -(back_t + card_t + card_fit) - pocket_t / 2; // pocket center, holder-local y
seat_sy      = nub_o_y + 0.7071 * seat_off;
seat_sz      = -0.7071 * seat_off;
dm_hold_ty   = seat_sy - 0.7071 * (pocket_cy + pad_z0);
dm_hold_tz   = seat_sz - 0.7071 * (pocket_cy - pad_z0);
stem_d       = bore_land_d - 2 * fit;
socket_w     = tang_w + 2 * fit;
socket_h     = tang_h + 2 * fit;
socket_out_w = socket_w + 2 * socket_wall;
socket_out_h = socket_h + 2 * socket_wall;
socket_len   = tang_len + 1;
cell         = (card_w - 2 * pattern_rim) / pattern_cols;
slot_w_in    = card_w + 2 * card_fit;          // slot interior width
slot_d_in    = card_t + card_fit;              // slot interior depth (front-back)
frame_w      = slot_w_in + 2 * rail_wall;
frame_depth  = back_t + slot_d_in + lip_t;     // back panel → lip front
flag_anchor  = [0, socket_len, -arm_drop];     // card bottom edge, below bore
// Holder/card orientation: "flat" lays the card horizontal (local height →
// downrange, pattern normal → straight up); +cant pitches the card top BACK
// under the gun ("under"), −cant pitches it forward ("forward").
flag_rot     = (cant_dir == "flat") ? [90, 0, 180]
             : (cant_dir == "under") ? [cant_deg, 0, 0] : [-cant_deg, 0, 0];
back_face    = -(back_t + slot_d_in);          // holder back panel outer plane
pad_w        = joint_w + 2 * joint_fit + 2 * joint_wall;
pad_t        = joint_t + joint_fit + 2.4;      // pocket depth + rear wall
head_h       = 8;                              // connector head below the pad
arm_bottom   = -(arm_drop - 10);               // arm stops short of the holder
arm_y0       = socket_len - arm_depth + 2;     // arm front face under the socket
// "under" mode arm routing: the card sweeps BACK, so a straight drop would
// pass through it. The arm instead runs diagonally down-FORWARD past the
// card's bottom-front corner (~y 17.6 at z ≈ −86), then the gusset crosses
// back underneath the card's bottom edge to the head.
under_foot_y = 21;                             // foot block front of that corner
under_foot_z = -(arm_drop + 7);                // just below the card bottom edge
// "flat" mode arm routing: the card extends FORWARD horizontally, so the arm
// drops vertically BEHIND its rear edge (y 1..9 — clear of the collar above
// and the holder's rear wall in front) and the gusset reaches down-forward
// to the head hanging under the card's rear portion. Nothing crosses above
// the pattern, so the face stays fully exposed.
flat_arm_y0  = 1;
flat_foot_z  = -(arm_drop - 6);

// ============================================================================
// Transition cone + 45°-down click nub, in the gun frame (bore axis Y,
// forward = +y) with the mounting face at y = 0. Shared by every stem.
module click_nub_assembly(base_d) {
  // Transition cone reinforces the nub's root against the cantilevered
  // holder (and keeps the stack self-supporting in print).
  rotate([-90, 0, 0]) cylinder(d1 = base_d, d2 = 5.5, h = 2.6);
  // The click nub, pivoted 45° DOWNWARD: a solid neck (its end face is
  // the depth stop against the pocket mouth), then two flexing prongs
  // that pass through the holder's slot and CLICK their barbs over the
  // pad's top edge.
  translate([0, (pocket_t + 3) * 0.3536, 0]) rotate([-45, 0, 0]) {
    // Neck: flush with the prongs' TOP face (the card-holder side), so
    // nothing stands proud of the fork on the side that slides along the
    // holder's back panel — the slot's front wall is that panel, and any
    // material above the prong plane would block insertion. The neck's
    // extra bulk (strength + depth stop) hangs on the back side only.
    translate([-(pocket_w + 3) / 2, 0, nub_t / 2 - (pocket_t + 3)])
      cube([pocket_w + 3, neck_len, pocket_t + 3]);
    for (sx = [-1, 1]) {
      // Prong body (runs past the latch plane to carry the barb ramp).
      translate([sx * (snap_slot / 2 + prong_w / 2) - prong_w / 2, neck_len, -nub_t / 2])
        cube([prong_w, prong_len + barb_len, nub_t]);
      // Barb: flat latch face toward the neck, ramped toward the tip so
      // insertion squeezes the prongs together until the click.
      hull() {
        translate([sx * (nub_w / 2 + snap_barb / 2) - snap_barb / 2, neck_len + prong_len, -nub_t / 2])
          cube([snap_barb, 0.1, nub_t]);
        translate([sx * nub_w / 2 - 0.05, neck_len + prong_len + barb_len - 0.1, -nub_t / 2])
          cube([0.1, 0.1, nub_t]);
      }
    }
  }
}

// ============================================================================
// PART 1 — bore stem. Bore axis = Y; muzzle crown plane at y = 0; stem runs
// into the barrel (−Y), collar + tang out the front (+Y).
module stem() {
  rotate([90, 0, 0]) union() {
    cylinder(d = stem_d, h = stem_len - 2);
    translate([0, 0, stem_len - 2]) cylinder(d1 = stem_d, d2 = stem_d - 1.6, h = 2);
  }
  // Sacrificial fit ribs, embedded 1 mm, standing rib_proud off the surface.
  if (use_ribs)
    for (i = [0 : rib_count - 1])
      rotate([0, i * 360 / rib_count, 0])
        translate([-rib_width / 2, -(stem_len - 6), stem_d / 2 - 1])
          cube([rib_width, stem_len - 10, 1 + rib_proud]);
  // Collar: seats on the crown, sets depth, keeps the stem concentric.
  rotate([-90, 0, 0]) cylinder(d = collar_d, h = collar_t);
  if (mount_style == "direct") {
    // Base screw out the front: the shared universal head screws on (one
    // turn to jam). Small transition disc blends collar → screw core.
    translate([0, collar_t, 0]) rotate([-90, 0, 0]) {
      cylinder(d1 = collar_d, d2 = thr_core, h = 1.5);
      uni_helix(stem_screw_l);
    }
  } else {
    // Transition cone collar → tang (self-supporting printed tang-down).
    translate([0, collar_t, 0]) rotate([-90, 0, 0]) cylinder(d1 = collar_d, d2 = 5.5, h = 2.6);
    // Tang (rectangular key) out the front — tilted down by tang_tilt so the
    // rod-to-tongue angle is 135° in "flat" mode (tail stays embedded in the
    // collar; the tilt pivots on the collar's front face).
    translate([0, collar_t, 0]) rotate([-tang_tilt, 0, 0])
      translate([-tang_w / 2, 0, -tang_h / 2])
        cube([tang_w, tang_len, tang_h]);
  }
}

// ============================================================================
// PART 1b — UNIVERSAL DIAL STEM (print & assemble, experimental).
//
// Four small parts. The HEAD — the click-nub connector — SCREWS ON AND OFF,
// so it prints ONCE; the three fit-critical parts replate in minutes while
// you tune clearances:
//   SLEEVE — slotted collet tube, relaxed Ø8.4, slips into any bore from
//            9 mm up; its knurled flange seats on the crown.
//   SPINE  — expansion cone at the tip, spline-flatted rod (keyed to the
//            sleeve so nothing rotates in the barrel), helical screw on top.
//   DIAL   — knurled nut riding the screw, bearing on the sleeve flange:
//            twisting it draws the cone into the sleeve's tapered mouth and
//            expands the fingers until they grip the bore — Ø8.4 relaxed →
//            ~Ø10.2 max (9 mm, .357 SIG, 10 mm, .40 S&W).
//   HEAD   — blind threaded socket + the standard 45° click nub; jams snug
//            on the socket's conical ceiling. Its roll angle doesn't matter:
//            you aim the nub by rotating the WHOLE stem in the bore before
//            tightening the dial.
//
// Assembly: push the spine's rod up through the sleeve from the tip end
// (flats aligned), spin the dial onto the screw, screw the head on snug.
// Render "stem-universal" (sleeve + spine + dial plate) and
// "stem-universal-head" (once).
uni_cone_d1   = 8.2;   // cone fat end (must still enter a 9 mm bore)
uni_cone_d2   = 4.9;   // cone tail = rod diameter
uni_cone_h    = 8;
uni_rod_flat  = 4.0;   // spline flats across the rod (anti-rotation key)
uni_sleeve_od = 8.4;   // relaxed OD — slips into the smallest (9 mm) bore
uni_mouth_id  = 6.4;   // tapered tip mouth the cone wedges into
uni_slot_w    = 2.2;   // 4 slots → 4 fingers
uni_fit       = 0.35;  // assembly clearances (keyed bore + threads)
// The shared screw (all stems' base, knob, head): MASSIVE round thread — a
// Ø4 helical lobe at pitch 10 with a 6 mm gap between turns and its own
// generous clearance — sized from print testing: anything finer binds on
// seam blobs and hole shrinkage.
thr_core      = 7.6;   // screw: core + one helical lobe rib (round thread)
thr_lobe      = 4.0;
thr_lobe_r    = 4.6;   // lobe center radius → major Ø ≈ 13.2
thr_pitch     = 10;    // one turn ≈ fully engaged
thr_fit       = 0.5;   // radial clearance for THREAD cavities (looser than
                       // uni_fit — printed thread pairs bind long before
                       // sliding fits do)
dial_od       = 20;    // knobs: bigger = more finger torque, thicker walls
uni_flange_od = 18;    // flanges: crown seat + knob/dial thrust bearing
                       // (wide enough to overlap the deep thread's cavity)

// Lengths (each part is modeled in its own print orientation).
uni_sleeve_l  = 25;    // 20 in the bore + 5 of flange
uni_rod_z1    = 31.5;  // spine: rod ends / screw starts
uni_scr_l     = 24;    // screw length: dial + head + expansion travel

// 2D profiles: rod with spline flats, and the screw thread section.
module uni_rod_2d(extra = 0) {
  intersection() {
    circle(d = uni_cone_d2 + 2 * extra);
    square([uni_rod_flat + 2 * extra, uni_cone_d2 + 2 * extra + 2], center = true);
  }
}
// Helical extrude, pitch-linked twist. Parts are threaded together BY HAND
// after printing, so phase never matters — only the `extra` clearance does.
// Defaults are the shared coarse thread (dial stem, cage stem base, head);
// the cage stem's TIP thread passes its own smaller profile.
module uni_helix(h, extra = 0, pitch = thr_pitch, lobe = thr_lobe, core = thr_core, lobe_r = thr_lobe_r) {
  linear_extrude(height = h, twist = -360 * h / pitch, slices = ceil(h * 3), convexity = 10)
    offset(delta = extra) union() {
      circle(d = core);
      translate([lobe_r, 0]) circle(d = lobe);
    }
}

// SPINE — prints as oriented, cone face on the bed (small footprint: brim).
module uni_spine() {
  // Expansion cone, fat end at the bed (the deepest point in the bore).
  cylinder(d1 = uni_cone_d1, d2 = uni_cone_d2, h = uni_cone_h);
  // Splined rod up through the sleeve.
  translate([0, 0, uni_cone_h - 0.1]) linear_extrude(uni_rod_z1 - uni_cone_h + 0.2) uni_rod_2d();
  // The screw: carries the dial low and the head on top.
  translate([0, 0, uni_rod_z1]) uni_helix(uni_scr_l);
}

// SLEEVE — prints flange-down (the bed face doubles as the dial's thrust
// face). Tip and its tapered mouth at the top.
module uni_sleeve() {
  difference() {
    union() {
      cylinder(d = uni_flange_od, h = 5);
      translate([0, 0, 5]) cylinder(d1 = uni_flange_od, d2 = uni_sleeve_od, h = 2);
      translate([0, 0, 5]) cylinder(d = uni_sleeve_od, h = uni_sleeve_l - 5);
    }
    // Keyed bore through the flange: spline flats lock spine rotation.
    translate([0, 0, -1]) linear_extrude(12) uni_rod_2d(uni_fit);
    // Finger bore.
    translate([0, 0, 10.9]) cylinder(d = 6.0, h = 11.7);
    // Tapered mouth at the tip: the cone's wedge seat.
    translate([0, 0, 22.5]) cylinder(d1 = 5.8, d2 = uni_mouth_id, h = uni_sleeve_l - 22.5 + 0.01);
    // Four slots → four fingers, attached low, free tips at the mouth.
    for (i = [0 : 3])
      rotate([0, 0, i * 90])
        translate([-uni_slot_w / 2, 1.6, 11])
          cube([uni_slot_w, uni_sleeve_od, uni_sleeve_l]);
    // Flange knurl flutes (grip while twisting the dial).
    for (i = [0 : 11])
      rotate([0, 0, i * 30])
        translate([uni_flange_od / 2, 0, -1])
          cylinder(d = 1.2, h = 6.5);
    // Tip lead-in chamfer: cut the ring outside a narrowing cone so the tip
    // tapers 8.4 → 7.8 over the last 1.2 mm.
    difference() {
      translate([0, 0, uni_sleeve_l - 1.2]) cylinder(d = uni_sleeve_od + 2, h = 1.21);
      translate([0, 0, uni_sleeve_l - 1.21]) cylinder(d1 = uni_sleeve_od, d2 = 7.8, h = 1.23);
    }
  }
}

// DIAL/KNOB — a plain knurled through-nut; prints flat on the bed. Tall
// enough to always hold a full turn of thread with travel to spare.
module uni_dial() {
  difference() {
    cylinder(d = dial_od, h = 12);
    translate([0, 0, -0.1]) uni_helix(12.2, thr_fit);
    // Knurl flutes.
    for (i = [0 : 23])
      rotate([0, 0, i * 15])
        translate([dial_od / 2, 0, -1])
          cylinder(d = 1.8, h = 14);
  }
}

// HEAD — the screw-on click-nub connector; PRINT ONCE. Blind threaded
// socket opening at the bed; the ceiling is a 45° internal cone, so it
// prints without supports and the spine's screw jams snug against it.
module uni_head() {
  difference() {
    cylinder(d = 20, h = 17);
    translate([0, 0, -0.1]) uni_helix(10.1, thr_fit);
    translate([0, 0, 9.9]) cylinder(d1 = 14.6, d2 = 1, h = 6);
    // Knurl flutes.
    for (i = [0 : 15])
      rotate([0, 0, i * 22.5])
        translate([10, 0, -1])
          cylinder(d = 1.8, h = 19);
  }
  translate([0, 0, 17]) rotate([90, 0, 0]) click_nub_assembly(20);
}

// The replate set: everything EXCEPT the head, each in print orientation.
module universal_stem() {
  translate([-18, 0, 0]) uni_spine();
  uni_sleeve();
  translate([18, 0, 0]) uni_dial();
}

// ============================================================================
// PART 1d — CAGE STEM (universal, screws at the base, experimental).
//
// The molly-bolt take on a universal stem — three chunky parts, no fine
// fingers to snap:
//   CAGE  — a lantern tube: knurled crown flange, four gently PRE-BOWED
//           ribs, and a solid rear ring. Relaxed Ø8.4 slips into any bore
//           from 9 mm up.
//   SPINE — tip disc, spline-flatted rod (keyed to the cage so nothing
//           rotates in the barrel), the screw at the BASE, and the click
//           nub integral on top.
//   KNOB  — knurled nut at the base. Screwing it in presses the cage
//           flange against the crown and PULLS the spine outward: the tip
//           disc compresses the cage lengthwise and the ribs BOW OUTWARD
//           until their broad mid-spans grip the bore. Molly-bolt leverage
//           means about an EIGHTH of a turn past snug is fully gripped —
//           and the same leverage makes the hold very strong at light
//           knob torque.
//   DISC  — the tapered tip disc SCREWS ONTO the rod's tip: both spine
//           ends are otherwise fat, so a bare rod is the only way it can
//           pass through the cage at all. Under load the joint is in pure
//           tension — exactly what a thread is for.
//   HEAD  — no integral nub: the base screw is the SAME thread as the
//           dial stem's, so the shared screw-on head
//           (aim-flag-stem-universal-head.stl, printed once) and the same
//           knob geometry serve both universal stems.
//
// The ribs are elastic, not hinged: peak bending strain stays under ~1 %
// at .40 expansion (print the cage in PETG anyway), and releasing the knob
// lets them spring straight so the stem slides back out. The pre-bow makes
// the buckling direction deterministic — always outward, never inward.
//
// Assembly, in order: 1) spin the KNOB onto the base screw from its top
// end and run it down; 2) slide the rod through the cage from the FRONT
// (flats aligned in the throat); 3) screw the DISC onto the tip until it
// jams on the rod shoulder; 4) screw the shared HEAD onto the base. Use:
// insert until the flange seats on the crown, aim the nub down, pinch the
// knurled flange, tighten the knob.
cage_od       = 8.4;   // relaxed OD — slips into the smallest (9 mm) bore
cage_wall     = 1.15;  // rib thickness
cage_bow      = 0.45;  // built-in outward bulge at mid-rib (buckling bias)
cage_len      = 24;    // rib span
cage_slot_w   = 2.4;   // 4 slots → 4 ribs
cage_ring_l   = 4;     // solid rear ring the tip disc bears on
cage_flange_l = 4;     // knurled crown flange (also the knob's thrust face)
cage_rib_z0   = cage_flange_l + 2;             // ribs start above the flare
cage_rib_z1   = cage_rib_z0 + cage_len;
cage_total    = cage_rib_z1 + cage_ring_l;     // 34: flange 4 + flare 2 + 24 + 4
// Spine sections (base screw shares thr_* with the dial stem/head/knob).
cage_scr_l    = 30;    // base screw: LONG — the knob needs real runway to
                       // keep pressing the stack through its full travel
                       // (print testing: 22 ran out before the squeeze bit)
cage_rod_l    = cage_total + 2;                // through the cage + travel
// Tip thread the taper disc screws onto. Sized to pass through EVERYTHING
// on its way in — the keyed throat (4.7 across the flats) and the rear ring
// bore (Ø5.6) — while the rod's own cross-section forms the shoulder the
// disc jams against. Plenty for the load: the joint sees pure tension.
cage_tip_l      = 7;
cage_tip_pitch  = 6;   // ~one turn seats the disc
cage_tip_lobe   = 1.8;
cage_tip_core   = 2.7;
cage_tip_lobe_r = 1.4; // major Ø 4.6 < 4.7 flat opening (must still pass
                       // the keyed throat on assembly — the hard cap here)

// CAGE — the expandable part, exported as its OWN file ("stem-cage-flex")
// so it can print in TPU: rubber ribs grip the bore far better than PETG,
// can never crack, and are gentler to seat. Prints flange-down; the bowed
// ribs are a solid of revolution (≤ ~3° of overhang) minus the four slots.
module cage_stem_cage() {
  steps = 16;
  difference() {
    rotate_extrude(convexity = 6, $fn = 96) polygon([
      [0, 0],
      [uni_flange_od / 2, 0],
      [uni_flange_od / 2, cage_flange_l],
      [cage_od / 2, cage_flange_l + 1],
      // Ribs: necked to Ø(od − 2·bow) at the ends, bulging to Ø od at
      // mid-span — the relaxed cage never exceeds cage_od, and the bulge
      // biases the buckling outward.
      each [for (i = [0 : steps]) [cage_od / 2 - cage_bow + cage_bow * sin(180 * i / steps), cage_rib_z0 + cage_len * i / steps]],
      [cage_od / 2, cage_rib_z1 + 1],
      [cage_od / 2, cage_total],
      [0, cage_total],
    ]);
    // Keyed throat through flange + flare: spline flats lock spine rotation.
    translate([0, 0, -1]) linear_extrude(cage_rib_z0 + 1.5) uni_rod_2d(uni_fit);
    // Inner cavity along the ribs, bowed with the outside (constant wall).
    rotate_extrude(convexity = 6, $fn = 96) polygon([
      [0, cage_rib_z0 - 0.4],
      each [for (i = [0 : steps]) [cage_od / 2 - cage_bow + cage_bow * sin(180 * i / steps) - cage_wall, cage_rib_z0 + cage_len * i / steps]],
      [0, cage_rib_z1 + 0.2],
    ]);
    // Rear ring bore: plain round, the rod just passes through.
    translate([0, 0, cage_rib_z1 - 0.3]) cylinder(d = 5.6, h = cage_ring_l + 1.3);
    // Four slots → four ribs.
    for (i = [0 : 3])
      rotate([0, 0, i * 90 + 45])
        translate([-cage_slot_w / 2, 1.5, cage_rib_z0])
          cube([cage_slot_w, cage_od, cage_len]);
    // Flange knurl flutes (counter-grip while tightening the knob).
    for (i = [0 : 11])
      rotate([0, 0, i * 30])
        translate([uni_flange_od / 2, 0, -1])
          cylinder(d = 1.2, h = cage_flange_l + 1.5);
  }
}

// STUD — the bare double-ended spine every screw-at-base universal shares:
// base screw, keyed rod (length per design), tip thread for the taper disc.
// Prints standing on the base screw's end face (add a brim).
module uni_stud(rod_l) {
  uni_helix(cage_scr_l);
  translate([0, 0, cage_scr_l - 0.1]) linear_extrude(rod_l + 0.2) uni_rod_2d();
  translate([0, 0, cage_scr_l + rod_l]) uni_helix(cage_tip_l, 0, cage_tip_pitch, cage_tip_lobe, cage_tip_core, cage_tip_lobe_r);
}

module cage_stem_spine() {
  uni_stud(cage_rod_l);
}

// DISC — the screw-on taper tip: a chamfered Ø8.2 disc with a through
// thread; it jams against the rod's shoulder (the flats are wider than the
// tip thread's core). Prints flat.
module cage_stem_disc() {
  difference() {
    union() {
      cylinder(d1 = 7, d2 = 8.2, h = 1);
      translate([0, 0, 0.99]) cylinder(d = 8.2, h = 3.01);
    }
    translate([0, 0, -0.2]) uni_helix(4.4, thr_fit, cage_tip_pitch, cage_tip_lobe, cage_tip_core, cage_tip_lobe_r);
  }
}

// The RIGID cage-stem parts plated in print orientation (PETG). The CAGE —
// the expandable part — is deliberately NOT on this plate: it exports on
// its own ("stem-cage-flex") so it can print in TPU. The knob is the SAME
// part as the dial stem's (uni_dial — identical thread), and the click-nub
// head is the shared aim-flag-stem-universal-head.stl, printed once.
module cage_stem() {
  translate([-16, 0, 0]) cage_stem_spine();
  translate([4, 0, 0]) uni_dial();
  translate([4, 18, 0]) cage_stem_disc();
}

// ============================================================================
// PART 1e — PLUG STEM (universal, squeezed-TPU gland — the test-plug take).
//
// The simplest and gentlest universal: a small TPU RING squeezed lengthwise
// between two rigid faces bulges radially — exactly how a plumber's pipe
// test plug or a bike seatpost expander grips a tube. Rubber conforms to
// the bore, grips like rubber does, SEALS the muzzle, releases with zero
// set, and the gland is a five-minute print that is physically impossible
// to break. Expansion happens just past the crown.
//
// Everything except the SEAT and GLAND is shared with the cage stem — the
// STUD (short version), the KNOB (uni_dial), the taper DISC, the HEAD:
//   SEAT  — a short rigid tube: knurled crown flange + Ø8.2 pilot; its
//           front face is one jaw of the squeeze.
//   GLAND — the TPU ring (Ø8.4 × Ø5.2 × 9). Print the SOFTEST TPU you
//           have: 95A grips 9 mm/.357 easily; 85A recommended to reach
//           .40 at light knob torque.
// Twisting the base knob pulls the stud outward; the taper disc squeezes
// the gland against the seat and it barrels out against the bore.
// Assembly: knob onto the base screw, stud through the seat (flats keyed),
// gland onto the rod, disc onto the tip.
plug_pilot_l  = 8;                       // seat pilot into the bore
plug_seat_l   = 4 + 2 + plug_pilot_l;    // flange + flare + pilot
plug_gland_od = 8.4;                     // relaxed: slips into a 9 mm bore
plug_gland_id = 5.2;                     // slides over the rod
plug_gland_l  = 9;
// Rod exactly spans the stack, so the base screw starts flush with the
// seat's flange face — the knob gets its FULL height of thread runway to
// drive the squeeze (print testing: a gap here wasted most of the travel).
plug_rod_l    = plug_seat_l + plug_gland_l;

// SEAT — prints flange-down; the keyed bore locks the stud's rotation.
module plug_seat() {
  difference() {
    union() {
      cylinder(d = uni_flange_od, h = 4);
      translate([0, 0, 4]) cylinder(d1 = uni_flange_od, d2 = 8.2, h = 2);
      translate([0, 0, 4]) cylinder(d = 8.2, h = plug_seat_l - 4);
    }
    translate([0, 0, -1]) linear_extrude(plug_seat_l + 2) uni_rod_2d(uni_fit);
    for (i = [0 : 11])
      rotate([0, 0, i * 30])
        translate([uni_flange_od / 2, 0, -1])
          cylinder(d = 1.2, h = 6.5);
  }
}

// GLAND — the TPU ring, its own export so it prints in its own material.
module plug_gland() {
  difference() {
    cylinder(d = plug_gland_od, h = plug_gland_l);
    translate([0, 0, -0.5]) cylinder(d = plug_gland_id, h = plug_gland_l + 1);
  }
}

// The rigid plug-stem parts plated (PETG); the gland exports separately.
module plug_stem() {
  translate([-18, 0, 0]) uni_stud(plug_rod_l);
  plug_seat();
  translate([18, 0, 0]) uni_dial();
  translate([18, 18, 0]) cage_stem_disc();
}

// ============================================================================
// Fiducial pattern DARK volumes, in card-local coordinates (card slab local
// y ∈ [−card_t, 0], camera face at y = 0): alternating checker cells plus a
// round dot in one white cell that breaks the checkerboard's 180° rotational
// symmetry. The tiles are RAISED — they sit on top of the flat card face
// (y ∈ [0, pattern_depth]) and print as extra layers in the second color;
// wherever the second color isn't, the face reads as a recess.
module pattern_volumes() {
  // Inset each tile by a hair so diagonally-adjacent cells never share a
  // corner edge — corner-touching cubes make the mesh non-manifold, which
  // trips slicer repair. 0.01 mm is far below anything a nozzle can render.
  eps = 0.01;
  for (i = [0 : pattern_cols - 1], j = [0 : pattern_cols - 1])
    if ((i + j) % 2 == 0)
      translate([-card_w / 2 + pattern_rim + i * cell + eps, 0, pattern_rim + j * cell + eps])
        cube([cell - 2 * eps, pattern_depth, cell - 2 * eps]);
  translate([
    -card_w / 2 + pattern_rim + (marker_cell[0] + 0.5) * cell,
    pattern_depth,
    pattern_rim + (marker_cell[1] + 0.5) * cell,
  ])
    rotate([90, 0, 0]) cylinder(d = cell * 0.4, h = pattern_depth);
}

// ============================================================================
// PART 3 — the card. A flat slab, pattern face at y = 0 (the raised tiles of
// card_inlay() print on top of it), chamfered bottom corners for easy entry
// into the frame slot.
// Bare slab + bottom corner chamfers — shared by every card variant.
module card_blank() {
  difference() {
    translate([-card_w / 2, -card_t, 0]) cube([card_w, card_t, card_h]);
    // Bottom corner chamfers (lead-in).
    for (sx = [-1, 1])
      translate([sx * card_w / 2, 0, 0]) rotate([0, 45, 0])
        cube([4, 2 * card_t + 2, 4], center = true);
  }
}

module card() {
  difference() {
    card_blank();
    if (!printed_pattern) {
      // Paper fallback: one big shallow recess for a glued grid.
      translate([-card_w / 2 + pattern_rim, -pattern_depth, pattern_rim])
        cube([card_w - 2 * pattern_rim, pattern_depth + 1, card_h - 2 * pattern_rim]);
    }
  }
}

module card_inlay() {
  pattern_volumes();
}

// ---- COLOR-QUADRANT pattern (recommended for AMS) --------------------------
// Four big quadrants. Positions below are CARD-LOCAL (col 0 = the gun's
// left); FACING the finished card — as the camera does — left/right mirror,
// so you see: GREEN top-left, RED top-right, BLUE bottom-right, white with a
// BLUE dot bottom-left. The tracker is handedness-proof, so only the colors'
// mutual layout matters, and that is fixed here. Saturated colors make camera
// detection near-unbreakable (color IS the correspondence), and quadrants are
// ~2× the checker cell size, so it tracks from farther away. White base + red
// + green + blue = exactly four AMS slots (the dot prints in blue).
// Export card + card-inlay-red + card-inlay-green + card-inlay-blue, import
// all four as ONE object, assign filaments per part.
quad_size = (card_w - 2 * pattern_rim) / 2;

// One raised quadrant tile at [col, row] (0/1, row 1 = top), eps-inset like
// the checker tiles so nothing shares edges.
module quadrant_tile(col, row) {
  eps = 0.01;
  translate([-card_w / 2 + pattern_rim + col * quad_size + eps, 0, pattern_rim + row * quad_size + eps])
    cube([quad_size - 2 * eps, pattern_depth, quad_size - 2 * eps]);
}

module card_inlay_red() {
  quadrant_tile(0, 1); // top-left
}
module card_inlay_green() {
  quadrant_tile(1, 1); // top-right
}
module card_inlay_blue() {
  quadrant_tile(0, 0); // bottom-left
  // The confirmation dot, centered in the white bottom-right quadrant.
  translate([
    -card_w / 2 + pattern_rim + 1.5 * quad_size,
    pattern_depth,
    pattern_rim + 0.5 * quad_size,
  ])
    rotate([90, 0, 0]) cylinder(d = quad_size * 0.3, h = pattern_depth);
}

// ---- CLICK-IN color card (print each color separately, press to click) ------
// Same quadrant layout as the AMS color card, but the four color patches and
// the confirmation dot are SEPARATE flat parts that press into pockets on the
// card body and CLICK under four small snap ridges — no AMS needed: print the
// body (matte white recommended, so the rims between tiles read as
// background), one tile each in matte red/green/blue, the dot-hole tile in
// matte white, and the dot in matte blue. Drop the dot into the white tile
// from BEHIND (its flange seats in the counterbore), then click every tile
// in; the pocket floor traps the flange so the dot can never fall out. Each
// pocket floor has a poke-hole so a paperclip from the back pops a tile out.
insert_t       = 1.0;   // tile thickness (5 layers @ 0.2)
insert_fit     = 0.15;  // per-side tile ↔ pocket clearance
insert_chamfer = 0.4;   // lead-in chamfer around the tile's bottom edge
pocket_floor   = 0.6;   // card material left under each pocket (3 layers)
pocket_rib     = 1.2;   // wall between/around pockets that carries the ridges
click_bite     = 0.45;  // ridge protrusion into the pocket; the ~0.3 mm net
                        // overlap past insert_fit is what clicks over the tile
click_r        = 0.4;   // snap ridge radius
click_len      = 5;     // snap ridge length
poke_d         = 3.5;   // poke-out hole through the pocket floor
dot_flange     = 1.2;   // per-side flange on the dot, trapped under the tile
dot_counter    = 0.45;  // counterbore depth in the white tile's back face

pocket_size  = quad_size - pocket_rib;          // 34.8 at the defaults
pocket_depth = card_t - pocket_floor;           // 1.4
tile_size    = pocket_size - 2 * insert_fit;    // 34.5
dot_d        = quad_size * 0.3;                 // matches the AMS card's dot
// Ridge center depth: the tile's top face sits at −(pocket_depth − insert_t);
// centering the ridge 0.15 above that leaves its bulge overhanging the tile
// edge by ~click_bite − insert_fit once seated.
click_y      = insert_t - pocket_depth + 0.15;

// Pocket cavity for quadrant [col, row] (card-local coords, cut into the
// face), plus its poke-out hole — off-center so it stays clear of the dot's
// flange when this is the white quadrant.
module insert_pocket_cavity(col, row) {
  x0 = -card_w / 2 + pattern_rim + col * quad_size + pocket_rib / 2;
  z0 = pattern_rim + row * quad_size + pocket_rib / 2;
  translate([x0, -pocket_depth, z0])
    cube([pocket_size, pocket_depth + 1, pocket_size]);
  translate([x0 + 6, -card_t - 1, z0 + 6])
    rotate([-90, 0, 0]) cylinder(d = poke_d, h = card_t + 2);
}

// Four snap ridges per pocket — horizontal half-round beads embedded in the
// walls, one centered per side. The inserted tile bows past them and clicks
// flat onto the floor; the ridges overhang its edges and hold it down.
module insert_pocket_ridges(col, row) {
  x0 = -card_w / 2 + pattern_rim + col * quad_size + pocket_rib / 2;
  z0 = pattern_rim + row * quad_size + pocket_rib / 2;
  xc = x0 + pocket_size / 2;
  zc = z0 + pocket_size / 2;
  off = pocket_size / 2 + click_r - click_bite;
  for (s = [-1, 1]) {
    translate([xc, click_y, zc + s * off])
      rotate([0, 90, 0]) cylinder(r = click_r, h = click_len, center = true);
    translate([xc + s * off, click_y, zc])
      cylinder(r = click_r, h = click_len, center = true);
  }
}

// The click-in card body: plain slab + four pockets + snap ridges (clipped
// to the slab so nothing stands proud of the pattern face).
module card_click() {
  difference() {
    card_blank();
    for (c = [0, 1], r = [0, 1]) insert_pocket_cavity(c, r);
  }
  intersection() {
    union() { for (c = [0, 1], r = [0, 1]) insert_pocket_ridges(c, r); }
    translate([-card_w / 2, -card_t, 0]) cube([card_w, card_t, card_h]);
  }
}

// One click-in color tile, built directly in print orientation (flat on the
// bed, pattern face up, lead-in chamfer around the bottom edge). Print 1×
// each in red, green, and blue.
module insert_tile() {
  hull() {
    translate([0, 0, insert_chamfer])
      linear_extrude(insert_t - insert_chamfer) square(tile_size, center = true);
    linear_extrude(0.01) square(tile_size - 2 * insert_chamfer, center = true);
  }
}

// The white tile: dot hole through the middle plus a counterbore in the back
// face that swallows the dot's flange, so the flange is sandwiched between
// tile and pocket floor once the tile clicks in.
module insert_tile_white() {
  difference() {
    insert_tile();
    translate([0, 0, -0.5]) cylinder(d = dot_d + 0.3, h = insert_t + 1);
    translate([0, 0, -0.5]) cylinder(d = dot_d + 2 * dot_flange + 0.3, h = 0.5 + dot_counter);
  }
}

// The blue confirmation dot: a flanged disc, printed flange-down. Total
// height = insert_t, so it sits flush with the white tile's face.
module insert_dot() {
  cylinder(d = dot_d + 2 * dot_flange, h = 0.4);
  cylinder(d = dot_d, h = insert_t);
}

// ---- SHAPE pattern (black + white ONLY — identity by topology) --------------
// Same quadrant layout as the color card, but each patch is identified by
// its HOLE COUNT instead of its hue: a solid DISK (0 holes) where red was,
// a RING (1 hole) where green was, a TWO-HOLE patch where blue was, plus
// the same dot in the white quadrant. Hole count survives any perspective,
// rotation, and blur-skew, and black-on-white is immune to white balance
// and low-light desaturation. Two filaments — print exactly like the
// checkerboard (zero-waste single swap, or 2-part AMS).
module shape_patch_at(col, row) {
  translate([
    -card_w / 2 + pattern_rim + (col + 0.5) * quad_size,
    pattern_depth,
    pattern_rim + (row + 0.5) * quad_size,
  ])
    rotate([90, 0, 0]) children();
}

module card_inlay_shapes() {
  d_patch = quad_size * 0.94;
  // DISK — top-left (red's slot).
  shape_patch_at(0, 1) cylinder(d = d_patch, h = pattern_depth);
  // RING — top-right (green's slot).
  shape_patch_at(1, 1) difference() {
    cylinder(d = d_patch, h = pattern_depth);
    translate([0, 0, -0.5]) cylinder(d = quad_size * 0.32, h = pattern_depth + 1);
  }
  // TWO-HOLE — bottom-left (blue's slot).
  shape_patch_at(0, 0) difference() {
    cylinder(d = d_patch, h = pattern_depth);
    for (sx = [-1, 1])
      translate([sx * quad_size * 0.19, 0, -0.5])
        cylinder(d = quad_size * 0.28, h = pattern_depth + 1);
  }
  // Dot — bottom-right white quadrant, same as the color card.
  shape_patch_at(1, 0) cylinder(d = quad_size * 0.3, h = pattern_depth);
  // ANCHOR dots at the pattern's edge midpoints: after the main lock the
  // tracker hunts these at known positions and least-squares refines the
  // pose over the full ~65 mm baseline — noticeably steadier aim traces.
  for (a = [[0, 0.9], [0, -0.9], [0.9, 0], [-0.9, 0]])
    translate([a[0] * quad_size, pattern_depth, pattern_rim + quad_size + a[1] * quad_size])
      rotate([90, 0, 0]) cylinder(d = quad_size * 0.16, h = pattern_depth);
}

// ============================================================================
// The card holder, in card-local coordinates: back panel + side walls +
// bottom stop + front lips, open at the top so the card slides in from above
// and clicks past two snap nubs. Card slot: x ±slot_w_in/2, y −slot_d_in..0.
module card_holder() {
  difference() {
    // Solid block: full frame footprint.
    translate([-frame_w / 2, -(back_t + slot_d_in), -bottom_wall])
      cube([frame_w, frame_depth, card_h + bottom_wall]);
    // Card slot, open through the top.
    translate([-slot_w_in / 2, -slot_d_in, 0])
      cube([slot_w_in, slot_d_in + 0.01, card_h + 1]);
    // Front window (lips remain on left/right/bottom; top open).
    translate([-(card_w / 2 - lip_w), -0.5, lip_w])
      cube([card_w - 2 * lip_w, lip_t + 1, card_h + 1]);
    // Entry chamfer at the slot's top rear so the card starts easily.
    translate([-slot_w_in / 2 - 1, -(slot_d_in + 0.8), card_h - 0.8])
      rotate([45, 0, 0]) cube([slot_w_in + 2, 2, 2]);
  }
  // Snap nubs on the lips near the top: rounded ridges protruding ~0.45 mm
  // into the slot; the card flexes past and clicks home.
  for (sx = [-1, 1])
    translate([sx * (card_w / 2 - lip_w / 2), nub_r / 2, card_h - 5])
      rotate([0, 90, 0]) cylinder(r = nub_r, h = 5, center = true);
}

// ============================================================================
// PART 2a — HOLDER: the card-holding object. Cant-free (the 45° lives in the
// connector), so it prints flat/upright cleanly. Rails + a joint pad on the
// back whose foot is flush with the holder's BOTTOM face (it prints straight
// off the buildplate — no supports) with a through slot, opening downward,
// that the connector's tang or the stem's prongs slide up into.
module holder() {
  card_holder();
  difference() {
    translate([-pad_w / 2, back_face - pad_t, pad_z0])
      cube([pad_w, pad_t, joint_len + joint_fit + 2.4]);
    // THROUGH slot: bounded in front by the holder's own back panel, open
    // at BOTH ends — the frame connector's tang still seats in it, and the
    // direct-mount stem's click prongs pass through so their barbs snap
    // over the pad's top edge.
    translate([-(joint_w / 2 + joint_fit), back_face - joint_t - joint_fit, pad_z0 - 1])
      cube([joint_w + 2 * joint_fit, joint_t + joint_fit + 0.01, joint_len + joint_fit + 2.4 + 2]);
  }
}

// ============================================================================
// PART 2b — CONNECTOR: stem-tang socket, drop arm, and a 45° head whose tang
// plugs up into the holder's pocket.
module connector() {
  if (cant_dir == "flat") {
    // Socket angled to match the 135° tongue (45° down-forward), cut flush
    // at the collar face so nothing pokes back into the slide's muzzle.
    difference() {
      rotate([-tang_tilt, 0, 0]) {
        difference() {
          translate([-socket_out_w / 2, 0, -socket_out_h / 2])
            cube([socket_out_w, socket_len + 2, socket_out_h]);
          translate([-socket_w / 2, -1, -socket_h / 2])
            cube([socket_w, socket_len + 1, socket_h]);
        }
      }
      translate([-50, -99.5, -50]) cube([100, 100, 100]); // flush at y = 0.5
    }
    // Strut: the "\" — from the socket's lower end straight down-back to the
    // foot behind the card's rear edge (the card is the "_").
    hull() {
      rotate([-tang_tilt, 0, 0])
        translate([-arm_width_u / 2, socket_len, -socket_out_h / 2])
          cube([arm_width_u, 2, socket_out_h]);
      translate([-arm_width_u / 2, flat_arm_y0, flat_foot_z])
        cube([arm_width_u, arm_depth, 1]);
    }
    // Gusset: foot → head, reaching down-forward under the card's rear edge.
    hull() {
      translate([-arm_width_u / 2, flat_arm_y0, flat_foot_z])
        cube([arm_width_u, arm_depth, 1]);
      translate(flag_anchor) rotate(flag_rot)
        translate([-pad_w / 2, back_face - pad_t, pad_z0 - head_h])
          cube([pad_w, pad_t, 1]);
    }
  } else if (cant_dir == "under") {
    // Straight socket block around the stem's tang.
    difference() {
      translate([-socket_out_w / 2, 0, -socket_out_h / 2])
        cube([socket_out_w, socket_len + 2, socket_out_h]);
      translate([-socket_w / 2, -1, -socket_h / 2])
        cube([socket_w, socket_len + 1, socket_h]);
    }
    // Diagonal arm: from under the socket down-forward, clearing the
    // swept-back card's bottom-front corner on the outside.
    hull() {
      translate([-arm_width_u / 2, arm_y0, -socket_out_h / 2])
        cube([arm_width_u, arm_depth, 1]);
      translate([-arm_width_u / 2, under_foot_y, under_foot_z])
        cube([arm_width_u, arm_depth, 1]);
    }
    // Gusset: foot → head, crossing UNDER the card's bottom edge.
    hull() {
      translate([-arm_width_u / 2, under_foot_y, under_foot_z])
        cube([arm_width_u, arm_depth, 1]);
      translate(flag_anchor) rotate(flag_rot)
        translate([-pad_w / 2, back_face - pad_t, pad_z0 - head_h])
          cube([pad_w, pad_t, 1]);
    }
  } else {
    // Straight socket block around the stem's tang.
    difference() {
      translate([-socket_out_w / 2, 0, -socket_out_h / 2])
        cube([socket_out_w, socket_len + 2, socket_out_h]);
      translate([-socket_w / 2, -1, -socket_h / 2])
        cube([socket_w, socket_len + 1, socket_h]);
    }
    // Drop arm — stops short of where the holder sits (separate parts).
    translate([-arm_width / 2, arm_y0, arm_bottom])
      cube([arm_width, arm_depth, -arm_bottom - socket_out_h / 2 + 1]);
    // Gusset bridging the arm foot to the head.
    hull() {
      translate([-arm_width / 2, arm_y0, arm_bottom])
        cube([arm_width, arm_depth, 1]);
      translate(flag_anchor) rotate(flag_rot)
        translate([-pad_w / 2, back_face - pad_t, pad_z0 - head_h])
          cube([pad_w, pad_t, 1]);
    }
  }
  // 45° head + tang, built in the holder's (card-local) frame so the two
  // parts are guaranteed to register.
  translate(flag_anchor) rotate(flag_rot) {
    // Head block: seats flat against the holder's back panel, its top face
    // against the pad's bottom edge (the tang's depth stop).
    translate([-pad_w / 2, back_face - pad_t, pad_z0 - head_h])
      cube([pad_w, pad_t, head_h]);
    // Tang, centered in the pocket's thickness.
    translate([-joint_w / 2, back_face - joint_t - joint_fit / 2, pad_z0 - 0.5])
      cube([joint_w, joint_t, joint_len + 0.5]);
  }
}

// Holder + card seated, for the assembled preview.
module holder_assembled() {
  translate(flag_anchor) rotate(flag_rot) holder();
}
module card_in_frame() {
  translate(flag_anchor) rotate(flag_rot)
    translate([0, -card_fit / 2, 0]) {
      card();
      color("Black") card_inlay();
    }
}

// ============================================================================
// Print orientations:
//  • stem      — vertical, tang down, bore tip up (brim recommended).
//  • holder    — upright on its bottom wall, rails vertical. The joint pad's
//    foot is coplanar with the bottom wall, so it stands on the buildplate
//    too — no supports anywhere.
//  • connector — natural attitude lifted onto the bed: arm as a column, 45°
//    head at the bottom (self-supporting). Brim recommended.
//  • card      — FLAT, pattern face up.
module stem_for_print() {
  if (mount_style == "direct") {
    // Rod vertical TIP DOWN, standing on the rod's flat tip end: a plain
    // straight print — rod, collar, base screw. No supports — use a
    // generous brim (the footprint is a ~7 mm disc).
    translate([0, 0, stem_len]) rotate([90, 0, 0]) stem();
  } else {
    // Rod vertical, tip up. With the tilted tongue the part stands on the
    // tongue's lowest edge — print with a brim and supports touching
    // buildplate (under the collar ring and tongue).
    translate([0, 0, collar_t + tang_len * cos(tang_tilt) + (tang_h / 2) * sin(tang_tilt)])
      rotate([-90, 0, 0]) stem();
  }
}
module holder_for_print() {
  translate([0, 0, bottom_wall]) holder();
}
module connector_for_print() {
  if (cant_dir == "flat" || cant_dir == "under") {
    // The flipped head has no flat bottom, so the part lies on its SIDE —
    // socket, diagonal arm, and head all rest on their flat x-faces. Enable
    // "supports touching buildplate" for the tang rib.
    translate([0, 0, pad_w / 2]) rotate([0, 90, 0]) connector();
  } else {
    translate([0, 0, arm_drop]) connector();
  }
}
module card_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card();
}
module card_inlay_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_inlay();
}
module card_inlay_red_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_inlay_red();
}
module card_inlay_green_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_inlay_green();
}
module card_inlay_blue_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_inlay_blue();
}
module card_inlay_shapes_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_inlay_shapes();
}
module card_click_for_print() {
  translate([0, 0, card_t]) rotate([90, 0, 0]) card_click();
}
// The insert tiles and dot are already modeled in print orientation.

if (render_part == "assembled") {
  stem();
  if (mount_style == "direct") {
    // The shared screw-on head, jammed on the base screw...
    color("Goldenrod") translate([0, head_seat_y, 0]) rotate([-90, 0, 0]) uni_head();
    // ...and the holder + card clicked onto ITS prongs, pattern face
    // turned up-forward.
    translate([0, dm_hold_ty, dm_hold_tz]) rotate([135, 0, 180]) {
      color("SlateGray") holder();
      color("WhiteSmoke") translate([0, -card_fit / 2, 0]) {
        card();
        color("Black") card_inlay();
      }
    }
  } else {
    color("SteelBlue") connector();
    color("SlateGray") holder_assembled();
    color("WhiteSmoke") card_in_frame();
  }
} else if (render_part == "stem") {
  stem_for_print();
} else if (render_part == "stem-universal") {
  universal_stem();
} else if (render_part == "stem-universal-head") {
  uni_head();
} else if (render_part == "stem-cage") {
  cage_stem();
} else if (render_part == "stem-cage-flex") {
  cage_stem_cage();
} else if (render_part == "stem-plug") {
  plug_stem();
} else if (render_part == "stem-plug-gland") {
  plug_gland();
} else if (render_part == "holder") {
  holder_for_print();
} else if (render_part == "connector") {
  connector_for_print();
} else if (render_part == "card") {
  card_for_print();
} else if (render_part == "card-inlay") {
  card_inlay_for_print();
} else if (render_part == "card-inlay-red") {
  card_inlay_red_for_print();
} else if (render_part == "card-inlay-green") {
  card_inlay_green_for_print();
} else if (render_part == "card-inlay-blue") {
  card_inlay_blue_for_print();
} else if (render_part == "card-inlay-shapes") {
  card_inlay_shapes_for_print();
} else if (render_part == "card-click") {
  card_click_for_print();
} else if (render_part == "card-click-tile") {
  insert_tile();
} else if (render_part == "card-click-tile-white") {
  insert_tile_white();
} else if (render_part == "card-click-dot") {
  insert_dot();
} else if (render_part == "card-shape-swap") {
  // One merged mesh: body + raised shape patches. Slice with a single
  // "change filament at layer" at z = card_t (2.0 mm) — white base below,
  // black shapes above, zero waste.
  card_for_print();
  card_inlay_shapes_for_print();
} else if (render_part == "card-swap") {
  // One merged mesh: body + raised tiles. Slice with a single "change
  // filament at layer" at z = card_t (2.0 mm) for the zero-waste two-color
  // card — only tile geometry exists above that plane.
  card_for_print();
  card_inlay_for_print();
} else {
  // "print": everything plated (single color).
  translate([-70, 0, 0]) stem_for_print();
  translate([-30, 60, 0]) holder_for_print();
  if (mount_style != "direct") translate([10, -20, 0]) connector_for_print();
  translate([0, 160, 0]) card_for_print();
}
