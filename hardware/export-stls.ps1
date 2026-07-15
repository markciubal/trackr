# Exports every part of the Glock 17 aim flag as separate STLs.
#
# The card is deliberately TWO files — card.stl (body with checker pockets)
# and card-inlay.stl (the tiles that fill them) — so the slicer can treat
# them as two parts of one object and assign a different filament to each.
# Never render/export them together from OpenSCAD: it would union them into
# one solid and the color boundary would be lost.
#
# Usage:  .\export-stls.ps1            (expects `openscad` on PATH)
#         .\export-stls.ps1 -OpenScad "C:\Program Files\OpenSCAD\openscad.exe"
param(
  [string]$OpenScad = ""
)

# Locate OpenSCAD: explicit param → PATH → standard install locations.
if (-not $OpenScad) {
  # Prefer openscad.com — the console binary that PowerShell can wait on and
  # read an exit code from (the .exe is a GUI app that detaches immediately).
  $candidates = @(
    "openscad.com",
    "openscad",
    "$env:ProgramFiles\OpenSCAD\openscad.com",
    "$env:ProgramFiles\OpenSCAD (Nightly)\openscad.com",
    "${env:ProgramFiles(x86)}\OpenSCAD\openscad.com",
    "$env:LOCALAPPDATA\Programs\OpenSCAD\openscad.com",
    "$env:ProgramFiles\OpenSCAD\openscad.exe",
    "$env:ProgramFiles\OpenSCAD (Nightly)\openscad.exe",
    "${env:ProgramFiles(x86)}\OpenSCAD\openscad.exe",
    "$env:LOCALAPPDATA\Programs\OpenSCAD\openscad.exe"
  )
  foreach ($candidate in $candidates) {
    $resolved = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($resolved) { $OpenScad = $resolved.Source; break }
  }
  if (-not $OpenScad) {
    Write-Error ("OpenSCAD not found. Install it from https://openscad.org/downloads.html " +
      "(or `winget install OpenSCAD.OpenSCAD`), or pass -OpenScad <path to openscad.exe>.")
    exit 1
  }
}
Write-Host "Using OpenSCAD: $OpenScad"

$src = Join-Path $PSScriptRoot "glock17-aim-flag.scad"
# Direct mount (default in the .scad): the flag stem's tang plugs straight
# into the holder's pocket — no connector. The connector can still be
# exported manually by setting mount_style = "frame" in the .scad.
$parts = @(
  "stem", "holder", "card",
  "card-inlay-shapes", "card-shape-swap",                   # B/W shape card (recommended)
  "card-inlay-red", "card-inlay-green", "card-inlay-blue",  # RGB color card (AMS)
  "card-click", "card-click-tile",                          # RGB color card, CLICK-IN
  "card-click-tile-white", "card-click-dot",                #   (single-color printers)
  "card-inlay", "card-swap"                                 # checkerboard card
)

foreach ($part in $parts) {
  $out = Join-Path $PSScriptRoot "glock17-aim-flag-$part.stl"
  Write-Host "Exporting $part -> $out"
  # \" so the quotes survive PowerShell's native-argument passing — OpenSCAD
  # must receive render_part="stem" (with quotes) for a string assignment.
  $define = 'render_part=\"' + $part + '\"'
  & $OpenScad -o $out -D $define $src
  if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenSCAD failed exporting '$part' (exit $LASTEXITCODE)."
    exit 1
  }
}

# Universal dial stem (9 mm through .40 — twist to expand): the replate set
# (sleeve + spine + dial) plus the screw-on head, which only prints once.
foreach ($uniPart in @("stem-universal", "stem-universal-head", "stem-cage", "stem-cage-flex", "stem-plug", "stem-plug-gland")) {
  $outUni = Join-Path $PSScriptRoot "aim-flag-$uniPart.stl"
  Write-Host "Exporting $uniPart -> $outUni"
  $define = 'render_part=\"' + $uniPart + '\"'
  & $OpenScad -o $outUni -D $define $src
  if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenSCAD failed exporting '$uniPart' (exit $LASTEXITCODE)."
    exit 1
  }
}

# .40 S&W stem (G22/23) — the stem is the only caliber-specific part; the
# holder and card are shared.
$out40 = Join-Path $PSScriptRoot "glock22-aim-flag-stem.stl"
Write-Host "Exporting stem (.40 S&W) -> $out40"
& $OpenScad -o $out40 -D 'render_part=\"stem\"' -D 'caliber=\"40\"' $src
if ($LASTEXITCODE -ne 0) {
  Write-Error "OpenSCAD failed exporting the .40 stem (exit $LASTEXITCODE)."
  exit 1
}

Write-Host ""
Write-Host "Done. Pattern card options:"
Write-Host ""
Write-Host "  RGB COLOR CARD (recommended - near-unbreakable camera detection):"
Write-Host "    1. Import ...-card.stl + ...-card-inlay-red/-green/-blue.stl together"
Write-Host "       (load as ONE object with multiple parts -> Yes)."
Write-Host "    2. Filaments: matte WHITE body, matte RED / GREEN / BLUE inlays."
Write-Host "       (The confirmation dot is part of the blue inlay.)"
Write-Host ""
Write-Host "  RGB COLOR CARD, CLICK-IN (no AMS - one color per print):"
Write-Host "    1. Print ...-card-click.stl in matte WHITE (flat, face up)."
Write-Host "    2. Print ...-card-click-tile.stl once each in matte RED, GREEN, BLUE."
Write-Host "    3. Print ...-card-click-tile-white.stl in matte WHITE and"
Write-Host "       ...-card-click-dot.stl in matte BLUE."
Write-Host "    4. Drop the dot into the white tile from BEHIND, then press every"
Write-Host "       tile into its pocket until it clicks. Facing the card: GREEN"
Write-Host "       top-left, RED top-right, BLUE bottom-right, white+dot bottom-left."
Write-Host ""
Write-Host "  CHECKERBOARD card, zero-waste single swap:"
Write-Host "    1. Slice glock17-aim-flag-card-swap.stl alone, flat as oriented."
Write-Host "    2. Right-click the layer slider at z = 2.0 mm -> 'Change filament'."
Write-Host "       Base = light matte, everything above = dark matte."
