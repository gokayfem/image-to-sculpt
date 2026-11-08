# Fitting

The hull is the right silhouette from every registered view and nothing else: flat where views do not cut,
faceted where they meet. The fit turns it into the subject's surface, one recorded pass at a time.

## One pass (`fit.one_pass`)

1. Mesh the distance field; make a Taubin-smoothed copy for the solve (no marching-cubes facets in the normals).
2. For each view, the photo first, rasterise the copy from the view's camera and build per-vertex targets:
   - **normals**: the view's normal map where the vertex is visible and faces the camera (weight ∝ facing²,
     agreement with the current normal within 40–70°). A generated view gets one global rotation (Kabsch,
     < 8°) and counts only where the photo does not see.
   - **silhouette**: a signed push along the normal so the projected outline lands on the view's outline
     (clipped per stage).
   - **depth**: the photo's metric depth, robust weight (Cauchy, 1.5 % of the subject).
3. **Symmetry** (if detected): each vertex finds its mirror partner across the plane; where one side is seen
   better, it lends its mirrored targets to the other. Never mirrors the field.
4. **Two solves.** Silhouette + depth through a very stiff thin-plate field (`lam_low`: whole forms move and
   resize). Normals through a flexible Laplacian-regularised integration (`lam`: the scale of change), on top.
   Focus passes add a mid band (`lam_mid`) so whole fingers or toes can move.
5. Soft step limit (`step × tanh(t / step)`), displace the **original** surface, re-distance exactly, mesh,
   record the step, measure every view (held-out included) at half resolution.

## Stages

| stage | passes | step | lam (normals) | lam_low | what moves |
|---|---|---|---|---|---|
| forms | 10 | 0.003 | 3000 (~4 % of the subject) | 1e6 | proportions, big volumes |
| surface | 14 | 0.002 | 100 (~1.7 %) | 3e5 | planes, muscles, folds |
| detail | 14 | 0.0012 | 6 (~0.5 %) | 3e5 | features: noses, brows, knuckles |
| polish | 4 | 0.001 | 6 | 3e5 | after cuts and smoothing |
| focus | 8 | 0.001 | 3 | 3e5 + mid 300 | a chosen region only (`--region`) |

Units are case units (1 = the subject's front extent), so these values transfer between subjects. The grid
resolution is `settings.res` (default 400 voxels per unit); `fit_scale` (default 1.0) sets the raster resolution
of the views during fitting.

## Ops

- **cut** (`i2s.py cut CASE [--region R] [--views front|all]`): a silhouette cut by smooth maximum (no creases),
  then the largest piece is kept. With the photo it opens every gap the photo shows (fingers, legs, arm and
  body). With generated views (`--views all`) it also opens gaps they show, but their outlines disagree by a few
  pixels: limit it to a region.
- **smooth** (`i2s.py smooth CASE [--radius VOX] [--region R]`): morphological opening with exact re-distancing.
  Removes webs, flaps and specks thinner than twice the radius (default 1.2 voxels). A region feathers it.
- **focus** (`fit --stage focus --region hands`): passes whose displacement is limited to a front-image region
  (spread a few centimetres over the surface), with a mid-stiffness band so whole parts can move.

Regions: `i2s.py parts CASE --prompt "hands" --prompt "face"` (SAM-3 on the photo → `parts/<label>.png`) or
`--box label:x0,y0,x1,y1`; or inline `box:0.3,0.1,0.7,0.4` (fractions of the working image). Join with `+`.

## Default plan (`i2s.py run`)

`forms → surface → detail → cut (photo) → smooth → polish → finish` (38 + 4 passes, 2 ops). Change it per case in
`settings.plan` (`[["fit", "forms"], ["cut", "front"], ...]`). A typical second round after reading the sheets:

```bash
python3 scripts/i2s.py parts CASE --prompt "hands" --prompt "face"
python3 scripts/i2s.py fit CASE --stage focus --region face
python3 scripts/i2s.py cut CASE --region hands
python3 scripts/i2s.py fit CASE --stage focus --region hands
python3 scripts/i2s.py smooth CASE --region box:0.2,0.3,0.8,0.7
python3 scripts/i2s.py fit CASE --stage polish
python3 scripts/i2s.py finish CASE
```

## Reading the per-pass log

Every pass logs train IoU, normal median, roughness ratio and the held-out view. Healthy: IoU and normals improve
over forms and surface, roughness settles between 0.8 and 1.2. Roughness above ~1.3 means lumps (too much
detail from noisy normals: fewer detail passes, or a smooth + polish). IoU falling during detail means normals
fight silhouettes (usually a misregistered view: `register CASE`). A held-out view that worsens while training
views improve means the fit is copying one generated view's quirks.
