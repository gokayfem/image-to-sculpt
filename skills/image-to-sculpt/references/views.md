# Views

The photo shows one side. Everything else comes from views of the same subject from other angles: generated
(nano banana edit of the front image) or imported (your own photos or renders).

## Generating

`i2s.py views CASE [--set minimal|standard|full] [--angles "az:el,..."]`

| set | angles (az, el) | calls |
|---|---|---|
| minimal | ±45, ±90, 180 | 5 + 1 held out |
| standard | minimal + −135, front from above (0, 30), back from above (180, 30), (−45, 30) | 9 + 1 held out |
| full | standard + (45, 30), front from below (0, −20), ±65 | 13 + 1 held out |

The held-out view is (135, 0): never used to fit, measured like the others. Each call costs about $0.15 plus
mask and normals (~$0.04), estimated; the case budget (`--budget` at `new`, `settings.budget_usd`) stops paid
calls before it is exceeded. Request ids go into `case.json` before the result is awaited.

The prompt (`inputs.PROMPT`) asks for the same subject, pose, proportions, details and materials, the same
distance and lens, the whole subject in frame on a plain light grey background without a ground shadow. Angles
are described as orbits ("the camera has orbited 45 degrees around the subject to the RIGHT, so we now see more
of the side that was on the right of the original image"), which works for subjects without a left and right.

## Judging a generated view

Look at every one before carving. Reject (set `role` to `rejected` in `case.json`, with a `reason`) or regenerate:

- a different pose (arms moved, head turned), an extra or missing limb, a different subject;
- the subject cropped by the frame (out-of-frame pixels count as unknown, but a crop hides real shape);
- a cast shadow or a floor joined to the subject in the mask (`views/<v>_mask.png`);
- a mirrored or swapped side (a scar or an emblem on the wrong side): registration may still place it, but its
  details are wrong.

Registration and screening catch most of these on their own (`iou_hull` < 0.80 → rejected). Say what you saw
either way.

## Registration

`hull` registers every view twice: against an inflated proxy of the front silhouette, then against the first
hull (every 10° of azimuth × elevations −15…45°, closed-form scale and offset, a quarter-resolution splat scored
on silhouette IoU + normal agreement − a small prior toward the requested angle), then refines on the exact
rasteriser. `case.json` keeps `requested` and `params`; they often differ by tens of degrees. That is expected:
the registered angle is the one the view was really taken from.

The search keeps |az| within 60° of the requested |az| and the elevation within 30° of the requested one.
Generators often swap left and right (a view asked for at +90° registered at −115° on the troll), which this
allows; they rarely swap front and back, which this forbids. That matters because on a left-right symmetric
subject a view at (az, el) has the same outline as one at (180° − az, −el): without the band, back views of a
bust registered as front views. Imported views without an angle search the whole circle.

`i2s.py register CASE` re-refines the training views against the current sculpt (useful after big shape
changes). The held-out view keeps its camera unless `--include-holdout`.

## Adding views where the sculpt is unsure

When a region keeps failing (webs that will not open, a back that stays flat, an armpit flap), ask for views
that look into it: `--angles "-30:25,30:25"` for gaps between arms and body seen from above, `0:-20` for under a
chin or a skirt, `150:0` for a back three-quarter. Then `hull` again (cheap) or `register` + more `fit` passes.
Smoothing does not fix what no view sees.

## Imported views

`views --import DIR`: images named with their angle (`az+090.png`, `az-045_el+30.png`, `az180_hi.png`) or listed
in `DIR/views.json` (`{"IMG_2041.jpg": {"az": 90, "el": 0}}`); unknown angles are found by the whole-circle search.
A `<name>_mask.png` or `<name>_normals.png` next to an image is used instead of calling fal. `--holdout FILE`
holds one out. Real photos are usually more consistent than generated views: registration IoUs above 0.95 are
normal.
