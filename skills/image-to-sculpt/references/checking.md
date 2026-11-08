# Checking

`i2s.py check CASE` renders the current sculpt from every registered camera and compares it with that view's
maps. Look at the sheet first, then the numbers.

## The sheet (`checks/sheet_*.png`)

One row per view (photo, training views, held-out view), five columns:

| column | look for |
|---|---|
| reference | the image the view came from |
| reference normals | what the surface should do (blue faces the camera, red right, green up) |
| sculpt | clay render of the sculpt from the same camera |
| normal error | angle per pixel (blue 0°, red ≥ 45°): red blobs are wrong surfaces, red lines are creases or misregistration |
| silhouette | grey both, green missing from the sculpt, pink extra in the sculpt |

## The numbers

| metric | meaning | default target (`acceptance.json`) |
|---|---|---|
| IoU | silhouette overlap | photo ≥ 0.97, training mean ≥ 0.92, held-out ≥ 0.85 |
| outline px | mean distance between outlines | photo ≤ 2.5 px |
| normals median | median angle to the normal map | photo ≤ 10°, training mean ≤ 18°, held-out ≤ 22° |
| normals < 15° | share of pixels within 15° | reported, no target |
| roughness × | sculpt roughness / reference roughness | training mean 0.75–1.25 |
| photo depth | mean depth error on the photo, /1000 of the subject | ≤ 15 |

Set the targets per case before fitting (a glossy or busy reference earns looser normal targets), then leave them.

Generated views are not ground truth: they disagree with each other and with the photo. A training mean that
keeps improving while the held-out view stops is the signal to stop fitting. The held-out *best-fit* row (the
view re-registered to the final shape) separates shape error from camera error.

## What to fix, and with what

| symptom on the sheet | cause | fix |
|---|---|---|
| green gap between legs/fingers in the photo row | the hull filled it; the soft term cannot open it | `cut CASE --region <part>` |
| pink web between arm and body in a side view | no view looked into the gap | new views (`views --angles`), then `cut --views all --region` |
| lumpy surface, roughness > 1.3 | noisy normals integrated too finely | `smooth`, `fit --stage polish`; fewer detail passes |
| flat face or hands | detail stage too stiff for them | `parts --prompt face`, `fit --stage focus --region face` |
| one view all red with a consistent tilt | misregistered or a different pose | `register CASE`, or reject the view |
| red creases along limbs | silhouette fighting normals | check registration; `fit --stage polish` |
| specks, loose bits | thin pieces left by cuts | `smooth` (drops loose pieces) |

Before claiming a fix, show the before/after sheets or close-ups (crop the same box of both).

## Acceptance

`acceptance.json` in each case holds the targets agreed at the start (from `templates/acceptance.json`). Do not
lower a target after a failure; report the failure.
