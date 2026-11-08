# Contract: {{SUBJECT}}

Input: `{{INPUT}}` (sha256 `{{SHA256}}`). Written before any view is generated or any surface is fitted.

## Goal

A watertight sculpt of {{SUBJECT}} that matches the input image in the image's own camera (silhouette,
normals, depth) and stays consistent with generated views of the sides the image does not show. The build is
recorded step by step and replayed in Blender.

## What judges it

- The input image: silhouette IoU, outline distance, normal angle, roughness against the image's own
  roughness, metric depth (when MoGe gives it). Targets in `acceptance.json`.
- Every training view: the same measures, registered camera.
- One held-out view: never used to fit, camera frozen before fitting. Reported separately.
- A person looking at the sheets, the views sheet and the turntable.

## Honesty rules

- What only generated views show (the back, the far side) is the generator's guess; the report says so.
- Generated views that disagree with the others are rejected, with the reason recorded; none is edited.
- Targets in `acceptance.json` are not lowered after a failure.
- Every step is recorded; a fix is a new step.
- Paid calls stay within `settings.budget_usd` (estimates), with request ids recorded.

## Free choices (fill in)

- Subject phrase used in prompts:
- View set and extra angles, and why:
- Symmetry (auto / on / off), and why:
- Regions for focus passes:
