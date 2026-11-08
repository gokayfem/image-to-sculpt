# Lessons

What went wrong while building the first sculpts with this method, and the rule each one became. Most of these
were found on a cartoon troll (four versions, from a primitive rig to a 74-step map fit) and generalise to any
subject.

## Views

- **Generated views are not at the angle you ask for.** Views requested at +45°, +90° and −90° registered at
  +19°, −123° and +35°. Registration searches the whole circle; the request is only a weak prior.
- **Silhouettes cannot tell front from back** on a left-right symmetric subject: (az, el) and (180° − az, −el)
  have the same outline, and the hull's normals carry no face to tell them apart. A whole-circle search put the
  back views of a bust and a bulldog in front. Generators swap left and right, not front and back, so the search
  keeps |az| near the requested |az|.
- **Registration needs a reasonable proxy and a fixed centre.** Re-centring the orbit on an under-constrained
  hull dragged the centre to the far end of the carve box and broke every camera. The centre moves only when
  side views constrain depth, and never far.
- **Regions no view sees need new views, not smoothing.** The webs between arm and body survived every
  smoothing and opening pass until views from above and from three-quarter angles looked into the gap. Ask the
  generator for angles that look into the problem.
- **One bad generation cuts real volume.** A view with a changed pose or an extra limb carves the sculpt; screen
  views against the hull of the others before the final carve.

## Fitting

- **Generated-view depth makes lumps.** Their depth maps disagree by centimetres after alignment; only the
  photo's depth is used. Their silhouettes and normals carry the information.
- **Averaging generated normals over the photo's flattens features.** Where the photo sees a surface, the photo's
  normals govern alone; generated views fill in where it does not.
- **Each generated view is tilted a few degrees overall.** One global rotation per view (Kabsch on the normals)
  before integration; without it, creases appear where the dominant view changes.
- **Blurring the normal targets flattens small features** (a nose, a brow). Never blur; let the solver's
  stiffness limit each stage instead (forms → surface → detail).
- **A smoothed fallback is a smoothing flow.** Where no view is sure, keep the current normal, not a smoothed one;
  otherwise every pass wears down the details.
- **Silhouette pulls in a flexible field make ridges along limbs.** Silhouettes and depth go through a very stiff
  field (whole forms move), normals through a flexible one (two solves per pass).
- **Displace the original surface, solve on a smoothed copy.** Solving against raw marching-cubes facets
  "corrects" the facets into staircases; displacing the smoothed copy shaves curved features every pass.

## Distance fields

- **Keep the field exact.** A field that is only distance-like (blended, offset, a primitive union) turns into
  voxel staircases once the surface moves: exact triangle distances in a band, a distance transform beyond.
- **Never mirror the field itself.** Sampling it at mirrored voxels left grid moiré rings and copied an
  off-centre feature into a second one. Symmetry acts on the targets.
- **Two sheets that nearly touch flip each other's signs.** Only the nearest sheet votes on inside/outside.
- **Zero-area marching-cubes triangles give NaN distances.** Skip them; their neighbours cover them.

## Gaps and webs

- **Closing fills the gap you are trying to open.** Use opening (erode, then dilate) for webs and flaps, and a
  silhouette cut (from the photo, smooth max) for gaps the photo shows.
- **A silhouette cut from generated views shreds thin parts** (fingers): their outlines disagree by a few
  pixels. Cut with the photo everywhere, and with generated views only in a region you chose.
- **Limit every cut to the zone that needs it.** An elevated view's cut cracked a shoulder it was never aimed at.

## Process

- **Record every step.** The build is the product: carve steps, every pass, every cut. The timelapse and the
  Blender replay come from the record, and a fix is a new step, never an edit.
- **Visual feedback at every checkpoint.** A map sheet per stage, close-ups of what changed, before/after pairs.
  People notice problems in images that no single number shows ("random shapes", a missing gap).
- **One number never stands for quality.** IoU went up while roughness doubled; normals improved while a gap
  closed. Report IoU, outline distance, normal angle, roughness and depth side by side.
