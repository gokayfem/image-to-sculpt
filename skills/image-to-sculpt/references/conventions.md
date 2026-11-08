# Conventions

Everything a new tool needs to read or write a case without breaking the others.

## World frame and units

- The photo's camera sits at the origin and looks along **+Y**; **+Z** is up, **+X** is image right.
- Units are normalised per case: the subject's larger front extent (width or height, from the photo's metric
  depth) is **1.0**. Every tolerance in the code is in these units (a step of 0.003 is 0.3 % of the subject).
- Without depth (no MoGe), 1 unit spans the larger mask extent at the estimated centre distance.

## Cameras

A view camera is the front camera **orbited about the vertical axis through the subject centre** `c` (in
`case.json` → `front.center`):

| param | meaning |
|---|---|
| `az` | degrees; + = the camera moved to the right of the original image (sees more of what was on the right) |
| `el` | degrees; + = the camera moved up and looks down |
| `s` | focal scale (generated views drift in apparent size) |
| `du`, `dv` | principal point offset in **pixels of the working resolution** |

`K = (fx, fy, cx, cy)` is the front camera at the working resolution (`front.K`). `geom.make_cam(params, c, K)`
gives `{"R": columns = right, forward, up in world, "p": position, "K": ...}`; `geom.scale_cam(cam, f)` gives the
same camera at f × the resolution. Do not store offsets at another resolution: a half-resolution offset applied
at full resolution shifted one generation of this pipeline's views by ~1 % of the subject.

Blender: `geom.blender_camera(cam, W, H)` → `matrix_world` (camera looks down its −Z), `lens` (36 mm sensor
width, `sensor_fit = HORIZONTAL`), `shift_x`, `shift_y` (fractions of the sensor width).

## Maps

| map | file | convention |
|---|---|---|
| mask | `ref/front_mask.png`, `views/<v>_mask.png` | 8-bit, > 127 = subject |
| normals | `ref/front_normals.png`, `views/<v>_normals.png` | camera frame: x right, y up, z toward the camera; RGB = (n + 1) × 127.5 (Marigold's encoding) |
| depth | `ref/front.npz` → `depth` | camera-axis distance in case units, NaN outside the mask; the photo only |
| front | `ref/front.npz` | `mask`, `depth`, `ncam` (float normals), `K`, `has_depth`, `has_normals` |

Every view image, mask and normal map is stored at the working size `work.size` (the input padded to the
nearest generator aspect ratio, then resized). Generated views are requested at that aspect ratio, so one lens
model fits all of them.

## Volume and meshes

- `build/field.npz`: `F` (float32 signed distance, negative inside), `lo` (grid origin), `vox` (voxel size);
  voxel i's centre is `lo + (i + 0.5) * vox`. Default `vox = 1/400` (`settings.res`).
- After every operation the field is **exact** near the surface (closest-point distance to the triangle mesh in a
  6-voxel band) and a Euclidean distance transform elsewhere (`sdf.consistent`).
- Meshes are `verts` (float32, world units) + `faces` (int32 triangles, outward winding).
- `build/steps/NNNN_<label>.npz`: `verts`, `faces`, `label`, `op` (`carve`, `remesh`, `fit`, `cut`, `smooth`,
  `finish`), `info` (json). Steps are append-only. `build/log.jsonl` has one line per step with metrics.

## case.json

`input` (file, sha256, size, kind), `subject`, `work` (size, aspect, pad box), `settings` (`res`, `hull_res`,
`budget_usd`, `fit_scale`, `symmetry`: auto/on/off, `view_set`, `plan`), `front` (K, unit_scale, center, sources),
`views` (per view: `requested`, `params`, `role` = train | holdout | rejected, `reason`, `iou_hull`, `source`,
`prompt`), `symmetry` (iou, yaw, normal, point, used), `hull`, `fal` (endpoint, request_id, purpose, est_usd),
`status` (stage → timestamp).

Exports (`out/sculpt.glb/.obj/.stl` from Blender) are centred on the vertical axis through the subject centre
with the lowest point at z = 0, Z up. `out/sculpt.obj`/`.stl` written by `finish` before Blender runs are in the
world frame above.
