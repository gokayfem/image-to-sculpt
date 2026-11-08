# Blender and delivery

## Replay (`i2s.py blender CASE`)

`film.job` writes `blender/job.json`: every step mesh, the final mesh, and the cameras already converted to
Blender (`matrix_world`, lens, shift), so `scripts/blender/build.py` needs only `bpy` and `numpy`. Blender 5.2+
runs headless:

```bash
blender -b --factory-startup --python skills/image-to-sculpt/scripts/blender/build.py -- CASE/blender/job.json
```

It renders (Workbench, clay matcap with cavity, 640 px wide by default):

- `frames/tl_NNNN.png`: every recorded step from the photo's camera (the timelapse);
- `frames/tt_NNNN.png`: 180 frames of a 360° turn about the vertical axis through the subject centre, from a
  slightly wider copy of the photo's camera;
- `frames/view_<name>.png`: the final sculpt from each registered view's camera;

and saves `blender/sculpt.blend` (the sculpt, the photo as the camera's background image, the turntable keyed
on a pivot empty) and `out/sculpt.glb`, `.obj`, `.stl` (centred, lowest point at z = 0, Z up).

`blender.log` holds Blender's output. Renders run one Blender at a time; do not start a second replay while one
is running.

## Films (`i2s.py film CASE`)

- `out/timelapse.mp4`, 24 fps: a title card (frame 0 is always a finished thumbnail: reference | final), then
  `[reference | sculpt]` for every recorded step (held so the build lasts about 12 s), a 1.5 s hold on the last
  step, the finished sculpt turning 360° (7.5 s), a 1 s hold.
- `out/turntable.mp4`: the turn alone.
- `out/views.png`: each view's image next to the sculpt from that view's camera, with IoU and normal error.
- `out/hero.png`: reference, front, 45°, 90° and 180° of the final sculpt.

## Report (`i2s.py report CASE`)

`out/REPORT.md` and `out/index.html` (a static page with the film, sheets and tables): input facts and sha256,
front sources, recorded steps by kind, symmetry, every view's requested and registered angle and screening
result, the hull and final metrics (training views, held-out, held-out best-fit), the per-step chart
(`checks/fit_passes.png`), the final map sheet, the fal ledger, and what is generated rather than measured.

## Presenting

Show the timelapse, the views sheet and the measured table. Say plainly which parts are weakest (from the sheet,
not from memory) and which surfaces only generated views constrain. Then stop: the user decides whether to
iterate (more views, focus passes) or keep it.
