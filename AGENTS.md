# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini) working on this repository. To use the
skill rather than develop it, see the README.

## Layout

```text
skills/image-to-sculpt/         the skill (this is what gets installed)
  SKILL.md                      entry point: goal, input routing, workflow, hard rules, reference routing
  references/                   inputs, views, fitting, checking, blender-and-delivery, conventions, lessons
  templates/                    CONTRACT.md, acceptance.json (copied into every case)
  scripts/i2s.py                CLI: doctor | imagine | new | prep | views | parts | hull | symmetry | register |
                                fit | cut | smooth | check | finish | blender | film | report | run | demo | status
  scripts/i2s/                  importable library (numpy, scipy, scikit-image, Pillow; fal-client optional)
    case        case folder, case.json, recorded steps, input normalisation
    falio       every paid call: budget, request id saved before waiting
    inputs      front maps (mask, MoGe depth + focal, Marigold normals), generated / imported views, parts
    geom        cameras, mesh helpers, the z-buffer rasteriser
    sdf         exact signed distance grid: marching cubes, closest-point distances, opening, resampling
    register    view registration (left/right swaps found, front/back kept) and refinement
    hull        inflated proxy, recorded visual-hull carve, symmetry plane
    fit         one map-fit pass, stages, cut / smooth / focus ops, finish and exports
    maps        per-view reference maps, the map checker and sheets
    film        Blender job + run, timelapse and turntable films, views sheet, hero image
    report      REPORT.md, per-step chart, review page
    synth       synthetic subjects rendered into case files (tests, `i2s.py demo`)
  scripts/blender/build.py      runs inside Blender 5.2+: replay steps, turntable, views, .blend, GLB/OBJ/STL
tests/                          python3 tests/run_tests.py [--slow]
adapters/                       copy-in prompts for tools without skill support
gallery/                        README input images (inputs only, never outputs)
.claude-plugin/                 Claude Code plugin + marketplace manifests
```

## Commands

```bash
python3 tests/run_tests.py                                  # fast tests (~1 min)
python3 tests/run_tests.py --slow                           # + the end-to-end synthetic pipeline
python3 skills/image-to-sculpt/scripts/i2s.py --help
python3 skills/image-to-sculpt/scripts/i2s.py demo /tmp/i2s-demo --quick   # whole pipeline offline
```

Blender 5.2+ runs as `blender -b --factory-startup --python scripts/blender/build.py -- CASE/blender/job.json`
(`i2s.py blender CASE` writes the job and runs it).

## Rules for changes

- Input-agnostic: no hardcoded subjects, part names, paths or angles in `skills/`. Subject-specific choices live in
  a case's own folder (`case.json` → `settings.plan`, regions in `parts/`), never in this repository.
- No outputs in the repository (cases, renders, meshes, films, reports): cases live outside it (`cases/` is
  git-ignored). Input images for the README are the only images, in `gallery/`.
- Every new module gets a test in `tests/` against a closed form or a synthetic subject with known shape.
- Keep the data contracts in `references/conventions.md` stable (camera params, map encodings, `field.npz`, step
  files); the Blender script and the report depend on them.
- Keep the distance field exact after every operation (`sdf.consistent` / `sdf.reinit`).
- No secrets in files or logs. `FAL_KEY` is read from the environment and never printed.
