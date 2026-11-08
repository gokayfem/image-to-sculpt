#!/usr/bin/env python3
"""image-to-sculpt CLI: one image of any subject -> a recorded, measured 3D sculpt in Blender.

  i2s.py doctor                                  check Python packages, Blender, ffmpeg, FAL_KEY (never printed)
  i2s.py imagine OUT.png "an idea"               an input image from text (idea-only requests)
  i2s.py new CASE --input IMG [--subject TEXT]   scaffold a case (input copied, sha256, padded working image)
  i2s.py prep CASE [--select TEXT] [--restyle]   front maps: mask, metric depth + focal length, normals
  i2s.py views CASE [--set standard]             generated views (+ masks, normals); or --import DIR
  i2s.py parts CASE --prompt hands --prompt face front-image regions for focus / cut / smooth
  i2s.py hull CASE                               register every view, screen, carve the visual hull (recorded)
  i2s.py fit CASE --stage forms|surface|detail|polish|focus|all [--region R] [--passes N]
  i2s.py cut CASE [--region R] [--views front|all]   silhouette cut: opens gaps the views see
  i2s.py smooth CASE [--radius VOX] [--region R]      opening: removes webs, flaps, specks
  i2s.py symmetry CASE on|off|auto              use the detected mirror plane or not (after hull)
  i2s.py check CASE [--tag T]                    map sheet + metrics for every view, held-out included
  i2s.py finish CASE                             final smoothing, OBJ/STL
  i2s.py blender CASE                            replay every step in Blender, turntable, views, .blend, GLB
  i2s.py film CASE                               timelapse.mp4 (+360), turntable.mp4, views.png, hero.png
  i2s.py report CASE                             REPORT.md, per-step chart, index.html
  i2s.py run CASE                                everything after `new`, in order (skips finished stages)
  i2s.py demo DIR [--subject figure|creature]    the whole pipeline on a synthetic subject, offline
  i2s.py status CASE
Regions R: a parts/ label ('hands'), 'box:x0,y0,x1,y1' (fractions or pixels), several joined with '+'.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from i2s import falio  # noqa: E402
from i2s.case import Case, new_case  # noqa: E402


def say(*a):
    print(*a, flush=True)


def cmd_doctor(a):
    ok = True
    for mod in ("numpy", "scipy", "skimage", "PIL", "fal_client", "matplotlib"):
        try:
            __import__(mod)
            say(f"  ok   {mod}")
        except ImportError:
            say(f"  {'MISSING' if mod not in ('fal_client', 'matplotlib') else 'optional'} {mod}")
            ok &= mod in ("fal_client", "matplotlib")
    from i2s.film import find_blender
    b = find_blender()
    if b:
        v = subprocess.run([b, "-b", "--version"], capture_output=True, text=True).stdout.splitlines()
        say(f"  ok   blender {v[0] if v else b}")
    else:
        say("  MISSING blender (set $BLENDER): needed for `blender` and `film`")
    say(f"  {'ok  ' if shutil.which('ffmpeg') else 'MISSING'} ffmpeg")
    say(f"  {'ok  ' if falio.have_key() else 'missing'} FAL_KEY {'(set; not shown)' if falio.have_key() else '(offline: alpha-mask inputs and imported views only)'}")
    return 0 if ok else 1


def cmd_imagine(a):
    from i2s.inputs import imagine
    say(imagine(a.prompt, a.out, a.aspect))


def cmd_new(a):
    c = new_case(a.case, a.input, a.subject, a.kind, a.long_side, a.budget)
    say(f"case {c.root}: working image {c.size[0]}x{c.size[1]} ({c.cfg['work']['aspect']}), "
        f"{'alpha cut-out' if c.cfg['input']['has_alpha_cutout'] else 'no alpha'}")


def cmd_prep(a):
    from i2s.inputs import prep
    f = prep(Case(a.case), a.select, a.restyle)
    say(json.dumps({k: v for k, v in f.items() if k != "K"}, indent=1))


def cmd_views(a):
    from i2s import inputs
    c = Case(a.case)
    if a.import_dir:
        say("imported", inputs.import_views(c, a.import_dir, a.holdout))
        return
    angles = None
    if a.angles:
        angles = [tuple(int(x) for x in s.split(":")) for s in a.angles.split(",")]
    say("generated", inputs.generate(c, a.set, angles, holdout=not a.no_holdout))
    say(f"fal spent (est.) ${c.fal_spent():.2f}")


def cmd_parts(a):
    from i2s.inputs import parts
    say(parts(Case(a.case), a.prompt or (), a.box or ()))


def cmd_hull(a):
    from i2s import hull
    c = Case(a.case)
    hull.build(c, log=say)


def cmd_register(a):
    from i2s import maps as MP
    from i2s import register as RG
    c = Case(a.case)
    V, Fc = c.mesh()
    for n in a.views or c.views(("train", "holdout")):
        if c.cfg["views"][n]["role"] == "holdout" and not a.include_holdout:
            continue
        vw = MP.View(c, n, 1.0)
        p, info = RG.refine(V, Fc, vw.mask, vw.ncam, c.center, c.K, c.cfg["views"][n]["params"])
        c.cfg["views"][n]["params"] = p
        say(f"  {n}: {info}")
    c.save()


def cmd_fit(a):
    from i2s import fit
    c = Case(a.case)
    stages = ["forms", "surface", "detail"] if a.stage == "all" else [a.stage]
    for s in stages:
        fit.run_stage(c, s, a.passes, a.region, sheet_every=a.sheet_every, log=say)


def cmd_cut(a):
    from i2s import fit
    fit.run_op(Case(a.case), "cut", a.region, a.views, log=say)


def cmd_smooth(a):
    from i2s import fit
    fit.run_op(Case(a.case), "smooth", a.region, radius=a.radius, log=say)


def cmd_check(a):
    from i2s import maps as MP
    c = Case(a.case)
    V, Fc = c.mesh()
    vs = MP.load_views(c, a.scale)
    res, maps = MP.evaluate(V, Fc, vs, want_maps=True)
    tag = a.tag or f"{c.n_steps():04d}"
    json.dump(res, open(c.p("checks", f"metrics_{tag}.json"), "w"), indent=1)
    out = MP.sheet(vs, res, maps, c.p("checks", f"sheet_{tag}.png"), f"check {tag}")
    for k in ("train_mean", "holdout_mean"):
        if k in res:
            say(k, {m: round(v, 3) for m, v in res[k].items()})
    say(out)


def cmd_finish(a):
    from i2s import fit
    fit.finish(Case(a.case), log=say)


def cmd_blender(a):
    from i2s import film
    film.run_blender(Case(a.case), a.width, a.frames, a.what.split(",") if a.what else None, log=say)


def cmd_film(a):
    from i2s import film
    c = Case(a.case)
    film.timelapse(c, log=say)
    m = {}
    p = c.p("checks", "metrics_final.json")
    if os.path.exists(p):
        m = json.load(open(p))
    say(film.views_sheet(c, m), film.hero(c))


def cmd_report(a):
    from i2s import report
    report.build(Case(a.case), log=say)


def run_all(c, log=say, quick=False):
    from i2s import film, fit, hull, inputs, report
    st = c.cfg.get("status", {})
    t0 = time.time()
    if "prep" not in st:
        log("== prep")
        inputs.prep(c)
    if not c.cfg.get("views") and falio.have_key() and c.cfg["settings"].get("generate_views", True):
        log("== views")
        inputs.generate(c, c.cfg["settings"].get("view_set", "standard"))
    if "hull" not in st:
        log("== hull")
        hull.build(c, log=log)
    plan = c.cfg["settings"].get("plan") or [["fit", "forms"], ["fit", "surface"], ["fit", "detail"], ["cut", "front"],
                                             ["smooth", None], ["fit", "polish"]]
    for k, (op, arg) in enumerate(plan):
        key = f"plan_{k}_{op}_{arg}"
        if key in st:
            continue
        log(f"== {op} {arg or ''}")
        if op == "fit":
            fit.run_stage(c, arg, (2 if quick else None), sheet_every=-1, log=log)
        elif op == "cut":
            fit.run_op(c, "cut", which=arg, log=log)
        elif op == "smooth":
            fit.run_op(c, "smooth", log=log)
        c.status(key)
    log("== finish")
    fit.finish(c, log=log)
    if film.find_blender() and shutil.which("ffmpeg"):
        log("== blender")
        film.run_blender(c, 480 if quick else 640, 72 if quick else 180, log=log)
        log("== film")
        film.timelapse(c, log=log)
        m = json.load(open(c.p("checks", "metrics_final.json")))
        film.views_sheet(c, m)
        film.hero(c)
    else:
        log("   (no Blender or ffmpeg: skipping replay and films)")
    log("== report")
    report.build(c, log=log)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {c.p('out')}")


def cmd_run(a):
    run_all(Case(a.case), quick=a.quick)


def cmd_demo(a):
    from i2s import synth
    say(f"building a synthetic '{a.subject}' case (known shape, generated-looking views with wrong angles)")
    kw = dict(size=(288, 384), res=128, hull_res=80) if a.quick else {}
    c, _ = synth.make_case(a.dir, a.subject, bad_view=True, **kw)
    run_all(c, quick=a.quick)


def cmd_symmetry(a):
    c = Case(a.case)
    sym = c.cfg.get("symmetry")
    if not sym:
        raise SystemExit("no symmetry plane yet: run hull first")
    c.cfg["settings"]["symmetry"] = a.mode
    sym["used"] = bool(sym["symmetric"]) if a.mode == "auto" else a.mode == "on"
    if a.why:
        sym["why"] = a.why
    c.save()
    say(f"symmetry {'on' if sym['used'] else 'off'} (plane yaw {sym['yaw_deg']:.1f} deg, mirror IoU {sym['iou']:.3f})")


def cmd_status(a):
    c = Case(a.case)
    say(json.dumps({"status": c.cfg.get("status"), "steps": c.n_steps(), "views": {
        n: (v.get("role"), v.get("params", {}).get("az")) for n, v in c.cfg.get("views", {}).items()},
        "symmetry": c.cfg.get("symmetry", {}).get("used"), "fal_usd_est": round(c.fal_spent(), 2)}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="i2s.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sp.add_parser("imagine")
    p.add_argument("out")
    p.add_argument("prompt")
    p.add_argument("--aspect", default="3:4")
    p.set_defaults(fn=cmd_imagine)
    p = sp.add_parser("new")
    p.add_argument("case")
    p.add_argument("--input", required=True)
    p.add_argument("--subject", help="a short noun phrase, e.g. 'stocky cartoon troll' (used in view prompts)")
    p.add_argument("--kind", default="auto", choices=["auto", "render", "ai-image", "photo", "drawing", "synthetic"])
    p.add_argument("--long-side", type=int, default=1024)
    p.add_argument("--budget", type=float, default=4.0, help="fal budget for this case, USD (estimates)")
    p.set_defaults(fn=cmd_new)
    p = sp.add_parser("prep")
    p.add_argument("case")
    p.add_argument("--select", help="text prompt to pick one subject in a busy image (SAM-3)")
    p.add_argument("--restyle", action="store_true", help="drawing/sketch: make a 3D render of it first")
    p.set_defaults(fn=cmd_prep)
    p = sp.add_parser("views")
    p.add_argument("case")
    p.add_argument("--set", default="standard", choices=["minimal", "standard", "full"])
    p.add_argument("--angles", help="extra angles 'az:el,az:el' (e.g. '-30:20,150:0')")
    p.add_argument("--no-holdout", action="store_true")
    p.add_argument("--import", dest="import_dir", help="folder of your own views (photos/renders of the subject)")
    p.add_argument("--holdout", help="with --import: the file to hold out")
    p.set_defaults(fn=cmd_views)
    p = sp.add_parser("parts")
    p.add_argument("case")
    p.add_argument("--prompt", action="append")
    p.add_argument("--box", action="append", help="label:x0,y0,x1,y1")
    p.set_defaults(fn=cmd_parts)
    for name, fn in (("hull", cmd_hull), ("finish", cmd_finish), ("report", cmd_report), ("film", cmd_film),
                     ("status", cmd_status)):
        p = sp.add_parser(name)
        p.add_argument("case")
        p.set_defaults(fn=fn)
    p = sp.add_parser("symmetry")
    p.add_argument("case")
    p.add_argument("mode", choices=["on", "off", "auto"])
    p.add_argument("--why", help="recorded in case.json")
    p.set_defaults(fn=cmd_symmetry)
    p = sp.add_parser("register")
    p.add_argument("case")
    p.add_argument("--views", nargs="*")
    p.add_argument("--include-holdout", action="store_true")
    p.set_defaults(fn=cmd_register)
    p = sp.add_parser("fit")
    p.add_argument("case")
    p.add_argument("--stage", default="all", choices=["forms", "surface", "detail", "polish", "focus", "all"])
    p.add_argument("--passes", type=int)
    p.add_argument("--region")
    p.add_argument("--sheet-every", type=int, default=0)
    p.set_defaults(fn=cmd_fit)
    p = sp.add_parser("cut")
    p.add_argument("case")
    p.add_argument("--region")
    p.add_argument("--views", default="front", choices=["front", "all"])
    p.set_defaults(fn=cmd_cut)
    p = sp.add_parser("smooth")
    p.add_argument("case")
    p.add_argument("--radius", type=float, help="opening radius in voxels (default 1.2)")
    p.add_argument("--region")
    p.set_defaults(fn=cmd_smooth)
    p = sp.add_parser("check")
    p.add_argument("case")
    p.add_argument("--tag")
    p.add_argument("--scale", type=float, default=0.5)
    p.set_defaults(fn=cmd_check)
    p = sp.add_parser("blender")
    p.add_argument("case")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--what", help="comma list: timelapse,turntable,views,export")
    p.set_defaults(fn=cmd_blender)
    p = sp.add_parser("run")
    p.add_argument("case")
    p.add_argument("--quick", action="store_true", help="2 passes per stage, small renders (a smoke test)")
    p.set_defaults(fn=cmd_run)
    p = sp.add_parser("demo")
    p.add_argument("dir")
    p.add_argument("--subject", default="figure", choices=["figure", "creature"])
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
