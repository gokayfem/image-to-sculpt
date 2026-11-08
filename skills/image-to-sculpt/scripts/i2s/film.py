"""Blender job + films and sheets.

  blender   write blender/job.json (cameras converted here, so the Blender script needs only bpy + numpy), run
            Blender headless: every recorded step from the photo's camera, a 360-degree turntable, each view
  film      out/timelapse.mp4: a title card (frame 0 is a finished thumbnail), then [reference | sculpt] for every
            recorded step, a hold, the finished sculpt turning 360 degrees, a hold. out/turntable.mp4.
  sheets    out/views.png (each view: reference | sculpt from its registered camera), out/hero.png
"""
import glob
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from . import geom as G
from .maps import font

BG = (14, 14, 16)


def find_blender():
    for c in (os.environ.get("BLENDER"), shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if c and os.path.exists(c):
            return c
    return None


def steps(case):
    return sorted(glob.glob(os.path.join(case.steps_dir(), "*.npz")))


def step_label(path):
    z = np.load(path)
    return str(z["label"])


def job(case, width=640, turntable_frames=180, what=None):
    W, H = case.size
    sc = width / W
    fr = case.p("blender", "frames", mk=True)
    cams = {"front": G.blender_camera(case.cam("front"), W, H), "views": {}}
    for n, v in case.cfg["views"].items():
        if v.get("params") and v.get("role") != "rejected":
            cams["views"][n] = G.blender_camera(case.cam(n), W, H)
    c = case.center
    fx, fy, _, _ = case.K
    zoom = 0.8
    K = (fx * zoom, fy * zoom, W / 2 - fx * zoom * c[0] / c[1], H / 2 + fy * zoom * c[2] / c[1])
    cams["turntable"] = G.blender_camera(G.make_cam({}, c, K), W, H)
    final = case.p("build", "final_mesh.npz")
    if not os.path.exists(final):
        final = case.p("build", "mesh.npz")
    j = {"steps": [{"path": p, "label": step_label(p)} for p in steps(case)], "final": final, "cameras": cams,
         "center": c.tolist(), "reference": case.p("ref", "front.png"), "frames_dir": fr, "scale": sc,
         "turntable_frames": turntable_frames, "blend": case.p("blender", "sculpt.blend"),
         "out_dir": case.p("out", mk=True), "what": what or ["timelapse", "turntable", "views", "export"]}
    path = case.p("blender", "job.json")
    json.dump(j, open(path, "w"), indent=1)
    return path


def run_blender(case, width=640, turntable_frames=180, what=None, log=print):
    exe = find_blender()
    if not exe:
        raise SystemExit("Blender not found: set $BLENDER or put blender on PATH (Blender 5.2+)")
    jp = job(case, width, turntable_frames, what)
    for f in glob.glob(case.p("blender", "frames", "*.png")):
        if what is None or any(os.path.basename(f).startswith({"timelapse": "tl_", "turntable": "tt_",
                                                               "views": "view_"}.get(w, "~")) for w in what):
            os.remove(f)
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blender", "build.py")
    with open(case.p("blender", "blender.log"), "w") as lf:
        r = subprocess.run([exe, "-b", "--factory-startup", "--python", script, "--", jp], stdout=lf,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"Blender failed (see {case.p('blender', 'blender.log')})")
    case.status("blender")
    log(f"  Blender: {len(glob.glob(case.p('blender', 'frames', 'tl_*.png')))} step frames, "
        f"{len(glob.glob(case.p('blender', 'frames', 'tt_*.png')))} turntable frames -> {case.p('blender', 'sculpt.blend')}")


def _panel(ref, frame, head, title, sub):
    W, H = frame.size
    c = Image.new("RGB", (2 * W, H + head), BG)
    c.paste(ref.resize((W, H), Image.LANCZOS), (0, head))
    c.paste(frame, (W, head))
    d = ImageDraw.Draw(c)
    d.text((16, 10), title, font=font(max(16, W // 28)), fill=(235, 235, 235))
    d.text((16, 14 + max(16, W // 28) + 4), sub[:150], font=font(max(12, W // 42)), fill=(175, 175, 175))
    return c


def _encode(frames_dir, out, fps=24):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)


def timelapse(case, fps=24, build_seconds=12.0, log=print):
    fr = case.p("blender", "frames")
    tl = sorted(glob.glob(os.path.join(fr, "tl_*.png")))
    tt = sorted(glob.glob(os.path.join(fr, "tt_*.png")))
    if not tl:
        raise SystemExit("no step frames: run i2s.py blender CASE first")
    labels = [step_label(p) for p in steps(case)]
    W, H = Image.open(tl[0]).size
    ref = Image.open(case.p("ref", "front.png")).convert("RGB")
    head = max(56, H // 9)
    tmp = case.p("blender", "film_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n = 0
    subj = case.cfg.get("subject") or "subject"
    nviews = len(case.views(("train",)))

    def put(img, k=1):
        nonlocal n
        for _ in range(k):
            img.save(os.path.join(tmp, f"{n:05d}.png"))
            n += 1
    hero = Image.open(tt[0] if tt else tl[-1]).convert("RGB").resize((W, H), Image.LANCZOS)
    put(_panel(ref, hero, head, f"image-to-sculpt · {subj}",
               f"one image -> {len(tl)} recorded steps, {nviews} views, fitted to every map"), int(fps * 0.75))
    hold = int(np.clip(round(build_seconds * fps / len(tl)), 1, 6))
    for k, p in enumerate(tl):
        put(_panel(ref, Image.open(p).convert("RGB"), head, f"image-to-sculpt · step {k + 1}/{len(tl)}",
                   labels[k] if k < len(labels) else ""), hold)
    last = Image.open(os.path.join(tmp, f"{n - 1:05d}.png"))
    put(last, int(fps * 1.5))
    for p in tt:
        put(_panel(ref, Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), head,
                   "image-to-sculpt · final sculpt, 360°", f"{len(tl)} recorded steps · {nviews} views"))
    if tt:
        put(Image.open(os.path.join(tmp, f"{n - 1:05d}.png")), fps)
    out = case.p("out", "timelapse.mp4", mk=True)
    _encode(tmp, out, fps)
    shutil.rmtree(tmp, ignore_errors=True)
    if tt:
        tmp2 = case.p("blender", "tt_tmp")
        shutil.rmtree(tmp2, ignore_errors=True)
        os.makedirs(tmp2)
        for k, p in enumerate(tt):
            shutil.copy(p, os.path.join(tmp2, f"{k:05d}.png"))
        _encode(tmp2, case.p("out", "turntable.mp4"), fps)
        shutil.rmtree(tmp2, ignore_errors=True)
    log(f"  film: {out} ({n} frames, {n / fps:.1f} s)")
    return out


def views_sheet(case, metrics=None, tile_h=300):
    fr = case.p("blender", "frames")
    names = ["front"] + [n for n, v in case.cfg["views"].items() if v.get("role") in ("train", "holdout")]
    tiles = []
    for n in names:
        rp = case.p("ref", "front.png") if n == "front" else case.p("views", f"{n}.png")
        sp = os.path.join(fr, f"view_{n}.png") if n != "front" else None
        if n == "front":
            tl = sorted(glob.glob(os.path.join(fr, "tl_*.png")))
            sp = tl[-1] if tl else None
        if not sp or not os.path.exists(sp):
            continue
        a = Image.open(rp).convert("RGB")
        b = Image.open(sp).convert("RGB").resize(a.size, Image.LANCZOS)
        w = int(a.width * tile_h / a.height)
        pair = Image.new("RGB", (2 * w + 6, tile_h + 30), BG)
        pair.paste(a.resize((w, tile_h)), (0, 30))
        pair.paste(b.resize((w, tile_h)), (w + 6, 30))
        role = "photo" if n == "front" else case.cfg["views"][n]["role"]
        m = (metrics or {}).get(n, {})
        txt = f"{n} [{role}]" + (f"  IoU {m['iou']:.3f}  normals {m.get('normal_med', float('nan')):.1f}°" if m else "")
        ImageDraw.Draw(pair).text((6, 6), txt, font=font(15), fill=(220, 220, 220))
        tiles.append(pair)
    if not tiles:
        return None
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    S = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 10) + 10), BG)
    for i, t in enumerate(tiles):
        S.paste(t, (10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 10)))
    out = case.p("out", "views.png")
    S.save(out)
    return out


def hero(case, height=520):
    fr = case.p("blender", "frames")
    tt = sorted(glob.glob(os.path.join(fr, "tt_*.png")))
    if not tt:
        return None
    n = len(tt)
    picks = [tt[0], tt[n // 8], tt[n // 4], tt[n // 2]]
    ref = Image.open(case.p("ref", "front.png")).convert("RGB")
    ims = [ref] + [Image.open(p).convert("RGB") for p in picks]
    ims = [i.resize((int(i.width * height / i.height), height), Image.LANCZOS) for i in ims]
    S = Image.new("RGB", (sum(i.width for i in ims) + 8 * (len(ims) + 1), height + 16), BG)
    x = 8
    for i in ims:
        S.paste(i, (x, 8))
        x += i.width + 8
    out = case.p("out", "hero.png")
    S.save(out)
    return out
