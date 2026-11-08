"""A case: one input image and everything derived from it, in one folder.

  case.json        input facts, subject, working size, front camera, centre, symmetry, views, fal ledger, status
  CONTRACT.md      what this sculpt is judged against (from templates/)
  input/           the original file, untouched (sha256 in case.json)
  ref/             front.png (the input padded to a generator aspect ratio, working size), front.npz (mask,
                   depth, ncam, K), raw fal outputs
  views/           <name>.png, <name>_mask.png, <name>_normals.png (generated or imported views), raw/
  parts/           <label>.png front-image region masks (SAM-3 text prompts or boxes) for focus/cut/smooth
  build/           steps/NNNN_<label>.npz (every recorded state), log.jsonl, field.npz, mesh.npz
  checks/          map sheets, metrics json, the per-pass chart
  blender/         sculpt.blend, frames/
  out/             timelapse.mp4, turntable.mp4, sculpt.glb/.obj/.stl, views.png, REPORT.md, index.html
"""
import datetime as dt
import hashlib
import json
import os
import re
import shutil

import numpy as np
from PIL import Image

from . import geom as G

ASPECTS = {"21:9": 21 / 9, "16:9": 16 / 9, "3:2": 1.5, "4:3": 4 / 3, "5:4": 1.25, "1:1": 1.0, "4:5": 0.8,
           "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16}
SKILL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATES = os.path.join(SKILL, "templates")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_label(s):
    """a region / part name safe to use as a file name: lowercase letters, digits and underscores only"""
    out = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    if not out:
        raise SystemExit(f"invalid name {s!r}: use letters and digits")
    return out


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


class Case:
    def __init__(self, path):
        self.root = os.path.abspath(path)
        self.cfg_path = os.path.join(self.root, "case.json")
        if not os.path.exists(self.cfg_path):
            raise SystemExit(f"no case at {self.root} (run: i2s.py new {path} --input IMAGE)")
        self.cfg = json.load(open(self.cfg_path))
        self._front = None

    # -------------------------------------------------------------- paths / persistence
    def p(self, *parts, mk=False):
        path = os.path.join(self.root, *parts)
        if mk:
            os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
        return path

    def save(self):
        tmp = self.cfg_path + ".tmp"
        json.dump(self.cfg, open(tmp, "w"), indent=1)
        os.replace(tmp, self.cfg_path)

    def status(self, key, value=True):
        self.cfg.setdefault("status", {})[key] = value if value is not True else now()
        self.save()

    @property
    def size(self):
        return tuple(self.cfg["work"]["size"])  # (W, H)

    @property
    def subject(self):
        return self.cfg.get("subject") or "subject"

    # -------------------------------------------------------------- front + cameras
    def front(self):
        if self._front is None:
            self._front = dict(np.load(self.p("ref", "front.npz")))
        return self._front

    @property
    def K(self):
        return tuple(self.cfg["front"]["K"])

    @property
    def center(self):
        return np.array(self.cfg["front"]["center"], float)

    def cam(self, name, scale=1.0):
        params = {"az": 0, "el": 0, "s": 1, "du": 0, "dv": 0}
        if name != "front":
            v = self.cfg["views"][name]
            req = v.get("requested") or {}
            params = v.get("params") or dict(params, az=req.get("az") or 0, el=req.get("el") or 0)
        return G.scale_cam(G.make_cam(params, self.center, self.K), scale)

    def views(self, roles=("train",)):
        return [n for n, v in self.cfg.get("views", {}).items() if v.get("role") in roles]

    # -------------------------------------------------------------- build record
    def steps_dir(self):
        return self.p("build", "steps", mk=True)

    def n_steps(self):
        d = self.steps_dir()
        return len([f for f in os.listdir(d) if f.endswith(".npz")])

    def write_step(self, V, Fc, label, op, info=None):
        """record one state of the sculpt (every op the timelapse and the Blender replay show)"""
        k = self.n_steps()
        slug = "".join(c if c.isalnum() or c in "-+" else "_" for c in label)[:48]
        path = os.path.join(self.steps_dir(), f"{k:04d}_{slug}.npz")
        np.savez_compressed(path, verts=V.astype(np.float32), faces=Fc.astype(np.int32), label=label, op=op,
                            info=json.dumps(info or {}))
        self.log({"step": k, "op": op, "label": label, "verts": int(len(V)), **(info or {})})
        return path

    def log(self, entry):
        entry = {"t": now(), **entry}
        with open(self.p("build", "log.jsonl", mk=True), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_log(self):
        p = self.p("build", "log.jsonl")
        return [json.loads(x) for x in open(p)] if os.path.exists(p) else []

    def mesh(self):
        z = np.load(self.p("build", "mesh.npz"))
        return z["verts"].astype(float), z["faces"].astype(np.int64)

    def save_mesh(self, V, Fc):
        np.savez_compressed(self.p("build", "mesh.npz", mk=True), verts=V.astype(np.float32), faces=Fc.astype(np.int32))

    # -------------------------------------------------------------- fal ledger
    def fal_spent(self):
        return float(sum(e.get("est_usd", 0.0) for e in self.cfg.get("fal", [])))


def nearest_aspect(w, h):
    r = w / h
    return min(ASPECTS.items(), key=lambda kv: abs(np.log(kv[1] / r)))


def border_colour(a):
    b = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(x) for x in np.median(b, 0))


def normalise_input(src, long_side=1024):
    """-> (RGB image padded to the nearest generator aspect ratio and resized to the working size, alpha mask or
    None, aspect name, pad box). Padding keeps the subject's pixels unscaled relative to each other, so every
    generated view can share the front camera's intrinsics."""
    im = Image.open(src)
    alpha = None
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        a = np.asarray(im)[..., 3]
        if (a < 250).mean() > 0.02:  # a real cut-out
            alpha = a
        bg = Image.new("RGBA", im.size, (208, 208, 210, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    name, r = nearest_aspect(w, h)
    if w / h < r:
        W0, H0 = int(round(h * r)), h
    else:
        W0, H0 = w, int(round(w / r))
    arr = np.asarray(im)
    canvas = np.empty((H0, W0, 3), np.uint8)
    canvas[:] = border_colour(arr)
    x0, y0 = (W0 - w) // 2, (H0 - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    sc = long_side / max(W0, H0)
    W, H = int(round(W0 * sc / 8)) * 8, int(round(H0 * sc / 8)) * 8
    out = Image.fromarray(canvas).resize((W, H), Image.LANCZOS)
    am = None
    if alpha is not None:
        A = np.zeros((H0, W0), np.uint8)
        A[y0:y0 + h, x0:x0 + w] = alpha
        am = np.asarray(Image.fromarray(A).resize((W, H), Image.BILINEAR)) > 127
    return out, am, name, [x0, y0, w, h, W0, H0]


def new_case(path, image, subject=None, kind="auto", long_side=1024, budget=4.0):
    root = os.path.abspath(path)
    if os.path.exists(os.path.join(root, "case.json")):
        raise SystemExit(f"{root} already has a case.json; pick a new folder")
    for d in ("input", "ref/raw", "views/raw", "parts", "build/steps", "checks", "blender/frames", "out"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    dst = os.path.join(root, "input", os.path.basename(image))
    shutil.copy2(image, dst)
    img, alpha, aspect, pad = normalise_input(dst, long_side)
    img.save(os.path.join(root, "ref", "front.png"))
    if alpha is not None:
        Image.fromarray(alpha.astype(np.uint8) * 255).save(os.path.join(root, "ref", "alpha_mask.png"))
    W, H = img.size
    src = Image.open(dst)
    cfg = {
        "version": 1,
        "created": now(),
        "input": {"file": os.path.relpath(dst, root), "sha256": sha256(dst), "size": list(src.size),
                  "mode": src.mode, "kind": kind, "has_alpha_cutout": alpha is not None},
        "subject": subject or "",
        "work": {"size": [W, H], "aspect": aspect, "pad_box": pad},
        "settings": {"res": 400, "hull_res": 160, "budget_usd": budget},
        "views": {},
        "fal": [],
        "status": {"new": now()},
    }
    json.dump(cfg, open(os.path.join(root, "case.json"), "w"), indent=1)
    for t in ("CONTRACT.md", "acceptance.json"):
        s = os.path.join(TEMPLATES, t)
        if os.path.exists(s):
            txt = open(s).read().replace("{{SUBJECT}}", subject or "the subject").replace(
                "{{INPUT}}", os.path.basename(image)).replace("{{SHA256}}", cfg["input"]["sha256"])
            open(os.path.join(root, t), "w").write(txt)
    return Case(root)
