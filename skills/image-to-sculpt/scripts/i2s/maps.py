"""Reference maps per view and the map checker.

Measures, per view (sculpt rendered from the view's registered camera, against that view's maps):
  iou            silhouette intersection over union
  sil_px         mean distance (px, working res) from the sculpt's outline to the reference outline
  normal_med     median angle (deg) between the sculpt's normals and the reference normal map, inside both masks
  normal_p15     % of those pixels under 15 deg
  depth_mm       front photo only: mean depth error in thousandths of the subject size (metric MoGe depth)
  rough_ratio    roughness (angle between a normal map and its 3 px blur) of the sculpt / of the reference;
                 1 = as smooth as the reference, > 1 = lumpier, < 1 = smoother
The held-out view is measured exactly like the others and never used to fit.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from . import geom as G

ROUGH_SIGMA = 3.0


def roughness(ncam, mask, sigma=ROUGH_SIGMA):
    m = mask.astype(np.float32)
    nb = np.stack([ndi.gaussian_filter(ncam[..., k] * m, sigma) for k in range(3)], -1)
    nb = G.unit(nb)
    ang = np.degrees(np.arccos(np.clip((nb * ncam).sum(-1), -1, 1)))
    inner = ndi.binary_erosion(mask, iterations=int(2 * sigma + 2))
    return np.where(inner, ang, np.nan)


def _resize_mask(m, size):
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127


def _resize_normals(n, size):
    out = np.stack([np.asarray(Image.fromarray(n[..., k].astype(np.float32)).resize(size, Image.BILINEAR))
                    for k in range(3)], -1)
    return G.unit(out)


class View:
    """one view's reference maps at `scale` x the working resolution, and its registered camera"""

    def __init__(self, case, name, scale=1.0, trust=None):
        W0, H0 = case.size
        self.W, self.H = int(round(W0 * scale)), int(round(H0 * scale))
        size = (self.W, self.H)
        self.name, self.scale = name, scale
        info = {} if name == "front" else case.cfg["views"][name]
        self.role = "front" if name == "front" else info.get("role", "train")
        if name == "front":
            T = case.front()
            img = Image.open(case.p("ref", "front.png")).convert("RGB")
            mask = T["mask"].astype(bool)
            ncam = T["ncam"] if bool(T.get("has_normals", True)) else None
            depth = T["depth"].astype(float)
            self.metric = bool(np.isfinite(depth).any())
            default_trust = 1.0
        else:
            img = Image.open(case.p("views", f"{name}.png")).convert("RGB")
            mask = np.asarray(Image.open(case.p("views", f"{name}_mask.png")).convert("L")) > 127
            npath = case.p("views", f"{name}_normals.png")
            ncam = None
            if os.path.exists(npath):
                a = np.asarray(Image.open(npath).convert("RGB")).astype(np.float32)
                ncam = G.unit(a / 127.5 - 1.0)
            depth = np.full(mask.shape, np.nan)
            self.metric = False
            el = abs(info.get("params", info.get("requested", {})).get("el", 0) or 0)
            default_trust = 0.6 if el < 15 else 0.45
        self.trust = default_trust if trust is None else trust
        if self.role == "holdout":
            self.trust = 0.0
        self.img = np.asarray(img.resize(size, Image.LANCZOS)) if scale != 1 else np.asarray(img)
        self.mask = _resize_mask(mask, size) if scale != 1 else mask
        if ncam is not None and scale != 1:
            ncam = _resize_normals(ncam, size)
        self.ncam = ncam
        if scale != 1 and self.metric:
            ok = np.isfinite(depth)
            dz = ndi.zoom(np.where(ok, depth, 0), scale, order=1, grid_mode=True, mode="nearest")
            wz = ndi.zoom(ok.astype(float), scale, order=1, grid_mode=True, mode="nearest")
            depth = np.where(wz > 0.99, dz / np.maximum(wz, 1e-6), np.nan)[:self.H, :self.W]
        self.depth = depth if self.metric else np.full((self.H, self.W), np.nan)
        self.dval = self.mask & np.isfinite(self.depth)
        self.set_cam(case.cam(name, scale))
        sdf = ndi.distance_transform_edt(~self.mask) - ndi.distance_transform_edt(self.mask)
        self.sdf_raw = sdf
        self.sdf = ndi.gaussian_filter(sdf, 1.0)
        gy, gx = np.gradient(self.sdf)
        gn = np.maximum(np.hypot(gx, gy), 1e-6)
        self.gx, self.gy = gx / gn, gy / gn
        inner = ndi.binary_erosion(self.mask, iterations=3)
        self.nvalid = ndi.gaussian_filter(inner.astype(np.float32), 1.5) * (self.ncam is not None)
        self.rough_ref = roughness(self.ncam, self.mask) if self.ncam is not None else None

    def set_cam(self, cam):
        self.cam = cam
        if self.ncam is not None:
            R = cam["R"]
            nc = self.ncam
            self.nworld = nc[..., :1] * R[:, 0] + nc[..., 1:2] * R[:, 2] - nc[..., 2:3] * R[:, 1]
        else:
            self.nworld = None


def load_views(case, scale=1.0, roles=("train", "holdout"), front=True):
    out = [View(case, "front", scale)] if front else []
    out += [View(case, n, scale) for n in case.views(roles)]
    return out


def affine_depth(dr, dt, iters=3):
    """robust fit dt ~ a * dr + b"""
    w = np.ones_like(dr)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([dr * w, w], 1)
        a, b = np.linalg.lstsq(A, dt * w, rcond=None)[0]
        r = np.abs(a * dr + b - dt)
        s = max(np.median(r) * 1.5, 1e-4)
        w = 1.0 / np.maximum(1.0, r / s)
    return float(a), float(b)


def compare(view, Rm):
    tm, rm = view.mask, Rm["mask"]
    inter = tm & rm
    out = {"iou": float(inter.sum() / max((tm | rm).sum(), 1))}
    edge = rm & ~ndi.binary_erosion(rm)
    out["sil_px"] = float(np.abs(view.sdf_raw[edge]).mean() / view.scale) if edge.any() else float("nan")
    inner = ndi.binary_erosion(inter, iterations=3)
    ang = None
    if view.ncam is not None and inner.any():
        ang = np.degrees(np.arccos(np.clip((Rm["ncam"] * view.ncam).sum(-1), -1, 1)))
        out["normal_med"] = float(np.median(ang[inner]))
        out["normal_p15"] = float((ang[inner] < 15).mean() * 100)
    if view.metric:
        both = inner & view.dval & np.isfinite(Rm["depth"])
        if both.sum() > 50:
            out["depth_mm"] = float(np.abs(Rm["depth"][both] - view.depth[both]).mean() * 1000)
    rr = None
    if view.rough_ref is not None:
        rr = roughness(Rm["ncam"], rm)
        sel = np.isfinite(rr) & np.isfinite(view.rough_ref)
        if sel.sum() > 100:
            out["rough_ratio"] = float(np.mean(rr[sel]) / max(np.mean(view.rough_ref[sel]), 1e-6))
    return out, ang, rr


def evaluate(V, Fc, views, want_maps=False):
    VN = G.vertex_normals(V, Fc)
    res, maps = {}, {}
    for vw in views:
        Rm = G.raster(V, Fc, vw.cam, vw.W, vw.H, VN=VN)
        m, ang, rr = compare(vw, Rm)
        m["role"] = vw.role
        res[vw.name] = m
        if want_maps:
            maps[vw.name] = (Rm, ang, rr)
    res.update(summarise(res))
    return res, maps


def summarise(res):
    out = {}
    tr = [r for k, r in res.items() if isinstance(r, dict) and r.get("role") == "train"]
    ho = [r for k, r in res.items() if isinstance(r, dict) and r.get("role") == "holdout"]
    for nm, group in (("train_mean", tr), ("holdout_mean", ho)):
        if group:
            keys = {k for r in group for k, v in r.items() if isinstance(v, float)}
            out[nm] = {k: float(np.nanmean([r[k] for r in group if k in r])) for k in sorted(keys)}
    return out


# ----------------------------------------------------------------------------- sheets


def font(size):
    for p in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def ncol(n):
    return np.clip((np.nan_to_num(n) + 1) * 127.5, 0, 255).astype(np.uint8)


def heat(a, vmax=45.0):
    t = np.clip(np.nan_to_num(a) / vmax, 0, 1)
    return (np.stack([t, 1.6 * t * (1 - t) + 0.15 * (1 - t), 1 - t], -1) * 255).astype(np.uint8)


def crop_box(mask, pad=24):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0, 0, mask.shape[1], mask.shape[0]
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, mask.shape[1]),
            min(ys.max() + pad, mask.shape[0]))


def sheet(views, res, maps, out, title, tile_h=260):
    cols = ["reference", "reference normals", "sculpt", "normal error", "silhouette (green missing, pink extra)"]
    rows = []
    for vw in views:
        Rm, ang, _ = maps[vw.name]
        x0, y0, x1, y1 = crop_box(vw.mask | Rm["mask"])
        sl = (slice(y0, y1), slice(x0, x1))
        clay = G.clay_shade(Rm["ncam"], Rm["mask"])
        sil = np.zeros(vw.mask.shape + (3,), np.uint8) + 20
        sil[vw.mask & Rm["mask"]] = (150, 150, 150)
        sil[vw.mask & ~Rm["mask"]] = (40, 220, 90)
        sil[~vw.mask & Rm["mask"]] = (240, 60, 170)
        tiles = [vw.img, ncol(vw.ncam) if vw.ncam is not None else np.zeros_like(vw.img),
                 clay, heat(ang) * (vw.mask & Rm["mask"])[..., None] if ang is not None else np.zeros_like(vw.img), sil]
        ims = []
        for t in tiles:
            im = Image.fromarray(np.ascontiguousarray(t[sl]))
            ims.append(im.resize((max(1, int(im.width * tile_h / im.height)), tile_h), Image.LANCZOS))
        m = res[vw.name]
        txt = (f"{vw.name} [{vw.role}]  IoU {m['iou']:.3f}  outline {m.get('sil_px', float('nan')):.1f}px  "
               f"normals {m.get('normal_med', float('nan')):.1f}° ({m.get('normal_p15', float('nan')):.0f}% <15°)  "
               f"roughness x{m.get('rough_ratio', float('nan')):.2f}"
               + (f"  depth {m['depth_mm']:.1f}/1000" if "depth_mm" in m else ""))
        rows.append((ims, txt))
    if not rows:
        return
    head, lab = 54, 28
    Wt = max(sum(i.width for i in ims) + 8 * (len(ims) - 1) for ims, _ in rows) + 24
    Ht = head + 26 + sum(tile_h + lab + 12 for _ in rows)
    S = Image.new("RGB", (Wt, Ht), (16, 16, 18))
    d = ImageDraw.Draw(S)
    d.text((12, 10), title, font=font(22), fill=(235, 235, 235))
    x = 12
    for c, w in zip(cols, [i.width for i in rows[0][0]]):
        d.text((x, head), c, font=font(13), fill=(160, 160, 160))
        x += w + 8
    y = head + 26
    for ims, txt in rows:
        d.text((12, y), txt, font=font(15), fill=(220, 220, 220))
        x = 12
        for im in ims:
            S.paste(im, (x, y + lab))
            x += im.width + 8
        y += tile_h + lab + 12
    S.save(out)
    return out
