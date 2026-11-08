"""Synthetic subjects with known shape, rendered into exactly the files a real case produces (front maps,
generated-looking views with wrong angles and noisy normals, an optional view of a different pose).

Used by the tests and by `i2s.py demo` (the whole pipeline offline, no API key, scored against ground truth).
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import geom as G
from . import sdf as S
from .case import new_case
from .inputs import view_name, write_front


def sphere(P, c, r):
    return np.linalg.norm(P - c, axis=-1) - r


def capsule(P, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = P - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def ellipsoid(P, c, rad):
    q = (P - c) / np.asarray(rad)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(rad), axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def figure(P, arms_up=False):
    """a stocky humanoid facing -Y (the camera): gaps between the arms and the body, and between the legs"""
    d = ellipsoid(P, (0, 0, 0.05), (0.17, 0.12, 0.25))
    d = smin(d, ellipsoid(P, (0, 0, -0.17), (0.15, 0.11, 0.11)), 0.04)
    d = smin(d, sphere(P, (0, -0.01, 0.42), 0.12), 0.05)
    d = smin(d, sphere(P, (0, -0.125, 0.41), 0.032), 0.015)  # nose
    for sx in (-1, 1):
        d = smin(d, sphere(P, (sx * 0.115, -0.01, 0.46), 0.035), 0.02)  # ears
        hand = (sx * 0.31, -0.03, -0.14) if not arms_up else (sx * 0.36, -0.03, 0.55)
        arm = np.minimum(capsule(P, (sx * 0.21, 0, 0.22), hand, 0.045), sphere(P, hand, 0.06))
        d = np.minimum(d, smin(arm, sphere(P, (sx * 0.2, 0, 0.22), 0.07), 0.03))
        leg = capsule(P, (sx * 0.08, 0, -0.2), (sx * 0.1, 0, -0.5), 0.06)
        leg = smin(leg, ellipsoid(P, (sx * 0.1, -0.04, -0.54), (0.065, 0.1, 0.04)), 0.03)
        d = np.minimum(d, leg)
    return d


def creature(P, arms_up=False):
    """a four-legged creature seen from three-quarters, curled tail to one side (not symmetric)"""
    d = ellipsoid(P, (0, 0, 0), (0.33, 0.15, 0.15))
    d = smin(d, capsule(P, (0.25, 0, 0.05), (0.38, -0.02, 0.25), 0.07), 0.05)
    d = smin(d, ellipsoid(P, (0.43, -0.03, 0.3), (0.11, 0.08, 0.08)), 0.04)
    d = smin(d, capsule(P, (0.48, -0.03, 0.29), (0.55, -0.03, 0.27), 0.04), 0.03)  # snout
    for sy in (-1, 1):
        d = smin(d, capsule(P, (0.42, sy * 0.05, 0.36), (0.40, sy * 0.07, 0.45), 0.022), 0.02)  # ears
        for sx in (-1, 1):
            leg = capsule(P, (sx * 0.22, sy * 0.08, -0.05), (sx * 0.24, sy * 0.09, -0.4), 0.045)
            d = np.minimum(d, smin(leg, sphere(P, (sx * 0.22, sy * 0.08, -0.04), 0.07), 0.03))
    tail = capsule(P, (-0.3, 0, 0.05), (-0.45, -0.08, 0.2), 0.03)
    tail = np.minimum(tail, capsule(P, (-0.45, -0.08, 0.2), (-0.42, -0.2, 0.3), 0.026))
    return np.minimum(d, tail)


SUBJECTS = {"figure": figure, "creature": creature}


def truth_mesh(fn, yaw, center, vox=1 / 150, **kw):
    lo, hi = np.array([-0.75, -0.75, -0.75]), np.array([0.75, 0.75, 0.75])
    n = np.ceil((hi - lo) / vox).astype(int)
    g = np.stack(np.meshgrid(*[lo[i] + (np.arange(n[i]) + 0.5) * vox for i in range(3)], indexing="ij"), -1)
    Rz = G.rz(np.radians(-yaw))
    Q = g.reshape(-1, 3) @ Rz.T  # rotate the query points = rotate the subject by +yaw
    F = np.concatenate([fn(Q[i:i + 2_000_000], **kw) for i in range(0, len(Q), 2_000_000)]).reshape(n)
    V, Fc = S.mesh_of(F.astype(np.float32), lo, vox)
    return V + np.asarray(center), Fc


def noisy_normals(ncam, mask, deg, seed):
    if deg <= 0:
        return ncam
    rng = np.random.default_rng(seed)
    noise = np.stack([ndi.gaussian_filter(rng.standard_normal(mask.shape), 6) for _ in range(3)], -1)
    noise *= np.radians(deg) / max(noise.std(), 1e-9)
    return G.unit(ncam + noise) * mask[..., None]


def render(V, Fc, cam, W, H, base=(0.74, 0.68, 0.6)):
    Rm = G.raster(V, Fc, cam, W, H, VN=G.vertex_normals(V, Fc))
    img = G.clay_shade(Rm["ncam"], Rm["mask"], base=base, bg=(0.84, 0.84, 0.85))
    return Rm, img


def make_case(path, subject="figure", size=(384, 512), yaw=20.0, n_views="standard", bad_view=False, seed=0,
              angle_noise=10.0, normal_noise=4.0, res=160, hull_res=96, truth_vox=1 / 150):
    """-> Case, truth dict (mesh in case units, true view params)"""
    W, H = size
    fn = SUBJECTS[subject]
    rng = np.random.default_rng(seed)
    center = np.array([0.0, 2.4, 0.0])
    V, Fc = truth_mesh(fn, yaw, center, truth_vox)
    fy = 0.5 * H / np.tan(np.radians(18))
    K = (fy, fy, W / 2, H / 2)
    front_cam = G.make_cam({}, center, K)
    Rm, img = render(V, Fc, front_cam, W, H)
    os.makedirs(path, exist_ok=True)
    src = os.path.join(path, "synthetic_input.png")
    Image.fromarray(img).save(src)
    case = new_case(path, src, subject=f"synthetic {subject}", kind="synthetic", long_side=max(W, H))
    os.remove(src)
    case.cfg["settings"].update(res=res, hull_res=hull_res, fit_scale=1.0)
    assert tuple(case.size) == (W, H), (case.size, W, H)
    depth = np.where(Rm["mask"], Rm["depth"], np.nan)
    ncam = noisy_normals(Rm["ncam"], Rm["mask"], normal_noise * 0.5, seed)
    write_front(case, Rm["mask"], depth, ncam, K, {"mask_source": "synthetic", "depth_source": "synthetic"})
    scale = case.cfg["front"]["unit_scale"]
    from .inputs import ANGLE_SETS, HOLDOUT
    angles = [(az, el, "train") for az, el in ANGLE_SETS[n_views]] + [(*HOLDOUT, "holdout")]
    truth = {"views": {}, "unit_scale": scale, "center": (center * scale).tolist()}
    for k, (az, el, role) in enumerate(angles):
        name = view_name(az, el)
        p = {"az": az + float(rng.normal(0, angle_noise)), "el": el + float(rng.normal(0, angle_noise * 0.4)),
             "s": 1 + float(rng.normal(0, 0.03)), "du": float(rng.normal(0, 5)), "dv": float(rng.normal(0, 5))}
        Rv, imv = render(V, Fc, G.make_cam(p, center, K), W, H)
        _save_view(case, name, Rv, imv, normal_noise, seed + 1 + k)
        case.cfg["views"][name] = {"requested": {"az": az, "el": el}, "role": role, "source": "synthetic"}
        truth["views"][name] = p
    if bad_view:  # a generation that changed the pose: screening must reject it
        Vb, Fb = truth_mesh(fn, yaw, center, truth_vox, arms_up=True)
        name = "az-030_badpose"
        p = {"az": -30.0, "el": 0.0, "s": 1.0, "du": 0.0, "dv": 0.0}
        Rv, imv = render(Vb, Fb, G.make_cam(p, center, K), W, H)
        _save_view(case, name, Rv, imv, normal_noise, seed + 99)
        case.cfg["views"][name] = {"requested": {"az": -30, "el": 0}, "role": "train", "source": "synthetic bad pose"}
        truth["views"][name] = p
    case.save()
    np.savez_compressed(case.p("synthetic_truth.npz"), verts=(V * scale).astype(np.float32), faces=Fc.astype(np.int32))
    json.dump(truth, open(case.p("synthetic_truth.json"), "w"), indent=1)
    return case, truth


def _save_view(case, name, Rm, img, normal_noise, seed):
    Image.fromarray(img).save(case.p("views", f"{name}.png"))
    Image.fromarray(Rm["mask"].astype(np.uint8) * 255).save(case.p("views", f"{name}_mask.png"))
    n = noisy_normals(Rm["ncam"], Rm["mask"], normal_noise, seed)
    Image.fromarray(np.clip((n + 1) * 127.5, 0, 255).astype(np.uint8)).save(case.p("views", f"{name}_normals.png"))


def chamfer(Va, Fa, Vb, Fb, n=40000):
    """mean symmetric surface distance between two meshes (case units = fraction of the subject's extent)"""
    from .register import surface_points
    Pa, _ = surface_points(Va, Fa, n, seed=1)
    Pb, _ = surface_points(Vb, Fb, n, seed=2)
    da, _, _ = S.mesh_closest(Pa, Vb, Fb)
    db, _, _ = S.mesh_closest(Pb, Va, Fa)
    return float((da.mean() + db.mean()) / 2), float(max(np.percentile(da, 95), np.percentile(db, 95)))
