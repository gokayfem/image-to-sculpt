"""Find the camera of every non-front view.

A generated view is never at the angle it was asked for: on the first case this toolkit was built from, views
requested at +45, +90 and -90 degrees registered at +19, -123 and +35. So the search covers the whole circle.

  1. global   every 10 deg of azimuth x 5 elevations, a closed-form scale and offset for each, scored on a
              quarter-resolution splat: silhouette IoU + normal agreement - a small prior toward the requested angle.
              |az| stays within 60 deg of the requested |az| and el within 30 deg: generators often swap left and
              right, rarely front and back (and a back view outlines like a front view on a symmetric subject)
     partner  on a left-right symmetric subject (az, el) and (180 - az, -el) have the same outline: the one
              nearer the requested angle wins a near-tie (generators rarely confuse front and back)
  2. local    coordinate search with halving steps on the best three, same splat
  3. refine   coordinate search on the exact rasteriser at half resolution
"""
import numpy as np
from scipy import ndimage as ndi

from . import geom as G


def surface_points(V, Fc, n=60000, seed=0):
    """area-weighted surface sample with face normals"""
    A = V[Fc]
    fn = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    area = np.linalg.norm(fn, axis=1)
    rng = np.random.default_rng(seed)
    f = rng.choice(len(Fc), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = A[f, 0] * (1 - s)[:, None] + A[f, 1] * (s * (1 - r2))[:, None] + A[f, 2] * (s * r2)[:, None]
    return P, G.unit(fn[f])


def splat_nm(P, N, cam, w, h):
    """z-buffered point splat: mask (closed) + camera-frame normals"""
    u, v, z = G.project(P, cam)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pix = v[ok].astype(int) * w + u[ok].astype(int)
    zz = z[ok]
    order = np.lexsort((zz, pix))
    ps = pix[order]
    first = order[np.r_[True, ps[1:] != ps[:-1]]] if len(ps) else order
    m = np.zeros(h * w, bool)
    m[pix[first]] = True
    R = cam["R"]
    Nk = N[ok][first]
    nc = np.zeros((h * w, 3))
    nc[pix[first]] = np.stack([Nk @ R[:, 0], Nk @ R[:, 2], -(Nk @ R[:, 1])], -1)
    m = m.reshape(h, w)
    mc = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=1))
    return mc, nc.reshape(h, w, 3), m


class Target:
    def __init__(self, mask, ncam, f):
        h, w = mask.shape
        self.w, self.h = w // f, h // f
        self.mask = np.asarray(ndi.zoom(mask.astype(float), 1 / f, order=1, grid_mode=True, mode="nearest") > 0.5)
        self.mask = self.mask[:self.h, :self.w]
        self.ncam = None
        if ncam is not None:
            n = np.stack([ndi.zoom(ncam[..., k], 1 / f, order=1, grid_mode=True, mode="nearest") for k in range(3)], -1)
            self.ncam = G.unit(n[:self.h, :self.w])
        ys, xs = np.nonzero(self.mask)
        self.hgt = ys.max() - ys.min() + 1
        self.ctr = np.array([xs.mean(), ys.mean()])
        self.bctr = np.array([(xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2])


def _score(P, N, cam, T, wn=0.3):
    m, nc, raw = splat_nm(P, N, cam, T.w, T.h)
    io = G.iou(m, T.mask)
    if T.ncam is None or wn == 0:
        return io, io
    both = raw & T.mask
    ag = float(np.clip((nc[both] * T.ncam[both]).sum(-1), -1, 1).mean()) if both.sum() > 20 else 0.0
    return io + wn * ag, io


def _closed_form(P, N, center, Kq, T, az, el):
    """scale from the projected height, offset from the bounding-box centres (no rendering needed)"""
    p = {"az": float(az), "el": float(el), "s": 1.0, "du": 0.0, "dv": 0.0}
    u, v, z = G.project(P, G.make_cam(p, center, Kq))
    ok = z > 0
    if ok.sum() < 30:
        return p
    u, v = u[ok], v[ok]
    v0, v1 = np.percentile(v, [0.3, 99.7])
    p["s"] = float(np.clip(T.hgt / max(v1 - v0, 1e-6), 0.6, 1.6))
    cx, cy = Kq[2], Kq[3]
    u0, u1 = np.percentile(u, [0.3, 99.7])
    pu, pv = cx + (0.5 * (u0 + u1) - cx) * p["s"], cy + (0.5 * (v0 + v1) - cy) * p["s"]
    p["du"], p["dv"] = float(T.bctr[0] - pu), float(T.bctr[1] - pv)
    return p


def _coord(f, p, steps, rounds=4):
    best = f(p)
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for k, st in steps.items():
                for sg in (1, -1):
                    q = dict(p)
                    q[k] = q.get(k, 0.0) + sg * st
                    val = f(q)
                    if val[0] > best[0] + 1e-5:
                        p, best, improved = q, val, True
                        break
        steps = {k: v / 2 for k, v in steps.items()}
    return p, best


def in_band(az, el, raz, rel, band, el_band):
    """generators swap left and right (az -> -az) but rarely front and back: keep |az| near |requested az|"""
    if raz is None:
        return True
    a = abs(((az + 180) % 360) - 180)
    return abs(a - abs(((raz + 180) % 360) - 180)) <= band and abs(el - rel) <= el_band


def register(V, Fc, mask, ncam, center, K, requested=None, f=4, prior=0.03, window=None, tie=0.03, band=60.0,
             el_band=30.0, verbose=False):
    """-> params (pixel offsets at the working resolution), quarter-res score, quarter-res IoU.
    window: search only +-window degrees of azimuth around the requested angle. band: |az| within band of the
    requested |az| (left/right swaps allowed, front/back flips not); None = the whole circle"""
    P, N = surface_points(V, Fc, 30000)
    T = Target(mask, ncam, f)
    Kq = tuple(k / f for k in K)
    raz = None if not requested or requested.get("az") is None else float(requested["az"])
    rel = 0.0 if not requested else float(requested.get("el") or 0.0)
    els = sorted({-15.0, 0.0, 15.0, 30.0, 45.0, rel})
    cands = []
    azs = np.arange(-180, 180, 10.0)
    if window and raz is not None:
        azs = raz + np.arange(-window, window + 0.1, 5.0)
    for az in azs:
        for el in els:
            if band is not None and not in_band(az, el, raz, rel, band, el_band):
                continue
            p = _closed_form(P, N, center, Kq, T, az, el)
            sc, io = _score(P, N, G.make_cam(p, center, Kq), T)
            pen = 0.0 if raz is None else prior * abs(((az - raz + 180) % 360) - 180) / 180
            cands.append((sc - pen, io, p))
    cands.sort(key=lambda c: -c[0])

    def fq(q):
        if band is not None and not in_band(q["az"], q["el"], raz, rel, band + 5, el_band + 5):
            return -9.0, 0.0
        sc, io = _score(P, N, G.make_cam(q, center, Kq), T)
        pen = 0.0 if raz is None else prior * abs(((q["az"] - raz + 180) % 360) - 180) / 180
        return sc - pen, io
    best = None
    for sc0, io0, p0 in cands[:3]:
        p, val = _coord(fq, dict(p0), {"az": 4.0, "el": 4.0, "s": 0.03, "du": 2.0, "dv": 2.0}, rounds=3)
        if best is None or val[0] > best[1][0]:
            best = (p, val)
        if verbose:
            print(f"   cand az {p0['az']:.0f} el {p0['el']:.0f} -> az {p['az']:.1f} el {p['el']:.1f} score {val[0]:.3f} "
                  f"iou {val[1]:.3f}")
    # silhouettes cannot tell (az, el) from (180 - az, -el) on a left-right symmetric subject (the back seen from
    # above outlines like the front seen from below); the generator rarely confuses front and back, so the
    # partner that is nearer the requested angle wins when it scores within `tie`
    p0, v0 = best
    if raz is not None:
        q = _closed_form(P, N, center, Kq, T, 180.0 - p0["az"], -p0["el"])
        q, vq = _coord(fq, q, {"az": 4.0, "el": 4.0, "s": 0.03, "du": 2.0, "dv": 2.0}, rounds=3)

        def dist(x):
            return abs(((x["az"] - raz + 180) % 360) - 180) + 0.5 * abs(x["el"] - rel)
        if vq[0] >= v0[0] - tie and dist(q) < dist(p0):
            if verbose:
                print(f"   front/back partner az {q['az']:.1f} el {q['el']:.1f} (score {vq[0]:.3f} vs {v0[0]:.3f})")
            best = (q, vq)
    p, (sc, io) = best
    p = dict(p, du=p["du"] * f, dv=p["dv"] * f)
    p["az"] = ((p["az"] + 180) % 360) - 180
    return p, float(sc), float(io)


def refine(V, Fc, mask, ncam, center, K, params, f=2, rounds=3):
    """exact-raster refinement at 1/f resolution: IoU + 0.1 x share of normals within 15 deg"""
    VN = G.vertex_normals(V, Fc)
    T = Target(mask, ncam, f)
    Kf = tuple(k / f for k in K)
    p0 = dict(params, du=params["du"] / f, dv=params["dv"] / f)

    def fr(q):
        Rm = G.raster(V, Fc, G.make_cam(q, center, Kf), T.w, T.h, VN=VN)
        io = G.iou(Rm["mask"], T.mask)
        if T.ncam is None:
            return io, io
        inner = ndi.binary_erosion(Rm["mask"] & T.mask, iterations=2)
        ag = float(((Rm["ncam"][inner] * T.ncam[inner]).sum(-1) > np.cos(np.radians(15))).mean()) if inner.any() else 0
        return io + 0.1 * ag, io

    start = fr(p0)
    p, (sc, io) = _coord(fr, p0, {"az": 1.0, "el": 1.0, "s": 0.01, "du": 1.0, "dv": 1.0}, rounds=rounds)
    p = dict(p, du=p["du"] * f, dv=p["dv"] * f)
    return p, {"before": [round(start[0], 4), round(start[1], 4)], "after": [round(sc, 4), round(io, 4)]}
