"""The first volume: a visual hull carved from every view, with each cut recorded as a build step.

  inflate   bootstrap proxy from the front alone: the silhouette inflated to round cross-sections (radius from
            the medial axis), its front face on the photo's depth. Only used to find the views' cameras.
  carve     voxels are kept where every view's silhouette (dilated by a tolerance: generated views are not
            exactly consistent) and the photo's depth allow. Beyond the frame, a view keeps space only where the
            subject itself runs off that edge.
  symmetry  the mirror plane (vertical, through the centre, any yaw) that best maps the hull onto itself.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis

from . import geom as G
from . import register as RG
from . import sdf as S
from .maps import View


def _grid(lo, hi, vox):
    n = np.maximum(np.ceil((np.asarray(hi) - lo) / vox).astype(int), 2)
    return n


def _centres(lo, n, vox, sl):
    """voxel centres of the x-slab sl (memory-friendly)"""
    ii = np.arange(n[0])[sl]
    g = np.meshgrid(ii, np.arange(n[1]), np.arange(n[2]), indexing="ij")
    return lo + (np.stack(g, -1).reshape(-1, 3) + 0.5) * vox


def mask_sdf(mask):
    return ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)  # px, > 0 outside


def _lookup(img, cam, X, default, clamp=False):
    """image value at each point's projection. clamp: out-of-frame points take the nearest border pixel's value,
    so space beyond the frame is kept only where the subject itself runs off that edge (a cropped view)"""
    u, v, z = G.project(X, cam)
    h, w = img.shape
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    if clamp:
        ui, vi = np.clip(ui, 0, w - 1), np.clip(vi, 0, h - 1)
    ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.full(len(X), default, float)
    out[ok] = img[vi[ok], ui[ok]]
    return out


def front_bounds(case, near_pad=0.1, depth_span=2.5):
    T = case.front()
    fx, fy, cx, cy = case.K
    ys, xs = np.nonzero(T["mask"])
    d0 = case.center[1] - (case.center[1] - np.nanmin(T["depth"]) if np.isfinite(T["depth"]).any() else 0.6)
    near, far = max(d0 - near_pad, 0.05), d0 + depth_span
    X = [((x + 0.5 - cx) / fx) * d for x in (xs.min() - 2, xs.max() + 2) for d in (near, far)]
    Z = [(-(y + 0.5 - cy) / fy) * d for y in (ys.min() - 2, ys.max() + 2) for d in (near, far)]
    return np.array([min(X), near, min(Z)]), np.array([max(X), far, max(Z)])


def inflate(case, vox=1 / 64):
    """front-only proxy: inflated silhouette between the photo's depth and an estimated back"""
    T = case.front()
    mask = T["mask"]
    fx, fy, cx, cy = case.K
    skel, dist = medial_axis(mask, return_distance=True)
    _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    R = dist[iy, ix]
    e = ndi.distance_transform_edt(mask)
    hpx = np.sqrt(np.maximum(e * (2 * R - e), 0))
    yc = case.center[1]
    if np.isfinite(T["depth"]).any():
        yf = np.where(np.isfinite(T["depth"]), T["depth"], yc)
        back = yf + 2 * hpx * yf / fx
    else:
        yf = yc - hpx * yc / fx
        back = yc + hpx * yc / fx
    lo, hi = front_bounds(case, depth_span=1.6)
    n = _grid(lo, hi, vox)
    occ = np.zeros(n, bool)
    cam = case.cam("front")
    for i0 in range(0, n[0], 16):
        X = _centres(lo, n, vox, slice(i0, i0 + 16))
        inside = _lookup(mask.astype(float), cam, X, 0) > 0.5
        y = X[:, 1]
        inside &= (y > _lookup(yf, cam, X, np.inf) - 0.01) & (y < _lookup(back, cam, X, -np.inf))
        occ[i0:i0 + 16] = inside.reshape(-1, n[1], n[2])
    fd = S.from_occupancy(occ, lo, vox)
    return fd


def carve(case, names, vox, lo=None, hi=None, tol_front=1.0, tol_frac=0.008, depth_tol=0.012, record=None,
          scale=1.0):
    """visual hull from the front + `names`. record(label, occ, lo, vox) is called after every cut."""
    if lo is None:
        lo, hi = front_bounds(case)
    n = _grid(lo, hi, vox)
    T = case.front()
    W, H = case.size
    tests = [("front silhouette", case.cam("front"), mask_sdf(T["mask"]), tol_front)]
    for nm in names:
        vw = View(case, nm, 1.0)
        ys, _ = np.nonzero(vw.mask)
        tol = max(2.0, tol_frac * (ys.max() - ys.min()))
        tests.append((f"view {nm}", vw.cam, vw.sdf_raw, tol))
    occ = np.ones(n, bool)
    if record:
        record("block", occ, lo, vox)
    has_depth = np.isfinite(T["depth"]).any()
    order = tests[:1] + ([("front depth", None, None, None)] if has_depth else []) + tests[1:]
    stats = []
    for label, cam, sd, tol in order:
        before = int(occ.sum())
        fcam = case.cam("front")
        for i0 in range(0, n[0], 16):
            sl = slice(i0, i0 + 16)
            cur = occ[sl]
            if not cur.any():
                continue
            X = _centres(lo, n, vox, sl)
            if cam is None:  # in front of the photo's surface by more than depth_tol
                d = _lookup(np.nan_to_num(T["depth"], nan=np.inf), fcam, X, np.inf)
                keep = ~(X[:, 1] < d - depth_tol)
            else:
                keep = _lookup(sd, cam, X, np.inf, clamp=True) <= tol
            occ[sl] = cur & keep.reshape(cur.shape)
        stats.append({"cut": label, "removed": before - int(occ.sum())})
        if record:
            record(label, occ, lo, vox)
    lab, nl = ndi.label(occ)
    if nl > 1:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, nl + 1))
        occ = lab == 1 + int(np.argmax(sizes))
        if record:
            record("keep the largest piece", occ, lo, vox)
    return occ, lo, vox, stats


def occ_mesh(occ, lo, vox, sigma=1.0):
    f = 0.5 - ndi.gaussian_filter(occ.astype(np.float32), sigma)
    return S.mesh_of(f * vox, lo, vox)


def bbox_of(occ, lo, vox, pad):
    idx = np.argwhere(occ)
    return lo + (idx.min(0) - pad) * vox, lo + (idx.max(0) + 1 + pad) * vox


def symmetry(occ, lo, vox, center, thr=0.84):
    """best vertical mirror plane through (near) the centre: normal (cos t, sin t, 0), offset o along it"""
    P = lo + (np.argwhere(occ) + 0.5) * vox
    if len(P) > 60000:
        P = P[np.random.default_rng(0).choice(len(P), 60000, replace=False)]
    n = np.array(occ.shape)
    c = np.asarray(center, float)

    def score(t, o):
        nv = np.array([np.cos(np.radians(t)), np.sin(np.radians(t)), 0.0])
        d = (P - c) @ nv - o
        M = P - 2 * d[:, None] * nv
        ii = np.floor((M - lo) / vox).astype(int)
        ok = np.all((ii >= 0) & (ii < n), 1)
        hit = np.zeros(len(P), bool)
        hit[ok] = occ[tuple(ii[ok].T)]
        m = hit.mean()
        return m / (2 - m)

    best = max(((score(t, o), t, o) for t in np.arange(-60, 61, 3.0) for o in np.arange(-0.1, 0.101, 0.02)))
    _, t0, o0 = best
    best = max(((score(t, o), t, o) for t in np.arange(t0 - 3, t0 + 3.01, 0.5) for o in np.arange(o0 - 0.02, o0 + 0.0201, 0.004)))
    sc, t, o = best
    nv = np.array([np.cos(np.radians(t)), np.sin(np.radians(t)), 0.0])
    return {"iou": float(sc), "yaw_deg": float(t), "normal": nv.tolist(), "point": (c + o * nv).tolist(),
            "symmetric": bool(sc >= thr)}


def build(case, log=print):
    """register every view against the proxy, carve, re-centre, re-register, screen, carve again (recorded)"""
    names = case.views(("train", "holdout"))
    if not names:
        log("no views: front-only sculpt (inflated silhouette); depth behind the photo is a guess")
    import time
    t0 = time.time()
    fd = inflate(case)
    V, Fc = fd.mesh()
    log(f"  proxy from the front silhouette ({time.time() - t0:.0f}s)")
    for nm in names:
        vw = View(case, nm, 1.0)
        p, sc, io = RG.register(V, Fc, vw.mask, vw.ncam, case.center, case.K, case.cfg["views"][nm].get("requested"))
        case.cfg["views"][nm].update(params=p, iou_proxy=round(io, 4))
        log(f"  {nm}: requested {case.cfg['views'][nm].get('requested')} -> az {p['az']:.0f} el {p['el']:.0f} "
            f"(IoU vs proxy {io:.2f})")
    case.save()
    log(f"  registered {len(names)} views against the proxy ({time.time() - t0:.0f}s)")
    train = case.views(("train",))
    if names:
        occ, lo, vox, _ = carve(case, train, 1 / 64)
        Vh, Fh = occ_mesh(occ, lo, vox)
        c_old = case.center
        # re-centre on the hull only where side views constrain its depth, and never by more than 0.25 units
        # (an unconstrained hull runs to the far end of the carve box and drags the orbit centre with it)
        sides = [nm for nm in train if 55 <= abs(case.cfg["views"][nm]["params"]["az"]) <= 125]
        c_new = np.array([(Vh[:, 0].min() + Vh[:, 0].max()) / 2, (Vh[:, 1].min() + Vh[:, 1].max()) / 2, c_old[2]])
        c_new = c_old + np.clip(c_new - c_old, -0.25, 0.25)
        if len(sides) < 1:
            c_new = c_old
        case.cfg["front"]["center"] = c_new.tolist()
        case.save()
        log(f"  centre {np.round(c_old, 3).tolist()} -> {np.round(c_new, 3).tolist()} ({len(sides)} side views)")
        for nm in names:  # cameras orbit the centre: search again around the new one
            vw = View(case, nm, 1.0)
            prev = case.cfg["views"][nm]
            p, sc, io = RG.register(Vh, Fh, vw.mask, vw.ncam, case.center, case.K,
                                    {"az": prev["params"]["az"], "el": prev["params"]["el"]}, prior=0.05)
            p, info = RG.refine(Vh, Fh, vw.mask, vw.ncam, case.center, case.K, p)
            case.cfg["views"][nm].update(params=p, iou_hull=info["after"][1])
        log(f"  re-registered against the first hull ({time.time() - t0:.0f}s)")
        # screen: a view the others cannot agree with is a different subject or pose
        for nm in train:
            io = case.cfg["views"][nm]["iou_hull"]
            if io < 0.8:
                case.cfg["views"][nm]["role"] = "rejected"
                case.cfg["views"][nm]["reason"] = f"silhouette IoU {io:.2f} < 0.80 against the hull of the other views"
                log(f"  rejected {nm}: IoU {io:.2f}")
        case.save()
        train = case.views(("train",))
    hres = case.cfg["settings"].get("hull_res", 160)
    if names:
        occ, lo, vox, _ = carve(case, train, 1 / 64)
        lo, hi = bbox_of(occ, lo, vox, 3)
    else:
        V, _ = fd.mesh()
        lo, hi = V.min(0) - 0.05, V.max(0) + 0.05
    k0 = case.n_steps()

    def record(label, occ, lo_, vox_):
        if occ.all():
            Vb = np.array([[x, y, z] for x in (lo_[0], hi[0]) for y in (lo_[1], hi[1]) for z in (lo_[2], hi[2])])
            Fb = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                           [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            case.write_step(Vb.astype(float), Fb, f"carve: {label}", "carve")
            return
        Vm, Fm = occ_mesh(occ, lo_, vox_)
        case.write_step(Vm, Fm, f"carve: {label}", "carve", {"voxels": int(occ.sum())})

    if names:
        occ, lo, vox, stats = carve(case, train, 1 / hres, lo, hi, record=record)
    else:
        F = fd.F
        occ, lo, vox, stats = F < 0, fd.lo, fd.vox, []
        record("inflated front silhouette (no other views)", occ, lo, vox)
    log(f"  carved ({time.time() - t0:.0f}s)")
    sym = symmetry(occ, lo, vox, case.center)
    mode = case.cfg["settings"].get("symmetry", "auto")
    sym["used"] = bool(sym["symmetric"]) if mode == "auto" else mode == "on"
    case.cfg["symmetry"] = sym
    case.save()
    log(f"  symmetry: IoU {sym['iou']:.3f} at yaw {sym['yaw_deg']:.1f} deg -> {'on' if sym['used'] else 'off'}")
    fh = S.from_occupancy(occ, lo, vox)
    Vr, Fr = fh.mesh()  # hull-resolution mesh: enough for camera refinement, ~10x fewer triangles
    res = case.cfg["settings"].get("res", 400)
    fd = S.resample(fh, 1.0 / res)
    fd.save(case.p("build", "field.npz", mk=True))
    V, Fc = fd.mesh()
    case.save_mesh(V, Fc)
    case.write_step(V, Fc, "hull at sculpt resolution", "remesh", {"res": res})
    log(f"  distance field at 1/{res} ({len(V)} vertices, {time.time() - t0:.0f}s)")
    # final camera refinement against the hull (the held-out camera is frozen here, before any fitting)
    for nm in names:
        if case.cfg["views"][nm]["role"] == "rejected":
            continue
        vw = View(case, nm, 1.0)
        p, info = RG.refine(Vr, Fr, vw.mask, vw.ncam, case.center, case.K, case.cfg["views"][nm]["params"], f=2,
                            rounds=2)
        case.cfg["views"][nm].update(params=p, iou_hull=info["after"][1])
    log(f"  final camera refinement ({time.time() - t0:.0f}s)")
    case.cfg["hull"] = {"steps": [k0, case.n_steps() - 1], "cuts": stats, "voxel_hull": vox, "res": res}
    case.save()
    case.status("hull")
    return fd
