"""Fit the volume to every map, one recorded pass at a time.

One pass:
  1. mesh the distance field; solve on a Taubin-smoothed copy (no marching-cubes facets in the normals)
  2. per view (the photo first): rasterise the sculpt from the view's camera and build per-vertex targets
       normals     the view's normal map where the vertex is visible and faces the camera; generated views get one
                   global rotation each (Kabsch) and only fill in where the photo does not see
       silhouette  push/pull along the normal so the outline lands on the view's outline
       depth       the photo's metric depth only (generated views' depths disagree by centimetres)
  3. symmetry (if the subject has a mirror plane): the better-seen side lends its targets to the worse-seen side
  4. two solves: silhouette + depth through a very stiff field (whole forms move, no ridges along limbs), then
     the normals through a flexible one whose stiffness sets the scale of the change (forms -> surface -> detail)
  5. displace the ORIGINAL surface, re-distance the field exactly (sdf.apply_mesh), mesh, record, measure.
Ops between stages: cut (silhouette cut that opens gaps the views see), smooth (opening: removes webs, flaps,
specks), focus (passes limited to front-image regions such as the face or hands).
"""
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.spatial import cKDTree

from . import geom as G
from . import maps as MP
from . import sdf as S

BASE = dict(alpha=7e4, wsil=2.0, fpow=2)
STAGES = {
    # lam: normal-field stiffness (crossover ~4 cm at 3000, ~1.7 cm at 100, ~5 mm at 6 for a 1-unit subject);
    # lam_low: the silhouette/depth field (~15-20 cm). Normal maps are never blurred: the stiffness alone limits
    # each stage (blurred targets flatten small features).
    "forms": dict(BASE, passes=10, step=0.003, mu=1.0, lam=3000.0, lam_low=1e6, wdepth=0.5, sil_clip=0.006, nsmooth=3),
    "surface": dict(BASE, passes=14, step=0.002, mu=2.0, lam=100.0, lam_low=3e5, wdepth=0.4, sil_clip=0.004, nsmooth=2),
    "detail": dict(BASE, passes=14, step=0.0012, mu=3.0, lam=6.0, lam_low=3e5, wdepth=0.3, sil_clip=0.003, nsmooth=2),
    "polish": dict(BASE, passes=4, step=0.001, mu=3.0, lam=6.0, lam_low=3e5, wdepth=0.3, sil_clip=0.003, nsmooth=2),
    "focus": dict(BASE, passes=8, step=0.001, mu=4.0, lam=3.0, lam_low=3e5, lam_mid=300.0, wdepth=0.3,
                  sil_clip=0.003, nsmooth=1),
}


# ----------------------------------------------------------------------------- targets


def view_targets(V, N, vw, Rm, prm):
    cam = vw.cam
    u, v, z = G.project(V, cam)
    W, H = vw.W, vw.H
    inb = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1) & (z > 0)
    ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
    zb = ndi.minimum_filter(Rm["depth"], 3)
    vis = inb & (z <= zb[vi, ui] + 0.003)
    r = G.unit(V - cam["p"])
    facing = -(N * r).sum(1)
    out = {}
    # -- normals
    if vw.nworld is not None:
        nt = G.unit(G.sample(vw.nworld, u, v))
        w = vw.trust * vis * G.smoothstep(0.15, 0.5, facing) * np.clip(facing, 0, 1) ** prm["fpow"] * G.sample(vw.nvalid, u, v)
        lo_a, hi_a = (70, 40) if vw.metric or vw.name == "front" else (55, 30)
        agree = G.smoothstep(np.cos(np.radians(lo_a)), np.cos(np.radians(hi_a)), (nt * N).sum(1))
        wn = w * agree * inb
        if vw.name != "front" and wn.sum() > 100:
            # one global rotation per generated view: normal estimates of different images disagree by a few
            # degrees overall, and integrating them leaves creases where the dominant view changes
            sel = wn > 0.05
            Hm = (nt[sel] * wn[sel, None]).T @ N[sel]
            U, _, Vt = np.linalg.svd(Hm)
            Rk = (U @ Vt).T
            if np.linalg.det(Rk) < 0:
                Rk = np.eye(3)
            ang = np.degrees(np.arccos(np.clip((np.trace(Rk) - 1) / 2, -1, 1)))
            if ang < 8.0:
                nt = G.unit(nt @ Rk.T)
            out["tilt_deg"] = float(ang)
    else:
        nt, wn = N.copy(), np.zeros(len(V))
    out["wn"], out["nt"] = wn, nt
    # -- depth (metric photo only)
    R = cam["R"]
    nf = N @ R[:, 1]
    if vw.metric:
        dt = G.sample(np.nan_to_num(vw.depth, nan=0.0), u, v)
        dok = G.sample(vw.dval.astype(np.float32), u, v) > 0.99
        dz = dt - z
        td = np.where(nf < -0.3, dz / np.where(nf < -0.3, nf, -1), 0.0)
        wd = vw.trust * prm["wdepth"] * vis * dok * G.smoothstep(0.3, 0.6, -nf) / (1 + (dz / 0.015) ** 2)
        td, wd = np.nan_to_num(td), np.nan_to_num(wd)
    else:
        td, wd = np.zeros(len(V)), np.zeros(len(V))
    # -- silhouette
    q = (V - cam["p"]) @ R
    d = N @ R
    fx, fy = cam["K"][0], cam["K"][1]
    du = fx * (d[:, 0] * q[:, 1] - q[:, 0] * d[:, 1]) / q[:, 1] ** 2
    dv = -fy * (d[:, 2] * q[:, 1] - q[:, 2] * d[:, 1]) / q[:, 1] ** 2
    sref = G.sample(vw.sdf, u, v)
    gJ = G.sample(vw.gx, u, v) * du + G.sample(vw.gy, u, v) * dv
    rmask = Rm["mask"]
    rs = G.sample(ndi.distance_transform_edt(~rmask) - ndi.distance_transform_edt(rmask), u, v)
    strong = gJ > 0.3 * fx / np.maximum(q[:, 1], 1e-3)
    outside = inb & (sref > 0.5) & strong
    expand = inb & (sref < -0.5) & (rs > -2.0) & strong
    ts = np.where(outside | expand, -sref / np.where(strong, gJ, 1), 0.0)
    ts = np.clip(ts, -prm["sil_clip"], prm["sil_clip"])
    ws = vw.trust * prm["wsil"] * (outside | expand) * G.smoothstep(0.5, 2.0, np.abs(sref))
    out["wp"] = wd + ws
    out["tp"] = (wd * td + ws * ts) / np.maximum(wd + ws, 1e-12)
    return out


# ----------------------------------------------------------------------------- solves


def factor(M):
    """sparse LU with a symmetric fill-reducing ordering (the systems are symmetric positive definite): ~3x faster
    and leaner than the default COLAMD ordering on 400k-vertex meshes"""
    return sla.splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


def coarse_graph(V, N, E, cell):
    """cluster vertices by grid cell AND dominant normal direction (two sheets facing each other across a thin
    gap never share a cluster) -> labels, centroids, counts, cluster edges"""
    key = np.floor((V - V.min(0)) / cell).astype(np.int64)
    ax = np.abs(N).argmax(1)
    sgn = (N[np.arange(len(N)), ax] > 0).astype(np.int64)
    key = np.c_[key, 2 * ax + sgn]
    _, lab = np.unique(key, axis=0, return_inverse=True)
    lab = lab.ravel()
    m = int(lab.max()) + 1
    cnt = np.bincount(lab, minlength=m).astype(float)
    C = np.stack([np.bincount(lab, weights=V[:, k], minlength=m) for k in range(3)], 1) / cnt[:, None]
    ec = np.sort(np.stack([lab[E[:, 0]], lab[E[:, 1]]], 1), 1)
    ec = np.unique(ec[ec[:, 0] != ec[:, 1]], axis=0)
    return lab, C, cnt, ec


def stiff_band(V, N, E, A, deg, tp, wp, lam, alpha, coarse_cell=None):
    """the silhouette/depth offset through a very stiff thin-plate field. On big meshes it is solved on a coarse
    cluster graph (summed data weights, the same lam/h^2 scaling, so the continuous problem is unchanged) and
    spread back by diffusion: the field is smooth over ~15 cm, the clusters are ~1 cm."""
    n = len(V)
    if coarse_cell is None or n < 60000:
        le = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
        h = float(np.median(le))
        L0 = sparse.eye(n) - sparse.diags(1 / deg) @ A
        M = sparse.diags(alpha * wp) + (lam / h ** 2) * (L0.T @ L0) + 2500.0 * sparse.eye(n)
        return factor(M).solve(alpha * wp * tp)
    lab, C, cnt, ec = coarse_graph(V, N, E, coarse_cell)
    m = len(C)
    Ac, degc = G.adjacency(ec, m)
    hc = float(np.median(np.linalg.norm(C[ec[:, 1]] - C[ec[:, 0]], axis=1)))
    wc = np.bincount(lab, weights=wp, minlength=m)
    bc = np.bincount(lab, weights=wp * tp, minlength=m)
    L0 = sparse.eye(m) - sparse.diags(1 / degc) @ Ac
    M = sparse.diags(alpha * wc) + (lam / hc ** 2) * (L0.T @ L0) + sparse.diags(2500.0 * cnt)
    tc = factor(M).solve(alpha * bc)
    return G.diffuse(tc[lab], A, deg, 20)


def solve(V, N, E, A, deg, nt, cn, tp, wp, prm):
    """normal integration (Nehab-style edge constraints) + position targets + a membrane"""
    n = len(V)
    Nt = G.unit(cn[:, None] * nt + (1 - cn[:, None]) * N)  # unsure: keep the current normal (never a smoothed one)
    i, j = E[:, 0], E[:, 1]
    ne = G.unit(Nt[i] + Nt[j])
    le = np.linalg.norm(V[j] - V[i], axis=1)
    h = float(np.median(le))
    ce = np.minimum(cn[i], cn[j]) + 0.15
    s = np.sqrt(prm["mu"] * ce) / np.maximum(le, 1e-5)
    m = len(E)
    Ae = sparse.csr_matrix((np.r_[s * (ne * N[j]).sum(1), -s * (ne * N[i]).sum(1)],
                            (np.r_[np.arange(m), np.arange(m)], np.r_[j, i])), shape=(m, n))
    b_e = -s * (ne * (V[j] - V[i])).sum(1)
    sp = np.sqrt(prm["alpha"] * wp)
    L = (sparse.eye(n) - sparse.diags(1 / deg) @ A) * (np.sqrt(prm["lam"]) / h)
    M = (Ae.T @ Ae + sparse.diags(sp ** 2) + L.T @ L + 2500.0 * sparse.eye(n)).tocsc()
    rhs = Ae.T @ b_e + sp * sp * tp
    t = factor(M).solve(rhs)
    return t, float(np.abs(M @ t - rhs).max() / max(np.abs(rhs).max(), 1e-12))


def solve2(V, N, E, A, deg, nt, cn, tp, wp, prm, wmid=None):
    """low band: silhouettes + depth through a very stiff field; optional mid band (focus regions) so whole
    fingers can move and the gaps between them open; high band: normals, flexible"""
    n = len(V)
    cell = prm.get("coarse_cell", 0.0125)
    t_low = stiff_band(V, N, E, A, deg, tp, wp, prm["lam_low"], prm["alpha"], cell)
    t_low = prm["step"] * np.tanh(t_low / prm["step"])
    if prm.get("lam_mid") and wmid is not None:
        le = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
        h = float(np.median(le))
        L0 = sparse.eye(n) - sparse.diags(1 / deg) @ A
        Mm = sparse.diags(prm["alpha"] * wp * wmid) + (prm["lam_mid"] / h ** 2) * (L0.T @ L0) + 2500.0 * sparse.eye(n)
        t_mid = factor(Mm).solve(prm["alpha"] * wp * wmid * (tp - t_low))
        t_low = t_low + wmid * prm["step"] * np.tanh(t_mid / prm["step"])
    t_high, info = solve(V + t_low[:, None] * N, N, E, A, deg, nt, cn, np.zeros(n), np.zeros(n), dict(prm, alpha=0.0))
    return t_low + t_high, info


# ----------------------------------------------------------------------------- symmetry, regions


def mirror(P, sym):
    n, p = np.array(sym["normal"]), np.array(sym["point"])
    return P - 2 * ((P - p) @ n)[:, None] * n


def mirror_vec(N, sym):
    n = np.array(sym["normal"])
    return N - 2 * (N @ n)[:, None] * n


def region_weight(V, Fc, A, deg, front_view, mask, it=25):
    """vertices the front photo sees inside `mask`, spread a few centimetres over the surface"""
    VN = G.vertex_normals(V, Fc)
    Rm = G.raster(V, Fc, front_view.cam, front_view.W, front_view.H, VN=VN)
    u, v, z = G.project(V, front_view.cam)
    W, H = front_view.W, front_view.H
    ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
    m = mask
    if m.shape != (H, W):
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H))) > 127
    seed = (z <= Rm["depth"][vi, ui] + 0.004) & m[vi, ui]
    w = G.diffuse(seed.astype(float), A, deg, it)
    return G.smoothstep(0.02, 0.25, w)


def load_region(case, spec):
    """'hands' -> parts/hands.png; 'box:x0,y0,x1,y1' (fractions or pixels); several joined by '+'"""
    W, H = case.size
    acc = np.zeros((H, W), bool)
    for s in spec.split("+"):
        if s.startswith("box:"):
            x0, y0, x1, y1 = [float(c) for c in s[4:].split(",")]
            if max(x0, y0, x1, y1) <= 1.0:
                x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
            acc[int(y0):int(y1), int(x0):int(x1)] = True
        else:
            p = case.p("parts", f"{s}.png")
            if not os.path.exists(p):
                raise SystemExit(f"no region '{s}': make it with i2s.py parts CASE --prompt '{s}' (or use box:...)")
            acc |= np.asarray(Image.open(p).convert("L").resize((W, H))) > 127
    return acc


# ----------------------------------------------------------------------------- one pass


def one_pass(fd, views, prm, sym=None, focus_mask=None, front_view=None):
    V0, Fc = fd.mesh()
    E = G.edges_of(Fc)
    A, deg = G.adjacency(E, len(V0))
    V = G.taubin(V0, A, deg)  # solve on a facet-free copy; displace the original surface
    N = G.vertex_normals(V, Fc)
    NT, WN, TP, WP = np.zeros_like(V), np.zeros(len(V)), np.zeros(len(V)), np.zeros(len(V))
    wf = np.zeros(len(V))
    tilts = {}
    for vw in sorted(views, key=lambda x: x.name != "front"):
        if vw.trust <= 0:
            continue
        Rm = G.raster(V, Fc, vw.cam, vw.W, vw.H, VN=N)
        T = view_targets(V, N, vw, Rm, prm)
        if "tilt_deg" in T:
            tilts[vw.name] = round(T["tilt_deg"], 2)
        if vw.name == "front":
            wf = G.smoothstep(0.02, 0.2, T["wn"])
        else:
            T["wn"] = T["wn"] * (1 - wf) ** 2  # where the photo sees, the photo governs
        NT += T["wn"][:, None] * T["nt"]
        WN += T["wn"]
        TP += T["wp"] * T["tp"]
        WP += T["wp"]
    nt = G.unit(NT + 1e-12 * N)
    cn = WN / (WN + 0.05)
    tp = TP / np.maximum(WP, 1e-12)
    s_frac = 0.0
    if sym and sym.get("used"):
        dist, jm = cKDTree(V).query(mirror(V, sym), workers=-1)
        s = G.diffuse(G.smoothstep(0.015, 0.006, dist), A, deg, 10)
        a_n = G.diffuse(s * np.clip(0.5 + cn[jm] - cn, 0, 1) * (1 - np.minimum(cn, cn[jm])), A, deg, 8)
        nt_m = mirror_vec(nt[jm], sym)
        nt = G.unit((1 - a_n)[:, None] * cn[:, None] * nt + (a_n * cn[jm])[:, None] * nt_m + 1e-12 * N)
        cn = (1 - a_n) * cn + a_n * cn[jm]
        cp = WP / (WP + 0.5)
        a_p = s * np.clip(0.5 + cp[jm] - cp, 0, 1) * (1 - np.minimum(cp, cp[jm]))
        P = V + tp[:, None] * N
        Pm = mirror(P[jm], sym)
        has_i, has_j = WP > 1e-9, WP[jm] > 1e-9
        a_p = np.where(has_j, np.where(has_i, a_p, s), 0.0)
        Ps = (1 - a_p)[:, None] * np.where(has_i[:, None], P, V) + a_p[:, None] * Pm
        tp = ((Ps - V) * N).sum(1)
        WP = (1 - a_p) * WP + a_p * WP[jm]
        s_frac = float((s > 0.5).mean())
    NW, CW = cn[:, None] * nt, cn.copy()
    for _ in range(prm["nsmooth"]):
        NW = 0.5 * NW + 0.5 * (A @ NW) / deg[:, None]
        CW = 0.5 * CW + 0.5 * (A @ CW) / deg
    nt, cn = G.unit(NW + 1e-12 * N), np.clip(CW, 0, 1)
    wfoc = None
    if focus_mask is not None:
        wfoc = region_weight(V, Fc, A, deg, front_view, focus_mask)
    t, info = solve2(V, N, E, A, deg, nt, cn, tp, WP, prm, wmid=wfoc)
    t = prm["step"] * np.tanh(t / prm["step"])  # a soft limit: a hard clip leaves steps
    if wfoc is not None:
        t = t * wfoc
    fd = S.apply_mesh(fd, V0, Fc, t, G.smooth_vec(N, A, deg, 3), A, deg)
    return fd, {"verts": int(len(V)), "solve_relres": info, "t_p1_p50_p99":
                np.percentile(t * 1000, [1, 50, 99]).round(2).tolist(), "seen_frac": float((cn > 0.5).mean()),
                "sym_frac": s_frac, "tilt_deg": tilts}


# ----------------------------------------------------------------------------- ops


def cut(fd, case, views, region=None, which="front", tol_front=1.0, tol_gen=2.5):
    """silhouette cut: remove what a view's silhouette excludes (smooth max, no creases). `which` = front | all.
    Opens the gaps between fingers, legs, arm and body that the soft silhouette term cannot."""
    idx = np.argwhere(fd.F < 8 * fd.vox)
    P = fd.centres(idx)
    keepreg = np.ones(len(P), bool)
    fv = next(v for v in views if v.name == "front")
    if region is not None:
        u, v, z = G.project(P, fv.cam)
        reg = ndi.binary_dilation(region, iterations=4)
        ui, vi = np.clip(u.astype(int), 0, fv.W - 1), np.clip(v.astype(int), 0, fv.H - 1)
        keepreg = reg[vi, ui]
    cutF = np.full(len(P), -np.inf)
    for vw in views:
        if vw.role == "holdout" or (which == "front" and vw.name != "front"):
            continue
        u, v, z = G.project(P, vw.cam)
        ok = (u >= 0) & (u < vw.W) & (v >= 0) & (v < vw.H) & (z > 0)
        o = np.full(len(P), -np.inf)
        tol = tol_front if vw.name == "front" else tol_gen
        o[ok] = (G.sample(vw.sdf_raw, u[ok], v[ok]) - tol) * z[ok] / vw.cam["K"][0]
        cutF = np.maximum(cutF, o)
    sel = keepreg & np.isfinite(cutF)
    F = fd.F[tuple(idx.T)]
    F[sel] = S.smooth_max(F[sel], cutF[sel], fd.vox)
    fd.F[tuple(idx.T)] = F
    fd = S.reinit(fd)
    fd, n_drop = S.keep_largest(fd)
    return fd, {"voxels_changed": int((sel & (cutF > F - 1e-9)).sum()), "pieces_dropped": n_drop}


def smooth(fd, r=None, region_w=None):
    """opening (removes webs, flaps and specks thinner than 2r) + drop loose pieces"""
    r = r or 1.2 * fd.vox
    fd = S.opening(fd, r, weight=region_w)
    fd, n_drop = S.keep_largest(fd)
    return fd, {"radius": r, "pieces_dropped": n_drop}


def region_voxels(fd, case, region, fv):
    """per-voxel weight for a front-image region (whole depth), feathered"""
    n = np.array(fd.F.shape)
    w = np.zeros(fd.F.shape, np.float32)
    reg = ndi.gaussian_filter(region.astype(np.float32), 3)
    for i0 in range(0, n[0], 16):
        g = np.stack(np.meshgrid(np.arange(i0, min(i0 + 16, n[0])), np.arange(n[1]), np.arange(n[2]), indexing="ij"), -1)
        P = fd.lo + (g.reshape(-1, 3) + 0.5) * fd.vox
        u, v, z = G.project(P, fv.cam)
        w[i0:i0 + 16] = G.sample(reg, u, v).reshape(g.shape[:3])
    return np.clip(w * 1.5, 0, 1)


# ----------------------------------------------------------------------------- runner


def run_stage(case, stage, passes=None, region=None, fit_scale=None, eval_scale=0.5, sheet_every=0, log=print):
    prm = dict(STAGES[stage])
    n = passes or prm["passes"]
    fit_scale = fit_scale or case.cfg["settings"].get("fit_scale", 1.0)
    fd = S.Field.load(case.p("build", "field.npz"))
    views = MP.load_views(case, fit_scale)
    evs = MP.load_views(case, eval_scale)
    fv = next(v for v in views if v.name == "front")
    focus = load_region(case, region) if region else None
    sym = case.cfg.get("symmetry")
    label = stage if not region else f"{stage} [{region}]"
    for k in range(n):
        t0 = time.time()
        fd, st = one_pass(fd, views, prm, sym, focus, fv)
        V, Fc = fd.mesh()
        want = (sheet_every > 0 and (k + 1) % sheet_every == 0) or (sheet_every < 0 and k == n - 1)
        res, maps = MP.evaluate(V, Fc, evs, want_maps=want)
        fd.save(case.p("build", "field.npz"))
        case.save_mesh(V, Fc)
        summ = {kk: {m: round(x, 4) for m, x in vv.items()} for kk, vv in res.items() if kk.endswith("_mean")}
        summ["front"] = {m: round(x, 4) for m, x in res["front"].items() if isinstance(x, float)}
        case.write_step(V, Fc, f"fit {label} {k + 1}/{n}", "fit", {"stage": stage, "region": region, "pass": k + 1,
                                                                    "seconds": round(time.time() - t0, 1), **st,
                                                                    "metrics": summ})
        tm, hm = res.get("train_mean", {}), res.get("holdout_mean", {})
        log(f"  {label} {k + 1}/{n} ({time.time() - t0:.0f}s)  train IoU {tm.get('iou', 0):.3f} normals "
            f"{tm.get('normal_med', float('nan')):.1f}° rough x{tm.get('rough_ratio', float('nan')):.2f}"
            + (f" | held-out IoU {hm.get('iou', 0):.3f} normals {hm.get('normal_med', float('nan')):.1f}°" if hm else ""))
        if maps:
            MP.sheet(evs, res, maps, case.p("checks", f"sheet_{case.n_steps() - 1:04d}.png"),
                     f"{label} pass {k + 1}/{n}")
    case.status(f"fit_{stage}")
    return fd


def run_op(case, op, region=None, which="front", radius=None, log=print):
    fd = S.Field.load(case.p("build", "field.npz"))
    views = MP.load_views(case, 1.0)
    fv = next(v for v in views if v.name == "front")
    reg = load_region(case, region) if region else None
    if op == "cut":
        fd, info = cut(fd, case, views, reg, which)
        label = f"cut: {which} silhouette" + (f" [{region}]" if region else "")
    elif op == "smooth":
        w = region_voxels(fd, case, reg, fv) if reg is not None else None
        fd, info = smooth(fd, radius * fd.vox if radius else None, w)
        label = f"smooth: opening r={info['radius'] * 1000:.1f}/1000" + (f" [{region}]" if region else "")
    else:
        raise SystemExit(f"unknown op {op}")
    V, Fc = fd.mesh()
    fd.save(case.p("build", "field.npz"))
    case.save_mesh(V, Fc)
    res, _ = MP.evaluate(V, Fc, MP.load_views(case, 0.5))
    info["metrics"] = {k: {m: round(x, 4) for m, x in v.items()} for k, v in res.items() if k.endswith("_mean")}
    case.write_step(V, Fc, label, op, info)
    log(f"  {label}: {json.dumps({k: v for k, v in info.items() if k != 'metrics'})}")
    return fd


def finish(case, taubin_it=12, log=print):
    """final surface: Taubin-smoothed copy of the field's mesh (no facets), recorded and exported"""
    fd = S.Field.load(case.p("build", "field.npz"))
    V, Fc = fd.mesh()
    A, deg = G.adjacency(G.edges_of(Fc), len(V))
    V = G.taubin(V, A, deg, it=taubin_it)
    np.savez_compressed(case.p("build", "final_mesh.npz"), verts=V.astype(np.float32), faces=Fc.astype(np.int32))
    res, maps = MP.evaluate(V, Fc, MP.load_views(case, 0.5), want_maps=True)
    case.write_step(V, Fc, "finish: smooth facets", "finish", {"metrics": {k: v for k, v in res.items()
                                                                            if k.endswith("_mean") or k == "front"}})
    json.dump(res, open(case.p("checks", "metrics_final.json"), "w"), indent=1)
    MP.sheet(MP.load_views(case, 0.5), res, maps, case.p("checks", "sheet_final.png"), "final sculpt against every map")
    write_obj(case.p("out", "sculpt.obj", mk=True), V, Fc)
    write_stl(case.p("out", "sculpt.stl"), V, Fc)
    case.status("finish")
    log(f"  final: {len(V)} vertices -> out/sculpt.obj, out/sculpt.stl")
    return V, Fc, res


def write_obj(path, V, Fc):
    with open(path, "w") as f:
        f.write("# image-to-sculpt\n")
        np.savetxt(f, V, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, Fc + 1, fmt="f %d %d %d")


def write_stl(path, V, Fc):
    T = V[Fc].astype(np.float32)
    n = G.unit(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])).astype(np.float32)
    rec = np.zeros(len(Fc), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, T
    with open(path, "wb") as f:
        f.write(b"image-to-sculpt".ljust(80, b" "))
        f.write(np.uint32(len(Fc)).tobytes())
        f.write(rec.tobytes())
