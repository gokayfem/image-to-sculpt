"""Cameras, mesh helpers and a numpy z-buffer rasteriser. Pure numpy/scipy, no bpy.

World frame (every case): the photo's camera sits at the origin looking along +Y, +Z up, +X to the right of the
image. Units are normalised per case so the subject's larger front extent (width or height) is 1.0.

A view camera is the front camera ORBITED about the vertical axis through the subject centre `c` by azimuth `az`
(degrees, +az = the camera moves to the right of the original image, so it sees more of what was on the right),
then raised by elevation `el` (degrees, camera moves up and looks down), with a focal scale `s` and a principal
point offset (du, dv) in pixels of the case's working resolution.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse


# ----------------------------------------------------------------------------- small helpers

def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def sample(img, u, v, order=1):
    """bilinear sample at continuous pixel coords (pixel i's centre at i + 0.5)"""
    c = [np.asarray(v) - 0.5, np.asarray(u) - 0.5]
    if img.ndim == 2:
        return ndi.map_coordinates(img, c, order=order, mode="nearest")
    return np.stack([ndi.map_coordinates(img[..., k], c, order=order, mode="nearest")
                     for k in range(img.shape[-1])], -1)


# ----------------------------------------------------------------------------- cameras

def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def pose(az, el, center):
    """-> (R, p): R columns = camera right, forward, up in world; p = camera position"""
    Ro = rz(math.radians(az)) @ rx(-math.radians(el))
    c = np.asarray(center, float)
    return Ro, c + Ro @ (-c)


def make_cam(params, center, K):
    """params: az, el, s, du, dv. K = (fx, fy, cx, cy) in pixels of the resolution the camera is used at."""
    R, p = pose(params.get("az", 0.0), params.get("el", 0.0), center)
    fx, fy, cx, cy = K
    s = params.get("s", 1.0)
    return {"R": R, "p": p, "K": (fx * s, fy * s, cx + params.get("du", 0.0), cy + params.get("dv", 0.0))}


def scale_cam(cam, f):
    """the same camera at f x the resolution"""
    return {"R": cam["R"], "p": cam["p"], "K": tuple(k * f for k in cam["K"])}


def project(X, cam):
    """world points -> u, v (continuous pixel coords), z (camera-axis depth)"""
    q = (np.asarray(X, float) - cam["p"]) @ cam["R"]
    fx, fy, cx, cy = cam["K"]
    z = q[:, 1]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cx + fx * q[:, 0] / zs, cy - fy * q[:, 2] / zs, z


def unproject(u, v, z, cam):
    """pixel coords + camera-axis depth -> world points"""
    fx, fy, cx, cy = cam["K"]
    d = np.stack([(np.asarray(u) - cx) / fx, np.ones_like(np.asarray(u, float)), -(np.asarray(v) - cy) / fy], -1)
    return cam["p"] + (d * np.asarray(z)[..., None]) @ cam["R"].T


def blender_camera(cam, W, H):
    """camera -> Blender object matrix (4x4, camera looks down its -Z, +Y up), lens (mm, 36 mm sensor width),
    shift_x, shift_y (fractions of the sensor width)"""
    R, p = cam["R"], cam["p"]
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = R[:, 0], R[:, 2], -R[:, 1], p
    fx, fy, cx, cy = cam["K"]
    return {"matrix": M.tolist(), "lens": fx * 36.0 / W, "shift_x": (W / 2 - cx) / W, "shift_y": (cy - H / 2) / W,
            "width": int(W), "height": int(H)}


# ----------------------------------------------------------------------------- meshes

def vertex_normals(V, Fc):
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, Fc[:, k], fn)
    return unit(N)


def edges_of(Fc):
    E = np.sort(np.r_[Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]], 1)
    return np.unique(E, axis=0)


def adjacency(E, n):
    A = sparse.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    return A, deg


def diffuse(x, A, deg, it):
    """it rounds of 50/50 neighbour averaging of a per-vertex scalar or vector"""
    for _ in range(it):
        x = 0.5 * x + 0.5 * ((A @ x) / (deg[:, None] if x.ndim == 2 else deg))
    return x


def smooth_vec(X, A, deg, it):
    for _ in range(it):
        X = unit(0.5 * X + 0.5 * (A @ X) / deg[:, None])
    return X


def taubin(V, A, deg, it=10, lam=0.5, mu=-0.53):
    """low-pass without shrinkage: removes marching-cubes facets, keeps the forms"""
    for _ in range(it):
        V = V + lam * ((A @ V) / deg[:, None] - V)
        V = V + mu * ((A @ V) / deg[:, None] - V)
    return V


def signed_volume(V, Fc):
    return float(np.einsum("ij,ij->i", V[Fc[:, 0]], np.cross(V[Fc[:, 1]], V[Fc[:, 2]])).sum() / 6.0)


# ----------------------------------------------------------------------------- rendering

def raster(V, Fc, cam, w, h, VN=None, maxpix=48):
    """z-buffer rasteriser for a closed triangle mesh. -> dict depth (camera axis, inf = empty), face (-1 = empty),
    mask, and with VN: nworld (interpolated world normal), ncam (camera frame: x right, y up, z toward camera)."""
    u, v, z = project(V, cam)
    tu, tv, tz = u[Fc], v[Fc], z[Fc]
    P = V[Fc]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    good = ((P.mean(1) - cam["p"]) * fn).sum(1) < 0  # back-face cull
    i0, i1 = np.ceil(tu.min(1) - 0.5).astype(int), np.floor(tu.max(1) - 0.5).astype(int)
    j0, j1 = np.ceil(tv.min(1) - 0.5).astype(int), np.floor(tv.max(1) - 0.5).astype(int)
    good &= (tz.min(1) > 0) & (i1 >= 0) & (i0 < w) & (j1 >= 0) & (j0 < h) & (i1 >= i0) & (j1 >= j0)
    good &= ((i1 - i0) < maxpix) & ((j1 - j0) < maxpix)
    fid = np.nonzero(good)[0]
    depth = np.full(h * w, np.inf)
    face = np.full(h * w, -1, np.int64)
    out = {"depth": depth.reshape(h, w), "face": face.reshape(h, w), "mask": np.zeros((h, w), bool)}
    if VN is not None:
        out["nworld"] = np.zeros((h, w, 3))
        out["ncam"] = np.zeros((h, w, 3))
    if not len(fid):
        return out
    x0, x1, x2 = tu[fid].T
    y0, y1, y2 = tv[fid].T
    z0, z1, z2 = tz[fid].T
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    ok_den = np.abs(den) > 1e-12
    den = np.where(ok_den, den, 1.0)
    I0, I1, J0, J1 = i0[fid], i1[fid], j0[fid], j1[fid]
    pix, zs, fs, b0, b1 = [], [], [], [], []
    for di in range(int((I1 - I0).max()) + 1):
        for dj in range(int((J1 - J0).max()) + 1):
            ii, jj = I0 + di, J0 + dj
            ok = ok_den & (ii <= I1) & (jj <= J1) & (ii >= 0) & (ii < w) & (jj >= 0) & (jj < h)
            if not ok.any():
                continue
            px, py = ii + 0.5, jj + 0.5
            l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
            l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
            l2 = 1 - l0 - l1
            hit = ok & (l0 >= -1e-7) & (l1 >= -1e-7) & (l2 >= -1e-7)
            if not hit.any():
                continue
            pix.append(jj[hit] * w + ii[hit])
            zs.append((l0 * z0 + l1 * z1 + l2 * z2)[hit])
            fs.append(fid[hit])
            b0.append(l0[hit])
            b1.append(l1[hit])
    if not pix:
        return out
    pix, zs, fs, b0, b1 = (np.concatenate(a) for a in (pix, zs, fs, b0, b1))
    order = np.lexsort((zs, pix))
    ps = pix[order]
    sel = order[np.r_[True, ps[1:] != ps[:-1]]]
    depth[pix[sel]] = zs[sel]
    face[pix[sel]] = fs[sel]
    out = {"depth": depth.reshape(h, w), "face": face.reshape(h, w), "mask": np.isfinite(depth).reshape(h, w)}
    if VN is not None:
        n = np.zeros((h * w, 3))
        f = Fc[fs[sel]]
        l0, l1 = b0[sel], b1[sel]
        n[pix[sel]] = VN[f[:, 0]] * l0[:, None] + VN[f[:, 1]] * l1[:, None] + VN[f[:, 2]] * (1 - l0 - l1)[:, None]
        n = unit(n) * (np.linalg.norm(n, axis=1, keepdims=True) > 0)
        out["nworld"] = n.reshape(h, w, 3)
        R = cam["R"]
        out["ncam"] = np.stack([n @ R[:, 0], n @ R[:, 2], -(n @ R[:, 1])], -1).reshape(h, w, 3)
    return out


def splat(pts, cam, w, h, close=1):
    """fast silhouette of a dense surface point sample (registration search)"""
    u, v, z = project(pts, cam)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), bool)
    m[v[ok].astype(int), u[ok].astype(int)] = True
    if close:
        m = ndi.binary_closing(m, iterations=close)
        m = ndi.binary_fill_holes(m)
    return m


def clay_shade(ncam, mask, base=(0.80, 0.72, 0.62), bg=(0.07, 0.07, 0.08)):
    """quick clay preview from a camera-frame normal map (key light upper left, soft fill), uint8 RGB"""
    L1 = unit(np.array([-0.45, 0.55, 0.70]))
    L2 = unit(np.array([0.6, -0.1, 0.8]))
    d = np.clip(ncam @ L1, 0, 1) * 0.75 + np.clip(ncam @ L2, 0, 1) * 0.2 + 0.18
    rim = np.clip(1 - ncam[..., 2], 0, 1) ** 3 * 0.15
    col = np.clip((d + rim)[..., None] * np.array(base), 0, 1)
    col = np.where(mask[..., None], col, np.array(bg))
    return (col * 255).astype(np.uint8)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))
