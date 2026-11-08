"""Signed distance grid: the sculpt's volume between passes.

A Field holds F (float32, negative inside), the grid origin `lo` and the voxel size `vox`; voxel i's centre is at
lo + (i + 0.5) * vox. Every operation ends with `consistent`: exact triangle distances in a band around the
surface, a Euclidean distance transform everywhere else. Fields that are only "distance-like" (smoothed
occupancy, offsets of offsets) turn into voxel staircases once the surface moves, so they are never kept.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure

from . import geom as G


class Field:
    def __init__(self, F, lo, vox):
        self.F, self.lo, self.vox = F.astype(np.float32), np.asarray(lo, float), float(vox)

    @property
    def shape(self):
        return self.F.shape

    def centres(self, idx):
        return self.lo + (np.asarray(idx) + 0.5) * self.vox

    def mesh(self):
        return mesh_of(self.F, self.lo, self.vox)

    def save(self, path):
        np.savez_compressed(path, F=self.F, lo=self.lo, vox=self.vox)

    @staticmethod
    def load(path):
        z = np.load(path)
        return Field(z["F"], z["lo"], float(z["vox"]))

    def copy(self):
        return Field(self.F.copy(), self.lo.copy(), self.vox)


def mesh_of(F, lo, vox, level=0.0):
    """marching cubes of the zero set, outward-facing triangles, padded so the mesh is closed"""
    P = np.pad(F, 1, constant_values=max(float(F.max()), vox))
    v, f, _, _ = measure.marching_cubes(P, level, spacing=(vox, vox, vox))
    V = (v - vox) + np.asarray(lo) + 0.5 * vox
    f = f.astype(np.int64)
    if G.signed_volume(V, f) < 0:
        f = f[:, ::-1]
    return V.astype(np.float64), f


def pt_tri_closest(P, A, B, C):
    """closest point on each triangle (Ericson), vectorised"""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    Q = np.empty_like(P)
    done = np.zeros(len(P), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for m, q in (((d1 <= 0) & (d2 <= 0), lambda m: A[m]),
                     ((d3 >= 0) & (d4 <= d3), lambda m: B[m]),
                     ((vc <= 0) & (d1 >= 0) & (d3 <= 0), lambda m: A[m] + (d1 / (d1 - d3))[m, None] * ab[m]),
                     ((d6 >= 0) & (d5 <= d6), lambda m: C[m]),
                     ((vb <= 0) & (d2 >= 0) & (d6 <= 0), lambda m: A[m] + (d2 / (d2 - d6))[m, None] * ac[m]),
                     ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
                      lambda m: B[m] + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[m, None] * (C[m] - B[m]))):
            m = m & ~done
            Q[m] = q(m)
            done |= m
        m = ~done
        den = 1.0 / (va + vb + vc)
        Q[m] = A[m] + ab[m] * (vb * den)[m, None] + ac[m] * (vc * den)[m, None]
    return Q


def mesh_closest(P, V, Fc, k=10, chunk=200_000):
    """exact closest point on a triangle mesh (k nearest face centres as candidates): distance, point, face"""
    T = V[Fc]
    tree = cKDTree(T.mean(1))
    D, Qo, Fo = np.empty(len(P)), np.empty_like(P), np.empty(len(P), np.int64)
    k = min(k, len(Fc))
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]
        _, j = tree.query(p, k=k, workers=-1)
        j = j.reshape(len(p), k)
        pp = np.repeat(p, k, axis=0)
        jj = j.ravel()
        Q = pt_tri_closest(pp, T[jj, 0], T[jj, 1], T[jj, 2])
        d = np.linalg.norm(pp - Q, axis=1).reshape(-1, k)
        d = np.where(np.isfinite(d), d, np.inf)  # zero-area triangles give NaN; their neighbours cover them
        a = d.argmin(1)
        r = np.arange(len(p))
        D[i:i + chunk] = d[r, a]
        Qo[i:i + chunk] = Q.reshape(-1, k, 3)[r, a]
        Fo[i:i + chunk] = j[r, a]
    return D, Qo, Fo


def consistent(fd, idx, vals):
    """exact values in the band, a Euclidean distance transform of the inside mask everywhere else"""
    F = fd.F
    F[tuple(idx.T)] = vals
    inside = F < 0
    D = np.where(inside, -(ndi.distance_transform_edt(inside) - 0.5),
                 ndi.distance_transform_edt(~inside) - 0.5) * fd.vox
    D[tuple(idx.T)] = vals
    fd.F = D.astype(np.float32)
    return fd


def band_idx(fd, band):
    return np.argwhere(np.abs(fd.F) < band)


def apply_mesh(fd, V, Fc, t, Nd, A, deg, band=None):
    """move the surface: displace the mesh by t along normals Nd, then set the band to the EXACT signed distance to
    the displaced mesh. Sign near the surface: a vote of the nearest sheet's vertices (so two sheets that nearly
    touch do not flip each other's signs); further out, the previous field offset by t."""
    band = band or 6 * fd.vox
    V2 = V + t[:, None] * Nd
    idx = band_idx(fd, band)
    P = fd.centres(idx)
    d, _, _ = mesh_closest(P, V2, Fc)
    N2 = G.smooth_vec(G.vertex_normals(V2, Fc), A, deg, 2)
    dk, jk = cKDTree(V2).query(P, k=8, workers=-1)
    wk = (1.0 / (dk + 0.12 * fd.vox)) * (dk <= d[:, None] + 0.6 * fd.vox)
    vote = (wk * ((P[:, None, :] - V2[jk]) * N2[jk]).sum(-1)).sum(1)
    dk4, jk4 = dk[:, :4], jk[:, :4]
    w4 = 1.0 / (dk4 + 0.2 * fd.vox)
    approx = fd.F[tuple(idx.T)] - (t[jk4] * w4).sum(1) / w4.sum(1)
    sgn = np.where(d < 1.6 * fd.vox, np.sign(vote), np.sign(approx))
    sgn = np.where(sgn == 0, 1.0, sgn)
    return consistent(fd, idx, (sgn * d).astype(np.float32))


def reinit(fd):
    """same surface, exact distances: re-derive the band from the field's own marching-cubes mesh"""
    V, Fc = fd.mesh()
    A, deg = G.adjacency(G.edges_of(Fc), len(V))
    return apply_mesh(fd, V, Fc, np.zeros(len(V)), G.vertex_normals(V, Fc), A, deg)


def offset_field(fd, V, t, band=None):
    """offset the field by a smooth per-vertex t (interpolated from the nearest sheet only), then make the grid
    consistent. For one-off ops with sharp steps, where displacing marching-cubes vertices would fold the mesh."""
    band = band or 6 * fd.vox
    idx = band_idx(fd, band)
    P = fd.centres(idx)
    dk, jk = cKDTree(V).query(P, k=8, workers=-1)
    wk = (1.0 / (dk + 0.2 * fd.vox)) * (dk <= dk[:, :1] + 0.6 * fd.vox)
    tk = (t[jk] * wk).sum(1) / wk.sum(1)
    return consistent(fd, idx, (fd.F[tuple(idx.T)] - tk).astype(np.float32))


def from_occupancy(occ, lo, vox, sigma=1.0):
    """voxel occupancy -> exact signed distance field of its smoothed boundary"""
    f = 0.5 - ndi.gaussian_filter(occ.astype(np.float32), sigma)
    fd = Field(np.where(occ, -vox, vox).astype(np.float32), lo, vox)
    fd.F = (f * 2 * vox).astype(np.float32)  # distance-like start, replaced by exact values below
    return reinit(fd)


def resample(fd, vox, pad=4):
    """the same surface on a grid with a different voxel size (tight bbox around the surface + pad voxels)"""
    V, _ = fd.mesh()
    lo = V.min(0) - pad * vox
    n = np.ceil((V.max(0) + pad * vox - lo) / vox).astype(int)
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    P = lo + (idx + 0.5) * vox
    c = ((P - fd.lo) / fd.vox - 0.5).T
    F = ndi.map_coordinates(fd.F, c, order=1, mode="nearest").reshape(n)
    return reinit(Field(F.astype(np.float32), lo, vox))


def keep_largest(fd):
    inside = fd.F < 0
    lab, nl = ndi.label(inside)
    if nl > 1:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, nl + 1))
        keep = lab == 1 + int(np.argmax(sizes))
        fd.F = np.where(inside & ~keep, np.abs(fd.F) + fd.vox, fd.F).astype(np.float32)
        return reinit(fd), nl - 1
    return fd, 0


def opening(fd, r, weight=None):
    """morphological opening by a ball of radius r (erode, then dilate; exact re-distancing between): removes
    flaps, webs and specks thinner than 2r. weight: per-voxel 0..1 to apply it only in a region."""
    F0 = fd.F.copy()
    fd.F = fd.F + r
    fd = reinit(fd)
    fd.F = fd.F - r
    fd = reinit(fd)
    if weight is not None:
        fd.F = (weight * fd.F + (1 - weight) * F0).astype(np.float32)
        fd = reinit(fd)
    return fd


def closing(fd, r, weight=None):
    """morphological closing by a ball of radius r: fills pits and cracks narrower than 2r"""
    F0 = fd.F.copy()
    fd.F = fd.F - r
    fd = reinit(fd)
    fd.F = fd.F + r
    fd = reinit(fd)
    if weight is not None:
        fd.F = (weight * fd.F + (1 - weight) * F0).astype(np.float32)
        fd = reinit(fd)
    return fd


def smooth_max(a, b, k):
    """C1 smooth maximum (cuts without creases)"""
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0, 1)
    return a * h + b * (1 - h) + k * h * (1 - h)
