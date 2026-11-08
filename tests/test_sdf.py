"""The distance field stays exact through every operation."""
import numpy as np

from i2s import geom as G
from i2s import sdf as S


def _ball_occ(r=0.25, vox=1 / 80, pad=0.08, fin=False):
    n = int(2 * (r + pad) / vox)
    lo = np.full(3, -(r + pad))
    g = lo + (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1) + 0.5) * vox
    occ = np.linalg.norm(g, axis=-1) < r
    if fin:  # a thin plate sticking out of the ball (a web or flap)
        occ |= (np.abs(g[..., 1]) < 0.6 * vox) & (np.abs(g[..., 2]) < 0.1) & (g[..., 0] > 0) & (g[..., 0] < r + 0.06)
    return occ, lo, vox


def test_from_occupancy_is_a_distance_field():
    occ, lo, vox = _ball_occ()
    fd = S.from_occupancy(occ, lo, vox)
    idx = np.argwhere(np.abs(fd.F) < 4 * vox)
    P = fd.centres(idx)
    exact = np.linalg.norm(P, axis=1) - 0.25
    err = np.abs(fd.F[tuple(idx.T)] - exact)
    assert np.median(err) < 0.35 * vox, np.median(err) / vox
    gz, gy, gx = np.gradient(fd.F, vox)
    gm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)[tuple(np.argwhere(np.abs(fd.F) < 8 * vox).T)]
    assert abs(np.median(gm) - 1) < 0.05  # |grad| = 1: a true distance field
    return f"median error {np.median(err) / vox:.2f} voxel"


def test_apply_mesh_offsets_exactly():
    occ, lo, vox = _ball_occ()
    fd = S.from_occupancy(occ, lo, vox)
    V, Fc = fd.mesh()
    r0 = np.linalg.norm(V, axis=1).mean()
    A, deg = G.adjacency(G.edges_of(Fc), len(V))
    t = np.full(len(V), 0.6 * vox)
    fd = S.apply_mesh(fd, V, Fc, t, G.vertex_normals(V, Fc), A, deg)
    V2, _ = fd.mesh()
    r1 = np.linalg.norm(V2, axis=1).mean()
    assert abs((r1 - r0) - 0.6 * vox) < 0.15 * vox, (r1 - r0) / vox


def test_opening_removes_a_thin_web_and_keeps_the_ball():
    occ, lo, vox = _ball_occ(fin=True)
    fd = S.from_occupancy(occ, lo, vox, sigma=0.5)
    V0, _ = fd.mesh()
    assert np.linalg.norm(V0, axis=1).max() > 0.28  # the fin is there
    fd = S.opening(fd, 1.5 * vox)
    fd, _ = S.keep_largest(fd)
    V1, _ = fd.mesh()
    rr = np.linalg.norm(V1, axis=1)
    assert rr.max() < 0.27, rr.max()
    assert abs(np.median(rr) - 0.25) < 0.6 * vox


def test_resample_keeps_the_surface():
    occ, lo, vox = _ball_occ()
    fd = S.from_occupancy(occ, lo, vox)
    f2 = S.resample(fd, vox / 2)
    V, _ = f2.mesh()
    assert abs(np.linalg.norm(V, axis=1).mean() - 0.25) < 0.5 * vox
