"""Cameras and the rasteriser against closed forms."""
import numpy as np

from i2s import geom as G
from i2s import sdf as S

K = (500.0, 500.0, 160.0, 200.0)
C = np.array([0.05, 2.0, -0.03])


def test_project_unproject_roundtrip():
    rng = np.random.default_rng(0)
    for az, el in ((0, 0), (37, 8), (-120, 25), (180, -10)):
        cam = G.make_cam({"az": az, "el": el, "s": 1.05, "du": 3, "dv": -2}, C, K)
        X = C + rng.normal(0, 0.2, (200, 3))
        u, v, z = G.project(X, cam)
        Y = G.unproject(u, v, z, cam)
        assert np.abs(Y - X).max() < 1e-9


def test_orbit_keeps_distance_and_looks_at_centre():
    d0 = np.linalg.norm(C)
    for az, el in ((45, 0), (90, 30), (-150, -20)):
        cam = G.make_cam({"az": az, "el": el}, C, K)
        assert abs(np.linalg.norm(cam["p"] - C) - d0) < 1e-9
        R = cam["R"]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        u, v, _ = G.project(C[None], G.make_cam({"az": az, "el": el}, C, K))
        u0, v0, _ = G.project(C[None], G.make_cam({}, C, K))
        assert abs(u[0] - u0[0]) < 1e-6 and abs(v[0] - v0[0]) < 1e-6  # the centre stays put on screen


def test_blender_camera_is_rigid():
    b = G.blender_camera(G.make_cam({"az": 30, "el": 10}, C, K), 320, 400)
    M = np.array(b["matrix"])[:3, :3]
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-12) and np.linalg.det(M) > 0
    assert abs(b["lens"] - 500 * 36 / 320) < 1e-9


def _sphere_mesh(r=0.3, c=(0.0, 2.0, 0.0), vox=0.01):
    n = int(2 * (r + 0.05) / vox)
    lo = np.array(c) - (r + 0.05)
    g = lo + (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1) + 0.5) * vox
    F = np.linalg.norm(g - np.array(c), axis=-1) - r
    return S.mesh_of(F.astype(np.float32), lo, vox)


def test_raster_sphere_depth_and_normals():
    V, Fc = _sphere_mesh()
    cam = G.make_cam({}, C, K)
    Rm = G.raster(V, Fc, cam, 320, 400, VN=G.vertex_normals(V, Fc))
    # analytic: ray through the pixel centre hits the sphere at the near root
    ys, xs = np.nonzero(Rm["mask"])
    sel = np.random.default_rng(0).choice(len(ys), 300, replace=False)
    ys, xs = ys[sel], xs[sel]
    d = np.stack([(xs + 0.5 - K[2]) / K[0], np.ones(len(xs)), -(ys + 0.5 - K[3]) / K[1]], -1)
    c = np.array([0.0, 2.0, 0.0])
    a, b, cc = (d * d).sum(1), -2 * (d @ c), c @ c - 0.3 ** 2
    disc = b * b - 4 * a * cc
    ok = disc > 0.02
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    err = np.abs(Rm["depth"][ys, xs] - t)[ok]
    assert np.median(err) < 0.002, np.median(err)
    n_true = G.unit(d * t[:, None] - c)
    n_ras = Rm["nworld"][ys, xs]
    ang = np.degrees(np.arccos(np.clip((n_true * n_ras).sum(1), -1, 1)))[ok]
    assert np.median(ang) < 2.0, np.median(ang)
    return f"depth err {np.median(err) * 1000:.2f}/1000, normal err {np.median(ang):.2f} deg"


def test_mesh_is_outward_and_closed():
    V, Fc = _sphere_mesh()
    assert G.signed_volume(V, Fc) > 0
    vol = G.signed_volume(V, Fc)
    assert abs(vol - 4 / 3 * np.pi * 0.3 ** 3) / (4 / 3 * np.pi * 0.3 ** 3) < 0.02
